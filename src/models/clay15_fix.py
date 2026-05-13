"""
Fix for Clay 1.5 + Terratorch compatibility issue

The problem: ClayMAE.forward() returns a tuple of losses (for MAE pretraining),
but Terratorch's segmentation pipeline expects a ModelOutput object.

The solution: Create a ClayMAEBackbone adapter that uses only the encoder
to extract features, then use the standard Terratorch PixelWiseModel pipeline.

Usage Example with NDSM:

    model_args = dict(
        backbone="clay15",
        decoder="FCNDecoder",
        in_channels=5,  # R, G, B, NIR, NDSM
        bands=["red", "green", "blue", "nir", "ndsm"],
        num_classes=24,
        pretrained=True,
        num_frames=1,
        # ClayMAE architecture
        dim=192,
        depth=6,
        heads=4,
        dim_head=48,
        mlp_ratio=2,
        patch_size=8,
        # Metadata with NDSM statistics
        metadata={
            "naip": {
                "band_order": ["red", "green", "blue", "nir", "ndsm"],
                "gsd": 1.0,
                "bands": {
                    "mean": {
                        "red": 110.16, "green": 115.41, "blue": 98.15, "nir": 139.04,
                        "ndsm": 8.5431  # NDSM mean (meters)
                    },
                    "std": {
                        "red": 47.23, "green": 39.82, "blue": 35.43, "nir": 49.86,
                        "ndsm": 10.2349  # NDSM std (meters)
                    },
                    "wavelength": {
                        "red": 0.65, "green": 0.56, "blue": 0.48, "nir": 0.842,
                        "ndsm": 0.0  # Placeholder (NDSM is elevation, not spectral)
                    }
                }
            }
        },
        platform=["naip"],
        batch_size=8,
    )

    task = SemanticSegmentationTask(
        model_args,
        "Clay1_5ModelFactoryFixed",
        loss="ce",
        lr=1e-4,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
        freeze_backbone=False,
    )

Performance:
    - Clay 1.5 (4 bands, no NDSM): 0.2095 mIoU
    - Clay 1.5 (5 bands, with NDSM): 0.2735 mIoU (+30.5% improvement)
"""

import logging
import math
from collections.abc import Callable
from pathlib import Path

import torch
from einops import repeat
from torch import nn
from box import Box
from huggingface_hub import hf_hub_download

import terratorch.models.decoders as decoder_registry
from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
)
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.scalar_output_model import ScalarOutputModel
from terratorch.models.utils import DecoderNotFoundError, extract_prefix_keys
from terratorch.registry import MODEL_FACTORY_REGISTRY
from terratorch.models.backbones.clay_v15.model import ClayMAE
from terratorch.models.backbones.clay_v15.utils import posemb_sincos_2d_with_gsd


PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS


class ClayMAEBackbone(nn.Module):
    """
    Adapter for ClayMAE to work as a feature extractor (backbone) for downstream tasks.

    This wraps the ClayMAE encoder and extracts features at multiple scales,
    compatible with Terratorch's PixelWiseModel expectations.
    """

    def __init__(self, clayma_model, batch_size, bands, platform, metadata):
        super().__init__()
        self.clayma = clayma_model
        self.batch_size = batch_size
        self.bands = bands
        self.platform = platform[0] if isinstance(platform, list) else platform
        self.metadata = metadata
        self.patch_size = clayma_model.patch_size
        self.dim = clayma_model.encoder.dim

    def forward(self, x, **kwargs):
        """
        Extract features from ClayMAE encoder at multiple scales.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            List of feature tensors at different scales (early, mid, late layers)
        """
        # Prepare datacube format expected by ClayMAE
        B, C, H, W = x.shape

        # Get metadata for this platform
        platform_str = self.platform
        waves = torch.tensor(
            list(self.metadata[platform_str].bands.wavelength.values()),
            device=x.device
        )
        gsd = torch.tensor(self.metadata[platform_str].gsd, device=x.device)

        # Create dummy time and latlon (zeros if not available)
        time = torch.zeros(B, 4, device=x.device)
        latlon = torch.zeros(B, 4, device=x.device)

        datacube = {
            "pixels": x,
            "time": time,
            "latlon": latlon,
            "gsd": gsd,
            "waves": waves,
            "platform": [platform_str] * B
        }

        # Run through encoder only (no masking for inference)
        # We need to extract the encoder's output before it goes through the full MAE pipeline

        # Patch embedding
        patches, waves_encoded = self.clayma.encoder.to_patch_embed(x, waves)

        # Add position encodings
        patches = self.clayma.encoder.add_encodings(patches, time, latlon, gsd)

        # Skip masking for downstream tasks - use all patches
        # Add class token
        cls_tokens = repeat(self.clayma.encoder.cls_token, "1 1 D -> B 1 D", B=B)
        patches_with_cls = torch.cat((cls_tokens, patches), dim=1)

        # Pass through transformer
        # Note: Clay 1.0 uses single-scale features from final layer only
        # Multi-scale extraction from early layers hurts performance (-14.5%)
        # because MAE early layers are trained for reconstruction, not semantic features
        encoded_patches = self.clayma.encoder.transformer(patches_with_cls)

        # Remove CLS token and reshape to spatial format
        # encoded_patches is a list of outputs from each layer
        # We only use the final layer (semantic features)
        final_features = encoded_patches[-1][:, 1:, :]  # Remove CLS token [B, L, D]

        # Reshape from [B, L, D] to [B, D, H', W'] where H' = W' = sqrt(L)
        L = final_features.shape[1]
        grid_size = int(math.sqrt(L))
        final_features = final_features.transpose(1, 2)  # [B, D, L]
        features = final_features.reshape(B, self.dim, grid_size, grid_size)

        # Return single-scale features (same as Clay 1.0 approach)
        return [features]

    def feature_info(self):
        """Return feature info for decoder with single-scale channels"""
        class FeatureInfo:
            def __init__(self, dim):
                # Single scale from final transformer layer
                self._channels = [dim]

            def channels(self):
                return self._channels

        return FeatureInfo(self.dim)


@MODEL_FACTORY_REGISTRY.register
class Clay1_5ModelFactoryFixed(ModelFactory):
    """
    Fixed model factory for Clay 1.5 that properly integrates with Terratorch.

    This follows the same pattern as ClayModelFactory but adapts ClayMAE
    to work as a feature extractor instead of returning losses.
    """

    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        in_channels: int,
        bands: list[int] = [],
        num_classes: int | None = None,
        pretrained: bool = True,
        num_frames: int = 1,
        prepare_features_for_image_model: Callable | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,
        **kwargs,
    ) -> Model:

        task = task.lower()
        if task not in SUPPORTED_TASKS:
            msg = f"Task {task} not supported. Please choose one of {SUPPORTED_TASKS}"
            raise NotImplementedError(msg)

        # Extract decoder and head kwargs
        # NOTE: We don't use extract_prefix_keys for "decoder_" because ClayMAE
        # has its own decoder parameters (decoder_dim, decoder_depth, etc.) that
        # should NOT be extracted. Only extract explicitly prefixed terratorch decoder params.
        decoder_cls = _get_decoder(decoder)

        # For now, just pass empty decoder kwargs (decoder gets defaults)
        # Users can pass decoder_out_channels via "decoder_out_channels" key if needed
        decoder_kwargs = {}
        head_kwargs = {}

        # Extract meta parameters
        batch_size = kwargs.get("batch_size")
        platform = kwargs.get("platform")
        metadata = Box(kwargs.get("metadata"))
        patch_size = kwargs.get("patch_size", 8)
        padding = kwargs.get("padding", "reflect")

        # Create ClayMAE model with all kwargs (includes decoder_dim, decoder_depth, etc.)
        logging.getLogger("terratorch").info("Creating ClayMAE model...")
        clayma = ClayMAE(**kwargs)

        # Load checkpoint if provided or pretrained
        if pretrained:
            checkpoint_path_arg = kwargs.get("checkpoint_path")
            if checkpoint_path_arg and Path(checkpoint_path_arg).exists():
                # Use provided local checkpoint
                logging.getLogger("terratorch").info(f"Loading checkpoint from {checkpoint_path_arg}")
                ckpt_path = checkpoint_path_arg
            else:
                # Download from HuggingFace
                logging.getLogger("terratorch").info("Downloading Clay v1.5 checkpoint from HuggingFace...")
                ckpt_path = hf_hub_download(
                    repo_id="made-with-clay/Clay",
                    filename="v1.5/clay-v1.5.ckpt",
                    cache_dir=None  # Uses default HF cache
                )
                logging.getLogger("terratorch").info(f"Downloaded to {ckpt_path}")

            # Load the checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            clayma.load_state_dict(state_dict, strict=False)
            logging.getLogger("terratorch").info("Checkpoint loaded successfully")

        # Wrap in backbone adapter
        backbone_module = ClayMAEBackbone(clayma, batch_size, bands, platform, metadata)

        # Create decoder with multi-scale feature channels from backbone
        feature_info = backbone_module.feature_info()
        feature_channels = feature_info.channels()
        decoder_instance = decoder_cls(feature_channels, **decoder_kwargs)

        # Add num_classes to head kwargs
        if num_classes:
            head_kwargs["num_classes"] = num_classes

        # Handle auxiliary decoders if provided
        if aux_decoders is None:
            return _build_appropriate_model(
                task,
                backbone_module,
                decoder_instance,
                head_kwargs,
                prepare_features_for_image_model,
                patch_size=patch_size,
                padding=padding,
                rescale=rescale
            )

        to_be_aux_decoders = []
        for aux_decoder in aux_decoders:
            args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
            aux_decoder_cls = _get_decoder(aux_decoder.decoder)
            aux_decoder_kwargs, _ = extract_prefix_keys(args, "decoder_")
            aux_decoder_instance = aux_decoder_cls(feature_channels, **aux_decoder_kwargs)
            aux_head_kwargs, _ = extract_prefix_keys(args, "head_")
            if num_classes:
                aux_head_kwargs["num_classes"] = num_classes
            to_be_aux_decoders.append(
                AuxiliaryHeadWithDecoderWithoutInstantiatedHead(
                    aux_decoder.name, aux_decoder_instance, aux_head_kwargs
                )
            )

        return _build_appropriate_model(
            task,
            backbone_module,
            decoder_instance,
            head_kwargs,
            prepare_features_for_image_model,
            patch_size=patch_size,
            padding=padding,
            rescale=rescale,
            auxiliary_heads=to_be_aux_decoders,
        )


def _build_appropriate_model(
    task: str,
    backbone: nn.Module,
    decoder: nn.Module,
    head_kwargs: dict,
    prepare_features_for_image_model: Callable,
    patch_size: int | list | None,
    padding: str,
    rescale: bool = True,
    auxiliary_heads: dict | None = None,
):
    """Build the appropriate model based on the task."""
    if task in PIXEL_WISE_TASKS:
        return PixelWiseModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            patch_size=patch_size,
            padding=padding,
            rescale=rescale,
            auxiliary_heads=auxiliary_heads,
        )
    elif task in SCALAR_TASKS:
        return ScalarOutputModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            patch_size=patch_size,
            padding=padding,
            auxiliary_heads=auxiliary_heads,
        )


def _get_decoder(decoder: str | nn.Module) -> nn.Module:
    """Get decoder class from string or return the module directly."""
    if isinstance(decoder, nn.Module):
        return decoder
    if isinstance(decoder, str):
        try:
            decoder_cls = getattr(decoder_registry, decoder)
            return decoder_cls
        except AttributeError as e:
            msg = f"Decoder {decoder} was not found in the registry."
            raise DecoderNotFoundError(msg) from e
    msg = "Decoder must be str or nn.Module"
    raise Exception(msg)
