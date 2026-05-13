from terratorch.registry import (
    TERRATORCH_BACKBONE_REGISTRY,
    TERRATORCH_DECODER_REGISTRY,
    TERRATORCH_HEAD_REGISTRY,
)
from terratorch.models.model import PixelWiseModel
from terratorch.models.backbones import TerramindViT
from terratorch.models.decoders import UNetDecoder
from terratorch.models.heads import SegmentationHead
import torch.nn as nn
from terratorch.models.factory import ModelFactory
from terratorch.models.model import PixelWiseModel
from typing import List, Any, Dict, Optional


class TerramindModelFactory(ModelFactory):
    def __init__(self):
        self.backbone_registry = TERRATORCH_BACKBONE_REGISTRY
        self.decoder_registry = TERRATORCH_DECODER_REGISTRY
        self.head_registry = TERRATORCH_HEAD_REGISTRY

    def build_model(self, model_args: Dict[str, Any]) -> PixelWiseModel:
        # Extract backbone arguments
        backbone_name = model_args.get("backbone", "terramind_v1_base")
        backbone_pretrained = model_args.get("backbone_pretrained", True)
        backbone_modalities = model_args.get("backbone_modalities", ["RGB"])

        # Extract backbone-specific arguments
        backbone_args = {}
        if "backbone_kwargs" in model_args:
            backbone_args = model_args["backbone_kwargs"]

        # Build the backbone with specific model name and pretrained setting
        backbone = self.backbone_registry.build(
            backbone_name,
            modalities=backbone_modalities,
            pretrained=backbone_pretrained,
            **backbone_args,
        )

        # Create the decoder with the specified config
        if "decoder" in model_args:
            decoder = self.decoder_registry.build(
                model_args["decoder"],
                in_channels=backbone.out_channels,
                decoder_channels=model_args.get(
                    "decoder_channels", [512, 256, 128, 64]
                ),
                **model_args.get("decoder_kwargs", {}),
            )
        else:
            # Default to UNetDecoder if not specified
            decoder = UNetDecoder(
                in_channels=backbone.out_channels,
                decoder_channels=model_args.get(
                    "decoder_channels", [5512, 256, 128, 64]
                ),
            )

        # Create the head (SegmentationHead)
        if "head" in model_args:
            head = self.head_registry.build(
                model_args["head"],
                in_channels=decoder.out_channels,
                num_classes=model_args["num_classes"],
                **model_args.get("head_kwargs", {}),
            )
        else:
            head = SegmentationHead(
                in_channels=decoder.out_channels,
                num_classes=model_args["num_classes"],
                dropout=model_args.get("head_dropout", 0.1),
            )

        # Create the final model with backbone, decoder, and head
        model = PixelWiseModel(
            backbone=backbone,
            decoder=decoder,
            head=head,
            necks=model_args.get("necks", []),
        )

        return model
