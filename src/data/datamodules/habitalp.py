from pathlib import Path

import geopandas as gpd
import pandas as pd
import kornia.augmentation as K
import rasterio
import torch
from torch import Generator, Tensor
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import RasterDataset, random_grid_cell_assignment
from torchgeo.datasets.splits import roi_split
from torchgeo.samplers import GridGeoSampler
from typing import Tuple

from ..datasets import HabitAlp2
from ..samplers import RandomGeoSamplerWithinRoi


class HabitAlp2DataModule(GeoDataModule):
    """LightningDataModule implementation for the HabitAlp2 dataset
    - Load raster images and masks inside a RasterDataset.
    - Preprocess data on creation of a dataset.
    - Apply augmentations (flip, resize) after transfer of batch to device.
    - Create Train/Val/Test Datasets with randomly assigned grid cells.
    """

    def __init__(
        self,
        image_paths: dict[Path] | list[dict[Path]],
        mask_path: str | list[str],
        n_classes: int,
        roi_shape_path: str | list[str] | None = None,
        batch_size: int = 64,
        patch_size: tuple[int, int] = (256, 256),
        num_workers: int = 6,
        res: float | None = None,
        train_batches_per_epoch: int = 512,
        val_batches_per_epoch: int = 32,
        class_names: list[str] | None = None,
        image_paths_pred: dict[Path] | None = None,
        prediction_mask_path: str | None = None,
        ndsm_aug_config: dict | None = None,
    ):
        """Initialize the data module.

        Args:
            image_paths (dict | list[dict]): Paths to input images. Can be a single dict or list of dicts for multi-year training.
            mask_path (str | list[str]): Path to label mask image. Can be a single path or list of paths for multi-year training.
            n_classes (int): Number of target classes without the nodata class
            roi_shape_path (str | list[str]): Path to ROI geopackage file. Can be a single path (applied to all years) or list of paths (one per year) for multi-year training. If None, full dataset extent is used. Defaults to None.
            batch_size (int, optional): Number of samples per batch. Defaults to 64.
            patch_size (tuple[int, int], optional): Size of image patches to sample (height, width). Defaults to (256, 256).
            num_workers (int, optional): Number of parallel workers for data loading. Defaults to 6.
            res (float, optional): Resolution of the dataset in units of CRS (defaults to the resolution of the first file found). Defaults to None.
            train_batches_per_epoch (int, optional): Number of training batches per epoch. Defaults to 512.
            val_batches_per_epoch (int, optional): Number of validation batches per epoch. Defaults to 32.
            class_names (list[str], optional): List with class names without the nodata class. Used for plot labels. Defaults to None.
            image_paths_pred (dict, optional): Paths to input images for prediction. Defaults to None.
            prediction_mask_path (str, optional): Path to the label mask image of the prediction dataset. Defaults to None.
            ndsm_aug_config (dict, optional): Configuration for nDSM-specific augmentations. Defaults to None.
        """
        super().__init__(
            dataset_class=RasterDataset,
            batch_size=batch_size,
            patch_size=patch_size,
            length = train_batches_per_epoch * batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        self.image_paths = image_paths
        self.mask_path = mask_path
        self.roi_shape_path = roi_shape_path
        self.n_classes = n_classes
        self.res = res
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches_per_epoch = val_batches_per_epoch
        self.class_names = class_names
        self.image_paths_pred = image_paths_pred
        self.prediction_mask_path = prediction_mask_path

        # Data loaders
        self.predict_batch_size = 1  # For inference, we use batch size of 1

        def get_image_stats(
            image_paths: dict | list[dict],
        ) -> Tuple[Tensor, Tensor]:
            """Calculate per-band mean and standard deviation statistics for single- or multi-band images.

            Args:
                image_paths: Paths to the raster input image file in uint8 format.
                            Can be a single dict or list of dicts for multi-year training.
                            For multi-year, uses first year's paths for statistics.

            Returns:
                Tuple containing:
                    - Tensor of normalized mean values for each band
                    - Tensor of normalized standard deviation values for each band
            """
            # For multi-year, use first year's image paths for computing statistics
            paths_to_use = image_paths[0] if isinstance(image_paths, list) else image_paths

            band_means = []
            band_stds = []

            # Iterate over every input file
            for dataset, path in paths_to_use.items():
                with rasterio.open(path) as src:
                    stats_all = src.stats(approx=False)
                    for band in range(src.count):
                        # Skip R and G bands for CIR datasets
                        if dataset == "cir" and band > 0:
                            break
                        stats = stats_all[band]
                        band_means.append(stats.mean)
                        band_stds.append(stats.std)

            return (torch.tensor(band_means), torch.tensor(band_stds))

        self.mean, self.std = get_image_stats(self.image_paths)

        self.train_aug_all_before = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomResizedCrop(
                size=self.patch_size, scale=(0.8, 1.0), ratio=(1, 1), p=1.0
            ),
            data_keys=["image", "mask"],
        )

        self.train_aug_img = K.AugmentationSequential(
            K.ColorJiggle(brightness=0.3, contrast=0.5, p=0.8),
            data_keys=["image"],
        )

        self.train_aug_rgb = K.AugmentationSequential(
            K.RandomHue(hue=(-0.05, 0.05), p=0.5),
            K.RandomSaturation(saturation=(0.7, 1.3), p=0.5),
            data_keys=["image"],
        )

        self.train_aug_all_after = K.AugmentationSequential(
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.1),
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=["image", "mask"],
        )

        # Create nDSM-specific augmentations if config provided
        self.ndsm_aug_config = ndsm_aug_config
        if ndsm_aug_config and ndsm_aug_config.get("enable", False):
            print("Creating nDSM-specific augmentations for domain adaptation...")
            ndsm_aug_list = []

            # Gaussian blur (primary augmentation for photogrammetry simulation)
            if "gaussian_blur" in ndsm_aug_config:
                gb_cfg = ndsm_aug_config["gaussian_blur"]
                ndsm_aug_list.append(
                    K.RandomGaussianBlur(
                        kernel_size=tuple(gb_cfg["kernel_size"]),
                        sigma=tuple(gb_cfg["sigma"]),
                        p=gb_cfg["p"]
                    )
                )

            # Downsample/upsample to simulate resolution loss
            if "downsample_upsample" in ndsm_aug_config:
                du_cfg = ndsm_aug_config["downsample_upsample"]
                ndsm_aug_list.append(
                    K.RandomResize(
                        scale=tuple(du_cfg["scale"]),
                        p=du_cfg["p"]
                    )
                )

            # Median blur for edge smoothing
            if "median_blur" in ndsm_aug_config:
                mb_cfg = ndsm_aug_config["median_blur"]
                ndsm_aug_list.append(
                    K.RandomMedianBlur(
                        kernel_size=tuple(mb_cfg["kernel_size"]),
                        p=mb_cfg["p"]
                    )
                )

            # Box blur for additional smoothing
            if "box_blur" in ndsm_aug_config:
                bb_cfg = ndsm_aug_config["box_blur"]
                ndsm_aug_list.append(
                    K.RandomBoxBlur(
                        kernel_size=tuple(bb_cfg["kernel_size"]),
                        p=bb_cfg["p"]
                    )
                )

            # Gaussian noise for photogrammetry artifacts
            if "gaussian_noise" in ndsm_aug_config:
                gn_cfg = ndsm_aug_config["gaussian_noise"]
                ndsm_aug_list.append(
                    K.RandomGaussianNoise(
                        mean=gn_cfg["mean"],
                        std=gn_cfg["std"],
                        p=gn_cfg["p"]
                    )
                )

            self.train_aug_ndsm = K.AugmentationSequential(
                *ndsm_aug_list,
                data_keys=["image"],
            )
            print(f"  → Created {len(ndsm_aug_list)} nDSM augmentations")
        else:
            self.train_aug_ndsm = None

        self.val_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=["image", "mask"],
            same_on_batch=True,
        )
        self.test_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=["image", "mask"],
            same_on_batch=True,
        )

        # For multi-year, use first year's image paths for channel calculation and band checks
        self._primary_image_paths = image_paths[0] if isinstance(image_paths, list) else image_paths
        self.in_channels = (
            len(self._primary_image_paths.keys())
            + (2 if "rgb" in self._primary_image_paths else 0)
            + (3 if "spot-4" in self._primary_image_paths else 0)
        )  # add bands because rgb and spot contain more channels

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply augmentations after batch is transferred to device.

        Args:
            batch: Dictionary containing image and mask data
            dataloader_idx: Index of the dataloader

        Returns:
            Batch with augmentations applied
        """

        if self.trainer.training:
            # Apply training augmentations to both images and masks
            x = batch["image"]  # [B, 6, H, W]
            y = batch["mask"].float().unsqueeze(1)  # Add channel dim [B, 1, H, W]

            # Apply same geometric transforms to both input and mask
            transformed = self.train_aug_all_before(x, y)
            # Apply intensity transforms to image bands only (Have to be rescaled to [0,1] first)
            transformed[0][:, self.dataset.is_image_band, :, :] = self.train_aug_img(
                transformed[0][:, self.dataset.is_image_band, :, :] / 255.0
            )
            # Apply RGB specific augmentations
            if "rgb" in self._primary_image_paths:
                transformed[0][:, [0, 1, 2], :, :] = self.train_aug_rgb(
                    transformed[0][:, [0, 1, 2], :, :]
                )
            transformed[0][:, self.dataset.is_image_band, :, :] *= 255.0

            # Apply nDSM-specific augmentations if enabled
            if self.train_aug_ndsm is not None and "ndsm" in self._primary_image_paths:
                # Find nDSM channel index
                try:
                    ndsm_idx = self.dataset.bands.index("ndsm")
                    # Extract nDSM channel, apply augmentations, put back
                    ndsm_channel = transformed[0][:, ndsm_idx:ndsm_idx+1, :, :]
                    augmented_ndsm = self.train_aug_ndsm(ndsm_channel)
                    transformed[0][:, ndsm_idx:ndsm_idx+1, :, :] = augmented_ndsm[0]
                except ValueError:
                    # ndsm not in bands, skip
                    pass

            # Normalize
            transformed = self.train_aug_all_after(transformed[0], transformed[1])

            batch["image"] = transformed[0]  # First output is transformed image
            batch["mask"] = (
                transformed[1].squeeze(1).long()
            )  # Second output is transformed mask
        else:
            # For validation/test, apply normalization to both
            x = batch["image"]  # [B, 6, H, W]
            y = batch["mask"].float().unsqueeze(1)  # Add channel dim [B, 1, H, W]

            transformed = self.val_aug(x, y)
            batch["image"] = transformed[0]
            batch["mask"] = transformed[1].squeeze(1).long()

        return batch

    def setup(
        self,
        stage: str = "fit",
        stride: int | None = None,
    ):
        """Set up datasets and samplers.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        # Parse ROI shape paths
        if self.roi_shape_path is None:
            self.roi = None
            self.roi_list = None
        elif isinstance(self.roi_shape_path, list):
            # Per-year ROIs for multi-year mode
            self.roi_list = []
            for roi_path in self.roi_shape_path:
                gdf = gpd.read_file(roi_path)
                self.roi_list.append(gdf.geometry.union_all())
            self.roi = self.roi_list[0]  # Use first ROI for samplers
        else:
            gdf = gpd.read_file(self.roi_shape_path)
            self.roi = gdf.geometry.union_all()
            self.roi_list = None

        if stage in ["fit", "validate", "test"] and self.train_dataset is None:
            if isinstance(self.image_paths, list):
                # Multi-year mode - create datasets and apply per-year ROIs before union
                print(f"Creating multi-year dataset with {len(self.image_paths)} year(s)...")
                datasets = []
                for i, (img_paths, mask_path) in enumerate(zip(self.image_paths, self.mask_path)):
                    ds = HabitAlp2(
                        image_paths=img_paths,
                        mask_path=mask_path,
                        res=self.res,
                    )
                    # Apply per-year ROI if available, otherwise use global ROI
                    if self.roi_list is not None:
                        ds = roi_split(ds, [self.roi_list[i]])[0]
                        print(f"  Year {i+1}: applied per-year ROI")
                    elif self.roi is not None:
                        ds = roi_split(ds, [self.roi])[0]
                        print(f"  Year {i+1}: applied global ROI")
                    datasets.append(ds)

                # Store band info from first dataset before union (UnionDataset doesn't have these)
                self.dataset = datasets[0]
                self.dataset.bands = datasets[0].bands
                self.dataset.is_image_band = datasets[0].is_image_band
                for ds in datasets[1:]:
                    union_ds = self.dataset | ds
                    # Preserve band attributes on union dataset
                    union_ds.bands = self.dataset.bands
                    union_ds.is_image_band = self.dataset.is_image_band
                    self.dataset = union_ds
                print(f"Combined dataset spans {len(datasets)} year(s)")
            else:
                # Single year mode (backward compatible)
                self.dataset = HabitAlp2(
                    image_paths=self.image_paths,
                    mask_path=self.mask_path,
                    res=self.res,
                )
                # Apply ROI split if ROI shape is provided
                if self.roi is not None:
                    # Preserve band attributes through ROI split
                    bands = self.dataset.bands
                    is_image_band = self.dataset.is_image_band
                    self.dataset = roi_split(self.dataset, [self.roi])[0]
                    self.dataset.bands = bands
                    self.dataset.is_image_band = is_image_band

            # Create Train/Val/Test Datasets with randomly assigned grid cells
            generator = Generator().manual_seed(0)
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = random_grid_cell_assignment(
                self.dataset,
                [0.7, 0.15, 0.15],
                grid_size=12,
                generator=generator,
            )  # here is the random grid cell split
            print(
                f"Assigned {self.train_dataset.__len__()}/{self.val_dataset.__len__()}/{self.test_dataset.__len__()} cells to train/val/test datasets."
            )
        if stage in ["fit"]:
            self.train_sampler = RandomGeoSamplerWithinRoi(
                self.train_dataset,
                size=self.patch_size,
                length=self.length,
                roi=self.roi,
            )
        if stage in ["fit", "validate"]:
            self.val_sampler = RandomGeoSamplerWithinRoi(
                self.val_dataset,
                size=self.patch_size,
                length=self.val_batches_per_epoch * self.batch_size,
                roi=self.roi,
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                self.test_dataset,
                size=self.patch_size,
                stride=self.patch_size if stride is None else stride,
            )
        if stage in ["predict"]:
            if self.image_paths_pred is None:
                raise ValueError(
                    "Argument 'image_paths_pred' must be provided for prediction stage."
                )
            if stride is None:
                raise ValueError(
                    "Argument 'stride' must be provided for prediction stage."
                )

            self.predict_dataset = HabitAlp2(
                image_paths=self.image_paths_pred,
                mask_path=self.mask_path,
                res=self.res,
            )

            self.predict_sampler = GridGeoSampler(
                self.predict_dataset,
                size=self.patch_size,
                stride=stride,
                roi=self.roi,
            )
