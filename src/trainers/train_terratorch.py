import sys
from pathlib import Path

BASE_FOLDER = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_FOLDER))

import argparse
import warnings
import gc
import torch
import wandb
from lightning.fabric import Fabric

from src.data.datamodules.habitalp import HabitAlp2DataModule
from src.trainers.terratorch.segmentation_tasks import ExtendedSemanticSegmentationTask
from src.trainers.utils import load_config_from_yaml, setup_training

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    # Location of configuration file
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to YAML config file with data paths and parameters",
    )

    args = parser.parse_args()

    if args.config:
        # Load configuration file
        config_path = Path(args.config)
        if not config_path.is_absolute():
            # Look for config relative to cwd
            script_dir = Path.cwd()
            config_path = script_dir / args.config

        if config_path.exists():
            print("Loading configuration file ...")
            config = load_config_from_yaml(config_path)
            data_folder = Path(config["data_folder"])
            hyperparameters = config["hyperparameters"]

            experiment_name = config["name"]
            n_classes = config["num_labels"]
            num_workers = config["num_dataloader_workers"]
            class_weights = config["precalculated_weights"]
            ignore_index = config["ignore_index"]
            matmul_precision = config["matmul_precision"]
            precision = config["precision"]
            gpu_id = config["gpu_id"]
            input_bands = hyperparameters["input_bands"]
            class_names = config["class_names"]
            wandb_key = config["wandb_key"]
            model_factory = config.get("model_factory", "ClayModelFactory")
            model_args_config = config.get("model_args", {})
            freeze_backbone = config.get("freeze_backbone", False)

            # Check for multi-year training config
            if "training_years" in config:
                # Multi-year mode
                print(f"Multi-year training mode: {len(config['training_years'])} year(s)")
                image_data_paths = [
                    {
                        key: data_folder / year_cfg["image_data_paths"][key]
                        for key in input_bands
                        if key in year_cfg["image_data_paths"]
                    }
                    for year_cfg in config["training_years"]
                ]
                mask_path = [data_folder / year_cfg["mask_path"] for year_cfg in config["training_years"]]
                # Check for per-year ROI paths
                if "roi_shape_path" in config["training_years"][0]:
                    roi_shape_path = [data_folder / year_cfg["roi_shape_path"] for year_cfg in config["training_years"]]
                    print(f"Using per-year ROI paths")
                else:
                    roi_shape_path = data_folder / config["roi_shape_path"]
            else:
                # Single year mode (backward compatible)
                image_data_paths = {
                    key: data_folder / config["image_data_paths"][key]
                    for key in input_bands
                    if key in config["image_data_paths"]
                }
                mask_path = data_folder / config["mask_path"]
                roi_shape_path = data_folder / config["roi_shape_path"]
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    else:
        experiment_name = "TERRATORCH_TEST_DANIEL_CLAY15_FIX"
        # Data paths
        data_folder = Path("/dibhome/daniel.kulmer/HabitAlp2.0/Originaldaten")
        mask_path = data_folder / "processed/mask/classes_v4_2013.tif"
        roi_shape_path = data_folder / "roi/habitalp_2013_boundary.gpkg"
        class_weights = [
            3.05754852e-09,
            5.61588104e-08,
            2.55059813e-07,
            1.31145305e-08,
            1.18684274e-08,
            2.41300286e-09,
            1.39145003e-08,
            2.71112867e-08,
            1.55613875e-08,
            8.56254672e-09,
            3.42943184e-08,
            4.17894324e-09,
            4.42492833e-09,
            2.31283622e-08,
            1.43593138e-08,
            5.54229727e-09,
            1.03878198e-08,
            4.66937618e-08,
            1.60617068e-08,
            1.63495321e-08,
            2.86372574e-09,
            1.65979327e-08,
            6.29546752e-09,
            4.21235049e-08,
        ]
        ignore_index = None

        image_data_paths = {
            "rgb": data_folder / "processed/orthophoto_gis_stmk/flug_2013_2015_rgb.tif",
            "cir": data_folder / "processed/orthophoto_gis_stmk/falschfarben_2013_2015.tif",
            "dtm": data_folder / "processed/elevation_gis_stmk_2010-2012/dtm.tif",
            # "dsm": data_folder / "processed/elevation_waldstmk_2010-2012/dsm.tif",
            # "slope": data_folder / "processed/elevation_waldstmk_2010-2012/slope.tif",
            # "aspect": data_folder / "processed/elevation_waldstmk_2010-2012/aspect.tif",
            # "tri": data_folder / "processed/elevation_waldstmk_2010-2012/tri.tif",
            # "tpi": data_folder / "processed/elevation_waldstmk_2010-2012/tpi.tif",
            # "roughness": data_folder / "processed/elevation_waldstmk_2010-2012/roughness.tif",
            # "curvature": data_folder / "processed/elevation_waldstmk_2010-2012/curvature.tif",
            # "planform_curvature": data_folder
            # / "processed/elevation_waldstmk_2010-2012/planform_curvature.tif",
            # "profile_curvature": data_folder
            # / "processed/elevation_waldstmk_2010-2012/profile_curvature.tif",
            "ndsm": data_folder / "processed/elevation_gis_stmk_2010-2012/ndsm.tif",
        }
        n_classes = 23
        gpu_id = 0
        num_workers = 4
        matmul_precision = "high"
        precision = None
        class_names = None

        # Training parameters
        hyperparameters = {
            "batch_size": 32,
            "patch_size": 256,
            "learning_rate": 1e-4,
            "loss": "ce",
            "monitor_metric": "val/mIoU",
            "patience": 50,
            "res": None,
            "train_batches_per_epoch": 512,
            "val_batches_per_epoch": 64,
            "backbone": "prithvi_eo_v2_300",
            "input_bands": [
                "BLUE",
                "GREEN",
                "RED",
                "NIR_NARROW",
                "NDSM",
                "DTM",
                # "DSM",
                # "SLOPE",
                # "ASPECT",
                # "TRI",
                # "TPI",
                # "ROUGHNESS",
                # "CURVATURE",
                # "PLANFORM_CURVATURE",
                # "PROFILE_CURVATURE",
            ],
            "use_pretrained_model_weights": True,
            "min_epochs": 20,
            "max_epochs": 60,
        }
        model_factory = "ClayModelFactory"
        model_args_config = {}
        freeze_backbone = False

    experiment_dir_base = BASE_FOLDER / "models"
    fabric = Fabric(precision=precision)

    # reduce cudnn workspace allocation (can lower peak memory)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # free any cached GPU memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create datamodule
    datamodule = HabitAlp2DataModule(
        image_data_paths,
        mask_path,
        n_classes=n_classes,
        roi_shape_path=roi_shape_path,
        batch_size=hyperparameters["batch_size"],
        patch_size=(hyperparameters["patch_size"], hyperparameters["patch_size"]),
        num_workers=num_workers,
        res=hyperparameters["res"],
        train_batches_per_epoch=hyperparameters["train_batches_per_epoch"],
        val_batches_per_epoch=hyperparameters["val_batches_per_epoch"],
        class_names=class_names,
        ndsm_aug_config=config.get("ndsm_augmentation", {}).get("value", None) if args.config else None,
    )

    datamodule.setup(stage="fit")

    # Model configuration
    if model_factory == "ClayModelFactory":
        # EXACTLY match notebook band names (including dsm instead of ndsm!)
        band_mapping = {
            'r': 'red',
            'g': 'green',
            'b': 'blue',
            'nir': 'nir',
            'dtm': 'dtm',
            'ndsm': 'dsm'  # Notebook uses 'dsm' even though data has ndsm!
        }
        bands_clay = [band_mapping.get(b.lower(), b.lower()) for b in datamodule.dataset.bands]
        model_args = dict(
            backbone=hyperparameters["backbone"],
            pretrained=hyperparameters["use_pretrained_model_weights"],
            num_frames=1,
            num_classes=datamodule.n_classes + 1,
            bands=bands_clay,
            in_channels=len(datamodule.dataset.bands),
            decoder="FCNDecoder",
        )
    else:
        # Config-driven model_args for EncoderDecoderFactory / PrithviModelFactory
        model_args = dict(model_args_config)
        model_args["num_classes"] = datamodule.n_classes + 1

    # Create task
    # Normalize class weights if they exist and sum is too small
    if class_weights is not None:
        weight_sum = sum(class_weights)
        if weight_sum < 1.0:  # Weights are way too small
            print(f"WARNING: Class weights sum to {weight_sum:.2e}, normalizing to sum to {n_classes}")
            class_weights = [w * n_classes / weight_sum for w in class_weights]
            print(f"Normalized weights now sum to: {sum(class_weights):.4f}")

    task = ExtendedSemanticSegmentationTask(
        model_args,
        model_factory,
        loss=hyperparameters["loss"],
        lr=hyperparameters["learning_rate"],
        ignore_index=ignore_index,
        optimizer=hyperparameters.get("optimizer", "AdamW"),
        optimizer_hparams=hyperparameters.get("optimizer_hparams", {"weight_decay": 0.0001}),
        freeze_backbone=freeze_backbone,
        plot_on_val=False,
        class_names=datamodule.class_names,
        class_weights=class_weights,
    )

    # Set Internal precision of float32 matrix multiplications
    # See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision(matmul_precision)

    # Setup wandb logging — configurable via yaml key `wandb_project`
    wandb_project = config.get("wandb_project", {})
    if isinstance(wandb_project, dict):
        wandb_project = wandb_project.get("value", "semantic-segmentation-terratorch")
    if not wandb_project:
        wandb_project = "semantic-segmentation-terratorch"

    trainer = setup_training(
        experiment_name=experiment_name,
        experiment_dir=experiment_dir_base / experiment_name,
        min_epochs=hyperparameters["min_epochs"],
        max_epochs=hyperparameters["max_epochs"],
        gpu_id=gpu_id,
        patience=hyperparameters["patience"],
        wandb_project=wandb_project,
        wandb_key=wandb_key,
        monitor_metric=hyperparameters["monitor_metric"],
        log_model=False,
    )

    # Start training
    trainer.fit(model=task, datamodule=datamodule)

    # Evaluate using test set and log final metrics
    trainer.test(model=task, datamodule=datamodule)

    wandb.finish()

    return task


if __name__ == "__main__":
    main()
