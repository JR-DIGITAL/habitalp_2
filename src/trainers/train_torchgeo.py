import sys
from pathlib import Path

BASE_FOLDER = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_FOLDER))

try:
    import osgeo  # Ensures gdal is configurated correctly
except ImportError:
    pass  # GDAL bindings are optional; rasterio brings its own GDAL
import argparse
import pandas as pd
import torch
import wandb
import yaml

from src.data.datamodules.habitalp import HabitAlp2DataModule
from src.trainers import setup_training, MultiClassSemanticSegmentationTask
from src.trainers.utils import load_config_from_yaml


def train_model(
    config: dict,
    logging: str="remote",
):
    """Prepare datamodule and SemanticSegmentationTask and perform model training while logging to wandb.

    Args:
        config (dict): Config dictionary.
        logging (str, optional): Either 'local' or 'remote' for W&B logging. Defaults to "remote".

    Returns:
        (HabitAlp2DataModule, SemanticSegmentationTask, Trainer): datamodule, task, trainer
    """
    hyperparameters = config["hyperparameters"]

    # Location to create experiment data folder inside
    experiment_dir = BASE_FOLDER / "models" / config["name"]

    datamodule = get_datamodule(
        config=config
    )
    config["in_channels"] = datamodule.in_channels

    task = MultiClassSemanticSegmentationTask(
        model=hyperparameters["model"],
        backbone=hyperparameters["backbone"],
        weights=hyperparameters["use_pretrained_model_weights"],
        in_channels=config["in_channels"],
        task="multiclass" if config["num_labels"] > 1 else "binary",
        num_classes=config["num_labels"]+1 if config["num_labels"] > 1 else None,
        loss=hyperparameters["loss"],
        class_weights=torch.Tensor(config["precalculated_weights"]) if config["precalculated_weights"] is not None else None,
        ignore_index=config["ignore_index"],
        lr=hyperparameters["learning_rate"],
        patience=hyperparameters["patience"],
    )

    # Set Internal precision of float32 matrix multiplications
    # See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision(config["matmul_precision"])

    trainer = setup_training(
        experiment_name=config["name"],
        experiment_dir=experiment_dir,
        min_epochs=hyperparameters["min_epochs"],
        max_epochs=hyperparameters["max_epochs"],
        gpu_id=config["gpu_id"],
        patience=hyperparameters["patience"],
        wandb_project=config["wandb_project"],
        logging=logging,
        wandb_key=config["wandb_key"],
        monitor_metric=hyperparameters["monitor_metric"],
        log_model=False,
    )

    # Train model
    _ = trainer.fit(model=task, datamodule=datamodule)

    # Area of training data for plots
    if config["train_data_area_path"]:
        train_data_area_path = Path(config["data_folder"]) / config["train_data_area_path"]
        train_data_area = (pd.read_csv(train_data_area_path)["Train"] / 10000).to_list()
    else:
        train_data_area = None

    # Evaluate using test set and log final metrics
    trainer.test(model=task, datamodule=datamodule)
    task.log_final_metrics(class_names=datamodule.class_names.copy(), train_data_area=train_data_area)

    # Save model
    torch.save(task.model.state_dict(), experiment_dir / f"{config['name']}.pt")

    # Add more config parameters to wandb
    trainer.logger.experiment.notes = config["notes"]
    # Add accelerator to config
    trainer.logger.experiment.config["accelerator"] = trainer.accelerator.name()

    wandb.finish()

    # Save model paths
    best_model_path = Path(trainer.checkpoint_callback.best_model_path).name
    model_paths = {
        "best_model_path": best_model_path,
        "last_model_path": "last.ckpt",
        "wandb_project": config["wandb_project"],
        "wandb_run_id": trainer.logger.experiment.id,
    }
    with open(experiment_dir / "model_paths.yaml", "w") as file:
        yaml.dump(model_paths, file)

    return datamodule, task, trainer


def get_datamodule(
    config: dict,
    prediction_year: int | None=None,
):
    """Get datamodule for semantic segmentation.

    Args:
        config (dict): Configuration for the model
        prediction_year (int, optional): Which year to predict on (either 2020 or 2024)

    Returns:
        HabitAlp2DataModule: Data module
    """
    hyperparameters = config.get("hyperparameters", config) # If called from sweep, hyperparameters are in config
    data_folder = Path(config["data_folder"])

    image_data_paths = config["image_data_paths"]
    image_data_paths = {key: data_folder / image_data_paths[key] for key in hyperparameters["input_bands"] if key in image_data_paths}
    mask_path = data_folder / config["mask_path"]
    roi_shape_path = data_folder / config["roi_shape_path"]
    if prediction_year is not None:
        image_data_paths_pred = config.get(f"image_data_paths_pred_{prediction_year}")
        image_data_paths_pred = {key: data_folder / image_data_paths_pred[key] for key in hyperparameters["input_bands"] if key in image_data_paths_pred}
    else:
        image_data_paths_pred = None

    return HabitAlp2DataModule(
        image_data_paths,
        mask_path,
        n_classes=config["num_labels"],
        roi_shape_path=roi_shape_path,
        batch_size=hyperparameters["batch_size"],
        patch_size=(hyperparameters["patch_size"], hyperparameters["patch_size"]),
        num_workers=config["num_dataloader_workers"],
        res=hyperparameters["res"],
        train_batches_per_epoch=hyperparameters["train_batches_per_epoch"],
        val_batches_per_epoch=hyperparameters["val_batches_per_epoch"],
        class_names=config["class_names"],
        image_paths_pred=image_data_paths_pred,
    )


def main():
    parser = argparse.ArgumentParser()
    # Location of configuration file
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to YAML config file with data paths and parameters",
        default="config-defaults.yaml",
    )

    args = parser.parse_args()

    # Load configuration file
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Look for config relative to cwd
        script_dir = Path.cwd()
        config_path = script_dir / args.config

    if config_path.exists():
        config = load_config_from_yaml(config_path)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Call training function
    datamodule, task, trainer = train_model(
        config=config,
    )

if __name__ == '__main__':
    main()
