from typing import Any
import kornia.augmentation as K
import matplotlib.pyplot as plt
from terratorch.tasks import SemanticSegmentationTask
from terratorch.tasks.segmentation_tasks import to_segmentation_prediction
from torchgeo.datasets.utils import unbind_samples
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
import wandb

BATCH_IDX_FOR_VALIDATION_PLOTTING = 10


class ExtendedSemanticSegmentationTask(SemanticSegmentationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.test_metrics[0].add_metrics(
            {
                "Class_F1Score": MulticlassF1Score(
                    num_classes=self.hparams["model_args"]["num_classes"],
                    ignore_index=self.hparams["ignore_index"],
                    average="none",
                ),
                "ConfusionMatrix": MulticlassConfusionMatrix(
                    num_classes=self.hparams["model_args"]["num_classes"],
                    ignore_index=self.hparams["ignore_index"],
                    normalize="true",
                ),
            }
        )

    def forward(self, *args, **kwargs):
        # Lightning 2.6+ reicht oft 'bounds' und 'transform' durch.
        # Wir filtern alles heraus, was nicht vom Modell-Encoder 
        # (PixelWiseModel / GenericEncoder) erwartet wird.
        forbidden_keys = ["bounds", "transform"]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in forbidden_keys}
        
        return super().forward(*args, **filtered_kwargs)

    def log_test_metric_plots(self):
        """Generate plots for metrics calculated on the test set."""
        logger = self.logger.experiment

        class_names = self.hparams["class_names"]
        if class_names is not None:
            class_names.insert(self.hparams["ignore_index"], "No Data")

        # Plot Confusion Matrix
        figsize = self.hparams["model_args"]["num_classes"] * 0.75 + 4
        fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=100)
        self.test_metrics[0]["test/ConfusionMatrix"].plot(ax=ax, labels=class_names)
        [x.set_horizontalalignment("right") for x in ax.get_xticklabels()]
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        logger.log({f"test/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        # Plot Jaccard Index
        fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
        self.test_metrics[0]["test/IoU"].plot(ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        class_names = labels if class_names is None else class_names
        ax.legend(
            class_names, bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1, fontsize=8
        )
        ax.set_title("Jaccard Index")
        fig.tight_layout()
        logger.log({f"test/jaccard_each_class": wandb.Image(fig)})
        plt.close(fig)

        # Plot F1 Score
        fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
        self.test_metrics[0]["test/Class_F1Score"].plot(ax=ax)
        ax.legend(
            class_names, bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1, fontsize=8
        )
        ax.set_title("F1 Score")
        fig.tight_layout()
        logger.log({f"test/f1score_each_class": wandb.Image(fig)})
        plt.close(fig)

    def on_test_epoch_end(self) -> None:
        for metrics in self.test_metrics:
            computed_metrics = metrics.compute()
            averaged_metrics = {
                k: v
                for k, v in computed_metrics.items()
                if k not in ["test/Class_F1Score", "test/ConfusionMatrix"]
            }
            self.log_dict(averaged_metrics, sync_dist=True)
            self.log_test_metric_plots()
            metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Overwrite function from terratorch to support wandb figure logging

        Compute the validation loss and additional metrics.
        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = self.squeeze_ground_truth(batch["mask"])

        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output = self.handle_full_or_tiled_inference(x, self.tiled_inference_on_validation, **rest)

        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat_hard = to_segmentation_prediction(model_output)
        self.val_metrics.update(y_hat_hard, y)

        if self._do_plot_samples(batch_idx):
            try:
                datamodule = self.trainer.datamodule
                aug = K.AugmentationSequential( # <--- Added denormalization
                    K.Denormalize(datamodule.mean, datamodule.std),
                    data_keys=None,
                    keepdim=True,
                )
                batch = aug(batch)
                batch["prediction"] = y_hat_hard

                if isinstance(batch["image"], dict):
                    rgb_modality = getattr(datamodule, "rgb_modality", None) or list(batch["image"].keys())[0]
                    batch["image"] = batch["image"][rgb_modality]

                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.val_dataset.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    if hasattr(summary_writer, "add_figure"):
                        summary_writer.add_figure(f"image/{batch_idx}", fig, global_step=self.global_step)
                    elif hasattr(summary_writer, "log_figure"):
                        summary_writer.log_figure(
                            self.logger.run_id, fig, f"epoch_{self.current_epoch}_{batch_idx}.png"
                        )
                    elif hasattr(summary_writer, "log"): # <--- Added this block
                        summary_writer.log(
                            {f"val_prediction/{batch_idx}": wandb.Image(fig)}
                        )
            except ValueError:
                pass
            finally:
                plt.close()