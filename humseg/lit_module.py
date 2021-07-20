import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from humseg.metrics import load_metrics


class LitSegmentation(pl.LightningModule):
    """Class for training segmentation models"""

    def __init__(self, cfg: DictConfig):
        super(LitSegmentation, self).__init__()

        self.cfg = cfg
        self.save_hyperparameters()
        self.model = instantiate(cfg.model)
        self.criterion = instantiate(cfg.criterion)

        self.metrics = load_metrics(cfg.metrics)

    def configure_optimizers(self):
        optimizer = instantiate(
            self.cfg.optimizer, params=self.model.parameters()
        )

        if (
            self.cfg.scheduler._target_
            == "torch.optim.lr_scheduler.OneCycleLR"
        ):
            scheduler = {
                "scheduler": instantiate(
                    self.cfg.scheduler,
                    optimizer=optimizer,
                    steps_per_epoch=len(self.train_dataloader()),
                ),
                "monitor": "val_loss",
            }
        else:
            scheduler = {
                "scheduler": instantiate(
                    self.cfg.scheduler, optimizer=optimizer
                ),
                "reduce_on_plateau": True,
                "monitor": "val_loss",
            }

        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        data, targets = batch

        out = self(data.float())

        train_loss = self.criterion(out, targets)

        with torch.no_grad():
            for name, metric in self.metrics.items():
                metric_value = metric(out, targets)

                self.log(
                    f"train_{name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

        # self.log("step", self.trainer.current_epoch)

        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return train_loss

    def validation_step(self, batch, batch_idx):

        data, targets = batch

        out = self(data.float())

        val_loss = self.criterion(out, targets)

        with torch.no_grad():
            for name, metric in self.metrics.items():
                metric_value = metric(out, targets)

                self.log(
                    f"val_{name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

        # self.log("step", self.trainer.current_epoch)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
