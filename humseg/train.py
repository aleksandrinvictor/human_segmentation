import logging
import os

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything


@hydra.main(config_path="./../conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    logger.info("Starting run:")
    logger.info(f"Model: {cfg.model._target_}")

    # Setup device
    if "device" in cfg.common:
        device = cfg.common.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    seed_everything(cfg.common.seed)

    datamodule = instantiate(cfg.datamodule, cfg=cfg)

    pl_logger = instantiate(
        cfg.logger,
        save_dir="../",
        name=f"{cfg.training.model_id}",
        group=f"{cfg.training.model_id}",
        reinit=True,
    )
    os.makedirs("./checkpoints", exist_ok=True)

    early_stop_callback = instantiate(cfg.callbacks.early_stopping)
    checkpoint_callback = instantiate(
        cfg.callbacks.checkpoint,
        dirpath="./checkpoints",
    )

    model = instantiate(cfg.lit_module, cfg=cfg)

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        gpus=1,
        profiler=cfg.training.profiler,
        default_root_dir=os.getcwd(),
        logger=pl_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)

    wandb.finish()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
