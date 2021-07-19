import argparse
import os
import pathlib
from glob import glob
from typing import cast

import numpy as np
import omegaconf
import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from torch.functional import Tensor

from humseg.dataset import HumsegDataModule
from humseg.html import get_html
from humseg.lit_module import LitSegmentation
from humseg.metrics import dice_score
from humseg.utils import encode_rle


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="./outputs/resnet34_basic")
    parser.add_argument("--output_path", default="./results")
    parser.add_argument("--threshold", default=0.5)

    return parser.parse_args()


class Predictor:
    def __init__(self, model_path: str, output_path: str) -> None:

        checkpoint_path = glob(os.path.join(model_path, "checkpoints/*.ckpt"))[
            0
        ]

        config_path = os.path.join(model_path, ".hydra/config.yaml")
        self.model = LitSegmentation.load_from_checkpoint(
            checkpoint_path=checkpoint_path
        )
        self.model.eval()

        cfg = omegaconf.OmegaConf.load(config_path)
        cfg = cast(DictConfig, cfg)

        cfg.data.basepath = "./data"
        self.dm = HumsegDataModule(cfg)

        self.output_path = output_path

        os.makedirs(output_path, exist_ok=True)

    def _predict_batch(self, batch: Tensor, threshold: float) -> np.ndarray:
        y = self.model(batch)

        y = torch.sigmoid(
            torch.nn.functional.interpolate(
                y, size=(320, 240), mode="bilinear"
            )
        )
        y = (y > threshold).type(y.dtype)

        y = y.detach().numpy().squeeze(axis=1) * 255

        return y

    def _predict(self, batch: Tensor, threshold: float) -> np.ndarray:
        y = self.model(batch)

        y = torch.sigmoid(y)
        y = (y > threshold).type(y.dtype)

        return y

    def predict_test(self, threshold: float) -> None:

        self.dm.setup("test")

        predictions = []
        for batch in self.dm.test_dataloader():

            batch_pred = self._predict_batch(batch, threshold)
            predictions.extend(batch_pred)

        _ = get_html(
            self.dm.test_paths,
            predictions,
            os.path.join(self.output_path, "result_test"),
        )

    def predict_valid(self, threshold: float) -> None:

        pred_template = pd.read_csv("./data/pred_valid_template.csv")

        self.dm.setup(stage="fit")

        predictions = []

        val_preds_to_score = []
        true_masks = []

        for batch_imgs, batch_masks in self.dm.val_dataloader(shuffle=False):

            batch_pred = self._predict_batch(batch_imgs, threshold)

            val_preds_to_score.extend(self._predict(batch_imgs, threshold))

            predictions.extend(batch_pred)
            true_masks.extend(batch_masks)

        rle_masks = [encode_rle(mask) for mask in predictions]

        preds_to_score = torch.unsqueeze(
            torch.cat(val_preds_to_score, axis=0), 1
        )
        true_masks = torch.unsqueeze(torch.cat(true_masks, axis=0), 1)

        val_dice_score = dice_score(preds_to_score, true_masks)
        print(f"val_dice_score: {val_dice_score}")

        pred_template["rle_mask"] = rle_masks

        pred_template.to_csv(
            os.path.join(self.output_path, "pred_valid.csv"), index=False
        )


if __name__ == "__main__":
    args = parse_args()

    path = pathlib.PurePath(args.model_path)

    print(f"Model: {path.name}")

    predictor = Predictor(args.model_path, args.output_path)

    predictor.predict_test(args.threshold)

    predictor.predict_valid(args.threshold)
