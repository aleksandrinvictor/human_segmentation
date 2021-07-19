import os
from glob import glob
from importlib import import_module
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, listconfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def get_instance(object_path: str) -> Callable:

    module_path, class_name = object_path.rsplit(".", 1)
    module = import_module(module_path)

    return getattr(module, class_name)


def load_augs(cfg: DictConfig) -> A.Compose:
    """
    Load albumentations
    Args:
        cfg:
    Returns:
        compose object
    """
    augs = []
    for a in cfg:
        if a["class_name"] == "albumentations.OneOf":  # type: ignore
            small_augs = []
            for small_aug in a["params"]:  # type: ignore
                # yaml can't contain tuples, so we need to convert manually
                params = {
                    k: (
                        v
                        if not isinstance(v, listconfig.ListConfig)
                        else tuple(v)
                    )
                    for k, v in small_aug["params"].items()  # type: ignore
                }

                aug = get_instance(small_aug["class_name"])(  # type: ignore
                    **params
                )
                small_augs.append(aug)
            aug = get_instance(a["class_name"])(small_augs)  # type: ignore
            augs.append(aug)

        else:
            if "params" in a.keys():  # type: ignore
                params = {
                    k: (v if type(v) != listconfig.ListConfig else tuple(v))
                    for k, v in a["params"].items()  # type: ignore
                }
            else:
                params = {}
            aug = get_instance(a["class_name"])(**params)  # type: ignore
            augs.append(aug)

    return A.Compose(augs)


class HumsegDataset(Dataset):
    def __init__(
        self,
        basepath: str,
        image_ids: List[str],
        transforms: A.Compose = None,
        split: str = "train",
    ) -> None:

        self.basepath = basepath
        self.image_ids = image_ids
        self.transforms = transforms
        self.split = split

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor, Optional[Tensor]]:

        file_id = self.image_ids[index]
        if self.split == "test":
            image_path = f"./data/{self.split}/{file_id}.jpg"
        else:
            image_path = os.path.join(
                self.basepath, f"{self.split}/{file_id}.jpg"
            )
            mask_path = os.path.join(
                self.basepath, f"{self.split}_mask/{file_id}.png"
            )

        # Load image
        image = self._load_image(image_path)

        # Load mask
        if self.split == "test":
            mask = None
        else:
            mask = self._load_image(mask_path, True)

        if self.transforms is not None:

            if self.split == "test":
                augmented = self.transforms(image=image)

                return augmented["image"]
            else:
                augmented = self.transforms(image=image, mask=mask)

                return augmented["image"], augmented["mask"]

        return torch.from_numpy(image), mask

    def _load_image(self, path: str, mask: bool = False) -> np.ndarray:
        """Helper function for loading image.
        If mask is loaded, it is loaded in grayscale (, 0) parameter.
        """
        if mask:
            img = cv2.imread(path, 0)
            img = img / 255
            img = np.expand_dims(img, -1)
        else:
            img = cv2.imread(path)

        return img


class HumsegDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.hparams = cfg

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit" or stage is None:

            train_paths = glob(
                os.path.join(self.hparams.data.basepath, "train/*.jpg")
            )
            train_ids = [Path(x).stem for x in train_paths]

            val_paths = glob(
                os.path.join(self.hparams.data.basepath, "valid/*.jpg")
            )
            val_ids = [Path(x).stem for x in val_paths]

            train_augs = load_augs(self.hparams.augmentation.train)
            val_augs = load_augs(self.hparams.augmentation.val)

            self.train_dataset = HumsegDataset(
                basepath=self.hparams.data.basepath,
                image_ids=train_ids,
                transforms=train_augs,
                split="train",
            )

            self.valid_dataset = HumsegDataset(
                basepath=self.hparams.data.basepath,
                image_ids=val_ids,
                transforms=val_augs,
                split="valid",
            )
        else:
            self.test_paths = glob("./data/test/*.jpg")
            test_ids = [Path(x).stem for x in self.test_paths]

            test_augs = load_augs(self.hparams.augmentation.test)

            self.test_dataset = HumsegDataset(
                basepath=self.hparams.data.basepath,
                image_ids=test_ids,
                transforms=test_augs,
                split="test",
            )

    def train_dataloader(self) -> DataLoader:
        num_workers = os.cpu_count()

        if num_workers is None:
            num_workers = 0

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.training.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
        )
        return train_dataloader

    def val_dataloader(self, shuffle: bool = True) -> DataLoader:
        num_workers = os.cpu_count()

        if num_workers is None:
            num_workers = 0

        val_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.training.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )
        return val_dataloader

    def test_dataloader(self) -> DataLoader:
        num_workers = os.cpu_count()

        if num_workers is None:
            num_workers = 0

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.training.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return test_dataloader
