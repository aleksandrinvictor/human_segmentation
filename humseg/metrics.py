from typing import Any, Callable, Dict, Union
from importlib import import_module

import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import Tensor


activation_mapping = {"softmax": nn.Softmax, "sigmoid": nn.Sigmoid}


def get_instance(object_path: str) -> Callable:

    module_path, class_name = object_path.rsplit(".", 1)
    module = import_module(module_path)

    return getattr(module, class_name)


def load_metrics(cfg: DictConfig) -> Union[Dict[str, Callable], None]:
    """Load metrics

    Parameters
    ----------
    cfg: DictConfig
        metrics config

    Returns
    -------
    Dict[str, Callable]
    """

    if cfg is None:
        return None

    metrics: Dict[str, Callable] = {}

    for a in cfg:
        if isinstance(a, dict) and "params" in a.keys():
            params: Dict[str, Any] = {
                k: (v if type(v) != ListConfig else tuple(v))
                for k, v in a["params"].items()
            }
        else:
            params = {}
        metric = get_instance(a["class_name"])(**params)  # type: ignore

        metrics[a["name"]] = metric  # type: ignore

    return metrics


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


class IoU:
    def __init__(
        self,
        eps: float = 1e-7,
        activation: str = None,
        threshold: float = 0.5,
    ) -> None:

        self.threshold = threshold
        self.eps = eps

        self.activation = activation

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:

        if self.activation is not None:
            activation = activation_mapping[self.activation]

            input = activation(input)

        input = _threshold(input, threshold=self.threshold)

        return jaccard_score(input, target, eps=self.eps)


class Dice:
    def __init__(
        self,
        eps: float = 1e-7,
        activation: str = None,
        threshold: float = 0.5,
    ) -> None:

        self.threshold = threshold
        self.eps = eps

        self.activation = activation

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:

        if self.activation is not None:
            activation = activation_mapping[self.activation]

            input = activation(input)

        input = _threshold(input, threshold=self.threshold)

        return dice_score(input, target, eps=self.eps)


def jaccard_score(input: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Parameters
    ----------
    input: Tensor
        input tensor with shape (batch_size, num_classes, height, width)
        must sum to 1 over c channel (such as after softmax)
    target: Tensor
        one hot target tensor with shape
        (batch_size, num_classes, height, width)

    Returns
    -------
    Tensor
        mean jaccard score
    """

    intersection = torch.sum(input * target, axis=(2, 3))
    union = (
        torch.sum(input, axis=(2, 3))
        + torch.sum(target, axis=(2, 3))
        - intersection
    )
    return torch.mean((intersection + eps) / (union + eps))


def dice_score(input: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Parameters
    ----------
    input: Tensor
        input tensor with shape (batch_size, num_classes, height, width)
        must sum to 1 over c channel (such as after softmax)
    target: Tensor
        one hot target tensor with shape
        (batch_size, num_classes, height, width)

    Returns
    -------
    Tensor
        mean dice score
    """

    numerator = 2 * torch.sum(input * target, axis=(2, 3))
    denominator = torch.sum(input, axis=(2, 3)) + torch.sum(
        target, axis=(2, 3)
    )

    return torch.mean((numerator + eps) / (denominator + eps))
