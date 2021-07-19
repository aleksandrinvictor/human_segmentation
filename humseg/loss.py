import torch
from torch import Tensor

from humseg.metrics import activation_mapping, dice_score


class SoftDiceLoss:
    def __init__(
        self,
        activation: str = None,
        eps: float = 1e-6,
    ) -> None:
        self.eps = eps

        self.activation = None

        if activation is not None:
            if activation in activation_mapping.keys():
                self.activation = activation_mapping[activation]()
            else:
                raise ValueError(
                    "Unknown activation, "
                    f"should be one of {activation_mapping.keys()}"
                )

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
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

        if self.activation is not None:
            input = self.activation(input)

        return 1 - dice_score(input, target, self.eps)


class BceDiceLoss:
    def __init__(self, bce_coef: float = 0.5, dice_coef: float = 0.5):

        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(activation="sigmoid")

        self.bce_coef = bce_coef
        self.dice_coef = dice_coef

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:

        return self.bce_coef * self.bce(
            y_pred, y_true
        ) + self.dice_coef * self.dice(y_pred, y_true)
