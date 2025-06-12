from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any, Optional, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from monai.losses import MaskedLoss, DiceLoss
from monai.utils import LossReduction

__all__ = ["PartialLoss"]

# Default activation for background-sum
_ACTIVATION = {
    'softmax': True,
    'use_softmax': True,
    'sigmoid': False,
    'other_act': None
}

_WEIGHT_NAME = ['class_weight', 'weight', 'pos_weight']


def get_labeled_classes(target: torch.Tensor) -> List[Tensor]:
    """
    Returns the indices of non-empty channels (channels with at least one nonzero element).

    Args:
        target (torch.Tensor): A tensor of shape [B, C, *spatial_dims] when one-hot encoded or [B, 1, *spatial_dims].
    Returns:
        torch.Tensor: A list of 1D tensor containing indices of non-empty channels for each sample in the batch.
    """
    labeled_classes = []
    for target_ in target:
        if target_.shape[0] == 1:
            labeled_classes.append(
                torch.unique(target_, return_inverse=True, sorted=True)[0].to(torch.int)
            )
        else:
            indices = target_ > 0
            while len(indices.shape) > 1:
                indices = indices.any(dim=1)
            indices = indices.nonzero(as_tuple=True)[0]
            labeled_classes.append(indices.to(torch.int))  #, target[indices]
    return labeled_classes

class PartialLoss(_Loss):
    """
    This is a wrapper class for the loss functions of Monai. It allows for partial losses
    to be applied to both input and target.

    See Also:
        - :py:class:`monai.losses.MaskedDiceLoss`
    """

    def __init__(
            self,
            loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | _Loss,
            mode: str = "ignore",
            ensure_softmax: bool = False,
            regularize_unlabeled: bool = False,
            lambda_reg: float = 1.0,
            bg_channel: int = 0,
            reg_sigmoid: bool = False,
            reg_softmax: bool = False,
            reg_other_act: Callable | None = None,
            *loss_args: Any,
            **loss_kwargs: Any
    ) -> None:
        """
        Args:
            loss: loss function to be wrapped, this could be a loss class or an instance of a loss class.
            loss_args: arguments to the loss function's constructor if `loss` is a class.
            loss_kwargs: keyword arguments to the loss function's constructor if `loss` is a class.
        """
        super().__init__()
        self.loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            loss(*loss_args, **loss_kwargs) if inspect.isclass(loss) else loss
        )
        if not callable(self.loss):
            raise ValueError("The loss function is not callable.")

        self.mode: str = mode
        if self.mode not in ["ignore", "bg-sum"]:
            raise ValueError(f"The mode {self.mode} is not supported. Please choose from ['ignore', 'bg-sum'].")

        if ensure_softmax:
            self._ensure_softmax()

        self.bg_channel = bg_channel
        self.regularize_unlabeled = regularize_unlabeled
        self.lambda_reg = lambda_reg
        self.weight = self._get_weight()

        if self.regularize_unlabeled:
            self.reg = DiceLoss(
                softmax=reg_softmax or ensure_softmax,
                sigmoid=reg_sigmoid,
                other_act=reg_other_act,
            )

    def _ensure_softmax(self):
        # For main loss
        for act, expected in _ACTIVATION.items():
            if hasattr(self.loss, act):
                setattr(self.loss, act, expected)
        # For sub-losses, e.g. DiceCELoss or DiceFocalLoss
        for attr in dir(self.loss):
            attr_value = getattr(self.loss, attr)
            if isinstance(attr_value, _Loss):
                for act, expected in _ACTIVATION.items():
                    if hasattr(attr_value, act):
                        setattr(attr_value, act, expected)

    def _get_weight(self):
        weight_dict = {}
        # For main loss
        for weight_name in _WEIGHT_NAME:
            if hasattr(self.loss, weight_name):
                weight_dict[self.loss] = getattr(self.loss, weight_name)
        # For sub-losses, e.g. DiceCELoss or DiceFocalLoss
        for attr in dir(self.loss):
            attr_value = getattr(self.loss, attr)
            if isinstance(attr_value, _Loss):
                for weight_name in _WEIGHT_NAME:
                    if hasattr(attr_value, weight_name):
                        weight_dict[attr_value] = getattr(attr_value, weight_name)
        return weight_dict

    def _set_weight(self, weight_mask: Tensor):
        for loss, weight in self.weight.items():
            weight = torch.as_tensor(weight) * weight_mask if weight is not None else weight_mask.to(torch.float)
            for weight_name in _WEIGHT_NAME:
                if hasattr(loss, weight_name):
                    # TODO: some losses, like BCEWithLogitsLoss, have multiple weights,
                    #  we should modify only the one corresponding to class weighting
                    if isinstance(loss, nn.BCEWithLogitsLoss) and weight_name == "weight":
                        continue
                    setattr(loss, weight_name, weight)

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            labeled_classes: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            labeled_classes: the shape should be 1N or BN.
        """
        if labeled_classes is None:
            warnings.warn("No labeled_classes value specified for the PartialLoss. Inferring them from target.")
            labeled_classes = get_labeled_classes(target)

        if input.shape[0] != len(labeled_classes) and len(labeled_classes) != 1:
            raise ValueError(
                f"Length of labeled_classes ({len(labeled_classes)}) must be one or equal to input batch size ({input.shape[0]}).")

        # Ensure tensor
        labeled_classes = [torch.as_tensor(labeled_classes_) for labeled_classes_ in labeled_classes]

        loss_list = []
        for input_, target_, labeled_classes_ in zip(input, target, labeled_classes):
            loss_ = 0.0
            if labeled_classes_.shape[0] != input_.shape[0]:  # Al least one unlabeled
                class_mask = torch.zeros(input.shape[1], device=input.device, dtype=torch.bool)
                class_mask[labeled_classes_] = True
                self._set_weight(class_mask)
                if self.regularize_unlabeled:
                    self.reg.class_weight = ~class_mask
                    if target_.shape[0] == input_.shape[0]:
                        # Workaround to obtain a foreground mask
                        reg_mask = torch.max(target_[torch.arange(target_.shape[0]) != self.bg_channel], dim=0)[0]
                    else:
                        reg_mask = target_ > 0
                    reg_mask = reg_mask.expand_as(input_)
                    loss_ -= self.reg(input_.unsqueeze(0), reg_mask.unsqueeze(0))
                if self.mode == "bg-sum":
                    class_mask[self.bg_channel] = False  # Include background for torch.logsumexp
                    bg_sum = torch.logsumexp(input_[~class_mask], 0, keepdim=True)
                    input_ = input_.clone()  # Avoid in-place modification
                    input_[~class_mask] = torch.finfo(input_.dtype).min  # Do Not participate in the softmax
                    input_[self.bg_channel] = bg_sum  # Replace background logits
            loss_ += self.loss(input_.unsqueeze(0), target_.unsqueeze(0))
            loss_list.append(loss_)

        # TODO: some of monai.losses (e.g. DiceCELoss and DiceFocalLoss) do not set properly self.reduction leading
        #  to a default "mean" from torch.nn.modules.loss._Loss
        loss: torch.Tensor | list[torch.Tensor]
        if loss_list is not None:
            if len(loss_list[0].shape):  # Workaround for reduction "none" in self.loss
                loss = torch.cat(loss_list, dim=0)
            elif getattr(self.loss, "reduction") == LossReduction.MEAN:  # Reduction "mean" or "sum" in self.loss
                loss = torch.mean(torch.stack(loss_list))
            elif getattr(self.loss, "reduction") == LossReduction.SUM:
                loss = torch.sum(torch.stack(loss_list))
        return loss

