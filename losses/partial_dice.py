from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import warnings
import torch.nn as nn
from torch import Tensor

from torch.nn.modules.loss import _Loss
from monai.losses import DiceLoss, FocalLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, look_up_option, pytorch_after


def get_labeled_classes(target: torch.Tensor) -> tuple[Tensor, Tensor] | Any:
    """
    Returns the indices of non-empty channels (channels with at least one nonzero element).

    Args:
        target (torch.Tensor): A tensor of shape [C, *spatial_dims] when one-hot encoded or [1, *spatial_dims].
    Returns:
        torch.Tensor: A 1D tensor containing indices of non-empty channels.
        torch.Tensor: Target with new indices [0, C'] where C' is the number of non-empty channels.
    """
    if target.shape[0] == 1:
        return torch.unique(target, return_inverse=True, sorted=True)[0].to(torch.int)
    else:
        indices = target > 0
        while len(indices.shape) > 1:
            indices = indices.any(dim=1)
        indices = indices.nonzero(as_tuple=True)[0]
        return indices.to(torch.int)  #, target[indices]


class PartialDiceCELoss(_Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss`` and ``torch.nn.BCEWithLogitsLoss()``.
    In this implementation, two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        weight: torch.Tensor | None = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        label_smoothing: float = 0.0,
        mode: str = "ignore",
        regularize_unlabeled: bool = False,
        lambda_reg: float = 1.0
    ) -> None:
        """
        Args:
            ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` and ``weight`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``. only used by the `DiceLoss`, not for the `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            weight: a rescaling weight given to each class for cross entropy loss for `CrossEntropyLoss`.
                or a weight of positive examples to be broadcasted with target used as `pos_weight` for `BCEWithLogitsLoss`.
                See ``torch.nn.CrossEntropyLoss()`` or ``torch.nn.BCEWithLogitsLoss()`` for more information.
                The weight is also used in `DiceLoss`.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.
            label_smoothing: a value in [0, 1] range. If > 0, the labels are smoothed
                by the given factor to reduce overfitting.
                Defaults to 0.0.

        """
        super().__init__()
        assert mode in ["ignore", "merge"]
        self.mode = mode
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.reduction = reduction
        weight = torch.as_tensor(weight) if weight is not None else None
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor
        
        self.include_background = include_background

        dice_weight: torch.Tensor | None
        if weight is not None and not include_background:
            dice_weight = weight[1:]
        else:
            dice_weight = weight

        self.dice = DiceLoss(
            include_background=include_background,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            weight=dice_weight,
        )
        if pytorch_after(1, 10):
            self.cross_entropy = nn.CrossEntropyLoss(
                reduction=reduction, label_smoothing=label_smoothing, weight=weight
            )
        else:
            self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction, weight=weight)
        self.binary_cross_entropy = nn.BCEWithLogitsLoss(reduction=reduction, weight=weight)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.to_onehot_y = to_onehot_y
        self.dice_reg = DiceLoss(
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr
        ) if regularize_unlabeled else None
        self.lambda_reg = lambda_reg

    def set_weights(self, valid, channels):
        # Weight mask
        mask = torch.zeros(channels)
        mask[valid] = 1.0
        if self.class_weight is not None:
            self.cross_entropy.weight = mask * self.class_weight
            self.binary_cross_entropy.weight = mask * self.class_weight
            self.dice.class_weight = self.class_weight if self.class_weight.ndim == 0 else self.class_weight[valid]
        else:
            self.cross_entropy.weight = mask
            self.binary_cross_entropy.weight = mask

    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input logits and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)
        
        return self.cross_entropy(input, target)  # type: ignore[no-any-return]

    def bce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary CrossEntropy loss for the input logits and target in one single class.

        """
        if not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.binary_cross_entropy(input, target)  # type: ignore[no-any-return]

    def forward(self, input: torch.Tensor, target: torch.Tensor, labeled_classes=None) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].
            labeled_classes: B lists or tensors of labeled classes

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 (without one-hot encoding) nor the same as input.

        Returns:
            torch.Tensor: value of the loss.

        """
        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )

        loss_list = []
        for ind, (input_, target_) in enumerate(zip(input, target)):
            # Get labeled and unlabeled classes
            labeled_classes_ = labeled_classes[ind] if labeled_classes is not None else get_labeled_classes(target_)
            unlabeled_classes_ = [c for c in range(len(input_)) if c not in labeled_classes_]

            # BNH[WD] or B1H[WD]
            input_, target_ = input_.unsqueeze(0), target_.unsqueeze(0)

            # Set loss weights
            n_pred_ch = input_.shape[1]
            self.set_weights(labeled_classes_, n_pred_ch)

            ce_loss = self.ce(input_, target_) if n_pred_ch != 1 else self.bce(input_, target_)
            # Sum to total loss
            loss_ = self.lambda_ce * ce_loss

            # DiceLoss pre-processing
            if self.sigmoid:
                input_ = torch.sigmoid(input_)

            if self.softmax:
                if n_pred_ch == 1:
                    warnings.warn("single channel prediction, `softmax=True` ignored.")
                else:
                    input_ = torch.softmax(input_, 1)

            if self.other_act is not None:
                input_ = self.other_act(input_)

            if self.to_onehot_y:
                if n_pred_ch == 1:
                    warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
                else:
                    target_ = one_hot(target_, num_classes=n_pred_ch)

            if len(unlabeled_classes_) > 0:
                # Regularize unlabeled classes
                if self.dice_reg is not None:
                    # Generate mask as the union of all labeled organs
                    if n_pred_ch == 1:
                        warnings.warn("single channel prediction, `regularize_unlabeled=True` ignored.")
                    else:
                        # Minimize overlap between unlabeled regions and mask
                        mask = torch.max(target_[:, 1:], dim=1, keepdim=True)[0]
                        loss_ -= self.lambda_reg * self.dice_reg(
                            input_[:, unlabeled_classes_],
                            mask.expand_as(input_[:, unlabeled_classes_])
                        )

                # Filter input and target for Dice calculation
                target_ = target_[:, labeled_classes_]
                if self.mode == "ignore":
                    input_ = input_[:, labeled_classes_]
                    input_ /= (input_.sum(dim=1, keepdim=True) + torch.finfo(torch.float32).eps)
                elif self.mode == "merge":
                    # Merge all unlabeled to background after activation, assume background is first labeled class
                    new_bg = input_[:, 0] + input_[:, unlabeled_classes_].sum(dim=1, keepdim=True)
                    input_ = torch.cat((new_bg, input_[:, labeled_classes_[1:]]), dim=1)

            # Compute Dice
            dice_loss = self.dice(input_, target_)
            loss_ += self.lambda_dice * dice_loss
            loss_list.append(loss_)

        loss: torch.Tensor | list[torch.Tensor]
        if loss_list is not None:
            if self.reduction == LossReduction.MEAN:
                loss = torch.mean(torch.stack(loss_list))
            elif self.reduction == LossReduction.SUM:
                loss = torch.sum(torch.stack(loss_list))
        return loss


class PartialDiceFocalLoss(_Loss):
    """
    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Focal Loss is shown in ``monai.losses.FocalLoss``.

    ``gamma`` and ``lambda_focal`` are only used for the focal loss.
    ``include_background``, ``weight``, ``reduction``, and ``alpha`` are used for both losses,
    and other parameters are only used for dice loss.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        gamma: float = 2.0,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
        alpha: float | None = None,
        mode: str = "ignore",
        regularize_unlabeled: bool = False,
        lambda_reg: float = 1.0
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``.
                for example: `other_act = torch.tanh`. only used by the `DiceLoss`, not for `FocalLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            gamma: value of the exponent gamma in the definition of the Focal loss.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_focal: the trade-off weight value for focal loss. The value should be no less than 0.0.
                Defaults to 1.0.
            alpha: value of the alpha in the definition of the alpha-balanced Focal loss. The value should be in
                [0, 1]. Defaults to None.
        """
        super().__init__()
        assert mode in ["ignore", "merge"]
        self.mode = mode
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=weight,
        )
        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=False,
            gamma=gamma,
            weight=weight,
            alpha=alpha,
            reduction="none",
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.dice_reg = DiceLoss(
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr
        ) if regularize_unlabeled else None
        self.lambda_reg = lambda_reg
        weight = torch.as_tensor(weight) if weight is not None else None
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor

    def set_weights(self, valid, channels):
        # Weight mask
        mask = torch.zeros(channels)
        mask[valid] = 1.0
        if self.class_weight is not None:
            self.focal.class_weight = mask * self.class_weight
            self.dice.class_weight = self.class_weight if self.class_weight.ndim == 0 else self.class_weight[valid]
        else:
            self.focal.class_weight = mask

    def forward(self, input: torch.Tensor, target: torch.Tensor, labeled_classes=None) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD]. The input should be the original logits
                due to the restriction of ``monai.losses.FocalLoss``.
            target: the shape should be BNH[WD] or B1H[WD].
            labeled_classes:

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 (without one-hot encoding) nor the same as input.

        Returns:
            torch.Tensor: value of the loss.
        """
        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )

        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        loss_list = []
        for ind, (input_, target_) in enumerate(zip(input, target)):
            # Get labeled and unlabeled classes
            labeled_classes_ = labeled_classes[ind] if labeled_classes is not None else get_labeled_classes(target_)
            unlabeled_classes_ = [c for c in range(len(input_)) if c not in labeled_classes_]

            # BNH[WD] or B1H[WD]
            input_, target_ = input_.unsqueeze(0), target_.unsqueeze(0)

            # Set loss weights
            n_pred_ch = input_.shape[1]
            self.set_weights(labeled_classes_, n_pred_ch)

            # original logits due to the restriction of monai.losses.FocalLoss.
            focal_loss = self.focal(input_, target_)
            # Sum to total loss
            loss_ = self.lambda_focal * focal_loss

            # DiceLoss pre-processing
            if self.sigmoid:
                input_ = torch.sigmoid(input_)

            if self.softmax:
                if n_pred_ch == 1:
                    warnings.warn("single channel prediction, `softmax=True` ignored.")
                else:
                    input_ = torch.softmax(input_, 1)

            if self.other_act is not None:
                input_ = self.other_act(input_)

            if len(unlabeled_classes_) > 0:
                # Regularize unlabeled classes
                if self.dice_reg is not None:
                    # Generate mask as the union of all labeled organs
                    if n_pred_ch == 1:
                        warnings.warn("single channel prediction, `regularize_unlabeled=True` ignored.")
                    else:
                        # Minimize overlap between unlabeled regions and mask
                        mask = torch.max(target_[:, 1:], dim=1, keepdim=True)[0]
                        loss_ -= self.lambda_reg * self.dice_reg(
                            input_[:, unlabeled_classes_],
                            mask.expand_as(input_[:, unlabeled_classes_])
                        )

                # Filter input and target for Dice calculation
                target_ = target_[:, labeled_classes_]
                if self.mode == "ignore":
                    input_ = input_[:, labeled_classes_]
                    input_ /= (input_.sum(dim=1, keepdim=True) + torch.finfo(torch.float32).eps)
                elif self.mode == "merge":
                    # Merge all unlabeled to background after activation, assume background is first labeled class
                    new_bg = input_[:, 0] + input_[:, unlabeled_classes_].sum(dim=1, keepdim=True)
                    input_ = torch.cat((new_bg, input_[:, labeled_classes_[1:]]), dim=1)

            # Compute Dice
            dice_loss = self.dice(input_, target_)
            loss_ += self.lambda_dice * dice_loss
            loss_list.append(loss_)

        loss: torch.Tensor | list[torch.Tensor]
        if loss_list is not None:
            if self.reduction == LossReduction.NONE:  # Reduction "none" in loss_fn
                loss = torch.cat(loss_list, dim=0)
            elif self.reduction == LossReduction.MEAN:  # Reduction "mean" or "sum" in loss_fn
                loss = torch.mean(torch.stack(loss_list))
            elif self.reduction == LossReduction.SUM:
                loss = torch.sum(torch.stack(loss_list))
        return loss

'''
class PartialLossWrapper(nn.Module):
    """
    Class wrapper to ignore unlabeled classes.
    """
    def __init__(
            self,
            loss_fn: Callable,
            mode: str = "ignore",
            regularize_unlabeled: bool = False,
            lambda_reg: float = 1.0
    ):
        super().__init__()
        assert mode in ["ignore", "bg-collapse"]
        self.loss_fn = loss_fn
        self.mode = mode
        self.regularize_unlabeled = regularize_unlabeled
        self.lambda_reg = lambda_reg
        if self.regularize_unlabeled:
            self.dice_reg = DiceLoss()  # Default Dice for regularization
            # Get activation for regularization based on loss_fn
            # We need it here because we apply activation before filtering unlabeled channels
            self.sigmoid = self.get_attr("sigmoid", False)
            self.softmax = self.get_attr("softmax", False)
            self.other_act = self.get_attr("other_act", None)

    def get_attr(self, attr_name, default=None):
        """
        Finds the first occurrence of `attr_name` in `self.loss_fn` or any attribute
        of `self.loss_fn` that is an instance of `nn.modules.loss._Loss`.
        Stores the found value in `self.<attr_name>` and sets the attribute to `default_value`.

        Parameters:
        - attr_name (str): The attribute name to find and modify.
        - default_value: The value to set when the attribute is found.

        Returns:
        - The original value of `attr_name` if found, otherwise default_value.
        """
        # Check if self.loss_fn itself has the attribute
        if hasattr(self.loss_fn, attr_name):
            return getattr(self.loss_fn, attr_name)

        # Search in attributes that are instances of `_Loss`
        for attr in dir(self.loss_fn):
            attr_value = getattr(self.loss_fn, attr)
            if isinstance(attr_value, _Loss) and hasattr(attr_value, attr_name):
                return getattr(attr_value, attr_name)

        # If not found, set self.<attr_name> to default_value
        return default

    def forward(self, input: torch.Tensor, target: torch.Tensor, labeled_classes=None) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].
            labeled_classes:

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 (without one-hot encoding) nor the same as input.

        Returns:
            torch.Tensor: value of the loss.

        """
        loss_list = []
        for ind, (input_, target_) in enumerate(zip(input, target)):
            if labeled_classes is not None:
                labeled_classes_ = labeled_classes[ind]
                if target_.shape[0] == 1:
                    # Keep only labeled classes
                    target_ = target_ * torch.isin(target_, torch.tensor(labeled_classes_, device=target_.device))
                    # Map to consecutive indices for one-hot
                    target_ = torch.bucketize(target_, torch.tensor(labeled_classes_, device=target_.device))
                else:
                    target_ = target_[labeled_classes_]
            else:
                labeled_classes_, target_ = get_labeled_classes(target_)
                labeled_classes_ = labeled_classes_.to(torch.int)  # Ensure integer classes labels
            unlabeled_classes_ = [c for c in range(len(input_)) if c not in labeled_classes_]

            loss_ = 0.0
            if len(unlabeled_classes_) > 0:  # No filter if all classes are labelled
                # Unlabeled regularization
                if self.regularize_unlabeled:
                    # Apply activation (need probability for DICE) and keep unlabeled classes
                    input_act_ = self.apply_activation(input_)
                    input_act_ = input_act_[unlabeled_classes_]

                    # Generate mask as the union of all labeled organs
                    if target_.shape[0] == 1:
                        mask = (target_ > 0).int()
                    else:
                        mask = torch.max(target_[1: ], dim=0, keepdim=True)[0]  # No background

                    # Minimize overlap between unlabeled regions and mask
                    loss_ -= self.lambda_reg * self.dice_reg(
                        input_act_.unsqueeze(0),
                        mask.expand_as(input_act_).unsqueeze(0)
                    )

                if self.mode == "ignore":
                    # Use only labeled classes
                    input_ = input_[labeled_classes_]
                elif self.mode == "bg-collapse":
                    # Merge all unlabeled to background
                    new_bg = input_[0] + input_[unlabeled_classes_].sum(dim=0)
                    input_ = torch.cat((new_bg.unsqueeze(0), input_[labeled_classes_][1:]), dim=0)

            # Supervised segmentation loss
            loss_ += self.loss_fn(input_.unsqueeze(0), target_.unsqueeze(0))
            loss_list.append(loss_)

        # TODO: some of monai.losses (e.g. DiceCELoss and DiceFocalLoss) do not set properly self.reduction leading
        #  to a default "mean" from torch.nn.modules.loss._Loss
        loss: torch.Tensor | list[torch.Tensor]
        if loss_list is not None:
            if len(loss_list[0].shape):  # Reduction "none" in loss_fn
                loss = torch.cat(loss_list, dim=0)
            elif getattr(self.loss_fn, "reduction") == LossReduction.MEAN:  # Reduction "mean" or "sum" in loss_fn
                loss = torch.mean(torch.stack(loss_list))
            elif getattr(self.loss_fn, "reduction") == LossReduction.SUM:
                loss = torch.sum(torch.stack(loss_list))
        return loss

    def apply_activation(self, input: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        return input
'''