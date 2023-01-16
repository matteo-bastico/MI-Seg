import torch
import warnings

from torch import Tensor
import torch.nn as nn
from typing import Union, List

__all__ = ['ConditionalInstanceNorm1d', 'ConditionalInstanceNorm2d', 'ConditionalInstanceNorm3d']


class _ConditionalInstanceNorm(nn.Module):
    def __init__(
        self,
        num_styles: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if not affine:
            warnings.warn("Ignored affine=False for ConditionalInstanceNorm1D, set to True")
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_styles = num_styles
        self.norms = nn.ModuleList([
            self._get_norm()(
                 num_features, eps, momentum, True, track_running_stats, **factory_kwargs)
            for _ in range(num_styles)
        ])

    def _get_norm(self):
        raise NotImplementedError

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _check_input_styles(self, input, styles):
        if input.dim() == self._get_no_batch_dim():
            if not isinstance(styles, (int, list, Tensor)) or (isinstance(styles, Tensor) and torch.numel(styles) != 1) \
                    or (isinstance(styles, list) and len(styles) != 1):
                raise ValueError('Expected one style when input is not a batch.')
        else:
            if not isinstance(styles, (list, Tensor)) or len(styles) != len(input):
                raise ValueError('Expected number of styles as batch size.')

    def _get_no_batch_dim(self):
        raise NotImplementedError

    def _handle_no_batch_input(self, input, style):
        if isinstance(style, list):
            style = style[0]
        elif isinstance(style, Tensor):
            style = style.item()
        return self.norms[style]((input.unsqueeze(0)).squeeze(0))

    def _apply_instance_norm(self, input, styles):
        return torch.stack([self.norms[styles[i]](input[i].unsqueeze(0)).squeeze(0) for i in range(len(styles))])

    def forward(self, input: Tensor, styles: Union[List, Tensor, int]) -> Tensor:
        self._check_input_dim(input)
        self._check_input_styles(input, styles)
        if input.dim() == self._get_no_batch_dim():
            return self._handle_no_batch_input(input, styles)

        return self._apply_instance_norm(input, styles)


class ConditionalInstanceNorm1d(_ConditionalInstanceNorm):
    def _get_norm(self):
        return nn.InstanceNorm1d

    def _get_no_batch_dim(self):
        return 2

    def _check_input_dim(self, input):
        if input.dim() not in (2, 3):
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class ConditionalInstanceNorm2d(_ConditionalInstanceNorm):
    def _get_norm(self):
        return nn.InstanceNorm2d

    def _get_no_batch_dim(self):
        return 3

    def _check_input_dim(self, input):
        if input.dim() not in (3, 4):
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class ConditionalInstanceNorm3d(_ConditionalInstanceNorm):
    def _get_norm(self):
        return nn.InstanceNorm3d

    def _get_no_batch_dim(self):
        return 4

    def _check_input_dim(self, input):
        if input.dim() not in (4, 5):
            raise ValueError('expected 4D or 5D input (got {}D input)'
                             .format(input.dim()))

