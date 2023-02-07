import itertools
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.utils import get_norm_layer
from torch.nn.modules.normalization import LayerNorm
from monai.utils import optional_import
from networks.norms.conditional_instance_norm import _ConditionalInstanceNorm

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "PatchMerging",
    "PatchMergingV2",
]


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_type: Union[Tuple, str] = "instance_cond", spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()

        self.norm_type = norm_type[0] if isinstance(norm_type, Tuple) else norm_type
        self.dim = dim
        if spatial_dims == 3:
            # Automatic adding normalized_shape for layer normalization
            if self.norm_type == "layer":
                if isinstance(norm_type, Tuple):
                    norm_type[1]["normalized_shape"] = 8 * dim
                else:
                    norm_type = (norm_type, {"normalized_shape": 8 * dim})

            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = get_norm_layer(name=norm_type, spatial_dims=spatial_dims, channels=8 * dim)

        elif spatial_dims == 2:
            # Automatic adding normalized_shape for layer normalization
            if self.norm_type == "layer":
                if isinstance(norm_type, Tuple):
                    norm_type[1]["normalized_shape"] = 4 * dim
                else:
                    norm_type = (norm_type, {"normalized_shape": 4 * dim})

            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = get_norm_layer(name=norm_type, spatial_dims=spatial_dims, channels=4 * dim)

    def forward(self, x, modalities=None):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )
            # Normalize
            if isinstance(self.norm, LayerNorm):
                x = self.norm(x)
            else:
                # All other norms types need rearrange
                x = rearrange(x, "n d h w c -> n c d h w")
                if isinstance(self.norm, _ConditionalInstanceNorm):
                    x = self.norm(x, modalities)
                else:
                    x = self.norm(x)
                x = rearrange(x, "n c d h w -> n d h w c")

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)
            # Normalize
            if isinstance(self.norm, LayerNorm):
                x = self.norm(x)
            else:
                # All other norms types need rearrange
                x = rearrange(x, "n h w c -> n c h w")
                if isinstance(self.norm, _ConditionalInstanceNorm):
                    x = self.norm(x, modalities)
                else:
                    x = self.norm(x)
                x = rearrange(x, "n c h w -> n h w c")

        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x, modalities=None):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")

        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)

        # x = self.norm(x)
        # Normalize -> can only be len(x_shape) == 5
        if isinstance(self.norm, LayerNorm):
            x = self.norm(x)
        else:
            # All other norms types need rearrange
            x = rearrange(x, "n d h w c -> n c d h w")
            if isinstance(self.norm, _ConditionalInstanceNorm):
                x = self.norm(x, modalities)
            else:
                x = self.norm(x)
            x = rearrange(x, "n c d h w -> n d h w c")
        x = self.reduction(x)
        return x