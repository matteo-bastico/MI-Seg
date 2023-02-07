from typing import Sequence, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from monai.utils import optional_import
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.layers import DropPath
from torch.nn.modules.normalization import LayerNorm
from ..blocks.window_attention import WindowAttention
from ..utils.swin_utils import get_window_size, window_partition, window_reverse
from ..layers.utils import get_norm_layer
from networks.norms.conditional_instance_norm import _ConditionalInstanceNorm

__all__ = [
    "SwinTransformerBlock",
]

rearrange, _ = optional_import("einops", name="rearrange")


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            shift_size: Sequence[int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer: str = "GELU",
            use_checkpoint: bool = False,
            norm_type: Union[Tuple, str] = "layer",
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            norm_type: normalization layers type
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm_type = norm_type[0] if isinstance(norm_type, Tuple) else norm_type
        # Automatic adding normalized_shape for layer normalization
        if self.norm_type == "layer":
            if isinstance(norm_type, Tuple):
                norm_type[1]["normalized_shape"] = dim
            else:
                norm_type = (norm_type, {"normalized_shape": dim})

        self.norm1 = get_norm_layer(name=norm_type,
                                    spatial_dims=len(self.window_size),  # This is same as spatial dimensions
                                    channels=dim)
        self.norm2 = get_norm_layer(name=norm_type,
                                    spatial_dims=len(self.window_size),
                                    channels=dim)

        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix, modalities=None):
        x_shape = x.size()
        if len(x_shape) == 5:
            # Normalize
            if isinstance(self.norm1, LayerNorm):
                x = self.norm1(x)
            else:
                # All other norms types need rearrange
                x = rearrange(x, "n d h w c -> n c d h w")
                if isinstance(self.norm1, _ConditionalInstanceNorm):
                    x = self.norm1(x, modalities)
                else:
                    x = self.norm1(x)
                x = rearrange(x, "n c d h w -> n d h w c")

            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            # Normalize
            if isinstance(self.norm1, LayerNorm):
                x = self.norm1(x)
            else:
                # All other norms types need rearrange
                x = rearrange(x, "n h w c -> n c h w")
                if isinstance(self.norm1, _ConditionalInstanceNorm):
                    x = self.norm1(x, modalities)
                else:
                    x = self.norm1(x)
                x = rearrange(x, "n c h w -> n h w c")

            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x, modalities=None):
        x_shape = x.size()
        # Second normalization
        if len(x_shape) == 5:
            # Normalize
            if isinstance(self.norm2, LayerNorm):
                x_norm = self.norm2(x)
            else:
                # All other norms types need rearrange
                x_norm = rearrange(x, "n d h w c -> n c d h w")
                if isinstance(self.norm2, _ConditionalInstanceNorm):
                    x_norm = self.norm2(x_norm, modalities)
                else:
                    x_norm = self.norm2(x_norm)
                x_norm = rearrange(x_norm, "n c d h w -> n d h w c")

        elif len(x_shape) == 4:
            # Normalize
            if isinstance(self.norm2, LayerNorm):
                x_norm = self.norm2(x)
            else:
                # All other norms types need rearrange
                x_norm = rearrange(x, "n h w c -> n c h w")
                if isinstance(self.norm2, _ConditionalInstanceNorm):
                    x_norm = self.norm2(x_norm, modalities)
                else:
                    x_norm = self.norm2(x_norm)
                x_norm = rearrange(x_norm, "n c h w -> n h w c")

        return self.drop_path(self.mlp(x_norm))

    def load_from(self, weights, n_block, layer):
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def forward(self, x, mask_matrix, modalities=None):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, modalities)
        else:
            x = self.forward_part1(x, mask_matrix, modalities)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, modalities)
        else:
            x = x + self.forward_part2(x, modalities)
        return x