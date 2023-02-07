from typing import Optional, Sequence, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.blocks.patch_embedding import PatchEmbed
from monai.utils import look_up_option, optional_import

from ..blocks.swin_transformer_block import SwinTransformerBlock
from ..blocks.patch_merging import PatchMerging, PatchMergingV2
from networks.utils.swin_utils import get_window_size, compute_mask

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "BasicLayer",
    "SwinTransformer",
    "MERGING_MODE",
]


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        norm_type: Union[Tuple, str] = "layer"
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.norm_type = norm_type[0] if isinstance(norm_type, Tuple) else norm_type
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_type=norm_type if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
                norm_type=norm_type
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if self.norm_type == "layer":
                if len(x_shape) == 5:
                    n, ch, d, h, w = x_shape
                    x = rearrange(x, "n c d h w -> n d h w c")
                    x = F.layer_norm(x, [ch])
                    x = rearrange(x, "n d h w c -> n c d h w")
                elif len(x_shape) == 4:
                    n, ch, h, w = x_shape
                    x = rearrange(x, "n c h w -> n h w c")
                    x = F.layer_norm(x, [ch])
                    x = rearrange(x, "n h w c -> n c h w")
            elif self.norm_type == "instance" or self.norm_type == "instance_cond":
                x = F.instance_norm(x)
            # TODO: Add implementation for group and batch norm
            '''
            elif self.norm_type[0] == "group":
                x = F.group_norm(x, num_groups=self.norm_type[1]["num_groups"])
            elif self.norm_type[0] == "batch":
                # IDK if it is correct to set running mean and var to None (I tried on the notebook and seem fine)
                x = F.batch_norm(x, None, None, training=True)
            '''
        return x

    def forward(self, x, normalize=True, modalities=None):
        x0 = self.patch_embed(x, modalities)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous(), modalities=modalities)
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous(), modalities=modalities)
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous(), modalities=modalities)
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous(), modalities=modalities)
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
        norm_type: Union[Tuple, str] = "layer"
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    use_checkpoint=use_checkpoint,
                    norm_type=norm_type
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_type=norm_type, spatial_dims=len(self.window_size))

    def forward(self, x, modalities=None):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask, modalities=modalities)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x, modalities=modalities)
            x = rearrange(x, "b d h w c -> b c d h w")

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask, modalities=modalities)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x, modalities=modalities)
            x = rearrange(x, "b h w c -> b c h w")
        return x