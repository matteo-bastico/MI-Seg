# From https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/blocks/transformerblock.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from monai.utils import optional_import
from monai.networks.blocks import CrossAttentionBlock, MLPBlock, SABlock
from networks.norms.conditional_instance_norm import _ConditionalInstanceNorm
from torch.nn.modules.normalization import LayerNorm

from ..layers.utils import get_norm_layer

rearrange, _ = optional_import("einops", name="rearrange")


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        causal: bool = False,
        sequence_length: int | None = None,
        with_cross_attention: bool = False,
        use_flash_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = True,
        norm_name: tuple | str = "layer"
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            mlp_dim (int): dimension of feedforward layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias(bool, optional): apply bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
            use_flash_attention: if True, use Pytorch's inbuilt flash attention for a memory efficient attention mechanism
                (see https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html).
            include_fc: whether to include the final linear layer. Default to True.
            use_combined_linear: whether to use a single linear layer for qkv projection, default to True.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.norm_name = norm_name[0] if isinstance(norm_name, tuple) else norm_name
        # Automatic adding normalized_shape for layer normalization
        if self.norm_name == "layer":
            if isinstance(norm_name, tuple):
                norm_name[1]["normalized_shape"] = hidden_size
            else:
                norm_name = (norm_name, {"normalized_shape": hidden_size})
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = get_norm_layer(name=norm_name,
                                    spatial_dims=1,  # spatial_dims is 1 because (B, N_p, F)
                                    channels=hidden_size)
        self.attn = SABlock(
            hidden_size,
            num_heads,
            dropout_rate,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
            causal=causal,
            sequence_length=sequence_length,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.norm2 = get_norm_layer(name=norm_name,
                                    spatial_dims=1,
                                    channels=hidden_size)
        self.with_cross_attention = with_cross_attention

        self.norm_cross_attn = get_norm_layer(name=norm_name,
                                    spatial_dims=1,
                                    channels=hidden_size)
        self.cross_attn = CrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias,
            causal=False,
            use_flash_attention=use_flash_attention,
        )

    def forward(self,
                x,
                modalities=None,
                context: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        if isinstance(self.norm1, _ConditionalInstanceNorm) and modalities is None:
            raise ValueError("Modalities must be passed to the forward step when norm_name is 'instance_cond'.")

        # First normalization
        if isinstance(self.norm1, LayerNorm):
            x_norm = self.norm1(x)
        else:
            # All other norms types need rearrange
            x_norm = rearrange(x, "n l c -> n c l")
            if isinstance(self.norm1, _ConditionalInstanceNorm):
                x_norm = self.norm1(x_norm, modalities)
            else:
                x_norm = self.norm1(x_norm)
            x_norm = rearrange(x_norm, "n c l -> n l c")
        # SABLock
        x = x + self.attn(x_norm, attn_mask=attn_mask)

        # CrossAttention
        if self.with_cross_attention:
            # Cross Attention normalization
            if isinstance(self.norm_cross_attn, LayerNorm):
                x_norm = self.norm_cross_attn(x)
            else:
                # All other norms types need rearrange
                x_norm = rearrange(x, "n l c -> n c l")
                if isinstance(self.norm_cross_attn, _ConditionalInstanceNorm):
                    x_norm = self.norm_cross_attn(x_norm, modalities)
                else:
                    x_norm = self.norm_cross_attn(x_norm)
                x_norm = rearrange(x_norm, "n c l -> n l c")
            x = x + self.cross_attn(x_norm, context=context)

        # Second normalization
        if isinstance(self.norm2, LayerNorm):
            x_norm = self.norm2(x)
        else:
            # All other norms types need rearrange
            x_norm = rearrange(x, "n l c -> n c l")
            if isinstance(self.norm2, _ConditionalInstanceNorm):
                x_norm = self.norm2(x_norm, modalities)
            else:
                x_norm = self.norm2(x_norm)
            x_norm = rearrange(x_norm, "n c l -> n l c")

        # MLP block
        x = x + self.mlp(x_norm)
        return x
