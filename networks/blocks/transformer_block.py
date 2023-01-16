# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn

from typing import Union, Tuple
from monai.utils import optional_import
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.selfattention import SABlock
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
        self, hidden_size: int,
            mlp_dim: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            qkv_bias: bool = False,
            norm_type: Union[Tuple, str] = "layer"
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            qkv_bias: apply bias term for the qkv linear layer

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.norm_type = norm_type[0] if isinstance(norm_type, Tuple) else norm_type
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias)

        # Automatic adding normalized_shape for layer normalization
        if self.norm_type == "layer":
            if isinstance(norm_type, Tuple):
                norm_type[1]["normalized_shape"] = hidden_size
            else:
                norm_type = (norm_type, {"normalized_shape": hidden_size})

        # spatial_dims is 1 because (B, N_p, F)
        self.norm1 = get_norm_layer(name=norm_type,
                                    spatial_dims=1,
                                    channels=hidden_size)
        self.norm2 = get_norm_layer(name=norm_type,
                                    spatial_dims=1,
                                    channels=hidden_size)

    def forward(self,
                x,
                modalities=None):
        if isinstance(self.norm1, _ConditionalInstanceNorm) and modalities is None:
            raise ValueError("Modalities must be passed to the forward step when encoder_norm_type is 'instance_cond'.")

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
        x = x + self.attn(x_norm)

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
