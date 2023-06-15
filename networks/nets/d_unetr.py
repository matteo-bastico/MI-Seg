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
import warnings
from typing import Sequence, Tuple, Union

import torch.nn as nn

from monai.utils import ensure_tuple_rep

from networks.norms.utils import parse_normalization
from ..blocks.dynunet_block import UnetOutBlock
from ..blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from ..nets.vit import ViT


class D_UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: Union[Sequence[int], int],
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "conv",
            conv_block: bool = True,
            res_block: bool = True,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            qkv_bias: bool = False,
            vit_norm_name: Union[Tuple, str] = "layer",
            decoder_norm_name: Union[Tuple, str] = "instance",
            encoder_norm_name: Union[Tuple, str] = "instance",
            freeze_encoder: bool = False
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block
            vit_norm_name: feature normalization type and arguments for ViT
            decoder_norm_name: feature normalization type and arguments for decoder layers
            encoder_norm_name: feature normalization type and arguments for encoder layers

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norms
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, decoder_norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norms
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, decoder_norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norms
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', decoder_norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        # We use 15 layers now because also the first latent space come form ViT
        self.num_layers = 15
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        # Here the big change, classification is True, we add the class token and the MLP head
        self.classification = True
        # There are two classes: Source and Target
        # The modality used to train is the source, all the others are Target
        # We try to push the vit_encoder to learn common latent spaces for Source and Target(s)
        self.num_classes = 2
        # If Tuple is (norm_type, args1, arg2, ...) else norm_type
        self.vit_norm_name = vit_norm_name[0] if isinstance(vit_norm_name, Tuple) else vit_norm_name
        # Here only the normalization of the ViT should be conditioned
        if self.vit_norm_name != "instance_cond":
            warnings.warn("Conditional normalization is suggested in the ViT backbone for D-UNETR model.")
        # These two normalizations can be just instance normalizations
        self.decoder_norm_name = decoder_norm_name[0] if isinstance(decoder_norm_name, Tuple) else decoder_norm_name
        self.encoder_norm_name = encoder_norm_name[0] if isinstance(encoder_norm_name, Tuple) else encoder_norm_name

        # TODO: Add implementation for layer norm in Unetr Blocks
        if self.decoder_norm_name == "layer" or self.encoder_norm_name == "layer":
            raise ValueError("Layer normalization not yet implemented for encoder and decoder blocks, please "
                             "select another normalization.")

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            num_classes=self.num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            norm_type=vit_norm_name,
            classification_reverse_gradient=True,
            alpha_reversal=1.,
            post_activation="Softmax"
        )
        self.encoder1 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size,
            num_layer=3,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=encoder_norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=encoder_norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=encoder_norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=encoder_norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]
        # Freeze Encoder
        if freeze_encoder:
            self.vit.requires_grad_(False)
            self.encoder1.requires_grad_(False)
            self.encoder2.requires_grad_(False)
            self.encoder3.requires_grad_(False)
            self.encoder4.requires_grad_(False)

    @classmethod
    def from_argparse_args(cls, args):
        # Set normalizations
        vit_norm_name = parse_normalization(args.vit_norm_name,
                                            not args.vit_norm_no_affine,
                                            args.num_groups,
                                            args.num_styles)
        decoder_norm_name = parse_normalization(args.decoder_norm_name,
                                                not args.decoder_norm_no_affine,
                                                args.num_groups,
                                                args.num_styles)
        encoder_norm_name = parse_normalization(args.encoder_norm_name,
                                                not args.encoder_norm_no_affine,
                                                args.num_groups,
                                                args.num_styles)
        return cls(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            conv_block=not args.no_conv_block,
            res_block=not args.no_res_block,
            dropout_rate=args.dropout_rate,
            spatial_dims=args.spatial_dims,
            qkv_bias=args.qkv_bias,
            vit_norm_name=vit_norm_name,
            decoder_norm_name=decoder_norm_name,
            encoder_norm_name=encoder_norm_name,
            freeze_encoder=args.freeze_encoder
        )

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self,
                x_in,
                modalities=None,
                return_classification=True
                ):
        # TODO: switch double input with single *data and upack it
        if (self.vit_norm_name == "instance_cond" or
            self.encoder_norm_name == "instance_cond" or
            self.decoder_norm_name == "instance_cond") and modalities is None:
            raise ValueError("Modalities must be passed to the forward step when encoder_norm_type is 'instance_cond'.")
        # Now here x has to be used for the adversarial loss
        # The old x is the hidden_states_out[12]
        pred, hidden_states_out = self.vit(x_in, modalities)
        # Remember that now first token in hidden states is [cls]
        # The hidden states should be 2, 5, 8, 11, 14
        # Not 3, 6, 9, 12 as in the original implementation (12 is out of index)
        x1 = hidden_states_out[2][:, 1:]
        enc1 = self.encoder1(self.proj_feat(x1), modalities)
        x2 = hidden_states_out[5][:, 1:]
        enc2 = self.encoder2(self.proj_feat(x2), modalities)
        x3 = hidden_states_out[8][:, 1:]
        enc3 = self.encoder3(self.proj_feat(x3), modalities)
        x4 = hidden_states_out[11][:, 1:]
        enc4 = self.encoder4(self.proj_feat(x4), modalities)
        dec4 = self.proj_feat(hidden_states_out[-1][:, 1:])
        dec3 = self.decoder5(dec4, enc4, modalities)
        dec2 = self.decoder4(dec3, enc3, modalities)
        dec1 = self.decoder3(dec2, enc2, modalities)
        out = self.decoder2(dec1, enc1, modalities)
        if return_classification:
            return pred, self.out(out)
        else:
            return self.out(out)

