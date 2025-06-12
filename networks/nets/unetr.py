# From https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn

from monai.utils import ensure_tuple_rep

from ..blocks.dynunet_block import UnetOutBlock
from ..blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from ..nets.vit import ViT


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "conv",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        vit_norm: tuple | str = "layer",
        dec_norm: tuple | str = "instance",
        enc_norm: tuple | str = "instance",
        freeze_enc: bool = False
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 3.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.
            vit_norm: feature normalization type and arguments for ViT
            dec_norm: feature normalization type and arguments for decoder layers
            enc_norm: feature normalization type and arguments for encoder layers

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, dec_norm='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, dec_norm='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), proj_type='conv', dec_norm='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False

        self.vit_norm = vit_norm[0] if isinstance(vit_norm, tuple) else vit_norm
        self.dec_norm = dec_norm[0] if isinstance(dec_norm, tuple) else dec_norm
        self.enc_norm = enc_norm[0] if isinstance(enc_norm, tuple) else enc_norm

        # TODO: Add implementation for layer norm in Unetr Blocks
        if self.dec_norm == "layer" or self.enc_norm == "layer":
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
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
            norm_name=vit_norm
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=enc_norm,
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
            norm_name=enc_norm,
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
            norm_name=enc_norm,
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
            norm_name=enc_norm,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=dec_norm,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=dec_norm,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=dec_norm,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=dec_norm,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]
        # Freeze enc
        if freeze_enc:
            self.vit.requires_grad_(False)
            self.enc1.requires_grad_(False)
            self.enc2.requires_grad_(False)
            self.enc3.requires_grad_(False)
            self.enc4.requires_grad_(False)

    @classmethod
    def from_cfg(cls, cfg):
        """
        # Set normalizations
        vit_norm = parse_normalization(cfg.vit_norm,
                                            cfg.vit_norm_affine,
                                            cfg.num_groups if hasattr(cfg, "num_groups") else None,
                                            cfg.num_styles if hasattr(cfg, "num_styles") else None)
        dec_norm = parse_normalization(cfg.dec_norm,
                                                cfg.dec_norm_affine,
                                                cfg.num_groups if hasattr(cfg, "num_groups") else None,
                                                cfg.num_styles if hasattr(cfg, "num_styles") else None)
        enc_norm = parse_normalization(cfg.enc_norm,
                                                cfg.enc_norm_affine,
                                                cfg.num_groups if hasattr(cfg, "num_groups") else None,
                                                cfg.num_styles if hasattr(cfg, "num_styles") else None)
        """
        return cls(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            img_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z),
            feature_size=cfg.feature_size,
            hidden_size=cfg.hidden_size,
            mlp_dim=cfg.mlp_dim,
            num_heads=cfg.num_heads,
            proj_type=cfg.proj_type,
            conv_block=cfg.conv_block,
            res_block=cfg.res_block,
            dropout_rate=cfg.dropout_rate,
            spatial_dims=cfg.spatial_dims,
            qkv_bias=cfg.qkv_bias,
            vit_norm=tuple(cfg.vit_norm) if not isinstance(cfg.vit_norm, str) else cfg.vit_norm,
            dec_norm=tuple(cfg.dec_norm) if not isinstance(cfg.dec_norm, str) else cfg.dec_norm,
            enc_norm=tuple(cfg.enc_norm) if not isinstance(cfg.enc_norm, str) else cfg.enc_norm,
            freeze_enc=cfg.freeze_enc
        )

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self,
                x_in,
                modalities=None):
        # TODO: switch double input with single *data and upack it
        if (self.vit_norm == "instance_cond" or
            self.enc_norm == "instance_cond" or
            self.dec_norm == "instance_cond") and modalities is None:
            raise ValueError("Modalities must be passed to the forward step when norm_name is 'instance_cond'.")

        x, hidden_states_out = self.vit(x_in, modalities)
        enc1 = self.encoder1(x_in, modalities)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2), modalities)
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3), modalities)
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4), modalities)
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4, modalities)
        dec2 = self.decoder4(dec3, enc3, modalities)
        dec1 = self.decoder3(dec2, enc2, modalities)
        out = self.decoder2(dec1, enc1, modalities)
        return self.out(out)
