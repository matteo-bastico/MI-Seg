# Fromm https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py

from __future__ import annotations
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from ..blocks.dynunet_block import UnetOutBlock
from ..blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from ..blocks.patch_merging import PatchMerging, PatchMergingV2
from ..nets.swin_transformer import SwinTransformer
from monai.utils.deprecate_utils import deprecated_arg


rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "SwinUNETR",
    "MERGING_MODE",
]

MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and "
                   "checks on the input size are run during forward().",
    )
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        patch_size: int = 2,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        feature_size: int = 24,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str | nn.Module = "merging",
        vit_norm: tuple | str = "layer",
        dec_norm: tuple | str = "instance",
        enc_norm: tuple | str = "instance",
        freeze_enc: bool = False,
        use_v2: bool = False,
    ) -> None:
        """
        Args:
            img_size: spatial dimension of input image.
                This argument is only used for checking that the input image size is divisible by the patch size.
                The tensor passed to forward() can have a dynamic shape as long as its spatial dimensions are divisible by 2**5.
                It will be removed in an upcoming version.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            patch_size: size of the patch token.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            norm_layer: normalization layer.
            patch_norm: whether to apply normalization to the patch embedding. Default is False.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self.patch_size = patch_size

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(window_size, spatial_dims)

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.vit_norm = vit_norm[0] if isinstance(vit_norm, tuple) else vit_norm
        self.dec_norm = dec_norm[0] if isinstance(dec_norm, tuple) else dec_norm
        self.enc_norm = enc_norm[0] if isinstance(enc_norm, tuple) else enc_norm

        # TODO: Add implementation for layer norm in Unetr Blocks
        if self.dec_norm == "layer" or self.enc_norm == "layer":
            raise ValueError("Layer normalization not yet implemented for encoder and decoder blocks, please "
                             "select another normalization.")

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            norm_name=vit_norm,
            use_v2=use_v2,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=enc_norm,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=enc_norm,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=enc_norm,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=enc_norm,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=enc_norm,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=dec_norm,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=dec_norm,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=dec_norm,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=dec_norm,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=dec_norm,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        # Freeze Encoder
        if freeze_enc:
            self.swinViT.requires_grad_(False)
            self.enc1.requires_grad_(False)
            self.enc2.requires_grad_(False)
            self.enc3.requires_grad_(False)
            self.enc4.requires_grad_(False)
            self.enc10.requires_grad_(False)

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
        if len(cfg.depth_swin_block) == 1:
            depths = (cfg.depth_swin_block[0],) * 4
        else:
            assert len(cfg.depth_swin_block) == 4, "The length of depth_swin_block should be 4"
            depths = cfg.depth_swin_block
        num_heads = tuple(2**i*cfg.num_heads for i in range(0, 4))
        """
        return cls(
            img_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z),
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            patch_size=cfg.patch_size,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            qkv_bias=cfg.qkv_bias,
            mlp_ratio=cfg.mlp_ratio,
            feature_size=cfg.feature_size,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            dropout_path_rate=cfg.dropout_path_rate,
            normalize=cfg.normalize,
            patch_norm=cfg.patch_norm,
            use_checkpoint=cfg.use_checkpoint,
            spatial_dims=cfg.spatial_dims,
            downsample=cfg.downsample,
            vit_norm=tuple(cfg.vit_norm) if not isinstance(cfg.vit_norm, str) else cfg.vit_norm,
            dec_norm=tuple(cfg.dec_norm) if not isinstance(cfg.dec_norm, str) else cfg.dec_norm,
            enc_norm=tuple(cfg.enc_norm) if not isinstance(cfg.enc_norm, str) else cfg.enc_norm,
            freeze_enc=cfg.freeze_enc,
            use_v2=cfg.use_v2,
        )

    def load_from(self, weights):
        layers1_0: BasicLayer = self.swinViT.layers1[0]  # type: ignore[assignment]
        layers2_0: BasicLayer = self.swinViT.layers2[0]  # type: ignore[assignment]
        layers3_0: BasicLayer = self.swinViT.layers3[0]  # type: ignore[assignment]
        layers4_0: BasicLayer = self.swinViT.layers4[0]  # type: ignore[assignment]
        wstate = weights["state_dict"]

        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(wstate["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(wstate["module.patch_embed.proj.bias"])
            for bname, block in layers1_0.blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")  # type: ignore[operator]

            if layers1_0.downsample is not None:
                d = layers1_0.downsample
                d.reduction.weight.copy_(wstate["module.layers1.0.downsample.reduction.weight"])  # type: ignore
                d.norm.weight.copy_(wstate["module.layers1.0.downsample.norm.weight"])  # type: ignore
                d.norm.bias.copy_(wstate["module.layers1.0.downsample.norm.bias"])  # type: ignore

            for bname, block in layers2_0.blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")  # type: ignore[operator]

            if layers2_0.downsample is not None:
                d = layers2_0.downsample
                d.reduction.weight.copy_(wstate["module.layers2.0.downsample.reduction.weight"])  # type: ignore
                d.norm.weight.copy_(wstate["module.layers2.0.downsample.norm.weight"])  # type: ignore
                d.norm.bias.copy_(wstate["module.layers2.0.downsample.norm.bias"])  # type: ignore

            for bname, block in layers3_0.blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")  # type: ignore[operator]

            if layers3_0.downsample is not None:
                d = layers3_0.downsample
                d.reduction.weight.copy_(wstate["module.layers3.0.downsample.reduction.weight"])  # type: ignore
                d.norm.weight.copy_(wstate["module.layers3.0.downsample.norm.weight"])  # type: ignore
                d.norm.bias.copy_(wstate["module.layers3.0.downsample.norm.bias"])  # type: ignore

            for bname, block in layers4_0.blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")  # type: ignore[operator]

            if layers4_0.downsample is not None:
                d = layers4_0.downsample
                d.reduction.weight.copy_(wstate["module.layers4.0.downsample.reduction.weight"])  # type: ignore
                d.norm.weight.copy_(wstate["module.layers4.0.downsample.norm.weight"])  # type: ignore
                d.norm.bias.copy_(wstate["module.layers4.0.downsample.norm.bias"])  # type: ignore

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in, modalities=None):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize, modalities)
        enc0 = self.encoder1(x_in, modalities)
        enc1 = self.encoder2(hidden_states_out[0], modalities)
        enc2 = self.encoder3(hidden_states_out[1], modalities)
        enc3 = self.encoder4(hidden_states_out[2], modalities)
        dec4 = self.encoder10(hidden_states_out[4], modalities)
        dec3 = self.decoder5(dec4, hidden_states_out[3], modalities)
        dec2 = self.decoder4(dec3, enc3, modalities)
        dec1 = self.decoder3(dec2, enc2, modalities)
        dec0 = self.decoder2(dec1, enc1, modalities)
        out = self.decoder1(dec0, enc0, modalities)
        logits = self.out(out)
        return logits
