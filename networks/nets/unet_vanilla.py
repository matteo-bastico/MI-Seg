import torch
import torch.nn as nn

from collections.abc import Sequence

from networks.norms.utils import parse_normalization
from networks.layers.factories import Act, Norm, Conv
from networks.blocks.convolutions import Convolution, ResidualUnit
from networks.layers.simplelayers import SequentialWIthModalities


class UNetVanilla(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            channels: Sequence[int],
            strides: Sequence[int],
            kernel_size: Sequence[int] | int = 3,
            up_kernel_size: Sequence[int] | int = 3,
            num_res_units: int = 0,
            act: tuple | str = Act.PRELU,
            norm_down: tuple | str = Norm.INSTANCE,
            norm_up: tuple | str = Norm.INSTANCE,
            dropout: float = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm_down = norm_down
        self.norm_up = norm_up
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        conv_type = Conv[Conv.CONV, self.dimensions]
        self.pre_conv = Convolution(
            spatial_dims=self.dimensions,
            in_channels=self.in_channels,
            out_channels=self.channels[0],
            kernel_size=self.kernel_size,
            strides=self.strides[0],
            conv_only=True
        )

        # Downsampling path
        self.down_path = nn.Sequential()
        self.saved_strides = []
        for scale in range(1, len(channels)):
            # With subunits 2 is what they call vanilla residual unit
            down_layer = SequentialWIthModalities()
            # Difference with the paper is "kernel size twice the stride for strided convolution."
            # But in their code kernel_size = stride for strided convolutions (row 79)
            # https://github.com/vanya2v/Multi-modal-learning/blob/master/dltk/core/modules/residual_units.py
            down_layer.append(ResidualUnit(
                self.dimensions,
                self.channels[scale-1],
                self.channels[scale],
                strides=self.strides[scale],
                kernel_size=self.kernel_size,
                subunits=2,
                act=self.act,
                norm=self.norm_down,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
                )
            )
            self.saved_strides.append(strides[scale])
            for i in range(1, self.num_res_units):
                # Stride 1 here
                down_layer.append(ResidualUnit(
                    self.dimensions,
                    self.channels[scale],
                    self.channels[scale],
                    strides=1,
                    kernel_size=self.kernel_size,
                    subunits=2,
                    act=self.act,
                    norm=self.norm_down,
                    dropout=self.dropout,
                    bias=self.bias,
                    adn_ordering=self.adn_ordering,
                )
                )
            self.down_path.append(down_layer)

        # Upsampling path
        self.up_path = nn.Sequential()
        for scale in range(len(self.channels) - 2, -1, -1):
            up_layer = nn.Sequential(
                nn.Upsample(scale_factor=self.saved_strides[scale]),
                ResidualUnit(
                    self.dimensions,
                    in_channels=channels[scale+1] + channels[scale],
                    out_channels=channels[scale],
                    strides=1,
                    kernel_size=self.kernel_size,
                    subunits=2,
                    act=self.act,
                    norm=self.norm_up,
                    dropout=self.dropout,
                    bias=self.bias,
                    adn_ordering=self.adn_ordering,
                    )
                )
            self.up_path.append(up_layer)

        self.out = Convolution(
            spatial_dims=self.dimensions,
            in_channels=self.channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            strides=1,
            conv_only=True
        )

    @classmethod
    def from_argparse_args(cls, args):
        # Set normalizations
        decoder_norm_name = parse_normalization(args.decoder_norm_name,
                                                not args.decoder_norm_no_affine,
                                                args.num_groups,
                                                args.num_styles)
        encoder_norm_name = parse_normalization(args.encoder_norm_name,
                                                not args.encoder_norm_no_affine,
                                                args.num_groups,
                                                args.num_styles)
        return cls(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=args.feature_size,
            strides=args.strides,
            kernel_size=args.kernel_size,
            up_kernel_size=args.up_kernel_size,
            num_res_units=args.num_res_units,
            act=args.activation,
            norm_down=encoder_norm_name,
            norm_up=decoder_norm_name,
            dropout=args.dropout_rate,
            bias=not args.no_bias,
            adn_ordering=args.adn_ordering
        )

    def forward(self, x: torch.Tensor, modalities=None) -> torch.Tensor:
        # First Conv
        x = self.pre_conv(x)
        # Down Path
        skips = [x]
        for down_layer in self.down_path:
            x = down_layer(x, modalities)
            skips.append(x)
        # Up path
        for scale, (up_sample, residual_unit) in enumerate(self.up_path):
            x = up_sample(x)
            # Skip Connection
            x = torch.concat((skips[len(self.channels) - 2 - scale], x), dim=1)
            x = residual_unit(x, modalities)
        # Out
        out = self.out(x)
        return out
