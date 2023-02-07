import torch
import warnings

from monai.utils import optional_import
from networks.nets.unetr import UNETR
from networks.nets.unet import UNet
from networks.nets.swin_unetr import SwinUNETR

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "model_from_argparse_args",
]


def model_from_argparse_args(args):
    model_name = args.model_name
    if model_name == 'unetr':
        model = UNETR.from_argparse_args(args)
    elif model_name == 'unet':
        model = UNet.from_argparse_args(args)
    elif model_name == 'swin_unetr':
        model = SwinUNETR.from_argparse_args(args)
    else:
        raise ValueError("Model {} not implemented. Please chose another model.".format(model_name))

    if args.pretrained:
        print("Loading pre-trained weights ...")
        pretrained = torch.load(args.pretrained)
        if pretrained['out.conv.conv.weight'].shape[0] != args.out_channels:
            warnings.warn('Number of out channels of the pre-trained model different from model out_channels, '
                          'skipping loading of output layer.')
            del pretrained['out.conv.conv.weight']
            del pretrained['out.conv.conv.bias']
        model.load_state_dict(pretrained, strict=False)  # strict=False deal with missing or added elements

    return model