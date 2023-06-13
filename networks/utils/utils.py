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
    elif model_name == 'pre_swin_unetr':
        model = SwinUNETR.from_argparse_args(args)
        # Must be pretrained path provided (https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt)
        pretrained = torch.load(args.pre_swin)
        state_dict = pretrained['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('module.', '').replace('fc1', 'linear1').replace('fc2', 'linear2')] = \
                state_dict.pop(key)
        print("Loaded pre-trained Swin-ViT")
        print(model.swinViT.load_state_dict(state_dict, strict=False))

    else:
        raise ValueError("Model {} not implemented. Please chose another model.".format(model_name))

    if args.pretrained:
        print("Loading pre-trained weights ...")
        pretrained = torch.load(args.pretrained)
        # Standard name is state_dict for the model in our checkpoints
        state_dict = pretrained['state_dict']
        # Not load output layer if number of channels is different (for UNETR and Swin-unetr)
        if 'out.conv.conv.weight' in state_dict.keys():
            if state_dict['out.conv.conv.weight'].shape[0] != args.out_channels:
                warnings.warn('Number of out channels of the pre-trained model different from model out_channels, '
                              'skipping loading of output layer.')
                del state_dict['out.conv.conv.weight']
                del state_dict['out.conv.conv.bias']
        # TODO: we remove the last layer from the dict but we should keep if, check if the last layer of unet is always
        # starting with model.2
        # Not load output layer if number of channels is different (for UNET)
        to_del = []
        for key in state_dict.keys():
            if 'model.2' in key:
                to_del.append(key)
        for key in to_del:
            del state_dict[key]
        model.load_state_dict(state_dict, strict=False)  # strict=False deal with missing or added elements

    return model