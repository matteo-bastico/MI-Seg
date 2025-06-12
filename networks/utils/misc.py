from __future__ import annotations

from collections.abc import Mapping
from collections import OrderedDict

import torch
import warnings

from monai.utils import optional_import
from torch import nn

from networks.nets.unetr import UNETR
from networks.nets.unet import UNet
from networks.nets.swin_unetr import SwinUNETR
from monai.networks.utils import get_state_dict

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "model_from_cfg",
    "copy_model_state",
]


def model_from_cfg(cfg):
    model = cfg.model
    if model == 'unetr':
        model = UNETR.from_cfg(cfg)
    elif model == 'unet':
        model = UNet.from_cfg(cfg)
    elif model == 'swin-unetr':
        model = SwinUNETR.from_cfg(cfg)
    return model


def copy_model_state(
    dst: torch.nn.Module | Mapping,
    src: torch.nn.Module | Mapping,
    dst_prefix="",
    mapping=None,
    exclude_vars=None,
    inplace=True,
    filter_func=None,
    logger=None,
):
    """
    Compute a module state_dict, of which the keys are the same as `dst`. The values of `dst` are overwritten
    by the ones from `src` whenever their keys match. The method provides additional `dst_prefix` for
    the `dst` key when matching them. `mapping` can be a `{"src_key": "dst_key"}` dict, indicating
    `dst[dst_prefix + dst_key] = src[src_key]`.
    This function is mainly to return a model state dict
    for loading the `src` model state into the `dst` model, `src` and `dst` can have different dict keys, but
    their corresponding values normally have the same shape.

    Args:
        dst: a pytorch module or state dict to be updated.
        src: a pytorch module or state dict used to get the values used for the update.
        dst_prefix: `dst` key prefix, so that `dst[dst_prefix + src_key]`
            will be assigned to the value of `src[src_key]`.
        mapping: a `{"src_key": "dst_key"}` dict, indicating that `dst[dst_prefix + dst_key]`
            to be assigned to the value of `src[src_key]`.
        exclude_vars: a regular expression to match the `dst` variable names,
            so that their values are not overwritten by `src`.
        inplace: whether to set the `dst` module with the updated `state_dict` via `load_state_dict`.
            This option is only available when `dst` is a `torch.nn.Module`.
        filter_func: a filter function used to filter the weights to be loaded.
            See 'filter_swinunetr' in "monai.networks.nets.swin_unetr.py".

    Examples:
        .. code-block:: python

            from monai.networks.nets import BasicUNet
            from monai.networks.utils import copy_model_state

            model_a = BasicUNet(in_channels=1, out_channels=4)
            model_b = BasicUNet(in_channels=1, out_channels=2)
            model_a_b, changed, unchanged = copy_model_state(
                model_a, model_b, exclude_vars="conv_0.conv_0", inplace=False)
            # dst model updated: 76 of 82 variables.
            model_a.load_state_dict(model_a_b)
            # <All keys matched successfully>

    Returns: an OrderedDict of the updated `dst` state, the changed, and unchanged keys.

    """
    src_dict = get_state_dict(src)
    dst_dict = OrderedDict(get_state_dict(dst))

    to_skip = {s_key for s_key in src_dict if exclude_vars and re.compile(exclude_vars).search(s_key)}

    # update dst with items from src
    all_keys, updated_keys = list(dst_dict), list()
    for s, val in src_dict.items():
        dst_key = f"{dst_prefix}{s}"
        if dst_key in dst_dict and dst_key not in to_skip and dst_dict[dst_key].shape == val.shape:
            dst_dict[dst_key] = val
            updated_keys.append(dst_key)
    for s in mapping if mapping else {}:
        dst_key = f"{dst_prefix}{mapping[s]}"
        if dst_key in dst_dict and dst_key not in to_skip:
            if dst_dict[dst_key].shape != src_dict[s].shape:
                warnings.warn(f"Param. shape changed from {dst_dict[dst_key].shape} to {src_dict[s].shape}.")
            dst_dict[dst_key] = src_dict[s]
            updated_keys.append(dst_key)
    if filter_func is not None:
        for key, value in src_dict.items():
            new_pair = filter_func(key, value)
            if new_pair is not None and new_pair[0] not in to_skip:
                dst_dict[new_pair[0]] = new_pair[1]
                updated_keys.append(new_pair[0])

    updated_keys = sorted(set(updated_keys))
    unchanged_keys = sorted(set(all_keys).difference(updated_keys))
    if logger is not None:
        logger.info(f"'dst' model updated: {len(updated_keys)} of {len(dst_dict)} variables.")
    if inplace and isinstance(dst, torch.nn.Module):
        if isinstance(dst, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            dst = dst.module
        print(dst.load_state_dict(dst_dict, strict=False))  # Only difference is here
    return dst_dict, updated_keys, unchanged_keys