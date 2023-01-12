import torch
import warnings
import torch.nn as nn

from typing import Union, List


class ConditionalInstanceNorm1D(nn.Module):
    def __init__(self, num_styles,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,  # We need this to be active !
                 track_running_stats=False,
                 device=None,
                 dtype=None):
        super().__init__()
        if not affine:
            warnings.warn("Ignored affine=False for ConditionalInstanceNorm1D, set to True")

        self.num_styles = num_styles
        self.norms = nn.ModuleList([
            nn.InstanceNorm1d(
                 num_features,
                 eps,
                 momentum,
                 True,
                 track_running_stats,
                 device,
                 dtype)
            for i in range(num_styles)
        ])

    def forward(self, x, style_ids):
        if isinstance(style_ids, (list, torch.Tensor)):
            out = torch.stack([self.norms[style_ids[i]](x[i].unsqueeze(0)).squeeze(0) for i in range(len(style_ids))])
        else:
            out = self.norms[style_ids]((x.unsqueeze(0)).squeeze(0))
        return out


class ConditionalInstanceNorm2D(nn.Module):
    def __init__(self, num_styles,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,  # We need this to be active !
                 track_running_stats=False,
                 device=None,
                 dtype=None):
        super().__init__()
        if not affine:
            warnings.warn("Ignored affine=False for ConditionalInstanceNorm2D, set to True")

        self.num_styles = num_styles
        self.norms = nn.ModuleList([
            nn.InstanceNorm2d(
                 num_features,
                 eps,
                 momentum,
                 True,
                 track_running_stats,
                 device,
                 dtype)
            for i in range(num_styles)
        ])

    def forward(self, x, style_ids):
        if isinstance(style_ids, (list, torch.Tensor)):
            out = torch.stack([self.norms[style_ids[i]](x[i].unsqueeze(0)).squeeze(0) for i in range(len(style_ids))])
        else:
            out = self.norms[style_ids]((x.unsqueeze(0)).squeeze(0))
        return out


class ConditionalInstanceNorm3D(nn.Module):
    def __init__(self, num_styles,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,  # We need this to be active !
                 track_running_stats=False,
                 device=None,
                 dtype=None):
        super().__init__()
        if not affine:
            warnings.warn("Ignored affine=False for ConditionalInstanceNorm2D, set to True")

        self.num_styles = num_styles
        self.norms = nn.ModuleList([
            nn.InstanceNorm3d(
                 num_features,
                 eps,
                 momentum,
                 True,
                 track_running_stats,
                 device,
                 dtype)
            for i in range(num_styles)
        ])

    def forward(self, x, style_ids: Union[List, int]):
        if isinstance(style_ids, (list, torch.Tensor)):
            out = torch.stack([self.norms[style_ids[i]](x[i].unsqueeze(0)).squeeze(0) for i in range(len(style_ids))])
        else:
            out = self.norms[style_ids]((x.unsqueeze(0)).squeeze(0))
        return out
