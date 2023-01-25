import torch
from monai.losses import DiceCELoss, DiceFocalLoss


def loss_from_argparse_args(args):
    if args.criterion == "dice_focal":
        criterion = DiceFocalLoss(
            include_background=not args.no_include_background,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=args.smooth_nr,
            smooth_dr=args.smooth_dr
        )
    elif args.criterion == "dice_ce":
        criterion = DiceCELoss(
            include_background=not args.no_include_background,
            to_onehot_y=True,
            softmax=True,
            squared_pred=args.squared_pred,
            smooth_nr=args.smooth_nr,
            smooth_dr=args.smooth_dr
        )
    else:
        raise ValueError("Criterion {} not implemented, please chose another optimizer.".format(args.criterion))
    return criterion


def optimizer_from_argparse_args(args, model):
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.reg_weight
        )
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.reg_weight
        )
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Optimization {} not implemented, please chose another optimizer.".format(args.optim_name))
    return optimizer
