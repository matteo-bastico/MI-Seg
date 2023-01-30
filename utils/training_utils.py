import torch
from monai.losses import DiceCELoss, DiceFocalLoss, GeneralizedDiceFocalLoss
from monai.optimizers.lr_scheduler import WarmupCosineSchedule


def loss_from_argparse_args(args):
    if args.criterion == "dice_focal":
        # In the loss we always include background because it can be important (especially with focal loss)
        criterion = DiceFocalLoss(
            # include_background=not args.no_include_background,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=args.smooth_nr,
            smooth_dr=args.smooth_dr
        )
    elif args.criterion == "dice_ce":
        criterion = DiceCELoss(
            # include_background=not args.no_include_background,
            to_onehot_y=True,
            softmax=True,
            squared_pred=args.squared_pred,
            smooth_nr=args.smooth_nr,
            smooth_dr=args.smooth_dr
        )
    elif args.criterion == "generalized_dice_focal":
        criterion = GeneralizedDiceFocalLoss(
            # include_background=not args.no_include_backgound,
            to_onehot_y=True,
            softmax=True,
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


def scheduler_from_argparse_args(args, optimizer):
    if args.scheduler == 'warmup_cosine':
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=args.warmup_epochs,
            t_total=args.max_epochs,
            cycles=args.cycles
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.t_max
        )
    elif args.scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=args.patience_scheduler,
        )
    elif args.scheduler == 'none' or args.scheduler is None:
        scheduler = None
    else:
        raise ValueError("Scheduler {} not implemented, please chose another optimizer.".format(args.scheduler))
    return scheduler