import torch
from monai.losses import DiceCELoss, DiceFocalLoss, GeneralizedDiceFocalLoss
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from losses import PartialLoss

def loss_from_cfg(cfg):
    """
    if hasattr(cfg, "partial"):
        if cfg.criterion == "dice_focal":
            # In the loss we always include background because it can be important (especially with focal loss)
            criterion = PartialDiceFocalLoss(
                to_onehot_y=True,
                softmax=True,
                squared_pred=cfg.squared_pred,
                smooth_nr=cfg.smooth_nr,
                smooth_dr=cfg.smooth_dr,
                mode=cfg.partial,
                regularize_unlabeled=cfg.regularize_unlabeled if hasattr(cfg, "regularize_unlabeled") else False,
                lambda_reg=cfg.lambda_reg if hasattr(cfg, "lambda_reg") else 1.0,
            )
        elif cfg.criterion == "dice_ce":
            criterion = PartialDiceCELoss(
                to_onehot_y=True,
                softmax=True,
                squared_pred=cfg.squared_pred,
                smooth_nr=cfg.smooth_nr,
                smooth_dr=cfg.smooth_dr,
                mode=cfg.partial,
                regularize_unlabeled=cfg.regularize_unlabeled if hasattr(cfg, "regularize_unlabeled") else False,
                lambda_reg=cfg.lambda_reg if hasattr(cfg, "lambda_reg") else 1.0,
            )
        else:
            raise ValueError("Criterion {} not implemented for partial loss, please chose dice_focal or dice_ce.".format(cfg.criterion))
    else:
    """
    if cfg.criterion == "dice_focal":
        # In the loss we always include background because it can be important (especially with focal loss)
        criterion = DiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=cfg.squared_pred,
            smooth_nr=cfg.smooth_nr,
            smooth_dr=cfg.smooth_dr
        )
    elif cfg.criterion == "dice_ce":
        criterion = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=cfg.squared_pred,
            smooth_nr=cfg.smooth_nr,
            smooth_dr=cfg.smooth_dr
        )
    elif cfg.criterion == "generalized_dice_focal":
        criterion = GeneralizedDiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            smooth_nr=cfg.smooth_nr,
            smooth_dr=cfg.smooth_dr
        )
    else:
        raise ValueError("Criterion {} not implemented, please chose another optimizer.".format(cfg.criterion))

    if hasattr(cfg, "partial"):
        criterion = PartialLoss(
            loss=criterion,
            mode=cfg.partial,
            regularize_unlabeled=cfg.regularize_unlabeled if hasattr(cfg, "regularize_unlabeled") else False,
            lambda_reg=cfg.lambda_reg if hasattr(cfg, "lambda_reg") else 1.0,
            ensure_softmax=True
        )

    return criterion


def optimizer_from_cfg(cfg, model):
    if cfg.optim_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.reg_weight
        )
    elif cfg.optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.reg_weight
        )
    elif cfg.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=True,
            weight_decay=cfg.reg_weight
        )
    else:
        raise ValueError("Optimization {} not implemented, please chose another optimizer.".format(cfg.optim_name))
    return optimizer


def scheduler_from_cfg(cfg, optimizer):
    if cfg.scheduler == 'warmup_cosine':
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=cfg.warmup_epochs,
            t_total=cfg.epochs,
            cycles=cfg.cycles
        )
    elif cfg.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cfg.t_max
        )
    elif cfg.scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=cfg.patience_scheduler,
        )
    elif cfg.scheduler == 'none' or cfg.scheduler is None:
        scheduler = None
    else:
        raise ValueError("Scheduler {} not implemented, please chose another optimizer.".format(cfg.scheduler))
    return scheduler