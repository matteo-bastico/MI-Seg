import os
import torch
import yaml
import torch.distributed as dist

from tensorboardX import SummaryWriter
from utils.launch import launch
from utils.logger import get_root_logger
from utils import comm
from functools import partial
from argparse import ArgumentParser, Namespace
from torch.amp import GradScaler
from data.multi_modal import get_loaders
from utils.trainer import train_epoch, val_epoch
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import ReduceLROnPlateau
from networks.utils.misc import model_from_cfg
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.training_utils import loss_from_cfg, optimizer_from_cfg, scheduler_from_cfg

from monai.transforms import AsDiscrete
from monai.metrics.meandice import DiceMetric
from monai.metrics import SurfaceDistanceMetric, GeneralizedDiceScore, Cumulative
from networks.utils.misc import copy_model_state
from networks.utils.swin_utils import filter_swinunetr


def build_writer(save_path, logger):
    logger.info("=> Building writer ...")
    writer = SummaryWriter(save_path) if comm.is_main_process() else None
    logger.info(f"Tensorboard writer logging dir: {save_path}")
    return writer


def save_checkpoint(
        model,
        epoch,
        logdir,
        filename="model.pt",
        best=0,
        optimizer=None,
        scheduler=None,
        scaler=None,
        logger=None,
):
    state_dict = model.state_dict() if not comm.get_world_size() > 1 else model.module.state_dict()
    save_dict = {"epoch": epoch, "best": best, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        save_dict["scaler"] = scaler.state_dict()
    filename = os.path.join(logdir, "model", filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(save_dict, filename)
    if logger is not None:
        logger.info(f"Saving checkpoint {filename}")


def write_per_modality(metric, modalities, writer, epoch, classes=None, modality_names=None):
    buffer = metric.get_buffer()
    modalities = modalities.get_buffer().cpu()  # Indices on cpu
    if modality_names is None:
        modality_names = list(range(len(torch.unique(modalities))))
    if classes is None:
        classes = list(range(buffer.shape[-1]))
    for m, modality in enumerate(torch.unique(modalities)):
        # Select only samples of that modality
        val = buffer[modalities == modality]
        # Reduce per modality (see monai.metrics.misc.py)
        nans = torch.isnan(val)
        not_nans = (~nans).float()
        t_zero = torch.zeros(1, device=val.device, dtype=val.dtype)
        not_nans = not_nans.sum(dim=0)
        val[nans] = 0
        # We have the surface distance per class here
        val = torch.where(not_nans > 0, val.sum(dim=0) / not_nans, t_zero)  # batch average
        for c, v in enumerate(val.tolist()):
            writer.add_scalar(
                f"val/{metric.__class__.__name__}/{modality_names[m]}/{classes[c + int(not metric.include_background)]}",
                v,
                epoch
            )
        # Average
        writer.add_scalar(
            f"val/{metric.__class__.__name__}/{modality_names[m]}/mean",
            torch.nanmean(val[not_nans > 0]).item(),
            epoch
        )


def main(cfg):
    # Folder to save model
    model_logdir = os.path.join(cfg.default_root_dir, cfg.study_name, cfg.trial_name)
    # Create folder if not exists
    os.makedirs(model_logdir, exist_ok=True)
    if comm.is_main_process():
        with open(os.path.join(model_logdir, "cfg.yaml"), "w") as f:
            yaml.dump(cfg.__dict__, f)

    logger = get_root_logger(
        log_file=os.path.join(model_logdir, "train.log"),
        file_mode="a" if args.resume is not None else "w",
        # file_mode="a" if cfg.resume else "w",
    )
    logger.info(f"Save path: {model_logdir}")
    writer = build_writer(model_logdir, logger)
    logger.info("=> Building model ...")
    model = model_from_cfg(cfg).to(device="cuda" if torch.cuda.is_available() else "cpu")
    logger.info(model)
    logger.info(f"Num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if hasattr(cfg, 'pretrained'):
        logger.info(f"=> Loading pretrained model: {cfg.pretrained} ...")
        pretrained = torch.load(
            cfg.pretrained,
            weights_only=True,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
        if 'state_dict' in pretrained.keys():
            pretrained = pretrained['state_dict']
        elif 'model' in pretrained.keys():
            pretrained = pretrained['model']
        _, loaded, not_loaded = copy_model_state(
            model,
            pretrained,
            filter_func=filter_swinunetr if cfg.model == "swin-unetr" else None,
            logger=logger,
        )
        logger.info(f"Loaded {loaded}")
        logger.info(f"Not loaded {not_loaded}")

    if comm.get_world_size() > 1:
        model = DDP(
            model,
            device_ids=[comm.get_local_rank()],
            output_device=comm.get_local_rank(),
            find_unused_parameters=cfg.find_unused_parameters if hasattr(cfg, 'find_unused_parameters') else False,
        )

    metrics = (
        DiceMetric(
            include_background=cfg.include_background,  # In the metric background is not relevant
            reduction='mean_batch',  # This will give the accuracy per class in averaged on batches
        ),
        GeneralizedDiceScore(
            reduction="mean_batch",
            include_background=cfg.include_background,
        )
    )
    """
    SurfaceDistanceMetric(
        include_background=cfg.include_background,
        symmetric=True,
        distance_metric='euclidean',
        reduction='mean_batch',  # This will give the accuracy per class in averaged on batches
    )
    """
    modalities = Cumulative()

    # Post-processing for accuracy computation
    post_label = AsDiscrete(to_onehot=cfg.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg.out_channels)
    logger.info("=> Building optimize, scheduler, scaler(amp) ...")
    # Define loss criterion from argparse cfg
    criterion = loss_from_cfg(cfg)
    # Set labeled_classes only if criterion is of partial or ignore unlabeled
    labeled_classes = None
    if hasattr(cfg, 'labeled_classes') and hasattr(cfg, 'partial'):
        labeled_classes = cfg.labeled_classes
    # Define optimizer
    optimizer = optimizer_from_cfg(cfg, model)
    # Get the lr scheduler
    scheduler = scheduler_from_cfg(cfg, optimizer)
    # Scaler
    scaler = None
    if cfg.amp:
        scaler = GradScaler()

    # Get dataloader for training
    cfg.test_mode = False
    logger.info("=> Building dataloaders for training and validation sets ...")
    train_loader, val_loader = get_loaders(cfg)

    logger.info(f"Totally {len(train_loader)} x {cfg.batch_size} (batch size) x {cfg.patches_training_sample} "
                f"(patches per sample) samples in train loader.")
    logger.info(f"Totally {len(val_loader)} samples in val loader.")

    # Create model inferer
    model_inferer = partial(
        sliding_window_inference,
        roi_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z),
        sw_batch_size=cfg.sw_batch_size,
        predictor=model,
        device=cfg.inferer_device,
        overlap=cfg.infer_overlap,
    )

    layer_names = {param: name for name, param in model.named_parameters()}
    for idx, param_group in enumerate(optimizer.param_groups):
        param_names = [layer_names[p] for p in param_group['params'] if p in layer_names]
        logger.info(f"Learning rate group {idx}: {param_group['lr']}, Layers: {param_names}")

    # Train and validation
    best = 0.0
    start_epoch = 1
    logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")

    # Resume checkpoint
    if args.resume is not None:
        logger.info(f"=> Resuming checkpoint {os.path.join(args.resume, "model/last.pt")} ...")
        # Load state-dict
        checkpoint = torch.load(
            os.path.join(args.resume, "model/last.pt"),
            map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
        model.load_state_dict(checkpoint['state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best = checkpoint['best']
        logger.info(f"=> Resumed at epoch {start_epoch} with best score {best:.4f}.")

    for epoch in range(start_epoch, cfg.epochs + 1):
        # Log learning rate
        if writer is not None:
            for idx, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar("lr/group" + str(idx), param_group['lr'], epoch)

        logger.info(f"Epoch: [{epoch}/{cfg.epochs}] Lr (groups): {scheduler.get_last_lr()}")
        if comm.get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        # Train one epoch
        epoch_loss = train_epoch(
            model,  # opt_model
            train_loader,
            optimizer,
            criterion,
            scaler,
            amp=cfg.amp,
            iters_to_accumulate=cfg.iters_to_accumulate,
            logger=logger,
            labeled_classes=labeled_classes,
        )
        # Step the scheduler, if not Reduce On Plateau
        if scheduler is not None:
            if not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

        logger.info(f"Train result: loss: {epoch_loss:.4f}")
        # Log train loss
        if writer is not None:
            writer.add_scalar("loss_epoch/loss", epoch_loss, epoch)
        if epoch % cfg.check_val_every_n_epoch == 0:
            logger.info(">>>>>>>>>>>>>>>> Start Validation >>>>>>>>>>>>>>>>")
            # Val one epoch
            val_loss = val_epoch(
                model,  # opt_model
                val_loader,
                criterion,
                post_label=post_label,
                post_pred=post_pred,
                model_inferer=model_inferer,
                amp=cfg.amp,
                logger=logger,
                metrics=metrics,
                modalities=modalities,
                labeled_classes=labeled_classes,
            )

            logger.info(f"Val result: Loss: {val_loss:.4f}")
            if writer is not None:
                writer.add_scalar("loss/train", epoch_loss, epoch // cfg.check_val_every_n_epoch)
                writer.add_scalar("loss/val", val_loss, epoch // cfg.check_val_every_n_epoch)

            # Sync DDP processes before calculating metrics
            # Sync also all metrics to avoid wrong reduction in write_per_modality
            logger.info("Syncing ...")
            comm.synchronize()
            modalities._sync()
            for metric in metrics:
                metric._sync()
                avg = metric.aggregate().mean().item()
                logger.info(f"Val result: {metric.__class__.__name__}: {avg:.4f}")
                if writer is not None:
                    write_per_modality(
                        metric,
                        modalities,
                        writer,
                        epoch // cfg.check_val_every_n_epoch,
                        cfg.classes,
                        cfg.modalities
                    )
                    writer.add_scalar(
                        f"val/{metric.__class__.__name__}/mean",
                        avg,
                        epoch // cfg.check_val_every_n_epoch,
                    )

            if comm.is_main_process():
                current = metrics[0].aggregate().mean().item()
                # Save model
                if current > best:
                    logger.info("New best ({:.4f} --> {:.4f}). ".format(best, current))
                    best = current
                    model_name = 'best.pt'
                else:
                    # Last model not best, useful to restart training from last checkpoint if it is not best
                    model_name = 'last.pt'
                save_checkpoint(
                    model,  # opt_model
                    epoch,
                    model_logdir,
                    model_name,
                    best=best,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    logger=logger,
                )
                logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

            # Reset metrics
            for metric in metrics:
                metric.reset()
            modalities.reset()

            # Step scheduler here if Reduce on plateau
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)

    if comm.is_main_process():
        writer.close()

    if comm.get_world_size() > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = ArgumentParser(description="Load configuration file.")
    parser.add_argument("--config", type=str, help="Path to the configuration file (YAML).")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs per machine.")
    parser.add_argument("--num_machines", type=int, default=1, help="Number of machines.")
    parser.add_argument("--machine_rank", type=int, default=0, help="Local rank for distributed training.")
    parser.add_argument("--dist_url", type=str, default="auto", help="URL used to set up distributed training.")
    parser.add_argument("--resume", type=str, default=None, help="Path to experiment to resume.")
    args = parser.parse_args()

    if args.resume is not None:
        with open(os.path.join(args.resume, "cfg.yaml"), "r") as f:
            cfg = yaml.safe_load(f)
    else:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    cfg = Namespace(**cfg)

    # Start training
    launch(
        main,
        cfg,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
    )