import os
import torch
import optuna

from functools import partial
import torch.distributed as dist
from argparse import ArgumentParser

import wandb
from monai.metrics import LossMetric
from torch.cuda.amp import GradScaler
from monai.transforms import AsDiscrete
from monai.metrics.meandice import DiceMetric
from data.multi_modal_pelvic import get_loaders
from utils.trainer import train_epoch, val_epoch
from networks.utils import model_from_argparse_args
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
from optuna.storages import JournalStorage, JournalFileStorage
from monai.metrics import SurfaceDistanceMetric, GeneralizedDiceScore
from utils.parser import add_model_argparse_args, add_data_argparse_args, add_tune_argparse_args
from utils.training_utils import loss_from_argparse_args, optimizer_from_argparse_args, scheduler_from_argparse_args


def objective(args, single_trial):
    if args.distributed:
        trial = optuna.integration.TorchDistributedTrial(single_trial)
    else:
        trial = single_trial
    # Start wandb logger if local rank is 0
    logger = None
    if args.local_rank == 0:
        logger = wandb.init(
            dir=os.path.join(args.default_root_dir, args.study_name),
            project=args.project,
            entity=args.entity,
            group=args.study_name,
            mode=args.wandb_mode,
            id=str(trial.number),
        )
    model = model_from_argparse_args(args).to(args.device)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # Overall dice metric
    dice = DiceMetric(
        include_background=not args.no_include_background,  # In the metric background is not relevant
        reduction='mean_batch',  # This will give the accuracy per class in averaged on batches
        get_not_nans=True  # Exclude nans from computation
    )
    # Define surface distance metric
    surface_distance = SurfaceDistanceMetric(
        include_background=not args.no_include_background,
        symmetric=True,
        distance_metric='euclidean',
        reduction='mean_batch',  # This will give the accuracy per class in averaged on batches
        get_not_nans=True  # Exclude nans from computation
    )
    # Add generalized dice
    additional_metrics = [
        GeneralizedDiceScore(
            include_background=not args.no_include_background,
        )
    ]
    # Post-processing for accuracy computation
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    # Define loss criterion from argparse args
    criterion = loss_from_argparse_args(args)
    # Define optimizer
    optimizer = optimizer_from_argparse_args(args, model)
    # Get dataloader for training
    args.test_mode = False
    train_loader, val_loader = get_loaders(args)
    # Create model inferer
    model_inferer = partial(
        sliding_window_inference,
        roi_size=(args.roi_x, args.roi_y, args.roi_z),
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    # Get the lr scheduler
    scheduler = scheduler_from_argparse_args(args, optimizer)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    # Train and validation
    for epoch in range(1, args.max_epochs + 1):
        # Log learning rate
        if logger is not None and scheduler is not None:
            for idx, param_group in enumerate(optimizer.param_groups):
                logger.log({"Charts/lr_group" + str(idx): param_group['lr']}, epoch)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        # Train one epoch
        epoch_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            args.device,
            scaler,
            amp=args.amp
        )
        # Step the scheduler, if not Reduce On Plateau
        if scheduler is not None:
            if not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
        # Log train loss
        if logger is not None:
            logger.log({"train/loss": epoch_loss}, epoch)
            print(f"Train Epoch {epoch}, loss {epoch_loss}")
        if epoch % args.check_val_every_n_epoch == 0:
            # Val one epoch
            val_loss, accuracy, surface, other_metrics = val_epoch(
                model,
                val_loader,
                criterion,
                args.device,
                dice,
                post_label=post_label,
                post_pred=post_pred,
                model_inferer=model_inferer,
                amp=args.amp,
                surface_distance=surface_distance,
                additional_metrics=additional_metrics,
                logger=logger,
                epoch=epoch
            )
            # Log validation results
            if logger is not None:
                logger.log({
                    "val_total_others/loss": val_loss,
                    "val_total_dice/avg": accuracy,
                    "val_total_surface_distance/avg": surface,
                    "val_total_others/generalized_dice_avg": other_metrics[0]
                }, epoch)
                print(f"Val Epoch {epoch}, loss {val_loss}")
                print(f"Val Epoch {epoch}, dice accuracy {accuracy}")
                print(f"Val Epoch {epoch}, surface {surface}")
                print(f"Val Epoch {epoch}, generalized dice {other_metrics[0]}")
            # Step scheduler here if Reduce on plateau
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
            # if args.distributed:
            trial.report(accuracy, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                if logger is not None:
                    logger.finish()
                raise optuna.exceptions.TrialPruned()
            '''
            else:
                single_trial.report(accuracy, epoch)
                # Handle pruning based on the intermediate value.
                if single_trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            '''
    if logger is not None:
        logger.finish()
    return accuracy


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_model_argparse_args(parser)
    parser = add_data_argparse_args(parser)
    parser = add_tune_argparse_args(parser)
    args = parser.parse_args()
    args.amp = not args.no_amp
    # Set-up world size and rank in args for distributed trainings
    # If we are in the slurm cluster, use slurm enviromental variables to set-up distributed and override args
    # Otherwise we run with signle GPU
    # TODO: distributed in a general environment not Slurm
    if "SLURM_NTASKS" in os.environ:
        args.world_size = int(os.environ["SLURM_NTASKS"])
        args.local_rank = int(os.environ["SLURM_LOCALID"])
        args.world_rank = int(os.environ["SLURM_PROCID"])
        print("World size: ", args.world_size)
        print("Local rank: ", args.local_rank)
        print("World rank: ", args.world_rank)
        args.device = torch.device(args.local_rank)
        if args.world_size > 1:
            args.distributed = True
            # Removed set_start_method causing DataLoader workers exiting unexpectedly
            #  torch.multiprocessing.set_start_method("spawn", force=True)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "23456"
            dist.init_process_group(
                backend="nccl", world_size=args.world_size, rank=args.local_rank
            )
        else:
            args.distributed = False
    else:
        args.distributed = False
        args.local_rank = 0
        args.device = "cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    print(args)
    # Create and start optuna study with defined storage method
    # JournalFileStorage is suggested if a database cannot be set up in NFS
    # It is also suggested to avoid SQLite
    # (https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)
    if not args.distributed:
        study = optuna.create_study(
            direction="maximize",
            # Specify the storage URL here.
            storage="sqlite:///" + os.path.join(args.default_root_dir, 'optuna', args.study_name + ".db"),
            study_name=args.study_name,
            load_if_exists=True  # Needed if we run parallelized optimization
        )
        # Start to optimization
        study.optimize(
            partial(objective, args),
            n_trials=args.n_trials
        )
    else:
        storage = JournalStorage(JournalFileStorage(os.path.join(args.default_root_dir, 'optuna',
                                                                 args.study_name + ".log")))
        study = None
        if args.local_rank == 0:
            study = optuna.create_study(
                direction="maximize",
                storage=storage,  # Specify the storage URL here.
                study_name=args.study_name,
                load_if_exists=True  # Needed if we run parallelized optimization
            )
            study.optimize(
                partial(objective, args),
                n_trials=args.n_trials
            )
        else:
            for _ in range(args.n_trials):
                try:
                    objective(args, None)
                except optuna.TrialPruned:
                    pass
