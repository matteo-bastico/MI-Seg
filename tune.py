import os
import wandb
import torch
import optuna

from pathlib import Path
from functools import partial
import torch.distributed as dist
from argparse import ArgumentParser
from monai.metrics import LossMetric
from torch.cuda.amp import GradScaler
from optuna.samplers import TPESampler
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


def save_checkpoint(model, epoch, logdir, filename="model.pt", best_acc=0, optimizer=None, scheduler=None, scaler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        save_dict["scaler"] = scaler.state_dict()
    filename = os.path.join(logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def set_trail_config(trial, args):
    # Batch size = batch_size*patches_training_sample, fix these to max, see google paper
    # args.batch_size = trial.suggest_categorical("batch_size", [2, 4])
    # args.patches_training_sample = trial.suggest_categorical("patches_training_sample", [1, 2, 4])
    # Suggestion for lr, scheduler and optimizer
    args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    args.reg_weight = trial.suggest_float("reg_weight", 1e-6, 1e-4)
    if args.scheduler == "warmup_cosine":
        args.warmup_epochs = trial.suggest_int("warmup_epochs", 50, 100)
        # We try with just one cycle if it is smoother
        # args.cycles = trial.suggest_int("cycles", 1, 4)
    elif args.scheduler == "cosine":
        args.t_max = trial.suggest_int("t_max", 400, args.max_epochs)
    elif args.scheduler == "reduce_on_plateau":
        args.patience_scheduler = trial.suggest_int("patience_scheduler", 2, 10)
    # Model
    args.feature_size = trial.suggest_categorical("feature_size", [8, 16, 32])
    if args.model_name == 'unet':
        args.num_layers = trial.suggest_int("num_layers", 3, 5)
        # Change strides based on nulber of layers
        if args.num_layers == 3:
            args.strides = [2, 2]
        elif args.num_layers == 4:
            args.strides = [2, 2, 2]
        elif args.num_layers == 5:
            args.strides = [2, 2, 2, 2]
        # args.num_res_units = trial.suggest_int("num_res_units", 2, 3)
    elif args.model_name == "unetr":
        args.num_heads = trial.suggest_categorical("num_heads", [8, 12, 16])
        # args.hidden_size = trial.suggest_categorical("hidden_size", [512, 768, 1024])
    return args


def objective(args, single_trial):
    if args.distributed:
        trial = optuna.integration.TorchDistributedTrial(single_trial)
    else:
        trial = single_trial
    args = set_trail_config(trial, args)
    # Folder to save model
    model_logdir = os.path.join(args.default_root_dir, args.study_name, str(trial.number))
    # Create folder if not exists
    Path(model_logdir).mkdir(parents=True, exist_ok=True)
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
            config=args  # Here config of experiment hyper-parameters
        )
    model = model_from_argparse_args(args).to(args.device)
    if args.distributed:
        find_unused_parameters = args.vit_norm_name == "instance_cond" or args.encoder_norm_name == "instance_cond" or \
                                 args.decoder_norm_name == "instance_cond"
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                    find_unused_parameters=find_unused_parameters)
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
    best_acc = 0.0
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
                # Save model
                if accuracy > best_acc:
                    print("New best ({:.6f} --> {:.6f}). ".format(best_acc, accuracy))
                    best_acc = accuracy
                    model_name = 'best.pt'
                else:
                    # Last model not best, useful to restart training from last checkpoint if it is not best
                    model_name = 'last.pt'
                save_checkpoint(
                    model,
                    epoch,
                    model_logdir,
                    model_name,
                    best_acc=best_acc,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler
                )
                # We don't load for now due to memory limitation
                '''
                artifact = wandb.Artifact(model_name.replace('.pt', ''), type='model')
                artifact.add_file(os.path.join(model_logdir, model_name))
                logger.log_artifact(artifact)
                '''
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
    return best_acc


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
            os.environ["MASTER_PORT"] = "23456" if args.port is None else args.port
            dist.init_process_group(
                backend="nccl", world_size=args.world_size, rank=args.local_rank
            )
        else:
            args.distributed = False
    else:
        args.distributed = False
        args.local_rank = 0
        args.device = "cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    # print(args)
    # Activate benchmarking for faster training
    torch.backends.cudnn.benchmark = True
    # Create and start optuna study with defined storage method
    # JournalFileStorage is suggested if a database cannot be set up in NFS
    # It is also suggested to avoid SQLite
    # (https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)
    if not args.distributed:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(),
            pruner=optuna.pruners.SuccessiveHalvingPruner(),
            # Specify the storage URL here.
            storage="sqlite:///" + os.path.join(args.default_root_dir, 'optuna', args.study_name + ".db"),
            study_name=args.study_name,
            load_if_exists=True  # Needed if we run parallelized optimization
        )
        # Start to optimization
        study.optimize(
            partial(objective, args),
            n_trials=args.n_trials,
            timeout=args.timeout
        )
    else:
        storage = JournalStorage(JournalFileStorage(os.path.join(args.default_root_dir, 'optuna',
                                                                 args.study_name + ".log")))
        study = None
        if args.local_rank == 0:
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(),
                pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=4, reduction_factor=3),
                storage=storage,  # Specify the storage URL here.
                study_name=args.study_name,
                load_if_exists=True  # Needed if we run parallelized optimization
            )
            study.optimize(
                partial(objective, args),
                n_trials=args.n_trials,
                timeout=args.timeout
            )
        else:
            for _ in range(args.n_trials):
                try:
                    objective(args, None)
                except optuna.TrialPruned:
                    pass
