import os
import time
import torch
import optuna
import torchmetrics

from functools import partial
import torch.distributed as dist
from argparse import ArgumentParser
from data.multi_modal_pelvic import get_loaders
from networks.utils import model_from_argparse_args
from monai.inferers import sliding_window_inference
from optuna.storages import JournalStorage, JournalFileStorage
from utils.utils import loss_from_argparse_args, optimizer_from_argparse_args
from utils.parser import add_model_argparse_args, add_data_argparse_args, add_tune_argparse_args


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    start_time = time.time()
    run_loss = torchmetrics.MeanMetric().to(device)
    for idx, batch in enumerate(loader):
        data, target = batch["image"], batch["label"]
        data, target = data.to(device), target.to(device)
        modality = None
        if "modality" in batch.keys():
            modality = batch["modality"]
            modality = modality.to(device)
        optimizer.zero_grad()
        output = model(data, modality)
        loss = criterion(output, target)
        print(loss)
        # TODO: AMP
        loss.backward()
        optimizer.step()
        # Update distributed running loss
        run_loss.update(loss)
        break
    # Total loss
    epoch_loss = run_loss.compute()
    run_loss.reset()
    return epoch_loss


def val_epoch(model, loader, criterion, device, acc_func, model_inferer=None):
    model.eval()
    start_time = time.time()
    run_loss = torchmetrics.MeanMetric().to(device)
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            data, target = batch["image"], batch["label"]
            data, target = data.to(device), target.to(device)
            modality = None
            if "modality" in batch.keys():
                modality = batch["modality"]
                modality = modality.to(device)
            # TODO: amp
            if model_inferer is not None:
                output = model_inferer(data, modalities=modality)
            else:
                output = model(data, modality)
            loss = criterion(output, target)
            print("validation ", loss)
            run_loss.update(loss)
            acc_func.update(output, target)
            # TODO: per modality loss
    epoch_loss = run_loss.compute()
    accuracy = acc_func.compute()
    run_loss.reset()
    acc_func.reset()
    return epoch_loss, accuracy


def objective(args, single_trial):
    if args.distributed:
        trial = optuna.integration.TorchDistributedTrial(single_trial)
    model = model_from_argparse_args(args).to(args.device)
    dice = torchmetrics.Dice(
        average='macro',  # Calculate the metric for each class separately,
        # and average the metrics across classes (with equal weights for each class).
        num_classes=args.out_channels,
        ignore_index=0,  # It is recommend set ignore_index to index of background class.
        mdmc_average='samplewise'
    ).to(args.device)
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
    # Train and validation
    for epoch in range(1, args.max_epochs + 1):
        # Train one epoch
        epoch_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            args.device
        )
        if epoch % args.check_val_every_n_epoch == 0:
            # Val one epoch
            val_loss, accuracy = val_epoch(
                model,
                val_loader,
                criterion,
                args.device,
                dice,
                model_inferer
            )
            trial.report(accuracy, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return accuracy


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_model_argparse_args(parser)
    parser = add_data_argparse_args(parser)
    parser = add_tune_argparse_args(parser)
    args = parser.parse_args()
    # Set-up world size and rank in args for distributed trainings
    # If we are in the slurm cluster, use slurm enviromental variables to set-up distributed and override args
    # TODO: distributed is supported only in Slurm environment
    if "SLURM_NTASKS" in os.environ:
        args.world_size = int(os.environ["SLURM_NTASKS"])
        args.local_rank = int(os.environ["SLURM_LOCALID"])
        args.world_rank = int(os.environ["SLURM_PROCID"])
        args.device = torch.device(args.local_rank)
        if args.world_size > 1:
            args.distributed = True
            torch.multiprocessing.set_start_method("spawn", force=True)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "23456"
            dist.init_process_group(
                backend="nccl", world_size=args.world_size, rank=args.local_rank
            )
    else:
        args.distributed = False
        args.device = "cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    print(args)
    # Create and start optuna study with defined storage method
    # JournalFileStorage is suggested if a database cannot be set up in NFS
    # It is also suggested to avoid SQLite
    # (https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)
    storage = JournalStorage(JournalFileStorage("journal.log"))
    study = optuna.create_study(
        direction="maximize",
        storage=storage,  # Specify the storage URL here.
        study_name=args.study_name,
        load_if_exists=True  # Needed if we run parallelized optimization
    )
    # Start to optimization
    study.optimize(
        partial(objective, args),
        n_trials=args.n_trials
    )
