import time
import torch

from utils import comm
from torch.amp import autocast
from monai.data import decollate_batch
from monai.metrics import LossMetric, Cumulative, CumulativeAverage
from torch.nn.parallel import DistributedDataParallel as DDP


def train_epoch(
        model,
        loader,
        optimizer,
        criterion,
        scaler,
        amp=True,
        iters_to_accumulate=1,
        logger=None,
        labeled_classes=None,  # Make sure criterion takes three inputs if this is provided
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    run_loss = CumulativeAverage()
    # Added zero grad here
    optimizer.zero_grad(set_to_none=True)  # set_to_none=True here can modestly improve performance
    for idx, batch in enumerate(loader):
        start_time = time.time()
        if torch.cuda.is_available():
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda(non_blocking=True)

        if (idx + 1) % iters_to_accumulate == 0 or idx + 1 == len(loader):
            with autocast(enabled=amp, device_type=device):
                output = model(
                    batch["image"],
                    batch["modality"] if "modality" in batch.keys() else None,
                )
                # You may wish to divide loss by iters_to_accumulate to average
                # across the effective (accumulated) global batch.
                criterion_params = (
                    (output, batch["label"], [labeled_classes[modality.item()] for modality in batch["modality"]])
                    if labeled_classes is not None
                    else (output, batch["label"])
                )
                loss = criterion(*criterion_params) / iters_to_accumulate
                run_loss.append(loss)
            # print(f"Train batch {idx}, loss {loss.item()}")
            # If AMP is active
            if amp:
                scaler.scale(loss).backward()
                # Grads DO match across ranks at this point, ready to step
                scaler.step(optimizer)
                # Only call scaler.update() for iterations where we actually step()ed,
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # set_to_none=True here can modestly improve performance
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # set_to_none=True here can modestly improve performance
        # Here accumulate gradient
        else:
            # If is DDP we need to activate the no_sync to correctly accumulate gradient
            if isinstance(model, DDP):
                # We're not stepping this iteration, so use no_sync to prevent DDP allreduces.
                # It appears we need to run forward and backward under no_sync()
                # to get the right no-allreduce behavior.
                with model.no_sync():
                    with autocast(enabled=amp, device_type=device):
                        output = model(
                            batch["image"],
                            batch["modality"] if "modality" in batch.keys() else None,
                        )
                        criterion_params = (
                            (output, batch["label"], [labeled_classes[modality.item()] for modality in batch["modality"]])
                            if labeled_classes is not None
                            else (output, batch["label"])
                        )
                        loss = criterion(*criterion_params) / iters_to_accumulate
                        run_loss.append(loss)
                    if amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
            else:
                # No need to de-sync model if not DDP
                with autocast(enabled=amp, device_type=device):
                    output = model(
                        batch["image"],
                        batch["modality"] if "modality" in batch.keys() else None,
                    )
                    criterion_params = (
                        (output, batch["label"], [labeled_classes[modality.item()] for modality in batch["modality"]])
                        if labeled_classes is not None
                        else (output, batch["label"])
                    )
                    loss = criterion(*criterion_params) / iters_to_accumulate
                    run_loss.append(loss)
                if amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

        logger.info(f"Train: [{idx + 1}/{len(loader)}] Batch {(time.time() - start_time):.4f} loss: {loss.item():.4f}")
    # Total loss
    epoch_loss = run_loss.aggregate().item()
    # Important to reset and free memory
    run_loss.reset()
    return epoch_loss


def val_epoch(
        model,
        loader,
        criterion,
        post_label,
        post_pred,
        metrics,
        model_inferer=None,
        amp=True,
        logger=None,
        modalities=None,
        labeled_classes=None,    # Make sure criterion takes three inputs if this is provided
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    run_loss = CumulativeAverage()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            start_time = time.time()
            if torch.cuda.is_available():
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].cuda(non_blocking=True)
            if "modality" in batch.keys() and modalities is not None:
                modalities.extend(batch["modality"])

            with autocast(enabled=amp, device_type=device):
                if model_inferer is not None:
                    output = model_inferer(
                        batch["image"],
                        modalities=batch["modality"] if "modality" in batch.keys() else modalities,
                    )
                else:
                    output = model(
                        batch["image"],
                        modalities=batch["modality"] if "modality" in batch.keys() else modalities,
                    )
            criterion_params = (
                (output, batch["label"], [labeled_classes[modality.item()] for modality in batch["modality"]])
                if labeled_classes is not None
                else (output, batch["label"])
            )
            loss = criterion(*criterion_params)
            logger.info(f"Val: [{idx + 1}/{len(loader)}] Loss {loss:.4f} [{time.time() - start_time:.4f}s]")
            run_loss.append(loss)
            # Post-process like in https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb
            val_labels_list = decollate_batch(batch["label"])
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(output)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            # Back to [BCHWD] for metrics computation
            # val_output_convert = torch.stack(val_output_convert)
            # val_labels_convert = torch.stack(val_labels_convert)
            for metric in metrics:
                metric(y_pred=val_output_convert, y=val_labels_convert)

    epoch_loss = run_loss.aggregate().item()
    return epoch_loss
