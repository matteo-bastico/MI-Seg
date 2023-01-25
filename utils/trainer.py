import time
import torch

from torch.cuda.amp import autocast
from monai.data import decollate_batch
from monai.metrics import LossMetric, Cumulative


def train_epoch(model, loader, optimizer, criterion, device, scaler, amp=True):
    model.train()
    start_time = time.time()
    run_loss = LossMetric(loss_fn=criterion)
    for idx, batch in enumerate(loader):
        data, target = batch["image"], batch["label"]
        data, target = data.to(device), target.to(device)
        modality = None
        if "modality" in batch.keys():
            modality = batch["modality"]
            modality = modality.to(device)
        optimizer.zero_grad()
        with autocast(enabled=amp):
            output = model(data, modality)
            loss = criterion(output, target)
            run_loss(output, target)
        print(f"Train batch {idx}, loss {loss.item()}")
        # If AMP is active
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # Update distributed running loss
        # run_loss.update(loss)
        break
    # Total loss
    epoch_loss = run_loss.aggregate(reduction='mean').item()
    run_loss.reset()
    return epoch_loss


def val_epoch(model, loader, criterion, device, acc_func, post_label, post_pred, model_inferer=None, amp=True):
    model.eval()
    start_time = time.time()
    run_loss = LossMetric(loss_fn=criterion)
    # Here we will store accuracy per modality
    acc_mod_cumulative = Cumulative()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            data, target = batch["image"], batch["label"]
            data, target = data.to(device), target.to(device)
            modality = None
            if "modality" in batch.keys():
                modality = batch["modality"]
                modality = modality.to(device)
            with autocast(enabled=amp):
                if model_inferer is not None:
                    output = model_inferer(data, modalities=modality)
                else:
                    output = model(data, modality)
            loss = criterion(output, target)
            run_loss(output, target)
            print(f"Val batch {idx}, loss {loss.item()}")
            # run_loss.update(loss)
            # Compute accuracy
            # acc_func.update(output, target.int())
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(output)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            batch_acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc_mod_cumulative.append(batch_acc, modality)
            '''
            if modality is not None:
                for mod, acc in zip(modality, batch_acc):
                    acc_per_modality.setdefault(mod.item(), []).append(acc)

    if acc_per_modality:
        for mod, acc in acc_per_modality.items():
            # First get average per class in each modality
            acc = torch.stack(acc)
            print(acc)
            acc = torch.nanmean(acc, dim=0)  # This is per class
            # Log
            print(acc)
            # Then total average per modality
            acc = torch.nanmean(acc)  # This is average per modality
            # Log
            print(acc.item())     
    '''
    # Here I will have a Tensor of size [N_batches, N_sample_per_batch, N_classes]
    acc, mod = acc_mod_cumulative.get_buffer()
    # Flatten on first dim to have [N_samples, N_classes]
    acc = acc.flatten(end_dim=1)
    mod = mod.flatten(end_dim=1)
    for m in torch.unique(mod):
        # Select only samples of that modality
        acc_m = acc[mod == m]
        # Reduce per modality (see monai.metrics.utils.py)
        nans = torch.isnan(acc_m)
        not_nans = (~nans).float()
        t_zero = torch.zeros(1, device=acc_m.device, dtype=acc_m.dtype)
        not_nans = not_nans.sum(dim=0)
        acc_m[nans] = 0
        # We have the accuracy per class here
        acc_m = torch.where(not_nans > 0, acc_m.sum(dim=0) / not_nans, t_zero)  # batch average
        print(f"Accuracy per class [modality {m}]: {acc_m.tolist()}")
        # Average accuracy. Note: as in monai we don't account for classes with all nans in the average
        # This can make the average accuracy among modalities different from the total average accuracy !!
        print(f"Average Accuracy [modality {m}]: {torch.nanmean(acc_m[not_nans > 0]).item()}")
    epoch_loss = run_loss.aggregate(reduction='mean').item()
    accuracy, not_nans = acc_func.aggregate()
    print(f"Accuracy per class [tot]: {accuracy.tolist()}")
    run_loss.reset()
    acc_func.reset()
    acc_mod_cumulative.reset()
    return epoch_loss, torch.nanmean(accuracy[not_nans > 0]).item()
