import torch

from functools import partial
from torch.cuda.amp import autocast
from argparse import ArgumentParser
from monai.data import decollate_batch
from monai.metrics import LossMetric, Cumulative
from monai.transforms import AsDiscrete
from monai.metrics.meandice import DiceMetric
from data.multi_modal import get_loaders
from networks.utils.utils import model_from_argparse_args
from monai.inferers import sliding_window_inference
from monai.metrics import SurfaceDistanceMetric
from utils.parser import add_model_argparse_args, add_data_argparse_args, add_tune_argparse_args


def compute_metric_modality(metric_func, include_background=0):
    metric, mod_metric = metric_func.get_buffer()
    # This is a workaround for the cuda problem of monai metrics
    metric = metric.cpu()
    mod_metric = mod_metric.cpu()
    for m in torch.unique(mod_metric):
        # Select only samples of that modality
        metric_m = metric[mod_metric == m]
        # Reduce per modality (see monai.metrics.utils.py)
        nans = torch.isnan(metric_m)
        not_nans = (~nans).float()
        t_zero = torch.zeros(1, device=metric_m.device, dtype=metric_m.dtype)
        not_nans = not_nans.sum(dim=0)
        metric_m[nans] = 0
        # We have the surface distance per class here
        metric_m = torch.where(not_nans > 0, metric_m.sum(dim=0) / not_nans, t_zero)  # batch average
        dict_surf_class_modality = {}
        for c, v in enumerate(metric_m.tolist()):
            dict_surf_class_modality[f"val_modality{m}/class{c + include_background}"] = v
        print(dict_surf_class_modality)
        # Average surface distance. Note: as in monai we don't account for classes with all nans in the average
        # This can make the average surface distance among modalities different from the total average accuracy
        # print(f"Average {label} [modality {m}]: {torch.nanmean(metric_m[not_nans > 0]).item()}")
        print({f"val_modality{m}/avg": torch.nanmean(metric_m[not_nans > 0]).item()})


def test(
        model,
        loader,
        device,
        acc_func,
        post_label,
        post_pred,
        model_inferer=None,
        amp=True,
        surface_distance=None):
    model.eval()
    # Here we will store accuracy per modality
    acc_mod_cumulative = Cumulative()
    if surface_distance is not None:
        surface_mod_cumulative = Cumulative()
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
                    output = model_inferer(data, modalities=modality, return_classification=False)
                else:
                    output = model(data, modality, return_classification=False)
            # In the print we assume validation batch
            # print(f"Val batch {idx} modality {modality.tolist()}, loss {loss}")
            # run_loss.update(loss)
            # Compute accuracy
            # acc_func.update(output, target.int())
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(output)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            # Back to [BCHWD] for metrics computation
            val_output_convert = torch.stack(val_output_convert)
            val_labels_convert = torch.stack(val_labels_convert)
            batch_acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            # print(f"Accuracy batch {idx} modality {modality.tolist()}, loss {batch_acc}")
            # acc_mod_cumulative.append(batch_acc, modality)
            acc_mod_cumulative.extend(batch_acc, modality)  # Extend is for append a batch-first array
            # Update surface distance
            if surface_distance is not None:
                batch_surface = surface_distance(y_pred=val_output_convert, y=val_labels_convert)
                surface_mod_cumulative.extend(batch_surface, modality)
                # print(f"Surface batch {idx} modality {modality.tolist()}, loss {batch_surface}")
    # log dice and surface distance per modality with auxiliary function
    # This is 1 if background is not included, to have correct class logging
    print("Dice per modality")
    include_background_acc = int(not acc_func.include_background)
    compute_metric_modality(acc_mod_cumulative, include_background_acc)
    print("Surface Distance per modality")
    if surface_distance is not None:
        include_background_surf = int(not surface_distance.include_background)
        compute_metric_modality(surface_mod_cumulative, include_background_surf)
    # log dice and surface distance total
    accuracy, not_nans = acc_func.aggregate()
    dict_acc_class = {}
    for c, v in enumerate(accuracy.tolist()):
        dict_acc_class[f"val_total_dice/class{c + include_background_acc}"] = v
    print(dict_acc_class)
    if surface_distance is not None:
        surface, not_nans_surface = surface_distance.aggregate()
        dict_surf_class = {}
        for c, v in enumerate(surface.tolist()):
            dict_surf_class[f"val_total_surface_distance/class{c + include_background_surf}"] = v
        print(dict_surf_class)
        surface_distance.reset()
        surface_mod_cumulative.reset()
    # Important to reset and free memory
    acc_func.reset()
    acc_mod_cumulative.reset()
    # Aggregate additional metric, if any
    if surface_distance is not None:
        return torch.nanmean(accuracy[not_nans > 0]).item(), \
               torch.nanmean(surface[not_nans_surface > 0]).item()
    else:
        return torch.nanmean(accuracy[not_nans > 0]).item()


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model = model_from_argparse_args(args)
    model_weights = checkpoint["state_dict"]
    model.load_state_dict(model_weights)
    model = model.to(args.device)
    # Load test dataloader
    args.test_mode = True
    test_loader = get_loaders(args)
    # Post-processing for accuracy computation
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
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
    # Create model inferer
    model_inferer = partial(
        sliding_window_inference,
        roi_size=(args.roi_x, args.roi_y, args.roi_z),
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    dice, surf_dist = test(
        model,
        test_loader,
        args.device,
        dice,
        post_label=post_label,
        post_pred=post_pred,
        model_inferer=model_inferer,
        amp=not args.no_amp,
        surface_distance=surface_distance
    )
    print("Average total Dice: ", dice)
    print("Average surface distance: ", surf_dist)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_model_argparse_args(parser)
    parser = add_data_argparse_args(parser)
    parser = add_tune_argparse_args(parser)
    parser.add_argument("--checkpoint", default="Test_unet/3hs4d1wo/checkpoints/5-6.ckpt", type=str, help="Checkpoint")
    args = parser.parse_args()
    args.device = "cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    args.distributed = False
    if args.device != "cpu":
        torch.cuda.set_device(args.device)
    main(args)