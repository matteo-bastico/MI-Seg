from functools import partial

import yaml
import torch
import os
import time

from argparse import ArgumentParser, Namespace
from monai.handlers import from_engine
from monai.data import decollate_batch
from monai.transforms import Compose, EnsureTyped, Activationsd, Invertd, AsDiscreted, SaveImaged, ToDeviced, Orientationd
from monai.transforms import AsDiscrete
from torch.amp import autocast
from monai.inferers import sliding_window_inference
from monai.metrics import SurfaceDistanceMetric, GeneralizedDiceScore, Cumulative, DiceMetric
from utils.logger import get_root_logger
from networks.utils.misc import model_from_cfg
from data.multi_modal import get_loaders

def log_metric(
        metric,
        modalities,
        logger,
        classes=None,
        modality_names=None
):
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
            logger.info(f"=> {modality_names[m]} / {classes[c + int(not metric.include_background)]}: {v:.4f}")
        # Average
        logger.info(f"=> {modality_names[m]} / Mean : {torch.nanmean(val[not_nans > 0]).item():.4f}")


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger(
        log_file=os.path.join(cfg.experiment, "test.log"),
        file_mode="w"
    )
    logger.info("=> Building model ...")
    model = model_from_cfg(cfg).to(device=device)
    logger.info(f"Num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    weight_path = os.path.join(cfg.experiment, "model", cfg.weight)
    logger.info(f"=> Loading weight at {weight_path} ...")
    load_dict = torch.load(weight_path, weights_only=True, map_location=device)
    model.load_state_dict(load_dict["state_dict"])
    logger.info(f"=> Loaded weight at {weight_path} (epoch {load_dict['epoch']}, best {load_dict['best']:.4f})")
    # Get dataloader for training
    logger.info("=> Building dataloader for testing ...")
    cfg.test_mode = True
    test_loader, preprocessing = get_loaders(cfg)
    logger.info(f"Totally {len(test_loader)} samples in test loader.")
    # Create model inferer
    model_inferer = partial(
        sliding_window_inference,
        roi_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z),
        sw_batch_size=cfg.sw_batch_size,
        predictor=model,
        overlap=cfg.infer_overlap
    )
    # Metrics
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
    ),
    """
    modalities = Cumulative()
    # Post-processing from https://github.com/Project-MONAI/tutorials/blob/main/modules/decollate_batch.ipynb
    postprocessing = Compose(
        [
            EnsureTyped(keys=["pred", "label"]),  # ensure Tensor type after `decollate`
            Activationsd(keys="pred", softmax=True),
            ToDeviced(keys=["pred", "label", "image"], device="cpu"),  # Move everything on cpu before inversion
            Invertd(
                keys="pred",
                transform=preprocessing,
                orig_keys="image",  # get the previously applied pre_transforms information on the `image` data field
                meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
                orig_meta_keys="image_meta_dict",  # use the meta data from `img_meta_dict` field when inverting
                nearest_interp=True,  # use nearest interpolation
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),  # Saving purposes
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=os.path.join(cfg.experiment, "result"),
                output_ext=".mhd",
                output_postfix="pred",
                resample=False,
                print_log=False,
            ),
            AsDiscreted(keys=["pred", "label"], to_onehot=cfg.out_channels),  # One-hot for metrics
            Orientationd(keys=["pred", "label"], axcodes="RAS"),  # Ensure same orientation for metrics
            # some dataset, e.g. OAI-ZIB, has different orientation for annotations
        ]
    )

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            start = time.time()
            if torch.cuda.is_available():
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].cuda(non_blocking=True)
            if "modality" in batch.keys() and modalities is not None:
                modalities.extend(batch["modality"])

            with autocast(enabled=cfg.amp, device_type=device):
                if model_inferer is not None:
                    batch["pred"] = model_inferer(
                        batch["image"],
                        modalities=batch["modality"] if "modality" in batch.keys() else None
                    )
                else:
                    batch["pred"] = model(
                        batch["image"],
                        modalities=batch["modality"] if "modality" in batch.keys() else None
                    )
            batch = [postprocessing(i) for i in decollate_batch(batch)]
            y_pred, y = from_engine(["pred", "label"])(batch)
            for metric in metrics:
                metric(y_pred=y_pred, y=y)
            logger.info(f"Test {idx+1}/{len(test_loader)} [{time.time() - start:.4f}s]")

    for metric in metrics:
        logger.info(f"Test result {metric.__class__.__name__}")
        log_metric(
            metric,
            modalities,
            logger,
            classes=cfg.classes,
            modality_names=cfg.modalities,
        )
        avg = metric.aggregate().mean().item()
        logger.info(f"=> Mean: {avg:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Load configuration file.")
    parser.add_argument("--experiment", type=str, help="Path to the experiment.")
    parser.add_argument("--weight", type=str, help="Checkpoint name.")
    args = parser.parse_args()

    with open(os.path.join(args.experiment, "cfg.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    cfg = Namespace(**cfg)
    cfg.experiment = args.experiment
    cfg.weight = args.weight
    cfg.distributed = False

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    main(cfg)