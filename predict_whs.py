import torch
import numpy as np
from functools import partial
from argparse import ArgumentParser
from monai import transforms
from monai.transforms import AsDiscrete
from networks.utils.utils import model_from_argparse_args
from monai.inferers import sliding_window_inference
from utils.parser import add_model_argparse_args
from monai.transforms.utils import allow_missing_keys_mode
from data.utils import load_decathlon_datalist_with_modality
from pathlib import Path

import os
import nibabel as nib


_MAP = {
    1: 500,
    2: 600,
    3: 420,
    4: 550,
    5: 205,
    6: 820,
    7: 850
}


def remap_tensor(tensor, map_dict):
    for key, value in map_dict.items():
        tensor[tensor == key] = value
    return tensor


def main(args):
    # Model loading here
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model = model_from_argparse_args(args)
    model_weights = checkpoint["state_dict"]
    model.load_state_dict(model_weights)
    model = model.to(args.device)
    post_pred = AsDiscrete(argmax=True)
    # Load sample dataloader
    predict_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS", allow_missing_keys=True),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
                allow_missing_keys=True
            ),
            transforms.ScaleIntensityd(keys=["image"]),
            transforms.SpatialPadd(keys=["image", "label"],
                                   spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   value=0,
                                   allow_missing_keys=True)
        ]
    )


    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    datalist = load_decathlon_datalist_with_modality(
        datalist_json,
        True,
        "test",
        base_dir=data_dir
    )

    model_inferer = partial(
        sliding_window_inference,
        roi_size=(args.roi_x, args.roi_y, args.roi_z),
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
        progress=True
    )

    path = Path(args.result_dir)
    path.mkdir(parents=True, exist_ok=True)

    for el in datalist:
        sample = predict_transforms({
            "image": el["image"],
            # This is just to save transformation and invert the model prediction
            "label": el["image"]
        })
        model.eval()
        with torch.no_grad():
            # unsqueeze(0) to remove batch direction
            prediction = model_inferer(
                sample["image"].unsqueeze(0).to(args.device),
                modalities=el['modality']
            )
        prediction = post_pred(prediction.squeeze(0)).cpu()
        prediction.applied_operations = sample["label"].applied_operations
        with allow_missing_keys_mode(predict_transforms):
            inverted = predict_transforms.inverse(
                data={"label": prediction}
            )
        # Remap to original gt values
        final_pred = remap_tensor(inverted["label"], _MAP)
        original_affine = sample["image_meta_dict"]["affine"].numpy()
        # Use same path to save the prediction
        img_name = sample["image_meta_dict"]["filename_or_obj"].split("/")[-1]
        nib.save(
            nib.Nifti1Image(final_pred[0].astype(np.uint16), original_affine),
            os.path.join(args.result_dir, img_name.replace("image", "label"))
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    # Model data same as in training
    parser = add_model_argparse_args(parser)
    parser.add_argument("--checkpoint", default="", type=str, help="Checkpoint")
    parser.add_argument("--sample", default="", type=str, help="Checkpoint")
    parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
    parser.add_argument("--no_gpu", action="store_true", help="not use GPU on single training")
    parser.add_argument("--data_dir", default="dataset/MM-WHS", type=str, help="dataset directory(ies)")
    parser.add_argument('--json_list', default='CT_test.json', help='Json list(s) of input dataset(s)', type=str)
    parser.add_argument('--result_dir', default='dataset/MM_WHS/MM_WHS_test/CT/', help='Directory for results', type=str)
    args = parser.parse_args()
    if len(args.feature_size) == 1:
        args.feature_size = args.feature_size[0]
    args.device = "cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    args.distributed = False
    if args.device != "cpu":
        torch.cuda.set_device(args.device)
    main(args)
