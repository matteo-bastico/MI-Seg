import os

from collections.abc import Sequence
from monai import transforms, data

from monai.data import load_decathlon_datalist
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from utils import comm


def get_loaders(args):
    data_dirs = args.data_dirs
    datalist_jsons = [os.path.join(data_dir, json_list) for data_dir, json_list in zip(args.data_dirs, args.json_lists)]
    train_transforms = [transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityd(keys=["image"]),
            transforms.RandZoomd(
                keys=["image", "label"],
                prob=args.zoom_prob,
                min_zoom=args.min_zoom[idx] if isinstance(args.min_zoom, Sequence) else args.min_zoom,
                max_zoom=args.max_zoom[idx] if isinstance(args.max_zoom, Sequence) else args.max_zoom,
                mode=['area', 'nearest'],
                padding_mode='constant'
            ),
            transforms.SpatialPadd(keys=["image", "label"],
                                   spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   value=0),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.patches_training_sample,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.randFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.randFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.randFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.randRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.randScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.randShiftIntensityd_prob),
            transforms.EnsureTyped(keys=["image", "label"]),
            # transforms.ToTensord(keys=["image", "label"]),
        ]
    ) for idx in range(len(datalist_jsons))]
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityd(keys=["image"]),
            transforms.SpatialPadd(keys=["image", "label"],
                                   spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   value=0),
            transforms.EnsureTyped(keys=["image", "label"]),
            # transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    use_normal_dataset = args.use_normal_dataset
    cache_num = args.cache_num
    loader_workers = args.loader_workers
    batch_size = args.batch_size
    num_workers = args.num_workers
    # Train
    if not args.test_mode:
        datalists = [load_decathlon_datalist(
            datalist_json,
            True,
            "training",
            base_dir=data_dir
        ) for data_dir, datalist_json in zip(data_dirs, datalist_jsons)]
        if use_normal_dataset:
            train_datasets = [data.Dataset(
                data=datalist,
                transform=train_transforms[idx],
            ) for idx, datalist in enumerate(datalists)]
        else:
            train_datasets = [data.CacheDataset(
                data=datalist,
                transform=train_transforms[idx],
                cache_num=cache_num,
                cache_rate=1.0,
                progress=False,
                num_workers=loader_workers
            ) for idx, datalist in enumerate(datalists)]
        train_dataset = ConcatDataset(train_datasets)
        train_sampler = DistributedSampler(train_dataset) if comm.get_world_size() > 1 else None
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=num_workers > 0,  # In the example from Jean Zay they don't use this
        )
        # Validation
        datalists = [load_decathlon_datalist(
            datalist_json,
            True,
            "validation",
            base_dir=data_dir
        ) for data_dir, datalist_json in zip(data_dirs, datalist_jsons)]
        val_datasets = [data.Dataset(data=datalist, transform=val_transforms) for datalist in datalists]
        val_dataset = ConcatDataset(val_datasets)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if comm.get_world_size() > 1 else None
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=num_workers > 0,  # In the example from Jean Zay they don't use this
        )
        return train_loader, val_loader
    else:
        test_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys="image", axcodes="RAS"),
                transforms.Spacingd(
                    keys="image", pixdim=(args.space_x, args.space_y, args.space_z),
                    mode="bilinear"
                ),
                transforms.ScaleIntensityd(keys="image"),
                transforms.SpatialPadd(keys="image",
                                       spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                       value=0),
                transforms.EnsureTyped(keys=["image", "label"]),
                # transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        # Validation
        datalists = [load_decathlon_datalist(
            datalist_json,
            True,
            "test",
            base_dir=data_dir
        ) for data_dir, datalist_json in zip(data_dirs, datalist_jsons)]
        test_datasets = [data.Dataset(data=datalist, transform=test_transforms) for datalist in datalists]
        test_dataset = ConcatDataset(test_datasets)
        test_sampler = DistributedSampler(test_dataset, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=num_workers > 0,  # In the example from Jean Zay they don't use this
        )
        return test_loader, test_transforms
