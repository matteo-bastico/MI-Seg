import os
from argparse import Namespace, ArgumentParser

from monai import transforms, data
from typing import List, Union, Any
from torch.utils.data import ConcatDataset
from pytorch_lightning import LightningDataModule, Trainer
from torch.utils.data.distributed import DistributedSampler
from data.utils import load_decathlon_datalist_with_modality
from pytorch_lightning.utilities.argparse import from_argparse_args


class MultiModalDataModule(LightningDataModule):
    def __init__(self,
                 data_dirs: List[str],
                 json_lists: List[str],
                 space_x: float,
                 space_y: float,
                 space_z: float,
                 roi_x: int,
                 roi_y: int,
                 roi_z: int,
                 patches_training_sample: int,
                 randFlipd_prob: float,
                 randRotate90d_prob: float,
                 randScaleIntensityd_prob: float,
                 randShiftIntensityd_prob: float,
                 use_normal_dataset: bool = False,
                 cache_num: int = 12,
                 loader_workers: int = 8,
                 batch_size: int = 8,
                 num_workers: int = 8,
                 ):
        super().__init__()
        self.data_dirs = data_dirs
        self.datalist_jsons = [os.path.join(data_dir, json_list) for data_dir, json_list in zip(data_dirs, json_lists)]
        self.train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(space_x, space_y, space_z),
                    mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityd(keys=["image"]),
                transforms.SpatialPadd(keys=["image", "label"],
                                       spatial_size=(roi_x, roi_y, roi_z),
                                       value=0),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(roi_x, roi_y, roi_z),
                    pos=1,
                    neg=1,
                    num_samples=patches_training_sample,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=randFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=randFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=randFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(keys=["image", "label"], prob=randRotate90d_prob, max_k=3),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=randScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=randShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(space_x, space_y, space_z),
                    mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityd(keys=["image"]),
                transforms.SpatialPadd(keys=["image", "label"],
                                       spatial_size=(roi_x, roi_y, roi_z),
                                       value=0),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        self.use_normal_dataset = use_normal_dataset
        self.cache_num = cache_num
        self.loader_workers = loader_workers
        self.batch_size = batch_size
        self.num_workers = num_workers

    @classmethod
    def from_argparse_args(
            cls, args: Union[Namespace, ArgumentParser], **kwargs: Any
    ) -> Union["LightningDataModule", "Trainer"]:
        '''
        return cls(
            data_dir=args.data_dir,
            json_lists=args.json_lists,
            space_x=args.space_x,
            space_y=args.space_y,
            space_z=args.space_z,
            roi_x=args.roi_x,
            roi_y=args.roi_y,
            roi_z=args.roi_z,
            patches_training_sample=args.patches_training_sample,
            randFlipd_prob=args.randFlipd_prob,
            randRotate90d_prob=args.randRotate90d_prob,
            randScaleIntensityd_prob=args.randScaleIntensityd_prob,
            randShiftIntensityd_prob=args.randShiftIntensityd_prob,
            use_normal_dataset=args.use_normal_dataset,
            cache_num=args.cache_num,
            loader_workers=args.loader_workers,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        '''
        return from_argparse_args(cls, args)

    def prepare_data(self):
        # download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        pass

    def setup(self, stage: str):
        if stage == "fit":
            # Train
            datalists = [load_decathlon_datalist_with_modality(
                datalist_json,
                True,
                "training",
                base_dir=data_dir
            ) for data_dir, datalist_json in zip(self.data_dirs, self.datalist_jsons)]
            if self.use_normal_dataset:
                train_datasets = [data.Dataset(
                    data=datalist,
                    transform=self.train_transforms,
                ) for datalist in datalists]
            else:
                train_datasets = [data.CacheDataset(
                    data=datalist,
                    transform=self.train_transforms,
                    cache_num=self.cache_num,
                    cache_rate=1.0,
                    num_workers=self.loader_workers
                ) for datalist in datalists]
            self.train_dataset = ConcatDataset(train_datasets)
            # Validation
            self.val_dataset = self._get_test_val_dataset("validation")

        if stage == "test":
            self.test_dataset = self._get_test_val_dataset("test")

        if stage == "predict":
            self.predict_dataset = self._get_test_val_dataset("test")

    def _get_test_val_dataset(self, split):
        datalists = [load_decathlon_datalist_with_modality(
            datalist_json,
            True,
            split,
            base_dir=data_dir
        ) for data_dir, datalist_json in zip(self.data_dirs, self.datalist_jsons)]
        test_datasets = [data.Dataset(data=datalist, transform=self.val_transforms) for datalist in datalists]
        return ConcatDataset(test_datasets)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            shuffle=True
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return data.DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


def get_loaders(args):
    data_dirs = args.data_dirs
    datalist_jsons = [os.path.join(data_dir, json_list) for data_dir, json_list in zip(args.data_dirs, args.json_lists)]
    train_transforms = transforms.Compose(
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
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
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
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    use_normal_dataset = args.use_normal_dataset
    cache_num = args.cache_num
    loader_workers = args.loader_workers
    batch_size = args.batch_size
    num_workers = args.num_workers
    # Train
    if not args.test_mode:
        datalists = [load_decathlon_datalist_with_modality(
            datalist_json,
            True,
            "training",
            base_dir=data_dir
        ) for data_dir, datalist_json in zip(data_dirs, datalist_jsons)]
        if use_normal_dataset:
            train_datasets = [data.Dataset(
                data=datalist,
                transform=train_transforms,
            ) for datalist in datalists]
        else:
            train_datasets = [data.CacheDataset(
                data=datalist,
                transform=train_transforms,
                cache_num=cache_num,
                cache_rate=1.0,
                num_workers=loader_workers
            ) for datalist in datalists]
        train_dataset = ConcatDataset(train_datasets)
        train_sampler = DistributedSampler(train_dataset) if args.distributed else None
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
        datalists = [load_decathlon_datalist_with_modality(
            datalist_json,
            True,
            "validation",
            base_dir=data_dir
        ) for data_dir, datalist_json in zip(data_dirs, datalist_jsons)]
        val_datasets = [data.Dataset(data=datalist, transform=train_transforms) for datalist in datalists]
        val_dataset = ConcatDataset(val_datasets)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
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
        # Validation
        datalists = [load_decathlon_datalist_with_modality(
            datalist_json,
            True,
            "test",
            base_dir=data_dir
        ) for data_dir, datalist_json in zip(data_dirs, datalist_jsons)]
        test_datasets = [data.Dataset(data=datalist, transform=val_transforms) for datalist in datalists]
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
        return test_loader
