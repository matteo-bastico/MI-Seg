import torch
import inspect
import numpy as np
import torch.nn as nn

from functools import partial
from typing import Union, Sequence
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from pytorch_lightning import LightningModule
from monai.inferers import sliding_window_inference
from networks.utils import model_from_argparse_args
from monai.optimizers.lr_scheduler import WarmupCosineSchedule


class LitMonai(LightningModule):
    def __init__(self,
                 model: nn.Module,
                 out_channels: int,
                 criterion: str = "dice_focal",
                 squared_pred: bool = True,
                 smooth_nr: float = 0.0,
                 smooth_dr: float = 1e-6,
                 learning_rate: float = 1e-4,
                 optim_name: str = 'adamw',
                 reg_weight: float = 1e-5,
                 momentum: float = 0.99,
                 roi_size: Union[Sequence[int], int] = (96, 96, 96),
                 infer_overlap: float = 0.5,
                 sw_batch_size: int = 1,
                 infer_cpu: bool = False,
                 batch_size: int = 1,
                 scheduler: str = 'reduce_on_plateau',
                 warmup_epochs: Union[int, None] = None,
                 patience: Union[int, None] = None,
                 check_val_every_n_epoch: Union[int, None] = None,
                 max_epochs: int = 5000,
                 include_background: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.model = model
        if criterion == "dice_focal":
            self.criterion = DiceFocalLoss(
                include_background=include_background,
                to_onehot_y=True,
                softmax=True,
                squared_pred=True,
                smooth_nr=smooth_nr,
                smooth_dr=smooth_dr
            )
        elif criterion == "dice_ce":
            self.criterion = DiceCELoss(
                include_background=include_background,
                to_onehot_y=True,
                softmax=True,
                squared_pred=squared_pred,
                smooth_nr=smooth_nr,
                smooth_dr=smooth_dr
            )
        else:
            raise ValueError("Criterion {} not implemented, please chose another optimizer.".format(criterion))
        self.post_label = AsDiscrete(
            to_onehot=out_channels
        )
        self.post_pred = AsDiscrete(
            argmax=True,
            to_onehot=out_channels
        )
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction=MetricReduction.MEAN,
            get_not_nans=True
        )
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optim_name = optim_name
        self.reg_weight = reg_weight
        self.momentum = momentum
        self.infer_cpu = infer_cpu
        self.model_inferer = partial(
            sliding_window_inference,
            predictor=self.model,
            roi_size=roi_size,
            overlap=infer_overlap,
            sw_batch_size=sw_batch_size,
            device=torch.device("cpu") if infer_cpu else None
        )
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.max_epochs = max_epochs
        # For hyperparameters saving, additional args to log, can be useful to load from checkpoint
        self.__dict__.update(kwargs)
        self.save_hyperparameters(ignore=[
            'model',
            'criterion',
            'post_label',
            'post_pred',
            'dice_metric',
            'model_inferer',
            'roi_size'  # roi_x, roi_y, roi_z should be already in kwargs
        ])

    @classmethod
    def from_argparse_args(cls, args):
        model = model_from_argparse_args(args)
        # Additional args to log, can be useful to load from checkpoint
        params = vars(args)
        class_kwargs = inspect.signature(cls.__init__).parameters
        additional_kwargs = {name: params[name] for name in params if name not in class_kwargs}
        return cls(
            model=model,
            out_channels=args.out_channels,
            criterion=args.criterion,
            squared_pred=args.squared_dice,
            smooth_nr=args.smooth_nr,
            smooth_dr=args.smooth_dr,
            learning_rate=args.lr,
            optim_name=args.optim_name,
            reg_weight=args.reg_weight,
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            infer_overlap=args.infer_overlap,
            sw_batch_size=args.sw_batch_size,
            infer_cpu=args.infer_cpu,
            batch_size=args.batch_size,
            scheduler=args.scheduler,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience / 2,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            max_epochs=args.max_epochs,
            include_background=not args.no_include_background,
            **additional_kwargs
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]
        modality = None
        if 'modality' in batch.keys():
            modality = batch['modality']
        logits = self.model(image, modality)  # Or self.forward(x)
        loss = self.criterion(logits, label)
        self.log("train/loss",
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=self.batch_size
                 )
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    # This method is called before training_epoch_end()
    def validation_epoch_end(self, validation_step_outputs):
        self._shared_eval_end(validation_step_outputs, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test")

    def test_epoch_end(self, test_step_outputs):
        self._shared_eval_end(test_step_outputs, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        image = batch["image"]
        label = batch["label"]
        modality = None
        if 'modality' in batch.keys():
            modality = batch['modality']
        logits = self.model_inferer(image, modalities=modality)
        if self.infer_cpu:
            label = label.cpu()
        loss = self.criterion(logits, label)
        output = [self.post_pred(i) for i in decollate_batch(logits)]
        label = [self.post_label(i) for i in decollate_batch(label)]
        accuracy = self.dice_metric(y_pred=output, y=label)
        avg_accuracy = torch.nanmean(accuracy)
        accuracy_per_class = torch.nanmean(accuracy, dim=0)
        accuracy_per_class_dict = {
            f"{prefix}/accuracy/class_{idx}": acc for idx, acc in enumerate(accuracy_per_class)
        }
        self.log_dict(
            accuracy_per_class_dict,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=1
        )
        self.log_dict({
            f"{prefix}/loss/avg": loss,
            f"{prefix}/accuracy/avg": avg_accuracy.item()},  # Here item is needed to avid error in Early Stopping
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=1  # This is forced in the loader
        )
        return {
            'loss': loss,
            'accuracy': avg_accuracy,
            'modality': modality
        }

    def _shared_eval_end(self, validation_step_outputs, prefix):
        # Here we compute and log the accuracy per modality
        # validation_step_outputs is a list of {
        #             'loss': loss,
        #             'accuracy': avg_accuracy,
        #             'modality': modality
        #         }
        validation_outputs = {k: [dic[k] for dic in validation_step_outputs] for k in validation_step_outputs[0]}
        validation_outputs = {k: torch.stack(v) for k, v in validation_outputs.items()}
        # TODO: here it crashed if not converted to numpy
        validation_outputs = {k: v.cpu().numpy() for k, v in validation_outputs.items()}
        # modality is a two dimension tensor -> to one dimension (assuming validation batch_size is 1)
        validation_outputs["modality"] = validation_outputs["modality"].squeeze()
        accuracy_per_modality = {}
        loss_per_modality = {}
        for modality in np.unique(validation_outputs['modality']):
            accuracy_per_modality[f"{prefix}/accuracy/modality_{int(modality)}"] = \
                np.nanmean(validation_outputs['accuracy'][validation_outputs['modality'] == modality])
            loss_per_modality[f"{prefix}/loss/modality_{int(modality)}"] = \
                np.nanmean(validation_outputs['loss'][validation_outputs['modality'] == modality])
        self.log_dict(accuracy_per_modality,
                      logger=True,
                      sync_dist=True
                      )
        self.log_dict(loss_per_modality,
                      logger=True,
                      sync_dist=True
                      )

    '''
    # By default, the predict_step() method runs the forward() method. 
    # In order to customize this behaviour, simply override the predict_step() method.
    '''

    def configure_optimizers(self):
        if self.optim_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.reg_weight
            )
        elif self.optim_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.reg_weight
            )
        elif self.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.reg_weight
            )
        else:
            raise ValueError("Optimization {} not implemented, please chose another optimizer.".format(self.optim_name))

        if self.scheduler == 'warmup_cosine':
            scheduler = WarmupCosineSchedule(
                optimizer=optimizer,
                warmup_steps=self.warmup_epochs,
                t_total=self.max_epochs
            )
        elif self.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.max_epochs
            )
        elif self.scheduler == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=self.patience,
            )
        else:
            raise ValueError("Scheduler {} not implemented, please chose another optimizer.".format(self.scheduler))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'monitor': "val/loss/avg",
                'frequency': self.check_val_every_n_epoch
            }
        }
