import warnings
import wandb

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from networks.lightning_monai import LitMonai
from data.multi_modal import MultiModalDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.parser import add_model_argparse_args, add_data_argparse_args
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def main(args):
    lit_data = MultiModalDataModule.from_argparse_args(args)
    lit_model = LitMonai.from_argparse_args(args)
    wandb_logger = WandbLogger(
        name=args.experiment_name if args.experiment_name else None,
        group=args.group if args.group else None,
        project=args.project if args.project else None,
        entity=args.entity if args.entity else None,
        log_model=False,  # Do not log the models online, too heavy
        mode=args.wandb_mode,
        save_dir=args.default_root_dir,
    )
    early_stop_callback = EarlyStopping(
        monitor="val/accuracy/avg",
        min_delta=args.min_delta,
        patience=args.patience,
        verbose=True,
        mode="max"
    )
    lr_monitor_callback = LearningRateMonitor(
        logging_interval='step',
        log_momentum=False
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val/accuracy/avg',  # Quantity to monitor.
        mode='max',
        verbose=True,
        save_last=True,  # Saves a copy of the checkpoint to a file last.ckpt whenever a checkpoint file gets saved.
        save_top_k=args.save_top_k,  # The best k models according to the quantity monitored will be saved.
        auto_insert_metric_name=False,  # better when metric name contains '/'
    )
    # wandb_logger.watch(lit_model, log="all")  # Too many gradients, avoid to log
    # Real training
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[
            early_stop_callback,
            lr_monitor_callback,
            checkpoint_callback
        ],
        logger=wandb_logger
    )
    # Not supported in ddp
    try:
        trainer.tune(lit_model, lit_data)
    except Exception as e:
        warnings.warn("Tuning skipped due to Exception: {}".format(e))

    trainer.fit(
        lit_model,
        lit_data,
        ckpt_path=args.ckpt_path if args.ckpt_path else None
    )
    trainer.test(
        lit_model,
        lit_data,
        ckpt_path='best'
    )
    # wandb_logger.experiment.unwatch(lit_model)
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = add_model_argparse_args(parser)
    parser = add_data_argparse_args(parser)
    args = parser.parse_args()
    # Fix the feature size because only vanilla_unet require a list, otherwise just int for the first layer
    if len(args.feature_size) == 1:
        args.feature_size = args.feature_size[0]
    print(args)
    main(args)
