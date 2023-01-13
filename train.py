import warnings
import wandb
import os.path
import matplotlib.pyplot as plt

from pathlib import Path
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from networks.lightning_monai import LitMonai
from data.multi_modal_pelvic import MultiModalPelvicDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.parser import add_model_argparse_args, add_data_argparse_args
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def main(args):
    lit_data = MultiModalPelvicDataModule.from_argparse_args(args)
    lit_model = LitMonai.from_argparse_args(args)
    wandb_logger = WandbLogger(
        name=args.experiment_name if args.experiment_name else None,
        group=args.group if args.group else None,
        project=args.project if args.project else None,
        entity=args.entity if args.entity else None,
        log_model=False,  # Do not log the models online, too heavy
        mode=args.wandb_mode
    )
    if args.auto_lr_find:
        # It is not supported in ddp, do on single node and then start other trainer with new lr
        # Suggestion: do with accelerator 'gpu'
        if args.devices is None or int(args.devices) > 1:
            warnings.warn("Provided devices > 1 or auto. If several devices are available, "
                          "the best learning rate will be calculated on single device. "
                          "Pytorch Lightning does not support learning rate tuning in ddp.")
        trainer = Trainer.from_argparse_args(args, devices=1, logger=False)
        lr_finder = trainer.tuner.lr_find(lit_model, lit_data)
        fig = lr_finder.plot(suggest=True)
        new_lr = lr_finder.suggestion()
        print("Best learning rate found for this trial with tuner: ", new_lr)
        lit_model.hparams.learning_rate = new_lr
        Path(os.path.join(trainer.default_root_dir, args.project, 'lr_finder')).mkdir(exist_ok=True, parents=True)
        plt.savefig(os.path.join(trainer.default_root_dir, args.project, 'lr_finder',
                                 wandb_logger.experiment.id + '.pdf'))
        args.lr = new_lr
        args.auto_lr_find = None

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
    # trainer.tune(lit_model, lit_data, lr_find_kwargs={'num_training': 10})
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
    print(args)
    main(args)
