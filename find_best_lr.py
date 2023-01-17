import warnings
import wandb
import time
import json
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
    args.auto_lr_find = True
    lit_data = MultiModalPelvicDataModule.from_argparse_args(args)
    lit_model = LitMonai.from_argparse_args(args)
    # It is not supported in ddp, do on single node and then start other trainer with new lr
    # Suggestion: do with accelerator 'gpu'
    if args.devices is not None:
        if int(args.devices) > 1:
            raise ValueError("Provided devices > 1 or auto. If several devices are available, "
                             "the best learning rate has to be calculated on single device. "
                             "Pytorch Lightning does not support learning rate tuning in ddp.")

    trainer = Trainer.from_argparse_args(args, devices=1, logger=False)
    lr_finder = trainer.tuner.lr_find(
        lit_model,
        lit_data,
        num_training=args.num_training
    )  # TODO: See if useful to add other parameters as argument
    fig = lr_finder.plot(suggest=True)
    new_lr = lr_finder.suggestion()
    print("Best learning rate found for this trial with tuner: ", new_lr)
    lit_model.hparams.learning_rate = new_lr
    now = time.strftime("%Y%m%d-%H%M%S")
    Path(os.path.join(trainer.default_root_dir, 'lr_finder', now)).mkdir(exist_ok=True, parents=True)
    plt.savefig(os.path.join(trainer.default_root_dir, 'lr_finder', now, 'plot.pdf'))
    with open(os.path.join(trainer.default_root_dir, 'lr_finder', now, 'args.json'), 'w+') as f:
        json.dump(args.__dict__, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = add_model_argparse_args(parser)
    parser = add_data_argparse_args(parser)
    parser.add_argument("--num_trainings", type=int, default=100, help="Number of training for lr_find")
    parser.add_argument_group("lr_tuner")

    args = parser.parse_args()
    print(args)
    main(args)
