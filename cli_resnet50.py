from lightning.pytorch import cli
from dataloaders.imagenet import ImageNetDataModule
from models.resnet50 import VisionModel
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

import os

os.environ['CURL_CA_BUNDLE'] = ''


class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        parser.link_arguments(
            "data.batch_size", "model.batch_size"
        )
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")


if __name__ == "__main__":
    CLI(VisionModel, ImageNetDataModule, save_config_callback=None, seed_everything_default=6)
