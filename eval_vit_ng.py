import torch
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from utils.dataset_utils import get_dataset
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.encoder_utils import get_pretrained_encoder
from utils.args_utils import parse_args
from models.vit_ng import ViT_NG
import timm
from torchvision import datasets, transforms
import os

def main():
    seed_everything(5)
    args = parse_args()

    # model
    model = ViT_NG(**args.__dict__)
    model.init_encoder()

    # dataset
    # train_dataset, test_dataset = get_dataset(dataset=args.dataset, data_path=args.data_path)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    #                           shuffle=True, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    #                          shuffle=False, pin_memory=True)

    data_config = timm.data.resolve_model_data_config(model.encoder)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    val_dataset = datasets.ImageFolder(root=os.path.join(args.root, 'val'), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.works,
                            shuffle=False, pin_memory=True)
    # train_transform = timm.data.create_transform(**data_config, is_training=True)
    # train_dataset = datasets.ImageFolder(root=os.path.join(args.root, 'train'), transform=train_transform)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.works,
    #                           shuffle=True, pin_memory=True)


    # trainer
    wandb_logger = WandbLogger(
        name=f"{args.dataset}-{args.run_name}",
        project=args.project,
        entity=args.entity,
        offline=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        gpus=args.num_gpus,
        max_epochs=args.epochs,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        logger=wandb_logger,
        enable_checkpointing=False,
        precision=16,
        callbacks=[lr_monitor]

    )
    # trainer.fit(model, train_loader, test_loader)
    trainer.test(model, val_loader)
    wandb.finish()


if __name__ == '__main__':
    main()
