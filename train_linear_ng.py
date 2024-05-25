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
from models.linear import MLP


def main():
    seed_everything(5)
    args = parse_args()

    # model
    model = MLP(**args.__dict__)
    model.init_encoder()

    # dataset
    train_dataset, test_dataset = get_dataset(dataset=args.dataset, data_path=args.data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=True)


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
    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, test_loader)
    wandb.finish()


if __name__ == '__main__':
    main()
