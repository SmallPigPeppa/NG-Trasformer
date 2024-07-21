import torch
from torch import nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchvision.models import resnet50
from lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy


class VisionModel(LightningModule):
    def __init__(
            self,
            model_alias: str,
            weights: str = None,
            embedding_dims: int = 2048,
            num_classes: int = 1000,
            temperature: float = 1.0,
            lr: float = 1e-3,
            lr_warmup_epochs: int = 5,
            weight_decay: float = 0.0,
            batch_size: int = 256,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = resnet50(weights=self.hparams.weights)
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy(
            num_classes=self.hparams.num_classes,
            task="multiclass",
            top_k=1
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        logits = self.model(x)
        loss = self.ce_loss(logits, labels)
        acc = self.acc(logits, labels)
        self.log('train/loss', loss, sync_dist=True)
        self.log('train/acc', acc, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        logits = self.model(x)
        loss = self.ce_loss(logits, labels)
        acc = self.acc(logits, labels)
        self.log('val/loss', loss, sync_dist=True)
        self.log('val/acc', acc, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.lr,
        #     weight_decay=self.hparams.weight_decay,
        #     momentum=0.9
        # )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.lr_warmup_epochs,
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=0.01 * self.hparams.lr,
            eta_min=0.01 * self.hparams.lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
