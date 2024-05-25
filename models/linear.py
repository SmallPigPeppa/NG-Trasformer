import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchvision.models import resnet50


class MLP(pl.LightningModule):
    def __init__(self, dim_feature, num_classes, lr, epochs, warmup_epochs, **kwargs):
        super(MLP, self).__init__()
        self.dim_feature = dim_feature
        self.num_calsses = num_classes
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.extra_args = kwargs
        self.encoder = None
        self.fc = nn.Linear(dim_feature, num_classes)

    def on_train_epoch_start(self):
        self.encoder.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=self.warmup_epochs,
                                                  max_epochs=self.epochs)
        return [optimizer], [scheduler]

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        out = self.fc(x)
        return out

    def share_step(self, batch, batch_idx):
        x, targets = batch
        logits = self.forward(x)
        # ce loss
        ce_loss = F.cross_entropy(logits, targets)
        # acc
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == targets) / targets.shape[0]
        return {"acc": acc, "loss": ce_loss}

    def training_step(self, batch, batch_idx):
        out = self.share_step(batch, batch_idx)
        log_dict = {"train/" + k: v for k, v in out.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return out

    def validation_step(self, batch, batch_idx):
        out = self.share_step(batch, batch_idx)
        log_dict = {"val/" + k: v for k, v in out.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return out

    def test_step(self, batch, batch_idx):
        out = self.share_step(batch, batch_idx)
        log_dict = {"test/" + k: v for k, v in out.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return out
