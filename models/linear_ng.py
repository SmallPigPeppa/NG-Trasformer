import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchvision.models import resnet50
from torchmetrics.classification.accuracy import Accuracy
import numpy as np


class MLP(pl.LightningModule):
    def __init__(self, dim_feature, num_class, lr, epochs, warmup_epochs, **kwargs):
        super(MLP, self).__init__()
        self.dim_feature = dim_feature
        self.num_class = num_class
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.extra_args = kwargs
        self.fc = nn.Linear(dim_feature, num_class)
        self.acc = Accuracy(num_classes=num_class, task="multiclass", top_k=1)
        self.encoder = None

        # Initialize energy values
        self.energy_values = torch.tensor(np.random.normal(size=num_class), dtype=torch.float32, device=self.device)

    def init_encoder(self):
        encoder = resnet50()
        if "cifar" in self.extra_args['dataset']:
            encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            encoder.maxpool = nn.Identity()

        state = torch.load(self.extra_args['pretrain_ckpt'], map_location="cpu")["state_dict"]
        for k in list(state.keys()):
            if "encoder" in k:
                state[k.replace("encoder.", "")] = state[k]
            if "backbone" in k:
                state[k.replace("backbone.", "")] = state[k]
            del state[k]

        encoder.fc = nn.Identity()
        encoder.load_state_dict(state, strict=False)
        print(f"Loaded {self.extra_args['pretrain_ckpt']}")
        # self.encoder = encoder.eval()
        self.encoder = encoder

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=self.warmup_epochs,
                                                  max_epochs=self.epochs)
        return [optimizer], [scheduler]

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        out = self.fc(x)
        # self.energy_values.to(self.device)
        # Sort the output and assign energy values
        sorted_indices = torch.argsort(out, dim=1, descending=True)
        assigned_energies = torch.gather(self.energy_values.expand(out.size(0), -1).to(self.device), 1, sorted_indices)
        # assigned_energies = assigned_energies * 10000

        # # Normalize the assigned energies with softmax
        # normalized_energies = F.softmax(assigned_energies, dim=1)
        # # return normalized_energies

        # Normalize the assigned energies by dividing by the sum
        sum_energies = assigned_energies.sum(dim=1, keepdim=True)
        normalized_energies = assigned_energies / sum_energies

        out = F.softmax(out, dim=1)
        y = out - out.detach() + normalized_energies

        y = out

        return y

    def share_step(self, batch, batch_idx):
        x, targets = batch
        logits = self.forward(x)
        # ce loss
        ce_loss = F.cross_entropy(logits, targets)
        # acc
        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, targets)
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


if __name__ == '__main__':
    from utils.args_utils import parse_args

    args = parse_args()
    model = MLP(**args.__dict__)
