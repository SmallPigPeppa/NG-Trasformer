import os
import timm
from typing import Optional
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Subset
from lightning import LightningDataModule
from torchvision.datasets import ImageFolder


class ImageNetDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int = 64,
            num_workers: int = 8,
            path: Optional[str] = None,
            config: Optional[str] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = config

    def setup(
            self,
            stage: Optional[str] = None,
    ) -> None:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_dataset = ImageFolder(root=os.path.join(self.path, 'train'), transform=train_transforms)
        self.val_dataset = ImageFolder(root=os.path.join(self.path, 'val'), transform=val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == '__main__':
    weight = 'vit_base_patch16_224'
    model = timm.create_model(weight, pretrained=False)
    data_config = timm.data.resolve_model_data_config(model)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    print(data_config)
