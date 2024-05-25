import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import os

def get_dataset(dataset, data_path):
    # assert dataset in ["cifar100", "imagenet100"]
    if dataset == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        train_transforms = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(brightness=63 / 255),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        val_transforms = transforms.Compose(
            [transforms.Resize(32),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                      transform=train_transforms,
                                                      download=True)
        test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                                     transform=val_transforms,
                                                     download=True)

    elif dataset == "imagenet100":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data_path = os.path.join(data_path, "imagenet100")
        train_tansforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_tansforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"),
                                             transform=train_tansforms)
        test_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"),
                                            transform=val_tansforms)

    return train_dataset, test_dataset



if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset(dataset="imagenet100", data_path="/ppio_net0/torch_ds")
    print(train_dataset[0])
