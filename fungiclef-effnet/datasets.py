"""
modified from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
# import torch
# from torchvision import datasets
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode

Image.MAX_IMAGE_PIXELS = None


# Training transforms
def get_train_transform(image_size, pretrained):
    """
    training transformations
    : param image_size: Image size of resize when applying transforms.
    """
    train_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
        # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        # transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_transform


# Validation transforms
def get_valid_transform(image_size, pretrained):
    """
    validation transformations
    """
    valid_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return valid_transform


# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained:  # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:  # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize


# def get_datasets_from_folder(pretrained):
#     """
#     Function to prepare the Datasets.
#     :param pretrained: Boolean, True or False.
#     Returns the training and validation datasets along
#     with the class names.
#     """
#     ROOT_DIR = Path('__file__').parent.absolute().parent / 'data'
#     dataset = datasets.ImageFolder(
#         ROOT_DIR,
#         transform=(get_train_transform(IMAGE_SIZE, pretrained))
#     )
#     dataset_test = datasets.ImageFolder(
#         ROOT_DIR,
#         transform=(get_valid_transform(IMAGE_SIZE, pretrained))
#     )
#     dataset_size = len(dataset)
#     # Calculate the validation dataset size.
#     valid_size = int(VALID_SPLIT * dataset_size)
#     # Radomize the data indices.
#     indices = torch.randperm(len(dataset)).tolist()
#     # Training and validation sets.
#     dataset_train = Subset(dataset, indices[:-valid_size])
#     dataset_valid = Subset(dataset_test, indices[-valid_size:])
#     return dataset_train, dataset_valid, dataset.classes


def get_datasets(pretrained, image_size, validation_frac, seed=42):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along
    with the class names.
    """
    data_dir = Path('__file__').parent.absolute().parent / 'data'
    metadata_dir = Path('__file__').parent.absolute().parent / 'metadata'
    dataset = CustomImageDataset(
        label_file_path=metadata_dir / "FungiCLEF2023_train_metadata_PRODUCTION.csv",
        img_dir=data_dir / "DF20",
        transform=(get_train_transform(image_size, pretrained))
    )
    targets = dataset.target
    train_indices, test_indices = train_test_split(np.arange(targets.shape[0]), stratify=targets,
                                                   test_size=validation_frac, seed=seed)
    train_dataset = Subset(dataset, indices=train_indices)
    val_dataset = Subset(dataset, indices=test_indices)
    return train_dataset, val_dataset, dataset.classes


class CustomImageDataset(Dataset):
    def __init__(self, label_file_path: str, img_dir: str, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(label_file_path, dtype={"class_id": "int64"})
        self.img_labels = self.img_labels[["image_path", "class_id"]]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = self.img_labels["class_id"].unique()
        self.target = self.img_labels["class_id"].values

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # from torchvision.io import read_image
        # image = read_image(img_path)
        image = Image.open(img_path).convert('RGB')  # hopefully this handles greyscale cases
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_data_loaders(dataset_train, dataset_valid, batch_size, num_workers):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    :param batch_size: batch_size.
    :param num_workers: Number of parallel processes for data preparation.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        # pin_memory=True,
        # pin_memory_device=device,
        timeout=15,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        # pin_memory=True,
        # pin_memory_device=device,
        timeout=15,
        persistent_workers=True,
    )
    return train_loader, valid_loader
