"""
modified from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/

switch to v2 of transforms
https://pytorch.org/vision/stable/transforms.html#v1-or-v2-which-one-should-i-use
"""

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
# from torchvision import datasets
from PIL import Image
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.v2 import InterpolationMode

Image.MAX_IMAGE_PIXELS = None
WORKER_TIMEOUT_SECS = 120


# Training transforms
def get_train_transform(image_size, pretrained):
    """
    training transformations
    : param image_size: Image size of resize when applying transforms.
    """
    train_transform = v2.Compose([
        v2.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        # v2.CenterCrop(224),
        v2.RandomCrop(224),
        v2.TrivialAugmentWide(interpolation=InterpolationMode.BICUBIC),
        # v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
        # v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        # v2.RandomAutocontrast(),
        # v2.ToTensor(),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        normalize_transform(pretrained)
    ])
    return train_transform


# Validation transforms
def get_valid_transform(image_size, pretrained):
    """
    validation transformations
    """
    valid_transform = v2.Compose([
        v2.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(224),
        # v2.ToTensor(),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        normalize_transform(pretrained)
    ])
    return valid_transform


# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained:  # Normalization for pre-trained weights.
        normalize = v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:  # Normalization when training from scratch.
        normalize = v2.Normalize(
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


def get_datasets(pretrained, image_size, validation_frac, oversample=False, undersample=False,
                 oversample_prop=0.1, equal_undersampled_val=True, seed=42):
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

    if equal_undersampled_val:
        # get 4 samples per class for validation
        sample_dict = Counter(targets)
        sample_dict = {k: 4 for k in sample_dict.keys()}
        under = RandomUnderSampler(sampling_strategy=sample_dict, random_state=seed)
        all_indices = np.array(list(np.arange(targets.shape[0])))
        test_indices, _ = under.fit_resample(all_indices.reshape(-1, 1), targets)
        test_indices = test_indices.squeeze()
        train_indices = np.delete(all_indices, test_indices)
    else:
        train_indices, test_indices = train_test_split(np.arange(targets.shape[0]), stratify=targets,
                                                       test_size=validation_frac, random_state=seed)

    if undersample:
        under = RandomUnderSampler(random_state=seed)
        if oversample:
            sample_dict = Counter(targets)
            majority = max(sample_dict.values())
            sample_dict = {k: max(v, int(majority * oversample_prop)) for k, v in sample_dict.items()}
            over = RandomOverSampler(sampling_strategy=sample_dict, random_state=seed)
            train_indices, _ = over.fit_resample(train_indices.reshape(-1, 1), targets[train_indices])
            train_indices = train_indices.squeeze()
            print(train_indices.shape)
        train_indices, _ = under.fit_resample(train_indices.reshape(-1, 1), targets[train_indices])
        train_indices = train_indices.squeeze()

    train_dataset = Subset(dataset, indices=train_indices)
    train_dataset.target = targets[train_indices]
    val_dataset = Subset(dataset, indices=test_indices)
    val_dataset.target = targets[test_indices]
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
        """If an image can't be read, print the error and return None"""
        try:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            # from torchvision.io import read_image
            # image = read_image(img_path)
            image = Image.open(img_path).convert('RGB')  # hopefully this handles greyscale cases
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
        except Exception as e:
            print("issue loading image")
            print(e)
            return None
        return image, label


def collate_fn(batch):
    """Filter None from the batch"""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_data_loaders(train_dataset, val_dataset, batch_size, num_workers, balanced_sampler):
    """
    Prepares the training and validation data loaders.
    :param train_dataset: The training dataset.
    :param val_dataset: The validation dataset.
    :param batch_size: batch_size.
    :param num_workers: Number of parallel processes for data preparation.
    Returns the training and validation data loaders.
    """

    if balanced_sampler:
        target = train_dataset.target
        # https://pytorch.org/docs/stable/data.html
        # https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        weight_per_class = 1. / class_sample_count
        weight_per_sample = np.array([weight_per_class[class_idx] for class_idx in target])
        weight_per_sample = torch.from_numpy(weight_per_sample)
        weight_per_sample = weight_per_sample.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_per_sample, len(weight_per_sample))
        # class_sample_counts = Counter(train_dataset.target)
        # class_sample_counts = [v for k, v in sorted(class_sample_counts.items(), key=lambda x: x[0])]
        # weights = 1 / torch.Tensor(class_sample_counts)
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                                  num_workers=num_workers,
                                  # pin_memory=True,
                                  # pin_memory_device=device,
                                  timeout=WORKER_TIMEOUT_SECS,
                                  persistent_workers=True,
                                  collate_fn=collate_fn,
                                  )

    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            # pin_memory=True,
            # pin_memory_device=device,
            timeout=WORKER_TIMEOUT_SECS,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        # pin_memory=True,
        # pin_memory_device=device,
        timeout=WORKER_TIMEOUT_SECS,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader
