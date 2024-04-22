"""
modified from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/

switch to v2 of transforms
https://pytorch.org/vision/stable/transforms.html#v1-or-v2-which-one-should-i-use
"""

import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
# from torchvision import datasets
from PIL import Image, ImageFile
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision.transforms.v2 import InterpolationMode

from paths import DATA_DIR, DATA_FROM_FOLDER_DIR, METADATA_DIR

# dataset loading issue
# https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
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
        # v2.TrivialAugmentWide(interpolation=InterpolationMode.BICUBIC),
        v2.AutoAugment(interpolation=InterpolationMode.BICUBIC),
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


def get_datasets(pretrained, image_size, validation_frac, oversample=False, undersample=False,
                 oversample_prop=0.1, equal_undersampled_val=True, seed=42):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along
    with the class names.
    """

    dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "FungiCLEF2023_train_metadata_PRODUCTION.csv",
        img_dir=DATA_DIR / "DF20",
        transform=(get_train_transform(image_size, pretrained))
    )
    val_dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "FungiCLEF2023_train_metadata_PRODUCTION.csv",
        img_dir=DATA_DIR / "DF20",
        transform=(get_valid_transform(image_size, pretrained))  # only difference
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
    val_dataset = Subset(val_dataset, indices=test_indices)
    val_dataset.target = targets[test_indices]
    return train_dataset, val_dataset, dataset.classes


class CustomImageDataset(Dataset):
    def __init__(self, label_file_path: str, img_dir: str,
                 keep_only: set | None = None,
                 exclude: set | None = None,
                 transform=None,
                 target_transform=None):
        self.img_labels = pd.read_csv(label_file_path, dtype={"class_id": "int64"})
        self.img_labels = self.img_labels[["image_path", "class_id"]]
        self.img_dir = img_dir
        self.keep_only = keep_only
        if self.keep_only is not None:
            self.img_labels = self.img_labels[self.img_labels["class_id"].isin(self.keep_only)]
        self.exclude = exclude
        if self.exclude is not None:
            self.img_labels = self.img_labels[~self.img_labels["class_id"].isin(self.exclude)]
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
            # https://pytorch.org/vision/main/_modules/torchvision/datasets/folder.html#ImageFolder
            with open(img_path, "rb") as f:
                image = Image.open(f).convert('RGB')  # hopefully this handles greyscale cases
            # image = transforms.ToPILImage()(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
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


def get_data_loaders(train_dataset, val_dataset, batch_size, num_workers, balanced_sampler,
                     timeout=WORKER_TIMEOUT_SECS):
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
                                  timeout=timeout,
                                  persistent_workers=True,
                                  collate_fn=collate_fn,
                                  )

    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            # pin_memory=True,
            # pin_memory_device=device,
            timeout=timeout,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        # pin_memory=True,
        # pin_memory_device=device,
        timeout=timeout,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader


def get_openset_datasets(pretrained, image_size, n_train=2000, n_val=200, seed=42):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training, validation, and test sets for the openset dataset.
    """

    # TODO: revisit the idea of training transformations for the open set discriminator training

    # set up the dataset twice but with different transformations
    train_dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "FungiCLEF2023_val_metadata_PRODUCTION.csv",
        img_dir=DATA_DIR / "DF21",
        keep_only={-1},
        transform=(get_train_transform(image_size, pretrained))
    )
    val_test_dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "FungiCLEF2023_val_metadata_PRODUCTION.csv",
        img_dir=DATA_DIR / "DF21",
        keep_only={-1},
        transform=(get_valid_transform(image_size, pretrained))  # this is the difference
    )

    indices = list(range(train_dataset.target.shape[0]))
    test_indices, train_indices = train_test_split(indices, test_size=n_train, random_state=seed)
    test_indices, val_indices = train_test_split(test_indices, test_size=n_val, random_state=seed)

    train_dataset = Subset(train_dataset, indices=train_indices)
    val_dataset = Subset(val_test_dataset, indices=val_indices)
    test_dataset = Subset(val_test_dataset, indices=test_indices)
    train_dataset.target = np.array([-1] * len(train_dataset))
    val_dataset.target = np.array([-1] * len(val_dataset))
    test_dataset.target = np.array([-1] * len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def get_closedset_test_dataset(pretrained, image_size):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training, validation, and test sets for the openset dataset.
    """

    # drop the unknown label
    val_test_dataset = CustomImageDataset(
        label_file_path=METADATA_DIR / "FungiCLEF2023_val_metadata_PRODUCTION.csv",
        img_dir=DATA_DIR / "DF21",
        exclude={-1},
        transform=(get_valid_transform(image_size, pretrained))
    )

    return val_test_dataset


def get_dataloader_combine_and_balance_datasets(dataset_1, dataset_2, batch_size, num_workers=16,
                                                timeout=WORKER_TIMEOUT_SECS,
                                                persistent_workers=True, unknowns=False):
    """
    https://pytorch.org/docs/stable/data.html
    https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
    """
    dataset = ConcatDataset([dataset_1, dataset_2])
    target = np.array(list(dataset_1.target) + list(dataset_2.target))
    target = target + 1 if unknowns else target  # -1 is now 0 to satisfy logic below
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight_per_class = 1. / class_sample_count
    weight_per_class[0] = 0.5 if unknowns else weight_per_class[0]
    weight_per_sample = np.array([weight_per_class[class_idx] for class_idx in target])
    weight_per_sample = torch.from_numpy(weight_per_sample)
    weight_per_sample = weight_per_sample.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_per_sample, len(weight_per_sample))
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers,
                            timeout=timeout,
                            persistent_workers=persistent_workers,
                            )
    return dataloader

# the imagefolder dataset will sort by class then filename, so for data splitting purposes, we can sort the
# image_path per class and get back the order in the dataset
# full_dataset = ImageFolder(root=DATA_FROM_FOLDER_DIR)
# def get_openset_datasets(pretrained, image_size, n_train=2000, n_val=200, seed=42):
#     """
#     Function to prepare the Datasets.
#     :param pretrained: Boolean, True or False.
#     Returns the training, validation, and test sets for the openset dataset.
#     """
#
#     # TODO: revisit the idea of training transformations for the open set discriminator training
#
#     # set up the dataset twice but with different transformations
#     train_dataset = CustomImageDataset(
#         label_file_path=METADATA_DIR / "FungiCLEF2023_val_metadata_PRODUCTION.csv",
#         img_dir=DATA_DIR / "DF21",
#         keep_only={-1},
#         transform=(get_train_transform(image_size, pretrained))
#     )
#     val_test_dataset = CustomImageDataset(
#         label_file_path=METADATA_DIR / "FungiCLEF2023_val_metadata_PRODUCTION.csv",
#         img_dir=DATA_DIR / "DF21",
#         keep_only={-1},
#         transform=(get_valid_transform(image_size, pretrained))  # this is the difference
#     )
#
#     indices = list(range(train_dataset.target.shape[0]))
#     test_indices, train_indices = train_test_split(indices, test_size=n_train, random_state=seed)
#     test_indices, val_indices = train_test_split(test_indices, test_size=n_val, random_state=seed)
#
#     train_dataset = Subset(train_dataset, indices=train_indices)
#     val_dataset = Subset(val_test_dataset, indices=val_indices)
#     test_dataset = Subset(val_test_dataset, indices=test_indices)
#     train_dataset.target = np.array([-1] * len(train_dataset))
#     val_dataset.target = np.array([-1] * len(val_dataset))
#     test_dataset.target = np.array([-1] * len(test_dataset))
#
#     return train_dataset, val_dataset, test_dataset
