"""
modified from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
from torch import nn as nn
from torch.utils.data import Dataset, Subset

from models import Generator, Discriminator

matplotlib.style.use('ggplot')
OUTPUT_DIR = Path('__file__').parent.absolute() / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_model(epochs, model, optimizer, criterion, pretrained, model_path):
    """
    Function to save the trained model to disk.
    """

    if not model_path:
        model_path = str(OUTPUT_DIR / f"model_pretrained_{pretrained}.pth")

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, model_path)


def checkpoint_model(epochs, model, optimizer, criterion, validation_loss, file_path):
    """
    Function to save the trained model to disk.
    """
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'validation_loss': validation_loss,
    }, file_path)


def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained, accuracy_plot_path, loss_plot_path):
    """
    Function to save the loss and accuracy plots to disk.
    """

    if not accuracy_plot_path:
        accuracy_plot_path = str(OUTPUT_DIR / f"accuracy_pretrained_{pretrained}.png")

    if not loss_plot_path:
        loss_plot_path = str(OUTPUT_DIR / f"loss_pretrained_{pretrained}.png")

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(accuracy_plot_path)

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plot_path)


class CustomImageDataset(Dataset):
    def __init__(self, img_dir: str, transform=None, target_transform=None):
        valid_extensions = {".jpg", ".jpeg", ".png"}
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = np.array([1.])
        self.img_paths = [img for img in Path(img_dir).iterdir() if img.suffix.lower() in valid_extensions]
        self.n_img = len(self.img_paths)
        self.target = np.array([1.] * self.n_img)

    def __len__(self):
        return self.n_img

    def __getitem__(self, idx):
        """If an image can't be read, print the error and return None"""
        try:
            img_path = self.img_paths[idx]
            image = Image.open(img_path).convert('RGB')  # hopefully this handles greyscale cases
            label = 1.
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


def get_train_val_datasets(dataset, max_total_examples=1000, val_frac=0.1):
    if len(dataset) > max_total_examples:
        subset_indices = torch.randperm(len(dataset))[:max_total_examples]
        dataset = Subset(dataset, subset_indices)
    eval_size = int(val_frac * len(dataset))
    train_indices = torch.randperm(len(dataset))[eval_size:]
    val_indices = torch.randperm(len(dataset))[:eval_size]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_loss_plots(generator_losses, discriminator_losses, model_name):
    # ## drawing the error curves
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(generator_losses, label="Gen")
    plt.plot(discriminator_losses, label="Dis")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'learningCurves_{model_name}.png', bbox_inches='tight', transparent=True)
    # plt.show()


def set_seed(seed=1337):
    print(f"Random Seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)


def build_models(nz, ngf, nc, ndf, device, n_gpu):
    """
    create and initialize the networks
    handle multi-GPU training if applicable
    """
    # build the networks and move them to GPU if applicable
    net_g = Generator(nz=nz, hidden_dim=ngf, nc=nc).to(device)
    net_d = Discriminator(nc=nc, hidden_dim=ndf).to(device)
    # # Handle multi-gpu if desired
    # if ('cuda' in device) and (n_gpu > 1):
    #     net_d = nn.DataParallel(net_d, list(range(n_gpu)))
    #     net_g = nn.DataParallel(net_g, list(range(n_gpu)))
    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    net_d.apply(weights_init)
    net_g.apply(weights_init)
    return net_g, net_d
