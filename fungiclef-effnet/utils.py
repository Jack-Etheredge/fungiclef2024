"""
modified from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
"""

import datetime
import random
import shutil
from pathlib import Path

import torch
from matplotlib import pyplot as plt
import matplotlib
from torch import nn as nn
from torch.utils.data import Subset

from openset_recognition_models import Generator, Discriminator

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


def checkpoint_model(epochs, model, dropout_rate, optimizer, criterion, validation_loss, file_path):
    """
    Function to save the trained model to disk.
    """
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'dropout_rate': dropout_rate,
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


def save_loss_plots(generator_losses, discriminator_losses, model_name, save_dir):
    # ## drawing the error curves
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(generator_losses, label="Gen")
    plt.plot(discriminator_losses, label="Dis")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{save_dir}/learningCurves_{model_name}.png', bbox_inches='tight', transparent=True)
    # plt.show()


def save_discriminator_loss_plot(discriminator_losses, model_name, save_dir):
    # ## drawing the error curves
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(discriminator_losses, label="Dis")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{save_dir}/learningCurves_{model_name}.png', bbox_inches='tight', transparent=True)
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


def copy_config(script_name, experiment_id):
    now_dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = Path("model_checkpoints") / experiment_id
    config_from = Path("conf") / "config.yaml"
    config_to = experiment_dir / f"config_{script_name}_{now_dt}.yaml"
    shutil.copyfile(config_from, config_to)


def get_model_features(image, model, device):
    """
    get features (intermediate representation before classification) from the model.
    handles both the case of metadata being present or not.
    """
    if isinstance(image, list):
        image, metadata = image
        metadata = metadata.to(device)
        image = image.to(device)
        outputs = model.forward_head(model.forward_features(image, metadata), pre_logits=True)
    else:
        image = image.to(device)
        outputs = model.forward_head(model.forward_features(image), pre_logits=True)

    return outputs


def get_model_preds(image, model, device):
    """
    get model preds. handles both the case of metadata being present or not.
    """
    if isinstance(image, list):
        image, metadata = image
        metadata = metadata.to(device)
        image = image.to(device)
        outputs = model(image, metadata)
    else:
        image = image.to(device)
        outputs = model(image)
    return outputs
