"""
modified from https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
"""

import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.style.use('ggplot')
OUTPUT_DIR = Path('__file__').parent.absolute() / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_model(epochs, model, optimizer, criterion, pretrained):
    """
    Function to save the trained model to disk.
    """
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, str(OUTPUT_DIR / f"model_pretrained_{pretrained}.pth"))


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


def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained):
    """
    Function to save the loss and accuracy plots to disk.
    """
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
    plt.savefig(str(OUTPUT_DIR / f"accuracy_pretrained_{pretrained}.png"))

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
    plt.savefig(str(OUTPUT_DIR / f"loss_pretrained_{pretrained}.png"))
