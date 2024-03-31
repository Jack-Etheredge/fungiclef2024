"""
modified from these sources:
https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/
"""
from pathlib import Path

from focal_loss.focal_loss import FocalLoss
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm

from model import build_model, unfreeze_model
from datasets import get_datasets, get_data_loaders
from utils import save_model, save_plots, checkpoint_model
from losses import SeesawLoss

CHECKPOINT_DIR = Path('__file__').parent.absolute() / "model_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=25,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-pt', '--pretrained', action='store_true',
    help='Whether to use pretrained weights or not'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=1e-4,
    help='Learning rate for training the model'
)
args = vars(parser.parse_args())


# Training function.
def train(model, trainloader, optimizer, criterion, loss_function_id, max_norm):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    m = nn.Softmax(dim=-1)
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        if loss_function_id == "focal":
            loss = criterion(m(outputs), labels)
        else:
            loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # gradient norm clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # this is a tunable hparam
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / i + 1
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion, loss_function_id):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    with torch.no_grad():
        m = nn.Softmax(dim=-1)
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            if loss_function_id == "focal":
                loss = criterion(m(outputs), labels)
            else:
                loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / i + 1
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


if __name__ == '__main__':

    # Learning_parameters.
    lr = args['learning_rate']
    epochs = args['epochs']
    pretrained = args['pretrained']
    pretrained = True
    early_stop_thresh = 10
    loss_function = "seesaw"
    batch_size = 32
    num_dataloader_workers = 8
    image_resize = 224
    validation_frac = 0.1
    max_norm = 1.0
    fine_tune_after_n_epochs = 4
    model_file_path = str(
        CHECKPOINT_DIR /
        f"best_model_{loss_function}_batch_{32}_lr_{lr: .3f}_unfreeze_epoch_{fine_tune_after_n_epochs}_trivialaug.pth")
    resume_from_checkpoint = model_file_path if Path(model_file_path).exists() else None

    torch.autograd.set_detect_anomaly(True)

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets(args['pretrained'], image_resize, validation_frac)
    n_classes = len(dataset_classes)
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, batch_size, num_dataloader_workers)
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    model = build_model(
        pretrained=pretrained,
        fine_tune=not fine_tune_after_n_epochs,
        num_classes=n_classes
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if resume_from_checkpoint:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        checkpoint = torch.load(resume_from_checkpoint)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.to(device)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    if loss_function == "seesaw":
        criterion = SeesawLoss(num_classes=n_classes, device=device)
    elif loss_function == "focal":
        criterion = FocalLoss(gamma=0.7)
    elif loss_function == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    best_validation_loss = float("inf")
    best_epoch = 0
    # Start the training.
    unfrozen = fine_tune_after_n_epochs == 0
    for epoch in range(epochs):

        if not unfrozen and epoch + 1 > fine_tune_after_n_epochs:
            model = unfreeze_model(model)
            print("all layers unfrozen")
            unfrozen = True

        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                  optimizer, criterion, loss_function, max_norm)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                     criterion, loss_function)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss: .3f}, training acc: {train_epoch_acc: .3f}")
        print(f"Validation loss: {valid_epoch_loss: .3f}, validation acc: {valid_epoch_acc: .3f}")
        print('-' * 50)
        # TODO: make a helper that can use loss or accuracy and minimizes or maximizes intelligently
        if valid_epoch_loss < best_validation_loss:
            best_validation_loss = valid_epoch_loss
            best_epoch = epoch
            print("updating best model")
            checkpoint_model(epoch + 1, model, optimizer, criterion, valid_epoch_loss, model_file_path)
            print("successfully updated best model")
        elif epoch - best_epoch > early_stop_thresh:
            print("Early stopped training at epoch %d" % epoch)
            break  # terminate the training loop
        time.sleep(5)

    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, pretrained)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained)
    print('TRAINING COMPLETE')
