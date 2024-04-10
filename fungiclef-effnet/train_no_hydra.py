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
from torch_lr_finder import LRFinder
import time
from tqdm.auto import tqdm

from closedset_model import build_model, unfreeze_model
from datasets import get_datasets, get_data_loaders
from utils import save_model, save_plots, checkpoint_model
from losses import SeesawLoss

CHECKPOINT_DIR = Path('__file__').parent.absolute() / "model_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# # construct the argument parser
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '-e', '--epochs', type=int, default=25,
#     help='Number of epochs to train our network for'
# )
# parser.add_argument(
#     '-pt', '--pretrained', action='store_true',
#     help='Whether to use pretrained weights or not'
# )
# parser.add_argument(
#     '-lr', '--learning-rate', type=float,
#     dest='learning_rate', default=1e-5,
#     help='Learning rate for training the model'
# )
# args = vars(parser.parse_args())


# def add_weight_decay(
#         model,
#         weight_decay=1e-5,
#         skip_list=(nn.InstanceNorm1d, nn.InstanceNorm2d, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
#     """
#     https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/12
#     Using .modules() with an isinstance() check
#     returns the model parameters for use with weight_decay
#     """
#     decay = []
#     no_decay = []
#     for module in model.modules():
#         params = [p for p in module.parameters() if p.requires_grad]
#         if isinstance(module, skip_list):
#             no_decay.extend(params)
#         else:
#             decay.extend(params)
#     return [
#         {'params': no_decay, 'weight_decay': 0.},
#         {'params': decay, 'weight_decay': weight_decay}]


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """
    alternative implementation
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/12
    returns the model parameters for use with weight_decay
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


# alternative implementation
# https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/7
# TODO: revise and try
@torch.no_grad()
def get_wd_params(model: nn.Module):
    decay = list()
    no_decay = list()
    for name, param in model.named_parameters():
        print('checking {}'.format(name))
        if hasattr(param, 'requires_grad') and not param.requires_grad:
            continue
        if 'weight' in name and 'norm' not in name and 'bn' not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    return decay, no_decay


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
    # lr = args['learning_rate']
    # epochs = args['epochs']
    # pretrained = args['pretrained']
    epochs = 200
    lr = 8e-4
    pretrained = True
    early_stop_thresh = 10
    loss_function = "seesaw"
    batch_size = 128
    num_dataloader_workers = 16
    image_resize = 224
    validation_frac = 0.1
    max_norm = 1.0
    undersample = False
    oversample = False
    equal_undersampled_val = True
    oversample_prop = 0.1
    dropout_rate = 0.5
    weight_decay = 1e-5
    balanced_sampler = False
    use_lr_finder = False
    if balanced_sampler and (oversample or undersample):
        raise ValueError("cannot use balanced sampler with oversample or undersample")
    fine_tune_after_n_epochs = 4
    skip_frozen_epochs_load_failed_model = True
    model_file_path = str(
        CHECKPOINT_DIR /
        f"best_model_{loss_function}_batch_{batch_size}_lr_{lr: .6f}_dropout_{dropout_rate: .2f}_weight_decay_{weight_decay: .6f}_unfreeze_epoch_{fine_tune_after_n_epochs}_over_{oversample}_over_prop_{oversample_prop}_under_{undersample}_balanced_sampler_{balanced_sampler}_equal_undersampled_val_{equal_undersampled_val}_trivialaug.pth")
    resume_from_checkpoint = model_file_path if Path(model_file_path).exists() else None

    torch.autograd.set_detect_anomaly(True)

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets(pretrained, image_resize, validation_frac,
                                                                 oversample=oversample, undersample=undersample,
                                                                 oversample_prop=oversample_prop)
    n_classes = len(dataset_classes)
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, batch_size, num_dataloader_workers,
                                                  balanced_sampler=balanced_sampler)
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    model = build_model(
        pretrained=pretrained,
        fine_tune=not fine_tune_after_n_epochs,
        num_classes=n_classes,
        dropout_rate=dropout_rate,
    ).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    parameters = add_weight_decay(model, weight_decay=weight_decay)
    optimizer = optim.AdamW(parameters, lr=lr)

    if resume_from_checkpoint:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        checkpoint = torch.load(resume_from_checkpoint)
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        parameters = add_weight_decay(model, weight_decay=weight_decay)
        optimizer = optim.AdamW(parameters, lr=lr)
        model.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"unable to load optimizer state due to {e}")
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

    if use_lr_finder:
        optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
        lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
        lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
        lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()  # to reset the model and optimizer to their initial state

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    best_validation_loss = float("inf")
    best_epoch = 0
    # Start the training.
    unfrozen = fine_tune_after_n_epochs == 0
    for epoch in range(epochs):

        if (not unfrozen and (skip_frozen_epochs_load_failed_model or
                              epoch + 1 > fine_tune_after_n_epochs)):
            model = unfreeze_model(model)
            print("all layers unfrozen")
            parameters = add_weight_decay(model, weight_decay=weight_decay)
            optimizer = optim.AdamW(parameters, lr=lr)
            unfrozen = True

        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        recreate_loader = True
        while recreate_loader:
            try:
                train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                          optimizer, criterion, loss_function, max_norm)
                recreate_loader = False
            except Exception as e:
                print("issue with trining")
                print(e)
                print("recreating data loaders and trying again")
                train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, batch_size,
                                                              num_dataloader_workers,
                                                              balanced_sampler=balanced_sampler)
                train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                          optimizer, criterion, loss_function, max_norm)

        recreate_loader = True
        while recreate_loader:
            try:
                valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                             criterion, loss_function)
                recreate_loader = False
            except Exception as e:
                print("issue with validation")
                print(e)
                print("recreating data loaders and trying again")
                train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, batch_size,
                                                              num_dataloader_workers,
                                                              balanced_sampler=balanced_sampler)
                valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                             criterion, loss_function)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss: .3f}, training acc: {train_epoch_acc: .3f}")
        print(f"Validation loss: {valid_epoch_loss: .3f}, validation acc: {valid_epoch_acc: .3f}")
        # TODO: make a helper that can use loss or accuracy and minimizes or maximizes intelligently
        if valid_epoch_loss < best_validation_loss:
            best_validation_loss = valid_epoch_loss
            best_epoch = epoch
            print("updating best model")
            checkpoint_model(epoch + 1, model, optimizer, criterion, valid_epoch_loss, model_file_path)
            print(">> successfully updated best model <<")
        elif epoch - best_epoch > early_stop_thresh:
            print("Early stopped training at epoch %d" % epoch)
            break  # terminate the training loop
        else:
            print(f"model did not improve from best epoch {best_epoch + 1} with loss {best_validation_loss: .3f}")
        print('-' * 50)
        time.sleep(5)

    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, pretrained)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained)
    print('TRAINING COMPLETE')
