"""
modified from these sources:
https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/
"""
import json
from datetime import datetime
from pathlib import Path

from focal_loss.focal_loss import FocalLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_lr_finder import LRFinder
import time
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from paths import CHECKPOINT_DIR
from closedset_model import build_model, unfreeze_model
from datasets import get_datasets, get_data_loaders
from evaluate import evaluate_experiment
from utils import save_plots, checkpoint_model
from losses import SeesawLoss, SupConLoss


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
def train(model, trainloader, optimizer, criterion, loss_function_id, max_norm, device='cpu'):
    model.train()
    print(f'Training with device {device}')
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
    return epoch_loss.cpu().numpy(), epoch_acc


# Validation function.
@torch.no_grad()
def validate(model, testloader, criterion, loss_function_id, device='cpu', scheduler=None):
    model.eval()
    print(f'Validation with device {device}')
    valid_running_loss = 0.0
    valid_running_correct = 0
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
    if scheduler:
        scheduler.step(epoch_loss)
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss.cpu().numpy(), epoch_acc


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    epochs = cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]
    pretrained = cfg["train"]["pretrained"]
    early_stop_thresh = cfg["train"]["early_stop_thresh"]
    loss_function = cfg["train"]["loss_function"]
    batch_size = cfg["train"]["batch_size"]
    num_dataloader_workers = cfg["train"]["num_dataloader_workers"]
    image_resize = cfg["train"]["image_resize"]
    validation_frac = cfg["train"]["validation_frac"]
    max_norm = cfg["train"]["max_norm"]
    undersample = cfg["train"]["undersample"]
    oversample = cfg["train"]["oversample"]
    equal_undersampled_val = cfg["train"]["equal_undersampled_val"]
    oversample_prop = cfg["train"]["oversample_prop"]
    dropout_rate = cfg["train"]["dropout_rate"]
    weight_decay = cfg["train"]["weight_decay"]
    balanced_sampler = cfg["train"]["balanced_sampler"]
    use_lr_finder = cfg["train"]["use_lr_finder"]
    fine_tune_after_n_epochs = cfg["train"]["fine_tune_after_n_epochs"]
    skip_frozen_epochs_load_failed_model = cfg["train"]["skip_frozen_epochs_load_failed_model"]
    lr_scheduler = cfg["train"]["lr_scheduler"]
    lr_scheduler_patience = cfg["train"]["lr_scheduler_patience"]

    for k, v in cfg["train"].items():
        if v == "None" or v == "null":
            url = "https://stackoverflow.com/questions/76567692/hydra-how-to-express-none-in-config-files"
            raise ValueError(f"`{k}` set to 'None' or 'null'. Use `null` for None values in hydra; see {url}")

    experiment_id = cfg["train"]["experiment_id"]
    if experiment_id is None:
        experiment_id = str(datetime.now())
        cfg["train"]["experiment_id"] = experiment_id

    if balanced_sampler and (oversample or undersample):
        raise ValueError("cannot use balanced sampler with oversample or undersample")

    # use experiment_id to save model checkpoint, graphs, predictions, performance metrics, etc
    experiment_dir = CHECKPOINT_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_file_path = str(experiment_dir / f"model.pth")
    accuracy_plot_path = str(experiment_dir / "accuracy_plot.png")
    loss_plot_path = str(experiment_dir / "loss_plot.png")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(str(experiment_dir / "experiment_config.json"), "w") as f:
        json.dump(config_dict, f)

    torch.autograd.set_detect_anomaly(True)

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets(pretrained, image_resize, validation_frac,
                                                                 oversample=oversample, undersample=undersample,
                                                                 oversample_prop=oversample_prop,
                                                                 equal_undersampled_val=equal_undersampled_val)
    n_classes = len(dataset_classes)
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
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

    if lr_scheduler == "reducelronplateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_scheduler_patience)
    else:
        scheduler = None
        print(f"not using lr scheduler, config set to {lr_scheduler}")

    # TODO: refine/replace this logic
    resume_from_checkpoint = model_file_path if Path(model_file_path).exists() else None
    if resume_from_checkpoint:
        print(f"resuming from checkpoint: {model_file_path}")
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
        # TODO: use the epoch and loss from the checkpoint
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.to(device)
    else:
        print(f"training new model: {experiment_id}")

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
                                                          optimizer, criterion, loss_function, max_norm, device)
                recreate_loader = False
            except Exception as e:
                print("issue with training")
                print(e)
                print("recreating data loaders and trying again")
                train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, batch_size,
                                                              num_dataloader_workers,
                                                              balanced_sampler=balanced_sampler)

        recreate_loader = True
        while recreate_loader:
            try:
                valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                             criterion, loss_function, device, scheduler)
                recreate_loader = False
            except Exception as e:
                print("issue with validation")
                print(e)
                print("recreating data loaders and trying again")
                train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, batch_size,
                                                              num_dataloader_workers,
                                                              balanced_sampler=balanced_sampler)

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

    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained, accuracy_plot_path=accuracy_plot_path,
               loss_plot_path=loss_plot_path)
    print('TRAINING COMPLETE')

    print('evaluating')
    evaluate_experiment(experiment_id=experiment_id)
    print('EVALUATION COMPLETE')


if __name__ == '__main__':
    train_model()
