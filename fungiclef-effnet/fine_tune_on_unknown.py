"""
Fine-tuning on unknown:
- given a model trained on the closed set training set with early stopping on a validation portion
    - use the validation portion of the training set for closed labels
    - split the open set into train, val, test

modified from these sources:
https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/
https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/
"""
import json
from pathlib import Path

from focal_loss.focal_loss import FocalLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_lr_finder import LRFinder
import time
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from paths import CHECKPOINT_DIR
from closedset_model import build_model
from datasets import get_datasets, get_openset_datasets, get_dataloader_combine_and_balance_datasets
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
def train(model, data_loader, optimizer, criterion, open_criterion, loss_function_id, max_norm, device='cpu'):
    model.train()
    print(f'Training with device {device}')
    running_loss = 0.0
    running_correct = 0
    m = nn.Softmax(dim=-1)
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        open_outputs, closed_outputs, _, closed_labels, n_open, n_closed = split_open_closed(outputs, labels)
        # Calculate the loss.
        if n_open > 0:
            if loss_function_id == "focal":
                loss = criterion(m(closed_outputs), closed_labels)
            else:
                loss = criterion(closed_outputs, closed_labels)
        else:
            loss = torch.tensor(0.)
        if n_closed > 0:
            open_loss = open_criterion(open_outputs)
        else:
            open_loss = torch.tensor(0.)

        total_samples = (n_closed + n_open)
        assert total_samples == image.shape[0]
        closed_proportion = n_closed / total_samples
        open_proportion = n_open / total_samples

        total_loss = loss.item() * closed_proportion + open_loss.item() * open_proportion
        running_loss += total_loss

        # Calculate the accuracy.
        _, preds = torch.max(closed_outputs.data, 1)
        running_correct += (preds == closed_labels).sum().item()
        # Backpropagation
        loss.backward()
        # gradient norm clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # this is a tunable hparam
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = running_loss / i + 1
    epoch_acc = 100. * (running_correct / len(data_loader.dataset))
    return epoch_loss.cpu().numpy(), epoch_acc


# Validation function.
@torch.no_grad()
def validate(model, data_loader, criterion, open_criterion, loss_function_id, device='cpu', scheduler=None):
    model.eval()
    print(f'Validation with device {device}')
    running_loss = 0.0
    running_correct = 0
    m = nn.Softmax(dim=-1)
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)

        # Forward pass.
        outputs = model(image)
        open_outputs, closed_outputs, _, closed_labels, n_open, n_closed = split_open_closed(outputs, labels)
        # Calculate the loss.
        if n_open > 0:
            if loss_function_id == "focal":
                loss = criterion(m(closed_outputs), closed_labels)
            else:
                loss = criterion(closed_outputs, closed_labels)
        else:
            loss = torch.tensor(0.)
        if n_closed > 0:
            open_loss = open_criterion(open_outputs)
        else:
            open_loss = torch.tensor(0.)

        total_samples = (n_closed + n_open)
        assert total_samples == image.shape[0]
        closed_proportion = n_closed / total_samples
        open_proportion = n_open / total_samples

        total_loss = loss.item() * closed_proportion + open_loss.item() * open_proportion
        running_loss += total_loss
        # Calculate the accuracy.
        _, preds = torch.max(closed_outputs.data, 1)
        running_correct += (preds == closed_labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = running_loss / i + 1
    if scheduler:
        scheduler.step(epoch_loss)
    epoch_acc = 100. * (running_correct / len(data_loader.dataset))
    return epoch_loss.cpu().numpy(), epoch_acc


def split_open_closed(outputs, labels):
    """
    get the open and closed labels and outputs from the labels and outputs to calculate losses separately
    """
    closed_indices = labels >= 0
    open_indices = labels == -1
    n_closed = closed_indices.sum()
    n_open = open_indices.sum()
    closed_labels = labels[closed_indices]
    open_labels = labels[open_indices]
    closed_outputs = outputs[closed_indices]
    open_outputs = outputs[open_indices]
    return open_outputs, closed_outputs, open_labels, closed_labels, n_open, n_closed


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    epochs = cfg["unknown-fine-tune"]["epochs"]
    lr = cfg["unknown-fine-tune"]["lr"]
    batch_size = cfg["unknown-fine-tune"]["batch_size"]
    early_stop_thresh = cfg["unknown-fine-tune"]["early_stop_thresh"]
    loss_function = cfg["unknown-fine-tune"]["loss_function"]
    use_lr_finder = cfg["unknown-fine-tune"]["use_lr_finder"]
    lr_scheduler = cfg["unknown-fine-tune"]["lr_scheduler"]
    lr_scheduler_patience = cfg["unknown-fine-tune"]["lr_scheduler_patience"]

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
    n_classes = cfg["train"]["n_classes"]

    image_size = cfg["open-set-recognition"]["image_size"]
    openset_n_train = cfg["open-set-recognition"]["openset_n_train"]
    openset_n_val = cfg["open-set-recognition"]["openset_n_val"]
    pretrained = cfg["open-set-recognition"]["pretrained"]

    for k, v in cfg["train"].items():
        if v == "None" or v == "null":
            url = "https://stackoverflow.com/questions/76567692/hydra-how-to-express-none-in-config-files"
            raise ValueError(f"'{k}' set to 'None' or 'null'."
                             f"Use null without quotes for None values in hydra: see {url}")

    experiment_id = cfg["unknown-fine-tune"]["experiment_id"]
    if experiment_id is None:
        raise ValueError(("fine tuning requires a trained model; "
                          "specify an experiment_id model directory with a valid model.pth checkpoint"))

    if balanced_sampler and (oversample or undersample):
        raise ValueError("cannot use balanced sampler with oversample or undersample")

    # use experiment_id to save model checkpoint, graphs, predictions, performance metrics, etc
    experiment_dir = CHECKPOINT_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_file_path = str(experiment_dir / f"model.pth")
    if not Path(model_file_path).exists():
        raise ValueError(f"Invalid path to model: {model_file_path}")
    updated_model_file_path = str(experiment_dir / f"model_fine_tuned_unknown.pth")
    accuracy_plot_path = str(experiment_dir / "unknown_fine_tuning_accuracy_plot.png")
    loss_plot_path = str(experiment_dir / "unknown_fine_tuning_loss_plot.png")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(str(experiment_dir / "unknown_fine_tuning_experiment_config.json"), "w") as f:
        json.dump(config_dict, f)

    torch.autograd.set_detect_anomaly(True)

    # Load the training and validation datasets.
    closed_dataset_train, closed_dataset_val, _ = get_datasets(pretrained, image_resize,
                                                               validation_frac,
                                                               oversample=oversample,
                                                               undersample=undersample,
                                                               oversample_prop=oversample_prop,
                                                               equal_undersampled_val=equal_undersampled_val)
    # closed_train_loader, closed_val_loader = get_data_loaders(closed_dataset_train, closed_dataset_val, batch_size,
    #                                                           num_dataloader_workers,
    #                                                           balanced_sampler=balanced_sampler)
    open_dataset_train, open_dataset_val, _ = get_openset_datasets(pretrained=pretrained, image_size=image_size,
                                                                   n_train=openset_n_train, n_val=openset_n_val)
    # open_train_loader = torch.utils.data.DataLoader(open_dataset_train, batch_size=batch_size,
    #                                                 shuffle=False, num_workers=n_workers, collate_fn=collate_fn,
    #                                                 timeout=worker_timeout_s)
    # open_val_loader = torch.utils.data.DataLoader(open_dataset_train, batch_size=batch_size,
    #                                               shuffle=False, num_workers=n_workers, collate_fn=collate_fn,
    #                                               timeout=worker_timeout_s)
    # closed_train_loader, closed_val_loader = get_data_loaders(closed_dataset_train, closed_dataset_val, batch_size,
    #                                                           num_dataloader_workers,
    #                                                           balanced_sampler=balanced_sampler)
    print("[[train]] combining dataloaders and balancing classes")
    train_loader = get_dataloader_combine_and_balance_datasets(closed_dataset_train, open_dataset_train,
                                                               batch_size=batch_size, unknowns=True)
    print("[[val]] combining dataloaders and balancing classes")
    val_loader = get_dataloader_combine_and_balance_datasets(closed_dataset_val, open_dataset_val,
                                                             batch_size=batch_size, unknowns=True)

    print(f"[INFO]: Number of closed set training images: {len(closed_dataset_train)}")
    print(f"[INFO]: Number of closed set validation images: {len(closed_dataset_val)}")
    print(f"[INFO]: Number of open set training images: {len(open_dataset_train)}")
    print(f"[INFO]: Number of open set validation images: {len(open_dataset_val)}")
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    model = build_model(
        pretrained=False,  # model weights will be loaded
        fine_tune=True,  # we want to fine-tune all the weights
        num_classes=n_classes,
        dropout_rate=dropout_rate,
    ).to(device)

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
            optimizer.param_groups[0]["lr"] = lr
            print("loaded optimizer state but replaced lr")
        except Exception as e:
            print(f"unable to load optimizer state due to {e}")
        model.to(device)
    else:
        parameters = add_weight_decay(model, weight_decay=weight_decay)
        optimizer = optim.AdamW(parameters, lr=lr)
        print(f"training new model: {experiment_id}")

    # make scheduler after the optimizer has been created/modified
    if lr_scheduler == "reducelronplateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_scheduler_patience)
    else:
        scheduler = None
        print(f"not using lr scheduler, config set to {lr_scheduler}")

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

    open_criterion = UnknownLoss()

    if use_lr_finder:
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
        lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()  # to reset the model and optimizer to their initial state

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    best_validation_loss = float("inf")
    best_epoch = 0
    # Start the training.
    for epoch in range(epochs):

        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"current learning rate: {curr_lr:.0e}")
        recreate_loader = True
        while recreate_loader:
            try:
                train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                          optimizer, criterion, open_criterion, loss_function, max_norm,
                                                          device)
                recreate_loader = False
            except Exception as e:
                print("issue with training")
                print(e)
                print("recreating training data loader and trying again")
                train_loader = get_dataloader_combine_and_balance_datasets(closed_dataset_train, open_dataset_train,
                                                                           batch_size=batch_size, unknowns=True)

        recreate_loader = True
        while recreate_loader:
            try:
                valid_epoch_loss, valid_epoch_acc = validate(model, val_loader,
                                                             criterion, open_criterion, loss_function, device,
                                                             scheduler)
                recreate_loader = False
            except Exception as e:
                print("issue with validation")
                print(e)
                print("recreating validation data loader and trying again")
                val_loader = get_dataloader_combine_and_balance_datasets(closed_dataset_val, open_dataset_val,
                                                                         batch_size=batch_size, unknowns=True)

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
            checkpoint_model(epoch + 1, model, optimizer, criterion, valid_epoch_loss, updated_model_file_path)
            print(">> successfully updated best model <<")
        elif epoch - best_epoch > early_stop_thresh:
            print(f"Early stopped training at epoch {epoch + 1}")
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


class UnknownLoss(torch.nn.Module):

    def forward(self, model_outs):
        # modified from https://github.com/RenHuan1999/FungiCLEF2023-UstcAIGroup/blob/main/models/custom_loss.py
        labels_nov = torch.ones_like(model_outs) / model_outs.shape[1]
        loss_cls_classes_nov = - (labels_nov * log_softmax(model_outs, dim=1)).sum(dim=-1)
        return loss_cls_classes_nov.mean()


if __name__ == '__main__':
    train_model()
