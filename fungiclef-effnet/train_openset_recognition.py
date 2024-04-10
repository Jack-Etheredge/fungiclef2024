"""
https://arxiv.org/pdf/2104.02939v3.pdf

The MLP discriminator in OpenGANfea takes a
D-dimensional feature as the input. Its architecture has a set of fully-connected layers (fc marked
with input-dimension and output-dimension), Batch
Normalization layers (BN) and LeakyReLU layers (hyper-parameter as 0.2): fc (D→64*8),
BN, LeakyReLU, fc (64*8→64*4),
BN, LeakyReLU, fc (64*4→64*2),
BN, LeakyReLU, fc (64*2→64*1), BN,
LeakyReLU, fc (64*1→1), Sigmoid.

• The MLP generator synthesizes a D-dimensional
feature given a 64-dimensional random vector: fc (64→64*8), BN, LeakyReLU,
fc (64*8→64*4), BN, LeakyReLU,
fc (64*4→64*2), BN, LeakyReLU,
fc (64*2→64*4), BN, LeakyReLU, fc
(64*4→D), Tanh.
"""

import os
import copy
import sys
from pathlib import Path

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf

import warnings  # ignore warnings

from utils import set_seed, build_models, save_discriminator_loss_plot
from paths import EMBEDDINGS_DIR


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class FeatDataset(Dataset):
    """
    a helper function to read cached off-the-shelf features per closed images
    """

    def __init__(self, data, label):
        self.data = data
        self.current_set_len = data.shape[0]
        self.label = label
        self.target = [label] * self.current_set_len

    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        curdata = self.data[idx]
        return curdata, self.label


def get_cached_data(h5_path):
    """
    load cached set of features from an h5py .h5 file to numpy then torch
    """
    with h5py.File(h5_path, "r") as hf:
        cached_data = hf["data"][:]
    whole_feat_vec = torch.from_numpy(cached_data)
    # whole_feat_vec.unsqueeze_(-1).unsqueeze_(-1)  # not needed since using MLP instead of CNN
    print(whole_feat_vec.shape)
    return whole_feat_vec


def save_model_state(discriminator, epoch, save_dir):
    """
    save a checkpoint of the model weights
    """

    cur_model_wts = copy.deepcopy(discriminator.state_dict())
    path_to_save_model_state = os.path.join(save_dir, f'epoch-{epoch + 1}-discriminator.pth')
    torch.save(cur_model_wts, path_to_save_model_state)


def train(discriminator, data, criterion, optimizer, device):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    optimizer.zero_grad()
    # Format batch
    embeddings, labels = data
    real_cpu = embeddings.to(device)
    label = labels.float().to(device)
    output = discriminator(real_cpu).view(-1)  # Forward pass real batch through D
    dis_loss_real = criterion(output, label)  # Calculate loss on all-real batch
    # Calculate gradients for D in backward pass
    dis_loss_real.backward()
    D_x = output.mean().item()
    dis_loss = dis_loss_real
    optimizer.step()  # Update D
    return dis_loss, D_x


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    embedder_experiment_id = cfg["open-set-recognition"]["embedder_experiment_id"]
    openset_embeddings_name = cfg["open-set-recognition"]["openset_embeddings_name"]
    closedset_embeddings_name = cfg["open-set-recognition"]["closedset_embeddings_name"]
    embedding_size = cfg["open-set-recognition"]["embedding_size"]  # Number of channels in the embedding
    openset_embedding_output_path = EMBEDDINGS_DIR / openset_embeddings_name
    closedset_embedding_output_path = EMBEDDINGS_DIR / closedset_embeddings_name

    # TODO: move these params to hydra
    exp_dir = Path(
        '__file__').parent.absolute() / "openset_recognition_discriminators"  # experiment directory, used for reading the init model
    project_name = f"{embedder_experiment_id}_dlr_1e-6_open1closed0_no_gen"  # we save all the checkpoints in this directory
    SEED = 999
    lr_d = 1e-6  # learning rate discriminator
    num_epochs = 100  # total number of epoch in training
    batch_size = 128
    nz = 100  # Size of z latent vector (i.e. size of generator input)
    hidden_dim_g = 64  # Size of feature maps in generator
    hidden_dim_d = 64  # Size of feature maps in discriminator
    n_gpu = 1  # Number of GPUs available. Use 0 for CPU mode.
    openset_label = 1
    closedset_label = 0
    openset_examples = 100

    warnings.filterwarnings("ignore")
    print(sys.version)
    print(torch.__version__)
    set_seed(SEED)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    save_dir = os.path.join(exp_dir, project_name)
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set device, which gpu to use.
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")

    torch.cuda.device_count()
    torch.cuda.empty_cache()

    _, discriminator = build_models(nz, hidden_dim_g, embedding_size, hidden_dim_d, device, n_gpu)

    print("making closedset dataset")
    closedset_embeddings = get_cached_data(closedset_embedding_output_path)
    closedset_dataset = FeatDataset(data=closedset_embeddings, label=closedset_label)

    print("making openset dataset")
    openset_embeddings = get_cached_data(openset_embedding_output_path)
    if openset_examples < openset_embeddings.shape[0]:
        openset_embeddings = openset_embeddings[:openset_examples, ...]
    openset_dataset = FeatDataset(data=openset_embeddings, label=openset_label)

    print("combining dataloaders and balancing classes")
    dataset = ConcatDataset([openset_dataset, closedset_dataset])
    target = np.array(openset_dataset.target + closedset_dataset.target)
    # https://pytorch.org/docs/stable/data.html
    # https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight_per_class = 1. / class_sample_count
    weight_per_sample = np.array([weight_per_class[class_idx] for class_idx in target])
    weight_per_sample = torch.from_numpy(weight_per_sample)
    weight_per_sample = weight_per_sample.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_per_sample, len(weight_per_sample))
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=16,
                            timeout=120,
                            persistent_workers=True,
                            )

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr_d)

    dis_losses = []
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            train_outs = train(discriminator, data, criterion, optimizerD, device)
            dis_loss, dis_x = train_outs

            # Output training stats
            if i % 200 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tD(x): %.4f'
                      % (epoch, num_epochs, i, len(dataloader), dis_loss.item(), dis_x))

            # Save Losses for plotting later
            dis_losses.append(dis_loss.item())

        save_model_state(discriminator, epoch, save_dir)
        save_discriminator_loss_plot(dis_losses, project_name)


if __name__ == "__main__":
    main()
