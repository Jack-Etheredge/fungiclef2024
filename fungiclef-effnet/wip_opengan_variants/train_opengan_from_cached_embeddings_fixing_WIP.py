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
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import warnings  # ignore warnings

from closedset_model import get_embedding_size
from utils import set_seed, build_models, save_loss_plots
from paths import EMBEDDINGS_DIR


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


def save_model_state(generator, discriminator, epoch, save_dir, save_generator=False):
    """
    save a checkpoint of the model weights
    """

    cur_model_wts = copy.deepcopy(discriminator.state_dict())
    path_to_save_model_state = os.path.join(save_dir, f'epoch-{epoch + 1}-discriminator.pth')
    torch.save(cur_model_wts, path_to_save_model_state)

    if save_generator:
        cur_model_wts = copy.deepcopy(generator.state_dict())
        path_to_save_model_state = os.path.join(save_dir, f'epoch-{epoch + 1}-generator.pth')
        torch.save(cur_model_wts, path_to_save_model_state)


def train(generator, discriminator, data, criterion, optimizerG, optimizerD, nz, device, real_label=1., fake_label=0.,
          label_smoothing_eps=0.0):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # using close&open&fake data to update D

    adversarial_loss_weight = 0.2

    embeddings, labels = data
    real_cpu = embeddings.to(device)
    real_open_closed_batch_size = real_cpu.size(0)
    fake_batch_size = (real_open_closed_batch_size) // 4

    tmp_weights = torch.full((real_open_closed_batch_size + fake_batch_size,), 1, device=device)
    tmp_weights[-fake_batch_size:] *= adversarial_loss_weight
    criterionD = nn.BCELoss(weight=tmp_weights)

    # generate fakes
    noise = torch.randn(fake_batch_size, nz, device=device)  # Generate batch of latent vectors
    fake = generator(noise)  # Generate fake image batch with G
    labels_fake = torch.full((fake_batch_size,), fake_label, device=device)

    # update discriminator with real open, real closed, and fakes
    # (fakes can simulate open or closed depending on which has the 1 label)
    optimizerD.zero_grad()
    X = torch.cat((real_cpu, fake.detach()), 0)
    label_total = torch.cat((labels, labels_fake), 0)
    label_total = label_total * (1 - label_smoothing_eps) + (label_smoothing_eps / 2)
    output = discriminator(X).view(-1)
    dis_loss = criterionD(output, label_total)
    dis_loss.backward()
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    optimizerG.zero_grad()
    label = torch.full((fake_batch_size,), real_label, device=device)
    label = label * (1 - label_smoothing_eps) + (label_smoothing_eps / 2)
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = discriminator(fake).view(-1)
    gen_loss = criterion(output, label)  # Calculate G's loss based on this output
    gen_loss.backward()  # Calculate gradients for G
    optimizerG.step()  # Update G
    return dis_loss.mean().item(), gen_loss.mean().item()


def train_openganfea(cfg: DictConfig) -> str:
    embedder_experiment_id = cfg["evaluate"]["experiment_id"]
    openset_embeddings_name = cfg["open-set-recognition"]["openset_embeddings_name"]
    closedset_embeddings_name = cfg["open-set-recognition"]["closedset_embeddings_name"]
    openset_embedding_output_path = EMBEDDINGS_DIR / openset_embeddings_name
    closedset_embedding_output_path = EMBEDDINGS_DIR / closedset_embeddings_name
    lr_d = cfg["open-set-recognition"]["dlr"]  # learning rate discriminator
    lr_g = cfg["open-set-recognition"]["glr"]  # learning rate generator
    seed = cfg["open-set-recognition"]["seed"]
    num_epochs = cfg["open-set-recognition"]["epochs"]
    batch_size = cfg["open-set-recognition"]["batch_size"]
    nz = cfg["open-set-recognition"]["noise_vector_size"]  # Size of z latent vector (i.e. size of generator input)
    hidden_dim_g = cfg["open-set-recognition"]["hidden_dim_g"]  # Size of feature maps in generator
    hidden_dim_d = cfg["open-set-recognition"]["hidden_dim_d"]  # Size of feature maps in discriminator
    openset_label = cfg["open-set-recognition"]["openset_label"]
    closedset_label = cfg["open-set-recognition"]["closedset_label"]

    # get embedding size from the trained evaluation (embedder) model
    model_id = cfg["evaluate"]["model_id"]
    nc = get_embedding_size(model_id=model_id)

    # experiment directory, used for reading the init model
    # TODO: move this to the paths.py module
    exp_dir = Path('__file__').parent.absolute() / "openset_recognition_discriminators"

    n_gpu = 1  # Number of GPUs available. Use 0 for CPU mode.

    # all checkpoints saved to this directory
    # TODO: move this string construction to hydra
    project_name = embedder_experiment_id
    project_name += f"adambeta0.5_dlr_{lr_d:.0e}_glr_{lr_g:.0e}_open{openset_label}closed{closedset_label}"

    warnings.filterwarnings("ignore")
    print(sys.version)
    print(torch.__version__)
    set_seed(seed)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    save_dir = os.path.join(exp_dir, project_name)
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set device, which gpu to use.
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")

    torch.cuda.device_count()
    torch.cuda.empty_cache()

    generator, discriminator = build_models(nz, hidden_dim_g, nc, hidden_dim_d, device, n_gpu)

    noise = torch.randn(batch_size, nz, device=device)
    fake = generator(noise)  # Generate fake image batch with G
    predLabel = discriminator(fake)
    print("sanity checks:")
    print("noise shape:", noise.shape)
    print("fake embedding from generator shape:", fake.shape)
    print("label from discriminator shape:", predLabel.shape)

    print("making closedset dataset")
    closedset_embeddings = get_cached_data(closedset_embedding_output_path)
    # if n_closed_examples < closedset_embeddings.shape[0]:
    #     closedset_embeddings = closedset_embeddings[:n_closed_examples, ...]
    closedset_dataset = FeatDataset(data=closedset_embeddings, label=closedset_label)

    print("making openset dataset")
    openset_embeddings = get_cached_data(openset_embedding_output_path)
    # if n_open_examples < openset_embeddings.shape[0]:
    #     openset_embeddings = openset_embeddings[:n_open_examples, ...]
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
    # TODO: this should be configured from hydra
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=16,
                            timeout=120,
                            # persistent_workers=True,
                            )

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    # https://neptune.ai/blog/gan-failure-modes beta is actually an important hyperparameter for GAN training
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    # optimizerD = optim.AdamW(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    # optimizerG = optim.AdamW(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    # TODO: remove batch norm from the params for AdamW since batch norm doesn't need weight decay

    gen_losses = []
    dis_losses = []
    iters = 0
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            train_outs = train(generator, discriminator, data, criterion, optimizerG, optimizerD, nz, device)
            dis_loss, gen_loss = train_outs

            # Output training stats
            if i % 200 == 0:
                print((f"[{epoch}/{num_epochs}][{i + 1}/{len(dataloader)}]"
                       f"\tLoss_D: {dis_loss: .4f}\tLoss_G: {gen_loss: .4f}"))

            # Save Losses for plotting later
            gen_losses.append(gen_loss)
            dis_losses.append(dis_loss)

            iters += 1

        save_model_state(generator, discriminator, epoch, save_dir)
        save_loss_plots(gen_losses, dis_losses, project_name, save_dir)

    return project_name


if __name__ == "__main__":
    # using this instead of @hydra.main decorator so main function can be called from elsewhere
    with initialize(version_base=None, config_path="conf", job_name="evaluate"):
        cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))
    train_openganfea(cfg)
