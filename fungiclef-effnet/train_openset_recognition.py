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

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import warnings  # ignore warnings

from opengan.utils import save_loss_plots, set_seed, build_models


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


def get_cached_data(np_path):
    """
    load cached set of features from a numpy file
    """
    cached_data = np.load(np_path)
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


def train(discriminator, data, criterion, optimizerD, device):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    optimizerD.zero_grad()
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
    optimizerD.step()  # Update D
    return dis_loss, D_x


def main():
    # HYPERPARAMETERS
    # TODO: move common elements to config
    FEAT_DIR = Path('__file__').parent.absolute() / "feats"
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    openset_embedding_output_path = FEAT_DIR / "resnet18imagenet1k_cifar10_embeddings.npy"
    closedset_embedding_output_path = FEAT_DIR / "resnet18imagenet1k_imagenet1k_embeddings.npy"
    exp_dir = "./opengan_exp"  # experiment directory, used for reading the init model
    project_name = "opengan_fea_Res18sc_mlpGAN_dlr_1e-6_open1closed0_100_examples_no_gen"  # we save all the checkpoints in this directory
    SEED = 999
    lr_d = 1e-6  # learning rate discriminator
    num_epochs = 25  # total number of epoch in training
    batch_size = 128
    nc = 512  # Number of channels in the embedding
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

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(batch_size, nz, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr_d)

    img_list = []
    gen_losses = []
    dis_losses = []
    iters = 0
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

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                # TODO: figure out what to do with this unused img_list

            iters += 1

        save_model_state(generator, discriminator, epoch, save_dir)
        save_loss_plots(gen_losses, dis_losses, project_name)


if __name__ == "__main__":
    main()
