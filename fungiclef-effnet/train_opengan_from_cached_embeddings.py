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
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import warnings  # ignore warnings

from closedset_model import get_embedding_size
from utils import set_seed, build_wgangp_models, save_loss_plots
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


def calc_gradient_penalty(discriminator, real_data, generated_data, device, gp_weight=10.):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()


def train(generator, discriminator, data, batch_count, criterion, optimizerG, optimizerD, nz, device, real_label=1.,
          fake_label=0., label_smoothing_eps=0.0, critic_iter=2):
    ## Train with all-real batch
    optimizerD.zero_grad()
    # Format batch
    embeddings, labels = data
    real_cpu = embeddings.to(device)
    label = labels.float().to(device)
    label = label * (1 - label_smoothing_eps) + (label_smoothing_eps / 2)
    output_real = discriminator(real_cpu).view(-1)  # Forward pass real batch through D
    dis_loss_real = criterion(output_real, label)  # Calculate loss on all-real batch
    # Calculate gradients for D in backward pass
    dis_loss_real.backward()
    optimizerD.step()

    ## WGAN-GP https://arxiv.org/pdf/1704.00028v3
    optimizerD.zero_grad()
    real_labeled_real_embeddings = embeddings[labels == real_label].to(device)
    real = real_labeled_real_embeddings
    b_size = real.size(0)
    noise = torch.randn(b_size, nz, device=device)  # Generate batch of latent vectors
    fake = generator(noise)  # Generate fake image batch with G

    d_real = discriminator(real).view(-1)
    d_generated = discriminator(fake.detach()).view(-1)
    gradient_penalty = calc_gradient_penalty(discriminator, real, fake, device)

    dis_loss = d_generated.mean() - d_real.mean() + gradient_penalty
    dis_loss.backward()
    optimizerD.step()  # Update D
    dis_loss = dis_loss.item()

    # train generator every n discriminator steps
    if batch_count % critic_iter == 0:
        optimizerG.zero_grad()
        noise = torch.randn(real_cpu.size(0), nz, device=device)  # Generate batch of latent vectors
        fake = generator(noise)
        d_generated = discriminator(fake).view(-1)
        gen_loss = -d_generated.mean()  # Calculate G's loss based on this output
        gen_loss.backward()  # Calculate gradients for G
        optimizerG.step()  # Update G
        gen_loss = gen_loss.item()
    else:
        gen_loss = 0.0

    return dis_loss, gen_loss


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
    project_name += f"dlr_{lr_d:.0e}_glr_{lr_g:.0e}_open{openset_label}closed{closedset_label}_wgangp_adam_ncritic2"

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

    generator, discriminator = build_wgangp_models(nz, hidden_dim_g, nc, hidden_dim_d, device, n_gpu)

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
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.9))
    optimizerG = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.9))
    # optimizerD = optim.AdamW(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.9))
    # optimizerG = optim.AdamW(generator.parameters(), lr=lr_g, betas=(0.0, 0.9))
    # TODO: remove batch norm and layer norm from the params for AdamW since batch norm doesn't need weight decay

    gen_losses = []
    dis_losses = []
    iters = 0
    last_gen_loss = 0.0
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            train_outs = train(generator, discriminator, data, iters, criterion, optimizerG, optimizerD, nz, device)
            dis_loss, gen_loss = train_outs

            # Output training stats
            if i % 200 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch, num_epochs, i, len(dataloader), dis_loss, gen_loss))

            if gen_loss is None:
                # repeat gen loss for plotting since it's not updated every step
                gen_loss = last_gen_loss
            else:
                last_gen_loss = gen_loss

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
