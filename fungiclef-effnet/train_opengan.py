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
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import warnings  # ignore warnings

from closedset_model import get_embedding_size
from utils import set_seed, build_models, save_loss_plots
from paths import EMBEDDINGS_DIR

from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from paths import CHECKPOINT_DIR, EMBEDDINGS_DIR
from datasets import collate_fn, get_openset_datasets, get_datasets
from closedset_model import build_model, load_model_for_inference
from utils import get_model_features


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


def train(embedder, generator, discriminator, data, criterion, optimizerG, optimizerD, nz, device, real_label=1.,
          fake_label=0.):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    optimizerD.zero_grad()
    # Format batch
    images, labels = data
    embeddings = get_model_features(images, embedder, device).detach().cpu()
    real_cpu = embeddings.to(device)
    label = labels.float().to(device)
    b_size = real_cpu.size(0)
    output = discriminator(real_cpu).view(-1)  # Forward pass real batch through D
    dis_loss_real = criterion(output, label)  # Calculate loss on all-real batch
    # Calculate gradients for D in backward pass
    dis_loss_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    noise = torch.randn(b_size, nz, device=device)  # Generate batch of latent vectors
    fake = generator(noise)  # Generate fake image batch with G
    label.fill_(fake_label)
    # Classify all fake batch with D
    output = discriminator(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    dis_loss = dis_loss_real + errD_fake  # Add the gradients from the all-real and all-fake batches
    optimizerD.step()  # Update D

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    optimizerG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = discriminator(fake).view(-1)
    gen_loss = criterion(output, label)  # Calculate G's loss based on this output
    gen_loss.backward()  # Calculate gradients for G
    D_G_z2 = output.mean().item()
    optimizerG.step()  # Update G
    return dis_loss, gen_loss, D_x, D_G_z1, D_G_z2


def train_openganfea(cfg: DictConfig) -> str:
    # select embedder from evaluate experiment_id
    embedder_experiment_id = cfg["evaluate"]["experiment_id"]
    model_id = cfg["evaluate"]["model_id"]
    image_size = cfg["evaluate"]["image_size"]
    use_metadata = cfg["evaluate"]["use_metadata"]

    # embedder setup
    # need batch size of the embedder model now instead of the lightweight GAN models
    batch_size = int(cfg["evaluate"]["batch_size"] * 2.5)
    n_workers = cfg["open-set-recognition"]["n_workers"]
    openset_n_train = cfg["open-set-recognition"]["openset_n_train"]
    openset_n_val = cfg["open-set-recognition"]["openset_n_val"]
    closedset_n_train = cfg["open-set-recognition"]["closedset_n_train"]
    pretrained = cfg["open-set-recognition"]["pretrained"]

    # for closed set dataset
    undersample = cfg["train"]["undersample"]
    oversample = cfg["train"]["oversample"]
    equal_undersampled_val = cfg["train"]["equal_undersampled_val"]
    oversample_prop = cfg["train"]["oversample_prop"]
    validation_frac = cfg["train"]["validation_frac"]

    # for data loaders
    worker_timeout_s = cfg["train"]["worker_timeout_s"]

    # for constructing model
    n_classes = cfg["train"]["n_classes"]

    # openGAN settings
    lr_d = cfg["open-set-recognition"]["dlr"]  # learning rate discriminator
    lr_g = cfg["open-set-recognition"]["glr"]  # learning rate generator
    seed = cfg["open-set-recognition"]["seed"]
    num_epochs = cfg["open-set-recognition"]["epochs"]
    nz = cfg["open-set-recognition"]["noise_vector_size"]  # Size of z latent vector (i.e. size of generator input)
    hidden_dim_g = cfg["open-set-recognition"]["hidden_dim_g"]  # Size of feature maps in generator
    hidden_dim_d = cfg["open-set-recognition"]["hidden_dim_d"]  # Size of feature maps in discriminator
    openset_label = cfg["open-set-recognition"]["openset_label"]
    closedset_label = cfg["open-set-recognition"]["closedset_label"]

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")

    print("constructing embedder (closed set classifier outputting from penultimate layer)")
    experiment_dir = CHECKPOINT_DIR / embedder_experiment_id
    model = load_model_for_inference(device, experiment_dir, model_id, n_classes)

    # get embedding size from the trained evaluation (embedder) model
    model_id = cfg["evaluate"]["model_id"]
    nc = get_embedding_size(model_id=model_id)

    # experiment directory, used for reading the init model
    # TODO: move this to the paths.py module
    exp_dir = Path('__file__').parent.absolute() / "openset_recognition_discriminators"

    n_gpu = 1  # Number of GPUs available. Use 0 for CPU mode.

    # all checkpoints saved to this directory
    # TODO: move this string construction to hydra
    project_name = f"{embedder_experiment_id}_dlr_{lr_d:.0e}_glr_{lr_g:.0e}_open{openset_label}closed{closedset_label}"

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

    training_augs = True
    openset_dataset, _, _ = get_openset_datasets(pretrained=pretrained, image_size=image_size,
                                                 n_train=openset_n_train, n_val=openset_n_val,
                                                 include_metadata=use_metadata, training_augs=training_augs)
    openset_dataset.target = [openset_label] * len(openset_dataset.target)
    # openset_loader = torch.utils.data.DataLoader(openset_dataset, batch_size=batch_size,
    #                                              shuffle=False, num_workers=n_workers, collate_fn=collate_fn,
    #                                              timeout=worker_timeout_s)
    closedset_dataset, _, _ = get_datasets(pretrained=pretrained, image_size=image_size,
                                           validation_frac=validation_frac,
                                           oversample=oversample, undersample=undersample,
                                           oversample_prop=oversample_prop,
                                           equal_undersampled_val=equal_undersampled_val,
                                           include_metadata=use_metadata, training_augs=training_augs)
    closedset_dataset.target = np.array([closedset_label] * len(closedset_dataset.target))
    closedset_dataset = Subset(closedset_dataset,
                               np.random.choice(closedset_dataset.target.shape[0], closedset_n_train, replace=False))
    closedset_dataset.target = [closedset_label] * len(closedset_dataset)
    # closedset_loader = torch.utils.data.DataLoader(closedset_dataset, batch_size=batch_size,
    #                                                shuffle=False, num_workers=n_workers, collate_fn=collate_fn,
    #                                                timeout=worker_timeout_s)

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
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler,
                            num_workers=12,
                            timeout=240,
                            collate_fn=collate_fn,
                            # persistent_workers=True,
                            )

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr_d)
    optimizerG = optim.Adam(generator.parameters(), lr=lr_g)

    gen_losses = []
    dis_losses = []
    iters = 0
    print("Starting Training Loop...")
    for epoch in tqdm(range(num_epochs)):
        for i, data in enumerate(dataloader):
            images, labels = data
            labels[labels > -1] = closedset_label
            labels[labels == -1] = openset_label
            data = images, labels
            train_outs = train(model, generator, discriminator, data, criterion, optimizerG, optimizerD, nz, device)
            dis_loss, gen_loss, dis_x, dis_g_z1, dis_g_z2 = train_outs

            # Output training stats
            if i % 200 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         dis_loss.item(), gen_loss.item(), dis_x, dis_g_z1, dis_g_z2))

            # Save Losses for plotting later
            gen_losses.append(gen_loss.item())
            dis_losses.append(dis_loss.item())

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
