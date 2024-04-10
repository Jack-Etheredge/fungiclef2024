"""
Create embeddings to use to train openGAN
"""

from pathlib import Path

import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset

from torchvision.transforms import v2, InterpolationMode
from tqdm import tqdm

from utils import CustomImageDataset, collate_fn
import hydra
from omegaconf import DictConfig, OmegaConf

FEAT_DIR = Path('__file__').parent.absolute() / "feats"
FEAT_DIR.mkdir(parents=True, exist_ok=True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    embedder_experiment_id = cfg["open-set-recognition"]["embedder_experiment_id"]
    embedder_layer_offset = cfg["open-set-recognition"]["embedder_layer_offset"]
    batch_size = cfg["open-set-recognition"]["batch_size"]
    n_workers = cfg["open-set-recognition"]["n_workers"]
    openset_embeddings_name = cfg["open-set-recognition"]["openset_embeddings_name"]
    closedset_embeddings_name = cfg["open-set-recognition"]["closedset_embeddings_name"]
    image_size = cfg["open-set-recognition"]["image_size"]

    openset_embedding_output_path = FEAT_DIR / f"{openset_embeddings_name}.npy"
    closedset_embedding_output_path = FEAT_DIR / f"{closedset_embeddings_name}.npy"

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")

    model = torchvision.models.resnet18(weights='DEFAULT')  # model is trained on Imagenet1K
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    # TODO: move the transformations to utils and get them for train, val, test
    transforms = v2.Compose([
        v2.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # TODO: replace with fungiclef and determine how to split the data
    # generate embeddings for CIFAR10 since it's not in the training set for Imagenet1K
    val_dir = Path("~").expanduser().absolute() / "datasets" / "cifar10"
    cifar_dataset = torchvision.datasets.CIFAR10(root=cifar10_dir, train=True,
                                                 download=True, transform=transforms)
    openset_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=n_workers)
    imagenet_dir = Path("~").expanduser().absolute() / "datasets" / "imagenet" / "test_images"
    imagenet_dataset = CustomImageDataset(img_dir=imagenet_dir, transform=transforms)
    closedset_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=n_workers,
                                                   collate_fn=collate_fn)

    embeddings = []
    print("creating embeddings for open set")
    with torch.no_grad():
        for data in tqdm(openset_loader):
            images, _ = data
            # calculate outputs by running images through the network
            outputs = feature_extractor(images).detach().cpu().numpy()
            embeddings.append(outputs)
    embeddings = np.concatenate(embeddings).squeeze()
    np.save(openset_embedding_output_path, embeddings)

    embeddings = []
    print("creating embeddings for closed set")
    with torch.no_grad():
        for data in tqdm(closedset_loader):
            images, _ = data
            # calculate outputs by running images through the network
            outputs = feature_extractor(images).detach().cpu().numpy()
            embeddings.append(outputs)
    embeddings = np.concatenate(embeddings).squeeze()
    np.save(closedset_embedding_output_path, embeddings)


if __name__ == "__main__":
    main()
