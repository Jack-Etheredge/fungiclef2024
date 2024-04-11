"""
Create embeddings to use to train openGAN
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from paths import CHECKPOINT_DIR, EMBEDDINGS_DIR
from datasets import collate_fn, get_openset_datasets, get_datasets
from closedset_model import build_model


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

    # for constructing model
    n_classes = cfg["train"]["n_classes"]

    openset_embedding_output_path = EMBEDDINGS_DIR / openset_embeddings_name
    closedset_embedding_output_path = EMBEDDINGS_DIR / closedset_embeddings_name

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")

    print("constructing embedder (closed set classifier outputting from penultimate layer)")
    model = build_model(
        pretrained=False,  # we don't need to load the imagenet weights since we're going to load fine-tuned weights
        fine_tune=False,  # we don't need to unfreeze any weights
        num_classes=n_classes,
        dropout_rate=0.5,  # doesn't matter since embeddings will be created in eval on a fixed model
    ).to(device)
    experiment_dir = CHECKPOINT_DIR / embedder_experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_file_path = str(experiment_dir / f"model.pth")
    checkpoint = torch.load(model_file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    # transforms = v2.Compose([
    #     v2.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
    #     v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    #     v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    openset_dataset_train, _, _ = get_openset_datasets(pretrained=pretrained, image_size=image_size,
                                                       n_train=openset_n_train, n_val=openset_n_val)
    openset_loader = torch.utils.data.DataLoader(openset_dataset_train, batch_size=batch_size,
                                                 shuffle=False, num_workers=n_workers, collate_fn=collate_fn,
                                                 timeout=120)  # TODO move timeout to hydra
    closedset_dataset, _, _ = get_datasets(pretrained=pretrained, image_size=image_size,
                                           validation_frac=validation_frac,
                                           oversample=oversample, undersample=undersample,
                                           oversample_prop=oversample_prop,
                                           equal_undersampled_val=equal_undersampled_val)
    closedset_dataset = Subset(closedset_dataset,
                               np.random.choice(closedset_dataset.target.shape[0], closedset_n_train, replace=False))
    closedset_loader = torch.utils.data.DataLoader(closedset_dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=n_workers, collate_fn=collate_fn,
                                                   timeout=120)  # TODO move timeout to hydra

    embeddings = []
    print("creating embeddings for open set")
    with torch.no_grad():
        for data in tqdm(openset_loader):
            images, _ = data
            images = images.to(device)
            # calculate outputs by running images through the network
            outputs = feature_extractor(images).detach().cpu().numpy()
            embeddings.append(outputs)
    embeddings = np.concatenate(embeddings).squeeze()
    with h5py.File(openset_embedding_output_path, 'w') as hf:
        hf.create_dataset("data", data=embeddings)
    print("saved embeddings with shape", embeddings.shape)

    embeddings = []
    print("creating embeddings for closed set")
    with torch.no_grad():
        for data in tqdm(closedset_loader):
            images, _ = data
            images = images.to(device)
            # calculate outputs by running images through the network
            outputs = feature_extractor(images).detach().cpu().numpy()
            embeddings.append(outputs)
    embeddings = np.concatenate(embeddings).squeeze()
    with h5py.File(closedset_embedding_output_path, 'w') as hf:
        hf.create_dataset("data", data=embeddings)
    print("saved embeddings with shape", embeddings.shape)


if __name__ == "__main__":
    main()
