"""
Create embeddings to use to train openGAN
"""
import h5py
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from paths import CHECKPOINT_DIR, EMBEDDINGS_DIR
from datasets import collate_fn, get_openset_datasets, get_datasets
from closedset_model import build_model, load_model_for_inference
from utils import get_model_features


def create_embeddings(cfg: DictConfig) -> None:
    # select embedder from evaluate experiment_id
    embedder_experiment_id = cfg["evaluate"]["experiment_id"]
    model_id = cfg["evaluate"]["model_id"]
    image_size = cfg["evaluate"]["image_size"]
    use_metadata = cfg["evaluate"]["use_metadata"]

    # embedder setup
    embedder_layer_offset = cfg["open-set-recognition"]["embedder_layer_offset"]
    batch_size = cfg["open-set-recognition"]["batch_size"]
    n_workers = cfg["open-set-recognition"]["n_workers"]
    openset_embeddings_name = cfg["open-set-recognition"]["openset_embeddings_name"]
    closedset_embeddings_name = cfg["open-set-recognition"]["closedset_embeddings_name"]
    openset_n_train = cfg["open-set-recognition"]["openset_n_train"]
    openset_n_val = cfg["open-set-recognition"]["openset_n_val"]
    closedset_n_train = cfg["open-set-recognition"]["closedset_n_train"]
    pretrained = cfg["open-set-recognition"]["pretrained"]
    openset_oversample_rate = cfg["open-set-recognition"]["openset_oversample_rate"]
    closedset_oversample_rate = cfg["open-set-recognition"]["closedset_oversample_rate"]

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

    openset_embedding_output_path = EMBEDDINGS_DIR / openset_embeddings_name
    closedset_embedding_output_path = EMBEDDINGS_DIR / closedset_embeddings_name

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")

    print("constructing embedder (closed set classifier outputting from penultimate layer)")
    experiment_dir = CHECKPOINT_DIR / embedder_experiment_id
    model = load_model_for_inference(device, experiment_dir, model_id, n_classes)
    # feature_extractor = torch.nn.Sequential(*list(model.children())[:embedder_layer_offset])

    training_augs = True
    if closedset_oversample_rate < 1 or openset_oversample_rate < 1:
        raise ValueError("Oversample rates must be integers >= 1 (with 1 being no oversampling).")
    if closedset_oversample_rate == 1 and openset_oversample_rate == 1:
        training_augs = False
        print("no oversampling of either closed set or open set data, so disabling training data augmentations.")
    openset_dataset_train, _, _ = get_openset_datasets(cfg, pretrained=pretrained, image_size=image_size,
                                                       n_train=openset_n_train, n_val=openset_n_val,
                                                       include_metadata=use_metadata, training_augs=training_augs)
    openset_loader = torch.utils.data.DataLoader(openset_dataset_train, batch_size=batch_size,
                                                 shuffle=False, num_workers=n_workers, collate_fn=collate_fn,
                                                 timeout=worker_timeout_s)
    closedset_dataset, _, _ = get_datasets(cfg, pretrained=pretrained, image_size=image_size,
                                           validation_frac=validation_frac,
                                           oversample=oversample, undersample=undersample,
                                           oversample_prop=oversample_prop,
                                           equal_undersampled_val=equal_undersampled_val,
                                           include_metadata=use_metadata, training_augs=training_augs)
    if closedset_n_train is not None:
        closedset_dataset = Subset(closedset_dataset,
                                   np.random.choice(closedset_dataset.target.shape[0], closedset_n_train,
                                                    replace=False))
    else:
        print(f"using all closed set training data: {len(closedset_dataset)} observations")
    closedset_loader = torch.utils.data.DataLoader(closedset_dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=n_workers, collate_fn=collate_fn,
                                                   timeout=worker_timeout_s)

    embeddings = []
    print("creating embeddings for open set")
    with torch.no_grad():
        for i in range(openset_oversample_rate):
            print(f"{i + 1} of {openset_oversample_rate} passes through dataset with {openset_n_train} observations")
            for data in tqdm(openset_loader):
                images, _ = data
                outputs = get_model_features(images, model, device).detach().cpu().numpy()
                embeddings.append(outputs)
    embeddings = np.concatenate(embeddings).squeeze()
    with h5py.File(openset_embedding_output_path, 'w') as hf:
        hf.create_dataset("data", data=embeddings)
    print("saved embeddings with shape", embeddings.shape)

    embeddings = []
    print("creating embeddings for closed set")
    n_closedset = len(closedset_dataset)
    with torch.no_grad():
        for i in range(closedset_oversample_rate):
            print(f"{i + 1} of {closedset_oversample_rate} passes through dataset with {n_closedset} observations")
            for data in tqdm(closedset_loader):
                images, _ = data
                outputs = get_model_features(images, model, device).detach().cpu().numpy()
                embeddings.append(outputs)
    embeddings = np.concatenate(embeddings).squeeze()
    with h5py.File(closedset_embedding_output_path, 'w') as hf:
        hf.create_dataset("data", data=embeddings)
    print("saved embeddings with shape", embeddings.shape)


if __name__ == "__main__":
    # using this instead of @hydra.main decorator so main function can be called from elsewhere
    with initialize(version_base=None, config_path="conf", job_name="evaluate"):
        cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))
    create_embeddings(cfg)
