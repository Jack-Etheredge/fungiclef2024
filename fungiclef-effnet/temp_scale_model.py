import torch
from hydra import compose, initialize

from temperature_scaling import ModelWithTemperature
from omegaconf import DictConfig, OmegaConf
from closedset_model import load_model_for_inference
from paths import CHECKPOINT_DIR
from datasets import get_data_loaders, get_datasets


def create_temperature_scaled_model(cfg: DictConfig) -> None:
    n_classes = cfg["train"]["n_classes"]
    pretrained = cfg["train"]["pretrained"]
    num_dataloader_workers = cfg["train"]["num_dataloader_workers"]
    validation_frac = cfg["train"]["validation_frac"]
    undersample = cfg["train"]["undersample"]
    oversample = cfg["train"]["oversample"]
    equal_undersampled_val = cfg["train"]["equal_undersampled_val"]
    oversample_prop = cfg["train"]["oversample_prop"]
    balanced_sampler = cfg["train"]["balanced_sampler"]

    experiment_id = cfg["evaluate"]["experiment_id"]
    model_id = cfg["evaluate"]["model_id"]
    image_resize = cfg["evaluate"]["image_size"]
    use_metadata = cfg["evaluate"]["use_metadata"]
    batch_size = cfg["evaluate"]["batch_size"]

    print(f"creating temperature scaled model {experiment_id}")

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

    experiment_dir = CHECKPOINT_DIR / experiment_id
    temp_scaled_model_filename = experiment_dir / "model_with_temperature.pth"

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets(cfg, pretrained, image_resize, validation_frac,
                                                                 oversample=oversample, undersample=undersample,
                                                                 oversample_prop=oversample_prop,
                                                                 equal_undersampled_val=equal_undersampled_val,
                                                                 include_metadata=use_metadata)
    _, valid_loader = get_data_loaders(dataset_train, dataset_valid, batch_size, num_dataloader_workers,
                                       balanced_sampler=balanced_sampler)

    model = load_model_for_inference(device, experiment_dir, model_id, n_classes)
    temp_scaled_model = ModelWithTemperature(model)
    print("temperature scaling model")
    temp_scaled_model.set_temperature(valid_loader)
    torch.save(temp_scaled_model.state_dict(), temp_scaled_model_filename)


if __name__ == "__main__":
    # using this instead of @hydra.main decorator so main function can be called from elsewhere
    with initialize(version_base=None, config_path="conf", job_name="evaluate"):
        cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))
    create_temperature_scaled_model(cfg)
