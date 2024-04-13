import torch

from temperature_scaling import ModelWithTemperature
import hydra
from omegaconf import DictConfig, OmegaConf
from closedset_model import build_model
from paths import CHECKPOINT_DIR
from datasets import get_data_loaders, get_datasets


@hydra.main(version_base=None, config_path="conf", config_name="config")
def create_temperature_scaled_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    n_classes = cfg["train"]["n_classes"]
    pretrained = cfg["train"]["pretrained"]
    batch_size = cfg["train"]["batch_size"]
    num_dataloader_workers = cfg["train"]["num_dataloader_workers"]
    image_resize = cfg["train"]["image_resize"]
    validation_frac = cfg["train"]["validation_frac"]
    undersample = cfg["train"]["undersample"]
    oversample = cfg["train"]["oversample"]
    equal_undersampled_val = cfg["train"]["equal_undersampled_val"]
    oversample_prop = cfg["train"]["oversample_prop"]
    balanced_sampler = cfg["train"]["balanced_sampler"]
    experiment_id = cfg["temp-scaling"]["experiment_id"]

    print(f"creating temperature scaled model {experiment_id}")

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

    experiment_dir = CHECKPOINT_DIR / experiment_id
    checkpoint_path = experiment_dir / "model.pth"
    temp_scaled_model_filename = experiment_dir / "model_with_temperature.pth"

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets(pretrained, image_resize, validation_frac,
                                                                 oversample=oversample, undersample=undersample,
                                                                 oversample_prop=oversample_prop,
                                                                 equal_undersampled_val=equal_undersampled_val)
    _, valid_loader = get_data_loaders(dataset_train, dataset_valid, batch_size, num_dataloader_workers,
                                       balanced_sampler=balanced_sampler)

    model = build_model(
        pretrained=False,
        fine_tune=False,
        num_classes=n_classes,  # this is all that matters. everything else will be overwritten by checkpoint state
        dropout_rate=0.5,
    ).to(device)
    model.eval()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    temp_scaled_model = ModelWithTemperature(model)
    print("temperature scaling model")
    temp_scaled_model.set_temperature(valid_loader)
    torch.save(temp_scaled_model.state_dict(), temp_scaled_model_filename)


if __name__ == "__main__":
    create_temperature_scaled_model()
