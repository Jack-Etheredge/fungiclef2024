"""
Run all steps to create and select an openGANfea discriminator and save it in the model checkpoint dir.
"""
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch import nn

from choose_openset_recognition_discriminator import choose_best_discriminator, create_feature_extractor_from_model, \
    load_model_for_inference
from closedset_model import get_embedding_size
from create_embeddings_openset_recognition import create_embeddings
from openset_recognition_models import Discriminator
from paths import CHECKPOINT_DIR
from train_opengan import train_openganfea


class CompositeOpenGANInferenceModel(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model, opengan_discriminator, embedder_layer_offset, openset_label):
        super().__init__()
        self.model = model
        self.opengan_discriminator = opengan_discriminator
        self.feature_extractor = create_feature_extractor_from_model(model, embedder_layer_offset)
        self.openset_label = openset_label

    def forward(self, input):
        model_preds = torch.argmax(self.model(input), dim=1)
        if model_preds.dim() == 1:
            model_preds = model_preds.unsqueeze(0)
        penultimate_layer_output = self.feature_extractor(input).squeeze()
        if penultimate_layer_output.dim() == 1:
            penultimate_layer_output = penultimate_layer_output.unsqueeze(0)
        # opengan_preds = (self.opengan_discriminator(penultimate_layer_output) > 0.5).int()
        # model_preds[opengan_preds == self.openset_label] = -1
        opengan_probas = self.opengan_discriminator(penultimate_layer_output)
        return model_preds, opengan_probas


def create_composite_model(cfg: DictConfig) -> nn.Module:
    experiment_id = cfg["evaluate"]["experiment_id"]
    experiment_dir = CHECKPOINT_DIR / experiment_id
    hidden_dim = cfg["open-set-recognition"]["hidden_dim_d"]
    model_id = cfg["evaluate"]["model_id"]
    use_timm = cfg["evaluate"]["use_timm"]
    n_classes = cfg["train"]["n_classes"]
    openset_label = cfg["open-set-recognition"]["openset_label"]
    embedder_layer_offset = cfg["open-set-recognition"]["embedder_layer_offset"]

    # get embedding size from the trained evaluation (embedder) model
    nc = get_embedding_size(model_id=model_id, use_timm=use_timm)

    # set device, which gpu to use.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"using device {device}")

    model = load_model_for_inference(device, experiment_dir, model_id, n_classes, use_timm)
    opengan_model = load_opengan_discriminator(device, experiment_dir, hidden_dim, nc)
    composite_model = CompositeOpenGANInferenceModel(model, opengan_model, embedder_layer_offset, openset_label)
    # torch.save(composite_model, "opengan_composite_model.pth")
    return composite_model


def load_opengan_discriminator(device, experiment_dir, hidden_dim, nc):
    discriminator_path = experiment_dir / "openganfea_model.pth"
    opengan_model = Discriminator(nc=nc, hidden_dim=hidden_dim).to(device)
    try:
        opengan_model.load_state_dict(torch.load(discriminator_path), strict=False)
    except:
        opengan_model = torch.load(discriminator_path)
    opengan_model.eval()
    return opengan_model


def train_and_select_discriminator(cfg: DictConfig) -> None:
    create_embeddings(cfg)
    discriminators_dir_name = train_openganfea(cfg)
    choose_best_discriminator(cfg, project_name=discriminators_dir_name)
    # create_composite_model(cfg)


if __name__ == "__main__":
    # using this instead of @hydra.main decorator so main function can be called from elsewhere
    with initialize(version_base=None, config_path="conf", job_name="evaluate"):
        cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))
    train_and_select_discriminator(cfg)