"""
Choose best discriminator checkpoint for openGANfea
- closed set is the dataset used to train the closed set classifier that generates the embeddings for openGANfea
- open set is the dataset used to train openGANfea
- this example uses imagenet1K pretrained resnet18 as the classification network, so imagenet is the closed set dataset
- this example uses cifar10 as the cross-dataset open set, so cifar10 is the open set dataset
"""
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from paths import CHECKPOINT_DIR
from tqdm import tqdm

from closedset_model import build_model, get_embedding_size
from datasets import get_openset_datasets, get_datasets, get_closedset_test_dataset, collate_fn
from openset_recognition_models import Discriminator
from utils import get_model_features


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def choose_best_discriminator(cfg: DictConfig, project_name=None) -> Path:
    hidden_dim = cfg["open-set-recognition"]["hidden_dim_d"]
    eval_fraction = 0.1
    max_total_examples = 10_000  # TODO: reincorporate this

    # select embedder from evaluate experiment_id
    embedder_experiment_id = cfg["evaluate"]["experiment_id"]
    model_id = cfg["evaluate"]["model_id"]
    image_size = cfg["evaluate"]["image_size"]
    use_metadata = cfg["evaluate"]["use_metadata"]

    # get embedding size from the trained evaluation (embedder) model
    nc = get_embedding_size(model_id=model_id)

    openset_label = float(cfg["open-set-recognition"]["openset_label"])
    closedset_label = float(cfg["open-set-recognition"]["closedset_label"])
    if project_name is None:
        lr_d = cfg["open-set-recognition"]["lr_d"]  # learning rate discriminator
        lr_g = cfg["open-set-recognition"]["lr_g"]  # learning rate generator
        project_name = (f"{embedder_experiment_id}_dlr_{lr_d:.0e}_glr_{lr_g:.0e}"
                        f"_open{int(openset_label)}closed{int(closedset_label)}")
    exp_dir = Path(
        '__file__').parent.absolute() / "openset_recognition_discriminators"  # experiment directory, used for reading the init model
    discriminator_dir = exp_dir / project_name

    embedder_layer_offset = cfg["open-set-recognition"]["embedder_layer_offset"]
    batch_size = cfg["open-set-recognition"]["batch_size"]
    n_workers = cfg["open-set-recognition"]["n_workers"]
    openset_embeddings_name = cfg["open-set-recognition"]["openset_embeddings_name"]
    closedset_embeddings_name = cfg["open-set-recognition"]["closedset_embeddings_name"]
    openset_n_train = cfg["open-set-recognition"]["openset_n_train"]
    openset_n_val = cfg["open-set-recognition"]["openset_n_val"]
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

    # set device, which gpu to use.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"using device {device}")

    # filtering for "discriminator" in model state checkpoint files necessary because generator and discriminator
    # states were saved in the same directory
    discriminators = [dis for dis in discriminator_dir.iterdir() if
                      dis.suffix == ".pth" and "discriminator" in dis.stem]

    experiment_dir = CHECKPOINT_DIR / embedder_experiment_id
    if not experiment_dir.exists():
        raise ValueError(f"no checkpoint directory for embedder at {experiment_dir}")

    print("constructing embedder (closed set classifier outputting from penultimate layer)")
    model = load_model_for_inference(device, experiment_dir, model_id, n_classes)
    # feature_extractor = create_feature_extractor_from_model(model, embedder_layer_offset)

    # transforms = v2.Compose([
    #     v2.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
    #     v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    #     v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    _, openset_dataset_val, openset_dataset_test = get_openset_datasets(pretrained=pretrained, image_size=image_size,
                                                                        n_train=openset_n_train, n_val=openset_n_val,
                                                                        include_metadata=use_metadata)
    open_set_selection_loader = torch.utils.data.DataLoader(openset_dataset_val, batch_size=batch_size,
                                                            shuffle=False, num_workers=4, collate_fn=collate_fn,
                                                            timeout=worker_timeout_s)
    open_set_evaluation_loader = torch.utils.data.DataLoader(openset_dataset_test, batch_size=batch_size,
                                                             shuffle=False, num_workers=4,
                                                             collate_fn=collate_fn,
                                                             timeout=worker_timeout_s)
    _, closedset_dataset_val, _ = get_datasets(pretrained=pretrained, image_size=image_size,
                                               validation_frac=validation_frac,
                                               oversample=oversample, undersample=undersample,
                                               oversample_prop=oversample_prop,
                                               equal_undersampled_val=equal_undersampled_val,
                                               include_metadata=use_metadata)
    closed_set_selection_loader = torch.utils.data.DataLoader(closedset_dataset_val, batch_size=batch_size,
                                                              shuffle=False, num_workers=4,
                                                              collate_fn=collate_fn,
                                                              timeout=worker_timeout_s)
    closedset_dataset_test = get_closedset_test_dataset(pretrained, image_size, use_metadata)
    closed_set_evaluation_loader = torch.utils.data.DataLoader(closedset_dataset_test, batch_size=batch_size,
                                                               shuffle=False, num_workers=4,
                                                               collate_fn=collate_fn,
                                                               timeout=worker_timeout_s)

    # generate embeddings and labels to select the model
    embeddings = []
    labels = []
    print("generate closed set embeddings and labels to select the model")
    for data in tqdm(closed_set_selection_loader):
        images, _ = data
        # squeeze to use MLP instead of CNN
        outputs = get_model_features(images, model, device).detach().cpu().numpy().squeeze()
        embeddings.append(outputs)
        labels.extend([closedset_label] * outputs.shape[0])
    print("generate open set embeddings and labels to select the model")
    for data in tqdm(open_set_selection_loader):
        images, _ = data
        # squeeze to use MLP instead of CNN
        outputs = get_model_features(images, model, device).detach().cpu().numpy().squeeze()
        embeddings.append(outputs)
        labels.extend([openset_label] * outputs.shape[0])
    embeddings = np.concatenate(embeddings)
    embeddings = torch.tensor(embeddings).to(device)

    # create the discriminator model
    discriminator_model = Discriminator(nc=nc, hidden_dim=hidden_dim).to(device)
    discriminator_model.eval()

    # choose the best model based on ROC-AUC
    best_discriminator_model = None
    best_discriminator_path = None
    best_rocauc = float("-inf")
    with torch.no_grad():
        print("selecting discriminator")
        for i, discriminator_path in enumerate(discriminators):
            print(f"discriminator {i + 1} of {len(discriminators)}")
            discriminator_model.load_state_dict(torch.load(discriminator_path), strict=False)
            preds = []
            for embedding in tqdm(embeddings):
                y_pred_proba = discriminator_model(embedding.unsqueeze(0)).detach().cpu().numpy()
                preds.append(y_pred_proba)
            preds = np.array(preds).squeeze()

            rocauc = roc_auc_score(labels, preds)
            print(f"roc-auc for discriminator {i + 1}: {rocauc}")
            if rocauc > best_rocauc:
                best_rocauc = rocauc
                best_discriminator_path = discriminator_path
                best_discriminator_model = discriminator_model
                print(f"new best discriminator: {best_discriminator_path} with selection roc-auc {best_rocauc}")
    print(f"best discriminator: {best_discriminator_path} with selection roc-auc {best_rocauc}")

    # generate embeddings and labels to evaluate the model
    embeddings = []
    labels = []
    print("generate closed set embeddings and labels to evaluate the model on test set")
    for data in tqdm(closed_set_evaluation_loader):
        images, _ = data
        # squeeze to use MLP instead of CNN
        outputs = get_model_features(images, model, device).detach().cpu().numpy().squeeze()
        embeddings.append(outputs)
        labels.extend([closedset_label] * outputs.shape[0])
    print("generate open set embeddings and labels to evaluate the model on test set")
    for data in tqdm(open_set_evaluation_loader):
        images, _ = data
        # squeeze to use MLP instead of CNN
        outputs = get_model_features(images, model, device).detach().cpu().numpy().squeeze()
        embeddings.append(outputs)
        labels.extend([openset_label] * outputs.shape[0])
    embeddings = np.concatenate(embeddings)
    embeddings = torch.tensor(embeddings).to(device)

    # evaluate selected model
    print("evaluating model on open dataset")
    preds = []
    with torch.no_grad():
        for embedding in tqdm(embeddings):
            y_pred_proba = best_discriminator_model(embedding.unsqueeze(0)).detach().cpu().numpy()
            preds.append(y_pred_proba)
    preds = np.array(preds).squeeze()

    # evaluate roc-auc
    eval_rocauc = roc_auc_score(labels, preds)
    print((f"best discriminator: {best_discriminator_path} "
           f"with selection roc-auc {best_rocauc} "
           f"and evaluation roc-auc {eval_rocauc}"))

    print("saving best discriminator to the embedder model directory")
    torch.save(best_discriminator_model.state_dict(), experiment_dir / "openganfea_model.pth")


def load_model_for_inference(device, experiment_dir, model_id, n_classes):
    model = build_model(
        pretrained=False,  # doesn't matter since the weights will be updated by the checkpoint
        fine_tune=False,  # we don't need to unfreeze any weights
        num_classes=n_classes,
        dropout_rate=0.5,  # doesn't matter since embeddings will be created in eval on a fixed model
        model_id=model_id,
    ).to(device)
    model_file_path = str(experiment_dir / f"model.pth")
    checkpoint = torch.load(model_file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# def create_feature_extractor_from_model(model, embedder_layer_offset):
#     feature_extractor = torch.nn.Sequential(*list(model.children())[:embedder_layer_offset])
#     return feature_extractor


if __name__ == "__main__":
    # using this instead of @hydra.main decorator so main function can be called from elsewhere
    with initialize(version_base=None, config_path="conf", job_name="evaluate"):
        cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))
    choose_best_discriminator(cfg)
