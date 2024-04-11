"""
Choose best discriminator checkpoint for openGANfea
- closed set is the dataset used to train the closed set classifier that generates the embeddings for openGANfea
- open set is the dataset used to train openGANfea
- this example uses imagenet1K pretrained resnet18 as the classification network, so imagenet is the closed set dataset
- this example uses cifar10 as the cross-dataset open set, so cifar10 is the open set dataset
"""
from pathlib import Path

import torch
import torchvision
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from paths import CHECKPOINT_DIR
from tqdm import tqdm

from closedset_model import build_model
from datasets import get_openset_datasets, get_datasets, get_closedset_test_dataset, collate_fn
from openset_recognition_models import Discriminator


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # TODO: move the common elements to a config
    nc = 1280  # pull from hydra
    hidden_dim = 64
    eval_fraction = 0.1
    max_total_examples = 10_000  # TODO: reincorporate this
    embedder_experiment_id = cfg["open-set-recognition"]["embedder_experiment_id"]
    openset_label = 0.
    closedset_label = 1.
    project_name = f"{embedder_experiment_id}_dlr_1e-6_glr_1e-6_open{int(openset_label)}closed{int(closedset_label)}"  # we save all the checkpoints in this directory
    exp_dir = Path(
        '__file__').parent.absolute() / "openset_recognition_discriminators"  # experiment directory, used for reading the init model
    discriminator_dir = exp_dir / project_name

    embedder_experiment_id = cfg["open-set-recognition"]["embedder_experiment_id"]
    embedder_layer_offset = cfg["open-set-recognition"]["embedder_layer_offset"]
    batch_size = cfg["open-set-recognition"]["batch_size"]
    n_workers = cfg["open-set-recognition"]["n_workers"]
    openset_embeddings_name = cfg["open-set-recognition"]["openset_embeddings_name"]
    closedset_embeddings_name = cfg["open-set-recognition"]["closedset_embeddings_name"]
    image_size = cfg["open-set-recognition"]["image_size"]
    openset_n_train = cfg["open-set-recognition"]["openset_n_train"]
    openset_n_val = cfg["open-set-recognition"]["openset_n_val"]
    pretrained = cfg["open-set-recognition"]["pretrained"]

    # for closed set dataset
    undersample = cfg["train"]["undersample"]
    oversample = cfg["train"]["oversample"]
    equal_undersampled_val = cfg["train"]["equal_undersampled_val"]
    oversample_prop = cfg["train"]["oversample_prop"]
    validation_frac = cfg["train"]["validation_frac"]

    # for constructing model
    n_classes = cfg["train"]["n_classes"]

    # set device, which gpu to use.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"using device {device}")

    # filtering for "discriminator" in model state checkpoint files necessary because generator and discriminator
    # states were saved in the same directory
    discriminators = [dis for dis in discriminator_dir.iterdir() if
                      dis.suffix == ".pth" and "discriminator" in dis.stem]

    print("constructing embedder (closed set classifier outputting from penultimate layer)")
    model = build_model(
        pretrained=pretrained,
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
    feature_extractor = torch.nn.Sequential(*list(model.children())[:embedder_layer_offset])

    # transforms = v2.Compose([
    #     v2.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
    #     v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    #     v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    _, openset_dataset_val, openset_dataset_test = get_openset_datasets(pretrained=pretrained, image_size=image_size,
                                                                        n_train=openset_n_train, n_val=openset_n_val)
    open_set_selection_loader = torch.utils.data.DataLoader(openset_dataset_val, batch_size=batch_size,
                                                            shuffle=False, num_workers=4, collate_fn=collate_fn,
                                                            timeout=120)  # TODO: move to hydra
    open_set_evaluation_loader = torch.utils.data.DataLoader(openset_dataset_test, batch_size=batch_size,
                                                             shuffle=False, num_workers=4,
                                                             collate_fn=collate_fn,
                                                             timeout=120)  # TODO: move to hydra
    _, closedset_dataset_val, _ = get_datasets(pretrained=pretrained, image_size=image_size,
                                               validation_frac=validation_frac,
                                               oversample=oversample, undersample=undersample,
                                               oversample_prop=oversample_prop,
                                               equal_undersampled_val=equal_undersampled_val)
    closed_set_selection_loader = torch.utils.data.DataLoader(closedset_dataset_val, batch_size=batch_size,
                                                              shuffle=False, num_workers=4,
                                                              collate_fn=collate_fn,
                                                              timeout=120)  # TODO: move to hydra
    closedset_dataset_test = get_closedset_test_dataset(pretrained, image_size)
    closed_set_evaluation_loader = torch.utils.data.DataLoader(closedset_dataset_test, batch_size=batch_size,
                                                               shuffle=False, num_workers=4,
                                                               collate_fn=collate_fn,
                                                               timeout=120)  # TODO: move to hydra

    # generate embeddings and labels to select the model
    embeddings = []
    labels = []
    print("generate closed set embeddings and labels to select the model")
    for data in tqdm(closed_set_selection_loader):
        images, _ = data
        images = images.to(device)
        outputs = feature_extractor(images).detach().cpu().numpy().squeeze()  # squeeze to use MLP instead of CNN
        embeddings.append(outputs)
        labels.extend([closedset_label] * outputs.shape[0])
    print("generate open set embeddings and labels to select the model")
    for data in tqdm(open_set_selection_loader):
        images, _ = data
        images = images.to(device)
        outputs = feature_extractor(images).detach().cpu().numpy().squeeze()  # squeeze to use MLP instead of CNN
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
    print("generate closed set embeddings and labels to select the model")
    for data in tqdm(closed_set_evaluation_loader):
        images, _ = data
        images = images.to(device)
        outputs = feature_extractor(images).detach().cpu().numpy().squeeze()  # squeeze to use MLP instead of CNN
        embeddings.append(outputs)
        labels.extend([closedset_label] * outputs.shape[0])
    print("generate open set embeddings and labels to select the model")
    for data in tqdm(open_set_evaluation_loader):
        images, _ = data
        images = images.to(device)
        outputs = feature_extractor(images).detach().cpu().numpy().squeeze()  # squeeze to use MLP instead of CNN
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


if __name__ == "__main__":
    main()
    # discriminator only:
    # best discriminator: seesaw_04_02_2024_dlr_1e-6_open1closed0_no_gen/epoch-11-discriminator.pth
    #   selection roc-auc 0.6117849906483791 and evaluation roc-auc 0.5604385963920865
    # GAN open 1 closed 0:
    # best discriminator: seesaw_04_02_2024_dlr_1e-6_open1closed0/epoch-10-discriminator.pth
    #   selection roc-auc 0.6020331982543641 and evaluation roc-auc 0.5775920077303734
    # GAN open 0 closed 1:
    # best discriminator: seesaw_04_02_2024_dlr_1e-6_glr_1e-6_open0closed1/epoch-20-discriminator.pth
    #   selection roc-auc 0.6231694201995013 and evaluation roc-auc 0.6029468184269686
    # softmax thresholding (overly optimistic; set threshold on the test set): roc-auc 0.5447452440605403
    # temp scaled softmax thresholding (overly optimistic; set threshold on the test set): roc-auc 0.544770863626587
