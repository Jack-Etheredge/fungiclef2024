"""
Choose best discriminator checkpoint for openGANfea
- closed set is the dataset used to train the closed set classifier that generates the embeddings for openGANfea
- open set is the dataset used to train openGANfea
- this example uses imagenet1K pretrained resnet18 as the classification network, so imagenet is the closed set dataset
- this example uses cifar10 as the cross-dataset open set, so cifar10 is the open set dataset
"""
from pathlib import Path

import numpy as np

import torch
import torchvision
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset

from torchvision.transforms import v2, InterpolationMode
from tqdm import tqdm

from opengan.models import Discriminator
from opengan.utils import CustomImageDataset, collate_fn, get_train_val_datasets


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    # HYPERPARAMETERS
    batch_size = 128
    # TODO: move the common elements to a config
    nc = 512
    hidden_dim = 64
    eval_fraction = 0.1
    n_workers = 16
    image_size = 224
    max_total_examples = 1000
    project_name = "opengan_fea_Res18sc_mlpGAN_dlr_1e-6_open1closed0_100_examples_no_gen"  # we save all the checkpoints in this directory
    experiment_dir = Path('__file__').parent.absolute() / "opengan_exp"
    discriminator_dir = experiment_dir / project_name
    openset_label = 1.
    closedset_label = 0.

    # set device, which gpu to use.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device {device}")

    # filtering for "discriminator" in model state checkpoint files necessary because generator and discriminator
    # states were saved in the same directory
    discriminators = [dis for dis in discriminator_dir.iterdir() if
                      dis.suffix == ".pth" and "discriminator" in dis.stem]

    model = torchvision.models.resnet18(weights='DEFAULT').to(device)  # model is trained on Imagenet1K
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    transforms = v2.Compose([
        v2.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # generate embeddings for CIFAR10 since it's not in the training set for Imagenet1K
    datasets_dir = Path("~").expanduser().absolute() / "datasets"
    imagenet_dir = datasets_dir / "imagenet" / "val_images"
    cifar10_dir = datasets_dir / "cifar10"
    mnist_dir = datasets_dir / "mnist"
    imagenet_dataset = CustomImageDataset(img_dir=imagenet_dir, transform=transforms)
    cifar10_dataset = torchvision.datasets.CIFAR10(root=cifar10_dir, train=False,
                                                   download=True, transform=transforms)
    mnist_dataset = torchvision.datasets.MNIST(root=mnist_dir, train=False,
                                               download=True, transform=transforms)
    selection_imagenet_ds, eval_imagenet_ds = get_train_val_datasets(imagenet_dataset, max_total_examples,
                                                                     eval_fraction)
    selection_cifar10_ds, eval_cifar10_ds = get_train_val_datasets(cifar10_dataset, max_total_examples, eval_fraction)
    # set up the model selection datasets (imagenet vs something other than cifar10)
    closed_set_selection_loader = torch.utils.data.DataLoader(selection_imagenet_ds, batch_size=batch_size,
                                                              shuffle=False, num_workers=n_workers,
                                                              collate_fn=collate_fn)
    open_set_selection_loader = torch.utils.data.DataLoader(selection_cifar10_ds, batch_size=batch_size,
                                                            shuffle=False, num_workers=n_workers, collate_fn=collate_fn)
    # set up the model evaluation datasets (different split of imagenet vs something other than cifar10)
    closed_set_evaluation_loader = torch.utils.data.DataLoader(eval_imagenet_ds, batch_size=batch_size,
                                                               shuffle=False, num_workers=n_workers,
                                                               collate_fn=collate_fn)
    open_set_evaluation_loader = torch.utils.data.DataLoader(eval_cifar10_ds, batch_size=batch_size,
                                                             shuffle=False, num_workers=n_workers,
                                                             collate_fn=collate_fn)
    open_set_two_evaluation_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size,
                                                                 shuffle=False, num_workers=n_workers,
                                                                 collate_fn=collate_fn)

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
    print("generate open set embeddings and labels to select the model")
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

    # generate embeddings and labels to evaluate the model on a third dataset
    embeddings = []
    labels = []
    print("generate open set embeddings and labels to select the model")
    for data in tqdm(closed_set_evaluation_loader):
        images, _ = data
        images = images.to(device)
        outputs = feature_extractor(images).detach().cpu().numpy().squeeze()  # squeeze to use MLP instead of CNN
        embeddings.append(outputs)
        labels.extend([closedset_label] * outputs.shape[0])
    print("generate open set embeddings and labels to select the model")
    for data in tqdm(open_set_two_evaluation_loader):
        images, _ = data
        images = images.to(device)
        outputs = feature_extractor(images).detach().cpu().numpy().squeeze()  # squeeze to use MLP instead of CNN
        embeddings.append(outputs)
        labels.extend([openset_label] * outputs.shape[0])
    embeddings = np.concatenate(embeddings)
    embeddings = torch.tensor(embeddings).to(device)

    # evaluate selected model
    print("evaluating model on additional (unseen) open dataset")
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
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_defaults_leakyrelu_adam_10x_relative_lr_tanh_to_leakyrelu_open0closed1_100_examples/epoch-1-discriminator.pth with selection roc-auc 0.8504518518518519 and evaluation roc-auc 0.8550000000000001
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_defaults_leakyrelu_adam_10x_relative_lr_tanh_to_leakyrelu_open0closed1_100_examples/epoch-1-discriminator.pth with selection roc-auc 0.8504518518518519 and evaluation roc-auc 0.8818729999999999
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_defaults_leakyrelu_adam_10x_relative_lr_tanh_to_leakyrelu_open1closed0_100_examples/epoch-2-discriminator.pth with selection roc-auc 0.8651820987654321 and evaluation roc-auc 0.8753999999999998
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_defaults_leakyrelu_adam_10x_relative_lr_tanh_to_leakyrelu_open1closed0_100_examples/epoch-2-discriminator.pth with selection roc-auc 0.8651820987654321 and evaluation roc-auc 0.9086049999999999
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_defaults_leakyrelu_adam_tanh_to_leakyrelu_open0closed1_100_examples/epoch-1-discriminator.pth with selection roc-auc 0.937320987654321 and evaluation roc-auc 0.9078
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_defaults_leakyrelu_adam_tanh_to_leakyrelu_open0closed1_100_examples/epoch-1-discriminator.pth with selection roc-auc 0.937320987654321 and evaluation roc-auc 0.9551200000000001
    # 1 for open set?
    # lower lr?
    # equal lr (better than 10x to generator)?
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_lr_1e-5_open0closed1_100_examples/epoch-10-discriminator.pth with selection roc-auc 0.9435703703703704 and evaluation roc-auc 0.7775000000000001
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_lr_1e-5_open0closed1_100_examples/epoch-10-discriminator.pth with selection roc-auc 0.9435703703703704 and evaluation roc-auc 0.842269
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_lr_1e-4_open1closed0_100_examples/epoch-1-discriminator.pth with selection roc-auc 0.9214716049382715 and evaluation roc-auc 0.7641
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_lr_1e-4_open1closed0_100_examples/epoch-1-discriminator.pth with selection roc-auc 0.9214716049382715 and evaluation roc-auc 0.801161
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_lr_1e-3_open1closed0_100_examples/epoch-2-discriminator.pth with selection roc-auc 0.8843407407407408 and evaluation roc-auc 0.7222000000000001
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_lr_1e-3_open1closed0_100_examples/epoch-2-discriminator.pth with selection roc-auc 0.8843407407407408 and evaluation roc-auc 0.721201
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_lr_1e-3_open0closed1_100_examples/epoch-1-discriminator.pth with selection roc-auc 0.832206172839506 and evaluation roc-auc 0.8015
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_lr_1e-3_open0closed1_100_examples/epoch-1-discriminator.pth with selection roc-auc 0.832206172839506 and evaluation roc-auc 0.868776
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_lr_1e-4_open1closed0_100_examples/epoch-1-discriminator.pth with selection roc-auc 0.9133913580246913 and evaluation roc-auc 0.745
    # best discriminator: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_lr_1e-4_open1closed0_100_examples/epoch-1-discriminator.pth with selection roc-auc 0.9133913580246913 and evaluation roc-auc 0.7925059999999999

    # comparing with and without generator
    # dlr_1e-5_glr_1e-3
    # with generator
    # best discriminator [same open set as training]: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_dlr_1e-5_glr_1e-3_open1closed0_100_examples/epoch-1-discriminator.pth with selection roc-auc 0.8610993827160495 and evaluation roc-auc 0.8275
    # best discriminator [novel open set]: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_dlr_1e-5_glr_1e-3_open1closed0_100_examples/epoch-1-discriminator.pth with selection roc-auc 0.8610993827160495 and evaluation roc-auc 0.91899
    # without generator
    # best discriminator [same open set as training]: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_dlr_1e-5_glr_1e-3_open1closed0_100_examples_no_gen/epoch-7-discriminator.pth with selection roc-auc 0.9752641975308642 and evaluation roc-auc 0.9806
    # best discriminator [novel open set]: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_dlr_1e-5_glr_1e-3_open1closed0_100_examples_no_gen/epoch-7-discriminator.pth with selection roc-auc 0.9752641975308642 and evaluation roc-auc 0.991023
    # dlr_1e-5_glr_1e-5
    # with generator
    # best discriminator [same open set as training]: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_dlr_1e-5_glr_1e-5_open1closed0_100_examples/epoch-3-discriminator.pth with selection roc-auc 0.9283827160493828 and evaluation roc-auc 0.8421
    # best discriminator [novel open set]: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_dlr_1e-5_glr_1e-5_open1closed0_100_examples/epoch-3-discriminator.pth with selection roc-auc 0.9283827160493828 and evaluation roc-auc 0.868428
    # dlr_1e-6_glr_1e-6
    # with generator
    # best discriminator [same open set as training]: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_dlr_1e-6_glr_1e-6_open1closed0_100_examples/epoch-25-discriminator.pth with selection roc-auc 0.9318111111111111 and evaluation roc-auc 0.9517
    # best discriminator [novel open set]: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_dlr_1e-6_glr_1e-6_open1closed0_100_examples/epoch-25-discriminator.pth with selection roc-auc 0.9318111111111111 and evaluation roc-auc 0.96971
    # without generator
    # best discriminator [same open set as training]: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_dlr_1e-6_open1closed0_100_examples_no_gen/epoch-23-discriminator.pth with selection roc-auc 0.9604679012345679 and evaluation roc-auc 0.9459
    # best discriminator [novel open set]: /home/jack/projects/fungiclef2024/opengan/opengan_exp/opengan_fea_Res18sc_mlpGAN_dlr_1e-6_open1closed0_100_examples_no_gen/epoch-23-discriminator.pth with selection roc-auc 0.9604679012345679 and evaluation roc-auc 0.993128
