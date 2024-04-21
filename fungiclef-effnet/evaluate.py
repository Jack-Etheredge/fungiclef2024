import json

import pandas as pd
import numpy as np
import os

from hydra import compose, initialize
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from torchvision.transforms import v2
from PIL import Image
import torch
from pathlib import Path
from PIL import ImageFile
from scipy.special import softmax
from scipy.stats import entropy

from closedset_model import build_model
from competition_metrics import evaluate
from create_opengan_discriminator import train_and_select_discriminator, create_composite_model
from temperature_scaling import ModelWithTemperature
from temp_scale_model import create_temperature_scaled_model

np.set_printoptions(precision=5)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# TODO: pull this transformation from the datasets module
TRANSFORMS = v2.Compose([
    v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
    v2.CenterCrop(224),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_threshold(max_prob_list, k):
    # Calculate Q1 and Q3 quartiles
    q1 = np.quantile(max_prob_list, 0.25)
    q3 = np.quantile(max_prob_list, 0.75)

    # Get the Interquartile Range (IQR)
    iqr = q3 - q1

    # Calculate Minimum threshold
    min_th = q1 - k * iqr

    return min_th


class PytorchWorker:
    """Run inference using PyTorch."""

    def __init__(self, model_path: str, number_of_categories: int = 1604, temp_scaling=False,
                 model_id="efficientnet_b0", use_timm=True, device="cpu"):
        self.number_of_categories = number_of_categories  # must be set before calling _load_model
        self.temp_scaling = temp_scaling  # must be set before calling _load_model
        self.model_id = model_id  # must be set before calling _load_model
        self.use_timm = use_timm  # must be set before calling _load_model
        self.model = self._load_model(model_path)
        self.transforms = TRANSFORMS
        self.device = device

    def _load_model(self, model_path):
        print("Setting up Pytorch Model")
        # model = models.efficientnet_b0()
        # model.classifier[1] = nn.Linear(in_features=1280, out_features=self.number_of_categories)
        model = build_model(
            model_id=self.model_id,
            pretrained=False,
            fine_tune=False,
            num_classes=self.number_of_categories,
            # this is all that matters. everything else will be overwritten by checkpoint state
            dropout_rate=0.5,
            use_timm=self.use_timm,
        ).to(self.device)
        model_ckpt = torch.load(model_path, map_location=self.device)
        if self.temp_scaling:
            model = ModelWithTemperature(model)
            model.load_state_dict(model_ckpt)
        else:
            model.load_state_dict(model_ckpt['model_state_dict'])

        return model.to(self.device).eval()

    def predict_image(self, image: np.ndarray) -> list():
        """Run inference using ONNX runtime.

        :param image: Input image as numpy array.
        :return: A list with logits and confidences.
        """

        logits = self.model(self.transforms(image).unsqueeze(0).to(self.device))

        return logits.tolist()


def make_submission(test_metadata, model_path, output_csv_path, images_root_path, temp_scaling,
                    opengan, model_id, use_timm, cfg):
    """Make submission file"""
    # TODO: use the dataloader with a larger batch size to speed up inference
    # TODO: pull this transformation from the datasets module

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using devide: {device}")

    # Consolidate the model building logic
    if opengan:
        model = create_composite_model(cfg).to(device)
    else:
        model = PytorchWorker(model_path, temp_scaling=temp_scaling, model_id=model_id, use_timm=use_timm,
                              device=device)

    # this allows use on both validation and test
    test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)
    test_metadata = test_metadata.drop_duplicates("observation_id", keep="first")

    predictions = []
    max_probas = []
    entropy_scores = []
    image_paths = test_metadata["image_path"]
    with torch.no_grad():
        for image_path in tqdm(image_paths):
            try:
                image_path = os.path.join(images_root_path, image_path)
                test_image = Image.open(image_path).convert("RGB")
                if opengan:
                    pred, proba = model(TRANSFORMS(test_image).unsqueeze(0).to(device)).item()
                    # TODO: .item() solution is brittle and won't work once multiple images are fed in a batch
                    max_probas.append(proba.item())
                    predictions.append(pred.item())
                else:
                    logits = model.predict_image(test_image)
                    probas = softmax(logits)
                    entropy_score = entropy(probas.squeeze())
                    predictions.append(np.argmax(probas))
                    max_probas.append(np.max(probas))
                    entropy_scores.append(entropy_score)
            except Exception as e:
                print(f"issue with image {image_path}: {e}")
                predictions.append(-1)
                max_probas.append(-1)

    test_metadata.loc[:, "class_id"] = predictions
    if opengan:
        test_metadata.loc[:, "opengan_proba"] = max_probas
        keep_cols = ["observation_id", "class_id", "opengan_proba"]
    else:
        test_metadata.loc[:, "max_proba"] = max_probas
        test_metadata.loc[:, "entropy"] = entropy_scores
        keep_cols = ["observation_id", "class_id", "max_proba", "entropy"]
    test_metadata[keep_cols].to_csv(output_csv_path, index=None)


def evaluate_experiment(cfg, experiment_id, temperature_scaling=False, opengan=False, from_outputs=False):
    experiment_dir = Path("model_checkpoints") / experiment_id
    model_pre = "opengan_composite_" if opengan else ""
    model_ext = "_with_temperature" if temperature_scaling else ""
    model_file = f"{model_pre}model{model_ext}.pth"
    model_path = str(experiment_dir / model_file)
    ext = "_opengan" if opengan else ""
    ext += "_ts" if temperature_scaling else ""
    predictions_output_csv_path = str(experiment_dir / f"submission_fine_tuned_thresholding{ext}.csv")
    if opengan:
        predictions_with_unknown_output_csv_path = predictions_output_csv_path
    else:
        predictions_with_unknown_output_csv_path = str(
            experiment_dir / f"submission_with_unknowns_fine_tuned_thresholding{ext}.csv")

    data_dir = Path('__file__').parent.absolute().parent / "data" / "DF21"
    metadata_file_path = "../metadata/FungiCLEF2023_val_metadata_PRODUCTION.csv"
    test_metadata = pd.read_csv(metadata_file_path)

    # TODO: set thresholds on val
    # TODO: evaluate on test - if the test set dataloader doesn't shuffle, should be able to use it for submission.csv
    # TODO: remove train and val unknown examples from the ground truth for submission.csv before running evaluation
    open_set_val_loader = None
    closed_set_val_loader = None
    open_set_test_loader = None
    closed_set_test_loader = None

    # Make predictions if they need to be made. Skip if already computed.
    if not from_outputs:
        make_submission(
            test_metadata=test_metadata,
            model_path=model_path,
            images_root_path=data_dir,
            output_csv_path=predictions_output_csv_path,
            temp_scaling=temperature_scaling,
            opengan=opengan,
            model_id=model_id,
            use_timm=use_timm,
            cfg=cfg,
        )

    # TODO: logic is fundamentally different for openGAN evaluation since no thresholds are needed
    if opengan:
        scores_output_path = str(experiment_dir / f"competition_metrics_scores_opengan.json")

        # Generate metrics from predictions
        test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)
        test_metadata.drop_duplicates("observation_id", keep="first", inplace=True)
        y_true = test_metadata["class_id"].values
        submission_df = pd.read_csv(predictions_output_csv_path)
        submission_df.drop_duplicates("observation_id", keep="first", inplace=True)
        y_pred = np.copy(submission_df["class_id"].values)

        # TODO: deduplicate logic
        # best_threshold = set_best_threshold()

        homebrewed_scores = calc_homebrewed_scores(y_true, y_pred, y_pred)
        save_metrics(homebrewed_scores, metadata_file_path, predictions_with_unknown_output_csv_path,
                     scores_output_path)

    else:
        for thresholding_method in ["_softmax", "_entropy"]:

            ext += thresholding_method
            metrics_output_csv_path = str(experiment_dir / f"threshold_scores{ext}.csv")
            scores_output_path = str(experiment_dir / f"competition_metrics_scores{ext}.json")

            # Generate metrics from predictions
            test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)
            test_metadata.drop_duplicates("observation_id", keep="first", inplace=True)
            y_true = test_metadata["class_id"].values
            submission_df = pd.read_csv(predictions_output_csv_path)
            submission_df.drop_duplicates("observation_id", keep="first", inplace=True)
            if thresholding_method == "_softmax":
                y_proba = submission_df["max_proba"].values
                threshold_range = np.arange(0, 1, 0.05)
            elif thresholding_method == "_entropy":
                # taking -entropy so that threshold logic will work for softmax or entropy
                # since high softmax proba == known, but high entropy == unknown
                y_proba = -submission_df["entropy"].values
                threshold_range = np.arange(y_proba.min(), y_proba.max(), 0.05)
            else:
                raise ValueError(f"unrecognized thresholding method {thresholding_method}")

            scores = []
            thresholds = []

            y_pred = np.copy(submission_df["class_id"].values)
            best_threshold, threshold = None, None
            score = f1_score(y_true, y_pred, average='macro')
            best_f1 = score
            thresholds.append(threshold)
            scores.append(score)
            print(threshold, score)

            for threshold in threshold_range:
                y_pred = np.copy(submission_df["class_id"].values)
                y_pred[y_proba < threshold] = -1
                score = f1_score(y_true, y_pred, average='macro')
                if score > best_f1:
                    best_f1 = score
                    best_threshold = threshold
                print(threshold, score)
                thresholds.append(threshold)
                scores.append(score)
            for k in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
                threshold = get_threshold(y_proba, k)
                y_pred = np.copy(submission_df["class_id"].values)
                y_pred[y_proba < threshold] = -1
                score = f1_score(y_true, y_pred, average='macro')
                if score > best_f1:
                    best_f1 = score
                    best_threshold = threshold
                print(f"iqr_k{k}", threshold, score)
                thresholds.append(threshold)
                scores.append(score)
            threshold_scores = pd.DataFrame()
            threshold_scores['threshold'] = thresholds
            threshold_scores['f1'] = scores
            threshold_scores.sort_values('f1').to_csv(metrics_output_csv_path, index=False)

            # reset y_pred and create y_pred_w_unknown using the best threshold before calculating homebrewed_scores
            y_pred = np.copy(submission_df["class_id"].values)
            y_pred_w_unknown = y_pred.copy()
            y_pred_w_unknown[y_proba < best_threshold] = -1
            print("best threshold")
            homebrewed_scores = calc_homebrewed_scores(y_true, y_pred, y_pred_w_unknown)

            # make and save the unknown output csv
            submission_df.loc[:, "class_id"] = y_pred_w_unknown
            submission_df.to_csv(predictions_with_unknown_output_csv_path, index=False)

            save_metrics(homebrewed_scores, metadata_file_path, predictions_with_unknown_output_csv_path,
                         scores_output_path)


def save_metrics(homebrewed_scores, metadata_file_path, predictions_with_unknown_output_csv_path, scores_output_path):
    # add additional competition metricspredictions_with_unknown_output_csv_path
    competition_metrics_scores = evaluate(
        test_annotation_file=metadata_file_path,
        user_submission_file=predictions_with_unknown_output_csv_path,
        phase_codename="prediction-based",
    )
    # deduplicate and flatten results
    competition_metrics_scores = competition_metrics_scores["submission_result"]
    # remove this metric since it's not used this year
    del competition_metrics_scores["Track 4: Classification Error with Special Cost for Unknown"]
    competition_metrics_scores.update(homebrewed_scores)
    with open(scores_output_path, "w") as f:
        json.dump(competition_metrics_scores, f)


def calc_homebrewed_scores(y_true, y_pred, y_pred_w_unknown):
    homebrewed_scores = dict()

    # check disparity between training validation accuracy and validation set accuracy
    y_pred_on_known = y_pred[~(y_true == -1)]
    y_true_on_known = y_true[~(y_true == -1)]
    balanced_accuracy_known_classes = balanced_accuracy_score(y_true_on_known, y_pred_on_known)
    homebrewed_scores['balanced_accuracy_known_classes'] = balanced_accuracy_known_classes
    print("balanced accuracy on known classes:", balanced_accuracy_known_classes)
    accuracy_known_classes = accuracy_score(y_true_on_known, y_pred_on_known)
    homebrewed_scores['accuracy_known_classes'] = accuracy_known_classes
    print("unbalanced accuracy on known classes:", accuracy_known_classes)

    # check unknown vs known binary f1 using best threshold
    y_true_known_vs_unknown = y_true.copy()
    y_true_known_vs_unknown[~(y_true == -1)] = 1  # 1 is known
    y_true_known_vs_unknown[(y_true == -1)] = 0  # 0 is unknown
    y_pred_known_vs_unknown = y_pred_w_unknown.copy()
    y_pred_known_vs_unknown[~(y_pred_w_unknown == -1)] = 1
    y_pred_known_vs_unknown[(y_pred_w_unknown == -1)] = 0
    f1_binary_known_vs_unknown = f1_score(y_true_known_vs_unknown, y_pred_known_vs_unknown, average='binary')
    homebrewed_scores['f1_binary_known_vs_unknown'] = f1_binary_known_vs_unknown
    print("F1 binary known vs unknown:", f1_binary_known_vs_unknown)
    f1_macro_known_vs_unknown = f1_score(y_true_known_vs_unknown, y_pred_known_vs_unknown, average='macro')
    homebrewed_scores['f1_macro_known_vs_unknown'] = f1_macro_known_vs_unknown
    print("F1 macro known vs unknown:", f1_macro_known_vs_unknown)
    roc_auc_known_vs_unknown = roc_auc_score(y_true_known_vs_unknown, y_pred_known_vs_unknown)
    homebrewed_scores['roc_auc_known_vs_unknown'] = roc_auc_known_vs_unknown
    print("roc_auc_known_vs_unknown:", roc_auc_known_vs_unknown)

    return homebrewed_scores


if __name__ == "__main__":
    # TODO: address issue of evaluating on all the unknowns despite some being used for train and val in fine-tuning
    # TODO: set threshold on val, evaluate on test (this should also solve the above given a val loader for unknowns)
    # TODO: use dataloader with a larger batch size to speed up inference
    # TODO: pull test transformations from the datasets module

    with initialize(version_base=None, config_path="conf", job_name="evaluate"):
        cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))

    experiment_id = cfg["evaluate"]["experiment_id"]
    use_timm = cfg["evaluate"]["use_timm"]
    model_id = cfg["evaluate"]["model_id"]

    outputs_precomputed = True
    print(f"evaluating experiment {experiment_id}")
    evaluate_experiment(cfg=cfg, experiment_id=experiment_id, from_outputs=outputs_precomputed)
    print("creating temperature scaled model")
    create_temperature_scaled_model(cfg)
    print(f"evaluating experiment {experiment_id} with temperature scaling")
    evaluate_experiment(cfg=cfg, experiment_id=experiment_id, temperature_scaling=True,
                        from_outputs=outputs_precomputed)

    # outputs_precomputed = False
    # print("training openGAN model")
    # # train_and_select_discriminator(cfg)
    # print(f"evaluating experiment {experiment_id} with openGAN")
    # evaluate_experiment(cfg=cfg, experiment_id=experiment_id, opengan=True, from_outputs=outputs_precomputed)
