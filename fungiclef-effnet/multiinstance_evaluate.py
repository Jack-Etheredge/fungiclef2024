import json

import pandas as pd
import numpy as np
import os

from hydra import compose, initialize
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from PIL import Image
import torch
from pathlib import Path
from PIL import ImageFile
from scipy.special import softmax
from scipy.stats import entropy

from closedset_model import build_model
from competition_metrics import evaluate
from create_opengan_discriminator import train_and_select_discriminator, create_composite_model
from datasets import get_valid_transform, encode_metadata_row
from paths import METADATA_DIR, DATA_DIR
from temperature_scaling import ModelWithTemperature
from temp_scale_model import create_temperature_scaled_model
from utils import copy_config

np.set_printoptions(precision=5)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PytorchWorker:
    """Run inference using PyTorch."""

    def __init__(self, model_path: str, number_of_categories: int = 1604, temp_scaling=False,
                 model_id="efficientnet_b0", device="cpu"):

        ########################################
        # must be set before calling _load_model
        self.number_of_categories = number_of_categories
        self.temp_scaling = temp_scaling
        self.model_id = model_id
        self.device = device
        ########################################

        self.transforms = TRANSFORMS
        # most other attributes must be set before calling _load_model, so call last
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        print("Setting up Pytorch Model")
        # model = models.efficientnet_b0()
        # model.classifier[1] = nn.Linear(in_features=1280, out_features=self.number_of_categories)
        model = build_model(
            model_id=self.model_id,
            pretrained=False,
            fine_tune=False,
            # this is all that matters. everything else will be overwritten by checkpoint state
            num_classes=self.number_of_categories,
            dropout_rate=0.0,  # doesn't matter for eval since pytorch will use identity at inference
        ).to(self.device)
        model_ckpt = torch.load(model_path, map_location=self.device)
        if self.temp_scaling:
            model = ModelWithTemperature(model)
            model.load_state_dict(model_ckpt)
        else:
            model.load_state_dict(model_ckpt['model_state_dict'])

        return model.to(self.device).eval()

    def predict_image(self, image: np.ndarray, metadata_row=None) -> list():
        """Run inference using ONNX runtime.

        :param image: Input image as numpy array.
        :param metadata_row: Input metadata encoded for model.
        :return: A list with logits and confidences.
        """

        transformed_images = self.transforms(image).unsqueeze(0).to(self.device)

        if metadata_row is not None:
            transformed_metadata = metadata_row.unsqueeze(0).to(self.device)
            logits = self.model(transformed_images, transformed_metadata)
        else:
            logits = self.model(transformed_images)

        return logits.tolist()


@torch.no_grad
def make_submission(test_metadata, model_path, output_csv_path, images_root_path, temp_scaling,
                    opengan, cfg):
    """Make submission file"""
    # TODO: use the dataloader with a larger batch size to speed up inference
    # TODO: pull this transformation from the datasets module

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using devide: {device}")

    model_id = cfg["evaluate"]["model_id"]

    # Consolidate the model building logic
    if opengan:
        model = create_composite_model(cfg, probas=True).to(device)
    else:
        model = PytorchWorker(model_path, temp_scaling=temp_scaling, model_id=model_id, device=device)

    # this allows use on both validation and test
    test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)

    probas_total = []
    opengan_probas_total = []
    for i, row in tqdm(test_metadata.iterrows(), total=len(test_metadata)):
        image_path = row["image_path"]
        image_path = os.path.join(images_root_path, image_path)
        test_image = Image.open(image_path).convert("RGB")
        if opengan:
            if cfg["evaluate"]["use_metadata"]:
                encoded_metadata_row = torch.tensor(np.array(encode_metadata_row(row)),
                                                    dtype=torch.float32).unsqueeze(0).to(device)
                transformed_image = TRANSFORMS(test_image).unsqueeze(0).to(device)
                logits, opengan_proba = model(transformed_image, encoded_metadata_row)
            else:
                logits, opengan_proba = model(TRANSFORMS(test_image).unsqueeze(0).to(device))
            # TODO: .item() solution is brittle and won't work once multiple images are fed in a batch
            probas = softmax(logits.item())
            probas_total.append(probas)
            opengan_probas_total.append(opengan_proba.item())
        else:
            if cfg["evaluate"]["use_metadata"]:
                encoded_metadata_row = torch.tensor(np.array(encode_metadata_row(row)), dtype=torch.float32)
                logits = model.predict_image(test_image, encoded_metadata_row)
            else:
                logits = model.predict_image(test_image)
            probas = softmax(logits)
            probas_total.append(probas)
            opengan_probas_total.append(probas)
    probas_total = np.array(probas_total)
    opengan_probas_total = np.array(opengan_probas_total)

    # update predictions and scores based on average of model probas
    predictions = []
    max_probas = []
    entropy_scores = []
    avg_opengan_probas = []
    # pandas unique preserves order
    unique_observations = test_metadata["observation_id"].unique()
    for obs_id in unique_observations:
        indices = (test_metadata["observation_id"].values == obs_id).nonzero()[0]

        obs_probas = probas_total[indices]  # should still work for single index
        avg_obs_probas = np.mean(obs_probas, axis=0)
        entropy_score = entropy(avg_obs_probas.squeeze())
        entropy_scores.extend([entropy_score] * len(indices))
        max_proba = np.max(avg_obs_probas)
        max_probas.extend([max_proba] * len(indices))

        obs_opengan_probas = opengan_probas_total[indices]
        obs_opengan_avg = np.mean(obs_opengan_probas, axis=0)
        avg_opengan_probas.extend([obs_opengan_avg] * len(indices))

        pred = np.argmax(avg_obs_probas, axis=-1).item()
        predictions.extend([pred] * len(indices))

    test_metadata.loc[:, "class_id"] = predictions
    if opengan:
        test_metadata.loc[:, "max_proba"] = avg_opengan_probas
        keep_cols = ["observation_id", "class_id", "max_proba"]
    else:
        test_metadata.loc[:, "max_proba"] = max_probas
        test_metadata.loc[:, "entropy"] = entropy_scores
        keep_cols = ["observation_id", "class_id", "max_proba", "entropy"]
    test_metadata.sort_index(inplace=True)  # revert sorting in case it matters
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
    predictions_with_unknown_output_csv_path = str(
        experiment_dir / f"submission_with_unknowns_fine_tuned_thresholding{ext}.csv")

    data_split = cfg['evaluate']['data_split']
    metadata_file_path = METADATA_DIR / f"{data_split}_split_openclosed_val.csv"
    test_metadata = pd.read_csv(metadata_file_path)
    test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)
    test_metadata = test_metadata.sort_values("observation_id")

    # Make predictions if they need to be made. Skip if already computed.
    if not from_outputs:
        make_submission(
            test_metadata=test_metadata,
            model_path=model_path,
            images_root_path=DATA_DIR,
            output_csv_path=predictions_output_csv_path,
            temp_scaling=temperature_scaling,
            opengan=opengan,
            cfg=cfg,
        )

    if opengan:
        closedset_label = cfg["open-set-recognition"]["closedset_label"]
        metrics_output_csv_path = str(experiment_dir / "threshold_scores_opengan.csv")
        scores_output_path = str(experiment_dir / "competition_metrics_scores_opengan.json")
        best_threshold_path = str(experiment_dir / "best_threshold_opengan.txt")
        # Generate metrics from predictions
        y_true = test_metadata["class_id"].values
        submission_df = pd.read_csv(predictions_output_csv_path)
        submission_df.drop_duplicates("observation_id", keep="first", inplace=True)
        # if closedset_label == 1, high proba == known
        # if closedset_label == 0, then a high proba == unknown, so probas need to be inverted to work with:
        # y_pred[y_proba < threshold] = -1
        y_proba = submission_df["max_proba"].values if closedset_label == 1 else -submission_df["max_proba"].values
        threshold_range = np.arange(y_proba.min(), y_proba.max(), 0.01)
        best_threshold = get_best_threshold(metrics_output_csv_path, submission_df, threshold_range, y_proba,
                                            y_true)
        with open(best_threshold_path, 'w') as f:
            f.write(str(best_threshold))
        homebrewed_scores = create_homebrwed_scores_and_save_predictions_csv(best_threshold,
                                                                             predictions_with_unknown_output_csv_path,
                                                                             submission_df, y_proba, y_true)
        save_metrics(homebrewed_scores, metadata_file_path, predictions_with_unknown_output_csv_path,
                     scores_output_path)

    else:
        for thresholding_method in ["_softmax", "_entropy", "_no_unknown_baseline"]:

            if thresholding_method == "_no_unknown_baseline":
                ext = thresholding_method
            else:
                ext += thresholding_method
            metrics_output_csv_path = str(experiment_dir / f"threshold_scores{ext}.csv")
            scores_output_path = str(experiment_dir / f"competition_metrics_scores{ext}.json")
            best_threshold_path = str(experiment_dir / f"best_threshold{ext}.txt")

            # Generate metrics from predictions
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
                threshold_range = np.arange(y_proba.min(), y_proba.max(), 0.01)
            elif thresholding_method == "_no_unknown_baseline":
                threshold_range = None
            else:
                raise ValueError(f"unrecognized thresholding method {thresholding_method}")

            best_threshold = get_best_threshold(metrics_output_csv_path, submission_df, threshold_range, y_proba,
                                                y_true)
            with open(best_threshold_path, 'w') as f:
                f.write(str(best_threshold))
            homebrewed_scores = create_homebrwed_scores_and_save_predictions_csv(best_threshold,
                                                                                 predictions_with_unknown_output_csv_path,
                                                                                 submission_df, y_proba, y_true)
            save_metrics(homebrewed_scores, metadata_file_path, predictions_with_unknown_output_csv_path,
                         scores_output_path)


def create_homebrwed_scores_and_save_predictions_csv(best_threshold, predictions_with_unknown_output_csv_path,
                                                     submission_df, y_proba, y_true):
    # reset y_pred and create y_pred_w_unknown using the best threshold before calculating homebrewed_scores
    y_pred = np.copy(submission_df["class_id"].values)
    y_pred_w_unknown = y_pred.copy()
    if not best_threshold is None:
        y_pred_w_unknown[y_proba < best_threshold] = -1
    homebrewed_scores = calc_homebrewed_scores(y_true, y_pred, y_pred_w_unknown)
    # make and save the unknown output csv
    submission_df.loc[:, "class_id"] = y_pred_w_unknown
    submission_df.to_csv(predictions_with_unknown_output_csv_path, index=False)
    return homebrewed_scores


def get_best_threshold(metrics_output_csv_path, submission_df, threshold_range, y_proba, y_true):
    if threshold_range is None:
        return None
    scores = []
    thresholds = []
    y_pred = np.copy(submission_df["class_id"].values)
    best_threshold, threshold = None, None
    # we actually want the *unbalanced* (micro) f1, since final competition eval is unbalanced?
    score = f1_score(y_true, y_pred, average='macro')
    best_f1 = score
    thresholds.append(threshold)
    scores.append(score)
    print(threshold, score)
    for threshold in threshold_range:
        y_pred = np.copy(submission_df["class_id"].values)
        y_pred[y_proba < threshold] = -1
        # we actually want the *unbalanced* (micro) f1, since final competition eval is unbalanced?
        score = f1_score(y_true, y_pred, average='macro')
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
        print(threshold, score)
        thresholds.append(threshold)
        scores.append(score)
    threshold_scores = pd.DataFrame()
    threshold_scores['threshold'] = thresholds
    threshold_scores['f1'] = scores
    threshold_scores.sort_values('f1').to_csv(metrics_output_csv_path, index=False)
    return best_threshold


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
    f1_micro_known_vs_unknown = f1_score(y_true_known_vs_unknown, y_pred_known_vs_unknown, average='micro')
    homebrewed_scores['f1_micro_known_vs_unknown'] = f1_micro_known_vs_unknown
    print("F1 micro known vs unknown:", f1_micro_known_vs_unknown)
    roc_auc_known_vs_unknown = roc_auc_score(y_true_known_vs_unknown, y_pred_known_vs_unknown)
    homebrewed_scores['roc_auc_known_vs_unknown'] = roc_auc_known_vs_unknown
    print("roc_auc_known_vs_unknown:", roc_auc_known_vs_unknown)

    return homebrewed_scores


if __name__ == "__main__":
    # TODO: address issue of evaluating on all the unknowns despite some being used for train and val in fine-tuning
    # TODO: set threshold on val, evaluate on test (this should also solve the above given a val loader for unknowns)
    # TODO: use dataloader with a larger batch size to speed up inference

    with initialize(version_base=None, config_path="conf", job_name="evaluate"):
        cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))

    experiment_id = cfg["evaluate"]["experiment_id"]
    model_id = cfg["evaluate"]["model_id"]
    image_size = cfg["evaluate"]["image_size"]

    TRANSFORMS = get_valid_transform(image_size=image_size, pretrained=True)

    copy_config("multiinstance_evaluate", experiment_id)

    outputs_precomputed = False

    print(f"evaluating experiment {experiment_id}")
    evaluate_experiment(cfg=cfg, experiment_id=experiment_id, from_outputs=outputs_precomputed)

    print("creating temperature scaled model")
    create_temperature_scaled_model(cfg)
    print(f"evaluating experiment {experiment_id} with temperature scaling")
    evaluate_experiment(cfg=cfg, experiment_id=experiment_id, temperature_scaling=True,
                        from_outputs=outputs_precomputed)

    print("training openGAN model")
    train_and_select_discriminator(cfg)
    print(f"evaluating experiment {experiment_id} with openGAN")
    evaluate_experiment(cfg=cfg, experiment_id=experiment_id, opengan=True, from_outputs=outputs_precomputed)
