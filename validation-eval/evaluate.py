import json

import pandas as pd
import numpy as np
import os

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm
import torchvision.models as models
import torch.nn as nn
# import torchvision.transforms as T
from torchvision.transforms import v2
from PIL import Image
import torch
from pathlib import Path
from PIL import ImageFile
from scipy.special import softmax
from competition_metrics import evaluate

np.set_printoptions(precision=5)
ImageFile.LOAD_TRUNCATED_IMAGES = True


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

    def __init__(self, model_path: str, number_of_categories: int = 1604):
        self.number_of_categories = number_of_categories
        self.model = self._load_model(model_path)
        # self.transforms = T.Compose([T.Resize((299, 299)),
        #                              T.ToTensor(),
        #                              # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        #                              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transforms = v2.Compose([
            v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            # v2.ToTensor(),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        print("Setting up Pytorch Model")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using devide: {self.device}")
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(in_features=1280, out_features=self.number_of_categories)
        model_ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(model_ckpt['model_state_dict'])

        return model.to(self.device).eval()

    def predict_image(self, image: np.ndarray) -> list():
        """Run inference using ONNX runtime.

        :param image: Input image as numpy array.
        :return: A list with logits and confidences.
        """

        logits = self.model(self.transforms(image).unsqueeze(0).to(self.device))

        return logits.tolist()


def make_submission(test_metadata, model_path, output_csv_path, images_root_path):
    """Make submission with given """

    model = PytorchWorker(model_path)

    # this allows use on both validation and test
    test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)
    test_metadata = test_metadata.drop_duplicates("observation_id", keep="first")

    predictions = []
    max_probas = []
    image_paths = test_metadata["image_path"]
    with torch.no_grad():
        for image_path in tqdm(image_paths):
            try:
                image_path = os.path.join(images_root_path, image_path)
                test_image = Image.open(image_path).convert("RGB")
                logits = model.predict_image(test_image)
                logits = softmax(logits)
                predictions.append(np.argmax(logits))
                max_probas.append(np.max(logits))
            except Exception as e:
                print(f"issue with image {image_path}: {e}")
                predictions.append(-1)
                max_probas.append(-1)

    test_metadata.loc[:, "class_id"] = predictions
    test_metadata.loc[:, "max_proba"] = max_probas
    test_metadata[["observation_id", "class_id", "max_proba"]].to_csv(output_csv_path, index=None)


if __name__ == "__main__":
    MODEL_PATH = "../fungiclef-effnet/model_checkpoints/best_model_seesaw_batch_128_lr_ 0.000800_dropout_ 0.50_weight_decay_ 0.000010_unfreeze_epoch_4_over_False_over_prop_0.1_under_False_balanced_sampler_False_equal_undersampled_val_True_trivialaug.pth"
    predictions_output_csv_path = "submission_fine_tuned_thresholding_seesaw_04_02_2024.csv"
    predictions_with_unknown_output_csv_path = "submission_with_unknowns_fine_tuned_thresholding_seesaw_04_02_2024.csv"
    metrics_output_csv_path = "threshold_scores_seesaw_04_02_2024.csv"
    scores_output_path = "seesaw_04_02_2024_scores.json"

    data_dir = Path('__file__').parent.absolute().parent / "data" / "DF21"
    metadata_file_path = "../metadata/FungiCLEF2023_val_metadata_PRODUCTION.csv"
    test_metadata = pd.read_csv(metadata_file_path)

    # Make predictions
    make_submission(
        test_metadata=test_metadata,
        model_path=MODEL_PATH,
        images_root_path=data_dir,
        output_csv_path=predictions_output_csv_path,
    )

    # Generate metrics from predictions
    test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)
    test_metadata.drop_duplicates("observation_id", keep="first", inplace=True)
    y_true = test_metadata["class_id"].values
    submission_df = pd.read_csv(predictions_output_csv_path)
    submission_df.drop_duplicates("observation_id", keep="first", inplace=True)
    y_proba = submission_df["max_proba"].values
    scores = []
    thresholds = []

    y_pred = np.copy(submission_df["class_id"].values)
    best_threshold, threshold = None, None
    score = f1_score(y_true, y_pred, average='macro')
    best_f1 = score
    thresholds.append(threshold)
    scores.append(score)

    for threshold in np.arange(0, 1, 0.05):
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
    y_pred = np.copy(submission_df["class_id"].values)
    y_pred[y_proba < best_threshold] = -1
    y_true_known_vs_unknown = y_true.copy()
    y_true_known_vs_unknown[~(y_true == -1)] = 1  # 1 is known
    y_true_known_vs_unknown[(y_true == -1)] = 0  # 0 is unknown
    y_pred_known_vs_unknown = y_pred.copy()
    y_pred_known_vs_unknown[~(y_pred == -1)] = 1
    y_pred_known_vs_unknown[(y_pred == -1)] = 0
    f1_binary_known_vs_unknown = f1_score(y_true_known_vs_unknown, y_pred_known_vs_unknown, average='binary')
    homebrewed_scores['f1_binary_known_vs_unknown'] = f1_binary_known_vs_unknown
    print("F1 binary known vs unknown:", f1_binary_known_vs_unknown)
    f1_macro_known_vs_unknown = f1_score(y_true_known_vs_unknown, y_pred_known_vs_unknown, average='macro')
    homebrewed_scores['f1_macro_known_vs_unknown'] = f1_macro_known_vs_unknown
    print("F1 macro known vs unknown:", f1_macro_known_vs_unknown)

    # make and save the unknown output csv
    submission_df.loc[:, "class_id"] = y_pred
    submission_df.to_csv(predictions_with_unknown_output_csv_path, index=False)

    # add additional competition metrics
    competition_metrics_scores = evaluate(
        test_annotation_file=metadata_file_path,
        user_submission_file=predictions_with_unknown_output_csv_path,
        phase_codename="prediction-based",
    )
    competition_metrics_scores.update(homebrewed_scores)
    with open(scores_output_path, "w") as f:
        json.dump(competition_metrics_scores, f)

    # TODO: add this evaluation functionality to the training script
    # TODO: refactor this code to be more reusable
