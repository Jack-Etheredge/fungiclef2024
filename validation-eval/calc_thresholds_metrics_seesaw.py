import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np


def get_threshold(max_prob_list, k):
    # Calculate Q1 and Q3 quartiles
    q1 = np.quantile(max_prob_list, 0.25)
    q3 = np.quantile(max_prob_list, 0.75)

    # Get the Interquartile Range (IQR)
    iqr = q3 - q1

    # Calculate Minimum threshold
    min_th = q1 - k * iqr

    return min_th


if __name__ == "__main__":
    input_csv_path = "submission_fine_tuned_thresholding_seesaw.csv"
    output_csv_path = "threshold_scores_seesaw.csv"
    data_dir = Path('__file__').parent.absolute().parent / "data" / "DF21"
    metadata_file_path = "./FungiCLEF2023_val_metadata_PRODUCTION.csv"
    test_metadata = pd.read_csv(metadata_file_path)
    test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)
    test_metadata.drop_duplicates("observation_id", keep="first", inplace=True)
    y_true = test_metadata["class_id"].values
    submission_df = pd.read_csv(input_csv_path)
    submission_df.drop_duplicates("observation_id", keep="first", inplace=True)
    y_proba = submission_df["max_proba"].values
    scores = []
    thresholds = []

    y_pred = np.copy(submission_df["class_id"].values)
    threshold = None
    score = f1_score(y_true, y_pred, average='macro')
    thresholds.append(threshold)
    scores.append(score)

    for threshold in np.arange(0, 1, 0.05):
        y_pred = np.copy(submission_df["class_id"].values)
        y_pred[y_proba < threshold] = -1
        score = f1_score(y_true, y_pred, average='macro')
        print(threshold, score)
        thresholds.append(threshold)
        scores.append(score)
    for k in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        threshold = get_threshold(y_proba, k)
        y_pred = np.copy(submission_df["class_id"].values)
        y_pred[y_proba < threshold] = -1
        score = f1_score(y_true, y_pred, average='macro')
        print(f"iqr_k{k}", threshold, score)
        thresholds.append(threshold)
        scores.append(score)
    threshold_scores = pd.DataFrame()
    threshold_scores['threshold'] = thresholds
    threshold_scores['f1'] = scores
    threshold_scores.sort_values('f1').to_csv(output_csv_path, index=False)
