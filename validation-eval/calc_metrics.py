import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

if __name__ == "__main__":

    data_dir = Path('__file__').parent.absolute().parent / "data" / "DF21"
    metadata_file_path = "./FungiCLEF2023_val_metadata_PRODUCTION.csv"
    test_metadata = pd.read_csv(metadata_file_path)
    test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)
    test_metadata.drop_duplicates("observation_id", keep="first", inplace=True)
    y_true = test_metadata["class_id"].values
    methods = {
        "keep_first": "submission.csv",
        "mean": "submission_average.csv",
        "max": "submission_max.csv"
    }
    for method, submission_file_path in methods.items():
        submission_df = pd.read_csv(submission_file_path)
        submission_df.drop_duplicates("observation_id", keep="first", inplace=True)
        y_pred = submission_df["class_id"].values
        print(method, f1_score(y_true, y_pred, average='macro'))


