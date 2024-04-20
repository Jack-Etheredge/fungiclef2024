"""
confirm the intuition from reading the evaluation code that unknown is treated as edible
track 2 edible-poisonous classification errors:
perfect_preds 0.0
poisonous-to-unknown 2.2601
edible-to-unknown 0.0
unknown-to-poisonous 0.327
unknown-to-edible 0.0
edible-to-poisonous 0.9774
poisonous-to-edible 2.2601
(100x penalty for poisonous -> edible but poisonous is much less common and this metric isn't class balanced)
"""

from pathlib import Path

import pandas as pd

from competition_metrics import evaluate


def make_preds_perfect(test_metadata):
    preds_perfect = test_metadata.copy()[["class_id", "observationID"]]
    preds_perfect = preds_perfect.drop_duplicates("observationID")
    return preds_perfect


def make_preds_poisonous_to_unknown(test_metadata, poisonous_sp):
    preds = test_metadata.copy()[["class_id", "observationID"]]
    preds = preds.drop_duplicates("observationID")
    preds.loc[preds["class_id"].isin(poisonous_sp), "class_id"] = -1
    return preds


def make_preds_edible_to_unknown(test_metadata, poisonous_sp):
    preds = test_metadata.copy()[["class_id", "observationID"]]
    preds = preds.drop_duplicates("observationID")
    preds.loc[~preds["class_id"].isin(poisonous_sp), "class_id"] = -1
    return preds


def make_preds_unknown_to_poisonous(test_metadata, poisonous_sp):
    preds = test_metadata.copy()[["class_id", "observationID"]]
    preds = preds.drop_duplicates("observationID")
    poisonous_example = list(poisonous_sp)[0]
    preds.loc[preds["class_id"] == -1, "class_id"] = poisonous_example
    return preds


def make_preds_unknown_to_edible(test_metadata, poisonous_sp):
    preds = test_metadata.copy()[["class_id", "observationID"]]
    preds = preds.drop_duplicates("observationID")
    edible_example = 0
    assert edible_example not in poisonous_sp
    preds.loc[preds["class_id"] == -1, "class_id"] = edible_example
    return preds


def make_preds_edible_to_poisonous(test_metadata, poisonous_sp):
    preds = test_metadata.copy()[["class_id", "observationID"]]
    preds = preds.drop_duplicates("observationID")
    poisonous_example = list(poisonous_sp)[0]
    preds.loc[~preds["class_id"].isin(poisonous_sp), "class_id"] = poisonous_example
    return preds


def make_preds_poisonous_to_edible(test_metadata, poisonous_sp):
    preds = test_metadata.copy()[["class_id", "observationID"]]
    preds = preds.drop_duplicates("observationID")
    edible_example = 0
    assert edible_example not in poisonous_sp
    preds.loc[preds["class_id"].isin(poisonous_sp), "class_id"] = edible_example
    return preds


def evaluate_experiment():
    metadata_dir = Path(__file__).parent.absolute() / ".." / "metadata"
    preds_dir = Path(__file__).parent.absolute() / "data_exploration"
    metadata_file_path = metadata_dir / "FungiCLEF2023_val_metadata_PRODUCTION.csv"
    test_metadata = pd.read_csv(metadata_file_path)
    preds_path = str(preds_dir / "eda_submission.csv")
    poisonous_def_df_path = metadata_dir / "poison_status_list.csv"
    poisonous_def_df = pd.read_csv(poisonous_def_df_path)
    poisonous_sp = set(poisonous_def_df[poisonous_def_df["poisonous"] == 1]["class_id"].unique())

    preds_dict = {
        "perfect_preds": make_preds_perfect(test_metadata),
        "poisonous-to-unknown": make_preds_poisonous_to_unknown(test_metadata, poisonous_sp),
        "edible-to-unknown": make_preds_edible_to_unknown(test_metadata, poisonous_sp),
        "unknown-to-poisonous": make_preds_unknown_to_poisonous(test_metadata, poisonous_sp),
        "unknown-to-edible": make_preds_unknown_to_edible(test_metadata, poisonous_sp),
        "edible-to-poisonous": make_preds_edible_to_poisonous(test_metadata, poisonous_sp),
        "poisonous-to-edible": make_preds_poisonous_to_edible(test_metadata, poisonous_sp),
    }

    track_2_results = {}
    for name, preds in preds_dict.items():
        print(name)
        preds.to_csv(preds_path, index=False)
        competition_metrics_scores = evaluate(
            test_annotation_file=metadata_file_path,
            user_submission_file=preds_path,
            phase_codename="prediction-based",
        )
        # print(competition_metrics_scores)
        track_2_results[name] = competition_metrics_scores['submission_result'][
            "Track 2: Cost for Poisonousness Confusion"]

    for k, v in track_2_results.items():
        print(k, v)


if __name__ == "__main__":
    evaluate_experiment()
