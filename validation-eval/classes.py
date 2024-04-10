from collections import Counter

import pandas as pd
from pathlib import Path
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == "__main__":
    data_dir = Path('__file__').parent.absolute().parent / "data"
    train_metadata_file_path = "../metadata/FungiCLEF2023_train_metadata_PRODUCTION.csv"
    train_metadata = pd.read_csv(train_metadata_file_path)
    val_metadata_file_path = "./FungiCLEF2023_val_metadata_PRODUCTION.csv"
    val_metadata = pd.read_csv(val_metadata_file_path)

    datasets = [
        ("train", train_metadata),
        ("val", val_metadata),
    ]
    for dataset, metadata_df in datasets:
        classes = metadata_df["class_id"]
        classes = Counter(classes)
        df = pd.DataFrame()
        df['class'] = list(classes.keys())
        df['count'] = list(classes.values())
        df['dataset'] = dataset
        df.sort_values("count", inplace=True)
        df.to_csv(f"../metadata/{dataset}_class_distribution.csv", index=False)

    train_counts = pd.DataFrame(train_metadata["class_id"].value_counts())
    train_counts.rename(columns={"count": "train_count"}, inplace=True)
    train_counts["train_percent"] = train_counts["train_count"] / train_counts["train_count"].sum() * 100
    train_counts.reset_index(inplace=True)
    val_counts = pd.DataFrame(val_metadata["class_id"].value_counts())
    val_counts.rename(columns={"count": "val_count"}, inplace=True)
    val_counts["val_percent"] = val_counts["val_count"] / val_counts["val_count"].sum() * 100
    val_counts.reset_index(inplace=True)
    merged = train_counts.merge(val_counts, how='outer')
    merged.fillna(0, inplace=True)
    merged["train_count"] = merged["train_count"].astype(int)
    merged["val_count"] = merged["val_count"].astype(int)
    merged = merged[["class_id", "train_count", "val_count", "train_percent", "val_percent"]]
    merged.to_csv(f"../metadata/train_val_class_distributions.csv", index=False)
