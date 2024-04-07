from collections import Counter

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import timm
import torchvision.transforms as T
from PIL import Image
import torch
from pathlib import Path
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == "__main__":
    data_dir = Path('__file__').parent.absolute().parent / "data"
    train_data_dir = data_dir / "DF20"
    val_data_dir = data_dir / "DF21"
    train_metadata_file_path = "../metadata/FungiCLEF2023_train_metadata_PRODUCTION.csv"
    train_metadata = pd.read_csv(train_metadata_file_path)
    val_metadata_file_path = "./FungiCLEF2023_val_metadata_PRODUCTION.csv"
    val_metadata = pd.read_csv(val_metadata_file_path)

    datasets = [
        ("train", train_metadata, train_data_dir),
        ("val", val_metadata, val_data_dir),
    ]
    for dataset, metadata_df, dataset_dir in datasets:
        classes = metadata_df["class_id"]
        classes = Counter(classes)
        df = pd.DataFrame()
        df['class'] = list(classes.keys())
        df['count'] = list(classes.values())
        df['dataset'] = dataset
        df.sort_values("count", inplace=True)
        df.to_csv(f"../metadata/{dataset}_class_distribution.csv", index=False)
