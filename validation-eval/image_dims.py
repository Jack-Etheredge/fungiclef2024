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

    dataset_ids = []
    images = []
    heights = []
    widths = []
    datasets = [
        ("train", train_metadata, train_data_dir),
        ("val", val_metadata, val_data_dir),
    ]
    for dataset, metadata_df, dataset_dir in datasets:
        image_paths = metadata_df["image_path"]
        for image_path in tqdm(image_paths):
            image_path = os.path.join(dataset_dir, image_path)
            img = Image.open(image_path).convert("RGB")
            dataset_ids.append(dataset)
            images.append(image_path)
            heights.append(img.height)
            widths.append(img.width)
    dims_df = pd.DataFrame()
    dims_df["dataset"] = dataset_ids
    dims_df["image"] = images
    dims_df["height"] = heights
    dims_df["width"] = widths
    dims_df.to_csv("../metadata/dataset_dims.csv", index=False)
