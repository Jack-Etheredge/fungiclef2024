"""
Convert original dataset into folders and create a new metadata file that maps to the original split
"""

import shutil

from pathlib import Path
import pandas as pd
from tqdm import tqdm

# from ..paths import DATA_DIR, DATA_FROM_FOLDER_DIR, METADATA_DIR
# TODO: refactor the codebase to have a sensible package structure
DATA_DIR = Path('~/datasets/fungiclef2024/data').expanduser().absolute()
DATA_FROM_FOLDER_DIR = Path('~/datasets/fungiclef2024/data_by_folder').expanduser().absolute()
METADATA_DIR = Path('__file__').parent.absolute().parent / 'metadata'

EXTENSIONS = ['.JPG', '.jpg']
TRAIN_DIR = DATA_DIR / "DF20"
VAL_DIR = DATA_DIR / "DF21"


def main():
    train_metadata = pd.read_csv(METADATA_DIR / "FungiCLEF2023_train_metadata_PRODUCTION.csv")
    val_metadata = pd.read_csv(METADATA_DIR / "FungiCLEF2023_val_metadata_PRODUCTION.csv")
    metadata_split = [(train_metadata, "train", TRAIN_DIR),
                      (val_metadata, "val", VAL_DIR)]
    image_paths = []
    image_labels = []
    image_stages = []
    for metadata, split, img_dir in metadata_split:
        metadata["original_data_split"] = split
        image_paths.extend([img_dir / img_pth for img_pth in metadata["image_path"]])
        image_labels.extend(metadata["class_id"].to_list())
        image_stages.extend([split] * len(metadata["image_path"]))
    combined_metadata = pd.concat([train_metadata, val_metadata], ignore_index=True)
    combined_metadata.drop(columns='filename', inplace=True)
    assert combined_metadata["image_path"].is_unique
    combined_metadata.to_csv(METADATA_DIR / "combined_metadata_dataset_from_folder.csv", index=False)

    problem_files = []
    for img_path, img_label, img_stage in tqdm(zip(image_paths, image_labels, image_stages), total=len(image_paths)):

        class_dir = DATA_FROM_FOLDER_DIR / img_stage / str(img_label)
        class_dir.mkdir(parents=True, exist_ok=True)

        dest = class_dir / img_path.name
        try:
            shutil.copy2(img_path, dest)
        except Exception as e:
            print(img_path, e)
            problem_files.append((img_path, img_label))

    print("issues copying following files:")
    print(problem_files)


if __name__ == "__main__":
    main()