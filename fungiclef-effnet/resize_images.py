"""
Offline resizing of images.
Pathing might be broken if not in root dir.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision.transforms import InterpolationMode, v2
from tqdm import tqdm

from paths import DATA_DIR, METADATA_DIR

RESIZED_DIR = Path("~").expanduser().absolute() / "datasets" / "resized_fungiclef"
RESIZED_DIR.mkdir(parents=True, exist_ok=True)

folder_metas = [("DF20", "FungiCLEF2023_train_metadata_PRODUCTION.csv"),
                ("DF21", "FungiCLEF2023_val_metadata_PRODUCTION.csv")]

for folder, metadata_file in folder_metas:

    df = pd.read_csv(METADATA_DIR / metadata_file)
    img_paths = df['image_path']

    out_dir = RESIZED_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = v2.Resize(384 * 2, interpolation=InterpolationMode.BICUBIC, antialias=True)

    print(f"save resized images to {DATA_DIR / folder}")
    for img_path in tqdm(img_paths):
        if (out_dir / img_path).exists():
            continue
        with open(DATA_DIR / folder / img_path, "rb") as f:
            image = Image.open(f).convert('RGB')
            image = transform(image)
        with open(out_dir / img_path, "wb") as f:
            image.save(f)
