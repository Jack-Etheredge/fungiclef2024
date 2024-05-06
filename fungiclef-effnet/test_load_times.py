"""
Test to determine whether offline resizing of images would improve performance.
Pathing might be broken if not in root dir.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision.transforms import InterpolationMode, v2
from tqdm import tqdm

from paths import DATA_DIR, METADATA_DIR

RESIZED_DIR = Path("~").expanduser().absolute() / "datasets" / "resized_fungiclef_train"
RESIZED_DIR.mkdir(parents=True, exist_ok=True)

metadata_file = METADATA_DIR / "FungiCLEF2023_train_metadata_PRODUCTION.csv"

df = pd.read_csv(metadata_file)

transform = v2.Resize(600, interpolation=InterpolationMode.BICUBIC, antialias=True)

n_rows_test = 3000
img_paths = df['image_path'].head(n_rows_test)

print("test full sized image open and resize:")
start = datetime.now()
for img_path in tqdm(img_paths):
    with open(DATA_DIR / "DF20" / img_path, "rb") as f:
        image = Image.open(f).convert('RGB')
        image = transform(image)
end = datetime.now()
print("elapsed seconds full size image open and resize", (end - start).total_seconds())

print("test full sized image open only:")
start = datetime.now()
for img_path in tqdm(img_paths):
    with open(DATA_DIR / "DF20" / img_path, "rb") as f:
        image = Image.open(f).convert('RGB')
end = datetime.now()
print("elapsed seconds full size image open only", (end - start).total_seconds())

print("create a dir to save resized images to")
for img_path in tqdm(img_paths):
    with open(DATA_DIR / "DF20" / img_path, "rb") as f:
        image = Image.open(f).convert('RGB')
        image = transform(image)
    with open(RESIZED_DIR / img_path, "wb") as f:
        image.save(f)

print("test resized image open only:")
start = datetime.now()
for img_path in tqdm(img_paths):
    with open(RESIZED_DIR / img_path, "rb") as f:
        image = Image.open(f).convert('RGB')
end = datetime.now()
print("elapsed seconds resized image open only", (end - start).total_seconds())

print("test resized image open and resize to same size:")
start = datetime.now()
for img_path in tqdm(img_paths):
    with open(RESIZED_DIR / img_path, "rb") as f:
        image = Image.open(f).convert('RGB')
        image = transform(image)
end = datetime.now()
print("elapsed seconds resized image open and resize to same size", (end - start).total_seconds())
