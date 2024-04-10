# modified from https://github.com/Poulinakis-Konstantinos/ML-util-functions/blob/master/scripts/Img_Premature_Ending-Detect_Fix.py

''' This script will detect and also fix the problem of Premature Ending in images.
This is caused when the image is corrupted in such a way that their hex code does not end with the default
D9. Opening the image with opencv and other image libraries is usually still possible, but the images might
produce errors during DL training or other tasks.
  Loading such an image with opencv and then saving it again can solve the problem. You can manually inspect
,using a notepad, that the image's hex finishes with D9 after the script has finished.
'''

import os
from pathlib import Path
import cv2
from tqdm import tqdm

EXTENSIONS = ['.JPG', '.jpg']
DATA_DIR = Path('__file__').parent.absolute().parent / 'data'
TRAIN_DIR = DATA_DIR / "DF20"
VAL_DIR = DATA_DIR / "DF21"


def detect_and_fix(img_path, img_name):
    # detect for premature ending
    try:
        with open(img_path, 'rb') as im:
            im.seek(-2, 2)
            if im.read() == b'\xff\xd9':
                print('Image OK :', img_name)
            else:
                # fix image
                img = cv2.imread(img_path)
                cv2.imwrite(img_path, img)
                print('FIXED corrupted image :', img_name)
    except(IOError, SyntaxError) as e:
        print(e)
        print("Unable to load/write Image : {} . Image might be destroyed".format(img_path))


def clean_dir(dir_path):
    for path in tqdm(os.listdir(str(dir_path))):
        # Make sure to change the extension if it is nor 'jpg' ( for example 'JPG','PNG' etc..)
        for extension in EXTENSIONS:
            if path.endswith(extension):
                img_path = os.path.join(dir_path, path)
                detect_and_fix(img_path=img_path, img_name=path)


if __name__ == "__main__":
    for dir_path in [TRAIN_DIR, VAL_DIR]:
        clean_dir(dir_path)
print("Process Finished")
