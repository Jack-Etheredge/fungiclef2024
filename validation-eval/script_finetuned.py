import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import torch
from pathlib import Path
from PIL import ImageFile
from scipy.special import softmax

np.set_printoptions(precision=5)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PytorchWorker:
    """Run inference using PyTorch."""

    def __init__(self, model_path: str, number_of_categories: int = 1604):
        self.number_of_categories = number_of_categories
        self.model = self._load_model(model_path)
        self.transforms = T.Compose([T.Resize((299, 299)),
                                     T.ToTensor(),
                                     # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def _load_model(self, model_path):
        print("Setting up Pytorch Model")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using devide: {self.device}")
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(in_features=1280, out_features=self.number_of_categories)
        model_ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(model_ckpt['model_state_dict'])

        return model.to(self.device).eval()

    def predict_image(self, image: np.ndarray) -> list():
        """Run inference using ONNX runtime.

        :param image: Input image as numpy array.
        :return: A list with logits and confidences.
        """

        logits = self.model(self.transforms(image).unsqueeze(0).to(self.device))

        return logits.tolist()


def make_submission(test_metadata, model_path, output_csv_path="./submission_fine_tuned_thresholding.csv",
                    images_root_path="/tmp/data/private_testset"):
    """Make submission with given """

    model = PytorchWorker(model_path)

    # this allows use on both validation and test
    test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)
    test_metadata = test_metadata.drop_duplicates("observation_id", keep="first")

    predictions = []
    max_probas = []
    image_paths = test_metadata["image_path"]
    with torch.no_grad():
        for image_path in tqdm(image_paths):
            try:
                image_path = os.path.join(images_root_path, image_path)
                test_image = Image.open(image_path).convert("RGB")
                logits = model.predict_image(test_image)
                logits = softmax(logits)
                predictions.append(np.argmax(logits))
                max_probas.append(np.max(logits))
            except Exception as e:
                print(f"issue with image {image_path}: {e}")
                predictions.append(-1)
                max_probas.append(-1)

    test_metadata["class_id"] = predictions
    test_metadata["max_proba"] = max_probas
    test_metadata[["observation_id", "class_id", "max_proba"]].to_csv(output_csv_path, index=None)


if __name__ == "__main__":
    MODEL_PATH = "best_model.pth"

    data_dir = Path('__file__').parent.absolute().parent / "data" / "DF21"
    metadata_file_path = "./FungiCLEF2023_val_metadata_PRODUCTION.csv"
    test_metadata = pd.read_csv(metadata_file_path)

    make_submission(
        test_metadata=test_metadata,
        model_path=MODEL_PATH,
        images_root_path=data_dir,
    )
