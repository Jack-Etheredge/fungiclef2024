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

class PytorchWorker:
    """Run inference using PyTorch."""

    def __init__(self, model_path: str, model_name: str, number_of_categories: int = 1604):
        self.number_of_categories = number_of_categories
        self.model = self._load_model(model_name, model_path)
        self.transforms = T.Compose([T.Resize((299, 299)),
                                     T.ToTensor(),
                                     T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def _load_model(self, model_name, model_path):
        print("Setting up Pytorch Model")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using devide: {self.device}")

        model = timm.create_model(model_name, num_classes=self.number_of_categories, pretrained=False)
        model_ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(model_ckpt)

        return model.to(self.device).eval()

    def predict_image(self, image: np.ndarray) -> list():
        """Run inference using ONNX runtime.

        :param image: Input image as numpy array.
        :return: A list with logits and confidences.
        """

        logits = self.model(self.transforms(image).unsqueeze(0).to(self.device))

        return logits.tolist()


def make_submission(test_metadata, model_path, model_name, output_csv_path="./submission.csv",
                    images_root_path="/tmp/data/private_testset"):
    """Make submission with given """

    model = PytorchWorker(model_path, model_name)

    # this allows use on both validation and test
    test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)

    predictions = []
    image_paths = test_metadata["image_path"]
    for image_path in tqdm(image_paths):
        try:
            image_path = os.path.join(images_root_path, image_path)
            test_image = Image.open(image_path).convert("RGB")
            logits = model.predict_image(test_image)
            predictions.append(np.argmax(logits))
        except Exception as e:
            print(f"issue with image {image_path}: {e}")
            predictions.append(-1)

    test_metadata["class_id"] = predictions
    user_pred_df = test_metadata.drop_duplicates("observation_id", keep="first")
    user_pred_df[["observation_id", "class_id"]].to_csv(output_csv_path, index=None)


if __name__ == "__main__":

    MODEL_PATH = "pytorch_model.bin"
    MODEL_NAME = "tf_efficientnet_b1.ap_in1k"

    data_dir = Path('__file__').parent.absolute().parent / "data" / "DF21"
    metadata_file_path = "./FungiCLEF2023_val_metadata_PRODUCTION.csv"
    test_metadata = pd.read_csv(metadata_file_path)

    make_submission(
        test_metadata=test_metadata,
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
        images_root_path=data_dir,
    )
