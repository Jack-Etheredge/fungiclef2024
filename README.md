# FungiCLEF 2024 training code

- The dataset is expected to be in `~/datasets/fungiclef2024`.
- This can be updated in `fungiclef-effnet/paths.py`

### Model training

1. Install poetry if you don't already have it.
2. Install the dependencies along with this project:

```shell
poetry install
poetry shell
pip install -e .
```

3. Train the model after making whatever changes you'd like:

```shell
poetry run fungiclef-effnet/train.py
```

4. To create temperature scaled versions of models as well as openWGAN-GP models:

```shell
poetry run fungiclef-effnet/evaluate.py
```
