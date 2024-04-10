from pathlib import Path

CHECKPOINT_DIR = Path('__file__').parent.absolute() / "model_checkpoints"
EMBEDDINGS_DIR = Path('__file__').parent.absolute() / "embeddings"


def setup_project():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    setup_project()
