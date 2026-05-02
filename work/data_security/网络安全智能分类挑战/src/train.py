from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
MODELS = ROOT / "models"


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Fill in training pipeline after reading the task data.")


if __name__ == "__main__":
    main()
