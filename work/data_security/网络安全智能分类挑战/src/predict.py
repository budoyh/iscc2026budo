from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"


def main() -> None:
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Fill in prediction pipeline after baseline is defined.")


if __name__ == "__main__":
    main()
