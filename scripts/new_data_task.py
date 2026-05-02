from pathlib import Path
import re
import shutil
import sys


def safe_name(value: str) -> str:
    value = re.sub(r"[^\w\-.]+", "_", value.strip(), flags=re.UNICODE)
    return value.strip("_") or "task"


def write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8", newline="\n")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: new_data_task.py <name>")
        return 2

    root = Path(__file__).resolve().parents[1]
    name = safe_name(" ".join(sys.argv[1:]))
    task_dir = root / "work" / "data_security" / name

    for child in [
        "raw",
        "processed",
        "notebooks",
        "src",
        "models",
        "submissions",
        "logs",
        "report",
    ]:
        (task_dir / child).mkdir(parents=True, exist_ok=True)

    readme_template = root / "data_security_template.md"
    if readme_template.exists() and not (task_dir / "README.md").exists():
        shutil.copyfile(readme_template, task_dir / "README.md")

    handoff_template = root / "data_security_handoff_template.md"
    if handoff_template.exists() and not (task_dir / "handoff_log.md").exists():
        shutil.copyfile(handoff_template, task_dir / "handoff_log.md")

    write_if_missing(
        task_dir / "experiments.csv",
        "time,run_id,model,features,seed,local_score,public_score,submission,notes\n",
    )

    write_if_missing(
        task_dir / "src" / "train.py",
        """\
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
MODELS = ROOT / "models"


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Fill in training pipeline after reading the task data.")


if __name__ == "__main__":
    main()
""",
    )

    write_if_missing(
        task_dir / "src" / "predict.py",
        """\
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
""",
    )

    print(task_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

