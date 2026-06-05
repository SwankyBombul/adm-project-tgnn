"""Download Yoochoose (RecSys Challenge 2015) from Kaggle into data/raw."""

import os
import shutil
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scr.utlis import get_project_root

DATASET_SLUG = "chadgostopp/recsys-challenge-2015"
NESTED_DATA_DIR = "yoochoose-data"


def download_dataset() -> None:
    project_root = get_project_root()
    load_dotenv(project_root / ".env")

    target_dir = project_root / "data" / "raw"

    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        print("ERROR: missing KAGGLE_USERNAME or KAGGLE_KEY in .env file.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory prepared at: {target_dir.resolve()}")

    import kaggle

    print("Authenticating with Kaggle API...")
    kaggle.api.authenticate()

    print(f"Downloading dataset ZIP: {DATASET_SLUG}...")
    kaggle.api.dataset_download_files(
        DATASET_SLUG,
        path=str(target_dir),
        unzip=False,
    )

    zip_files = list(target_dir.glob("*.zip"))
    if not zip_files:
        print("ERROR: dataset ZIP file was not found after download.")
        return

    zip_path = zip_files[0]

    print(f"Extracting dataset ZIP: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    nested_data_dir = target_dir / NESTED_DATA_DIR

    if nested_data_dir.exists() and nested_data_dir.is_dir():
        print(
            f"Detected nested '{NESTED_DATA_DIR}' folder. "
            "Moving contents into data/raw..."
        )

        for item in nested_data_dir.iterdir():
            destination = target_dir / item.name

            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()

            shutil.move(str(item), str(destination))

        nested_data_dir.rmdir()

    zip_path.unlink()

    print(f"Download complete. Dataset is ready in: {target_dir.resolve()}")


if __name__ == "__main__":
    download_dataset()
