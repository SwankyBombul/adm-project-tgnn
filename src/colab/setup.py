"""Bootstrap a Colab runtime: Drive mount, local data unpack, checkpoint dirs."""

from __future__ import annotations

import zipfile
from pathlib import Path

from dataclasses import dataclass

from src.config.train_config import PROCESSED_ZIP_NAME, TrainConfig
from src.utlis import get_project_root

DEFAULT_DRIVE_MOUNT = Path("/content/drive")
DEFAULT_MYDRIVE = DEFAULT_DRIVE_MOUNT / "MyDrive"


@dataclass
class DriveLayoutCheck:
    drive_project_dir: Path
    processed_zip: Path
    checkpoint_root: Path
    ok: bool
    errors: list[str]


def check_drive_layout(drive_project_dir: Path) -> DriveLayoutCheck:
    """Validate the expected Google Drive project folder before training."""
    drive_project_dir = Path(drive_project_dir)
    processed_zip = drive_project_dir / "data" / PROCESSED_ZIP_NAME
    checkpoint_root = drive_project_dir / "checkpoints"
    errors: list[str] = []

    if not drive_project_dir.is_dir():
        errors.append(f"Missing Drive project directory: {drive_project_dir}")
    if not processed_zip.is_file():
        errors.append(
            f"Missing processed archive: {processed_zip} "
            f"(expected data/{PROCESSED_ZIP_NAME} on Drive)."
        )

    return DriveLayoutCheck(
        drive_project_dir=drive_project_dir,
        processed_zip=processed_zip,
        checkpoint_root=checkpoint_root,
        ok=not errors,
        errors=errors,
    )


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def mount_google_drive(mount_point: Path = DEFAULT_DRIVE_MOUNT) -> Path:
    """Mount Google Drive in Colab; no-op elsewhere."""
    if not is_colab():
        return DEFAULT_MYDRIVE

    from google.colab import drive

    drive.mount(str(mount_point))
    return mount_point / "MyDrive"


def unzip_processed_archive(
    zip_path: Path,
    *,
    extract_parent: Path | None = None,
) -> Path:
    """Unpack ``processed.zip`` and return the local ``processed/`` directory.

    Expects the archive to contain a top-level ``processed/`` folder (standard
    ``zip -r processed.zip processed`` layout). If the archive root already
    looks like processed output (``meta.json`` or variant subfolders), files are
    placed under ``extract_parent/processed/``.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"Processed archive not found: {zip_path}")

    extract_parent = extract_parent or (get_project_root() / "data")
    extract_parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        top_levels = {name.split("/")[0] for name in names if name and not name.endswith("/")}

        if top_levels == {"processed"} or (
            len(top_levels) == 1 and "processed" in top_levels
        ):
            archive.extractall(extract_parent)
            processed_dir = extract_parent / "processed"
        else:
            processed_dir = extract_parent / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            archive.extractall(processed_dir)

    if not processed_dir.is_dir():
        raise RuntimeError(f"Expected processed directory at {processed_dir}")

    return processed_dir


def prepare_colab_session(config: TrainConfig) -> TrainConfig:
    """Mount Drive, unpack data to local disk, and ensure checkpoint directory exists."""
    if config.mount_google_drive and is_colab():
        mount_google_drive()

    if config.unpack_processed_zip:
        zip_path = config.processed_zip_path
        if zip_path is None:
            if is_colab():
                raise ValueError(
                    "unpack_processed_zip=True requires drive_project_dir "
                    "(path to data/processed.zip on Drive)."
                )
        elif zip_path.is_file():
            local_processed = unzip_processed_archive(zip_path)
            config.data_root = local_processed
        elif is_colab():
            raise FileNotFoundError(
                f"Processed archive not found on Drive: {zip_path}. "
                "Upload data/processed.zip to the project Drive folder."
            )

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return config
