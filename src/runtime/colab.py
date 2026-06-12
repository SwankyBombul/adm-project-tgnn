"""Bootstrap a Colab runtime: Drive mount, local data unpack, checkpoint dirs."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from src.common.paths import get_project_root

PROCESSED_ZIP_NAME = "processed.zip"
DEFAULT_DRIVE_MOUNT = Path("/content/drive")
DEFAULT_MYDRIVE = DEFAULT_DRIVE_MOUNT / "MyDrive"


@dataclass
class DriveLayoutCheck:
    drive_project_dir: Path
    processed_zip: Path
    checkpoint_root: Path
    ok: bool
    errors: list[str]


@dataclass
class ColabSession:
    """Paths after mounting Drive and unpacking data locally in Colab."""

    drive_project_dir: Path
    local_processed_root: Path
    checkpoint_dir: Path


def _drive_layout_hints(drive_project_dir: Path) -> list[str]:
    hints: list[str] = []
    if not is_colab():
        return hints

    if not DEFAULT_DRIVE_MOUNT.is_dir():
        hints.append(
            "Google Drive is not mounted yet. "
            "Call mount_google_drive() before checking paths."
        )
        return hints

    if not drive_project_dir.is_dir() and DEFAULT_MYDRIVE.is_dir():
        folders = sorted(path.name for path in DEFAULT_MYDRIVE.iterdir() if path.is_dir())
        preview = ", ".join(folders[:12]) or "(no folders found)"
        hints.append(f"Top-level folders in MyDrive: {preview}")
        hints.append(
            "Update DRIVE_PROJECT_DIR in the notebook config cell to match your folder."
        )
    return hints


def check_drive_layout(drive_project_dir: Path) -> DriveLayoutCheck:
    """Validate the expected Google Drive project folder (call after drive.mount)."""
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

    errors.extend(_drive_layout_hints(drive_project_dir))

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
    """Unpack ``processed.zip`` and return the local ``processed/`` directory."""
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


def prepare_colab_session(
    drive_project_dir: str | Path,
    *,
    run_name: str = "default",
    model_name: str = "gru4rec",
    mount_drive: bool = True,
    unpack_zip: bool = True,
) -> ColabSession:
    """Mount Drive, validate layout, unpack data locally, ensure checkpoint dir exists."""
    drive_project_dir = Path(drive_project_dir)
    checkpoint_dir = drive_project_dir / "checkpoints" / model_name / run_name

    if mount_drive and is_colab():
        mount_google_drive()

    layout = check_drive_layout(drive_project_dir)
    if not layout.ok:
        raise FileNotFoundError("Drive layout invalid:\n" + "\n".join(layout.errors))

    zip_path = layout.processed_zip
    if unpack_zip:
        if not zip_path.is_file():
            raise FileNotFoundError(
                f"Processed archive not found on Drive: {zip_path}. "
                "Upload data/processed.zip to the project Drive folder."
            )
        local_processed_root = unzip_processed_archive(zip_path)
    else:
        local_processed_root = get_project_root() / "data" / "processed"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return ColabSession(
        drive_project_dir=drive_project_dir,
        local_processed_root=local_processed_root,
        checkpoint_dir=checkpoint_dir,
    )
