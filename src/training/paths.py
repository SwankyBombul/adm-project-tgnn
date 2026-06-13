"""Predictable on-disk layout for trained model checkpoints."""

from __future__ import annotations

from pathlib import Path

from src.common.paths import get_project_root

SAVED_MODELS_ROOT = "saved_models"
DEFAULT_MODEL_NAME = "gru4rec"
CHECKPOINT_FILENAME = "best.ckpt"


def saved_model_dir(model: str, run_name: str, project_root: Path | None = None) -> Path:
    """Return ``saved_models/<model>/<run_name>/`` under the project root."""
    root = project_root or get_project_root()
    return root / SAVED_MODELS_ROOT / model / run_name


def resolve_saved_checkpoint(
    model: str,
    run_name: str,
    alias: str = "best",
    project_root: Path | None = None,
) -> Path:
    """Resolve ``best`` to ``saved_models/<model>/<run_name>/best.ckpt``."""
    if alias not in {"best", "last"}:
        path = Path(alias)
        if not path.is_absolute():
            path = (project_root or get_project_root()) / path
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    run_dir = saved_model_dir(model, run_name, project_root=project_root)
    if alias == "best":
        ckpt = run_dir / CHECKPOINT_FILENAME
    else:
        ckpt = run_dir / "last.ckpt"

    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}. Train with fit first or pass --ckpt_path PATH."
        )
    return ckpt
