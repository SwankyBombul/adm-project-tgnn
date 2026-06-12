"""Training configuration for local runs and Google Colab + Drive."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config.wandb_settings import WANDB_ENTITY, WANDB_PROJECT
from src.utlis import get_project_root

DEFAULT_PROCESSED_VARIANT = "subsample_1_32_clicks_only"
PROCESSED_ZIP_NAME = "processed.zip"


@dataclass
class TrainConfig:
    """Settings for model training, checkpointing, and experiment logging.

  Colab workflow (typical):
      1. Mount Google Drive and point ``drive_project_dir`` at the project folder.
      2. Unpack ``data/processed.zip`` into local ``data/processed/`` for fast I/O.
      3. Save checkpoints under ``drive_project_dir/checkpoints/``.
      4. Log metrics to Weights & Biases.
    """

    model_name: str = "gru4rec"
    processed_variant: str = DEFAULT_PROCESSED_VARIANT
    data_root: Path = field(
        default_factory=lambda: get_project_root() / "data" / "processed"
    )

    # Colab / Google Drive
    drive_project_dir: Path | None = None
    mount_google_drive: bool = True
    unpack_processed_zip: bool = True
    processed_zip_name: str = PROCESSED_ZIP_NAME

    # Checkpoints (written to Drive when ``drive_project_dir`` is set)
    run_name: str | None = None
    checkpoint_every_epochs: int = 1
    keep_last_n_checkpoints: int = 3

    # Training
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 1e-3
    num_workers: int = 2
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    checkpoint_monitor: str = "val/recall@20"
    early_stopping_patience: int | None = 3

    # Weights & Biases
    wandb_enabled: bool = True
    wandb_project: str = WANDB_PROJECT
    wandb_entity: str | None = WANDB_ENTITY
    wandb_run_name: str | None = None
    wandb_tags: tuple[str, ...] = ()

    @property
    def processed_dir(self) -> Path:
        return self.data_root / self.processed_variant

    @property
    def meta_path(self) -> Path:
        return self.processed_dir / "meta.json"

    @property
    def checkpoint_root(self) -> Path:
        base = self.drive_project_dir or (get_project_root() / "checkpoints")
        return base / "checkpoints" / self.model_name

    @property
    def checkpoint_dir(self) -> Path:
        run = self.run_name or self.wandb_run_name or "default"
        return self.checkpoint_root / run

    @property
    def processed_zip_path(self) -> Path | None:
        if self.drive_project_dir is None:
            return None
        return self.drive_project_dir / "data" / self.processed_zip_name

    @classmethod
    def for_colab(
        cls,
        drive_project_dir: str | Path,
        *,
        model_name: str = "gru4rec",
        processed_variant: str = DEFAULT_PROCESSED_VARIANT,
        wandb_run_name: str | None = None,
        **kwargs: Any,
    ) -> TrainConfig:
        """Factory for the standard Colab + Drive layout."""
        return cls(
            model_name=model_name,
            processed_variant=processed_variant,
            drive_project_dir=Path(drive_project_dir),
            mount_google_drive=True,
            unpack_processed_zip=True,
            wandb_run_name=wandb_run_name,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("data_root", "drive_project_dir"):
            value = data[key]
            if value is not None:
                data[key] = str(value)
        data["processed_dir"] = str(self.processed_dir)
        data["checkpoint_dir"] = str(self.checkpoint_dir)
        if self.processed_zip_path is not None:
            data["processed_zip_path"] = str(self.processed_zip_path)
        return data
