"""Training framework: shared LightningModule skeleton and checkpoint paths."""

from src.training.base_module import EVAL_DATALOADER_NAMES, NextItemLitModule
from src.training.paths import (
    CHECKPOINT_FILENAME,
    DEFAULT_MODEL_NAME,
    SAVED_MODELS_ROOT,
    resolve_saved_checkpoint,
    saved_model_dir,
)

__all__ = [
    "CHECKPOINT_FILENAME",
    "DEFAULT_MODEL_NAME",
    "EVAL_DATALOADER_NAMES",
    "NextItemLitModule",
    "SAVED_MODELS_ROOT",
    "resolve_saved_checkpoint",
    "saved_model_dir",
]
