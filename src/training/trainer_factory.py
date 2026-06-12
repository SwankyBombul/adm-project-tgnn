"""Build a Lightning Trainer wired to project config (Colab, wandb, Drive)."""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.train_config import TrainConfig
from src.config.wandb_settings import expected_wandb_settings


def build_lightning_trainer(config: TrainConfig) -> pl.Trainer:
    defaults = expected_wandb_settings()
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename="epoch_{epoch:03d}",
            monitor=config.checkpoint_monitor,
            mode="max",
            save_top_k=config.keep_last_n_checkpoints,
            save_last=True,
        )
    ]

    if config.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=config.checkpoint_monitor,
                mode="max",
                patience=config.early_stopping_patience,
            )
        )

    logger: WandbLogger | None = None
    if config.wandb_enabled:
        logger = WandbLogger(
            project=config.wandb_project or defaults.project,
            entity=config.wandb_entity or defaults.entity,
            name=config.wandb_run_name,
            tags=list(config.wandb_tags),
            save_dir=str(config.checkpoint_root),
            log_model=False,
        )

    use_gpu = config.device.startswith("cuda") and torch.cuda.is_available()
    return pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )
