"""Weights & Biases integration for Colab and local training."""

from __future__ import annotations

from typing import Any

from src.config.train_config import TrainConfig
from src.config.wandb_settings import expected_wandb_settings


class WandbLogger:
    """Thin wrapper so training code does not import wandb directly everywhere."""

    def __init__(self, run: Any | None) -> None:
        self._run = run

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled:
            return
        self._run.log(metrics, step=step)

    def watch(self, model: Any, log: str = "gradients", log_freq: int = 100) -> None:
        if not self.enabled:
            return
        import wandb

        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        if not self.enabled:
            return
        import wandb

        wandb.finish()


def init_wandb(config: TrainConfig, extra_config: dict[str, Any] | None = None) -> WandbLogger:
    if not config.wandb_enabled:
        return WandbLogger(None)

    import wandb

    run_config = config.to_dict()
    if extra_config:
        run_config.update(extra_config)

    defaults = expected_wandb_settings()
    run = wandb.init(
        project=config.wandb_project or defaults.project,
        entity=config.wandb_entity or defaults.entity,
        name=config.wandb_run_name or f"{config.model_name}-{config.processed_variant}",
        tags=list(config.wandb_tags),
        config=run_config,
    )
    return WandbLogger(run)
