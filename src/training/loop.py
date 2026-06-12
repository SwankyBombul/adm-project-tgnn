"""Generic training loop skeleton shared across models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.config.train_config import TrainConfig
from src.training.checkpoints import save_checkpoint
from src.training.wandb_logger import WandbLogger


@dataclass
class EpochResult:
    loss: float
    metrics: dict[str, float]


class TrainLoop:
    """Minimal epoch loop with checkpointing and W&B logging hooks."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        config: TrainConfig,
        logger: WandbLogger,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        evaluate: Callable[[nn.Module, DataLoader], dict[str, float]] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        self.loss_fn = loss_fn
        self.evaluate = evaluate
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        self.model.to(self.device)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            inputs, lengths, targets = self._to_device(batch)
            logits = self.model(inputs, lengths)
            loss = self.loss_fn(logits, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _to_device(self, batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        inputs, lengths, targets = batch
        return (
            inputs.to(self.device),
            lengths.to(self.device),
            targets.to(self.device),
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> list[EpochResult]:
        history: list[EpochResult] = []
        checkpoint_dir = self.config.checkpoint_dir

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            metrics: dict[str, float] = {"train/loss": train_loss}

            if val_loader is not None and self.evaluate is not None:
                val_metrics = self.evaluate(self.model, val_loader)
                metrics.update({f"val/{name}": value for name, value in val_metrics.items()})

            self.logger.log(metrics, step=epoch)
            result = EpochResult(loss=train_loss, metrics=metrics)
            history.append(result)

            if epoch % self.config.checkpoint_every_epochs == 0:
                ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                save_checkpoint(
                    ckpt_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    config=self.config.to_dict(),
                )
                self._prune_checkpoints(checkpoint_dir)

        return history

    def _prune_checkpoints(self, checkpoint_dir: Path) -> None:
        keep = self.config.keep_last_n_checkpoints
        if keep <= 0:
            return
        checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
        for stale in checkpoints[:-keep]:
            stale.unlink(missing_ok=True)
