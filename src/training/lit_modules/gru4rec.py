"""GRU4Rec LightningModule with ranking metrics on validation."""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
from torch import Tensor, nn

from src.evaluation.metrics import DEFAULT_KS, batch_ranking_metrics
from src.models.gru4rec import GRU4Rec


class GRU4RecLitModule(pl.LightningModule):
    def __init__(
        self,
        num_embeddings: int,
        learning_rate: float = 1e-3,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int = 0,
        metric_ks: tuple[int, ...] = DEFAULT_KS,
        pop_baseline_metrics: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("pop_baseline_metrics",))
        self.model = GRU4Rec(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )
        self.learning_rate = learning_rate
        self.metric_ks = metric_ks
        self.pop_baseline_metrics = pop_baseline_metrics or {}
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, item_ids: Tensor, lengths: Tensor) -> Tensor:
        return self.model(item_ids, lengths)

    def _shared_step(self, batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        item_ids, lengths, targets = batch
        logits = self(item_ids, lengths)
        loss = self.loss_fn(logits, targets)
        return loss, logits, targets

    def training_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        loss, _, _ = self._shared_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        perplexity = torch.exp(loss.detach().clamp(max=20.0))
        self.log("train/perplexity", perplexity, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        loss, logits, targets = self._shared_step(batch)
        batch_size = targets.size(0)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        val_perplexity = torch.exp(loss.detach().clamp(max=20.0))
        self.log(
            "val/perplexity",
            val_perplexity,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        for name, value in batch_ranking_metrics(logits, targets, self.metric_ks).items():
            log_name = f"val/{name}"
            self.log(log_name, value, on_step=False, on_epoch=True, batch_size=batch_size)

    def on_validation_epoch_end(self) -> None:
        for name, value in self.pop_baseline_metrics.items():
            self.log(f"val/{name}", value, prog_bar=name == "recall@20_pop")

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
