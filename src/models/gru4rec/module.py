"""GRU4Rec LightningModule with ranking metrics on validation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from src.evaluation.baselines import popularity_top_k_gru_indices
from src.evaluation.metrics import DEFAULT_KS
from src.models.gru4rec.model import GRU4Rec
from src.training.base_module import NextItemLitModule


class GRU4RecLitModule(NextItemLitModule):
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
        compute_pop_baseline: bool = True,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            metric_ks=metric_ks,
            pop_baseline_metrics=pop_baseline_metrics,
            compute_pop_baseline=compute_pop_baseline,
        )
        self.save_hyperparameters(ignore=("pop_baseline_metrics",))
        self.model = GRU4Rec(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )

    def forward(self, item_ids: Tensor, lengths: Tensor) -> Tensor:
        return self.model(item_ids, lengths)

    def compute_logits_and_targets(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        item_ids, lengths, targets = batch
        return self(item_ids, lengths), targets

    def popularity_indices(self, processed_dir: Path) -> list[int]:
        return popularity_top_k_gru_indices(processed_dir, k=self.pop_baseline_k)
