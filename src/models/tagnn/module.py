"""TAGNN LightningModule."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim.lr_scheduler import StepLR

from src.evaluation.baselines import popularity_top_k_gru_indices
from src.evaluation.metrics import DEFAULT_KS
from src.evaluation.sampled import build_candidate_sets
from src.models.tagnn.model import TAGNN
from src.training.base_module import NextItemLitModule


class TAGNNLitModule(NextItemLitModule):
    def __init__(
        self,
        num_embeddings: int,
        learning_rate: float = 1e-3,
        hidden_dim: int = 100,
        gnn_steps: int = 1,
        nonhybrid: bool = False,
        pad_idx: int = 0,
        weight_decay: float = 1e-5,
        lr_decay_step: int = 3,
        lr_decay_gamma: float = 0.1,
        metric_ks: tuple[int, ...] = DEFAULT_KS,
        pop_baseline_metrics: dict[str, float] | None = None,
        compute_pop_baseline: bool = True,
        eval_num_negatives: int = 99,
        eval_seed: int | None = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            metric_ks=metric_ks,
            pop_baseline_metrics=pop_baseline_metrics,
            compute_pop_baseline=compute_pop_baseline,
            eval_num_negatives=eval_num_negatives,
            eval_seed=eval_seed,
        )
        self.save_hyperparameters(ignore=("pop_baseline_metrics",))
        self.weight_decay = weight_decay
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma
        self.model = TAGNN(
            num_embeddings=num_embeddings,
            hidden_dim=hidden_dim,
            gnn_steps=gnn_steps,
            nonhybrid=nonhybrid,
            pad_idx=pad_idx,
        )

    def compute_logits_and_targets(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        alias_inputs, adjacency, items, mask, targets = batch
        node_hidden = self.model(items, adjacency)
        seq_hidden = self.model.sequence_hidden(node_hidden, alias_inputs)
        logits = self.model.compute_logits(seq_hidden, mask)
        return logits, targets

    def compute_sampled_scores_and_targets(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        alias_inputs, adjacency, items, mask, targets = batch
        generator = self._eval_candidate_generator
        if generator is None:
            raise RuntimeError("test candidate generator not initialized")
        candidates = build_candidate_sets(
            targets,
            self.model.num_embeddings,
            self.eval_num_negatives,
            generator=generator,
        )
        node_hidden = self.model(items, adjacency)
        seq_hidden = self.model.sequence_hidden(node_hidden, alias_inputs)
        scores = self.model.compute_logits_for_candidates(seq_hidden, mask, candidates.ids)
        return scores, targets, candidates.ids

    def popularity_indices(self, processed_dir: Path) -> list[int]:
        return popularity_top_k_gru_indices(processed_dir, k=self.pop_baseline_k)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = StepLR(
            optimizer,
            step_size=self.lr_decay_step,
            gamma=self.lr_decay_gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
