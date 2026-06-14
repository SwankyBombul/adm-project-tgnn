"""Shared LightningModule skeleton for next-item recommendation models."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from torch import Tensor, nn

from src.evaluation.baselines import evaluate_pop_baseline
from src.evaluation.metrics import DEFAULT_KS, batch_ranking_metrics
from src.evaluation.sampled import batch_sampled_ranking_metrics

EVAL_DATALOADER_NAMES = ("test_internal", "challenge_test")
POP_BASELINE_KS = (5, 10, 20)


class NextItemLitModule(pl.LightningModule):
    """Base class: train/val during ``fit``, held-out eval via Lightning ``test`` (CLI ``evaluate``)."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        metric_ks: tuple[int, ...] = DEFAULT_KS,
        pop_baseline_metrics: dict[str, float] | None = None,
        compute_pop_baseline: bool = True,
        pop_baseline_k: int = 20,
        eval_num_negatives: int = 99,
        eval_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.metric_ks = metric_ks
        self.compute_pop_baseline = compute_pop_baseline
        self.pop_baseline_k = pop_baseline_k
        self.pop_baseline_metrics = pop_baseline_metrics or {}
        self._eval_pop_baselines: dict[str, dict[str, float]] = {}
        self.eval_num_negatives = eval_num_negatives
        self.eval_seed = eval_seed
        self._eval_candidate_generator: torch.Generator | None = None
        self.loss_fn = nn.CrossEntropyLoss()

    @abstractmethod
    def compute_logits_and_targets(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Map a batch to full-catalog logits and target item indices."""

    @abstractmethod
    def compute_sampled_scores_and_targets(self, batch: Any) -> tuple[Tensor, Tensor, Tensor]:
        """Map a batch to sampled scores, targets, and candidate item indices."""

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.loss_fn(logits, targets)

    @abstractmethod
    def popularity_indices(self, processed_dir: Path) -> list[int]:
        """Return item indices for the global popularity baseline."""

    def _processed_dir_from_datamodule(self) -> Path | None:
        datamodule = self.trainer.datamodule
        processed_dir = getattr(datamodule, "processed_dir", None)
        return Path(processed_dir) if processed_dir is not None else None

    def _compute_pop_baseline(self, dataloader: Any) -> dict[str, float]:
        processed_dir = self._processed_dir_from_datamodule()
        if processed_dir is None:
            return {}
        pop_indices = self.popularity_indices(processed_dir)
        return evaluate_pop_baseline(
            dataloader,
            pop_indices,
            ks=POP_BASELINE_KS,
        )

    def _eval_base_seed(self) -> int:
        if self.eval_seed is not None:
            return int(self.eval_seed)
        return int(torch.initial_seed())

    def _init_test_candidate_generator(self, dataloader_idx: int) -> None:
        base_seed = self._eval_base_seed()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(base_seed + int(dataloader_idx))
        self._eval_candidate_generator = generator

    def on_fit_start(self) -> None:
        if not self.compute_pop_baseline or self.pop_baseline_metrics:
            return
        datamodule = self.trainer.datamodule
        if datamodule is None:
            return
        self.pop_baseline_metrics = self._compute_pop_baseline(datamodule.val_dataloader())

    def on_test_start(self) -> None:
        if not self.compute_pop_baseline:
            return
        datamodule = self.trainer.datamodule
        if datamodule is None:
            return
        loaders = datamodule.test_dataloader()
        if not isinstance(loaders, list):
            loaders = [loaders]
        for name, loader in zip(EVAL_DATALOADER_NAMES, loaders, strict=True):
            self._eval_pop_baselines[name] = self._compute_pop_baseline(loader)

    def on_test_batch_start(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0:
            self._init_test_candidate_generator(dataloader_idx)

    def _log_perplexity(self, prefix: str, loss: Tensor, batch_size: int) -> None:
        perplexity = torch.exp(loss.detach().clamp(max=20.0))
        self.log(
            f"{prefix}/perplexity",
            perplexity,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def _log_split_metrics(
        self,
        prefix: str,
        loss: Tensor,
        logits: Tensor,
        targets: Tensor,
        batch_size: int,
        *,
        prog_bar_loss: bool = False,
    ) -> None:
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar_loss,
            batch_size=batch_size,
        )
        self._log_perplexity(prefix, loss, batch_size)
        for name, value in batch_ranking_metrics(logits, targets, self.metric_ks).items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

    def _log_sampled_split_metrics(
        self,
        prefix: str,
        scores: Tensor,
        candidate_ids: Tensor,
        targets: Tensor,
        batch_size: int,
        *,
        prog_bar_recall20: bool = False,
    ) -> None:
        for name, value in batch_sampled_ranking_metrics(
            scores,
            candidate_ids,
            targets,
            self.metric_ks,
        ).items():
            self.log(
                f"{prefix}/sampled_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar_recall20 and name == "recall@20",
                batch_size=batch_size,
            )

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        logits, targets = self.compute_logits_and_targets(batch)
        loss = self.compute_loss(logits, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_perplexity("train", loss, targets.size(0))
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        logits, targets = self.compute_logits_and_targets(batch)
        loss = self.compute_loss(logits, targets)
        self._log_split_metrics(
            "val",
            loss,
            logits,
            targets,
            targets.size(0),
            prog_bar_loss=True,
        )

    def on_validation_epoch_end(self) -> None:
        for name, value in self.pop_baseline_metrics.items():
            self.log(f"val/{name}", value, prog_bar=name == "recall@20_pop")

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        prefix = EVAL_DATALOADER_NAMES[dataloader_idx]
        scores, targets, candidate_ids = self.compute_sampled_scores_and_targets(batch)
        self._log_sampled_split_metrics(
            prefix,
            scores,
            candidate_ids,
            targets,
            targets.size(0),
            prog_bar_recall20=prefix == "challenge_test",
        )

    def on_test_epoch_end(self) -> None:
        for split_name, metrics in self._eval_pop_baselines.items():
            for name, value in metrics.items():
                self.log(
                    f"{split_name}/{name}",
                    value,
                    prog_bar=split_name == "challenge_test" and name == "recall@20_pop",
                )

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
