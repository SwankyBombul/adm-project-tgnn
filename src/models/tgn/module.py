"""TGN LightningModule with configurable CE / BCE training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from src.evaluation.baselines import popularity_top_k_tgn_indices
from src.evaluation.metrics import DEFAULT_KS
from src.models.tgn.dataset import TGNEventTensors, TGNExampleBatch, load_events_tensors
from lightning.pytorch.utilities import rank_zero_info

from src.models.tgn.model import TGNModel
from src.training.base_module import NextItemLitModule

LossMode = Literal["ce", "bce"]

# Replay train events in chunks during eval warmup (avoids huge tensors + shows progress).
_WARMUP_CHUNK_EVENTS = 50_000
_WARMUP_LOG_EVERY_EVENTS = 100_000


class TGNLitModule(NextItemLitModule):
    def __init__(
        self,
        num_items: int,
        num_sessions_train: int,
        *,
        loss_mode: LossMode = "bce",
        num_negatives: int = 1,
        memory_dim: int = 172,
        time_dim: int = 100,
        embedding_dim: int = 100,
        n_neighbors: int = 10,
        item_embed_chunk_size: int = 512,
        fast_eval: bool = True,
        learning_rate: float = 1e-4,
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
        self.loss_mode = loss_mode
        self.num_negatives = num_negatives
        self.fast_eval = fast_eval
        self._reuse_bce_train_memory = False
        self.model = TGNModel(
            num_items=num_items,
            num_sessions=num_sessions_train,
            memory_dim=memory_dim,
            time_dim=time_dim,
            embedding_dim=embedding_dim,
            n_neighbors=n_neighbors,
            item_embed_chunk_size=item_embed_chunk_size,
        )
        self._train_events: TGNEventTensors | None = None
        self._eval_events: TGNEventTensors | None = None
        self._bce_criterion = torch.nn.BCEWithLogitsLoss()

    def set_event_tensors(
        self,
        train_events: TGNEventTensors | None = None,
        eval_events: TGNEventTensors | None = None,
    ) -> None:
        if train_events is not None:
            self._train_events = train_events
        if eval_events is not None:
            self._eval_events = eval_events

    def on_train_epoch_start(self) -> None:
        self.model.reset_state()
        self._reuse_bce_train_memory = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_epoch_end(self) -> None:
        if self.loss_mode == "bce":
            self._reuse_bce_train_memory = True

    def on_validation_epoch_end(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_fit_start(self) -> None:
        super().on_fit_start()
        datamodule = self.trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "attach_events_to_module"):
            datamodule.attach_events_to_module(self)

    def on_validation_epoch_start(self) -> None:
        datamodule = self.trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "set_eval_split"):
            datamodule.set_eval_split(self, "val")
        if self.loss_mode == "bce" and self._reuse_bce_train_memory:
            rank_zero_info(
                "TGN eval: reusing BCE train memory (skipping train warmup replay)"
            )
            self._reuse_bce_train_memory = False
            return
        self._warmup_for_eval()

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx != 0:
            return
        datamodule = self.trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "set_eval_split"):
            return
        from src.training.base_module import EVAL_DATALOADER_NAMES

        split = EVAL_DATALOADER_NAMES[dataloader_idx]
        datamodule.set_eval_split(self, split)
        self._warmup_for_eval()

    def _warmup_for_eval(self) -> None:
        self.model.reset_state()
        if self._train_events is None:
            return
        max_eid = int(self._train_events.event_id.max().item())
        if max_eid < 0:
            return
        rank_zero_info(
            f"TGN eval warmup: replaying {max_eid + 1:,} train events "
            f"(chunk size {_WARMUP_CHUNK_EVENTS:,})..."
        )
        pos = 0
        while pos <= max_eid:
            end = min(pos + _WARMUP_CHUNK_EVENTS - 1, max_eid)
            self.model.replay_events_up_to(self._train_events, end)
            if end + 1 >= max_eid + 1 or (end + 1) % _WARMUP_LOG_EVERY_EVENTS == 0:
                rank_zero_info(f"TGN eval warmup: {end + 1:,} / {max_eid + 1:,} events")
            pos = end + 1

    def compute_logits_and_targets(
        self,
        batch: TGNExampleBatch | tuple,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(batch, TGNExampleBatch):
            raise TypeError(f"Expected TGNExampleBatch, got {type(batch)}")
        if self._eval_events is None:
            raise RuntimeError("eval event tensors not set on TGNLitModule")
        return self.model.forward_eval_batch(
            batch,
            self._eval_events,
            fast_eval=self.fast_eval,
        )

    def popularity_indices(self, processed_dir: Path) -> list[int]:
        return popularity_top_k_tgn_indices(processed_dir, k=self.pop_baseline_k)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        if self.loss_mode == "ce":
            if self._eval_events is None:
                raise RuntimeError("CE training requires train event tensors on module")
            logits, targets = self.model.forward_ce_examples(batch, self._eval_events)
            loss = self.compute_loss(logits, targets)
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self._log_perplexity("train", loss, targets.size(0))
            return loss

        event_batch = batch
        pos_logits, neg_logits = self.model.score_pos_neg(
            event_batch["session_idx"],
            event_batch["item_idx_tgn"],
            event_batch["t_sec"],
            event_batch["msg"],
            num_negatives=self.num_negatives,
        )
        loss = self._bce_criterion(pos_logits, torch.ones_like(pos_logits))
        loss = loss + self._bce_criterion(neg_logits, torch.zeros_like(neg_logits))
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.loss_mode in ("bce", "ce"):
            self.model.detach_memory()

    def on_validation_model_eval(self) -> None:
        # Drop pending raw messages before Lightning calls model.eval().
        self.model.memory._reset_message_store()

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

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def load_split_events(path: Path, device: torch.device) -> TGNEventTensors:
    return load_events_tensors(path).to(device)
