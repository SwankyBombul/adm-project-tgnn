"""TGN LightningModule: BCE training, sampled validation and test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.evaluation.baselines import popularity_top_k_tgn_indices
from src.evaluation.metrics import DEFAULT_KS
from src.evaluation.sampled import build_candidate_sets
from src.models.tgn.dataset import TGNEventTensors, TGNExampleBatch, load_events_tensors
from lightning.pytorch.utilities import rank_zero_info

from src.models.tgn.model import TGNModel
from src.training.base_module import EVAL_DATALOADER_NAMES, NextItemLitModule

# Replay train events in chunks during eval warmup (avoids huge tensors + shows progress).
_WARMUP_CHUNK_EVENTS = 50_000
_WARMUP_LOG_EVERY_EVENTS = 100_000


class TGNLitModule(NextItemLitModule):
    def __init__(
        self,
        num_items: int,
        num_sessions_train: int,
        *,
        num_negatives: int = 1,
        eval_num_negatives: int = 99,
        eval_seed: int | None = None,
        memory_dim: int = 172,
        time_dim: int = 100,
        embedding_dim: int = 100,
        n_neighbors: int = 10,
        item_embed_chunk_size: int = 512,
        fast_eval_item_chunk_size: int = 4096,
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
            eval_num_negatives=eval_num_negatives,
            eval_seed=eval_seed,
        )
        self.save_hyperparameters(ignore=("pop_baseline_metrics",))
        self.num_negatives = num_negatives
        self.fast_eval = fast_eval
        self.model = TGNModel(
            num_items=num_items,
            num_sessions=num_sessions_train,
            memory_dim=memory_dim,
            time_dim=time_dim,
            embedding_dim=embedding_dim,
            n_neighbors=n_neighbors,
            item_embed_chunk_size=item_embed_chunk_size,
            fast_eval_item_chunk_size=fast_eval_item_chunk_size,
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
        self.model.set_session_offset(0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_fit_start(self) -> None:
        super().on_fit_start()
        datamodule = self.trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "attach_events_to_module"):
            datamodule.attach_events_to_module(self)

    def _init_validation_candidate_generator(self) -> None:
        base_seed = self._eval_base_seed()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(base_seed)
        self._eval_candidate_generator = generator

    def on_validation_epoch_start(self) -> None:
        datamodule = self.trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "set_eval_split"):
            datamodule.set_eval_split(self, "val")
        self._init_validation_candidate_generator()
        self._warmup_for_eval()
        self.model.reset_replay_cursor()
        if datamodule is not None and hasattr(datamodule, "session_offset"):
            self.model.set_session_offset(datamodule.session_offset("val"))

    def on_test_batch_start(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0:
            datamodule = self.trainer.datamodule
            split = EVAL_DATALOADER_NAMES[dataloader_idx]
            if datamodule is not None and hasattr(datamodule, "set_eval_split"):
                datamodule.set_eval_split(self, split)
            self._warmup_for_eval()
            self.model.reset_replay_cursor()
            if datamodule is not None and hasattr(datamodule, "session_offset"):
                self.model.set_session_offset(datamodule.session_offset(split))
        super().on_test_batch_start(batch, batch_idx, dataloader_idx)

    def _warmup_for_eval(self) -> None:
        self.model.reset_state()
        self.model.set_session_offset(0)
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

    def compute_sampled_scores_and_targets(
        self,
        batch: TGNExampleBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(batch, TGNExampleBatch):
            raise TypeError(f"Expected TGNExampleBatch, got {type(batch)}")
        if self._eval_events is None:
            raise RuntimeError("eval event tensors not set on TGNLitModule")
        generator = self._eval_candidate_generator
        if generator is None:
            raise RuntimeError("test candidate generator not initialized")

        candidates = build_candidate_sets(
            batch.target_item_idx_tgn,
            self.model.num_items,
            self.eval_num_negatives,
            generator=generator,
        )
        scores, targets = self.model.forward_eval_sampled(
            batch,
            self._eval_events,
            candidates.ids,
            fast_eval=self.fast_eval,
        )
        return scores, targets, candidates.ids

    def popularity_indices(self, processed_dir: Path) -> list[int]:
        return popularity_top_k_tgn_indices(processed_dir, k=self.pop_baseline_k)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        pos_logits, neg_logits = self.model.score_pos_neg(
            batch["session_idx"],
            batch["item_idx_tgn"],
            batch["t_sec"],
            batch["msg"],
            num_negatives=self.num_negatives,
        )
        loss = self._bce_criterion(pos_logits, torch.ones_like(pos_logits))
        loss = loss + self._bce_criterion(neg_logits, torch.zeros_like(neg_logits))
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self.model.detach_memory()

    def on_validation_model_eval(self) -> None:
        # Drop pending raw messages before Lightning calls model.eval().
        self.model.memory._reset_message_store()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        if not isinstance(batch, TGNExampleBatch):
            raise TypeError(f"Expected TGNExampleBatch, got {type(batch)}")
        scores, targets, candidate_ids = self.compute_sampled_scores_and_targets(batch)
        self._log_sampled_split_metrics(
            "val",
            scores,
            candidate_ids,
            targets,
            targets.size(0),
            prog_bar=True,
        )

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def load_split_events(path: Path, device: torch.device) -> TGNEventTensors:
    return load_events_tensors(path).to(device)
