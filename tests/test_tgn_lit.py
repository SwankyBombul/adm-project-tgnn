"""Tests for TGN LightningModule."""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.tgn.dataset import TGNExampleBatch, load_events_tensors
from src.models.tgn.module import TGNLitModule
from tests.tgn_fixtures import write_tgn_processed_dir


def _example_batch() -> TGNExampleBatch:
    return TGNExampleBatch(
        session_idx=torch.tensor([0, 1]),
        target_item_idx_tgn=torch.tensor([2, 4]),
        target_t_sec=torch.tensor([2.0, 7.0]),
        target_event_id=torch.tensor([2, 7]),
        prefix_last_event_id=torch.tensor([1, 6]),
    )


def test_tgn_lit_bce_training_step() -> None:
    module = TGNLitModule(num_items=5, num_sessions_train=2, loss_mode="bce", embedding_dim=16, memory_dim=32, time_dim=16, n_neighbors=2)
    batch = {
        "session_idx": torch.tensor([0, 1]),
        "item_idx_tgn": torch.tensor([1, 2]),
        "t_sec": torch.tensor([1.0, 4.0]),
        "msg": torch.zeros(2, 4),
    }
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0


def test_tgn_lit_ce_logits_shape(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    events = load_events_tensors(processed / "train" / "tgn" / "events.parquet")
    module = TGNLitModule(
        num_items=5,
        num_sessions_train=2,
        loss_mode="ce",
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
    )
    module.set_event_tensors(train_events=events, eval_events=events)
    logits, targets = module.model.forward_ce_examples(_example_batch(), events)
    assert logits.shape == (2, 5)
    assert targets.tolist() == [2, 4]


def test_tgn_lit_eval_logits_shape(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    events = load_events_tensors(processed / "train" / "tgn" / "events.parquet")
    module = TGNLitModule(
        num_items=5,
        num_sessions_train=2,
        loss_mode="bce",
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
        fast_eval=True,
    )
    module.set_event_tensors(eval_events=events)
    logits, targets = module.compute_logits_and_targets(_example_batch())
    assert logits.shape == (2, 5)
    assert targets.tolist() == [2, 4]


def test_tgn_fast_eval_matches_shapes(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    events = load_events_tensors(processed / "val" / "tgn" / "events.parquet")
    module = TGNLitModule(
        num_items=5,
        num_sessions_train=2,
        loss_mode="bce",
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
    )
    module.set_event_tensors(train_events=events, eval_events=events)
    module.model.reset_state()
    batch = _example_batch()
    full_logits, targets = module.model.forward_eval_batch(batch, events, fast_eval=False)
    fast_logits, _ = module.model.forward_eval_batch(batch, events, fast_eval=True)
    assert full_logits.shape == fast_logits.shape == (2, 5)
    assert full_logits.dtype == torch.float32
    assert fast_logits.dtype == torch.float32
