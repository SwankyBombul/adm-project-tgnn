"""Tests for TGN LightningModule."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch

from src.evaluation.sampled import build_candidate_sets
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
    module = TGNLitModule(
        num_items=5,
        num_sessions_train=2,
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
    )
    batch = {
        "session_idx": torch.tensor([0, 1]),
        "item_idx_tgn": torch.tensor([1, 2]),
        "t_sec": torch.tensor([1.0, 4.0]),
        "msg": torch.zeros(2, 4),
    }
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0


def test_tgn_lit_bce_training_step_multi_negative() -> None:
    module = TGNLitModule(
        num_items=5,
        num_sessions_train=2,
        num_negatives=4,
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
    )
    batch = {
        "session_idx": torch.tensor([0, 1]),
        "item_idx_tgn": torch.tensor([1, 2]),
        "t_sec": torch.tensor([1.0, 4.0]),
        "msg": torch.zeros(2, 4),
    }
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0


def test_tgn_lit_sampled_validation_scores_shape(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    events = load_events_tensors(processed / "train" / "tgn" / "events.parquet")
    module = TGNLitModule(
        num_items=5,
        num_sessions_train=2,
        eval_num_negatives=2,
        eval_seed=0,
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
        fast_eval=False,
    )
    module.set_event_tensors(eval_events=events)
    module._eval_candidate_generator = torch.Generator().manual_seed(0)
    module.model.reset_state()

    scores, targets, candidate_ids = module.compute_sampled_scores_and_targets(_example_batch())
    assert scores.shape == (2, 3)
    assert candidate_ids.shape == (2, 3)
    assert targets.tolist() == [2, 4]


def test_tgn_lit_sampled_candidates_reproducible() -> None:
    targets = torch.tensor([1, 3])
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    first = build_candidate_sets(targets, num_items=6, num_negatives=2, generator=gen_a)
    second = build_candidate_sets(targets, num_items=6, num_negatives=2, generator=gen_b)
    assert torch.equal(first.ids, second.ids)


def test_tgn_lit_full_catalog_eval_still_available(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    events = load_events_tensors(processed / "train" / "tgn" / "events.parquet")
    module = TGNLitModule(
        num_items=5,
        num_sessions_train=2,
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
        fast_eval=False,
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


def test_tgn_lit_validation_step_logs_sampled_metrics(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    events = load_events_tensors(processed / "train" / "tgn" / "events.parquet")
    module = TGNLitModule(
        num_items=5,
        num_sessions_train=2,
        eval_num_negatives=2,
        eval_seed=0,
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
        fast_eval=False,
    )
    module.set_event_tensors(train_events=events, eval_events=events)
    module.trainer = MagicMock()
    module._init_validation_candidate_generator()
    module.model.reset_state()

    logged: dict[str, torch.Tensor] = {}
    module.log = lambda name, value, **kwargs: logged.update({name: value})  # noqa: ARG005

    module.validation_step(_example_batch(), batch_idx=0)
    assert "val/sampled_recall@20" in logged
    assert "val/loss" not in logged


def test_tgn_lit_test_step_logs_sampled_metrics(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    events = load_events_tensors(processed / "train" / "tgn" / "events.parquet")
    module = TGNLitModule(
        num_items=5,
        num_sessions_train=2,
        eval_num_negatives=2,
        eval_seed=0,
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
        fast_eval=False,
    )
    module.set_event_tensors(train_events=events, eval_events=events)
    module.trainer = MagicMock()
    module.trainer.datamodule = None
    module._eval_candidate_generator = torch.Generator().manual_seed(0)
    module.model.reset_state()

    logged: dict[str, torch.Tensor] = {}
    module.log = lambda name, value, **kwargs: logged.update({name: value})  # noqa: ARG005

    batch = _example_batch()
    module.test_step(batch, batch_idx=0, dataloader_idx=0)
    assert "test_internal/sampled_recall@20" in logged


def test_tgn_lit_load_expanded_memory_checkpoint() -> None:
    small = TGNLitModule(
        num_items=5,
        num_sessions_train=2,
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
    )
    large = TGNLitModule(
        num_items=5,
        num_sessions_train=6,
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
    )
    checkpoint = small.state_dict()
    large.load_state_dict(checkpoint, strict=False)
    prefix = checkpoint["model.memory.memory"].shape[0]
    assert torch.allclose(
        large.model.memory.memory[:prefix],
        small.model.memory.memory,
    )
    assert large.model.memory.memory.shape[0] == 5 + 6
