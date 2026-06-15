"""Tests for TGNModel forward / memory updates."""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.tgn.dataset import TGNExampleBatch, load_events_tensors
from src.models.tgn.model import TGNModel
from tests.tgn_fixtures import write_tgn_processed_dir


def _example_batch() -> TGNExampleBatch:
    return TGNExampleBatch(
        session_idx=torch.tensor([0, 1]),
        target_item_idx_tgn=torch.tensor([2, 4]),
        target_t_sec=torch.tensor([2.0, 7.0]),
        target_event_id=torch.tensor([2, 7]),
        prefix_last_event_id=torch.tensor([1, 6]),
    )


def test_tgn_bce_sequential_batches_no_crash() -> None:
    """Regression: wrong GNN edge indexing crashed CUDA after a few BCE steps."""
    model = TGNModel(
        num_items=5,
        num_sessions=2,
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
    )
    model.reset_state()
    for step in range(12):
        session_idx = torch.tensor([step % 2])
        item_idx = torch.tensor([step % 5])
        t_sec = torch.tensor([float(step)])
        msg = torch.zeros(1, 4)
        pos, neg = model.score_pos_neg(session_idx, item_idx, t_sec, msg)
        assert pos.shape == (1,)
        assert neg.shape == (1,)
        model.detach_memory()


def test_tgn_bce_multi_negative_shapes() -> None:
    model = TGNModel(
        num_items=5,
        num_sessions=2,
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
    )
    model.reset_state()
    session_idx = torch.tensor([0, 1])
    item_idx = torch.tensor([1, 2])
    t_sec = torch.tensor([1.0, 2.0])
    msg = torch.zeros(2, 4)
    pos, neg = model.score_pos_neg(
        session_idx,
        item_idx,
        t_sec,
        msg,
        num_negatives=3,
    )
    assert pos.shape == (2,)
    assert neg.shape == (2, 3)


def test_tgn_forward_eval_sampled_matches_full_catalog(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    events = load_events_tensors(processed / "train" / "tgn" / "events.parquet")
    num_items = 5
    model = TGNModel(
        num_items=num_items,
        num_sessions=2,
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
    )
    batch = _example_batch()
    candidate_ids = torch.stack([torch.randperm(num_items) for _ in range(batch.target_item_idx_tgn.size(0))])

    model.reset_state()
    full_logits, targets = model.forward_eval_batch(batch, events, fast_eval=True)
    model.reset_state()
    sampled_scores, sampled_targets = model.forward_eval_sampled(
        batch,
        events,
        candidate_ids,
        fast_eval=True,
    )

    assert torch.equal(targets, sampled_targets)
    assert sampled_scores.shape == candidate_ids.shape
    for row in range(batch.target_item_idx_tgn.size(0)):
        for col in range(num_items):
            item_id = int(candidate_ids[row, col].item())
            assert sampled_scores[row, col] == full_logits[row, item_id]


def test_tgn_bce_sequential_batches_cuda() -> None:
    if not torch.cuda.is_available():
        return
    model = TGNModel(
        num_items=5,
        num_sessions=2,
        embedding_dim=16,
        memory_dim=32,
        time_dim=16,
        n_neighbors=2,
    ).cuda()
    model.reset_state()
    for step in range(12):
        session_idx = torch.tensor([step % 2], device="cuda")
        item_idx = torch.tensor([step % 5], device="cuda")
        t_sec = torch.tensor([float(step)], device="cuda")
        msg = torch.zeros(1, 4, device="cuda")
        pos, neg = model.score_pos_neg(session_idx, item_idx, t_sec, msg)
        assert pos.shape == (1,)
        assert neg.shape == (1,)
        model.detach_memory()
