"""Tests for TGNModel forward / memory updates."""

from __future__ import annotations

import torch

from src.models.tgn.model import TGNModel


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
