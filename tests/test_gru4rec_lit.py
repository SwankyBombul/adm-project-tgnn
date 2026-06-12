"""Tests for GRU4Rec LightningModule."""

from __future__ import annotations

import torch

from src.models.gru4rec import GRU4RecLitModule


def test_gru4rec_lit_module_forward_step() -> None:
    module = GRU4RecLitModule(num_embeddings=50, learning_rate=1e-3)
    item_ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
    lengths = torch.tensor([2, 1])
    targets = torch.tensor([4, 5])

    loss, logits, _ = module._shared_step((item_ids, lengths, targets))
    assert loss.ndim == 0
    assert logits.shape == (2, 50)
