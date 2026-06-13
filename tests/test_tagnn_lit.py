"""Tests for TAGNN LightningModule."""

from __future__ import annotations

import torch

from src.models.tagnn import TAGNNLitModule
from src.models.tagnn.graph_batch import tagnn_collate_fn


def test_tagnn_lit_module_logits_shape() -> None:
    module = TAGNNLitModule(num_embeddings=20, hidden_dim=16, gnn_steps=1)
    batch = tagnn_collate_fn([([1, 2, 3], 4), ([2, 3], 5)])

    logits, targets = module.compute_logits_and_targets(batch)
    loss = module.compute_loss(logits, targets)

    assert logits.shape == (2, 20)
    assert targets.tolist() == [4, 5]
    assert loss.ndim == 0
