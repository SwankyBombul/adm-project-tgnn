"""Tests for TAGNN LightningModule."""

from __future__ import annotations

import torch

from src.evaluation.sampled import build_candidate_sets
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


def test_tagnn_compute_logits_for_candidates_shape() -> None:
    module = TAGNNLitModule(num_embeddings=20, hidden_dim=16, gnn_steps=1)
    batch = tagnn_collate_fn([([1, 2, 3], 4), ([2, 3], 5)])
    alias_inputs, adjacency, items, mask, targets = batch
    node_hidden = module.model(items, adjacency)
    seq_hidden = module.model.sequence_hidden(node_hidden, alias_inputs)
    generator = torch.Generator().manual_seed(0)
    candidates = build_candidate_sets(targets, 20, 2, generator=generator)
    scores = module.model.compute_logits_for_candidates(seq_hidden, mask, candidates.ids)
    assert scores.shape == (2, 3)


def test_tagnn_sampled_scores_and_targets() -> None:
    module = TAGNNLitModule(
        num_embeddings=20, hidden_dim=16, gnn_steps=1, eval_num_negatives=2, eval_seed=0
    )
    batch = tagnn_collate_fn([([1, 2, 3], 4), ([2, 3], 5)])
    module._eval_candidate_generator = torch.Generator().manual_seed(0)
    scores, targets, candidate_ids = module.compute_sampled_scores_and_targets(batch)
    assert scores.shape == candidate_ids.shape == (2, 3)
    assert targets.tolist() == [4, 5]
