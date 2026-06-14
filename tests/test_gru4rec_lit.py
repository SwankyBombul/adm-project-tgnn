"""Tests for GRU4Rec LightningModule."""

from __future__ import annotations

import torch

from src.evaluation.sampled import build_candidate_sets
from src.models.gru4rec import GRU4RecLitModule


def test_gru4rec_lit_module_compute_logits_and_targets() -> None:
    module = GRU4RecLitModule(num_embeddings=50, learning_rate=1e-3)
    item_ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
    lengths = torch.tensor([2, 1])
    targets = torch.tensor([4, 5])

    logits, out_targets = module.compute_logits_and_targets((item_ids, lengths, targets))
    loss = module.compute_loss(logits, out_targets)

    assert loss.ndim == 0
    assert logits.shape == (2, 50)
    assert out_targets.tolist() == targets.tolist()


def test_gru4rec_score_candidates_shape() -> None:
    module = GRU4RecLitModule(num_embeddings=50, eval_num_negatives=3, eval_seed=0)
    item_ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
    lengths = torch.tensor([2, 1])
    targets = torch.tensor([4, 5])
    generator = torch.Generator().manual_seed(0)
    candidates = build_candidate_sets(targets, 50, 3, generator=generator)
    scores = module.model.score_candidates(item_ids, lengths, candidates.ids)
    assert scores.shape == (2, 4)


def test_gru4rec_sampled_scores_and_targets() -> None:
    module = GRU4RecLitModule(num_embeddings=50, eval_num_negatives=3, eval_seed=0)
    item_ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
    lengths = torch.tensor([2, 1])
    targets = torch.tensor([4, 5])
    module._eval_candidate_generator = torch.Generator().manual_seed(0)
    scores, out_targets, candidate_ids = module.compute_sampled_scores_and_targets(
        (item_ids, lengths, targets)
    )
    assert scores.shape == candidate_ids.shape == (2, 4)
    assert out_targets.tolist() == targets.tolist()
