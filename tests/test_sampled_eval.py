"""Tests for sampled evaluation helpers."""

from __future__ import annotations

import torch

from src.evaluation.sampled import (
    batch_sampled_ranking_metrics,
    build_candidate_sets,
    sample_negative_items,
)


def test_build_candidate_sets_shape_and_contains_target() -> None:
    targets = torch.tensor([1, 3])
    candidates = build_candidate_sets(targets, num_items=5, num_negatives=2, generator=torch.Generator().manual_seed(0))
    assert candidates.ids.shape == (2, 3)
    assert candidates.num_negatives == 2
    for row, target in enumerate(targets.tolist()):
        assert int(target) in candidates.ids[row].tolist()
        assert len(set(candidates.ids[row].tolist())) == 3


def test_build_candidate_sets_reproducible_with_generator() -> None:
    targets = torch.tensor([1, 3, 0])
    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(42)
    first = build_candidate_sets(targets, num_items=6, num_negatives=3, generator=gen_a)
    second = build_candidate_sets(targets, num_items=6, num_negatives=3, generator=gen_b)
    assert torch.equal(first.ids, second.ids)


def test_build_candidate_sets_shuffles_columns() -> None:
    targets = torch.tensor([2, 2, 2])
    gen = torch.Generator().manual_seed(7)
    candidates = build_candidate_sets(targets, num_items=5, num_negatives=2, generator=gen)
    unshuffled = torch.tensor([[2, 0, 1]] * 3)
    assert not torch.equal(candidates.ids, unshuffled)


def test_sample_negative_items_single_and_multi() -> None:
    targets = torch.tensor([0, 1])
    single = sample_negative_items(targets, num_items=4, num_negatives=1)
    assert single.shape == (2,)
    multi = sample_negative_items(targets, num_items=4, num_negatives=3)
    assert multi.shape == (2, 3)


def test_sample_negative_items_popularity_weighted() -> None:
    targets = torch.tensor([0, 1, 2, 3])
    weights = torch.tensor([0.01, 0.01, 0.97, 0.01], dtype=torch.float32)
    gen = torch.Generator().manual_seed(123)
    negatives = sample_negative_items(
        targets,
        num_items=4,
        num_negatives=2,
        sampling="popularity",
        popularity_weights=weights,
        generator=gen,
    )
    assert negatives.shape == (4, 2)
    assert int((negatives == 2).sum().item()) >= 4


def test_batch_sampled_ranking_metrics_perfect_and_miss() -> None:
    candidate_ids = torch.tensor(
        [
            [1, 0, 2],
            [0, 1, 2],
        ]
    )
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.1, 0.1],
        ]
    )
    targets = torch.tensor([1, 2])

    metrics = batch_sampled_ranking_metrics(scores, candidate_ids, targets, ks=(1, 5, 10, 20))
    assert metrics["recall@1"] == 0.0
    assert metrics["hit@1"] == metrics["recall@1"]
    assert metrics["recall@5"] == 1.0
    assert "mrr@20" in metrics
    assert "ndcg@20" in metrics
