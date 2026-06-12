"""Tests for ranking metrics."""

from __future__ import annotations

import pytest
import torch

from src.evaluation.baselines import pop_recall_at_k
from src.evaluation.metrics import (
    batch_ranking_metrics,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k_perfect_ranking() -> None:
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.1, 0.1],
        ]
    )
    targets = torch.tensor([1, 0])

    assert recall_at_k(scores, targets, k=1) == 1.0
    assert recall_at_k(scores, targets, k=5) == 1.0


def test_mrr_at_k_first_and_second_position() -> None:
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.9, 0.4, 0.8],
        ]
    )
    targets = torch.tensor([1, 2])

    assert mrr_at_k(scores, targets, k=2) == 0.75


def test_ndcg_at_k_top_one() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.2]])
    targets = torch.tensor([1])
    assert ndcg_at_k(scores, targets, k=20) == 1.0


def test_batch_ranking_metrics_keys() -> None:
    scores = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    metrics = batch_ranking_metrics(scores, targets, ks=(1, 5, 10, 20))

    assert "hit@1" in metrics
    assert "recall@5" in metrics
    assert "recall@10" in metrics
    assert "recall@20" in metrics
    assert "mrr@20" in metrics
    assert "ndcg@20" in metrics


def test_pop_recall_constant_recommendation() -> None:
    targets = torch.tensor([1, 2, 99])
    pop = [1, 2, 3]
    assert pop_recall_at_k(targets, pop, k=2) == pytest.approx(2 / 3)
