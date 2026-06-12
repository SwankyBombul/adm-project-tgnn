"""Session recommendation ranking metrics (full-catalog, single target)."""

from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_KS = (1, 5, 10, 20)


def _validate_inputs(scores: Tensor, targets: Tensor) -> None:
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got shape {tuple(scores.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be 1D, got shape {tuple(targets.shape)}")
    if scores.size(0) != targets.size(0):
        raise ValueError("scores and targets batch sizes must match")


def recall_at_k(scores: Tensor, targets: Tensor, k: int) -> float:
    """Recall@K / Hit Rate@K for a single ground-truth item per row."""
    return float(batch_recall_at_k(scores, targets, k).item())


def mrr_at_k(scores: Tensor, targets: Tensor, k: int) -> float:
    """Mean Reciprocal Rank@K."""
    return float(batch_mrr_at_k(scores, targets, k).item())


def ndcg_at_k(scores: Tensor, targets: Tensor, k: int) -> float:
    """NDCG@K with a single relevant item per row (IDCG = 1)."""
    return float(batch_ndcg_at_k(scores, targets, k).item())


def batch_recall_at_k(scores: Tensor, targets: Tensor, k: int) -> Tensor:
    _validate_inputs(scores, targets)
    topk = scores.topk(min(k, scores.size(1)), dim=1).indices
    hits = (topk == targets.unsqueeze(1)).any(dim=1)
    return hits.float().mean()


def batch_mrr_at_k(scores: Tensor, targets: Tensor, k: int) -> Tensor:
    _validate_inputs(scores, targets)
    topk = scores.topk(min(k, scores.size(1)), dim=1).indices
    matches = topk == targets.unsqueeze(1)
    has_hit = matches.any(dim=1)
    ranks = matches.int().argmax(dim=1).float() + 1.0
    reciprocal = torch.where(has_hit, 1.0 / ranks, torch.zeros_like(ranks))
    return reciprocal.mean()


def batch_ndcg_at_k(scores: Tensor, targets: Tensor, k: int) -> Tensor:
    _validate_inputs(scores, targets)
    topk = scores.topk(min(k, scores.size(1)), dim=1).indices
    matches = topk == targets.unsqueeze(1)
    has_hit = matches.any(dim=1)
    ranks = matches.int().argmax(dim=1).float() + 1.0
    ndcg = torch.where(has_hit, 1.0 / torch.log2(ranks + 1.0), torch.zeros_like(ranks))
    return ndcg.mean()


def batch_ranking_metrics(
    scores: Tensor,
    targets: Tensor,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, Tensor]:
    """Compute recall, MRR, and NDCG for multiple values of K."""
    metrics: dict[str, Tensor] = {}
    for k in ks:
        metrics[f"recall@{k}"] = batch_recall_at_k(scores, targets, k)
        if k == 1:
            metrics["hit@1"] = metrics["recall@1"]
        if k == 20:
            metrics["mrr@20"] = batch_mrr_at_k(scores, targets, k)
            metrics["ndcg@20"] = batch_ndcg_at_k(scores, targets, k)
    return metrics
