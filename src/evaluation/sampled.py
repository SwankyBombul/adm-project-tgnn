"""Sampled candidate sets and ranking metrics for efficient evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.evaluation.metrics import DEFAULT_KS


@dataclass(frozen=True)
class CandidateSet:
    """Per-row candidate item indices (target plus negatives), column order shuffled."""

    ids: Tensor
    num_negatives: int

    @property
    def num_candidates(self) -> int:
        return int(self.ids.size(1))


def sample_negative_items(
    positive_items: Tensor,
    num_items: int,
    num_negatives: int = 1,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Uniform negative item indices in ``0 .. num_items - 1``."""
    if num_negatives < 1:
        raise ValueError(f"num_negatives must be >= 1, got {num_negatives}")
    batch_size = positive_items.size(0)
    device = positive_items.device
    neg = torch.randint(
        0,
        num_items,
        (batch_size, num_negatives),
        device=device,
        generator=generator,
    )
    for _ in range(3):
        clash = neg.eq(positive_items.unsqueeze(1))
        if not clash.any():
            break
        resample_count = int(clash.sum().item())
        neg[clash] = torch.randint(
            0,
            num_items,
            (resample_count,),
            device=device,
            generator=generator,
        )
    return neg.squeeze(1) if num_negatives == 1 else neg


def _unique_negative_rows(
    targets: Tensor,
    negatives: Tensor,
    num_items: int,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    """Ensure each row's negatives are unique and differ from the target."""
    batch_size, num_negatives = negatives.shape
    device = targets.device
    for row in range(batch_size):
        target = int(targets[row].item())
        seen = {target}
        for col in range(num_negatives):
            item = int(negatives[row, col].item())
            attempts = 0
            while item in seen and attempts < num_items * 2:
                item = int(
                    torch.randint(
                        0,
                        num_items,
                        (1,),
                        device=device,
                        generator=generator,
                    ).item()
                )
                attempts += 1
            negatives[row, col] = item
            seen.add(item)
    return negatives


def build_candidate_sets(
    targets: Tensor,
    num_items: int,
    num_negatives: int,
    *,
    generator: torch.Generator | None = None,
) -> CandidateSet:
    """Build shuffled candidate pools: target plus ``num_negatives`` uniform negatives per row."""
    if targets.ndim != 1:
        raise ValueError(f"targets must be 1D, got shape {tuple(targets.shape)}")
    if num_negatives < 1:
        raise ValueError(f"num_negatives must be >= 1, got {num_negatives}")
    if int(targets.max().item()) >= num_items or int(targets.min().item()) < 0:
        raise ValueError("targets must be valid item indices for the catalog")

    negatives = sample_negative_items(
        targets,
        num_items,
        num_negatives,
        generator=generator,
    )
    if num_negatives == 1:
        negatives = negatives.unsqueeze(1)
    negatives = _unique_negative_rows(targets, negatives, num_items, generator=generator)

    ids = torch.cat([targets.unsqueeze(1), negatives], dim=1)
    num_candidates = ids.size(1)
    noise = torch.rand(
        ids.size(0),
        num_candidates,
        device=ids.device,
        generator=generator,
    )
    perm = noise.argsort(dim=1)
    return CandidateSet(ids=ids.gather(1, perm), num_negatives=num_negatives)


def batch_sampled_ranking_metrics(
    scores: Tensor,
    candidate_ids: Tensor,
    target_ids: Tensor,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, Tensor]:
    """Ranking metrics when only a candidate subset is scored per row."""
    if scores.shape != candidate_ids.shape:
        raise ValueError("scores and candidate_ids must have the same shape")
    if scores.size(0) != target_ids.size(0):
        raise ValueError("scores and target_ids batch sizes must match")

    metrics: dict[str, Tensor] = {}
    num_candidates = scores.size(1)
    for k in ks:
        topk = scores.topk(min(k, num_candidates), dim=1).indices
        topk_items = candidate_ids.gather(1, topk)
        hits = (topk_items == target_ids.unsqueeze(1)).any(dim=1)
        metrics[f"recall@{k}"] = hits.float().mean()
        if k == 1:
            metrics["hit@1"] = metrics["recall@1"]
        if k == 20:
            matches = topk_items == target_ids.unsqueeze(1)
            has_hit = matches.any(dim=1)
            ranks = matches.int().argmax(dim=1).float() + 1.0
            reciprocal = torch.where(has_hit, 1.0 / ranks, torch.zeros_like(ranks))
            metrics["mrr@20"] = reciprocal.mean()
            ndcg = torch.where(has_hit, 1.0 / torch.log2(ranks + 1.0), torch.zeros_like(ranks))
            metrics["ndcg@20"] = ndcg.mean()

    return metrics
