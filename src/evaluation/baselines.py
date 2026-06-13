"""Non-learned baselines loaded from preprocessing metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.artifacts import load_meta
from src.artifacts.vocab import load_gru_item2idx, load_tgn_item2idx


def popularity_top_k(meta: dict[str, Any], k: int = 20) -> list[int]:
    """Return the top-*k* raw item IDs by train-click popularity from ``meta.json``."""
    popularity = meta.get("stats", {}).get("popularity", {})
    top_ids = popularity.get("top20_item_ids", [])
    if len(top_ids) < k:
        raise ValueError(f"meta.json only stores {len(top_ids)} popular items, requested k={k}")
    return list(top_ids[:k])


def popularity_top_k_tgn_indices(processed_dir: Path, k: int = 20) -> list[int]:
    """Map global popularity baseline IDs to TGN item indices."""
    meta = load_meta(processed_dir)
    item2idx = load_tgn_item2idx(processed_dir / "vocab")
    indices: list[int] = []
    for raw_id in popularity_top_k(meta, k=k):
        if raw_id in item2idx:
            indices.append(item2idx[raw_id])
    return indices


def popularity_top_k_gru_indices(processed_dir: Path, k: int = 20) -> list[int]:
    """Map global popularity baseline IDs to GRU4Rec item indices."""
    meta = load_meta(processed_dir)
    item2idx = load_gru_item2idx(processed_dir / "vocab")
    indices: list[int] = []
    for raw_id in popularity_top_k(meta, k=k):
        if raw_id in item2idx:
            indices.append(item2idx[raw_id])
    return indices


def pop_recall_at_k(targets: Tensor, pop_indices: list[int], k: int) -> float:
    """Hit rate when recommending the same global top-*k* list to every session."""
    if not pop_indices:
        return 0.0
    pop_tensor = torch.tensor(pop_indices[:k], device=targets.device, dtype=targets.dtype)
    hits = (targets.unsqueeze(1) == pop_tensor.unsqueeze(0)).any(dim=1)
    return float(hits.float().mean().item())


@torch.inference_mode()
def evaluate_pop_baseline(
    dataloader: DataLoader,
    pop_indices: list[int],
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """Compute popularity baseline metrics over a split (targets only)."""
    device = torch.device("cpu")
    all_targets: list[Tensor] = []

    for batch in dataloader:
        if hasattr(batch, "target_item_idx_tgn"):
            targets = batch.target_item_idx_tgn
        else:
            targets = batch[-1]
        if not isinstance(targets, Tensor):
            raise TypeError(f"Expected target tensor as last batch element, got {type(targets)}")
        all_targets.append(targets.to(device))

    targets = torch.cat(all_targets, dim=0)
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}_pop"] = pop_recall_at_k(targets, pop_indices, k)
    return metrics
