"""Non-learned baselines loaded from preprocessing metadata."""

from __future__ import annotations

from typing import Any


def popularity_top_k(meta: dict[str, Any], k: int = 20) -> list[int]:
    """Return the top-*k* item IDs by train-click popularity from ``meta.json``."""
    popularity = meta.get("stats", {}).get("popularity", {})
    top_ids = popularity.get("top20_item_ids", [])
    if len(top_ids) < k:
        raise ValueError(f"meta.json only stores {len(top_ids)} popular items, requested k={k}")
    return list(top_ids[:k])
