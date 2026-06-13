"""Bipartite session–item node indexing for TGN."""

from __future__ import annotations

import torch
from torch import Tensor


def item_global_id(item_idx_tgn: Tensor | int) -> Tensor | int:
    """Item nodes occupy ``0 .. num_items - 1`` (stable across splits)."""
    return item_idx_tgn


def session_global_id(session_idx: Tensor | int, num_items: int) -> Tensor | int:
    """Session nodes occupy ``num_items .. num_items + num_sessions - 1``."""
    if isinstance(session_idx, Tensor):
        return session_idx + num_items
    return int(session_idx) + num_items


def num_nodes(num_items: int, num_sessions: int) -> int:
    return num_items + num_sessions


def edge_endpoints(
    session_idx: Tensor,
    item_idx_tgn: Tensor,
    num_items: int,
) -> tuple[Tensor, Tensor]:
    """Map bipartite endpoints to global node ids (session -> item)."""
    src = session_global_id(session_idx, num_items)
    dst = item_global_id(item_idx_tgn)
    return src, dst


def item_global_ids(num_items: int, device: torch.device | None = None) -> Tensor:
    """All item node ids ``0 .. num_items - 1``."""
    return torch.arange(num_items, device=device, dtype=torch.long)
