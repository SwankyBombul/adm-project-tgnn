"""Collate helpers for TGN training batches."""

from __future__ import annotations

import torch
from torch import Tensor

from src.evaluation.sampled import sample_negative_items
from src.models.tgn.dataset import TGNExampleBatch

__all__ = ["sample_negative_items", "tgn_event_collate_fn", "tgn_example_collate_fn"]


def tgn_example_collate_fn(samples: list[TGNExampleBatch]) -> TGNExampleBatch:
    return TGNExampleBatch(
        session_idx=torch.stack([s.session_idx for s in samples]),
        target_item_idx_tgn=torch.stack([s.target_item_idx_tgn for s in samples]),
        target_t_sec=torch.stack([s.target_t_sec for s in samples]),
        target_event_id=torch.stack([s.target_event_id for s in samples]),
        prefix_last_event_id=torch.stack([s.prefix_last_event_id for s in samples]),
    )


def tgn_event_collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {
        "session_idx": torch.cat([b["session_idx"] for b in batch], dim=0),
        "item_idx_tgn": torch.cat([b["item_idx_tgn"] for b in batch], dim=0),
        "t_sec": torch.cat([b["t_sec"] for b in batch], dim=0),
        "msg": torch.cat([b["msg"] for b in batch], dim=0),
    }
