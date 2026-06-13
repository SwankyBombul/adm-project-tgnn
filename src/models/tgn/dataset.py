"""TGN datasets backed by processed parquet artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import Dataset

MSG_COLUMNS = ("cat_bucket_idx", "price_log", "quantity", "event_type")


@dataclass(frozen=True)
class TGNEventTensors:
    """Columnar event stream for a split."""

    event_id: Tensor
    session_idx: Tensor
    item_idx_tgn: Tensor
    t_sec: Tensor
    msg: Tensor
    num_sessions: int

    @property
    def num_events(self) -> int:
        return int(self.event_id.size(0))

    def slice_events(self, start: int, end: int) -> dict[str, Tensor]:
        """Inclusive ``event_id`` range ``[start, end]``."""
        mask = (self.event_id >= start) & (self.event_id <= end)
        session_idx = self.session_idx[mask]
        item_idx = self.item_idx_tgn[mask]
        return {
            "session_idx": session_idx,
            "item_idx_tgn": item_idx,
            "t_sec": self.t_sec[mask],
            "msg": self.msg[mask],
        }

    def to(self, device: torch.device) -> TGNEventTensors:
        return TGNEventTensors(
            event_id=self.event_id.to(device),
            session_idx=self.session_idx.to(device),
            item_idx_tgn=self.item_idx_tgn.to(device),
            t_sec=self.t_sec.to(device),
            msg=self.msg.to(device),
            num_sessions=self.num_sessions,
        )


def load_events_tensors(path: Path) -> TGNEventTensors:
    table = pq.read_table(
        path,
        columns=["event_id", "session_idx", "item_idx_tgn", "t_sec", *MSG_COLUMNS],
    )
    event_id = torch.as_tensor(table.column("event_id").to_numpy(), dtype=torch.long)
    session_idx = torch.as_tensor(table.column("session_idx").to_numpy(), dtype=torch.long)
    item_idx = torch.as_tensor(table.column("item_idx_tgn").to_numpy(), dtype=torch.long)
    t_sec = torch.as_tensor(table.column("t_sec").to_numpy(), dtype=torch.float32)
    msg = torch.stack(
        [
            torch.as_tensor(table.column(col).to_numpy(), dtype=torch.float32)
            for col in MSG_COLUMNS
        ],
        dim=-1,
    )
    num_sessions = int(session_idx.max().item()) + 1 if len(session_idx) else 0
    return TGNEventTensors(
        event_id=event_id,
        session_idx=session_idx,
        item_idx_tgn=item_idx,
        t_sec=t_sec,
        msg=msg,
        num_sessions=num_sessions,
    )


@dataclass(frozen=True)
class TGNExampleBatch:
    session_idx: Tensor
    target_item_idx_tgn: Tensor
    target_t_sec: Tensor
    target_event_id: Tensor
    prefix_last_event_id: Tensor


class TGNExampleDataset(Dataset[TGNExampleBatch]):
    """Next-click supervision rows from ``examples.parquet``."""

    def __init__(self, path: Path, *, sort_by_target: bool = True) -> None:
        df = pd.read_parquet(path)
        if sort_by_target:
            df = df.sort_values("target_event_id", kind="mergesort").reset_index(drop=True)
        self._session_idx = torch.as_tensor(df["session_idx"].to_numpy(), dtype=torch.long)
        self._target_item = torch.as_tensor(
            df["target_item_idx_tgn"].to_numpy(), dtype=torch.long
        )
        self._target_t = torch.as_tensor(df["target_t_sec"].to_numpy(), dtype=torch.float32)
        self._target_event = torch.as_tensor(df["target_event_id"].to_numpy(), dtype=torch.long)
        self._prefix_last = torch.as_tensor(
            df["prefix_last_event_id"].to_numpy(), dtype=torch.long
        )

    def __len__(self) -> int:
        return int(self._session_idx.size(0))

    def __getitem__(self, index: int) -> TGNExampleBatch:
        return TGNExampleBatch(
            session_idx=self._session_idx[index],
            target_item_idx_tgn=self._target_item[index],
            target_t_sec=self._target_t[index],
            target_event_id=self._target_event[index],
            prefix_last_event_id=self._prefix_last[index],
        )


class TGNEventStreamDataset(Dataset[dict[str, Tensor]]):
    """Chronological mini-batches over ``events.parquet`` for BCE training."""

    def __init__(self, path: Path, batch_size: int) -> None:
        self.events = load_events_tensors(path)
        self.batch_size = batch_size
        n = self.events.num_events
        self._num_batches = max(1, (n + batch_size - 1) // batch_size) if n else 0

    def __len__(self) -> int:
        return self._num_batches

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        start = index * self.batch_size
        end = min(self.events.num_events, start + self.batch_size)
        return {
            "session_idx": self.events.session_idx[start:end],
            "item_idx_tgn": self.events.item_idx_tgn[start:end],
            "t_sec": self.events.t_sec[start:end],
            "msg": self.events.msg[start:end],
        }
