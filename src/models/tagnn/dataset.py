"""TAGNN dataset loading from processed pickle or parquet artifacts."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.models.tagnn.graph_batch import (
    SessionGraph,
    _truncate_item_ids,
    build_session_graph,
)


def _as_int_list(item_ids: Sequence[int] | np.ndarray) -> list[int]:
    if isinstance(item_ids, np.ndarray):
        return item_ids.astype(np.int64, copy=False).tolist()
    if hasattr(item_ids, "tolist") and not isinstance(item_ids, (str, bytes)):
        converted = item_ids.tolist()
        if converted is not item_ids:
            return _as_int_list(converted)
    return [int(item) for item in item_ids]


def _load_tagnn_frame(examples_path: Path) -> pd.DataFrame:
    if not examples_path.is_file():
        raise FileNotFoundError(examples_path)
    if examples_path.suffix == ".pkl":
        with examples_path.open("rb") as handle:
            records = pickle.load(handle)
        return pd.DataFrame(records)
    return pd.read_parquet(examples_path)


class TAGNNDataset(Dataset):
    """Load TAGNN examples for a single split (pkl or parquet)."""

    def __init__(
        self,
        examples_path: Path,
        max_seq_len: int | None = 50,
        precompute_graphs: bool = True,
    ) -> None:
        frame = _load_tagnn_frame(examples_path)
        if "history_len" not in frame.columns:
            raise ValueError(f"TAGNN examples must include history_len: {examples_path}")

        self._item_ids: list[list[int]] = []
        for item_ids, history_len in zip(
            frame["item_ids"].tolist(),
            frame["history_len"].to_numpy(dtype=np.int64),
            strict=True,
        ):
            history = _as_int_list(item_ids)[: int(history_len)]
            self._item_ids.append(history)
        self._targets = frame["target_item_idx"].to_numpy(dtype=np.int64)
        self.max_seq_len = max_seq_len
        self._lengths: list[int] | None = None
        self._graphs: list[SessionGraph] | None = None
        if precompute_graphs:
            self._graphs = [
                build_session_graph(_truncate_item_ids(item_ids, max_seq_len))
                for item_ids in self._item_ids
            ]

    def __len__(self) -> int:
        return len(self._targets)

    def example_length(self, index: int) -> int:
        if self._lengths is not None:
            return self._lengths[index]
        if self._graphs is not None:
            return len(self._graphs[index][0])
        return len(_truncate_item_ids(self._item_ids[index], self.max_seq_len))

    def example_lengths(self) -> list[int]:
        if self._lengths is None:
            self._lengths = [self.example_length(i) for i in range(len(self))]
        return self._lengths

    def __getitem__(self, index: int) -> tuple[SessionGraph | list[int], int]:
        if self._graphs is not None:
            return self._graphs[index], int(self._targets[index])
        return self._item_ids[index], int(self._targets[index])
