"""GRU4Rec dataset and batching."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.preprocessing.config import PAD_IDX

HistoryLike = Sequence[int] | np.ndarray


def _as_int_list(history: HistoryLike) -> list[int]:
    """Normalize parquet / numpy history values to a plain Python list."""
    if isinstance(history, np.ndarray):
        return history.astype(np.int64, copy=False).tolist()
    return [int(item) for item in history]


class GRU4RecDataset(Dataset):
    """Load ``gru4rec_examples.parquet`` for a single split."""

    def __init__(self, examples_path: Path) -> None:
        if not examples_path.is_file():
            raise FileNotFoundError(examples_path)
        frame = pd.read_parquet(examples_path)
        self._history = frame["history_item_idx"].tolist()
        self._targets = frame["target_item_idx"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return _as_int_list(self._history[index]), int(self._targets[index])


def gru4rec_collate_fn(
    batch: list[tuple[HistoryLike, int]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad variable-length histories; return (padded, lengths, targets)."""
    histories, targets = zip(*batch, strict=True)
    history_lists = [_as_int_list(history) for history in histories]
    lengths = torch.tensor([len(history) for history in history_lists], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) else 0

    padded = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)
    for row_idx, history in enumerate(history_lists):
        if len(history) > 0:
            padded[row_idx, : len(history)] = torch.tensor(history, dtype=torch.long)

    target_tensor = torch.tensor(targets, dtype=torch.long)
    return padded, lengths, target_tensor
