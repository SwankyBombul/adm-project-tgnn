"""Chronological train/val/test_internal assignment by full sessions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SPLIT_NAMES = ("train", "val", "test_internal")


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: pd.Timestamp
    val_end: pd.Timestamp


def compute_split_boundaries(
    clicks: pd.DataFrame,
    ratios: tuple[float, float, float],
) -> SplitBoundaries:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    sorted_ts = clicks["timestamp"].sort_values()
    n = len(sorted_ts)
    train_end = sorted_ts.iloc[int(n * ratios[0]) - 1]
    val_end = sorted_ts.iloc[int(n * (ratios[0] + ratios[1])) - 1]
    return SplitBoundaries(train_end=train_end, val_end=val_end)


def assign_session_splits(
    clicks: pd.DataFrame,
    boundaries: SplitBoundaries,
) -> pd.Series:
    """Map session_id -> split name using time of first click in session."""
    first_click = clicks.groupby("session_id")["timestamp"].min()

    def _bucket(ts: pd.Timestamp) -> str:
        if ts <= boundaries.train_end:
            return "train"
        if ts <= boundaries.val_end:
            return "val"
        return "test_internal"

    return first_click.map(_bucket)


def split_frame(df: pd.DataFrame, session_split: pd.Series) -> dict[str, pd.DataFrame]:
    split_col = df["session_id"].map(session_split)
    out: dict[str, pd.DataFrame] = {}
    for name in SPLIT_NAMES:
        out[name] = df[split_col == name].copy()
    return out
