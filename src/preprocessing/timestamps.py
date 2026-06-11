"""Timestamp normalization relative to train start."""

from __future__ import annotations

import pandas as pd


def compute_t_min(train_clicks: pd.DataFrame) -> pd.Timestamp:
    return train_clicks["timestamp"].min()


def to_t_sec(series: pd.Series, t_min: pd.Timestamp) -> pd.Series:
    delta = series - t_min
    return delta.dt.total_seconds().astype("float64")
