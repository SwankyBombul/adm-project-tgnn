"""Chronological subsampling by full sessions."""

from __future__ import annotations

import pandas as pd


def subsample_sessions(
    clicks: pd.DataFrame,
    buys: pd.DataFrame,
    fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Keep the oldest `fraction` of sessions (by first click time)."""
    if fraction >= 1.0:
        return clicks.copy(), buys.copy(), clicks["session_id"].unique()

    session_start = clicks.groupby("session_id")["timestamp"].min().sort_values()
    n_keep = max(1, int(len(session_start) * fraction))
    kept = session_start.iloc[:n_keep].index

    clicks_sub = clicks[clicks["session_id"].isin(kept)].copy()
    buys_sub = buys[buys["session_id"].isin(kept)].copy()
    return clicks_sub, buys_sub, kept
