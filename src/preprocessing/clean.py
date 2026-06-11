"""Row-level cleaning on click and buy tables."""

from __future__ import annotations

import pandas as pd


def remove_exact_duplicates(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates(subset=keys, keep="first").copy()
    out.attrs["duplicates_removed"] = before - len(out)
    return out


def add_row_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["row_order"] = range(len(out))
    return out
