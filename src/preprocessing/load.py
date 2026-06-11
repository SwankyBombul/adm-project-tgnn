"""Load raw Yoochoose .dat files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CLICK_COLUMNS = ["session_id", "timestamp", "item_id", "category"]
BUY_COLUMNS = ["session_id", "timestamp", "item_id", "price", "quantity"]


def _read_dat(path: Path, names: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(
        path,
        sep=",",
        names=names,
        dtype={
            "session_id": "int64",
            "item_id": "int64",
            "category": "string",
            "price": "float64",
            "quantity": "float64",
        },
        parse_dates=["timestamp"],
        date_format="%Y-%m-%dT%H:%M:%S.%f%z",
    )


def load_clicks(raw_dir: Path) -> pd.DataFrame:
    df = _read_dat(raw_dir / "yoochoose-clicks.dat", CLICK_COLUMNS)
    df["category"] = df["category"].astype("string")
    return df


def load_buys(raw_dir: Path) -> pd.DataFrame:
    return _read_dat(raw_dir / "yoochoose-buys.dat", BUY_COLUMNS)


def load_test(raw_dir: Path) -> pd.DataFrame:
    df = _read_dat(raw_dir / "yoochoose-test.dat", CLICK_COLUMNS)
    df["category"] = df["category"].astype("string")
    return df
