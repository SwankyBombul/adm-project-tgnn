"""Resolve paths to processed training artifacts on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

SplitName = Literal["train", "val", "test_internal", "challenge_test"]
ModelFormat = Literal["gru4rec", "tagnn", "tgn"]


def split_examples_path(
    processed_dir: Path,
    split: SplitName,
    model: ModelFormat,
) -> Path:
    split_dir = processed_dir / split
    if model == "gru4rec":
        return split_dir / "gru4rec_examples.parquet"
    if model == "tagnn":
        return split_dir / "tagnn_examples.pkl"
    if model == "tgn":
        return split_dir / "tgn" / "examples.parquet"
    raise ValueError(f"Unsupported model format: {model}")
