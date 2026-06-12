"""Load preprocessing metadata and resolve artifact paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

SplitName = Literal["train", "val", "test_internal", "challenge_test"]
ModelFormat = Literal["gru4rec", "tagnn", "tgn"]


def load_meta(processed_dir: Path) -> dict[str, Any]:
    meta_path = processed_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.json in {processed_dir}")
    with meta_path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def gru4rec_vocab_size(meta: dict[str, Any]) -> int:
    return int(meta["index_conventions"]["gru4rec"]["embedding_num_embeddings"])
