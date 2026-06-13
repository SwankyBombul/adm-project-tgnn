"""Load preprocessing metadata written to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_meta(processed_dir: Path) -> dict[str, Any]:
    meta_path = processed_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.json in {processed_dir}")
    with meta_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def gru4rec_vocab_size(meta: dict[str, Any]) -> int:
    return int(meta["index_conventions"]["gru4rec"]["embedding_num_embeddings"])


def tgn_num_items(meta: dict[str, Any]) -> int:
    """Number of item nodes in TGN space (known items + UNK)."""
    tgn = meta["index_conventions"]["tgn"]
    unk_idx = int(tgn["unk_idx"])
    return unk_idx + 1


def tgn_num_sessions(meta: dict[str, Any], split: str = "train") -> int:
    """Best-effort session count for a split (train uses global ``n_sessions``)."""
    stats = meta.get("stats", {})
    per_split = stats.get(f"{split}_sessions")
    if per_split is not None:
        return int(per_split)
    if split == "train" and "n_sessions" in stats:
        return int(stats["n_sessions"])
    raise KeyError(
        f"Cannot resolve session count for split {split!r}; "
        "compute from events.parquet in the data module."
    )
