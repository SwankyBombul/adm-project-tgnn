"""Read item vocabulary artifacts written by preprocessing (read-only)."""

from __future__ import annotations

import json
from pathlib import Path


def load_tgn_item2idx(vocab_dir: Path) -> dict[int, int]:
    """Map raw item ID → TGN item index (0..n_items for UNK)."""
    vocab_path = vocab_dir / "item_vocab.json"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing item vocabulary: {vocab_path}")
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    n_items = int(payload["n_items"])
    unk_tgn = int(payload["unk_tgn_idx"])
    out: dict[int, int] = {}
    for raw_id, gru_idx in payload["item2idx"].items():
        known = int(gru_idx) <= n_items
        out[int(raw_id)] = int(gru_idx) - 1 if known else unk_tgn
    return out


def load_gru_item2idx(vocab_dir: Path) -> dict[int, int]:
    """Load raw item ID → GRU4Rec index mapping from ``vocab/item_vocab.json``."""
    vocab_path = vocab_dir / "item_vocab.json"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing item vocabulary: {vocab_path}")
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    return {int(raw_id): int(idx) for raw_id, idx in payload["item2idx"].items()}
