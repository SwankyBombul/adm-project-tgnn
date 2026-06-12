"""Read item vocabulary artifacts written by preprocessing (read-only)."""

from __future__ import annotations

import json
from pathlib import Path


def load_gru_item2idx(vocab_dir: Path) -> dict[int, int]:
    """Load raw item ID → GRU4Rec index mapping from ``vocab/item_vocab.json``."""
    vocab_path = vocab_dir / "item_vocab.json"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing item vocabulary: {vocab_path}")
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    return {int(raw_id): int(idx) for raw_id, idx in payload["item2idx"].items()}
