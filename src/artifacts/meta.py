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
