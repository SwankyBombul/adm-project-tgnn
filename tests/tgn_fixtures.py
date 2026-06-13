"""Synthetic TGN artifacts for unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_tgn_processed_dir(
    root: Path,
    *,
    num_items: int = 5,
    num_sessions: int = 2,
    n_events: int = 8,
) -> Path:
    processed = root / "subsample_test_clicks_only"
    for split in ("train", "val", "test_internal", "challenge_test"):
        tgn_dir = processed / split / "tgn"
        tgn_dir.mkdir(parents=True, exist_ok=True)
        events = pd.DataFrame(
            {
                "event_id": list(range(n_events)),
                "session_id": [0, 0, 0, 1, 1, 1, 1, 1][:n_events],
                "session_idx": [0, 0, 0, 1, 1, 1, 1, 1][:n_events],
                "item_id": [10, 11, 12, 20, 21, 22, 23, 24][:n_events],
                "item_idx_tgn": [0, 1, 2, 0, 1, 2, 3, 4][:n_events],
                "t_sec": [float(i) for i in range(n_events)],
                "event_type": [0] * n_events,
                "cat_bucket_idx": [0] * n_events,
                "price_log": [0.0] * n_events,
                "quantity": [0.0] * n_events,
                "is_click": [True] * n_events,
                "split": [split] * n_events,
            }
        )
        events.to_parquet(tgn_dir / "events.parquet", index=False)
        examples = pd.DataFrame(
            {
                "example_id": [0, 1],
                "split": [split, split],
                "session_id": [0, 1],
                "session_idx": [0, 1],
                "target_item_idx_tgn": [2, 4],
                "target_t_sec": [2.0, 7.0],
                "target_event_id": [2, 7],
                "prefix_last_event_id": [1, 6],
            }
        )
        examples.to_parquet(tgn_dir / "examples.parquet", index=False)

    vocab_dir = processed / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    (vocab_dir / "item_vocab.json").write_text(
        json.dumps(
            {
                "pad_idx": 0,
                "n_items": num_items,
                "unk_gru_idx": num_items + 1,
                "unk_tgn_idx": num_items,
                "item2idx": {str(10 + i): i + 1 for i in range(num_items)},
                "idx2item": {str(i + 1): 10 + i for i in range(num_items)},
                "unk_raw_item_id": -1,
            }
        ),
        encoding="utf-8",
    )
    meta = {
        "index_conventions": {
            "gru4rec": {"embedding_num_embeddings": num_items + 2},
            "tgn": {"unk_idx": num_items, "item_idx_range": f"0..{num_items - 1}"},
        },
        "stats": {
            "n_sessions": num_sessions,
            "popularity": {
                "top20_item_ids": [10 + i for i in range(min(20, num_items))],
            },
        },
    }
    (processed / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return processed
