"""Tests for popularity baselines (artifacts-only, no preprocessing imports)."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.baselines import popularity_top_k, popularity_top_k_gru_indices


def test_popularity_top_k_gru_indices_from_artifacts(tmp_path: Path) -> None:
    processed = tmp_path / "subsample_1_32_clicks_only"
    vocab_dir = processed / "vocab"
    vocab_dir.mkdir(parents=True)

    (processed / "meta.json").write_text(
        json.dumps(
            {
                "stats": {
                    "popularity": {
                        "top20_item_ids": [100, 200, 300],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (vocab_dir / "item_vocab.json").write_text(
        json.dumps(
            {
                "item2idx": {"100": 1, "200": 2, "999": 3},
                "idx2item": {"1": 100, "2": 200, "3": 999},
                "n_items": 3,
                "unk_raw_item_id": -1,
            }
        ),
        encoding="utf-8",
    )

    assert popularity_top_k(json.loads((processed / "meta.json").read_text()), k=2) == [
        100,
        200,
    ]
    assert popularity_top_k_gru_indices(processed, k=2) == [1, 2]
