"""Unit tests for preprocessing helpers."""

from __future__ import annotations

import pandas as pd

from src.preprocessing.category import classify_category
from src.preprocessing.split import assign_session_splits, compute_split_boundaries
from src.preprocessing.vocab import build_item_vocab


def test_classify_category_buckets() -> None:
    assert classify_category("0") == "no_category"
    assert classify_category("S") == "special_offer"
    assert classify_category("3") == "product_category"
    assert classify_category("2089286907") == "brand_context"


def test_session_split_assigns_whole_sessions() -> None:
    clicks = pd.DataFrame(
        {
            "session_id": [1, 1, 2, 2],
            "timestamp": pd.to_datetime(
                [
                    "2014-04-01T00:00:00.000000+00:00",
                    "2014-04-02T00:00:00.000000+00:00",
                    "2014-04-03T00:00:00.000000+00:00",
                    "2014-04-04T00:00:00.000000+00:00",
                ],
                utc=True,
            ),
            "item_id": [10, 11, 20, 21],
            "category": ["0", "0", "0", "0"],
        }
    )
    boundaries = compute_split_boundaries(clicks, (0.5, 0.25, 0.25))
    splits = assign_session_splits(clicks, boundaries)
    assert splits.loc[1] == "train"
    assert splits.loc[2] == "val"


def test_vocab_gru_and_tgn_indices() -> None:
    train = pd.DataFrame(
        {
            "session_id": [1, 1],
            "item_id": [100, 200],
            "timestamp": pd.to_datetime(
                ["2014-04-01T00:00:00.000000+00:00"] * 2, utc=True
            ),
            "category": ["0", "S"],
        }
    )
    vocab = build_item_vocab(train)
    assert vocab.gru_index(100, True) == 1
    assert vocab.tgn_index(100, True) == 0
    assert vocab.gru_index(999, False) == vocab.n_items + 1
    assert vocab.tgn_index(999, False) == vocab.n_items
