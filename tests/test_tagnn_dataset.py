"""Tests for TAGNN dataset input construction."""

from __future__ import annotations

import pickle
from pathlib import Path

from src.models.tagnn.dataset import TAGNNDataset


def _write_examples(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(rows, handle)


def test_tagnn_dataset_uses_history_without_target(tmp_path: Path) -> None:
    examples_path = tmp_path / "tagnn_examples.pkl"
    _write_examples(
        examples_path,
        [{"item_ids": [1, 2, 3], "target_item_idx": 3, "history_len": 2}],
    )

    item_ids, target = TAGNNDataset(examples_path, precompute_graphs=False)[0]

    assert item_ids == [1, 2]
    assert target == 3
