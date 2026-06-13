"""Tests for TAGNN LightningDataModule."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from src.artifacts.paths import split_examples_path
from src.data_modules.tagnn import TAGNNDataModule, _resolve_num_workers


def _write_tagnn_split(processed: Path, split: str, rows: list[dict]) -> None:
    split_dir = processed / split
    split_dir.mkdir(parents=True, exist_ok=True)
    with (split_dir / "tagnn_examples.pkl").open("wb") as handle:
        pickle.dump(rows, handle)


def _make_processed_dir(tmp_path: Path) -> Path:
    processed = tmp_path / "subsample_1_32_clicks_only"
    meta = {
        "index_conventions": {"gru4rec": {"embedding_num_embeddings": 128}},
        "stats": {},
    }
    processed.mkdir(parents=True)
    (processed / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    train_rows = [
        {"item_ids": [1, 2, 3], "target_item_idx": 3, "history_len": 2},
        {"item_ids": [4, 5], "target_item_idx": 5, "history_len": 1},
    ]
    val_rows = [{"item_ids": [1, 2], "target_item_idx": 2, "history_len": 1}]
    test_internal_rows = [{"item_ids": [2, 3, 4], "target_item_idx": 4, "history_len": 2}]
    challenge_rows = [{"item_ids": [5, 6], "target_item_idx": 6, "history_len": 1}]
    _write_tagnn_split(processed, "train", train_rows)
    _write_tagnn_split(processed, "val", val_rows)
    _write_tagnn_split(processed, "test_internal", test_internal_rows)
    _write_tagnn_split(processed, "challenge_test", challenge_rows)
    return processed


def test_split_examples_path_prefers_pkl(tmp_path: Path) -> None:
    split_dir = tmp_path / "val"
    split_dir.mkdir()
    pkl_path = split_dir / "tagnn_examples.pkl"
    parquet_path = split_dir / "tagnn_examples.parquet"
    pkl_path.write_bytes(b"")
    parquet_path.write_bytes(b"")

    resolved = split_examples_path(tmp_path, "val", "tagnn")
    assert resolved == pkl_path


def test_tagnn_datamodule_loaders(tmp_path: Path) -> None:
    processed = _make_processed_dir(tmp_path)
    dm = TAGNNDataModule(processed_dir=processed, batch_size=2, num_workers=0)
    assert dm.num_embeddings == 128

    dm.setup("fit")
    train_batch = next(iter(dm.train_dataloader()))
    assert train_batch[4].shape[0] <= 2

    dm.setup("test")
    loaders = dm.test_dataloader()
    assert len(loaders) == 2
    assert next(iter(loaders[0]))[4].shape[0] >= 1


def test_resolve_num_workers_forces_zero_on_windows(monkeypatch) -> None:
    monkeypatch.setattr("src.data_modules.tagnn.sys.platform", "win32")
    with pytest.warns(UserWarning, match="forced to 0 on Windows"):
        assert _resolve_num_workers(4) == 0
    assert _resolve_num_workers(0) == 0
