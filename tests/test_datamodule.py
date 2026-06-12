"""Tests for GRU4Rec LightningDataModule."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_modules.gru4rec import GRU4RecDataModule


def _write_gru4rec_split(processed: Path, split: str, rows: list[dict]) -> None:
    split_dir = processed / split
    split_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    pq.write_table(table, split_dir / "gru4rec_examples.parquet")


def _make_processed_dir(tmp_path: Path) -> Path:
    processed = tmp_path / "subsample_1_32_clicks_only"
    meta = {
        "index_conventions": {"gru4rec": {"embedding_num_embeddings": 128}},
        "stats": {},
    }
    processed.mkdir(parents=True)
    (processed / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    train_rows = [
        {"history_item_idx": [1, 2], "target_item_idx": 3},
        {"history_item_idx": [4], "target_item_idx": 5},
    ]
    val_rows = [{"history_item_idx": [1], "target_item_idx": 2}]
    _write_gru4rec_split(processed, "train", train_rows)
    _write_gru4rec_split(processed, "val", val_rows)
    return processed


def test_gru4rec_datamodule_vocab_size_and_loaders(tmp_path: Path) -> None:
    processed = _make_processed_dir(tmp_path)
    dm = GRU4RecDataModule(processed_dir=processed, batch_size=2, num_workers=0)
    assert dm.num_embeddings == 128

    dm.setup("fit")
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))

    assert train_batch[0].shape[0] <= 2
    assert train_batch[2].shape[0] == train_batch[0].shape[0]
    assert val_batch[2].shape[0] == val_batch[0].shape[0]
