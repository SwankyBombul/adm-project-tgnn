"""Tests for TGNDataModule."""

from __future__ import annotations

from pathlib import Path

from src.data_modules.tgn import TGNDataModule
from tests.tgn_fixtures import write_tgn_processed_dir


def test_tgn_datamodule_setup_bce(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    dm = TGNDataModule(processed, loss_mode="bce", event_batch_size=4, example_batch_size=2)
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch["session_idx"].numel() <= 4


def test_tgn_datamodule_setup_ce(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    dm = TGNDataModule(processed, loss_mode="ce", example_batch_size=2)
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.target_item_idx_tgn.numel() == 2
