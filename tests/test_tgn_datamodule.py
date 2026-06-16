"""Tests for TGNDataModule."""

from __future__ import annotations

from pathlib import Path

import torch

from src.data_modules.tgn import TGNDataModule
from tests.tgn_fixtures import write_tgn_processed_dir


def test_tgn_datamodule_setup(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    dm = TGNDataModule(processed, event_batch_size=4, example_batch_size=2)
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch["session_idx"].numel() <= 4


def test_tgn_datamodule_sampling_weights_are_normalized(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    dm = TGNDataModule(processed, event_batch_size=4, example_batch_size=2)
    weights = dm.train_item_sampling_weights(alpha=1.0)
    assert weights.shape == (dm.num_items,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
