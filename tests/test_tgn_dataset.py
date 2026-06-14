"""Tests for TGN datasets."""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.tgn.dataset import TGNExampleDataset, load_events_tensors
from tests.tgn_fixtures import write_tgn_processed_dir


def test_load_events_tensors(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    events = load_events_tensors(processed / "train" / "tgn" / "events.parquet")
    assert events.num_events == 8
    assert events.num_sessions == 2
    assert events.msg.shape == (8, 4)
    assert events.device == torch.device("cpu")
    if torch.cuda.is_available():
        cuda_events = events.to(torch.device("cuda"))
        assert cuda_events.device.type == "cuda"


def test_example_dataset_sorted(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    ds = TGNExampleDataset(processed / "train" / "tgn" / "examples.parquet")
    assert len(ds) == 2
    batch = ds[0]
    assert int(batch.target_event_id.item()) == 2
