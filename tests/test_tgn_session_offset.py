"""Tests for disjoint global session nodes across TGN splits."""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.tgn.model import TGNModel
from src.models.tgn.node_ids import session_global_id
from tests.tgn_fixtures import write_tgn_processed_dir


def test_session_offset_maps_disjoint_nodes() -> None:
    model = TGNModel(num_items=5, num_sessions=8)
    model.set_session_offset(2)
    local = torch.tensor([0, 1])
    global_ids = model._session_global(local)
    assert global_ids.tolist() == session_global_id(local + 2, 5).tolist()


def test_datamodule_session_offsets_are_cumulative(tmp_path: Path) -> None:
    processed = write_tgn_processed_dir(tmp_path)
    from src.data_modules.tgn import TGNDataModule

    dm = TGNDataModule(processed, val_max_examples=2)
    dm.setup("fit")

    assert dm.session_offset("train") == 0
    assert dm.session_offset("val") == dm.train_events.num_sessions
    assert dm.session_offset("test_internal") == (
        dm.train_events.num_sessions + dm.val_events.num_sessions
    )
