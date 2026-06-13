"""Tests for GRU4Rec data utilities and W&B defaults."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.artifacts import gru4rec_vocab_size, load_meta
from src.config.wandb_settings import WANDB_ENTITY, WANDB_PROJECT, expected_wandb_settings
from src.models.gru4rec import gru4rec_collate_fn


def test_wandb_defaults() -> None:
    settings = expected_wandb_settings()
    assert settings.entity == WANDB_ENTITY == "project-nn"
    assert settings.project == WANDB_PROJECT == "adm-project-tgnn"


def test_collect_viewer_entities_from_user_object() -> None:
    from types import SimpleNamespace

    from src.config.wandb_settings import _collect_viewer_entities

    viewer = SimpleNamespace(username="koostosh", entity="koostosh", teams=[])
    assert _collect_viewer_entities(viewer) == {"koostosh"}


def test_gru4rec_collate_handles_numpy_histories() -> None:
    batch = [(np.array([1, 2], dtype=np.int64), 3), (np.array([4], dtype=np.int64), 5)]
    padded, lengths, targets = gru4rec_collate_fn(batch)

    assert padded[0].tolist() == [1, 2]
    assert padded[1].tolist() == [4, 0]
    assert lengths.tolist() == [2, 1]
    assert targets.tolist() == [3, 5]


def test_gru4rec_collate_pads_with_zero() -> None:
    batch = [([1, 2], 3), ([4], 5)]
    padded, lengths, targets = gru4rec_collate_fn(batch)

    assert padded.shape == (2, 2)
    assert padded[0].tolist() == [1, 2]
    assert padded[1].tolist() == [4, 0]
    assert lengths.tolist() == [2, 1]
    assert targets.tolist() == [3, 5]


def test_gru4rec_vocab_size_from_meta() -> None:
    meta = {"index_conventions": {"gru4rec": {"embedding_num_embeddings": 19373}}}
    assert gru4rec_vocab_size(meta) == 19373


def test_load_meta_from_processed_dir(tmp_path: Path) -> None:
    processed = tmp_path / "subsample_1_32_clicks_only"
    processed.mkdir()
    (processed / "meta.json").write_text('{"stats": {}}', encoding="utf-8")
    assert load_meta(processed)["stats"] == {}
