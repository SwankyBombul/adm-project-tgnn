"""Tests for Colab runtime helpers and GRU4Rec data utilities."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from src.artifacts import gru4rec_vocab_size, load_meta
from src.config.wandb_settings import WANDB_ENTITY, WANDB_PROJECT, expected_wandb_settings
from src.models.gru4rec import gru4rec_collate_fn
from src.runtime import check_drive_layout, prepare_colab_session, unzip_processed_archive


def test_wandb_defaults() -> None:
    settings = expected_wandb_settings()
    assert settings.entity == WANDB_ENTITY == "project-nn"
    assert settings.project == WANDB_PROJECT == "adm-project-tgnn"


def test_collect_viewer_entities_from_user_object() -> None:
    from types import SimpleNamespace

    from src.config.wandb_settings import _collect_viewer_entities

    viewer = SimpleNamespace(username="koostosh", entity="koostosh", teams=[])
    assert _collect_viewer_entities(viewer) == {"koostosh"}


def test_check_drive_layout_missing_zip(tmp_path: Path) -> None:
    project_dir = tmp_path / "drive_project"
    project_dir.mkdir()
    layout = check_drive_layout(project_dir)
    assert not layout.ok
    assert any("processed.zip" in error for error in layout.errors)


def test_prepare_colab_session_paths(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive_project"
    data_dir = drive_root / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "processed.zip").write_bytes(b"not-a-real-zip")

    session = prepare_colab_session(
        drive_root,
        run_name="smoke",
        mount_drive=False,
        unpack_zip=False,
    )
    assert session.checkpoint_dir == drive_root / "checkpoints" / "gru4rec" / "smoke"


def test_unzip_processed_archive_with_processed_prefix(tmp_path: Path) -> None:
    variant_dir = tmp_path / "staging" / "processed" / "subsample_1_32_clicks_only"
    variant_dir.mkdir(parents=True)
    (variant_dir / "meta.json").write_text('{"ok": true}', encoding="utf-8")

    zip_path = tmp_path / "processed.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in variant_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(tmp_path / "staging").as_posix())

    extracted = unzip_processed_archive(zip_path, extract_parent=tmp_path / "local_data")
    meta = json.loads((extracted / "subsample_1_32_clicks_only" / "meta.json").read_text())
    assert meta["ok"] is True


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
