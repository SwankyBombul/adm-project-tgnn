"""Tests for training scaffold (config, colab data unpack, collate)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch

from src.colab.setup import unzip_processed_archive
from src.colab.setup import check_drive_layout
from src.config.train_config import TrainConfig
from src.config.wandb_settings import WANDB_ENTITY, WANDB_PROJECT, expected_wandb_settings
from src.data.gru4rec import gru4rec_collate_fn
from src.data.meta import gru4rec_vocab_size, load_meta


def test_wandb_defaults() -> None:
    settings = expected_wandb_settings()
    assert settings.entity == WANDB_ENTITY == "project-nn"
    assert settings.project == WANDB_PROJECT == "adm-project-tgnn"


def test_check_drive_layout_missing_zip(tmp_path: Path) -> None:
    project_dir = tmp_path / "drive_project"
    project_dir.mkdir()
    layout = check_drive_layout(project_dir)
    assert not layout.ok
    assert any("processed.zip" in error for error in layout.errors)


def test_train_config_colab_paths(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive_project"
    config = TrainConfig.for_colab(
        drive_root,
        run_name="smoke",
        processed_variant="subsample_1_32_clicks_only",
    )

    assert config.processed_zip_path == drive_root / "data" / "processed.zip"
    assert config.checkpoint_dir == drive_root / "checkpoints" / "gru4rec" / "smoke"
    assert "processed_dir" in config.to_dict()


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
