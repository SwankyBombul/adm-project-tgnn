"""Tests for LightningCLI config linking helpers."""

from __future__ import annotations

import json
from pathlib import Path

from jsonargparse import Namespace

from src.utils.cli import link_gru4rec_num_embeddings, resolve_checkpoint_path


def test_link_gru4rec_num_embeddings_namespace(tmp_path: Path) -> None:
    processed = tmp_path / "subsample_1_32_clicks_only"
    processed.mkdir()
    meta = {"index_conventions": {"gru4rec": {"embedding_num_embeddings": 12345}}}
    (processed / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    data_cfg = Namespace(
        init_args=Namespace(processed_dir=str(processed)),
    )
    model_cfg = Namespace(init_args=Namespace(num_embeddings=0))

    link_gru4rec_num_embeddings(data_cfg, model_cfg)

    assert model_cfg.init_args.num_embeddings == 12345


def test_link_gru4rec_num_embeddings_dict(tmp_path: Path) -> None:
    processed = tmp_path / "subsample_1_32_clicks_only"
    processed.mkdir()
    meta = {"index_conventions": {"gru4rec": {"embedding_num_embeddings": 99}}}
    (processed / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    data_cfg = {"init_args": {"processed_dir": str(processed)}}
    model_cfg: dict = {}

    link_gru4rec_num_embeddings(data_cfg, model_cfg)

    assert model_cfg["init_args"]["num_embeddings"] == 99


def test_resolve_checkpoint_path_best_prefers_best_in_name(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    run_dir = root / "run-a" / "checkpoints"
    run_dir.mkdir(parents=True)
    older = run_dir / "epoch_001.ckpt"
    newer_best = run_dir / "best-epoch_004.ckpt"
    older.write_text("old", encoding="utf-8")
    newer_best.write_text("new", encoding="utf-8")

    import time

    time.sleep(0.01)
    newer_best.touch()

    resolved = resolve_checkpoint_path("best", root)
    assert resolved is not None
    assert Path(resolved).name == "best-epoch_004.ckpt"


def test_resolve_checkpoint_path_last_prefers_last_ckpt(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    run_dir = root / "run-a" / "checkpoints"
    run_dir.mkdir(parents=True)
    (run_dir / "epoch_001.ckpt").write_text("e1", encoding="utf-8")
    last = run_dir / "last.ckpt"
    last.write_text("last", encoding="utf-8")

    resolved = resolve_checkpoint_path("last", root)
    assert resolved is not None
    assert Path(resolved).name == "last.ckpt"
