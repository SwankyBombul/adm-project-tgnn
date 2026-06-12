"""Tests for LightningCLI config linking helpers."""

from __future__ import annotations

import json
from pathlib import Path

from jsonargparse import Namespace

from src.utils.cli import link_gru4rec_num_embeddings


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
