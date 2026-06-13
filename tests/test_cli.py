"""Tests for LightningCLI config linking and saved-model paths."""

from __future__ import annotations

import json
from pathlib import Path

from jsonargparse import Namespace

from src.training.paths import resolve_saved_checkpoint, saved_model_dir
from src.utils.cli import (
    configure_fit_saved_model_dirs,
    infer_model_name,
    infer_run_name,
    link_gru4rec_num_embeddings,
    link_model_config_from_meta,
    resolve_checkpoint_path,
)


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


def test_link_model_config_from_meta_gru4rec(tmp_path: Path) -> None:
    processed = tmp_path / "subsample_1_32_clicks_only"
    processed.mkdir()
    meta = {"index_conventions": {"gru4rec": {"embedding_num_embeddings": 42}}}
    (processed / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    data_cfg = {"init_args": {"processed_dir": str(processed)}}
    model_cfg = {
        "class_path": "src.models.gru4rec.module.GRU4RecLitModule",
        "init_args": {"num_embeddings": 0},
    }

    link_model_config_from_meta(data_cfg, model_cfg)

    assert model_cfg["init_args"]["num_embeddings"] == 42


def test_link_model_config_from_meta_tagnn(tmp_path: Path) -> None:
    processed = tmp_path / "subsample_1_32_clicks_only"
    processed.mkdir()
    meta = {"index_conventions": {"gru4rec": {"embedding_num_embeddings": 77}}}
    (processed / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    data_cfg = {"init_args": {"processed_dir": str(processed)}}
    model_cfg = {
        "class_path": "src.models.tagnn.module.TAGNNLitModule",
        "init_args": {"num_embeddings": 0},
    }

    link_model_config_from_meta(data_cfg, model_cfg)

    assert model_cfg["init_args"]["num_embeddings"] == 77


def test_infer_model_name_and_run_name() -> None:
    data_cfg = Namespace(class_path="src.data_modules.gru4rec.GRU4RecDataModule")
    sub_cfg = Namespace(
        trainer=Namespace(
            logger=Namespace(
                init_args=Namespace(name="gru4rec-baseline"),
            ),
        ),
    )

    assert infer_model_name(data_cfg) == "gru4rec"
    assert infer_run_name(sub_cfg) == "gru4rec-baseline"
    assert infer_run_name(Namespace(trainer=Namespace(logger=False))) == "default"


def test_configure_fit_saved_model_dirs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.training.paths.get_project_root", lambda: tmp_path)
    sub_cfg = Namespace(
        trainer=Namespace(
            default_root_dir="saved_models/gru4rec",
            logger=Namespace(
                class_path="lightning.pytorch.loggers.WandbLogger",
                init_args=Namespace(save_dir="old"),
            ),
            callbacks=[
                Namespace(
                    class_path="lightning.pytorch.callbacks.ModelCheckpoint",
                    init_args=Namespace(dirpath=None, filename="best-epoch_{epoch:03d}"),
                ),
            ],
        ),
    )

    run_dir = configure_fit_saved_model_dirs(sub_cfg, "gru4rec", "smoke-run")

    assert run_dir == saved_model_dir("gru4rec", "smoke-run", project_root=tmp_path)
    assert run_dir.is_dir()
    assert sub_cfg.trainer.default_root_dir == str(run_dir)
    assert sub_cfg.trainer.callbacks[0].init_args.dirpath == str(run_dir)
    assert sub_cfg.trainer.logger.init_args.save_dir == str(run_dir)


def test_resolve_checkpoint_path_best(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.training.paths.get_project_root", lambda: tmp_path)
    run_dir = saved_model_dir("gru4rec", "baseline", project_root=tmp_path)
    run_dir.mkdir(parents=True)
    ckpt = run_dir / "best.ckpt"
    ckpt.write_text("weights", encoding="utf-8")

    resolved = resolve_checkpoint_path("best", model_name="gru4rec", run_name="baseline")
    assert resolved == str(ckpt)


def test_resolve_saved_checkpoint_explicit_path(tmp_path: Path) -> None:
    ckpt = tmp_path / "custom.ckpt"
    ckpt.write_text("weights", encoding="utf-8")

    resolved = resolve_saved_checkpoint(
        "gru4rec",
        "any-run",
        alias=str(ckpt),
        project_root=tmp_path,
    )
    assert resolved == ckpt
