"""LightningCLI with project defaults, saved-model paths, and config linking."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from jsonargparse import Namespace
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from src.common.paths import get_project_root
from src.training.paths import DEFAULT_MODEL_NAME, resolve_saved_checkpoint, saved_model_dir

_DEFAULT_CONFIG = str(get_project_root() / "config" / "default.yaml")
_SUBCOMMANDS = ("fit", "test")


def _nested_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _set_init_arg(cfg: Any, key: str, value: Any) -> None:
    init_args = _nested_get(cfg, "init_args")
    if init_args is None:
        init_args = Namespace()
        if isinstance(cfg, dict):
            cfg["init_args"] = init_args
        else:
            cfg.init_args = init_args
    if isinstance(init_args, dict):
        init_args[key] = value
    else:
        setattr(init_args, key, value)


def _set_subcfg_value(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def _resolve_processed_dir(data_cfg: Any) -> Path | None:
    data_init = _nested_get(data_cfg, "init_args")
    processed_dir = _nested_get(data_init, "processed_dir")
    if processed_dir is None:
        return None
    path = Path(processed_dir)
    return path if path.is_absolute() else get_project_root() / path


def link_gru4rec_num_embeddings(data_cfg: Any, model_cfg: Any) -> None:
    """Set ``model.init_args.num_embeddings`` from processed ``meta.json``."""
    processed_dir = _resolve_processed_dir(data_cfg)
    if processed_dir is None:
        return

    from src.artifacts import gru4rec_vocab_size, load_meta

    _set_init_arg(model_cfg, "num_embeddings", gru4rec_vocab_size(load_meta(processed_dir)))


def link_tgn_num_items(
    data_cfg: Any,
    model_cfg: Any,
    *,
    include_challenge_test: bool = True,
) -> None:
    """Set TGN ``num_items`` and session count from split event artifacts."""
    processed_dir = _resolve_processed_dir(data_cfg)
    if processed_dir is None:
        return

    from src.artifacts import load_meta, tgn_num_items
    from src.models.tgn.dataset import load_events_tensors
    from src.artifacts.paths import split_events_path

    meta = load_meta(processed_dir)
    _set_init_arg(model_cfg, "num_items", tgn_num_items(meta))
    total_sessions = 0
    splits = ["train", "val", "test_internal"]
    if include_challenge_test:
        splits.append("challenge_test")
    for split in splits:
        path = split_events_path(processed_dir, split)
        if path.is_file():
            total_sessions += load_events_tensors(path).num_sessions
    _set_init_arg(model_cfg, "num_sessions_train", total_sessions)


def link_model_config_from_meta(
    data_cfg: Any,
    model_cfg: Any,
    *,
    include_tgn_challenge_sessions: bool = True,
) -> None:
    """Link model hyperparameters from artifacts based on the configured model class."""
    model_class = str(_nested_get(model_cfg, "class_path") or "").lower()
    if "gru4rec" in model_class or "tagnn" in model_class:
        link_gru4rec_num_embeddings(data_cfg, model_cfg)
    if "tgn" in model_class:
        link_tgn_num_items(
            data_cfg,
            model_cfg,
            include_challenge_test=include_tgn_challenge_sessions,
        )


def infer_model_name(data_cfg: Any) -> str:
    """Infer saved-model folder name from the configured data module."""
    class_path = str(_nested_get(data_cfg, "class_path") or "").lower()
    if "gru4rec" in class_path:
        return "gru4rec"
    if "tagnn" in class_path:
        return "tagnn"
    if "tgn" in class_path:
        return "tgn"
    return DEFAULT_MODEL_NAME


def infer_run_name(sub_cfg: Any) -> str:
    """Read W&B run name from trainer logger config (fallback: ``default``)."""
    trainer_cfg = _nested_get(sub_cfg, "trainer")
    logger_cfg = _nested_get(trainer_cfg, "logger")
    if not logger_cfg or logger_cfg is False:
        return "default"
    logger_init = _nested_get(logger_cfg, "init_args")
    run_name = _nested_get(logger_init, "name")
    return str(run_name) if run_name else "default"


def configure_fit_saved_model_dirs(
    sub_cfg: Any,
    model_name: str,
    run_name: str,
) -> Path:
    """Point trainer, checkpoint callback, and W&B logger at ``saved_models/<model>/<run_name>/``."""
    run_dir = saved_model_dir(model_name, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer_cfg = _nested_get(sub_cfg, "trainer")
    _set_subcfg_value(trainer_cfg, "default_root_dir", str(run_dir))

    callbacks = _nested_get(trainer_cfg, "callbacks") or []
    for callback_cfg in callbacks:
        class_path = str(_nested_get(callback_cfg, "class_path") or "")
        if class_path.endswith("ModelCheckpoint"):
            _set_init_arg(callback_cfg, "dirpath", str(run_dir))
            _set_init_arg(callback_cfg, "filename", "best")

    logger_cfg = _nested_get(trainer_cfg, "logger")
    if logger_cfg and logger_cfg is not False:
        _set_init_arg(logger_cfg, "save_dir", str(run_dir))

    return run_dir


def resolve_checkpoint_path(
    ckpt_path: str | None,
    *,
    model_name: str,
    run_name: str,
) -> str | None:
    """Resolve ``best`` / ``last`` to ``saved_models/<model>/<run_name>/best.ckpt``."""
    if ckpt_path is None or ckpt_path not in {"best", "last"}:
        return ckpt_path
    return str(resolve_saved_checkpoint(model_name, run_name, ckpt_path))


class AdmLightningCLI(LightningCLI):
    """CLI entry point: ``fit`` and ``evaluate`` (mapped to Lightning ``test``)."""

    def __init__(self, **kwargs: Any) -> None:
        parser_kwargs = {
            subcommand: {"default_config_files": [_DEFAULT_CONFIG]}
            for subcommand in _SUBCOMMANDS
        }
        parser_kwargs.update(kwargs.pop("parser_kwargs", {}) or {})
        super().__init__(
            save_config_callback=SaveConfigCallback,
            save_config_kwargs={"overwrite": True},
            parser_kwargs=parser_kwargs,
            auto_configure_optimizers=False,
            **kwargs,
        )

    def add_arguments_to_parser(self, parser: Any) -> None:
        parser.add_argument("--ignore_warnings", default=False, type=bool)

    def before_instantiate_classes(self) -> None:
        sub_cfg = self.config[self.subcommand]
        if sub_cfg.get("ignore_warnings"):
            warnings.filterwarnings("ignore")

        data_cfg = sub_cfg.get("data")
        model_cfg = sub_cfg.get("model")
        if data_cfg and model_cfg:
            link_model_config_from_meta(
                data_cfg,
                model_cfg,
                include_tgn_challenge_sessions=self.subcommand == "test",
            )

        model_name = infer_model_name(data_cfg) if data_cfg else DEFAULT_MODEL_NAME
        run_name = infer_run_name(sub_cfg)

        if self.subcommand == "fit":
            configure_fit_saved_model_dirs(sub_cfg, model_name, run_name)

        if self.subcommand == "test":
            ckpt_path = _nested_get(sub_cfg, "ckpt_path")
            if ckpt_path in ("best", "last"):
                resolved = resolve_checkpoint_path(
                    ckpt_path,
                    model_name=model_name,
                    run_name=run_name,
                )
                _set_subcfg_value(sub_cfg, "ckpt_path", resolved)
