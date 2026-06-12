"""LightningCLI with project defaults and GRU4Rec config linking."""

from __future__ import annotations

import warnings
from typing import Any

from jsonargparse import Namespace
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from src.common.paths import get_project_root

_DEFAULT_CONFIG = str(get_project_root() / "config" / "default.yaml")


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


def link_gru4rec_num_embeddings(data_cfg: Any, model_cfg: Any) -> None:
    """Set ``model.init_args.num_embeddings`` from processed ``meta.json``."""
    data_init = _nested_get(data_cfg, "init_args")
    processed_dir = _nested_get(data_init, "processed_dir")
    if processed_dir is None:
        return

    from pathlib import Path

    from src.artifacts import gru4rec_vocab_size, load_meta

    path = Path(processed_dir)
    if not path.is_absolute():
        path = get_project_root() / path
    _set_init_arg(model_cfg, "num_embeddings", gru4rec_vocab_size(load_meta(path)))


class AdmLightningCLI(LightningCLI):
    """CLI entry point: ``fit``, ``validate``, ``test`` with layered YAML configs."""

    def __init__(self, **kwargs: Any) -> None:
        parser_kwargs = {
            subcommand: {"default_config_files": [_DEFAULT_CONFIG]}
            for subcommand in ("fit", "validate", "test", "predict")
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
            link_gru4rec_num_embeddings(data_cfg, model_cfg)
