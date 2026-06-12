"""LightningCLI with project defaults and GRU4Rec config linking."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from jsonargparse import Namespace
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from src.common.paths import get_project_root

_DEFAULT_CONFIG = str(get_project_root() / "config" / "default.yaml")
_DEFAULT_CHECKPOINT_ROOT = get_project_root() / "checkpoints" / "gru4rec"


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


def _set_subcfg_value(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def resolve_checkpoint_path(
    ckpt_path: str | None,
    checkpoint_root: Path | None = None,
) -> str | None:
    """Resolve ``best`` / ``last`` aliases to a concrete ``.ckpt`` file on disk.

    Lightning only knows ``best`` after ``fit`` in the same process; standalone
    ``validate`` needs an explicit path — we search under ``checkpoint_root``.
    """
    if ckpt_path is None or ckpt_path not in {"best", "last"}:
        return ckpt_path

    root = checkpoint_root or _DEFAULT_CHECKPOINT_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {root}. Train first or pass --ckpt_path PATH."
        )

    ckpts = list(root.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No .ckpt files under {root}. Train first or pass --ckpt_path PATH."
        )

    if ckpt_path == "best":
        preferred = [p for p in ckpts if "best" in p.as_posix().lower()]
        candidates = preferred or ckpts
    else:
        preferred = [p for p in ckpts if p.name == "last.ckpt"]
        candidates = preferred or ckpts

    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def _checkpoint_root_from_config(sub_cfg: Any) -> Path:
    trainer_cfg = _nested_get(sub_cfg, "trainer")
    default_root_dir = _nested_get(trainer_cfg, "default_root_dir") or "checkpoints/gru4rec"
    root = Path(default_root_dir)
    return root if root.is_absolute() else get_project_root() / root


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

        if self.subcommand in ("validate", "test", "predict"):
            ckpt_path = _nested_get(sub_cfg, "ckpt_path")
            if ckpt_path in ("best", "last"):
                resolved = resolve_checkpoint_path(ckpt_path, _checkpoint_root_from_config(sub_cfg))
                _set_subcfg_value(sub_cfg, "ckpt_path", resolved)
