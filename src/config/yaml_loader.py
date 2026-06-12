"""Load preprocessing YAML configs into typed dataclass kwargs."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from src.common.paths import get_project_root

PREPROCESS_SECTIONS = ("paths", "preprocessing")

_PATH_KEYS = frozenset({"raw_dir", "output_root"})


def config_dir(project_root: Path | None = None) -> Path:
    return (project_root or get_project_root()) / "config"


def resolve_config_path(path: str | Path, project_root: Path | None = None) -> Path:
    """Resolve a config file path (absolute or relative to project root)."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    rooted = (project_root or get_project_root()) / candidate
    if rooted.is_file():
        return rooted.resolve()
    raise FileNotFoundError(f"Config file not found: {path}")


def load_yaml_file(path: str | Path, project_root: Path | None = None) -> dict[str, Any]:
    config_path = resolve_config_path(path, project_root=project_root)
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data


def _flatten_sections(
    data: dict[str, Any],
    *,
    sections: tuple[str, ...],
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for section in sections:
        payload = data.get(section)
        if payload is None:
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"Config section '{section}' must be a mapping")
        flat.update(payload)
    return flat


def _resolve_path_value(value: Any, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _coerce_value(key: str, value: Any, project_root: Path) -> Any:
    if key in _PATH_KEYS and value is not None:
        return _resolve_path_value(value, project_root)
    if key == "split_ratios" and isinstance(value, list):
        return tuple(value)
    return value


def _validate_keys(flat: dict[str, Any], dataclass_type: type) -> None:
    allowed = {field.name for field in fields(dataclass_type)}
    unknown = sorted(set(flat) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown config keys for {dataclass_type.__name__}: {', '.join(unknown)}"
        )


def preprocess_config_kwargs(
    data: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root or get_project_root()
    flat = _flatten_sections(data, sections=PREPROCESS_SECTIONS)
    from src.preprocessing.config import PreprocessConfig

    _validate_keys(flat, PreprocessConfig)
    return {key: _coerce_value(key, value, root) for key, value in flat.items()}
