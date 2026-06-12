"""Tests for YAML configuration loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

from src.common.paths import get_project_root
from src.config import config_dir, resolve_config_path
from src.preprocessing.config import PreprocessConfig


def test_config_dir_points_to_project_config_folder() -> None:
    assert config_dir() == get_project_root() / "config"


def test_resolve_config_path_relative() -> None:
    path = resolve_config_path("config/data/gru4rec_yoochoose.yaml")
    assert path.is_file()
    assert path.name == "gru4rec_yoochoose.yaml"


def test_preprocess_config_from_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "preprocess.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """
            paths:
              raw_dir: data/raw
              output_root: data/processed
            preprocessing:
              subsample_fraction: 0.001
              include_buys: true
              split_ratios: [0.7, 0.15, 0.15]
            """
        ).strip(),
        encoding="utf-8",
    )
    config = PreprocessConfig.from_yaml(yaml_path, project_root=tmp_path)
    assert config.subsample_fraction == 0.001
    assert config.include_buys is True
    assert config.split_ratios == (0.7, 0.15, 0.15)
    assert config.raw_dir == tmp_path / "data" / "raw"
