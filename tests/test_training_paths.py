"""Tests for saved_models path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.paths import resolve_saved_checkpoint, saved_model_dir


def test_saved_model_dir_layout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.training.paths.get_project_root", lambda: tmp_path)
    run_dir = saved_model_dir("gru4rec", "baseline")
    assert run_dir == tmp_path / "saved_models" / "gru4rec" / "baseline"


def test_resolve_saved_checkpoint_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        resolve_saved_checkpoint("gru4rec", "missing-run", "best", project_root=tmp_path)
