"""Filesystem helpers shared across the project."""

from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project repository root."""
    return Path(__file__).resolve().parent.parent.parent
