"""Shared primitives: paths, constants, no domain logic.

Package layout (refactor conventions):

- ``common/`` — paths and constants used everywhere
- ``artifacts/`` — read-only access to processed data on disk (``meta.json``, paths)
- ``preprocessing/`` — writes artifacts; downstream code reads them via ``artifacts/``
- ``models/<name>/`` — one subpackage per model (e.g. ``gru4rec/``: model, dataset, module)
- ``data_modules/`` — LightningDataModule per model
- ``main.py`` + ``utils/cli.py`` — LightningCLI entry (``fit``, ``evaluate``, …)
- ``training/`` — shared LightningModule skeleton and checkpoint paths
- ``evaluation/`` — ranking metrics and baselines
"""

from src.common.constants import PAD_IDX
from src.common.paths import get_project_root

__all__ = ["PAD_IDX", "get_project_root"]
