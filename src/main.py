"""LightningCLI entry: ``uv run python -m src.main fit`` / ``evaluate``."""

from __future__ import annotations

import sys

from src.utils.cli import AdmLightningCLI


def cli_main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        sys.argv[1] = "test"
    AdmLightningCLI()


if __name__ == "__main__":
    cli_main()
