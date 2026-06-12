"""LightningCLI entry: ``uv run python -m src.main fit -c config/data/gru4rec_yoochoose.yaml -c config/model/gru4rec.yaml``"""

from __future__ import annotations

from src.utils.cli import AdmLightningCLI


def cli_main() -> None:
    AdmLightningCLI()


if __name__ == "__main__":
    cli_main()
