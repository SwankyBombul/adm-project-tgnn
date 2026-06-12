"""CLI entry point: uv run python -m src.preprocessing"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.config import PreprocessConfig
from src.preprocessing.pipeline import run_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Yoochoose for GRU4Rec, TAGNN, and TGN."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML preprocessing config (e.g. config/preprocessing.yaml).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Session subsample fraction (default: 1/32, or value from YAML).",
    )
    parser.add_argument(
        "--include-buys",
        action="store_true",
        help="Include buy events in TGN stream (separate output directory).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory with yoochoose-*.dat files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for processed outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing output directory if present.",
    )
    args = parser.parse_args()

    if args.config is not None:
        config = PreprocessConfig.from_yaml(args.config)
    else:
        config = PreprocessConfig(
            subsample_fraction=args.fraction if args.fraction is not None else 1 / 32,
        )

    if args.fraction is not None:
        config.subsample_fraction = args.fraction
    if args.include_buys:
        config.include_buys = True
    if args.raw_dir is not None:
        config.raw_dir = args.raw_dir
    if args.output_root is not None:
        config.output_root = args.output_root

    out_dir = config.output_dir()
    if out_dir.exists():
        if args.force:
            import shutil

            shutil.rmtree(out_dir)
        else:
            raise SystemExit(
                f"Output exists: {out_dir}. Pass --force to overwrite."
            )

    run_preprocessing(config)


if __name__ == "__main__":
    main()
