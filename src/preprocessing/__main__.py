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
        "--fraction",
        type=float,
        default=1 / 32,
        help="Session subsample fraction (default: 1/32).",
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

    config = PreprocessConfig(
        subsample_fraction=args.fraction,
        include_buys=args.include_buys,
    )
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
