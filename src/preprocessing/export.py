"""Write processed artifacts to disk."""

from __future__ import annotations

import gc
import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.config import PreprocessConfig
from src.preprocessing.examples import ExampleBatch
from src.preprocessing.vocab import ItemVocab

_STREAM_BATCH_SIZE = 100_000


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _write_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _append_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df.to_parquet(path, index=False, engine="pyarrow", append=True)
    else:
        df.to_parquet(path, index=False)


def write_examples_streaming(
    example_iter: Iterator[tuple[dict, dict, dict]],
    output_dir: Path,
    export_sequence_models: bool,
) -> int:
    """Stream examples to disk in batches (for full challenge_test)."""
    tgn_path = output_dir / "tgn" / "examples.parquet"
    gru_path = output_dir / "gru4rec_examples.parquet" if export_sequence_models else None
    tagnn_path = (
        output_dir / "tagnn_examples.parquet" if export_sequence_models else None
    )

    for path in (p for p in (tgn_path, gru_path, tagnn_path) if p is not None):
        if path.exists():
            path.unlink()

    gru_rows: list[dict] = []
    tagnn_rows: list[dict] = []
    tgn_rows: list[dict] = []
    total = 0

    def flush() -> None:
        nonlocal gru_rows, tagnn_rows, tgn_rows
        if not tgn_rows:
            return
        _append_parquet(pd.DataFrame(tgn_rows), tgn_path)
        if gru_path is not None:
            _append_parquet(pd.DataFrame(gru_rows), gru_path)
        if tagnn_path is not None:
            _append_parquet(pd.DataFrame(tagnn_rows), tagnn_path)
        gru_rows = []
        tagnn_rows = []
        tgn_rows = []

    for gru_row, tagnn_row, tgn_row in example_iter:
        gru_rows.append(gru_row)
        tagnn_rows.append(tagnn_row)
        tgn_rows.append(tgn_row)
        total += 1
        if len(tgn_rows) >= _STREAM_BATCH_SIZE:
            flush()

    flush()
    gc.collect()
    return total


def write_split_artifacts(
    split_name: str,
    examples: ExampleBatch,
    tgn_events: pd.DataFrame,
    output_dir: Path,
    include_buys: bool,
    export_sequence_models: bool,
) -> None:
    split_dir = output_dir / split_name
    tgn_dir = split_dir / "tgn"
    tgn_dir.mkdir(parents=True, exist_ok=True)

    _write_parquet(tgn_events, tgn_dir / "events.parquet")
    _write_parquet(examples.tgn, tgn_dir / "examples.parquet")

    if export_sequence_models:
        _write_parquet(examples.gru4rec, split_dir / "gru4rec_examples.parquet")
        _write_pickle(
            examples.tagnn.to_dict(orient="records"),
            split_dir / "tagnn_examples.pkl",
        )


def write_challenge_events(tgn_events: pd.DataFrame, output_dir: Path) -> None:
    tgn_dir = output_dir / "challenge_test" / "tgn"
    tgn_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(tgn_events, tgn_dir / "events.parquet")


def write_meta(
    output_dir: Path,
    config: PreprocessConfig,
    vocab: ItemVocab,
    boundaries: dict[str, str],
    stats: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": config.to_dict(),
        "boundaries": boundaries,
        "stats": stats,
        "index_conventions": {
            "gru4rec": {
                "pad_idx": 0,
                "item_idx_range": f"1..{vocab.n_items}",
                "unk_idx": vocab.n_items + 1,
                "embedding_num_embeddings": vocab.gru_vocab_size,
            },
            "tagnn": {
                "node_per_click": True,
                "edges": "consecutive (i -> i+1) built in training code",
                "item_idx_scheme": "same as gru4rec",
            },
            "tgn": {
                "item_idx_range": f"0..{vocab.n_items - 1}",
                "unk_idx": vocab.n_items,
                "bipartite": "session_idx -> item_idx",
                "edge_attr": [
                    "cat_bucket_idx",
                    "price_log",
                    "quantity",
                    "event_type",
                ],
                "positive_edges": "next click only (is_click=True)",
            },
        },
        "exports": {
            "sequence_models": not config.include_buys,
            "tgn_events_include_buys": config.include_buys,
        },
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
