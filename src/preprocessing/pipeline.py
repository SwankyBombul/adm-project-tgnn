"""End-to-end preprocessing orchestration."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.category import BUCKET_NAMES
from src.preprocessing.clean import add_row_order, remove_exact_duplicates
from src.preprocessing.config import PreprocessConfig
from src.preprocessing.events import (
    assign_session_idx,
    build_event_stream,
    build_item_buckets,
    events_to_tgn_frame,
)
from src.preprocessing.examples import (
    build_examples_for_split,
    iter_examples_for_split,
)
from src.preprocessing.export import (
    write_challenge_events,
    write_examples_streaming,
    write_meta,
    write_split_artifacts,
)
from src.preprocessing.load import load_buys, load_clicks, load_test
from src.preprocessing.split import (
    SPLIT_NAMES,
    assign_session_splits,
    compute_split_boundaries,
    split_frame,
)
from src.preprocessing.subsample import subsample_sessions
from src.preprocessing.timestamps import compute_t_min
from src.preprocessing.vocab import ItemVocab, build_item_vocab


def _attach_click_metadata(clicks: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    click_events = events.loc[events["is_click"], [
        "session_id",
        "timestamp",
        "row_order",
        "event_id",
        "session_idx",
        "t_sec",
        "item_idx_gru",
        "item_idx_tgn",
    ]]
    return clicks.merge(
        click_events,
        on=["session_id", "timestamp", "row_order"],
        how="left",
        suffixes=("", "_ev"),
    )


def _popularity_stats(train_clicks: pd.DataFrame, vocab: ItemVocab) -> dict[str, Any]:
    counts = train_clicks["item_id"].value_counts()
    total = len(train_clicks)
    props = counts / total
    top20_items = set(counts.head(20).index)
    return {
        "n_clicks": int(total),
        "top20_item_ids": [int(x) for x in top20_items],
        "top1pct_click_share": float(props.head(max(1, len(props) // 100)).sum()),
    }


def run_preprocessing(config: PreprocessConfig) -> Path:
    output_dir = config.output_dir()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Remove it or choose another variant."
        )

    print("Loading raw data...")
    clicks = load_clicks(config.raw_dir)
    buys = load_buys(config.raw_dir)
    test_raw = load_test(config.raw_dir)

    if config.remove_exact_duplicates:
        clicks = remove_exact_duplicates(
            clicks, ["session_id", "item_id", "timestamp"]
        )
    clicks = add_row_order(clicks)
    buys = add_row_order(buys)

    print(f"Subsampling sessions (fraction={config.subsample_fraction})...")
    clicks, buys, _kept = subsample_sessions(clicks, buys, config.subsample_fraction)

    boundaries = compute_split_boundaries(clicks, config.split_ratios)
    session_split = assign_session_splits(clicks, boundaries)
    clicks_splits = split_frame(clicks, session_split)
    buys_splits = split_frame(buys, session_split)

    train_clicks = clicks_splits["train"]
    vocab = build_item_vocab(train_clicks)
    known_items = set(vocab.item2idx.keys())
    item_buckets = build_item_buckets(train_clicks)
    t_min = compute_t_min(train_clicks)

    vocab.save(output_dir / "vocab")
    bucket_map = {name: idx for idx, name in enumerate(BUCKET_NAMES)}
    (output_dir / "vocab" / "cat_bucket2idx.json").write_text(
        json.dumps(bucket_map, indent=2), encoding="utf-8"
    )

    stats: dict[str, Any] = {
        "n_sessions": int(clicks["session_id"].nunique()),
        "n_clicks": int(len(clicks)),
        "n_buys": int(len(buys)),
        "n_items_train_vocab": vocab.n_items,
        "duplicates_removed": int(clicks.attrs.get("duplicates_removed", 0)),
        "popularity": _popularity_stats(train_clicks, vocab),
    }

    export_sequence = not config.include_buys
    boundary_payload = {
        "train_end": str(boundaries.train_end),
        "val_end": str(boundaries.val_end),
    }

    for split_name in SPLIT_NAMES:
        print(f"Processing split: {split_name}...")
        split_clicks = clicks_splits[split_name]
        split_buys = buys_splits[split_name]

        events = build_event_stream(
            split_clicks,
            split_buys,
            vocab,
            item_buckets,
            known_items,
            t_min,
            include_buys=config.include_buys,
        )
        events = assign_session_idx(events, split_name)
        clicks_enriched = _attach_click_metadata(split_clicks, events)

        examples = build_examples_for_split(
            clicks=clicks_enriched,
            events=events,
            vocab=vocab,
            known_items=known_items,
            split_name=split_name,
            min_session_clicks=config.min_session_clicks,
            eval_mode=config.eval_mode,
            is_train=(split_name == "train"),
        )

        write_split_artifacts(
            split_name=split_name,
            examples=examples,
            tgn_events=events_to_tgn_frame(events),
            output_dir=output_dir,
            include_buys=config.include_buys,
            export_sequence_models=export_sequence,
        )

        stats[f"{split_name}_examples"] = {
            "gru4rec": len(examples.gru4rec),
            "tagnn": len(examples.tagnn),
            "tgn": len(examples.tgn),
            "events": len(events),
        }

    print("Processing challenge test...")
    empty_buys = buys.iloc[0:0].copy()
    del clicks, buys, clicks_splits, buys_splits
    gc.collect()

    if config.remove_exact_duplicates:
        test_raw = remove_exact_duplicates(
            test_raw, ["session_id", "item_id", "timestamp"]
        )
    test_raw = add_row_order(test_raw)
    test_events = build_event_stream(
        test_raw,
        empty_buys,
        vocab,
        item_buckets,
        known_items,
        t_min,
        include_buys=False,
    )
    test_events = assign_session_idx(test_events, "challenge_test")
    tgn_events_frame = events_to_tgn_frame(test_events)
    write_challenge_events(tgn_events_frame, output_dir)

    test_enriched = _attach_click_metadata(test_raw, test_events)
    cold_start_items = int(
        test_raw.loc[~test_raw["item_id"].isin(known_items), "item_id"].nunique()
    )
    n_test_events = len(test_events)

    challenge_example_iter = iter_examples_for_split(
        clicks=test_enriched,
        events=test_events,
        vocab=vocab,
        known_items=known_items,
        split_name="challenge_test",
        min_session_clicks=config.min_session_clicks,
        eval_mode=config.eval_mode,
        is_train=False,
    )
    n_challenge_examples = write_examples_streaming(
        challenge_example_iter,
        output_dir / "challenge_test",
        export_sequence_models=export_sequence,
    )

    del test_raw, test_events, test_enriched, tgn_events_frame
    gc.collect()

    stats["challenge_test_examples"] = {
        "gru4rec": n_challenge_examples if export_sequence else 0,
        "tgn": n_challenge_examples,
        "events": n_test_events,
        "cold_start_items": cold_start_items,
    }

    write_meta(output_dir, config, vocab, boundary_payload, stats)
    print(f"Done. Output written to: {output_dir}")
    return output_dir
