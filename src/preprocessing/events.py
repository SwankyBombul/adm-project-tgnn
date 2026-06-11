"""Build unified temporal event streams for TGN."""

from __future__ import annotations

import pandas as pd

from src.preprocessing.category import bucket_to_idx, classify_category, item_category_mode
from src.preprocessing.timestamps import to_t_sec
from src.preprocessing.vocab import ItemVocab

EVENT_CLICK = 0
EVENT_BUY = 1


def enrich_clicks(
    clicks: pd.DataFrame,
    vocab: ItemVocab,
    item_buckets: pd.Series,
    known_items: set[int],
    t_min: pd.Timestamp,
) -> pd.DataFrame:
    out = clicks.copy()
    out["t_sec"] = to_t_sec(out["timestamp"], t_min)
    out["event_type"] = EVENT_CLICK
    out["is_click"] = True
    out["price_log"] = 0.0
    out["quantity"] = 0.0
    out["known_item"] = out["item_id"].isin(known_items)
    out["item_idx_gru"] = out["item_id"].map(
        lambda x: vocab.gru_index(x, x in known_items)
    )
    out["item_idx_tgn"] = out["item_id"].map(
        lambda x: vocab.tgn_index(x, x in known_items)
    )
    out["cat_bucket"] = out["item_id"].map(item_buckets).fillna("missing")
    out["cat_bucket_idx"] = out["cat_bucket"].map(bucket_to_idx).astype("int64")
    return out


def enrich_buys(
    buys: pd.DataFrame,
    vocab: ItemVocab,
    known_items: set[int],
    t_min: pd.Timestamp,
) -> pd.DataFrame:
    out = buys.copy()
    out["t_sec"] = to_t_sec(out["timestamp"], t_min)
    out["event_type"] = EVENT_BUY
    out["is_click"] = False
    out["category"] = pd.NA
    out["cat_bucket"] = "missing"
    out["cat_bucket_idx"] = bucket_to_idx("missing")
    import numpy as np

    out["price_log"] = np.log1p(out["price"].clip(lower=0).fillna(0))
    out["quantity"] = out["quantity"].fillna(0)
    out["known_item"] = out["item_id"].isin(known_items)
    out["item_idx_gru"] = out["item_id"].map(
        lambda x: vocab.gru_index(x, x in known_items)
    )
    out["item_idx_tgn"] = out["item_id"].map(
        lambda x: vocab.tgn_index(x, x in known_items)
    )
    return out


def build_event_stream(
    clicks: pd.DataFrame,
    buys: pd.DataFrame,
    vocab: ItemVocab,
    item_buckets: pd.Series,
    known_items: set[int],
    t_min: pd.Timestamp,
    include_buys: bool,
) -> pd.DataFrame:
    click_events = enrich_clicks(clicks, vocab, item_buckets, known_items, t_min)
    frames = [click_events]
    if include_buys and len(buys) > 0:
        frames.append(enrich_buys(buys, vocab, known_items, t_min))

    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values(
        ["session_id", "timestamp", "event_type", "row_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    events["event_id"] = range(len(events))
    return events


def build_item_buckets(train_clicks: pd.DataFrame) -> pd.Series:
    return item_category_mode(train_clicks)


def assign_session_idx(events: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Map raw session_id to 0..N-1 within a split."""
    out = events.copy()
    unique_sessions = out["session_id"].drop_duplicates().sort_values()
    mapping = {sid: idx for idx, sid in enumerate(unique_sessions)}
    out["session_idx"] = out["session_id"].map(mapping).astype("int64")
    out["split"] = split_name
    return out


def events_to_tgn_frame(events: pd.DataFrame) -> pd.DataFrame:
    """Columns required for TemporalData construction in training code."""
    return events[
        [
            "event_id",
            "session_id",
            "session_idx",
            "item_id",
            "item_idx_tgn",
            "t_sec",
            "event_type",
            "cat_bucket_idx",
            "price_log",
            "quantity",
            "is_click",
            "split",
        ]
    ].copy()
