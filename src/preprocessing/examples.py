"""Next-click example generation for train (all steps) and val/test (last click)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocessing.vocab import ItemVocab


@dataclass
class ExampleBatch:
    gru4rec: pd.DataFrame
    tagnn: pd.DataFrame
    tgn: pd.DataFrame


def _prefix_stats(event_ids: np.ndarray, target_event_id: int) -> tuple[int, int]:
    mask = event_ids < target_event_id
    if not mask.any():
        return -1, 0
    prefix_ids = event_ids[mask]
    return int(prefix_ids.max()), int(mask.sum())


def iter_examples_for_split(
    clicks: pd.DataFrame,
    events: pd.DataFrame,
    vocab: ItemVocab,
    known_items: set[int],
    split_name: str,
    min_session_clicks: int,
    eval_mode: str,
    is_train: bool,
) -> Iterator[tuple[dict, dict, dict]]:
    """Yield (gru4rec, tagnn, tgn) example dicts session-by-session without extra copies."""
    clicks_sorted = clicks.sort_values(
        ["session_id", "timestamp", "row_order"], kind="mergesort"
    )
    events_sorted = events.sort_values(
        ["session_id", "timestamp", "event_type", "row_order"], kind="mergesort"
    )

    click_groups = clicks_sorted.groupby("session_id", sort=False)
    event_groups = events_sorted.groupby("session_id", sort=False)

    example_id = 0
    for (session_id, click_df), (_, ev_df) in zip(click_groups, event_groups, strict=True):
        if len(click_df) < min_session_clicks:
            continue
        if click_df["event_id"].isna().any():
            raise ValueError(f"Missing event_id for session {session_id} in {split_name}")

        n_clicks = len(click_df)
        if is_train:
            target_positions = range(1, n_clicks)
        elif eval_mode == "last_click":
            target_positions = [n_clicks - 1]
        else:
            raise ValueError(f"Unsupported eval_mode: {eval_mode}")

        event_ids = ev_df["event_id"].to_numpy(dtype=np.int64, copy=False)

        for target_pos in target_positions:
            history_clicks = click_df.iloc[:target_pos]
            target_row = click_df.iloc[target_pos]

            history_item_idx = [
                int(vocab.gru_index(int(i), int(i) in known_items))
                for i in history_clicks["item_id"].to_numpy()
            ]
            target_item_idx = int(
                vocab.gru_index(
                    int(target_row["item_id"]),
                    int(target_row["item_id"]) in known_items,
                )
            )

            target_event_id = int(target_row["event_id"])
            prefix_last_event_id, prefix_num_events = _prefix_stats(
                event_ids, target_event_id
            )

            gru_row = {
                "example_id": example_id,
                "split": split_name,
                "session_id": int(session_id),
                "target_click_pos": int(target_pos),
                "history_item_idx": history_item_idx,
                "target_item_idx": target_item_idx,
            }
            tagnn_row = {
                "example_id": example_id,
                "split": split_name,
                "session_id": int(session_id),
                "item_ids": history_item_idx,
                "target_item_idx": target_item_idx,
                "history_len": len(history_item_idx),
            }
            tgn_row = {
                "example_id": example_id,
                "split": split_name,
                "session_id": int(session_id),
                "session_idx": int(target_row["session_idx"]),
                "target_item_idx_tgn": int(
                    vocab.tgn_index(
                        int(target_row["item_id"]),
                        int(target_row["item_id"]) in known_items,
                    )
                ),
                "target_t_sec": float(target_row["t_sec"]),
                "target_event_id": target_event_id,
                "prefix_last_event_id": prefix_last_event_id,
                "prefix_num_events": prefix_num_events,
            }
            yield gru_row, tagnn_row, tgn_row
            example_id += 1


def build_examples_for_split(
    clicks: pd.DataFrame,
    events: pd.DataFrame,
    vocab: ItemVocab,
    known_items: set[int],
    split_name: str,
    min_session_clicks: int,
    eval_mode: str,
    is_train: bool,
) -> ExampleBatch:
    """Collect all examples in memory (splits subsampled — safe for train/val/test)."""
    gru_rows: list[dict] = []
    tagnn_rows: list[dict] = []
    tgn_rows: list[dict] = []

    for gru_row, tagnn_row, tgn_row in iter_examples_for_split(
        clicks=clicks,
        events=events,
        vocab=vocab,
        known_items=known_items,
        split_name=split_name,
        min_session_clicks=min_session_clicks,
        eval_mode=eval_mode,
        is_train=is_train,
    ):
        gru_rows.append(gru_row)
        tagnn_rows.append(tagnn_row)
        tgn_rows.append(tgn_row)

    return ExampleBatch(
        gru4rec=pd.DataFrame(gru_rows),
        tagnn=pd.DataFrame(tagnn_rows),
        tgn=pd.DataFrame(tgn_rows),
    )


def count_examples_for_split(
    clicks: pd.DataFrame,
    events: pd.DataFrame,
    vocab: ItemVocab,
    known_items: set[int],
    split_name: str,
    min_session_clicks: int,
    eval_mode: str,
    is_train: bool,
) -> int:
    return sum(
        1
        for _ in iter_examples_for_split(
            clicks=clicks,
            events=events,
            vocab=vocab,
            known_items=known_items,
            split_name=split_name,
            min_session_clicks=min_session_clicks,
            eval_mode=eval_mode,
            is_train=is_train,
        )
    )
