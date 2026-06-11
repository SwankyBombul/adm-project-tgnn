"""Next-click example generation for train (all steps) and val/test (last click)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.preprocessing.vocab import ItemVocab


@dataclass
class ExampleBatch:
    gru4rec: pd.DataFrame
    tagnn: pd.DataFrame
    tgn: pd.DataFrame


def _click_rows_per_session(clicks: pd.DataFrame) -> dict[int, pd.DataFrame]:
    ordered = clicks.sort_values(
        ["session_id", "timestamp", "row_order"], kind="mergesort"
    )
    return {sid: grp.reset_index(drop=True) for sid, grp in ordered.groupby("session_id")}


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
    """Build GRU4Rec, TAGNN, and TGN supervision rows for one split."""
    gru_rows: list[dict] = []
    tagnn_rows: list[dict] = []
    tgn_rows: list[dict] = []

    session_clicks = _click_rows_per_session(clicks)
    events_by_session = {
        sid: grp.reset_index(drop=True)
        for sid, grp in events.sort_values(
            ["session_id", "timestamp", "event_type", "row_order"], kind="mergesort"
        ).groupby("session_id")
    }

    example_id = 0
    for session_id, click_df in session_clicks.items():
        if len(click_df) < min_session_clicks:
            continue
        if click_df["event_id"].isna().any():
            raise ValueError(f"Missing event_id for session {session_id} in {split_name}")

        click_positions = list(range(len(click_df)))
        if is_train:
            target_positions = click_positions[1:]
        elif eval_mode == "last_click":
            target_positions = [click_positions[-1]]
        else:
            raise ValueError(f"Unsupported eval_mode: {eval_mode}")

        ev_df = events_by_session[session_id]

        for target_pos in target_positions:
            history_clicks = click_df.iloc[:target_pos]
            target_row = click_df.iloc[target_pos]

            history_item_idx = [
                int(vocab.gru_index(int(i), int(i) in known_items))
                for i in history_clicks["item_id"]
            ]
            target_item_idx = int(
                vocab.gru_index(
                    int(target_row["item_id"]),
                    int(target_row["item_id"]) in known_items,
                )
            )

            gru_rows.append(
                {
                    "example_id": example_id,
                    "split": split_name,
                    "session_id": int(session_id),
                    "target_click_pos": int(target_pos),
                    "history_item_idx": history_item_idx,
                    "target_item_idx": target_item_idx,
                }
            )

            tagnn_rows.append(
                {
                    "example_id": example_id,
                    "split": split_name,
                    "session_id": int(session_id),
                    "item_ids": history_item_idx + [target_item_idx],
                    "target_item_idx": target_item_idx,
                    "history_len": len(history_item_idx),
                }
            )

            target_event_id = int(target_row["event_id"])
            prefix = ev_df[ev_df["event_id"] < target_event_id]

            tgn_rows.append(
                {
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
                    "prefix_last_event_id": int(prefix["event_id"].max())
                    if len(prefix)
                    else -1,
                    "prefix_num_events": len(prefix),
                }
            )
            example_id += 1

    return ExampleBatch(
        gru4rec=pd.DataFrame(gru_rows),
        tagnn=pd.DataFrame(tagnn_rows),
        tgn=pd.DataFrame(tgn_rows),
    )
