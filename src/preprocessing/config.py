"""Configuration for the Yoochoose preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.common.constants import PAD_IDX
from src.common.paths import get_project_root


@dataclass
class PreprocessConfig:
    """Pipeline settings aligned with EDA decisions and model input contracts."""

    subsample_fraction: float = 1 / 32
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    min_session_clicks: int = 2
    remove_exact_duplicates: bool = True
    merge_consecutive_repeats: bool = False
    include_buys: bool = False
    eval_mode: str = "last_click"
    raw_dir: Path = field(default_factory=lambda: get_project_root() / "data" / "raw")
    output_root: Path = field(default_factory=lambda: get_project_root() / "data" / "processed")

    def output_dir(self) -> Path:
        frac_label = f"subsample_{self._fraction_label()}"
        variant = "with_buys" if self.include_buys else "clicks_only"
        return self.output_root / f"{frac_label}_{variant}"

    def _fraction_label(self) -> str:
        if self.subsample_fraction >= 1.0:
            return "full"
        denom = round(1 / self.subsample_fraction)
        return f"1_{denom}"

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        project_root: Path | None = None,
        **overrides: Any,
    ) -> PreprocessConfig:
        """Load preprocessing config from ``config/*.yaml``; ``overrides`` win on conflicts."""
        from src.config.yaml_loader import load_yaml_file, preprocess_config_kwargs

        data = load_yaml_file(path, project_root=project_root)
        kwargs = preprocess_config_kwargs(data, project_root=project_root)
        kwargs.update(overrides)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["raw_dir"] = str(self.raw_dir)
        data["output_root"] = str(self.output_root)
        data["output_dir"] = str(self.output_dir())
        data["split_ratios"] = list(self.split_ratios)
        return data


# --- Model input contracts (for downstream training code) ---
#
# GRU4Rec
#   - File: {split}/gru4rec_examples.parquet
#   - Columns: example_id, session_id, history_item_idx (list[int]), target_item_idx (int)
#   - history uses PAD_IDX=0 only in batching; stored lists have no padding
#   - item_idx: 1..n_items known, n_items+1 UNK
#   - Train: all next-click steps; val/test: last click only
#
# TAGNN (session graph, per-click nodes — same as SR-GNN/TAGNN papers)
#   - File: {split}/tagnn_sessions.pkl list[dict]
#   - Each session: item_ids (list[int], one per click), target_item_idx, session_id
#   - Graph built in model: nodes 0..len-1, edges (i -> i+1)
#   - Train: one record per next-click step; val/test: last click only
#
# TGN (PyG TemporalData-compatible export)
#   - File: {split}/tgn/events.parquet
#   - Columns: event_id, session_idx, item_idx, t_sec, event_type,
#              cat_bucket_idx, price_log, quantity, is_click
#   - Bipartite link: session_idx -> item_idx at time t_sec
#   - item_idx: 0..n_items-1 known, n_items UNK (no padding)
#   - include_buys=False: only clicks, event_type=0, price/quantity=0
#   - include_buys=True: clicks + buys; positives for next-click only on is_click rows
#   - File: {split}/tgn/examples.parquet — next-click supervision rows
#
# Challenge test (clicks only, no subsample): challenge_test/
