"""Category bucketing for click edge features."""

from __future__ import annotations

import pandas as pd

BUCKET_NAMES = [
    "no_category",
    "special_offer",
    "product_category",
    "brand_context",
    "other",
    "missing",
]


def classify_category(value: str) -> str:
    if pd.isna(value):
        return "missing"
    if value == "S":
        return "special_offer"
    if value == "0":
        return "no_category"
    if value in {str(i) for i in range(1, 13)}:
        return "product_category"
    if value.isdigit() and len(value) >= 8:
        return "brand_context"
    return "other"


def bucket_to_idx(bucket: str) -> int:
    return BUCKET_NAMES.index(bucket)


def item_category_mode(clicks: pd.DataFrame) -> pd.Series:
    """Most frequent category bucket per item (train-only)."""
    buckets = clicks["category"].astype(str).map(classify_category)
    tmp = clicks.assign(cat_bucket=buckets)
    mode = tmp.groupby("item_id")["cat_bucket"].agg(
        lambda s: s.value_counts().index[0]
    )
    return mode
