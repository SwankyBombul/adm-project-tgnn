"""Item vocabulary and index mappings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.preprocessing.config import PAD_IDX


@dataclass
class ItemVocab:
    item2idx: dict[int, int]
    idx2item: dict[int, int]
    unk_item_id: int
    n_items: int

    @property
    def gru_vocab_size(self) -> int:
        """Embedding size for GRU4Rec: pad + items + unk."""
        return self.n_items + 2

    @property
    def tgn_num_items(self) -> int:
        """Item nodes in TGN: known items + UNK index."""
        return self.n_items + 1

    def gru_index(self, raw_item_id: int, known: bool) -> int:
        if not known:
            return self.n_items + 1
        return self.item2idx[raw_item_id]

    def tgn_index(self, raw_item_id: int, known: bool) -> int:
        if not known:
            return self.n_items
        return self.item2idx[raw_item_id] - 1

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "pad_idx": PAD_IDX,
            "n_items": int(self.n_items),
            "unk_gru_idx": int(self.n_items + 1),
            "unk_tgn_idx": int(self.n_items),
            "item2idx": {str(int(k)): int(v) for k, v in self.item2idx.items()},
            "idx2item": {str(int(k)): int(v) for k, v in self.idx2item.items()},
            "unk_raw_item_id": int(self.unk_item_id),
        }
        (directory / "item_vocab.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def build_item_vocab(train_clicks: pd.DataFrame) -> ItemVocab:
    unique_items = sorted(train_clicks["item_id"].unique())
    item2idx = {item_id: idx + 1 for idx, item_id in enumerate(unique_items)}
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}
    return ItemVocab(
        item2idx=item2idx,
        idx2item=idx2item,
        unk_item_id=-1,
        n_items=len(unique_items),
    )


def load_item_vocab(directory: Path) -> ItemVocab:
    payload = json.loads((directory / "item_vocab.json").read_text(encoding="utf-8"))
    item2idx = {int(k): int(v) for k, v in payload["item2idx"].items()}
    idx2item = {int(k): int(v) for k, v in payload["idx2item"].items()}
    return ItemVocab(
        item2idx=item2idx,
        idx2item=idx2item,
        unk_item_id=int(payload["unk_raw_item_id"]),
        n_items=int(payload["n_items"]),
    )
