"""Tests for popularity baselines (artifacts-only, no preprocessing imports)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.evaluation.baselines import evaluate_pop_baseline, popularity_top_k, popularity_top_k_gru_indices


class _TargetOnlyDataset(Dataset):
    def __init__(self, targets: list[int]) -> None:
        self._targets = targets

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> int:
        return self._targets[index]


def _gru4rec_collate(batch: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = torch.tensor(batch, dtype=torch.long)
    return torch.zeros(len(batch), 1), torch.ones(len(batch)), targets


def _tagnn_collate(batch: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = torch.tensor(batch, dtype=torch.long)
    return (
        torch.zeros(len(batch), 1),
        torch.zeros(len(batch), 1, 2),
        torch.zeros(len(batch), 1),
        torch.ones(len(batch), 1),
        targets,
    )


def test_evaluate_pop_baseline_supports_gru4rec_and_tagnn_batches() -> None:
    pop_indices = [1, 2, 3]
    dataset = _TargetOnlyDataset([1, 4, 2])

    gru_loader = DataLoader(dataset, batch_size=2, collate_fn=_gru4rec_collate)
    tagnn_loader = DataLoader(dataset, batch_size=2, collate_fn=_tagnn_collate)

    gru_metrics = evaluate_pop_baseline(gru_loader, pop_indices, ks=(1, 2))
    tagnn_metrics = evaluate_pop_baseline(tagnn_loader, pop_indices, ks=(1, 2))

    assert gru_metrics["recall@1_pop"] == tagnn_metrics["recall@1_pop"]
    assert gru_metrics["recall@2_pop"] == tagnn_metrics["recall@2_pop"]
    assert gru_metrics["recall@2_pop"] == pytest.approx(2 / 3)


def test_popularity_top_k_gru_indices_from_artifacts(tmp_path: Path) -> None:
    processed = tmp_path / "subsample_1_32_clicks_only"
    vocab_dir = processed / "vocab"
    vocab_dir.mkdir(parents=True)

    (processed / "meta.json").write_text(
        json.dumps(
            {
                "stats": {
                    "popularity": {
                        "top20_item_ids": [100, 200, 300],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (vocab_dir / "item_vocab.json").write_text(
        json.dumps(
            {
                "item2idx": {"100": 1, "200": 2, "999": 3},
                "idx2item": {"1": 100, "2": 200, "3": 999},
                "n_items": 3,
                "unk_raw_item_id": -1,
            }
        ),
        encoding="utf-8",
    )

    assert popularity_top_k(json.loads((processed / "meta.json").read_text()), k=2) == [
        100,
        200,
    ]
    assert popularity_top_k_gru_indices(processed, k=2) == [1, 2]
