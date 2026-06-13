"""Tests for NextItemLitModule metric prefixes and hooks."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.training.base_module import EVAL_DATALOADER_NAMES, NextItemLitModule


class _DummyNextItemModule(NextItemLitModule):
    def __init__(self) -> None:
        super().__init__(compute_pop_baseline=False)
        self.proj = nn.Linear(4, 8)

    def compute_logits_and_targets(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, targets = batch
        return self.proj(features), targets

    def popularity_indices(self, processed_dir: Path) -> list[int]:
        return [1, 2, 3]


def test_eval_dataloader_names() -> None:
    assert EVAL_DATALOADER_NAMES == ("test_internal", "challenge_test")


def test_test_step_logs_split_prefix() -> None:
    module = _DummyNextItemModule()
    logged: list[str] = []
    module.log = lambda name, value, **kwargs: logged.append(name)  # type: ignore[method-assign]

    features = torch.randn(2, 4)
    targets = torch.tensor([1, 2])
    module.test_step((features, targets), batch_idx=0, dataloader_idx=0)
    module.test_step((features, targets), batch_idx=0, dataloader_idx=1)

    assert any(name.startswith("test_internal/") for name in logged)
    assert any(name.startswith("challenge_test/") for name in logged)
    assert any(name == "test_internal/recall@20" for name in logged)
    assert any(name == "challenge_test/recall@20" for name in logged)


def test_validation_step_logs_val_prefix() -> None:
    module = _DummyNextItemModule()
    logged: list[str] = []
    module.log = lambda name, value, **kwargs: logged.append(name)  # type: ignore[method-assign]

    features = torch.randn(2, 4)
    targets = torch.tensor([1, 2])
    module.validation_step((features, targets), batch_idx=0)

    assert any(name.startswith("val/") for name in logged)
    assert any(name == "val/recall@20" for name in logged)
