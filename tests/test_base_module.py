"""Tests for NextItemLitModule metric prefixes and hooks."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.evaluation.sampled import build_candidate_sets
from src.training.base_module import EVAL_DATALOADER_NAMES, NextItemLitModule


class _DummyNextItemModule(NextItemLitModule):
    def __init__(self) -> None:
        super().__init__(compute_pop_baseline=False, eval_num_negatives=2, eval_seed=0)
        self.proj = nn.Linear(4, 8)

    def compute_logits_and_targets(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, targets = batch
        return self.proj(features), targets

    def compute_sampled_scores_and_targets(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, targets = batch
        logits = self.proj(features)
        generator = self._eval_candidate_generator
        if generator is None:
            raise RuntimeError("test candidate generator not initialized")
        candidates = build_candidate_sets(
            targets,
            logits.size(1),
            self.eval_num_negatives,
            generator=generator,
        )
        scores = logits.gather(1, candidates.ids)
        return scores, targets, candidates.ids

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
    module.on_test_batch_start((features, targets), batch_idx=0, dataloader_idx=0)
    module.test_step((features, targets), batch_idx=0, dataloader_idx=0)
    module.on_test_batch_start((features, targets), batch_idx=0, dataloader_idx=1)
    module.test_step((features, targets), batch_idx=0, dataloader_idx=1)

    assert any(name.startswith("test_internal/") for name in logged)
    assert any(name.startswith("challenge_test/") for name in logged)
    assert any(name == "test_internal/sampled_recall@20" for name in logged)
    assert any(name == "challenge_test/sampled_recall@20" for name in logged)


def test_validation_step_logs_val_prefix() -> None:
    module = _DummyNextItemModule()
    logged: list[str] = []
    module.log = lambda name, value, **kwargs: logged.append(name)  # type: ignore[method-assign]

    features = torch.randn(2, 4)
    targets = torch.tensor([1, 2])
    module.validation_step((features, targets), batch_idx=0)

    assert any(name.startswith("val/") for name in logged)
    assert any(name == "val/recall@20" for name in logged)
