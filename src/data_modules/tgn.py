"""TGN LightningDataModule for processed Yoochoose artifacts."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from src.artifacts import load_meta, split_events_path, split_examples_path, tgn_num_items
from src.common.paths import get_project_root
from src.models.tgn.dataset import TGNEventStreamDataset, TGNExampleDataset, load_events_tensors
from src.models.tgn.temporal_batch import tgn_event_collate_fn, tgn_example_collate_fn


def _resolve_num_workers(num_workers: int) -> int:
    if sys.platform == "win32" and num_workers > 0:
        warnings.warn(
            "TGNDataModule: num_workers is forced to 0 on Windows.",
            stacklevel=3,
        )
        return 0
    return num_workers


class TGNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        processed_dir: str | Path,
        *,
        event_batch_size: int = 200,
        example_batch_size: int = 32,
        val_max_examples: int | None = 2000,
        num_workers: int = 0,
        pin_memory: bool | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("pin_memory",))

        root = get_project_root()
        path = Path(processed_dir)
        self.processed_dir = path if path.is_absolute() else (root / path).resolve()
        meta = load_meta(self.processed_dir)
        self.num_items = tgn_num_items(meta)
        self.event_batch_size = event_batch_size
        self.example_batch_size = example_batch_size
        self.val_max_examples = val_max_examples
        self.num_workers = _resolve_num_workers(num_workers)
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        self.pin_memory = pin_memory

        self.train_events = load_events_tensors(
            split_events_path(self.processed_dir, "train")
        )
        self.num_sessions_train = self.train_events.num_sessions
        self._train_item_counts = torch.bincount(
            self.train_events.item_idx_tgn,
            minlength=self.num_items,
        ).to(dtype=torch.float32)
        self._session_offsets: dict[str, int] | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_event_dataset = TGNEventStreamDataset(
                split_events_path(self.processed_dir, "train"),
                self.event_batch_size,
            )
        if stage in ("fit", "validate", None):
            self.val_example_dataset = TGNExampleDataset(
                split_examples_path(self.processed_dir, "val", "tgn"),
                sort_by_prefix=True,
                max_examples=self.val_max_examples if stage in ("fit", None) else None,
            )
            self.val_events = load_events_tensors(
                split_events_path(self.processed_dir, "val")
            )
        if stage in ("test", None):
            self.test_internal_dataset = TGNExampleDataset(
                split_examples_path(self.processed_dir, "test_internal", "tgn"),
                sort_by_target=True,
            )
            self.test_internal_events = load_events_tensors(
                split_events_path(self.processed_dir, "test_internal")
            )
            self.challenge_dataset = TGNExampleDataset(
                split_examples_path(self.processed_dir, "challenge_test", "tgn"),
                sort_by_target=True,
            )
            self.challenge_events = load_events_tensors(
                split_events_path(self.processed_dir, "challenge_test")
            )

    def on_before_batch_transfer(self, batch, dataloader_idx: int = 0):  # noqa: ANN001
        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int = 0):  # noqa: ANN001
        if isinstance(batch, dict):
            return {k: v.to(device) for k, v in batch.items()}
        from src.models.tgn.dataset import TGNExampleBatch

        if isinstance(batch, TGNExampleBatch):
            return TGNExampleBatch(
                session_idx=batch.session_idx.to(device),
                target_item_idx_tgn=batch.target_item_idx_tgn.to(device),
                target_t_sec=batch.target_t_sec.to(device),
                target_event_id=batch.target_event_id.to(device),
                prefix_last_event_id=batch.prefix_last_event_id.to(device),
            )
        return batch

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_event_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=tgn_event_collate_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_example_dataset,
            batch_size=self.example_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=tgn_example_collate_fn,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> list[DataLoader]:
        kwargs = {
            "batch_size": self.example_batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "collate_fn": tgn_example_collate_fn,
            "pin_memory": self.pin_memory,
        }
        return [
            DataLoader(self.test_internal_dataset, **kwargs),
            DataLoader(self.challenge_dataset, **kwargs),
        ]

    def attach_events_to_module(self, module) -> None:  # noqa: ANN001
        """Wire event tensors into ``TGNLitModule`` (train-sized memory graph)."""
        device = module.device
        module.set_event_tensors(
            train_events=self.train_events.to(device),
            eval_events=self.train_events.to(device),
        )
        module.model.set_session_offset(0)

    def session_offset(self, split: str) -> int:
        if self._session_offsets is None:
            offsets: dict[str, int] = {}
            cumulative = 0
            for split_name in ("train", "val", "test_internal", "challenge_test"):
                offsets[split_name] = cumulative
                path = split_events_path(self.processed_dir, split_name)
                if path.is_file():
                    cumulative += load_events_tensors(path).num_sessions
            self._session_offsets = offsets
        return self._session_offsets[split]

    def set_eval_split(self, module, split: str) -> None:  # noqa: ANN001
        device = module.device
        if split == "train":
            events = self.train_events.to(device)
        elif split == "val":
            events = self.val_events.to(device)
        elif split == "test_internal":
            events = self.test_internal_events.to(device)
        elif split == "challenge_test":
            events = self.challenge_events.to(device)
        else:
            raise ValueError(split)
        module.set_event_tensors(eval_events=events)
        module.model.set_session_offset(self.session_offset(split))

    def train_item_sampling_weights(self, *, alpha: float = 1.0) -> torch.Tensor:
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        counts = self._train_item_counts
        # Additive smoothing keeps zero-count items sampleable when needed.
        weights = (counts + 1.0).pow(alpha)
        total = weights.sum()
        if float(total.item()) <= 0.0:
            return torch.full_like(weights, 1.0 / float(weights.numel()))
        return weights / total
