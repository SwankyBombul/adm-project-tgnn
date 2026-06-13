"""GRU4Rec LightningDataModule for processed Yoochoose artifacts."""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from src.artifacts import gru4rec_vocab_size, load_meta, split_examples_path
from src.common.paths import get_project_root
from src.models.gru4rec.dataset import GRU4RecDataset, gru4rec_collate_fn


class GRU4RecDataModule(pl.LightningDataModule):
    """Load train/val/eval splits from processed parquet artifacts."""

    def __init__(
        self,
        processed_dir: str | Path,
        batch_size: int = 256,
        num_workers: int = 2,
        pin_memory: bool | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("pin_memory",))

        root = get_project_root()
        path = Path(processed_dir)
        self.processed_dir = path if path.is_absolute() else (root / path).resolve()

        meta = load_meta(self.processed_dir)
        self.num_embeddings = gru4rec_vocab_size(meta)

        self.batch_size = batch_size
        self.num_workers = num_workers
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        self.pin_memory = pin_memory

        self.train_dataset: GRU4RecDataset | None = None
        self.val_dataset: GRU4RecDataset | None = None
        self.test_internal_dataset: GRU4RecDataset | None = None
        self.challenge_test_dataset: GRU4RecDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            train_path = split_examples_path(self.processed_dir, "train", "gru4rec")
            self.train_dataset = GRU4RecDataset(train_path)
        if stage in ("fit", "validate", None):
            val_path = split_examples_path(self.processed_dir, "val", "gru4rec")
            self.val_dataset = GRU4RecDataset(val_path)
        if stage in ("test", None):
            test_internal_path = split_examples_path(
                self.processed_dir, "test_internal", "gru4rec"
            )
            challenge_test_path = split_examples_path(
                self.processed_dir, "challenge_test", "gru4rec"
            )
            self.test_internal_dataset = GRU4RecDataset(test_internal_path)
            self.challenge_test_dataset = GRU4RecDataset(challenge_test_path)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader().")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=gru4rec_collate_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before val_dataloader().")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=gru4rec_collate_fn,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> list[DataLoader]:
        if self.test_internal_dataset is None or self.challenge_test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader().")
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "collate_fn": gru4rec_collate_fn,
            "pin_memory": self.pin_memory,
        }
        return [
            DataLoader(self.test_internal_dataset, **loader_kwargs),
            DataLoader(self.challenge_test_dataset, **loader_kwargs),
        ]
