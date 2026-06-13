"""TAGNN LightningDataModule for processed Yoochoose artifacts."""

from __future__ import annotations

import sys
import warnings
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from src.artifacts import gru4rec_vocab_size, load_meta, split_examples_path
from src.common.paths import get_project_root
from src.models.tagnn.batch_sampler import LengthBucketBatchSampler
from src.models.tagnn.dataset import TAGNNDataset
from src.models.tagnn.graph_batch import tagnn_collate_fn


def _resolve_num_workers(num_workers: int) -> int:
    """Windows spawn + Lightning cannot reliably start DataLoader worker processes."""
    if sys.platform == "win32" and num_workers > 0:
        warnings.warn(
            "TAGNNDataModule: num_workers is forced to 0 on Windows "
            "(multiprocessing DataLoader is unreliable with Lightning).",
            stacklevel=3,
        )
        return 0
    return num_workers


class TAGNNDataModule(pl.LightningDataModule):
    """Load train/val/eval TAGNN splits from processed artifacts."""

    def __init__(
        self,
        processed_dir: str | Path,
        batch_size: int = 256,
        num_workers: int = 2,
        pin_memory: bool | None = None,
        max_seq_len: int | None = 50,
        precompute_graphs: bool = True,
        length_bucketing: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("pin_memory",))

        root = get_project_root()
        path = Path(processed_dir)
        self.processed_dir = path if path.is_absolute() else (root / path).resolve()

        meta = load_meta(self.processed_dir)
        self.num_embeddings = gru4rec_vocab_size(meta)

        self.batch_size = batch_size
        self.num_workers = _resolve_num_workers(num_workers)
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        self.pin_memory = pin_memory
        self.max_seq_len = max_seq_len
        if precompute_graphs and self.num_workers > 0:
            precompute_graphs = False
        self.precompute_graphs = precompute_graphs
        self.length_bucketing = length_bucketing

        self.train_dataset: TAGNNDataset | None = None
        self.val_dataset: TAGNNDataset | None = None
        self.test_internal_dataset: TAGNNDataset | None = None
        self.challenge_test_dataset: TAGNNDataset | None = None

    def _make_dataset(self, examples_path: Path) -> TAGNNDataset:
        return TAGNNDataset(
            examples_path,
            max_seq_len=self.max_seq_len,
            precompute_graphs=self.precompute_graphs,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            train_path = split_examples_path(self.processed_dir, "train", "tagnn")
            self.train_dataset = self._make_dataset(train_path)
        if stage in ("fit", "validate", None):
            val_path = split_examples_path(self.processed_dir, "val", "tagnn")
            self.val_dataset = self._make_dataset(val_path)
        if stage in ("test", None):
            test_internal_path = split_examples_path(
                self.processed_dir, "test_internal", "tagnn"
            )
            challenge_test_path = split_examples_path(
                self.processed_dir, "challenge_test", "tagnn"
            )
            self.test_internal_dataset = self._make_dataset(test_internal_path)
            self.challenge_test_dataset = self._make_dataset(challenge_test_path)

    def _collate_fn(self):
        return partial(tagnn_collate_fn, max_seq_len=self.max_seq_len)

    def _loader_kwargs(self) -> dict:
        kwargs: dict = {
            "num_workers": self.num_workers,
            "collate_fn": self._collate_fn(),
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader().")
        loader_kwargs = self._loader_kwargs()
        if self.length_bucketing:
            lengths = self.train_dataset.example_lengths()
            return DataLoader(
                self.train_dataset,
                batch_sampler=LengthBucketBatchSampler(
                    lengths,
                    batch_size=self.batch_size,
                    shuffle=True,
                ),
                **loader_kwargs,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **loader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before val_dataloader().")
        loader_kwargs = self._loader_kwargs()
        if self.length_bucketing:
            lengths = self.val_dataset.example_lengths()
            return DataLoader(
                self.val_dataset,
                batch_sampler=LengthBucketBatchSampler(
                    lengths,
                    batch_size=self.batch_size,
                    shuffle=False,
                ),
                **loader_kwargs,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    def test_dataloader(self) -> list[DataLoader]:
        if self.test_internal_dataset is None or self.challenge_test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader().")
        loader_kwargs = self._loader_kwargs()
        loaders = []
        for dataset in (self.test_internal_dataset, self.challenge_test_dataset):
            if self.length_bucketing:
                lengths = dataset.example_lengths()
                loaders.append(
                    DataLoader(
                        dataset,
                        batch_sampler=LengthBucketBatchSampler(
                            lengths,
                            batch_size=self.batch_size,
                            shuffle=False,
                        ),
                        **loader_kwargs,
                    )
                )
            else:
                loaders.append(
                    DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        **loader_kwargs,
                    )
                )
        return loaders
