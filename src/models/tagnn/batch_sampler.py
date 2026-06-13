"""Batch samplers for TAGNN training."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group similar-length sessions so batch padding stays small.

    Sorts examples by sequence length, forms fixed-size batches from consecutive
    indices, then shuffles batch order each epoch.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        indices = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        if self.shuffle:
            random.shuffle(batches)
        yield from batches
