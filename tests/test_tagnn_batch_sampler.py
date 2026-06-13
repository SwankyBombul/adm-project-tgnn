"""Tests for TAGNN length bucketing."""

from __future__ import annotations

from src.models.tagnn.batch_sampler import LengthBucketBatchSampler


def test_length_bucket_batch_sampler_groups_similar_lengths() -> None:
    lengths = [1, 1, 2, 2, 10, 10, 3, 3]
    sampler = LengthBucketBatchSampler(lengths, batch_size=2, shuffle=False)
    batches = list(sampler)
    assert batches == [[0, 1], [2, 3], [6, 7], [4, 5]]
