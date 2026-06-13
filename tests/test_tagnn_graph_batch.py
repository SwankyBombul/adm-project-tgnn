"""Tests for TAGNN session graph batching."""

from __future__ import annotations

import numpy as np

from src.models.tagnn.graph_batch import build_session_graph, tagnn_collate_fn


def test_build_session_graph_sequential_edges() -> None:
    alias, adj, items, mask = build_session_graph([1, 2, 3])
    assert mask == [1, 1, 1]
    assert items == [1, 2, 3]
    assert alias == [0, 1, 2]
    assert adj.shape == (3, 6)
    assert float(adj.sum()) > 0


def test_build_session_graph_repeated_items_share_nodes() -> None:
    alias, adj, items, mask = build_session_graph([1, 2, 1, 3])
    assert items == [1, 2, 3]
    assert alias == [0, 1, 0, 2]
    assert adj.shape == (3, 6)


def test_build_session_graph_stops_at_padding() -> None:
    alias, adj, items, mask = build_session_graph([1, 2, 0, 0])
    assert mask == [1, 1, 0, 0]
    assert items == [1, 2]
    assert adj.shape == (2, 4)
    assert alias == [0, 1, 0, 0]


def test_tagnn_collate_fn_shapes() -> None:
    batch = [([1, 2, 3], 4), ([5, 6], 7)]
    alias, adj, items, mask, targets = tagnn_collate_fn(batch)

    assert alias.shape == (2, 3)
    assert adj.shape == (2, 3, 6)
    assert items.shape == (2, 3)
    assert mask.shape == (2, 3)
    assert targets.tolist() == [4, 7]


def test_tagnn_collate_truncates_long_sessions() -> None:
    long_seq = list(range(1, 80))
    alias, _, _, mask, _ = tagnn_collate_fn([(long_seq, 79)], max_seq_len=50)
    assert sum(mask[0].tolist()) == 50


def test_tagnn_collate_precomputed_graphs() -> None:
    graph_a = build_session_graph([1, 2, 3])
    graph_b = build_session_graph([4, 5])
    alias, adj, items, mask, targets = tagnn_collate_fn([(graph_a, 6), (graph_b, 7)])
    assert alias.shape == (2, 3)
    assert adj.shape == (2, 3, 6)
    assert targets.tolist() == [6, 7]
    assert sum(mask[1].tolist()) == 2
