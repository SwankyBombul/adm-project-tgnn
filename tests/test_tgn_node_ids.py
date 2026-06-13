"""Tests for TGN node id helpers."""

from __future__ import annotations

import torch

from src.models.tgn.node_ids import edge_endpoints, num_nodes, session_global_id


def test_session_global_id_offsets_items() -> None:
    assert session_global_id(0, num_items=5) == 5
    assert session_global_id(torch.tensor([0, 1]), num_items=5).tolist() == [5, 6]


def test_edge_endpoints_bipartite() -> None:
    src, dst = edge_endpoints(
        torch.tensor([0, 1]),
        torch.tensor([2, 3]),
        num_items=10,
    )
    assert src.tolist() == [10, 11]
    assert dst.tolist() == [2, 3]
    assert num_nodes(10, 3) == 13
