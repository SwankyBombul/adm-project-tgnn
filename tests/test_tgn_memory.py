"""Tests for SafeTGNMemory eval transition."""

from __future__ import annotations

import torch

from src.models.tgn.memory import SafeTGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator


def test_safe_tgn_memory_skips_full_graph_flush_on_eval() -> None:
    memory = SafeTGNMemory(
        1000,
        4,
        memory_dim=16,
        time_dim=8,
        message_module=IdentityMessage(4, 16, 8),
        aggregator_module=LastAggregator(),
    )
    memory.train()
    src = torch.tensor([0, 1])
    dst = torch.tensor([2, 3])
    t = torch.tensor([1, 2])
    msg = torch.zeros(2, 4)
    memory.update_state(src, dst, t, msg)
    memory.eval()  # would OOM / spike on stock TGNMemory with huge num_nodes
    assert not memory.training
