"""Tests for SafeTGNMemory eval transition."""

from __future__ import annotations

import torch

from src.models.tgn.memory import SafeTGNMemory
from src.models.tgn.model import TGNModel
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator


def test_neighbor_loader_follows_module_device() -> None:
    model = TGNModel(
        num_items=5, num_sessions=2, embedding_dim=8, memory_dim=16, time_dim=8, n_neighbors=2
    )
    cpu_loader = model._neighbor_loader_for_device()
    assert cpu_loader.neighbors.device.type == "cpu"
    if torch.cuda.is_available():
        model.cuda()
        gpu_loader = model._neighbor_loader_for_device()
        assert gpu_loader.neighbors.device.type == "cuda"
        assert gpu_loader is not cpu_loader


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
