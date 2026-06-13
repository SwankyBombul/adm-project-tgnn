"""Temporal graph attention embedding (PyG TGN example)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import TransformerConv


class GraphAttentionEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        msg_dim: int,
        time_enc: nn.Module,
    ) -> None:
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels,
            out_channels // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(
        self,
        x: Tensor,
        last_update: Tensor,
        edge_index: Tensor,
        t: Tensor,
        msg: Tensor,
    ) -> Tensor:
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)
