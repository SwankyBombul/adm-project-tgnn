"""Link prediction decoder and full-catalog scoring."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LinkDecoder(nn.Module):
    """MLP decoder on pairs of node embeddings (Twitter TGN ``MergeLayer`` style)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.lin_src = nn.Linear(channels, channels)
        self.lin_dst = nn.Linear(channels, channels)
        self.lin_final = nn.Linear(channels, 1)

    def forward(self, z_src: Tensor, z_dst: Tensor) -> Tensor:
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).squeeze(-1)

    def score_all_items(
        self,
        session_emb: Tensor,
        item_emb: Tensor,
        *,
        chunk_size: int = 4096,
    ) -> Tensor:
        """Full-catalog logits ``(batch, num_items)`` via chunked MLP scoring."""
        batch_size = session_emb.size(0)
        num_items = item_emb.size(0)
        logits = session_emb.new_empty(batch_size, num_items)
        for start in range(0, num_items, chunk_size):
            end = min(num_items, start + chunk_size)
            chunk = item_emb[start:end]
            src = session_emb.unsqueeze(1).expand(-1, chunk.size(0), -1)
            dst = chunk.unsqueeze(0).expand(batch_size, -1, -1)
            flat_src = src.reshape(-1, session_emb.size(-1))
            flat_dst = dst.reshape(-1, chunk.size(-1))
            scores = self.forward(flat_src, flat_dst).view(batch_size, chunk.size(0))
            logits[:, start:end] = scores
        return logits
