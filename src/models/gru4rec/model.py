"""GRU4Rec: Embedding → GRU → linear next-item classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GRU4Rec(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, num_embeddings)

    def encode(self, item_ids: Tensor, lengths: Tensor) -> Tensor:
        """Return the final GRU hidden state: shape (batch, hidden_dim)."""
        embedded = self.embedding(item_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return hidden[-1]

    def forward(self, item_ids: Tensor, lengths: Tensor) -> Tensor:
        """Return logits for the next item: shape (batch, num_embeddings)."""
        return self.output(self.encode(item_ids, lengths))

    def score_candidates(
        self,
        item_ids: Tensor,
        lengths: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        """Score a candidate subset per row: shape (batch, num_candidates)."""
        hidden = self.encode(item_ids, lengths)
        weight = self.output.weight[candidate_ids]
        bias = self.output.bias[candidate_ids]
        return (hidden.unsqueeze(1) * weight).sum(dim=-1) + bias
