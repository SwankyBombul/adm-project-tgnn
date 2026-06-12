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

    def forward(self, item_ids: Tensor, lengths: Tensor) -> Tensor:
        """Return logits for the next item: shape (batch, num_embeddings)."""
        embedded = self.embedding(item_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        last_hidden = hidden[-1]
        return self.output(last_hidden)
