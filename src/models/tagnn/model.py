"""TAGNN: Target Attentive Graph Neural Network (SIGIR 2020).

Ported from https://github.com/CRIPAC-DIG/TAGNN (model.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module, Parameter


class GNN(Module):
    """Gated graph neural network propagation (SR-GNN / TAGNN)."""

    def __init__(self, hidden_size: int, step: int = 1) -> None:
        super().__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.empty(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.empty(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.empty(self.gate_size))
        self.b_hh = Parameter(torch.empty(self.gate_size))
        self.b_iah = Parameter(torch.empty(hidden_size))
        self.b_oah = Parameter(torch.empty(hidden_size))
        self.linear_edge_in = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(hidden_size, hidden_size, bias=True)

    def gnn_cell(self, adjacency: Tensor, hidden: Tensor) -> Tensor:
        n_nodes = adjacency.size(1)
        input_in = (
            torch.matmul(adjacency[:, :, :n_nodes], self.linear_edge_in(hidden)) + self.b_iah
        )
        input_out = (
            torch.matmul(adjacency[:, :, n_nodes:], self.linear_edge_out(hidden)) + self.b_oah
        )
        inputs = torch.cat([input_in, input_out], dim=2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, dim=2)
        h_r, h_i, h_n = gh.chunk(3, dim=2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        return new_gate + input_gate * (hidden - new_gate)

    def forward(self, adjacency: Tensor, hidden: Tensor) -> Tensor:
        for _ in range(self.step):
            hidden = self.gnn_cell(adjacency, hidden)
        return hidden


class TAGNN(Module):
    """Session graph model with target attentive decoding."""

    def __init__(
        self,
        num_embeddings: int,
        hidden_dim: int = 100,
        gnn_steps: int = 1,
        nonhybrid: bool = False,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.nonhybrid = nonhybrid
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(num_embeddings, hidden_dim, padding_idx=pad_idx)
        self.gnn = GNN(hidden_dim, step=gnn_steps)
        self.linear_one = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_two = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_three = nn.Linear(hidden_dim, 1, bias=False)
        self.linear_transform = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.linear_t = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.target_attn_chunk_size = 4096
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, items: Tensor, adjacency: Tensor) -> Tensor:
        """Return node hidden states: (batch, max_n_node, hidden_dim)."""
        hidden = self.embedding(items)
        return self.gnn(adjacency, hidden)

    def sequence_hidden(self, node_hidden: Tensor, alias_inputs: Tensor) -> Tensor:
        """Map node states to sequence positions via alias indices."""
        batch_size, max_seq = alias_inputs.shape
        batch_idx = torch.arange(batch_size, device=alias_inputs.device).unsqueeze(1)
        batch_idx = batch_idx.expand(-1, max_seq)
        return node_hidden[batch_idx, alias_inputs]

    def compute_logits(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """Full-catalog logits (batch, num_embeddings)."""
        lengths = mask.long().sum(dim=1).clamp(min=1)
        last_idx = lengths - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        ht = hidden[batch_idx, last_idx]

        q1 = self.linear_one(ht).unsqueeze(1)
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        alpha = F.softmax(alpha, dim=1)
        mask_f = mask.unsqueeze(-1).float()
        session_repr = torch.sum(alpha * hidden * mask_f, dim=1)

        if not self.nonhybrid:
            session_repr = self.linear_transform(torch.cat([session_repr, ht], dim=1))

        masked_hidden = hidden * mask_f
        qt = self.linear_t(masked_hidden)
        qt_t = qt.transpose(1, 2)

        item_emb = self.embedding.weight
        session_broadcast = session_repr.unsqueeze(1)
        logits = hidden.new_zeros(hidden.size(0), self.num_embeddings)

        chunk_size = self.target_attn_chunk_size
        for start in range(0, self.num_embeddings, chunk_size):
            end = min(start + chunk_size, self.num_embeddings)
            chunk_emb = item_emb[start:end]
            attn_scores = torch.einsum("cd,bds->bcs", chunk_emb, qt_t)
            beta = F.softmax(attn_scores, dim=-1)
            target_repr = beta @ masked_hidden
            combined = session_broadcast + target_repr
            logits[:, start:end] = torch.sum(combined * chunk_emb.unsqueeze(0), dim=-1)

        return logits

    def compute_logits_for_candidates(
        self,
        hidden: Tensor,
        mask: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        """Sampled-catalog logits: shape (batch, num_candidates)."""
        lengths = mask.long().sum(dim=1).clamp(min=1)
        last_idx = lengths - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        ht = hidden[batch_idx, last_idx]

        q1 = self.linear_one(ht).unsqueeze(1)
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        alpha = F.softmax(alpha, dim=1)
        mask_f = mask.unsqueeze(-1).float()
        session_repr = torch.sum(alpha * hidden * mask_f, dim=1)

        if not self.nonhybrid:
            session_repr = self.linear_transform(torch.cat([session_repr, ht], dim=1))

        masked_hidden = hidden * mask_f
        qt = self.linear_t(masked_hidden)
        qt_t = qt.transpose(1, 2)

        candidate_emb = self.embedding.weight[candidate_ids]
        session_broadcast = session_repr.unsqueeze(1)
        attn_scores = torch.einsum("bch,bhs->bcs", candidate_emb, qt_t)
        beta = F.softmax(attn_scores, dim=-1)
        target_repr = beta @ masked_hidden
        combined = session_broadcast + target_repr
        return torch.sum(combined * candidate_emb, dim=-1)
