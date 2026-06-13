"""Session graph construction for TAGNN batches (port of CRIPAC-DIG/TAGNN utils)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from src.common.constants import PAD_IDX


def _as_int_list(item_ids: Sequence[int] | np.ndarray) -> list[int]:
    if isinstance(item_ids, np.ndarray):
        return item_ids.astype(np.int64, copy=False).tolist()
    if hasattr(item_ids, "tolist") and not isinstance(item_ids, (str, bytes)):
        converted = item_ids.tolist()
        if converted is not item_ids:
            return _as_int_list(converted)
    return [int(item) for item in item_ids]


def build_session_graph(
    item_ids: Sequence[int] | np.ndarray,
) -> tuple[list[int], np.ndarray, list[int], list[int]]:
    """Build alias inputs, adjacency, unique items, and mask for one session.

    Returns:
        alias_inputs: sequence position -> index in unique node list
        adjacency: (n_node, 2 * n_node) normalized in/out edges
        items: unique item ids padded (caller pads to batch max_n_node)
        mask: 1 for real sequence positions
    """
    u_input = _as_int_list(item_ids)
    mask = [1 if item != PAD_IDX else 0 for item in u_input]

    non_pad = [item for item in u_input if item != PAD_IDX]
    if not non_pad:
        return [], np.zeros((0, 0), dtype=np.float32), [], mask

    node = np.unique(np.asarray(non_pad, dtype=np.int64))
    n_node = int(node.shape[0])
    items = node.tolist()
    item_to_idx = {int(item): idx for idx, item in enumerate(node)}

    adj = np.zeros((n_node, n_node), dtype=np.float64)
    for idx in range(len(u_input) - 1):
        if u_input[idx + 1] == PAD_IDX:
            break
        src = item_to_idx[u_input[idx]]
        dst = item_to_idx[u_input[idx + 1]]
        adj[src, dst] = 1.0

    sum_in = adj.sum(axis=0)
    sum_in[sum_in == 0] = 1.0
    adj_in = adj / sum_in

    sum_out = adj.sum(axis=1)
    sum_out[sum_out == 0] = 1.0
    adj_out = (adj.T / sum_out).T

    adjacency = np.concatenate([adj_in, adj_out], axis=0).T.astype(np.float32)
    alias_inputs = [item_to_idx[item] if item != PAD_IDX else 0 for item in u_input]
    return alias_inputs, adjacency, items, mask


def _truncate_item_ids(item_ids: list[int], max_seq_len: int | None) -> list[int]:
    if max_seq_len is None or len(item_ids) <= max_seq_len:
        return item_ids
    return item_ids[-max_seq_len:]


SessionGraph = tuple[list[int], np.ndarray, list[int], list[int]]
TAGNNExample = tuple[list[int] | SessionGraph, int]


def _is_precomputed_graph(value: list[int] | SessionGraph) -> bool:
    return isinstance(value, tuple) and len(value) == 4


def _pad_graph_batch(
    graph_parts: list[SessionGraph],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    max_seq = max(len(alias) for alias, _, _, _ in graph_parts)
    max_n_node = max(len(items) for _, _, items, _ in graph_parts)
    if max_n_node == 0:
        max_n_node = 1

    alias_batch: list[list[int]] = []
    items_batch: list[list[int]] = []
    masks: list[list[int]] = []
    adj_batch: list[np.ndarray] = []

    for alias, adj, items, mask in graph_parts:
        alias_batch.append(alias + [0] * (max_seq - len(alias)))
        masks.append(mask + [0] * (max_seq - len(mask)))
        padded_items = items + [PAD_IDX] * (max_n_node - len(items))
        items_batch.append(padded_items)

        if adj.size == 0:
            adj = np.zeros((max_n_node, 2 * max_n_node), dtype=np.float32)
        else:
            n_node = adj.shape[0]
            padded_adj = np.zeros((max_n_node, 2 * max_n_node), dtype=np.float32)
            padded_adj[:n_node, : adj.shape[1]] = adj
            adj = padded_adj
        adj_batch.append(adj)

    alias_tensor = torch.tensor(alias_batch, dtype=torch.long)
    adj_tensor = torch.tensor(np.stack(adj_batch), dtype=torch.float32)
    items_tensor = torch.tensor(items_batch, dtype=torch.long)
    mask_tensor = torch.tensor(masks, dtype=torch.long)
    return alias_tensor, adj_tensor, items_tensor, mask_tensor


def tagnn_collate_fn(
    batch: list[TAGNNExample],
    max_seq_len: int | None = 50,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate TAGNN examples into padded graph tensors."""
    sequences_or_graphs, targets = zip(*batch, strict=True)
    if _is_precomputed_graph(sequences_or_graphs[0]):
        graph_parts = list(sequences_or_graphs)  # type: ignore[arg-type]
    else:
        seq_lists = [
            _truncate_item_ids(_as_int_list(seq), max_seq_len)  # type: ignore[arg-type]
            for seq in sequences_or_graphs
        ]
        max_seq = max(len(seq) for seq in seq_lists)
        graph_parts = []
        for seq in seq_lists:
            padded = seq + [PAD_IDX] * (max_seq - len(seq))
            graph_parts.append(build_session_graph(padded))

    alias_tensor, adj_tensor, items_tensor, mask_tensor = _pad_graph_batch(graph_parts)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return alias_tensor, adj_tensor, items_tensor, mask_tensor, targets_tensor
