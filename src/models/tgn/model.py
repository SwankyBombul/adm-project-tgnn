"""TGN model: PyG memory + temporal attention + link decoder."""

from __future__ import annotations

from dataclasses import dataclass

from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader

from src.models.tgn.memory import SafeTGNMemory

from src.models.tgn.dataset import TGNEventTensors, TGNExampleBatch
from src.models.tgn.decoder import LinkDecoder
from src.models.tgn.embedding import GraphAttentionEmbedding
from src.models.tgn.node_ids import edge_endpoints, item_global_ids, num_nodes, session_global_id
from src.models.tgn.temporal_batch import sample_negative_items


@dataclass
class TGNReplayState:
    last_event_id: int = -1


class TGNModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_sessions: int,
        *,
        memory_dim: int = 172,
        time_dim: int = 100,
        embedding_dim: int = 100,
        msg_dim: int = 4,
        n_neighbors: int = 10,
        item_embed_chunk_size: int = 512,
        fast_eval_item_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.num_sessions = num_sessions
        self.n_neighbors = n_neighbors
        self.msg_dim = msg_dim
        self.item_embed_chunk_size = item_embed_chunk_size
        self.fast_eval_item_chunk_size = fast_eval_item_chunk_size
        self._num_nodes = num_nodes(num_items, num_sessions)

        self.memory = SafeTGNMemory(
            self._num_nodes,
            msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=msg_dim,
            time_enc=self.memory.time_enc,
        )
        self.mem_to_emb = nn.Linear(memory_dim, embedding_dim)
        self.decoder = LinkDecoder(embedding_dim)
        self._neighbor_loader: LastNeighborLoader | None = None
        self._replay = TGNReplayState()
        self._session_offset: int = 0
        self._edge_t = torch.zeros(0, dtype=torch.long)
        self._edge_msg = torch.zeros(0, msg_dim)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _neighbor_loader_for_device(self) -> LastNeighborLoader:
        """(Re)build ``LastNeighborLoader`` when the module moves across devices."""
        dev = self.device
        loader = self._neighbor_loader
        if loader is None or loader.neighbors.device != dev:
            self._neighbor_loader = LastNeighborLoader(
                self._num_nodes, size=self.n_neighbors, device=dev
            )
        return self._neighbor_loader

    def to(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        out = super().to(*args, **kwargs)
        self._neighbor_loader = None
        return out

    def _memory_time(self, t_sec: Tensor) -> Tensor:
        """PyG ``TGNMemory.last_update`` stores integer timestamps."""
        return t_sec.to(dtype=torch.long)

    def set_num_sessions(self, num_sessions: int) -> None:
        """Resize neighbor loader for a split with a different session count."""
        if num_sessions == self.num_sessions:
            return
        self.num_sessions = num_sessions
        self._num_nodes = num_nodes(self.num_items, num_sessions)
        self._neighbor_loader = None

    def set_session_offset(self, offset: int) -> None:
        """Map per-split local ``session_idx`` to disjoint global session nodes."""
        self._session_offset = int(offset)

    def reset_state(self) -> None:
        self.memory.reset_state()
        self._neighbor_loader_for_device().reset_state()
        self._replay = TGNReplayState()
        dev = self.device
        self._edge_t = torch.zeros(0, dtype=torch.long, device=dev)
        self._edge_msg = torch.zeros(0, self.msg_dim, device=dev)

    def _append_edge_store(self, t_sec: Tensor, msg: Tensor) -> None:
        """Store per-interaction attrs indexed by ``LastNeighborLoader`` edge ids."""
        t = self._memory_time(t_sec)
        self._edge_t = torch.cat([self._edge_t, t])
        self._edge_msg = torch.cat([self._edge_msg, msg])

    def _check_batch_indices(
        self,
        session_idx: Tensor,
        item_idx_tgn: Tensor,
    ) -> None:
        if session_idx.numel():
            max_global = int(session_idx.max().item()) + self._session_offset
            if max_global >= self.num_sessions:
                raise ValueError(
                    f"global session id {max_global} >= num_sessions {self.num_sessions}"
                )
        if item_idx_tgn.numel() and int(item_idx_tgn.max().item()) >= self.num_items:
            raise ValueError(
                f"item_idx_tgn max {int(item_idx_tgn.max().item())} >= num_items "
                f"{self.num_items}"
            )

    def detach_memory(self) -> None:
        self.memory.detach()

    def _global_endpoints(
        self,
        session_idx: Tensor,
        item_idx_tgn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return edge_endpoints(
            session_idx + self._session_offset,
            item_idx_tgn,
            self.num_items,
        )

    def _compute_embeddings(self, n_id: Tensor, *, memory_only: bool = False) -> Tensor:
        if memory_only:
            z, _ = self.memory(n_id)
            return self.mem_to_emb(z)
        neighbor_loader = self._neighbor_loader_for_device()
        all_id, edge_index, e_id = neighbor_loader(n_id)
        assoc = torch.empty(self._num_nodes, dtype=torch.long, device=self.device)
        assoc[all_id] = torch.arange(all_id.size(0), device=self.device)
        z, last_update = self.memory(all_id)
        if edge_index.numel() == 0:
            out = self.mem_to_emb(z)
        else:
            edge_t = self._edge_t[e_id].to(dtype=z.dtype)
            edge_msg = self._edge_msg[e_id]
            out = self.gnn(z, last_update, edge_index, edge_t, edge_msg)
        return out[assoc[n_id]]

    def _embed_nodes(self, node_ids: Tensor) -> Tensor:
        unique_ids, inverse = node_ids.unique(return_inverse=True)
        emb = self._compute_embeddings(unique_ids)
        return emb[inverse]

    def _item_embeddings(self) -> Tensor:
        item_ids = item_global_ids(self.num_items, device=self.device)
        return self._compute_embeddings(item_ids)

    def _score_full_catalog(
        self,
        session_emb: Tensor,
        *,
        memory_only_items: bool = False,
        item_chunk_size: int | None = None,
    ) -> Tensor:
        """Full-catalog logits without embedding all items in one forward pass."""
        batch_size = session_emb.size(0)
        logits = session_emb.new_empty(batch_size, self.num_items)
        item_ids = item_global_ids(self.num_items, device=self.device)
        chunk_size = item_chunk_size or self.item_embed_chunk_size
        for start in range(0, self.num_items, chunk_size):
            end = min(self.num_items, start + chunk_size)
            chunk_emb = self._compute_embeddings(
                item_ids[start:end],
                memory_only=memory_only_items,
            )
            logits[:, start:end] = self.decoder.score_all_items(
                session_emb,
                chunk_emb,
                chunk_size=chunk_emb.size(0),
            )
        return logits

    def _iter_eval_prefix_groups(self, batch: TGNExampleBatch) -> Iterator[tuple[int, Tensor]]:
        """Yield ``(prefix_last_event_id, row_indices)`` sorted for incremental replay."""
        n = batch.target_item_idx_tgn.size(0)
        order = batch.prefix_last_event_id.argsort()
        prefixes = batch.prefix_last_event_id[order]
        pos = 0
        while pos < n:
            prefix_end = int(prefixes[pos].item())
            end = pos + 1
            while end < n and int(prefixes[end].item()) == prefix_end:
                end += 1
            yield prefix_end, order[pos:end]
            pos = end

    def _embed_candidate_items(
        self,
        candidate_ids: Tensor,
        *,
        memory_only: bool,
    ) -> Tensor:
        """Embed candidate item indices with shape ``(batch, num_candidates, dim)``."""
        flat = candidate_ids.reshape(-1)
        unique_items, inverse = flat.unique(return_inverse=True)
        emb = self._compute_embeddings(unique_items, memory_only=memory_only)
        return emb[inverse].view(*candidate_ids.shape, emb.size(-1))

    def reset_replay_cursor(self) -> None:
        """Reset incremental replay position (e.g. when switching to val events)."""
        self._replay = TGNReplayState()

    def _update_from_tensors(
        self,
        session_idx: Tensor,
        item_idx_tgn: Tensor,
        t_sec: Tensor,
        msg: Tensor,
    ) -> None:
        self._check_batch_indices(session_idx, item_idx_tgn)
        src, dst = self._global_endpoints(session_idx, item_idx_tgn)
        t = self._memory_time(t_sec)
        self.memory.update_state(src, dst, t, msg)
        self._append_edge_store(t_sec, msg)
        self._neighbor_loader_for_device().insert(src, dst)

    def replay_events_up_to(
        self,
        events: TGNEventTensors,
        end_event_id: int,
    ) -> None:
        """Replay events with ``event_id`` in ``(last, end]`` without gradients."""
        if end_event_id <= self._replay.last_event_id:
            return
        start = self._replay.last_event_id + 1
        chunk = events.slice_events(start, end_event_id)
        if chunk["session_idx"].numel() == 0:
            self._replay.last_event_id = end_event_id
            return
        with torch.no_grad():
            self._update_from_tensors(
                chunk["session_idx"],
                chunk["item_idx_tgn"],
                chunk["t_sec"],
                chunk["msg"],
            )
        self._replay.last_event_id = end_event_id

    def _session_global(self, session_idx: Tensor) -> Tensor:
        return session_global_id(session_idx + self._session_offset, self.num_items)

    def score_pos_neg(
        self,
        session_idx: Tensor,
        item_idx_tgn: Tensor,
        t_sec: Tensor,
        msg: Tensor,
        *,
        num_negatives: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """BCE training step on an event batch (memory updated after scoring)."""
        src, dst = self._global_endpoints(session_idx, item_idx_tgn)
        neg_items = sample_negative_items(item_idx_tgn, self.num_items, num_negatives)
        _, neg_dst = self._global_endpoints(session_idx, neg_items)
        z_src = self._embed_nodes(src)
        z_dst = self._embed_nodes(dst)
        pos_logits = self.decoder(z_src, z_dst)

        if num_negatives == 1:
            neg_logits = self.decoder(z_src, self._embed_nodes(neg_dst))
        else:
            raise NotImplementedError("num_negatives > 1 not implemented")

        self._update_from_tensors(session_idx, item_idx_tgn, t_sec, msg)
        return pos_logits, neg_logits

    def forward_eval_batch(
        self,
        batch: TGNExampleBatch,
        events: TGNEventTensors,
        *,
        fast_eval: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Evaluation: replay prefixes only (no target-event memory updates)."""
        targets = batch.target_item_idx_tgn
        n = targets.size(0)
        logits = torch.empty(
            n,
            self.num_items,
            device=targets.device,
            dtype=torch.float32,
        )
        item_chunk = self.fast_eval_item_chunk_size if fast_eval else self.item_embed_chunk_size

        with torch.no_grad():
            for prefix_end, idx in self._iter_eval_prefix_groups(batch):
                self.replay_events_up_to(events, prefix_end)
                session_emb = self._compute_embeddings(
                    self._session_global(batch.session_idx[idx]),
                    memory_only=fast_eval,
                )
                logits[idx] = self._score_full_catalog(
                    session_emb,
                    memory_only_items=fast_eval,
                    item_chunk_size=item_chunk,
                )

        return logits, targets

    def forward_eval_sampled(
        self,
        batch: TGNExampleBatch,
        events: TGNEventTensors,
        candidate_ids: Tensor,
        *,
        fast_eval: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Sampled evaluation: score only ``candidate_ids`` per row after prefix replay."""
        if candidate_ids.size(0) != batch.target_item_idx_tgn.size(0):
            raise ValueError("candidate_ids batch size must match the example batch")
        targets = batch.target_item_idx_tgn
        n = targets.size(0)
        num_candidates = candidate_ids.size(1)
        scores = torch.empty(
            n,
            num_candidates,
            device=targets.device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            for prefix_end, idx in self._iter_eval_prefix_groups(batch):
                self.replay_events_up_to(events, prefix_end)
                group_candidates = candidate_ids[idx]
                session_emb = self._compute_embeddings(
                    self._session_global(batch.session_idx[idx]),
                    memory_only=fast_eval,
                )
                item_emb = self._embed_candidate_items(
                    group_candidates,
                    memory_only=fast_eval,
                )
                scores[idx] = self.decoder.score_candidates(session_emb, item_emb)

        return scores, targets
