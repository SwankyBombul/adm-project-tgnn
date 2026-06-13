"""TGN model: PyG memory + temporal attention + link decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader

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
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.num_sessions = num_sessions
        self.n_neighbors = n_neighbors
        self.msg_dim = msg_dim
        self._num_nodes = num_nodes(num_items, num_sessions)

        self.memory = TGNMemory(
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
        self._neighbor_loader = LastNeighborLoader(
            self._num_nodes, size=self.n_neighbors, device=self.device
        )
        self._replay = TGNReplayState()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _memory_time(self, t_sec: Tensor) -> Tensor:
        """PyG ``TGNMemory.last_update`` stores integer timestamps."""
        return t_sec.to(dtype=torch.long)

    def set_num_sessions(self, num_sessions: int) -> None:
        """Resize neighbor loader for a split with a different session count."""
        if num_sessions == self.num_sessions:
            return
        self.num_sessions = num_sessions
        self._num_nodes = num_nodes(self.num_items, num_sessions)
        self._neighbor_loader = LastNeighborLoader(
            self._num_nodes, size=self.n_neighbors, device=self.device
        )

    def reset_state(self) -> None:
        self.memory.reset_state()
        self._neighbor_loader.reset_state()
        self._replay = TGNReplayState()

    def detach_memory(self) -> None:
        self.memory.detach()

    def _global_endpoints(
        self,
        session_idx: Tensor,
        item_idx_tgn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return edge_endpoints(session_idx, item_idx_tgn, self.num_items)

    def _compute_embeddings(self, n_id: Tensor) -> Tensor:
        all_id, edge_index, e_id = self._neighbor_loader(n_id)
        z, last_update = self.memory(all_id)
        if edge_index.numel() == 0:
            out = self.mem_to_emb(z)
        else:
            msg = torch.zeros(
                edge_index.size(1), self.msg_dim, device=self.device, dtype=z.dtype
            )
            assoc = self.memory._assoc
            src_local = assoc[edge_index[0]]
            t = last_update[src_local].to(dtype=z.dtype)
            out = self.gnn(z, last_update, edge_index, t, msg)
        id_to_pos = {int(node.item()): pos for pos, node in enumerate(all_id)}
        positions = torch.tensor(
            [id_to_pos[int(node.item())] for node in n_id],
            device=n_id.device,
            dtype=torch.long,
        )
        return out[positions]

    def _embed_nodes(self, node_ids: Tensor) -> Tensor:
        unique_ids, inverse = node_ids.unique(return_inverse=True)
        emb = self._compute_embeddings(unique_ids)
        return emb[inverse]

    def _item_embeddings(self) -> Tensor:
        item_ids = item_global_ids(self.num_items, device=self.device)
        return self._compute_embeddings(item_ids)

    def _update_from_tensors(
        self,
        session_idx: Tensor,
        item_idx_tgn: Tensor,
        t_sec: Tensor,
        msg: Tensor,
    ) -> None:
        src, dst = self._global_endpoints(session_idx, item_idx_tgn)
        t = self._memory_time(t_sec)
        self.memory.update_state(src, dst, t, msg)
        self._neighbor_loader.insert(src, dst)

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
        return session_global_id(session_idx, self.num_items)

    def _session_embeddings(self, session_idx: Tensor) -> Tensor:
        return self._embed_nodes(self._session_global(session_idx))

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

    def forward_ce_examples(
        self,
        batch: TGNExampleBatch,
        events: TGNEventTensors,
    ) -> tuple[Tensor, Tensor]:
        """CE path: replay prefixes, score full catalog, then commit target events."""
        logits_list: list[Tensor] = []
        targets = batch.target_item_idx_tgn
        for i in range(targets.size(0)):
            prefix_end = int(batch.prefix_last_event_id[i].item())
            self.replay_events_up_to(events, prefix_end)
            session_emb = self._session_embeddings(batch.session_idx[i : i + 1])
            item_emb = self._item_embeddings()
            logits_list.append(self.decoder.score_all_items(session_emb, item_emb))
            target_eid = int(batch.target_event_id[i].item())
            self.replay_events_up_to(events, target_eid)
        return torch.cat(logits_list, dim=0), targets

    def forward_eval_batch(
        self,
        batch: TGNExampleBatch,
        events: TGNEventTensors,
    ) -> tuple[Tensor, Tensor]:
        """Evaluation: replay prefixes only (no target-event memory updates)."""
        logits_list: list[Tensor] = []
        targets = batch.target_item_idx_tgn
        with torch.no_grad():
            for i in range(targets.size(0)):
                prefix_end = int(batch.prefix_last_event_id[i].item())
                self.replay_events_up_to(events, prefix_end)
                session_emb = self._session_embeddings(batch.session_idx[i : i + 1])
                item_emb = self._item_embeddings()
                logits_list.append(self.decoder.score_all_items(session_emb, item_emb))
        return torch.cat(logits_list, dim=0), targets
