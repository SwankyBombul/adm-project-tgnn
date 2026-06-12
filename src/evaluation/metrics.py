"""Session recommendation ranking metrics.

Contract (to be finalized with the team):
  - ``scores``: (batch, num_items) — higher is better
  - ``targets``: (batch,) — ground-truth item indices in the same index space as scores
  - Full-catalog ranking unless a separate candidate set is introduced later
"""

from __future__ import annotations

import torch
from torch import Tensor


def recall_at_k(scores: Tensor, targets: Tensor, k: int) -> float:
    """Recall@K / Hit Rate@K — implementation pending team agreement on details."""
    raise NotImplementedError("Recall@K will be implemented in the next training iteration.")

    # Planned shape checks:
    # if scores.ndim != 2 or targets.ndim != 1:
    #     raise ValueError(...)
    # topk = scores.topk(k, dim=1).indices
    # hits = (topk == targets.unsqueeze(1)).any(dim=1)
    # return hits.float().mean().item()


def mrr_at_k(scores: Tensor, targets: Tensor, k: int) -> float:
    """MRR@K — implementation pending team agreement on details."""
    raise NotImplementedError("MRR@K will be implemented in the next training iteration.")
