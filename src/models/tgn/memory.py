"""TGN memory module tweaks for large bipartite graphs."""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.nn import TGNMemory


class SafeTGNMemory(TGNMemory):
    """``TGNMemory`` without the eval-mode flush over all ``num_nodes``.

    PyG flushes the raw message store on ``train(False)`` via a single
    ``_update_memory(torch.arange(num_nodes))`` call, which can allocate
  GiBs on GPU for large session graphs. We reset the message store instead;
    ``TGNLitModule._warmup_for_eval`` rebuilds memory from train events.
    """

    def train(self, mode: bool = True):
        if self.training and not mode:
            self._reset_message_store()
        return nn.Module.train(self, mode)
