"""GRU4Rec baseline: model, dataset, and Lightning module.

Training: ``uv run python -m src.main fit -c config/data/gru4rec_yoochoose.yaml -c config/model/gru4rec.yaml``
"""

from src.models.gru4rec.dataset import GRU4RecDataset, gru4rec_collate_fn
from src.models.gru4rec.model import GRU4Rec
from src.models.gru4rec.module import GRU4RecLitModule

__all__ = [
    "GRU4Rec",
    "GRU4RecDataset",
    "GRU4RecLitModule",
    "gru4rec_collate_fn",
]
