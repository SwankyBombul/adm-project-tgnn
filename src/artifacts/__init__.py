"""Read-only access to processed data artifacts (output of ``src.preprocessing``)."""

from src.artifacts.meta import gru4rec_vocab_size, load_meta, tgn_num_items, tgn_num_sessions
from src.artifacts.paths import ModelFormat, SplitName, split_events_path, split_examples_path
from src.artifacts.vocab import load_gru_item2idx, load_tgn_item2idx

__all__ = [
    "ModelFormat",
    "SplitName",
    "gru4rec_vocab_size",
    "load_gru_item2idx",
    "load_tgn_item2idx",
    "load_meta",
    "split_events_path",
    "split_examples_path",
    "tgn_num_items",
    "tgn_num_sessions",
]
