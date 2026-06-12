"""Read-only access to processed data artifacts (output of ``src.preprocessing``)."""

from src.artifacts.meta import gru4rec_vocab_size, load_meta
from src.artifacts.paths import ModelFormat, SplitName, split_examples_path
from src.artifacts.vocab import load_gru_item2idx

__all__ = [
    "ModelFormat",
    "SplitName",
    "gru4rec_vocab_size",
    "load_gru_item2idx",
    "load_meta",
    "split_examples_path",
]
