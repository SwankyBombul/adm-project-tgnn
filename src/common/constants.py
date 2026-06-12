"""Cross-cutting constants shared by preprocessing and training code."""

# GRU4Rec / TAGNN index conventions (also documented in meta.json).
PAD_IDX = 0
# Known items use 1..n_items; UNK uses n_items + 1.
