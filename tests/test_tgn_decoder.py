"""Tests for TGN link decoder."""

from __future__ import annotations

import torch

from src.models.tgn.decoder import LinkDecoder


def test_link_decoder_full_catalog_shape() -> None:
    decoder = LinkDecoder(channels=16)
    session_emb = torch.randn(3, 16)
    item_emb = torch.randn(7, 16)
    logits = decoder.score_all_items(session_emb, item_emb, chunk_size=4)
    assert logits.shape == (3, 7)
