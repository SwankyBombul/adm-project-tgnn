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


def test_link_decoder_score_candidates_matches_forward() -> None:
    decoder = LinkDecoder(channels=16)
    session_emb = torch.randn(2, 16)
    item_emb = torch.randn(2, 4, 16)
    scores = decoder.score_candidates(session_emb, item_emb)
    assert scores.shape == (2, 4)
    for row in range(2):
        for col in range(4):
            expected = decoder(session_emb[row], item_emb[row, col])
            assert torch.allclose(scores[row, col], expected)
