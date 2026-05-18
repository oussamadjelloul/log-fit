"""Tests for src/select_backbone.py — L2 quantile heuristic."""

from __future__ import annotations

import pytest

from src.select_backbone import (
    LONGFORMER_BACKBONE,
    ROBERTA_BACKBONE,
    WORD_QUANTILE_THRESHOLD,
    backbone_token_limit,
    select_backbone,
)
from src.types import LengthDistribution


class TestSelectBackbone:
    """[L2] q_0.8(words) <= 512 → RoBERTa, else Longformer."""

    def test_threshold_constant_matches_spec(self):
        assert WORD_QUANTILE_THRESHOLD == 512

    def test_below_threshold_picks_roberta(self):
        """p80 = 165 (real HDFS subset) -> RoBERTa."""
        dist = LengthDistribution(
            min=9, max=2857, p50=165.0, p80=165.0, p95=174.0, p99=192.0
        )
        assert select_backbone(dist) == ROBERTA_BACKBONE

    def test_above_threshold_picks_longformer(self):
        """p80 = 9764 (real BGL subset) -> Longformer."""
        dist = LengthDistribution(
            min=30, max=16005, p50=2388.0, p80=9764.4, p95=11480.6, p99=11953.8
        )
        assert select_backbone(dist) == LONGFORMER_BACKBONE

    def test_exactly_at_threshold_picks_roberta(self):
        """p80 == 512 should stay with RoBERTa (boundary is inclusive)."""
        dist = LengthDistribution(
            min=10, max=1024, p50=400.0, p80=512.0, p95=600.0, p99=800.0
        )
        assert select_backbone(dist) == ROBERTA_BACKBONE

    def test_one_over_threshold_picks_longformer(self):
        dist = LengthDistribution(
            min=10, max=1024, p50=400.0, p80=513.0, p95=600.0, p99=800.0
        )
        assert select_backbone(dist) == LONGFORMER_BACKBONE

    def test_custom_threshold(self):
        dist = LengthDistribution(
            min=10, max=2000, p50=600.0, p80=800.0, p95=1200.0, p99=1500.0
        )
        # With default threshold 512: p80=800 > 512 -> Longformer
        assert select_backbone(dist) == LONGFORMER_BACKBONE
        # With custom threshold 1000: p80=800 <= 1000 -> RoBERTa
        assert select_backbone(dist, quantile_threshold=1000) == ROBERTA_BACKBONE


class TestBackboneTokenLimit:
    def test_roberta_limit_is_512(self):
        assert backbone_token_limit(ROBERTA_BACKBONE) == 512

    def test_longformer_limit_is_4096(self):
        assert backbone_token_limit(LONGFORMER_BACKBONE) == 4096

    def test_case_insensitive_matching(self):
        assert backbone_token_limit("RoBERTa-base") == 512
        assert backbone_token_limit("ALLENAI/LONGFORMER-BASE-4096") == 4096

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            backbone_token_limit("gpt-neo")

    def test_error_includes_expected_options(self):
        with pytest.raises(ValueError) as exc_info:
            backbone_token_limit("bert-base-uncased")
        msg = str(exc_info.value)
        assert "roberta-base" in msg
        assert "longformer" in msg.lower()
