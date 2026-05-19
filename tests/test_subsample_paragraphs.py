"""Tests for src/subsample_paragraphs.py — paper L21 stratified subsample.

Reference: logfit-repro-spec-v1.5.md Component 1.6 (v1.5.1).
"""

from __future__ import annotations

import pytest

from src.subsample_paragraphs import (
    ANOMALY_SEED_OFFSET,
    DEFAULT_MAX_ANOMALY,
    DEFAULT_MAX_NORMAL,
    DEFAULT_SEED,
    NORMAL_SEED_OFFSET,
    SubsampleSummary,
    subsample_paragraphs,
)
from src.types import Paragraph


def _make_paragraphs(n_normal: int, n_anomaly: int) -> list[Paragraph]:
    """Synthetic paragraph list. IDs: 'n_0' ... , 'a_0' ..."""
    ps = [
        Paragraph(paragraph_id=f"n_{i}", lines=["x"], label=0)
        for i in range(n_normal)
    ]
    ps.extend(
        Paragraph(paragraph_id=f"a_{i}", lines=["y"], label=1)
        for i in range(n_anomaly)
    )
    return ps


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_paper_l21_defaults(self):
        assert DEFAULT_MAX_NORMAL == 25_000
        assert DEFAULT_MAX_ANOMALY == 2_000
        assert DEFAULT_SEED == 42

    def test_seed_offsets_distinct(self):
        """Normal and anomaly streams must use distinct offsets so they
        decouple — same principle as splits.py D_NEW6."""
        assert NORMAL_SEED_OFFSET != ANOMALY_SEED_OFFSET


# ---------------------------------------------------------------------------
# Basic behavior — capping
# ---------------------------------------------------------------------------


class TestCapping:
    def test_both_below_cap_passthrough(self):
        """Input smaller than both caps → output is the full input."""
        ps = _make_paragraphs(100, 50)
        out, summary = subsample_paragraphs(
            ps, max_normal=1000, max_anomaly=1000
        )
        assert len(out) == 150
        assert summary.output_normal == 100
        assert summary.output_anomaly == 50
        assert summary.normal_was_capped is False
        assert summary.anomaly_was_capped is False

    def test_normal_above_cap_caps(self):
        ps = _make_paragraphs(500, 50)
        out, summary = subsample_paragraphs(
            ps, max_normal=100, max_anomaly=1000
        )
        normal_out = [p for p in out if p.label == 0]
        anomaly_out = [p for p in out if p.label == 1]
        assert len(normal_out) == 100
        assert len(anomaly_out) == 50    # all passed through
        assert summary.normal_was_capped is True
        assert summary.anomaly_was_capped is False

    def test_anomaly_above_cap_caps(self):
        ps = _make_paragraphs(50, 500)
        out, summary = subsample_paragraphs(
            ps, max_normal=1000, max_anomaly=100
        )
        normal_out = [p for p in out if p.label == 0]
        anomaly_out = [p for p in out if p.label == 1]
        assert len(normal_out) == 50
        assert len(anomaly_out) == 100
        assert summary.normal_was_capped is False
        assert summary.anomaly_was_capped is True

    def test_both_above_cap_caps_both(self):
        ps = _make_paragraphs(500, 300)
        out, summary = subsample_paragraphs(
            ps, max_normal=100, max_anomaly=50
        )
        normal_out = [p for p in out if p.label == 0]
        anomaly_out = [p for p in out if p.label == 1]
        assert len(normal_out) == 100
        assert len(anomaly_out) == 50
        assert summary.normal_was_capped is True
        assert summary.anomaly_was_capped is True

    def test_paper_l21_caps_default(self):
        """Default caps match paper L21 (25k + 2k)."""
        ps = _make_paragraphs(30_000, 5_000)
        out, summary = subsample_paragraphs(ps)
        normal_out = [p for p in out if p.label == 0]
        anomaly_out = [p for p in out if p.label == 1]
        assert len(normal_out) == 25_000
        assert len(anomaly_out) == 2_000

    def test_tb_subset_shape(self):
        """Realistic TB subset shape (77k normal + 45k anomaly) → 25k+2k."""
        ps = _make_paragraphs(77_347, 45_106)
        out, summary = subsample_paragraphs(ps)
        assert summary.output_normal == 25_000
        assert summary.output_anomaly == 2_000
        assert summary.normal_was_capped is True
        assert summary.anomaly_was_capped is True


# ---------------------------------------------------------------------------
# Stratification — labels of output match expectations
# ---------------------------------------------------------------------------


class TestStratification:
    def test_normal_output_all_label_0(self):
        ps = _make_paragraphs(200, 50)
        out, _ = subsample_paragraphs(ps, max_normal=50, max_anomaly=50)
        normal_out = [p for p in out if p.label == 0]
        assert all(p.paragraph_id.startswith("n_") for p in normal_out)

    def test_anomaly_output_all_label_1(self):
        ps = _make_paragraphs(50, 200)
        out, _ = subsample_paragraphs(ps, max_normal=50, max_anomaly=50)
        anomaly_out = [p for p in out if p.label == 1]
        assert all(p.paragraph_id.startswith("a_") for p in anomaly_out)


# ---------------------------------------------------------------------------
# Seed-offset decoupling (paper L21 D_NEW6.1 invariant)
# ---------------------------------------------------------------------------


class TestSeedOffsetDecoupling:
    """Changing one label's cap MUST NOT shift which paragraphs of the
    other label are kept. Achieved via NORMAL_SEED_OFFSET vs
    ANOMALY_SEED_OFFSET."""

    def test_normal_cap_change_does_not_affect_anomaly_output(self):
        ps = _make_paragraphs(500, 300)
        out_a, _ = subsample_paragraphs(
            ps, max_normal=100, max_anomaly=50, seed=42
        )
        out_b, _ = subsample_paragraphs(
            ps, max_normal=200, max_anomaly=50, seed=42   # normal cap changed
        )
        anomaly_a = {p.paragraph_id for p in out_a if p.label == 1}
        anomaly_b = {p.paragraph_id for p in out_b if p.label == 1}
        assert anomaly_a == anomaly_b, (
            "Normal cap change perturbed anomaly sample — streams not decoupled."
        )

    def test_anomaly_cap_change_does_not_affect_normal_output(self):
        ps = _make_paragraphs(500, 300)
        out_a, _ = subsample_paragraphs(
            ps, max_normal=100, max_anomaly=50, seed=42
        )
        out_b, _ = subsample_paragraphs(
            ps, max_normal=100, max_anomaly=200, seed=42   # anomaly cap changed
        )
        normal_a = {p.paragraph_id for p in out_a if p.label == 0}
        normal_b = {p.paragraph_id for p in out_b if p.label == 0}
        assert normal_a == normal_b


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_output(self):
        ps = _make_paragraphs(500, 300)
        out_a, _ = subsample_paragraphs(
            ps, max_normal=100, max_anomaly=50, seed=42
        )
        out_b, _ = subsample_paragraphs(
            ps, max_normal=100, max_anomaly=50, seed=42
        )
        ids_a = [p.paragraph_id for p in out_a]
        ids_b = [p.paragraph_id for p in out_b]
        assert ids_a == ids_b

    def test_different_seeds_different_output(self):
        ps = _make_paragraphs(500, 300)
        out_a, _ = subsample_paragraphs(
            ps, max_normal=100, max_anomaly=50, seed=42
        )
        out_b, _ = subsample_paragraphs(
            ps, max_normal=100, max_anomaly=50, seed=43
        )
        ids_a = {p.paragraph_id for p in out_a}
        ids_b = {p.paragraph_id for p in out_b}
        assert ids_a != ids_b


# ---------------------------------------------------------------------------
# Order preservation
# ---------------------------------------------------------------------------


class TestOrderPreservation:
    def test_output_preserves_input_order(self):
        """Retained paragraphs appear in output in their original input
        index order (matters for BGL/TB which are time-sorted)."""
        ps = _make_paragraphs(200, 100)
        # Build a lookup from id → input position
        input_position = {p.paragraph_id: i for i, p in enumerate(ps)}
        out, _ = subsample_paragraphs(
            ps, max_normal=50, max_anomaly=20, seed=42
        )
        out_positions = [input_position[p.paragraph_id] for p in out]
        assert out_positions == sorted(out_positions), (
            "Output not in input order — order preservation broken."
        )

    def test_passthrough_preserves_exact_input_order(self):
        """When no capping triggers, output IS the input (same order)."""
        ps = _make_paragraphs(50, 30)
        out, _ = subsample_paragraphs(
            ps, max_normal=1000, max_anomaly=1000, seed=42
        )
        assert [p.paragraph_id for p in out] == [p.paragraph_id for p in ps]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_input_empty_output(self):
        out, summary = subsample_paragraphs([], max_normal=100, max_anomaly=50)
        assert out == []
        assert summary.input_normal == 0
        assert summary.input_anomaly == 0
        assert summary.output_normal == 0
        assert summary.output_anomaly == 0

    def test_all_normal_input(self):
        ps = _make_paragraphs(200, 0)
        out, summary = subsample_paragraphs(
            ps, max_normal=50, max_anomaly=100
        )
        assert len([p for p in out if p.label == 0]) == 50
        assert len([p for p in out if p.label == 1]) == 0
        assert summary.normal_was_capped is True
        assert summary.anomaly_was_capped is False

    def test_all_anomaly_input(self):
        ps = _make_paragraphs(0, 200)
        out, summary = subsample_paragraphs(
            ps, max_normal=100, max_anomaly=50
        )
        assert len([p for p in out if p.label == 0]) == 0
        assert len([p for p in out if p.label == 1]) == 50

    def test_zero_normal_cap(self):
        ps = _make_paragraphs(100, 50)
        out, _ = subsample_paragraphs(ps, max_normal=0, max_anomaly=100)
        assert len([p for p in out if p.label == 0]) == 0
        assert len([p for p in out if p.label == 1]) == 50

    def test_zero_anomaly_cap(self):
        ps = _make_paragraphs(100, 50)
        out, _ = subsample_paragraphs(ps, max_normal=100, max_anomaly=0)
        assert len([p for p in out if p.label == 0]) == 100
        assert len([p for p in out if p.label == 1]) == 0

    def test_negative_normal_cap_raises(self):
        ps = _make_paragraphs(10, 10)
        with pytest.raises(ValueError, match="non-negative"):
            subsample_paragraphs(ps, max_normal=-1, max_anomaly=10)

    def test_negative_anomaly_cap_raises(self):
        ps = _make_paragraphs(10, 10)
        with pytest.raises(ValueError, match="non-negative"):
            subsample_paragraphs(ps, max_normal=10, max_anomaly=-1)

    def test_hdfs_subset_shape_no_op(self):
        """HDFS subset shape (14941 + 698) is below both caps → no-op."""
        ps = _make_paragraphs(14_941, 698)
        out, summary = subsample_paragraphs(ps)   # defaults: 25k+2k
        assert len(out) == 14_941 + 698
        assert summary.normal_was_capped is False
        assert summary.anomaly_was_capped is False

    def test_bgl_subset_shape_only_anomaly_capped(self):
        """BGL subset shape (24070 + 2555) → only anomaly capped to 2000."""
        ps = _make_paragraphs(24_070, 2_555)
        out, summary = subsample_paragraphs(ps)   # defaults: 25k+2k
        assert summary.output_normal == 24_070    # passthrough
        assert summary.output_anomaly == 2_000    # capped
        assert summary.normal_was_capped is False
        assert summary.anomaly_was_capped is True


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------


class TestSubsampleSummary:
    def test_summary_fields_populated(self):
        ps = _make_paragraphs(100, 50)
        _, summary = subsample_paragraphs(
            ps, max_normal=80, max_anomaly=30, seed=42
        )
        assert summary.input_normal == 100
        assert summary.input_anomaly == 50
        assert summary.output_normal == 80
        assert summary.output_anomaly == 30
        assert summary.normal_was_capped is True
        assert summary.anomaly_was_capped is True
        assert summary.seed == 42
        assert summary.max_normal == 80
        assert summary.max_anomaly == 30

    def test_summary_no_op_case(self):
        ps = _make_paragraphs(10, 5)
        _, summary = subsample_paragraphs(
            ps, max_normal=100, max_anomaly=100
        )
        assert summary.normal_was_capped is False
        assert summary.anomaly_was_capped is False
        assert summary.output_normal == 10
        assert summary.output_anomaly == 5
