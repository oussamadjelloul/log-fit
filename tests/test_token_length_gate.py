"""Tests for src/token_length_gate.py — §1.6 token-length validation.

Strategy:
- Pure logic (threshold checks) tested via _build_summary_from_token_counts
  with synthetic token-count lists. No tokenizer / network dependency.
- The actual tokenization function tokenize_paragraph_token_counts is tested
  in a single integration test that runs only if transformers is importable
  AND the fast tokenizer for roberta-base is available locally.

Mapping to v1.3 spec Component 13 (gate-related tests):
- test_backbone_limit                 → L1, L2 backbone limits
- test_hard_fail_max_above_5x         → §1.6 HARD FAIL
- test_supervisor_flag_p95_above_limit → §1.6 SUPERVISOR_FLAG
- test_truncation_rate_correct        → §1.6 reporting
- test_empty_paragraphs_handled       → robustness
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from src.token_length_gate import (
    HARD_FAIL_MULTIPLIER,
    LONGFORMER_LIMIT,
    ROBERTA_LIMIT,
    _build_summary_from_token_counts,
    get_backbone_limit,
    run_gate,
)
from src.types import Paragraph, TokenLengthSummary


# ---------------------------------------------------------------------------
# Backbone-limit lookup
# ---------------------------------------------------------------------------


class TestBackboneLimit:
    def test_roberta_base(self):
        assert get_backbone_limit("roberta-base") == ROBERTA_LIMIT
        assert ROBERTA_LIMIT == 512

    def test_longformer(self):
        assert get_backbone_limit("allenai/longformer-base-4096") == LONGFORMER_LIMIT
        assert LONGFORMER_LIMIT == 4096

    def test_case_insensitive(self):
        assert get_backbone_limit("RoBERTa-Base") == ROBERTA_LIMIT
        assert get_backbone_limit("Longformer-X") == LONGFORMER_LIMIT

    def test_longformer_takes_precedence_over_roberta(self):
        """Longformer fine-tuned from RoBERTa may have both substrings in its
        name. Longformer rule must win to avoid using the 512 limit."""
        # Hypothetical: 'roberta-longformer' should resolve to Longformer
        assert get_backbone_limit("roberta-longformer-base") == LONGFORMER_LIMIT

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            get_backbone_limit("gpt-2")


# ---------------------------------------------------------------------------
# §1.6 threshold checks — pure logic, no tokenizer
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_all_under_limit_no_flag(self, capsys):
        # All token counts below RoBERTa limit (512)
        token_counts = [100, 200, 300, 400, 500]
        summary = _build_summary_from_token_counts(
            token_counts, "roberta-base", raise_on_hard_fail=True
        )
        captured = capsys.readouterr()
        assert "SUPERVISOR_FLAG" not in captured.out
        assert "HARD FAIL" not in captured.out
        assert summary.truncation_rate_at_backbone_limit == 0.0
        assert summary.distribution.max == 500

    def test_supervisor_flag_when_p95_above_limit(self, capsys):
        # p95 above 512 but max stays under 5×512 = 2560
        token_counts = [100, 200, 300, 400, 600, 700, 800, 900, 1000, 1200]
        # p95 of this distribution is ~1180, well above 512
        summary = _build_summary_from_token_counts(
            token_counts, "roberta-base", raise_on_hard_fail=True
        )
        captured = capsys.readouterr()
        assert "SUPERVISOR_FLAG" in captured.out
        assert "HARD FAIL" not in captured.out
        # Truncation rate: 6 of 10 paragraphs > 512 = 60%
        # (values 600, 700, 800, 900, 1000, 1200 are above 512)
        assert summary.truncation_rate_at_backbone_limit == 0.6

    def test_hard_fail_raises_when_max_above_5x_limit(self):
        # max = 3000 > 5 × 512 = 2560 → HARD FAIL
        token_counts = [100, 200, 300, 3000]
        with pytest.raises(ValueError, match="HARD FAIL"):
            _build_summary_from_token_counts(
                token_counts, "roberta-base", raise_on_hard_fail=True
            )

    def test_hard_fail_warn_only_does_not_raise(self, capsys):
        token_counts = [100, 200, 300, 3000]
        # raise_on_hard_fail=False — should print warning but not raise
        summary = _build_summary_from_token_counts(
            token_counts, "roberta-base", raise_on_hard_fail=False
        )
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "HARD FAIL" in captured.out
        assert summary.distribution.max == 3000

    def test_exact_5x_threshold_is_inclusive_passing(self):
        """max == 5 × limit should NOT trip HARD FAIL (strict inequality)."""
        token_counts = [100, 200, 5 * ROBERTA_LIMIT]
        # Should not raise
        summary = _build_summary_from_token_counts(
            token_counts, "roberta-base", raise_on_hard_fail=True
        )
        assert summary.distribution.max == 5 * ROBERTA_LIMIT

    def test_just_above_5x_threshold_raises(self):
        token_counts = [100, 200, 5 * ROBERTA_LIMIT + 1]
        with pytest.raises(ValueError, match="HARD FAIL"):
            _build_summary_from_token_counts(
                token_counts, "roberta-base", raise_on_hard_fail=True
            )


# ---------------------------------------------------------------------------
# Longformer thresholds
# ---------------------------------------------------------------------------


class TestLongformerThresholds:
    def test_longformer_limit_512_value_safe(self, capsys):
        # 512 tokens is safe for Longformer (limit 4096)
        token_counts = [100, 200, 512]
        summary = _build_summary_from_token_counts(
            token_counts, "allenai/longformer-base-4096", raise_on_hard_fail=True
        )
        captured = capsys.readouterr()
        assert "SUPERVISOR_FLAG" not in captured.out
        assert summary.truncation_rate_at_backbone_limit == 0.0

    def test_longformer_supervisor_flag_above_4096(self, capsys):
        # p95 well above 4096 but max under 5×4096 = 20480
        token_counts = [1000, 2000, 3000, 5000, 6000, 8000, 10000, 15000, 18000, 19000]
        summary = _build_summary_from_token_counts(
            token_counts, "allenai/longformer-base-4096", raise_on_hard_fail=True
        )
        captured = capsys.readouterr()
        assert "SUPERVISOR_FLAG" in captured.out

    def test_longformer_hard_fail_above_20480(self):
        # max = 25000 > 5 × 4096 = 20480 → HARD FAIL
        token_counts = [1000, 2000, 25000]
        with pytest.raises(ValueError, match="HARD FAIL"):
            _build_summary_from_token_counts(
                token_counts,
                "allenai/longformer-base-4096",
                raise_on_hard_fail=True,
            )


# ---------------------------------------------------------------------------
# Truncation rate
# ---------------------------------------------------------------------------


class TestTruncationRate:
    def test_zero_when_all_under_limit(self):
        summary = _build_summary_from_token_counts(
            [100, 200, 300], "roberta-base", raise_on_hard_fail=True
        )
        assert summary.truncation_rate_at_backbone_limit == 0.0

    def test_one_when_all_above_limit_below_hard_fail(self, capsys):
        # All above 512, below 2560 → SUPERVISOR_FLAG fires, no HARD FAIL
        token_counts = [600, 700, 800, 900]
        summary = _build_summary_from_token_counts(
            token_counts, "roberta-base", raise_on_hard_fail=True
        )
        assert summary.truncation_rate_at_backbone_limit == 1.0
        captured = capsys.readouterr()
        assert "SUPERVISOR_FLAG" in captured.out

    def test_half_when_half_above_limit(self):
        # 5 of 10 above 512 → truncation rate = 0.5
        token_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        # Quiet the SUPERVISOR_FLAG print noise by using raise_on_hard_fail=False
        # (HARD FAIL not actually triggered here, max=1000 < 2560)
        summary = _build_summary_from_token_counts(
            token_counts, "roberta-base", raise_on_hard_fail=True
        )
        assert summary.truncation_rate_at_backbone_limit == 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_paragraph_list_handled(self):
        summary = _build_summary_from_token_counts(
            [], "roberta-base", raise_on_hard_fail=True
        )
        assert summary.distribution.min == 0
        assert summary.distribution.max == 0
        assert summary.truncation_rate_at_backbone_limit == 0.0

    def test_single_paragraph_token_count(self):
        summary = _build_summary_from_token_counts(
            [256], "roberta-base", raise_on_hard_fail=True
        )
        assert summary.distribution.min == 256
        assert summary.distribution.max == 256
        assert summary.distribution.p50 == 256.0

    def test_summary_carries_tokenizer_name(self):
        summary = _build_summary_from_token_counts(
            [100], "roberta-base", raise_on_hard_fail=True
        )
        assert summary.tokenizer == "roberta-base"


# ---------------------------------------------------------------------------
# Integration: file-based gate (without real tokenizer)
# ---------------------------------------------------------------------------


class TestRunGateFileIO:
    """Tests run_gate's file IO path using monkeypatched tokenize fn."""

    def test_updates_existing_preparation_summary_json(
        self, tmp_path: Path, monkeypatch
    ):
        # Create a fake paragraphs.pkl
        paragraphs = [
            Paragraph(
                paragraph_id="p1",
                lines=["line 1", "line 2"],
                label=0,
            ),
            Paragraph(
                paragraph_id="p2",
                lines=["line 3"],
                label=1,
            ),
        ]
        pkl_path = tmp_path / "paragraphs.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(paragraphs, f)

        # Create a fake preparation_summary.json (matching prep module output)
        prep_summary = {
            "dataset": "hdfs",
            "window_seconds": None,
            "total_paragraphs": 2,
            "token_length_distribution": None,
        }
        from src.utils.io import save_json
        save_json(prep_summary, tmp_path / "preparation_summary.json")

        # Monkeypatch the tokenize function to avoid loading a real tokenizer
        import src.token_length_gate as mod
        monkeypatch.setattr(
            mod,
            "tokenize_paragraph_token_counts",
            lambda paras, backbone: [50, 100],  # synthetic, well under 512
        )

        summary = run_gate(
            paragraphs_pkl_path=pkl_path,
            backbone_name="roberta-base",
            update_preparation_summary=True,
            raise_on_hard_fail=True,
        )

        # Verify it returned a sensible summary
        assert isinstance(summary, TokenLengthSummary)
        assert summary.tokenizer == "roberta-base"
        assert summary.distribution.max == 100

        # Verify preparation_summary.json was updated
        from src.utils.io import load_json
        updated = load_json(tmp_path / "preparation_summary.json")
        assert updated["token_length_distribution"] is not None
        assert updated["token_length_distribution"]["tokenizer"] == "roberta-base"
        assert updated["token_length_distribution"]["distribution"]["max"] == 100

    def test_writes_standalone_when_no_existing_summary(
        self, tmp_path: Path, monkeypatch
    ):
        paragraphs = [Paragraph(paragraph_id="p1", lines=["x"], label=0)]
        pkl_path = tmp_path / "paragraphs.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(paragraphs, f)

        # NO preparation_summary.json in tmp_path

        import src.token_length_gate as mod
        monkeypatch.setattr(
            mod,
            "tokenize_paragraph_token_counts",
            lambda paras, backbone: [10],
        )

        run_gate(
            paragraphs_pkl_path=pkl_path,
            backbone_name="roberta-base",
            update_preparation_summary=True,
            raise_on_hard_fail=True,
        )

        # A standalone token_length_summary.json should exist
        standalone = tmp_path / "token_length_summary.json"
        assert standalone.exists()


# ---------------------------------------------------------------------------
# Real tokenizer integration (skipped unless transformers+tokenizer available)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_roberta_tokenizer_available():
    """Skip dependent tests if transformers isn't importable or the
    roberta-base tokenizer isn't locally cached."""
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
        return True
    except (ImportError, OSError):
        pytest.skip(
            "Real roberta-base tokenizer not available in this environment. "
            "Pre-download via "
            "`python -c 'from transformers import AutoTokenizer; "
            "AutoTokenizer.from_pretrained(\"roberta-base\")'`."
        )


class TestRealTokenizerIntegration:
    """One end-to-end check with the actual roberta-base tokenizer.

    Verifies that the spec Component 3.1 tokenization formula
    (`<s>` + lines + `</s>` per line) is implemented correctly.
    """

    def test_simple_paragraph_token_count(
        self, real_roberta_tokenizer_available
    ):
        from src.token_length_gate import tokenize_paragraph_token_counts

        # A paragraph with two short lines
        paragraphs = [
            Paragraph(
                paragraph_id="p1",
                lines=["hello world", "foo bar baz"],
                label=0,
            )
        ]
        counts = tokenize_paragraph_token_counts(paragraphs, "roberta-base")
        assert len(counts) == 1
        # Token count should be: 1 (<s>) + len("hello world") + 1 (</s>)
        # + len("foo bar baz") + 1 (</s>)
        # Approximate: each English word ≈ 1-2 tokens
        # Lower bound: 1 + 2 + 1 + 3 + 1 = 8; upper bound ~15
        assert 8 <= counts[0] <= 20

    def test_empty_lines_handled(self, real_roberta_tokenizer_available):
        from src.token_length_gate import tokenize_paragraph_token_counts

        paragraphs = [
            Paragraph(paragraph_id="p1", lines=[], label=0),
        ]
        counts = tokenize_paragraph_token_counts(paragraphs, "roberta-base")
        # Empty-lines paragraph counted as just <s> + </s>
        assert counts == [2]
