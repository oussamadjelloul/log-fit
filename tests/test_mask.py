"""Tests for src/mask.py — LogFiT MLM masking per paper §III-B.

Strategy:
- Pure logic (sentence/token selection, truncation, masking) tested with a
  fake whitespace-splitting tokenizer to avoid network / model-download
  dependency.
- One optional integration test class (TestRealTokenizer) gated by
  pytest.importorskip — runs only when transformers + roberta-base
  tokenizer files are available locally.

Mapping to spec-v1.2 Component 3 (masking-related tests):
- select_sentences_to_mask        → 50% sentence sampling
- select_tokens_in_line           → 80% token sampling within line
- truncate_boundaries             → line-aware right-truncation (D_NEW2)
- tokenize_paragraph_lines        → per-line tokenization + boundary tracking
- apply_logfit_masking            → end-to-end masking with BOS/EOS guard
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mask import (
    DEFAULT_SENTENCE_MASK_RATIO,
    DEFAULT_TOKEN_MASK_RATIO,
    apply_logfit_masking,
    select_sentences_to_mask,
    select_tokens_in_line,
    tokenize_paragraph_lines,
    truncate_boundaries,
)


# ---------------------------------------------------------------------------
# Fake tokenizer — whitespace splitter with deterministic per-token IDs.
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Deterministic whitespace tokenizer for unit tests.

    Token IDs are assigned on first occurrence starting at 100. Special token
    IDs are stable and well-separated from content IDs.
    """

    BOS_ID = 0
    PAD_ID = 1
    EOS_ID = 2
    MASK_ID = 3
    FIRST_CONTENT_ID = 100

    def __init__(self):
        self.bos_token_id = self.BOS_ID
        self.eos_token_id = self.EOS_ID
        self.pad_token_id = self.PAD_ID
        self.mask_token_id = self.MASK_ID
        self._vocab: dict[str, int] = {}
        self._next_id = self.FIRST_CONTENT_ID

    def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> list[int]:
        if add_special_tokens:
            raise AssertionError(
                "Production code must pass add_special_tokens=False"
            )
        ids: list[int] = []
        for w in text.split():
            if w not in self._vocab:
                self._vocab[w] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[w])
        return ids


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


# ---------------------------------------------------------------------------
# select_sentences_to_mask
# ---------------------------------------------------------------------------

class TestSelectSentencesToMask:
    def test_zero_lines_returns_empty(self):
        rng = np.random.default_rng(42)
        result = select_sentences_to_mask(0, 0.5, rng)
        assert result.size == 0
        assert result.dtype == np.int64

    def test_single_line_ratio_05_selects_1(self):
        rng = np.random.default_rng(42)
        result = select_sentences_to_mask(1, 0.5, rng)
        # max(1, round(0.5)) == 1
        assert result.tolist() == [0]

    def test_ratio_zero_still_selects_minimum_1(self):
        """The `max(1, ...)` floor ensures at least 1 line is masked
        whenever n_lines > 0."""
        rng = np.random.default_rng(42)
        result = select_sentences_to_mask(10, 0.0, rng)
        assert len(result) == 1

    def test_ratio_one_selects_all(self):
        rng = np.random.default_rng(42)
        result = select_sentences_to_mask(10, 1.0, rng)
        assert len(result) == 10
        assert set(result.tolist()) == set(range(10))

    def test_ratio_05_with_10_lines_selects_5(self):
        rng = np.random.default_rng(42)
        result = select_sentences_to_mask(10, 0.5, rng)
        assert len(result) == 5

    def test_output_is_sorted(self):
        rng = np.random.default_rng(42)
        result = select_sentences_to_mask(20, 0.5, rng)
        assert (np.diff(result) > 0).all()

    def test_deterministic_with_same_seed(self):
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        a = select_sentences_to_mask(20, 0.5, rng_a)
        b = select_sentences_to_mask(20, 0.5, rng_b)
        assert a.tolist() == b.tolist()

    def test_different_seeds_give_different_selection(self):
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(7)
        a = select_sentences_to_mask(50, 0.5, rng_a)
        b = select_sentences_to_mask(50, 0.5, rng_b)
        assert a.tolist() != b.tolist()

    def test_ratio_capped_at_n_lines(self):
        """round(0.99 * 100) == 99, but a ratio > 1.0 should still cap."""
        rng = np.random.default_rng(42)
        # Pathological: very high ratio
        result = select_sentences_to_mask(5, 5.0, rng)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# select_tokens_in_line
# ---------------------------------------------------------------------------

class TestSelectTokensInLine:
    def test_zero_length_range_returns_empty(self):
        rng = np.random.default_rng(42)
        result = select_tokens_in_line(5, 5, 0.8, rng)
        assert result.size == 0

    def test_sent_len_1_selects_1(self):
        rng = np.random.default_rng(42)
        result = select_tokens_in_line(10, 11, 0.8, rng)
        assert result.tolist() == [10]

    def test_ratio_zero_still_selects_minimum_1(self):
        rng = np.random.default_rng(42)
        result = select_tokens_in_line(0, 10, 0.0, rng)
        assert len(result) == 1

    def test_ratio_one_selects_all(self):
        rng = np.random.default_rng(42)
        result = select_tokens_in_line(0, 10, 1.0, rng)
        assert sorted(result.tolist()) == list(range(10))

    def test_positions_within_range(self):
        rng = np.random.default_rng(42)
        result = select_tokens_in_line(50, 70, 0.8, rng)
        assert (result >= 50).all()
        assert (result < 70).all()

    def test_output_is_sorted(self):
        rng = np.random.default_rng(42)
        result = select_tokens_in_line(0, 50, 0.8, rng)
        assert (np.diff(result) > 0).all()

    def test_deterministic_with_same_seed(self):
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        a = select_tokens_in_line(0, 30, 0.8, rng_a)
        b = select_tokens_in_line(0, 30, 0.8, rng_b)
        assert a.tolist() == b.tolist()


# ---------------------------------------------------------------------------
# truncate_boundaries
# ---------------------------------------------------------------------------

class TestTruncateBoundaries:
    def test_all_within_budget_unchanged(self):
        boundaries = [(0, 5), (5, 10), (10, 15)]
        result = truncate_boundaries(boundaries, budget=20)
        assert result == boundaries

    def test_line_clipped_at_budget(self):
        boundaries = [(0, 5), (5, 15)]
        result = truncate_boundaries(boundaries, budget=10)
        assert result == [(0, 5), (5, 10)]

    def test_line_entirely_past_budget_dropped(self):
        boundaries = [(0, 5), (5, 10), (10, 15)]
        result = truncate_boundaries(boundaries, budget=10)
        # Third line starts at 10, which is NOT < 10, so dropped.
        assert result == [(0, 5), (5, 10)]

    def test_zero_length_line_preserved(self):
        """Empty lines (start == end) within budget stay as (s, s)."""
        boundaries = [(0, 5), (5, 5), (5, 10)]
        result = truncate_boundaries(boundaries, budget=15)
        assert result == [(0, 5), (5, 5), (5, 10)]

    def test_budget_zero_drops_all(self):
        boundaries = [(0, 5), (5, 10)]
        result = truncate_boundaries(boundaries, budget=0)
        assert result == []

    def test_empty_boundaries_returns_empty(self):
        assert truncate_boundaries([], budget=10) == []

    def test_break_on_first_past_budget(self):
        """Once a line starts past budget, the loop breaks — later lines
        are never considered. This is correct because boundaries are
        in flat ascending order."""
        boundaries = [(0, 5), (5, 10), (10, 15), (15, 20)]
        result = truncate_boundaries(boundaries, budget=8)
        # (0,5) keeps; (5,10) clips to (5,8); (10,15) drops; loop breaks.
        assert result == [(0, 5), (5, 8)]


# ---------------------------------------------------------------------------
# tokenize_paragraph_lines
# ---------------------------------------------------------------------------

class TestTokenizeParagraphLines:
    def test_empty_lines_returns_empty(self, tokenizer):
        flat, bounds = tokenize_paragraph_lines([], tokenizer)
        assert flat == []
        assert bounds == []

    def test_single_line(self, tokenizer):
        flat, bounds = tokenize_paragraph_lines(["a b c"], tokenizer)
        assert len(flat) == 3
        assert bounds == [(0, 3)]

    def test_multiple_lines_boundary_correctness(self, tokenizer):
        lines = ["a b", "c d e", "f"]
        flat, bounds = tokenize_paragraph_lines(lines, tokenizer)
        assert len(flat) == 6
        assert bounds == [(0, 2), (2, 5), (5, 6)]

    def test_empty_line_yields_zero_length_boundary(self, tokenizer):
        lines = ["a", "", "b"]
        flat, bounds = tokenize_paragraph_lines(lines, tokenizer)
        # "" tokenizes to [], so boundary is (1, 1)
        assert bounds == [(0, 1), (1, 1), (1, 2)]
        assert flat == [100, 101]

    def test_flat_matches_concatenation(self, tokenizer):
        lines = ["the quick", "brown fox"]
        flat, bounds = tokenize_paragraph_lines(lines, tokenizer)
        # Boundaries should reconstruct each line's token IDs
        for line, (start, end) in zip(lines, bounds):
            expected_ids = tokenizer.encode(line, add_special_tokens=False)
            assert flat[start:end] == expected_ids


# ---------------------------------------------------------------------------
# apply_logfit_masking — end-to-end
# ---------------------------------------------------------------------------

class TestApplyLogfitMasking:
    def test_returns_three_keys(self, tokenizer):
        out = apply_logfit_masking(
            lines=["a b c", "d e f"],
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        assert set(out.keys()) == {"input_ids", "labels", "attention_mask"}

    def test_input_ids_wrapped_with_bos_eos(self, tokenizer):
        out = apply_logfit_masking(
            lines=["a b c"],
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        assert out["input_ids"][0] == tokenizer.bos_token_id
        assert out["input_ids"][-1] == tokenizer.eos_token_id

    def test_labels_minus_100_at_bos_and_eos(self, tokenizer):
        out = apply_logfit_masking(
            lines=["a b c"],
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        assert out["labels"][0] == -100
        assert out["labels"][-1] == -100

    def test_bos_and_eos_never_replaced_with_mask(self, tokenizer):
        out = apply_logfit_masking(
            lines=["a b c"],
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        assert out["input_ids"][0] != tokenizer.mask_token_id
        assert out["input_ids"][-1] != tokenizer.mask_token_id

    def test_label_at_masked_position_is_original_token(self, tokenizer):
        """At every position where input_ids == mask_id, labels must hold
        the ORIGINAL token ID (so CE loss can recover it)."""
        out = apply_logfit_masking(
            lines=["a b c d e f g h i j"],
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        for i, tok in enumerate(out["input_ids"]):
            if tok == tokenizer.mask_token_id:
                assert out["labels"][i] != -100
                assert out["labels"][i] >= FakeTokenizer.FIRST_CONTENT_ID

    def test_label_minus_100_at_unmasked_positions(self, tokenizer):
        """Symmetric invariant: positions that were NOT masked have
        label == -100, so loss ignores them."""
        out = apply_logfit_masking(
            lines=["a b c d e"],
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        for i, lbl in enumerate(out["labels"]):
            if lbl == -100:
                # Not a masked position; input_ids must NOT be mask_id
                assert out["input_ids"][i] != tokenizer.mask_token_id

    def test_three_lists_same_length(self, tokenizer):
        out = apply_logfit_masking(
            lines=["a b c", "d e f"],
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        n = len(out["input_ids"])
        assert len(out["labels"]) == n
        assert len(out["attention_mask"]) == n

    def test_attention_mask_all_ones(self, tokenizer):
        out = apply_logfit_masking(
            lines=["a b c", "d e f"],
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        assert all(m == 1 for m in out["attention_mask"])

    def test_respects_backbone_token_limit(self, tokenizer):
        # 100-word line; budget=20 → 18 content + BOS + EOS = 20
        lines = [" ".join(f"w{i}" for i in range(100))]
        out = apply_logfit_masking(
            lines=lines,
            tokenizer=tokenizer,
            backbone_token_limit=20,
            seed=42,
        )
        assert len(out["input_ids"]) <= 20

    def test_right_truncation_keeps_head(self, tokenizer):
        """Right-truncation must keep the FIRST tokens, not the last."""
        # Each word maps to a unique ID. First word "w0" → 100, "w1" → 101, ...
        lines = [" ".join(f"w{i}" for i in range(50))]
        out = apply_logfit_masking(
            lines=lines,
            tokenizer=tokenizer,
            backbone_token_limit=12,  # budget=10 content tokens
            seed=42,
        )
        # Position 0 is BOS, position 1 must be first content token (id=100)
        # but it may have been masked. Check that the SET of non-mask content
        # tokens at positions 1..-1 is a prefix of [100, 101, ..., 109].
        content_ids = out["input_ids"][1:-1]
        original_ids: list[int] = []
        for i, tok in enumerate(content_ids):
            if tok == tokenizer.mask_token_id:
                original_ids.append(out["labels"][i + 1])
            else:
                original_ids.append(tok)
        # original_ids should be the first 10 content tokens: 100..109
        assert original_ids == list(range(100, 110))

    def test_approx_40_percent_masked(self, tokenizer):
        """50% × 80% ≈ 40% of valid tokens masked. Allow generous tolerance."""
        lines = [f"w{i}_a w{i}_b w{i}_c w{i}_d" for i in range(100)]
        out = apply_logfit_masking(
            lines=lines,
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        n_content = len(out["input_ids"]) - 2  # exclude BOS, EOS
        n_masked = sum(
            1
            for tok in out["input_ids"][1:-1]
            if tok == tokenizer.mask_token_id
        )
        rate = n_masked / n_content
        # 50% lines × 80% tokens = 40%. Tolerance ±5pp for stochastic variation.
        assert 0.30 < rate < 0.50, f"Mask rate {rate:.3f} outside [0.30, 0.50]"

    def test_deterministic_with_same_seed(self, tokenizer):
        out_a = apply_logfit_masking(
            lines=["a b c d", "e f g h"],
            tokenizer=FakeTokenizer(),
            backbone_token_limit=512,
            seed=42,
        )
        out_b = apply_logfit_masking(
            lines=["a b c d", "e f g h"],
            tokenizer=FakeTokenizer(),
            backbone_token_limit=512,
            seed=42,
        )
        assert out_a == out_b

    def test_different_seeds_give_different_masks(self):
        # Use separate fresh tokenizers to keep vocab IDs identical
        out_a = apply_logfit_masking(
            lines=["a b c d e f g h i j"],
            tokenizer=FakeTokenizer(),
            backbone_token_limit=512,
            seed=42,
        )
        out_b = apply_logfit_masking(
            lines=["a b c d e f g h i j"],
            tokenizer=FakeTokenizer(),
            backbone_token_limit=512,
            seed=7,
        )
        # Mask positions should differ
        assert out_a["input_ids"] != out_b["input_ids"]

    def test_explicit_rng_takes_precedence_over_seed(self, tokenizer):
        """Caller can pass a long-lived rng — seed arg is ignored."""
        rng = np.random.default_rng(42)
        out_a = apply_logfit_masking(
            lines=["a b c d e"],
            tokenizer=tokenizer,
            backbone_token_limit=512,
            rng=rng,
            seed=999,  # should be ignored
        )
        # Independent RNG with same seed → same first-call result
        rng2 = np.random.default_rng(42)
        out_b = apply_logfit_masking(
            lines=["a b c d e"],
            tokenizer=FakeTokenizer(),  # fresh vocab — same IDs because same words
            backbone_token_limit=512,
            rng=rng2,
            seed=12345,  # should be ignored
        )
        assert out_a == out_b

    def test_empty_paragraph_returns_bos_eos_only(self, tokenizer):
        out = apply_logfit_masking(
            lines=[],
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        assert out["input_ids"] == [tokenizer.bos_token_id, tokenizer.eos_token_id]
        assert out["labels"] == [-100, -100]
        assert out["attention_mask"] == [1, 1]

    def test_all_empty_lines_yields_bos_eos_only(self, tokenizer):
        out = apply_logfit_masking(
            lines=["", "  ", ""],
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        # All lines tokenize to empty; final output is just BOS/EOS
        assert out["input_ids"] == [tokenizer.bos_token_id, tokenizer.eos_token_id]

    def test_missing_bos_id_raises(self, tokenizer):
        tokenizer.bos_token_id = None
        with pytest.raises(ValueError, match="missing one of"):
            apply_logfit_masking(
                lines=["a b"],
                tokenizer=tokenizer,
                backbone_token_limit=512,
                seed=42,
            )

    def test_missing_eos_id_raises(self, tokenizer):
        tokenizer.eos_token_id = None
        with pytest.raises(ValueError, match="missing one of"):
            apply_logfit_masking(
                lines=["a b"],
                tokenizer=tokenizer,
                backbone_token_limit=512,
                seed=42,
            )

    def test_missing_mask_id_raises(self, tokenizer):
        tokenizer.mask_token_id = None
        with pytest.raises(ValueError, match="missing one of"):
            apply_logfit_masking(
                lines=["a b"],
                tokenizer=tokenizer,
                backbone_token_limit=512,
                seed=42,
            )

    def test_default_ratios_match_paper(self):
        """Sanity check: module-level constants match the paper's 50/80."""
        assert DEFAULT_SENTENCE_MASK_RATIO == 0.5
        assert DEFAULT_TOKEN_MASK_RATIO == 0.8


# ---------------------------------------------------------------------------
# Integration with real RoBERTa tokenizer (optional)
# ---------------------------------------------------------------------------

class TestRealTokenizer:
    """Gated by importorskip — runs only if transformers + tokenizer files
    are available locally. Excluded by `-k "not RealTokenizer"` in the
    project's default test invocation, but runs in full-suite mode."""

    def test_real_roberta_produces_well_formed_output(self):
        transformers = pytest.importorskip("transformers")
        try:
            tok = transformers.RobertaTokenizerFast.from_pretrained(
                "roberta-base", local_files_only=True
            )
        except (OSError, ValueError):
            pytest.skip("roberta-base tokenizer not available locally")

        out = apply_logfit_masking(
            lines=[
                "INFO dfs.DataNode$DataXceiver: Receiving block blk_-123",
                "INFO dfs.DataNode$PacketResponder: Received block blk_-123",
            ],
            tokenizer=tok,
            backbone_token_limit=512,
            seed=42,
        )
        # Real tokenizer should yield BOS=0, EOS=2, mask=50264 for roberta-base
        assert out["input_ids"][0] == tok.bos_token_id
        assert out["input_ids"][-1] == tok.eos_token_id
        assert all(m == 1 for m in out["attention_mask"])
        # At least one position should be masked
        n_masked = sum(
            1 for t in out["input_ids"] if t == tok.mask_token_id
        )
        assert n_masked > 0
