"""Tests for src/variability_bgl.py — D17 v1.2 BGL variability test.

Strategy:
- Pure-logic functions (substitution, filtering, case preservation) tested
  offline with synthetic data. No NLTK / WordNet dependency.
- NLTK-dependent functions tested in a class that skips gracefully if NLTK
  or its corpora are unavailable.
"""

from __future__ import annotations

import pytest

from src.types import Paragraph
from src.variability_bgl import (
    DEFAULT_TOP_K_VERBS,
    LEMMA_SEPARATOR_RE,
    VERB_POS_TAGS,
    VariabilityReport,
    _preserve_case,
    apply_substitutions_to_paragraphs,
    build_substitutions_from_lemmas,
    is_single_word_lemma,
    substitute_in_lines,
)


# ---------------------------------------------------------------------------
# is_single_word_lemma — D17 v1.2 filter
# ---------------------------------------------------------------------------


class TestIsSingleWordLemma:
    def test_simple_word_accepted(self):
        assert is_single_word_lemma("acquire") is True
        assert is_single_word_lemma("receive") is True

    def test_underscore_rejected(self):
        """WordNet uses underscores in multi-word lemmas like 'look_up'."""
        assert is_single_word_lemma("look_up") is False
        assert is_single_word_lemma("set_up") is False

    def test_hyphen_rejected(self):
        assert is_single_word_lemma("get-together") is False

    def test_whitespace_rejected(self):
        assert is_single_word_lemma("get together") is False

    def test_empty_string_rejected(self):
        assert is_single_word_lemma("") is False

    def test_unicode_letter_accepted(self):
        # Lemma with only letters (any case) is single-word
        assert is_single_word_lemma("Acquire") is True


# ---------------------------------------------------------------------------
# _preserve_case
# ---------------------------------------------------------------------------


class TestPreserveCase:
    def test_all_caps_preserved(self):
        assert _preserve_case("RECEIVING", "acquire") == "ACQUIRE"

    def test_title_case_preserved(self):
        assert _preserve_case("Receiving", "acquire") == "Acquire"

    def test_lowercase_preserved(self):
        assert _preserve_case("receiving", "acquire") == "acquire"

    def test_single_uppercase_letter_treated_as_lowercase_origin(self):
        # 'A' alone is treated as lowercase result for safety (avoids
        # spurious uppercasing for single-letter words like 'I')
        result = _preserve_case("A", "acquire")
        # Either "acquire" (lowercase) or "Acquire" (capitalize) is fine,
        # but it should not be all-caps "ACQUIRE" from single-letter origin.
        assert result.lower() == "acquire"
        assert not result.isupper() or len(result) <= 1

    def test_mixed_case_origin_returns_lowercase(self):
        assert _preserve_case("ReCeIvInG", "acquire") == "acquire"


# ---------------------------------------------------------------------------
# substitute_in_lines
# ---------------------------------------------------------------------------


class TestSubstituteInLines:
    def test_no_substitutions_returns_lines_unchanged(self):
        lines = ["INFO Receiving block blk_123"]
        result, count = substitute_in_lines(lines, {})
        assert result == lines
        assert count == 0

    def test_single_substitution_with_case_preserved(self):
        lines = ["INFO Receiving block blk_123"]
        result, count = substitute_in_lines(lines, {"receiving": "acquiring"})
        assert result == ["INFO Acquiring block blk_123"]
        assert count == 1

    def test_multiple_occurrences_in_same_line_all_replaced(self):
        lines = ["receiving and receiving and receiving"]
        result, count = substitute_in_lines(lines, {"receiving": "acquiring"})
        assert result == ["acquiring and acquiring and acquiring"]
        assert count == 3

    def test_whole_word_only_no_prefix_match(self):
        """'receive' should NOT match inside 'receiver' or 'received'."""
        lines = ["Receiver got data"]
        result, count = substitute_in_lines(lines, {"receive": "acquire"})
        assert result == ["Receiver got data"]
        assert count == 0

    def test_whole_word_match_at_boundaries(self):
        """Word boundary matches around punctuation correctly."""
        lines = ["sent: receive,done"]
        result, count = substitute_in_lines(lines, {"receive": "acquire"})
        assert result == ["sent: acquire,done"]
        assert count == 1

    def test_multiple_substitutions_at_once(self):
        lines = ["Receiving block, then sending packet"]
        result, count = substitute_in_lines(
            lines,
            {"receiving": "acquiring", "sending": "transmitting"},
        )
        assert "Acquiring" in result[0]
        assert "transmitting" in result[0]
        assert count == 2

    def test_all_caps_substitution(self):
        lines = ["WARNING RECEIVING failed"]
        result, count = substitute_in_lines(lines, {"receiving": "acquiring"})
        assert result == ["WARNING ACQUIRING failed"]
        assert count == 1

    def test_multiple_lines_independent(self):
        lines = ["receive block", "no match here", "receive other"]
        result, count = substitute_in_lines(lines, {"receive": "acquire"})
        assert result == ["acquire block", "no match here", "acquire other"]
        assert count == 2

    def test_special_regex_chars_in_verb_escaped(self):
        """Verbs with regex special chars should be matched literally."""
        # Implausible but defensive: a 'verb' with a dot
        lines = ["foo bar"]
        # If escape weren't used, '.' would match any char.
        result, count = substitute_in_lines(lines, {"foo.bar": "baz"})
        # 'foo bar' should NOT match 'foo.bar' (literal dot expected)
        assert result == ["foo bar"]
        assert count == 0


# ---------------------------------------------------------------------------
# build_substitutions_from_lemmas — D17 filters
# ---------------------------------------------------------------------------


class TestBuildSubstitutionsFromLemmas:
    def test_accepts_simple_substitution(self):
        accepted, no_lemma, mw, ident = build_substitutions_from_lemmas(
            {"receive": "acquire"}
        )
        assert accepted == {"receive": "acquire"}
        assert no_lemma == []
        assert mw == {}
        assert ident == []

    def test_filters_none_lemma(self):
        accepted, no_lemma, mw, ident = build_substitutions_from_lemmas(
            {"foobaz": None}
        )
        assert accepted == {}
        assert no_lemma == ["foobaz"]

    def test_filters_multiword_lemma(self):
        accepted, no_lemma, mw, ident = build_substitutions_from_lemmas(
            {"check": "look_up"}
        )
        assert accepted == {}
        assert mw == {"check": "look_up"}
        assert no_lemma == []

    def test_filters_identity_substitution(self):
        """If lemma is the same as the verb, no point substituting."""
        accepted, no_lemma, mw, ident = build_substitutions_from_lemmas(
            {"receive": "receive"}
        )
        assert accepted == {}
        assert ident == ["receive"]

    def test_identity_check_is_case_insensitive(self):
        accepted, no_lemma, mw, ident = build_substitutions_from_lemmas(
            {"Receive": "receive"}
        )
        assert accepted == {}
        assert ident == ["Receive"]

    def test_mixed_filters_all_categories(self):
        """Cover all four outcome categories in one call."""
        verb_to_lemma = {
            "receive": "acquire",      # accepted
            "foobaz": None,            # no lemma
            "check": "look_up",        # multiword
            "send": "send",            # identity
        }
        accepted, no_lemma, mw, ident = build_substitutions_from_lemmas(
            verb_to_lemma
        )
        assert accepted == {"receive": "acquire"}
        assert no_lemma == ["foobaz"]
        assert mw == {"check": "look_up"}
        assert ident == ["send"]

    def test_empty_input(self):
        accepted, no_lemma, mw, ident = build_substitutions_from_lemmas({})
        assert accepted == {}
        assert no_lemma == []
        assert mw == {}
        assert ident == []


# ---------------------------------------------------------------------------
# apply_substitutions_to_paragraphs
# ---------------------------------------------------------------------------


class TestApplySubstitutionsToParagraphs:
    def test_returns_same_count_with_modified_content(self):
        paragraphs = [
            Paragraph(paragraph_id="p1", lines=["receive block"], label=1),
            Paragraph(paragraph_id="p2", lines=["normal line"], label=0),
        ]
        new_paras, n_modified, n_subs = apply_substitutions_to_paragraphs(
            paragraphs, {"receive": "acquire"}
        )
        assert len(new_paras) == 2
        assert n_modified == 1
        assert n_subs == 1
        assert new_paras[0].lines == ["acquire block"]
        assert new_paras[1].lines == ["normal line"]

    def test_preserves_paragraph_metadata(self):
        paragraphs = [
            Paragraph(
                paragraph_id="bgl_30s_w42",
                lines=["receive"],
                label=1,
                source_window_id=42,
                start_timestamp=1700000000.0,
            ),
        ]
        new_paras, _, _ = apply_substitutions_to_paragraphs(
            paragraphs, {"receive": "acquire"}
        )
        assert new_paras[0].paragraph_id == "bgl_30s_w42"
        assert new_paras[0].label == 1
        assert new_paras[0].source_window_id == 42
        assert new_paras[0].start_timestamp == 1700000000.0

    def test_empty_substitutions_returns_unmodified_paragraphs(self):
        paragraphs = [
            Paragraph(paragraph_id="p1", lines=["receive"], label=1),
        ]
        new_paras, n_modified, n_subs = apply_substitutions_to_paragraphs(
            paragraphs, {}
        )
        assert n_modified == 0
        assert n_subs == 0
        assert new_paras[0].lines == ["receive"]

    def test_empty_paragraph_list(self):
        new_paras, n_modified, n_subs = apply_substitutions_to_paragraphs(
            [], {"x": "y"}
        )
        assert new_paras == []
        assert n_modified == 0
        assert n_subs == 0

    def test_paragraph_with_empty_lines(self):
        paragraphs = [Paragraph(paragraph_id="p1", lines=[], label=1)]
        new_paras, n_modified, n_subs = apply_substitutions_to_paragraphs(
            paragraphs, {"x": "y"}
        )
        assert new_paras[0].lines == []
        assert n_modified == 0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_top_k_is_10(self):
        """D13: top-10 verbs."""
        assert DEFAULT_TOP_K_VERBS == 10

    def test_verb_pos_tags_includes_all_forms(self):
        """All Penn Treebank verb tags should be recognized."""
        assert "VB" in VERB_POS_TAGS    # base
        assert "VBD" in VERB_POS_TAGS   # past tense
        assert "VBG" in VERB_POS_TAGS   # gerund
        assert "VBN" in VERB_POS_TAGS   # past participle
        assert "VBP" in VERB_POS_TAGS   # non-3rd-person singular present
        assert "VBZ" in VERB_POS_TAGS   # 3rd-person singular present

    def test_lemma_separator_regex(self):
        """The separator regex matches underscores, hyphens, whitespace."""
        assert LEMMA_SEPARATOR_RE.search("a_b") is not None
        assert LEMMA_SEPARATOR_RE.search("a-b") is not None
        assert LEMMA_SEPARATOR_RE.search("a b") is not None
        assert LEMMA_SEPARATOR_RE.search("ab") is None


# ---------------------------------------------------------------------------
# VariabilityReport
# ---------------------------------------------------------------------------


class TestVariabilityReport:
    def test_default_construction_safe(self):
        r = VariabilityReport()
        assert r.top_verbs == []
        assert r.substitutions == {}
        assert r.paragraphs_modified == 0
        assert r.total_substitutions_applied == 0


# ---------------------------------------------------------------------------
# NLTK-dependent integration (skipped if NLTK / corpora unavailable)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_nltk_available():
    """Skip dependent tests if NLTK or its required corpora aren't ready."""
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        from nltk.tag import pos_tag
        from nltk.tokenize import word_tokenize
        # Smoke check: requires wordnet + tagger + tokenizer to actually work
        word_tokenize("hello world")
        pos_tag(["hello"])
        wn.synsets("receive", pos="v")
        return True
    except (ImportError, LookupError):
        pytest.skip(
            "NLTK or one of {wordnet, averaged_perceptron_tagger, punkt} "
            "is not available. Install + download with:\n"
            "  pip install nltk\n"
            "  python -m nltk.downloader wordnet "
            "averaged_perceptron_tagger punkt"
        )


class TestNLTKIntegration:
    def test_identify_top_k_verbs_returns_verbs(self, real_nltk_available):
        from src.variability_bgl import identify_top_k_verbs

        paragraphs = [
            Paragraph(
                paragraph_id="p1",
                lines=[
                    "Receiving block blk_123 from server",
                    "Sending response to client",
                    "Receiving block blk_456 from server",
                    "Closing connection",
                    "Receiving block blk_789 acknowledged",
                ],
                label=0,
            )
        ]
        verbs = identify_top_k_verbs(paragraphs, k=5)
        # Should include at least 'receiving' or 'sending' or similar
        assert len(verbs) > 0
        assert all(v.islower() for v in verbs)
        # At least one expected verb appears
        assert any(v in {"receiving", "sending", "closing"} for v in verbs)

    def test_get_first_synset_first_lemma_common_verb(
        self, real_nltk_available
    ):
        from src.variability_bgl import get_first_synset_first_lemma

        lemma = get_first_synset_first_lemma("receive")
        assert lemma is not None
        assert isinstance(lemma, str)
        # The lemma for 'receive' in its first verb-sense synset should be
        # 'receive' itself, but at minimum a string is returned.

    def test_get_first_synset_first_lemma_nonexistent_word(
        self, real_nltk_available
    ):
        from src.variability_bgl import get_first_synset_first_lemma

        lemma = get_first_synset_first_lemma("xyzqwerty12345")
        assert lemma is None
