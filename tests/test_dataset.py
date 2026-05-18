"""Tests for src/dataset.py — MaskedSentenceDataset + MaskedSentenceCollator.

Strategy:
- All Dataset behavior tested with a fake whitespace tokenizer (no network).
- Collator tested with a minimal stub exposing only pad_token_id.
- One optional integration test class (TestRealTokenizerIntegration) gated
  by pytest.importorskip — wires Dataset → Collator with a real tokenizer.

Mapping to spec-v1.2 Component 4.1:
- Determinism contract: seed + iteration order → identical masks
- Freshness contract: same idx, two calls → different masks (RNG advances)
- Collator: pad_token_id for input_ids, -100 for labels, 0 for attention_mask
"""

from __future__ import annotations

import pytest
import torch

from src.dataset import MaskedSentenceCollator, MaskedSentenceDataset
from src.types import Paragraph
from tests.test_mask import FakeTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paragraphs(n: int, lines_per: int = 3, words_per: int = 4) -> list[Paragraph]:
    """Build n synthetic Paragraph records with deterministic content."""
    paragraphs = []
    for i in range(n):
        lines = [
            " ".join(f"p{i}_l{j}_w{k}" for k in range(words_per))
            for j in range(lines_per)
        ]
        paragraphs.append(
            Paragraph(paragraph_id=f"p_{i}", lines=lines, label=0)
        )
    return paragraphs


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def paragraphs():
    return _make_paragraphs(10)


# ---------------------------------------------------------------------------
# MaskedSentenceDataset — construction / validation
# ---------------------------------------------------------------------------

class TestDatasetConstruction:
    def test_empty_paragraphs_raises(self, tokenizer):
        with pytest.raises(ValueError, match="at least one paragraph"):
            MaskedSentenceDataset(
                paragraphs=[],
                tokenizer=tokenizer,
                backbone_token_limit=512,
            )

    def test_invalid_token_limit_raises(self, tokenizer, paragraphs):
        with pytest.raises(ValueError, match="backbone_token_limit"):
            MaskedSentenceDataset(
                paragraphs=paragraphs,
                tokenizer=tokenizer,
                backbone_token_limit=1,
            )

    def test_invalid_r_sent_raises(self, tokenizer, paragraphs):
        with pytest.raises(ValueError, match="r_sent"):
            MaskedSentenceDataset(
                paragraphs=paragraphs,
                tokenizer=tokenizer,
                backbone_token_limit=512,
                r_sent=1.5,
            )

    def test_invalid_r_tok_raises(self, tokenizer, paragraphs):
        with pytest.raises(ValueError, match="r_tok"):
            MaskedSentenceDataset(
                paragraphs=paragraphs,
                tokenizer=tokenizer,
                backbone_token_limit=512,
                r_tok=-0.1,
            )

    def test_len_matches_paragraph_count(self, tokenizer, paragraphs):
        ds = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=tokenizer,
            backbone_token_limit=512,
        )
        assert len(ds) == 10


# ---------------------------------------------------------------------------
# MaskedSentenceDataset — __getitem__ contract
# ---------------------------------------------------------------------------

class TestDatasetGetItem:
    def test_returns_three_keys(self, tokenizer, paragraphs):
        ds = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        out = ds[0]
        assert set(out.keys()) == {"input_ids", "labels", "attention_mask"}

    def test_keys_are_lists_of_int(self, tokenizer, paragraphs):
        ds = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        out = ds[0]
        for k in ("input_ids", "labels", "attention_mask"):
            assert isinstance(out[k], list)
            assert all(isinstance(v, int) for v in out[k])

    def test_bos_and_eos_at_endpoints(self, tokenizer, paragraphs):
        ds = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        out = ds[0]
        assert out["input_ids"][0] == tokenizer.bos_token_id
        assert out["input_ids"][-1] == tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# MaskedSentenceDataset — determinism + freshness contracts
# ---------------------------------------------------------------------------

class TestDatasetDeterminism:
    def test_same_seed_same_walk_produces_identical_outputs(
        self, paragraphs
    ):
        """The deterministic contract: same construction seed + same
        access pattern → identical sequence of outputs."""
        ds_a = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=FakeTokenizer(),
            backbone_token_limit=512,
            seed=42,
        )
        ds_b = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=FakeTokenizer(),
            backbone_token_limit=512,
            seed=42,
        )
        for idx in range(len(ds_a)):
            assert ds_a[idx] == ds_b[idx]

    def test_different_seeds_diverge(self, paragraphs):
        ds_a = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=FakeTokenizer(),
            backbone_token_limit=512,
            seed=42,
        )
        ds_b = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=FakeTokenizer(),
            backbone_token_limit=512,
            seed=7,
        )
        # At least one example should differ
        diffs = sum(
            1 for idx in range(len(ds_a)) if ds_a[idx] != ds_b[idx]
        )
        assert diffs > 0


class TestDatasetFreshness:
    def test_same_idx_two_calls_yield_different_masks(
        self, tokenizer, paragraphs
    ):
        """Freshness contract: RNG advances per __getitem__ call, so
        calling ds[0] twice produces DIFFERENT masks. This is the
        "on-the-fly per epoch" behavior the paper specifies."""
        ds = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        out_a = ds[0]
        out_b = ds[0]
        # Probability of identical masks under 50%×80% over ~12 tokens
        # is astronomically small.
        assert out_a["input_ids"] != out_b["input_ids"]

    def test_reset_rng_returns_to_initial_state(
        self, tokenizer, paragraphs
    ):
        """reset_rng() restores construction-time RNG state, so a
        re-walk reproduces the original sequence."""
        ds = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        first_walk = [ds[idx] for idx in range(len(ds))]
        ds.reset_rng()
        second_walk = [ds[idx] for idx in range(len(ds))]
        assert first_walk == second_walk

    def test_two_epoch_walk_diverges(self, tokenizer, paragraphs):
        """Two consecutive walks WITHOUT reset must differ — that's how
        epochs get different masks for the same paragraph."""
        ds = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        first = [ds[idx] for idx in range(len(ds))]
        second = [ds[idx] for idx in range(len(ds))]
        # At least one paragraph should produce a different mask
        # between the two "epochs".
        diffs = sum(1 for a, b in zip(first, second) if a != b)
        assert diffs > 0


# ---------------------------------------------------------------------------
# MaskedSentenceCollator — construction
# ---------------------------------------------------------------------------

class _NoPadTokenizer:
    """Stub tokenizer with no pad_token_id — for error-path tests."""

    pad_token_id = None


class TestCollatorConstruction:
    def test_no_pad_token_raises(self):
        with pytest.raises(ValueError, match="pad_token_id"):
            MaskedSentenceCollator(_NoPadTokenizer())

    def test_accepts_valid_tokenizer(self, tokenizer):
        # Should not raise
        c = MaskedSentenceCollator(tokenizer)
        assert c.pad_token_id == tokenizer.pad_token_id


# ---------------------------------------------------------------------------
# MaskedSentenceCollator — padding behavior
# ---------------------------------------------------------------------------

class TestCollatorPadding:
    def test_empty_batch_raises(self, tokenizer):
        c = MaskedSentenceCollator(tokenizer)
        with pytest.raises(ValueError, match="empty batch"):
            c([])

    def test_single_item_batch_unchanged(self, tokenizer):
        c = MaskedSentenceCollator(tokenizer)
        feature = {
            "input_ids": [0, 5, 6, 2],
            "labels": [-100, -100, 6, -100],
            "attention_mask": [1, 1, 1, 1],
        }
        batch = c([feature])
        assert batch["input_ids"].shape == (1, 4)
        assert batch["input_ids"].tolist() == [[0, 5, 6, 2]]
        assert batch["labels"].tolist() == [[-100, -100, 6, -100]]
        assert batch["attention_mask"].tolist() == [[1, 1, 1, 1]]

    def test_pads_to_max_length(self, tokenizer):
        c = MaskedSentenceCollator(tokenizer)
        features = [
            {
                "input_ids": [0, 5, 2],
                "labels": [-100, 5, -100],
                "attention_mask": [1, 1, 1],
            },
            {
                "input_ids": [0, 5, 6, 7, 2],
                "labels": [-100, -100, 6, 7, -100],
                "attention_mask": [1, 1, 1, 1, 1],
            },
        ]
        batch = c(features)
        # max_len = 5
        assert batch["input_ids"].shape == (2, 5)
        assert batch["labels"].shape == (2, 5)
        assert batch["attention_mask"].shape == (2, 5)

    def test_pad_conventions(self, tokenizer):
        c = MaskedSentenceCollator(tokenizer)
        features = [
            {
                "input_ids": [0, 5, 2],
                "labels": [-100, 5, -100],
                "attention_mask": [1, 1, 1],
            },
            {
                "input_ids": [0, 5, 6, 7, 2],
                "labels": [-100, -100, 6, 7, -100],
                "attention_mask": [1, 1, 1, 1, 1],
            },
        ]
        batch = c(features)
        pad = tokenizer.pad_token_id
        # Padded positions in shorter sample (indices 3, 4)
        assert batch["input_ids"][0, 3].item() == pad
        assert batch["input_ids"][0, 4].item() == pad
        assert batch["labels"][0, 3].item() == -100
        assert batch["labels"][0, 4].item() == -100
        assert batch["attention_mask"][0, 3].item() == 0
        assert batch["attention_mask"][0, 4].item() == 0

    def test_real_content_unchanged(self, tokenizer):
        """Padding must NOT corrupt content positions."""
        c = MaskedSentenceCollator(tokenizer)
        features = [
            {
                "input_ids": [0, 5, 2],
                "labels": [-100, 5, -100],
                "attention_mask": [1, 1, 1],
            },
            {
                "input_ids": [0, 5, 6, 7, 2],
                "labels": [-100, -100, 6, 7, -100],
                "attention_mask": [1, 1, 1, 1, 1],
            },
        ]
        batch = c(features)
        # Real positions in shorter sample (indices 0..2)
        assert batch["input_ids"][0, :3].tolist() == [0, 5, 2]
        assert batch["labels"][0, :3].tolist() == [-100, 5, -100]
        assert batch["attention_mask"][0, :3].tolist() == [1, 1, 1]

    def test_returns_int64_tensors(self, tokenizer):
        c = MaskedSentenceCollator(tokenizer)
        feature = {
            "input_ids": [0, 5, 2],
            "labels": [-100, 5, -100],
            "attention_mask": [1, 1, 1],
        }
        batch = c([feature])
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long

    def test_inconsistent_feature_lengths_raises(self, tokenizer):
        c = MaskedSentenceCollator(tokenizer)
        bad_feature = {
            "input_ids": [0, 5, 2],
            "labels": [-100, 5],  # too short
            "attention_mask": [1, 1, 1],
        }
        with pytest.raises(ValueError, match="Inconsistent lengths"):
            c([bad_feature])


# ---------------------------------------------------------------------------
# Dataset + Collator integration
# ---------------------------------------------------------------------------

class TestDatasetCollatorIntegration:
    def test_dataset_outputs_collate_cleanly(self, tokenizer, paragraphs):
        ds = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        collator = MaskedSentenceCollator(tokenizer)
        features = [ds[i] for i in range(4)]
        batch = collator(features)
        assert batch["input_ids"].shape[0] == 4
        # All three tensors have matching shape
        assert batch["input_ids"].shape == batch["labels"].shape
        assert batch["input_ids"].shape == batch["attention_mask"].shape

    def test_mini_epoch_walk_produces_valid_tensors(
        self, tokenizer, paragraphs
    ):
        ds = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=tokenizer,
            backbone_token_limit=512,
            seed=42,
        )
        collator = MaskedSentenceCollator(tokenizer)
        # Simulate two batches of 5
        for batch_start in (0, 5):
            features = [ds[i] for i in range(batch_start, batch_start + 5)]
            batch = collator(features)
            assert batch["input_ids"].shape[0] == 5
            assert batch["input_ids"].dtype == torch.long


# ---------------------------------------------------------------------------
# Integration with real RoBERTa tokenizer (optional)
# ---------------------------------------------------------------------------

class TestRealTokenizerIntegration:
    """Gated by importorskip — runs only if transformers + tokenizer files
    are available locally. Excluded by `-k "not RealTokenizer"` in the
    project's default test invocation."""

    def test_end_to_end_with_roberta(self):
        transformers = pytest.importorskip("transformers")
        try:
            tok = transformers.RobertaTokenizerFast.from_pretrained(
                "roberta-base", local_files_only=True
            )
        except (OSError, ValueError):
            pytest.skip("roberta-base tokenizer not available locally")

        paragraphs = [
            Paragraph(
                paragraph_id="p_0",
                lines=[
                    "INFO dfs.DataNode$DataXceiver: Receiving block blk_-1",
                    "INFO dfs.DataNode$PacketResponder: Received block blk_-1",
                ],
                label=0,
            ),
            Paragraph(
                paragraph_id="p_1",
                lines=[
                    "INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock",
                ],
                label=0,
            ),
        ]
        ds = MaskedSentenceDataset(
            paragraphs=paragraphs,
            tokenizer=tok,
            backbone_token_limit=512,
            seed=42,
        )
        collator = MaskedSentenceCollator(tok)
        features = [ds[i] for i in range(len(ds))]
        batch = collator(features)

        assert batch["input_ids"].shape[0] == 2
        # At least one mask token somewhere
        assert (batch["input_ids"] == tok.mask_token_id).any().item()
