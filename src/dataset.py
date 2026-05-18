"""HuggingFace-compatible Dataset and Collator for LogFiT masked LM training.

This module bridges the masking algorithm in `src.mask` with HF `Trainer`:

- `MaskedSentenceDataset` wraps a list of `Paragraph` records and serves
  freshly-masked examples to the Trainer. Each `__getitem__` call advances a
  single internal numpy `Generator`, producing FRESH masks per epoch while
  remaining fully deterministic given the initial seed and a fixed iteration
  order. This matches the paper's intent of "on-the-fly per epoch" masking
  (spec-v1.2 Component 3 / 4.1).

- `MaskedSentenceCollator` pads variable-length sequences in a batch to the
  per-batch max length using HF's standard conventions: `pad_token_id` for
  `input_ids`, `-100` for `labels` (loss-ignore), and `0` for `attention_mask`.

Determinism contract (decisions-v1.2 D8 / T7):
    The Dataset advances a single RNG per `__getitem__`. Given an identical
    construction seed and identical iteration order, two runs produce the
    exact same sequence of masks. Per-paragraph hash seeding (the v1.0
    design) was REMOVED in v1.1 because (i) Python `hash()` is randomized
    per process and (ii) deterministic per-paragraph seeding locks unlucky
    paragraphs into FN territory under single-pass scoring. See decisions
    v1.2 finding B2.

Single-worker requirement:
    Must be used with `dataloader_num_workers=0`. Multi-worker DataLoaders
    fork RNG state non-deterministically. Config defaults pin this (see
    spec Component 4.1 / configs/*.yaml).

Reference: logfit-repro-spec-v1.2.md Components 3 + 4.1; decisions-v1.2 D8.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.mask import (
    DEFAULT_SENTENCE_MASK_RATIO,
    DEFAULT_TOKEN_MASK_RATIO,
    apply_logfit_masking,
)
from src.types import Paragraph


class MaskedSentenceDataset(Dataset):
    """Wraps Paragraph records; serves freshly-masked examples to HF Trainer.

    Args:
        paragraphs: List of Paragraph records (already filtered to the
            target split — usually train_normal for fine-tuning).
        tokenizer: HF tokenizer with `bos_token_id`, `eos_token_id`,
            `mask_token_id`, and an `encode(text, add_special_tokens=False)`
            method returning `list[int]`. RobertaTokenizerFast and
            LongformerTokenizerFast both qualify.
        backbone_token_limit: Max sequence length INCLUDING BOS+EOS
            (512 for RoBERTa, 4096 for Longformer).
        r_sent: Fraction of sentences (lines) to mask per paragraph.
            Default 0.5 per paper.
        r_tok: Fraction of tokens to mask within selected lines.
            Default 0.8 per paper.
        seed: Initial RNG seed. The RNG advances once per `__getitem__`.

    Determinism vs. freshness:
        - DETERMINISM: same seed + same iteration order → same masks.
        - FRESHNESS: `ds[i]` called twice in the same instance returns
          DIFFERENT masks because the RNG advanced. This is by design.
          HF Trainer does not re-call `__getitem__` on the same index
          within an epoch, so freshness only manifests across epochs.

    Edge cases:
        - Paragraphs with empty `lines` list are returned as `[BOS, EOS]`
          with no masked positions (no-op loss for that example).
        - Paragraphs where every line tokenizes to empty likewise yield
          `[BOS, EOS]`.
    """

    def __init__(
        self,
        paragraphs: list[Paragraph],
        tokenizer: Any,
        backbone_token_limit: int,
        r_sent: float = DEFAULT_SENTENCE_MASK_RATIO,
        r_tok: float = DEFAULT_TOKEN_MASK_RATIO,
        seed: int = 42,
    ):
        if not paragraphs:
            raise ValueError(
                "MaskedSentenceDataset requires at least one paragraph"
            )
        if backbone_token_limit < 2:
            raise ValueError(
                f"backbone_token_limit must be >= 2 (BOS+EOS); "
                f"got {backbone_token_limit}"
            )
        if not (0.0 <= r_sent <= 1.0):
            raise ValueError(f"r_sent must be in [0, 1]; got {r_sent}")
        if not (0.0 <= r_tok <= 1.0):
            raise ValueError(f"r_tok must be in [0, 1]; got {r_tok}")

        self.paragraphs = paragraphs
        self.tokenizer = tokenizer
        self.backbone_token_limit = backbone_token_limit
        self.r_sent = r_sent
        self.r_tok = r_tok
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paragraphs)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        para = self.paragraphs[idx]
        return apply_logfit_masking(
            lines=para.lines,
            tokenizer=self.tokenizer,
            backbone_token_limit=self.backbone_token_limit,
            sentence_mask_ratio=self.r_sent,
            token_mask_ratio=self.r_tok,
            rng=self._rng,
        )

    def reset_rng(self) -> None:
        """Reset the internal RNG to the construction seed.

        Useful for diagnostics — calling this puts the Dataset back in the
        state it was just after construction. Not called by HF Trainer in
        production. Tests rely on this for deterministic re-walks.
        """
        self._rng = np.random.default_rng(self.seed)


class MaskedSentenceCollator:
    """Pad a list of masked-sequence dicts into a tensor batch for HF Trainer.

    The Dataset emits variable-length sequences (per-paragraph token count
    varies). HF Trainer needs a batch with equal lengths. This collator
    right-pads to the batch's max length using HF conventions:

        - input_ids: padded with `tokenizer.pad_token_id`
        - labels:    padded with `-100` (CE loss ignores -100)
        - attention_mask: padded with `0`

    Returns int64 (`torch.long`) CPU tensors; HF Trainer moves to device.

    Args:
        tokenizer: HF tokenizer with `pad_token_id` (both RoBERTa and
            Longformer define `<pad>` = 1 by default).
    """

    def __init__(self, tokenizer: Any):
        if (
            not hasattr(tokenizer, "pad_token_id")
            or tokenizer.pad_token_id is None
        ):
            raise ValueError(
                f"Tokenizer {type(tokenizer).__name__} has no pad_token_id. "
                "RoBERTa and Longformer both define one by default; check "
                "tokenizer loading."
            )
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(
        self, features: list[dict[str, list[int]]]
    ) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError(
                "MaskedSentenceCollator received an empty batch"
            )

        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []

        for f in features:
            seq_len = len(f["input_ids"])
            # Consistency check — caller's contract is that all three lists
            # are equal length. We assert rather than silently ignore.
            if len(f["labels"]) != seq_len or len(f["attention_mask"]) != seq_len:
                raise ValueError(
                    f"Inconsistent lengths in feature: input_ids={seq_len}, "
                    f"labels={len(f['labels'])}, "
                    f"attention_mask={len(f['attention_mask'])}"
                )
            pad_len = max_len - seq_len
            batch_input_ids.append(
                list(f["input_ids"]) + [self.pad_token_id] * pad_len
            )
            batch_labels.append(list(f["labels"]) + [-100] * pad_len)
            batch_attention_mask.append(
                list(f["attention_mask"]) + [0] * pad_len
            )

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                batch_attention_mask, dtype=torch.long
            ),
        }
