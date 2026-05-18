"""LogFiT MLM masking per paper §III.B.

Strategy:
- 50% of sentences (lines) within a paragraph selected uniformly at random.
- Within each selected sentence, 80% of tokens replaced with [MASK].
- Other tokens left intact (NO random/keep replacement, unlike BERT MLM).
- Labels: original token ID at masked positions, -100 elsewhere (HF convention).

This is more aggressive than BERT's 15%/(80/10/10) scheme. The intent is to
force reconstruction of entire log lines, not arbitrary subwords.

Pipeline:
1. Tokenize each line without special tokens; track line boundaries in flat
   token coordinates.
2. Right-truncate to backbone_token_limit - 2 (BOS + EOS slots reserved).
   Per v1.4 D_NEW2, truncation is the SUPERVISOR_FLAG path for time-windowed
   configs; mask.py does the truncation, gate.py does the flagging.
3. Sample 50% of (post-truncation) lines, then 80% of tokens within each.
4. Replace selected token IDs with mask_token_id; build labels.
5. Wrap with BOS and EOS.

Reference: logfit-repro-spec-v1.3.md §1.1; configs/*.yaml masking section;
decisions-v1.4.md D_NEW2.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_SENTENCE_MASK_RATIO = 0.5
DEFAULT_TOKEN_MASK_RATIO = 0.8


def select_sentences_to_mask(
    n_lines: int,
    sentence_mask_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose line indices to mask. Returns sorted np.int64 array.

    At least 1 line is selected if n_lines > 0 (even at ratio=0). Ratio is
    capped at 1.0 (all lines).
    """
    if n_lines <= 0:
        return np.array([], dtype=np.int64)
    n_to_mask = max(1, round(sentence_mask_ratio * n_lines))
    n_to_mask = min(n_to_mask, n_lines)
    indices = rng.choice(n_lines, size=n_to_mask, replace=False)
    return np.sort(indices).astype(np.int64)


def select_tokens_in_line(
    start: int,
    end: int,
    token_mask_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose token positions within [start, end) to mask. Sorted ascending.

    At least 1 token is masked if (end - start) > 0. Ratio is capped at 1.0.
    """
    sent_len = end - start
    if sent_len <= 0:
        return np.array([], dtype=np.int64)
    n_to_mask = max(1, round(token_mask_ratio * sent_len))
    n_to_mask = min(n_to_mask, sent_len)
    positions = rng.choice(
        np.arange(start, end), size=n_to_mask, replace=False
    )
    return np.sort(positions).astype(np.int64)


def truncate_boundaries(
    boundaries: list[tuple[int, int]],
    budget: int,
) -> list[tuple[int, int]]:
    """Right-truncate line boundaries to fit within `budget` tokens.

    - Line fully past budget: dropped.
    - Line straddling budget: end clipped to budget.
    - Line entirely within budget: unchanged.

    Zero-length lines (start == end) after clipping are preserved as
    (start, start) so caller can distinguish "line was here but empty" from
    "line was dropped".
    """
    out: list[tuple[int, int]] = []
    for start, end in boundaries:
        if start >= budget:
            break
        out.append((start, min(end, budget)))
    return out


def tokenize_paragraph_lines(
    lines: list[str],
    tokenizer: Any,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize each line without special tokens.

    Returns (flat_token_ids, line_boundaries) where line_boundaries[i] is
    the (start, end) range in flat coordinates for lines[i]; end exclusive.
    """
    flat: list[int] = []
    boundaries: list[tuple[int, int]] = []
    for line in lines:
        ids = tokenizer.encode(line, add_special_tokens=False)
        start = len(flat)
        flat.extend(ids)
        boundaries.append((start, len(flat)))
    return flat, boundaries


def apply_logfit_masking(
    lines: list[str],
    tokenizer: Any,
    backbone_token_limit: int,
    sentence_mask_ratio: float = DEFAULT_SENTENCE_MASK_RATIO,
    token_mask_ratio: float = DEFAULT_TOKEN_MASK_RATIO,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> dict[str, list[int]]:
    """Apply LogFiT MLM masking to a paragraph.

    Args:
        lines: List of log lines (strings) constituting the paragraph.
        tokenizer: HF tokenizer with bos_token_id, eos_token_id,
            mask_token_id, and an encode(text, add_special_tokens=False)
            method that returns a list[int].
        backbone_token_limit: Max sequence length INCLUDING BOS and EOS
            (e.g., 512 for RoBERTa, 4096 for Longformer).
        sentence_mask_ratio: Fraction of lines to mask. Default 0.5.
        token_mask_ratio: Fraction of tokens to mask within selected lines.
            Default 0.8.
        rng: numpy RNG. If None, one is built from `seed`.
        seed: Seed for the RNG. Used only if `rng` is None.

    Returns:
        dict with keys 'input_ids', 'labels', 'attention_mask' (each list[int]
        of equal length, length <= backbone_token_limit).

        - input_ids: BOS, then real/masked tokens, then EOS.
        - labels: -100 at every position EXCEPT masked positions, where it
          holds the original token ID. (HF Trainer ignores -100 in loss.)
        - attention_mask: all 1s. Padding to batch_max is the collator's job.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id
    if bos is None or eos is None or mask_id is None:
        raise ValueError(
            f"Tokenizer {type(tokenizer).__name__} is missing one of "
            f"bos_token_id, eos_token_id, mask_token_id."
        )

    # Step 1: tokenize line-by-line, track boundaries
    flat, boundaries = tokenize_paragraph_lines(lines, tokenizer)

    # Step 2: right-truncate. Reserve 2 slots for BOS/EOS.
    budget = max(0, backbone_token_limit - 2)
    if len(flat) > budget:
        flat = flat[:budget]
    boundaries = truncate_boundaries(boundaries, budget)

    # Step 3+4: select sentences, then tokens within each, then mask
    labels_flat = [-100] * len(flat)
    sent_indices = select_sentences_to_mask(
        len(boundaries), sentence_mask_ratio, rng
    )
    for si in sent_indices:
        start, end = boundaries[int(si)]
        positions = select_tokens_in_line(start, end, token_mask_ratio, rng)
        for pos in positions:
            p = int(pos)
            labels_flat[p] = flat[p]
            flat[p] = mask_id

    # Step 5: wrap with BOS/EOS
    input_ids = [bos] + flat + [eos]
    labels = [-100] + labels_flat + [-100]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
