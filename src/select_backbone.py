"""Backbone selection per spec Component 2.

Implements the L2 heuristic from paper section IV-A: pick RoBERTa for short
paragraphs, Longformer for long ones, based on the 0.8 quantile of paragraph
word counts.

The threshold is 512 words (matching RoBERTa's token limit). Paragraphs whose
0.8 quantile in words exceeds 512 are likely to overflow RoBERTa's 512-token
context window after BPE subword tokenization, so Longformer is selected.

Reference: logfit-repro-spec-v1.3.md Component 2; locked spec [L2].
"""

from __future__ import annotations

from src.types import BackboneName, LengthDistribution

# [L2] Word-count quantile above which Longformer is required.
WORD_QUANTILE_THRESHOLD = 512

# Backbone identifiers (must match HuggingFace model hub names).
ROBERTA_BACKBONE: BackboneName = "roberta-base"
LONGFORMER_BACKBONE: BackboneName = "allenai/longformer-base-4096"


def select_backbone(
    word_length_distribution: LengthDistribution,
    quantile_threshold: int = WORD_QUANTILE_THRESHOLD,
) -> BackboneName:
    """Pick RoBERTa or Longformer based on word-count distribution.

    [L2] If the 0.8 quantile of paragraph word counts is ≤ threshold,
    paragraphs typically fit in RoBERTa's 512-token context window;
    otherwise Longformer-base-4096 is selected.

    Parameters
    ----------
    word_length_distribution : LengthDistribution
        Per-paragraph word counts from `compute_length_distribution`.
        Typically the `word_length_distribution` field of `PreparationSummary`.
    quantile_threshold : int
        Word-count threshold at the 0.8 quantile. Default 512 (matches paper
        and RoBERTa's token limit).

    Returns
    -------
    BackboneName : "roberta-base" or "allenai/longformer-base-4096"
    """
    if word_length_distribution.p80 <= quantile_threshold:
        return ROBERTA_BACKBONE
    return LONGFORMER_BACKBONE


def backbone_token_limit(backbone_name: str) -> int:
    """Return the maximum-input-tokens limit for a backbone.

    Used by `src/token_length_gate.py` to compute truncation rates and
    SUPERVISOR_FLAG / HARD FAIL conditions.
    """
    if "roberta-base" in backbone_name.lower():
        return 512
    if "longformer-base-4096" in backbone_name.lower():
        return 4096
    raise ValueError(
        f"Unknown backbone {backbone_name!r}; cannot determine token limit. "
        f"Expected 'roberta-base' or 'allenai/longformer-base-4096'."
    )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from src.utils.io import load_json

    parser = argparse.ArgumentParser(
        description="Select backbone for a prepared dataset per L2 quantile rule"
    )
    parser.add_argument(
        "--prep-dir",
        type=Path,
        required=True,
        help="Directory containing preparation_summary.json",
    )
    args = parser.parse_args()

    summary_path = args.prep_dir / "preparation_summary.json"
    summary = load_json(summary_path)
    wld = summary["word_length_distribution"]

    distribution = LengthDistribution(
        min=wld["min"],
        max=wld["max"],
        p50=wld["p50"],
        p80=wld["p80"],
        p95=wld["p95"],
        p99=wld["p99"],
    )

    selected = select_backbone(distribution)
    limit = backbone_token_limit(selected)

    print(f"Backbone selection for {args.prep_dir.name}:")
    print(f"  Word-length p80:    {distribution.p80:.1f}")
    print(f"  Threshold:           {WORD_QUANTILE_THRESHOLD}")
    print(f"  Selected backbone:   {selected}")
    print(f"  Token limit:         {limit}")
