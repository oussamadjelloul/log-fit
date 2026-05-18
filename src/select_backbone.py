"""Backbone selection per L2 quantile heuristic.

Spec L2 — pure function: given a LengthDistribution, compare its 0.8-quantile
to a threshold; if ≤ threshold, pick RoBERTa; else, pick Longformer.

The caller decides which distribution to pass:

- **Phase 1 (paper-faithful):** pass word-length distribution from
  `preparation_summary.json`. Matches paper §III.A heuristic exactly,
  including its word/token unit conflation (see v1.4 decisions M1).
- **Phase 2 (v1.4 corrected):** pass token-length distribution from
  `preparation_summary.json` (populated after token-length gate runs).
  Matches v1.4 D_NEW1 rule.

The function itself is identical for both modes — only the input distribution
differs. The threshold value (default 512) matches the RoBERTa context window.

Reference: logfit-repro-spec-v1.3.md §L2; logfit-repro-decisions-v1.4.md D_NEW1.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.token_length_gate import get_backbone_limit
from src.types import LengthDistribution

ROBERTA_BACKBONE = "roberta-base"
LONGFORMER_BACKBONE = "allenai/longformer-base-4096"

# [L2] Threshold for 0.8-quantile of the input distribution. Named
# WORD_QUANTILE_THRESHOLD for compatibility with v1.3 tests; under v1.4
# D_NEW1, the caller should pass a token-length distribution and the
# threshold value (512) is interpreted as a token count. The constant
# value is unchanged — only its semantic interpretation by the caller.
WORD_QUANTILE_THRESHOLD = 512


def select_backbone(
    distribution: LengthDistribution,
    quantile_threshold: int = WORD_QUANTILE_THRESHOLD,
) -> str:
    """Apply the L2 quantile heuristic.

    Returns:
        ROBERTA_BACKBONE  if `distribution.p80 <= quantile_threshold`
        LONGFORMER_BACKBONE  otherwise

    The boundary is inclusive at the threshold (p80 == 512 → RoBERTa).
    """
    if distribution.p80 <= quantile_threshold:
        return ROBERTA_BACKBONE
    return LONGFORMER_BACKBONE


def backbone_token_limit(backbone_name: str) -> int:
    """Return the token-length context window of a given backbone.

    Currently supports `roberta-base` (512) and
    `allenai/longformer-base-4096` (4096). Case-insensitive; matches by
    substring ('roberta' or 'longformer' anywhere in the name). Longformer
    takes precedence when both substrings are present (e.g.,
    'roberta-longformer').
    """
    try:
        return get_backbone_limit(backbone_name)
    except ValueError as e:
        # Re-raise with a test-friendly message that includes the canonical
        # backbone names so callers can fix the typo without consulting docs.
        raise ValueError(
            f"Unknown backbone {backbone_name!r}. Expected "
            f"'{ROBERTA_BACKBONE}' or '{LONGFORMER_BACKBONE}' "
            f"(or a substring match thereof)."
        ) from e


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------


@dataclass
class BackboneDecision:
    """Persistable record of a backbone-selection decision.

    Captures both the input distribution and the resulting choice so the
    decision is auditable downstream. The `distribution_kind` field
    documents whether word-length or token-length was used (Phase 1 vs
    Phase 2 of v1.4 D_NEW1).
    """

    chosen_backbone: str
    distribution_kind: str  # "word_length" or "token_length"
    distribution: LengthDistribution
    quantile_threshold: int
    rationale: str


def select_and_record(
    distribution: LengthDistribution,
    distribution_kind: str,
    quantile_threshold: int = WORD_QUANTILE_THRESHOLD,
) -> BackboneDecision:
    """Run selection and capture the full decision record."""
    chosen = select_backbone(distribution, quantile_threshold)
    rationale = (
        f"p80({distribution_kind}) = {distribution.p80:.1f} "
        f"{'<=' if distribution.p80 <= quantile_threshold else '>'} "
        f"threshold {quantile_threshold} → {chosen}"
    )
    return BackboneDecision(
        chosen_backbone=chosen,
        distribution_kind=distribution_kind,
        distribution=distribution,
        quantile_threshold=quantile_threshold,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_distribution_from_prep_summary(
    summary_path: Path,
    use_token_length: bool,
) -> tuple[LengthDistribution, str]:
    """Load either the word- or token-length distribution from a
    preparation_summary.json. Returns (distribution, distribution_kind)."""
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if use_token_length:
        tld = data.get("token_length_distribution")
        if tld is None:
            raise ValueError(
                f"{summary_path} has no token_length_distribution. "
                f"Run the §1.6 token-length gate first (see "
                f"src/token_length_gate.py)."
            )
        dist_data = tld["distribution"]
        return (
            LengthDistribution(
                min=int(dist_data["min"]),
                max=int(dist_data["max"]),
                p50=float(dist_data["p50"]),
                p80=float(dist_data["p80"]),
                p95=float(dist_data["p95"]),
                p99=float(dist_data["p99"]),
            ),
            "token_length",
        )
    else:
        wld = data.get("word_length_distribution")
        if wld is None:
            raise ValueError(
                f"{summary_path} has no word_length_distribution."
            )
        return (
            LengthDistribution(
                min=int(wld["min"]),
                max=int(wld["max"]),
                p50=float(wld["p50"]),
                p80=float(wld["p80"]),
                p95=float(wld["p95"]),
                p99=float(wld["p99"]),
            ),
            "word_length",
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Backbone selection via L2 quantile heuristic. "
            "Phase 1 paper-faithful: pass --word-length (paper's heuristic). "
            "Phase 2 v1.4 corrected: pass --token-length "
            "(after running the §1.6 gate)."
        )
    )
    parser.add_argument(
        "--preparation-summary",
        type=Path,
        required=True,
        help="Path to preparation_summary.json from prep output",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--word-length",
        action="store_true",
        help="Paper-faithful: use word_length_distribution (Phase 1 of D_NEW1)",
    )
    mode.add_argument(
        "--token-length",
        action="store_true",
        help="v1.4 corrected: use token_length_distribution (Phase 2 of D_NEW1)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=WORD_QUANTILE_THRESHOLD,
        help=f"L2 quantile threshold (default {WORD_QUANTILE_THRESHOLD})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write BackboneDecision JSON. "
             "Defaults to <preparation_summary_dir>/backbone_decision.json",
    )
    args = parser.parse_args()

    distribution, dist_kind = _load_distribution_from_prep_summary(
        args.preparation_summary,
        use_token_length=args.token_length,
    )
    decision = select_and_record(
        distribution,
        distribution_kind=dist_kind,
        quantile_threshold=args.threshold,
    )

    output_path = (
        args.output
        if args.output is not None
        else args.preparation_summary.parent / "backbone_decision.json"
    )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(decision), f, indent=2)

    print(f"\nBackbone selection ({dist_kind} mode):")
    print(f"  {decision.rationale}")
    print(f"  Token limit for chosen backbone: "
          f"{backbone_token_limit(decision.chosen_backbone)}")
    print(f"  Wrote {output_path}")


if __name__ == "__main__":
    main()
