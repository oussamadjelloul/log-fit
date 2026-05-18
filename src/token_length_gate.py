"""Token-length validation gate per spec Component 1.6.

Tokenizes paragraphs through the chosen backbone's tokenizer and validates
that the resulting token-length distribution is compatible with the
backbone's context window.

Decisions implemented:
- [L1, L2]   Backbone limits: roberta-base = 512, longformer-base-4096 = 4096.
- [§1.6]     HARD FAIL when `max > 5 × backbone_limit` — right-truncation
             discards more than 80% of paragraph content above this point.
- [§1.6]     SUPERVISOR_FLAG when `p95 > backbone_limit` — more than 5% of
             paragraphs need truncation; disclose rather than silently accept.
- [Component 3.1]  Per-line tokenization with </s> after each line, matching
                    the training-time masking pipeline. Token count =
                    1 (<s>) + sum_per_line(len(line_tokens) + 1).
- [§1.6 v1.2 / reviewer F11]  Longformer attention-window padding check is a
                               training-smoke-test concern, NOT a gate concern.
                               Deferred to src/train.py.

Reference: logfit-repro-spec-v1.3.md Component 1.6.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

from src.types import (
    BackboneName,
    LengthDistribution,
    Paragraph,
    TokenLengthSummary,
)
from src.utils.io import load_json, save_json
from src.utils.stats import compute_length_distribution

# Backbone context-window limits per L1, L2
ROBERTA_LIMIT = 512
LONGFORMER_LIMIT = 4096

# [§1.6] HARD FAIL when max exceeds this multiplier of backbone_limit.
HARD_FAIL_MULTIPLIER = 5


# ---------------------------------------------------------------------------
# Pure logic (no tokenizer required — fully testable offline)
# ---------------------------------------------------------------------------


def get_backbone_limit(backbone_name: str) -> int:
    """Return the token-length limit for a backbone.

    Recognizes 'roberta' or 'longformer' substrings in the backbone name.
    """
    name = backbone_name.lower()
    if "longformer" in name:
        return LONGFORMER_LIMIT
    if "roberta" in name:
        return ROBERTA_LIMIT
    raise ValueError(
        f"Unknown backbone {backbone_name!r}. Expected 'roberta' or "
        f"'longformer' substring."
    )


def _build_summary_from_token_counts(
    token_counts: list[int],
    backbone_name: str,
    raise_on_hard_fail: bool = True,
) -> TokenLengthSummary:
    """Pure-logic core: turn a list of per-paragraph token counts into a
    validated TokenLengthSummary, applying [§1.6] thresholds.

    Separated so tests can verify the gate behavior without loading a
    transformers tokenizer (which requires a model download or local cache).
    """
    backbone_limit = get_backbone_limit(backbone_name)
    hard_fail_threshold = HARD_FAIL_MULTIPLIER * backbone_limit

    distribution = compute_length_distribution(token_counts)

    if token_counts:
        truncation_rate = (
            sum(1 for tc in token_counts if tc > backbone_limit)
            / len(token_counts)
        )
    else:
        truncation_rate = 0.0

    summary = TokenLengthSummary(
        tokenizer=backbone_name,
        distribution=distribution,
        truncation_rate_at_backbone_limit=truncation_rate,
    )

    # [§1.6] HARD FAIL check
    if distribution.max > hard_fail_threshold:
        msg = (
            f"HARD FAIL: max token length {distribution.max:,} > "
            f"{HARD_FAIL_MULTIPLIER} × {backbone_limit} = "
            f"{hard_fail_threshold:,} (backbone: {backbone_name}). "
            f"Right-truncation would discard > "
            f"{(1 - 1 / HARD_FAIL_MULTIPLIER) * 100:.0f}% of paragraph "
            f"content. Per decisions-v1.3 §1.6."
        )
        if raise_on_hard_fail:
            raise ValueError(msg)
        print(f"WARNING (not raised, raise_on_hard_fail=False): {msg}")

    # [§1.6] SUPERVISOR_FLAG check
    if distribution.p95 > backbone_limit:
        print(
            f"SUPERVISOR_FLAG: p95 token length {distribution.p95:,.0f} > "
            f"backbone limit {backbone_limit} ({backbone_name}). "
            f"Truncation rate at backbone limit: "
            f"{truncation_rate * 100:.2f}% of paragraphs. "
            f"See decisions-v1.3 §1.6."
        )

    return summary


# ---------------------------------------------------------------------------
# Tokenization (requires transformers)
# ---------------------------------------------------------------------------


def tokenize_paragraph_token_counts(
    paragraphs: list[Paragraph],
    backbone_name: str,
) -> list[int]:
    """For each paragraph, compute its token length under spec Component 3.1
    tokenization: `<s>` + (line_tokens + `</s>`) per line.

    Uses the fast tokenizer in batch mode for speed; ~100x faster than
    per-line single-call tokenization on the same paragraph.

    Note: this requires `transformers` to be importable AND the tokenizer to
    be either pre-downloaded to the HF cache or accessible via network. On
    Narval, login nodes have network; compute nodes typically don't, so
    pre-download in a login-node session before submitting SLURM jobs.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers is required to run the token-length gate. "
            "Install via `pip install transformers` or use an env that has it."
        ) from e

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            backbone_name, use_fast=True
        )
    except OSError as e:
        raise OSError(
            f"Could not load tokenizer for {backbone_name!r}. On Narval, "
            f"download once on a login node (`python -c \"from transformers "
            f"import AutoTokenizer; AutoTokenizer.from_pretrained("
            f"'{backbone_name}')\"`) and confirm TRANSFORMERS_CACHE is set "
            f"consistently across login and compute nodes."
        ) from e

    token_counts: list[int] = []
    for p in paragraphs:
        if not p.lines:
            token_counts.append(2)  # just <s> </s>
            continue

        encodings = tokenizer(
            p.lines,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # 1 for opening <s>, then for each line: len(tokens) + 1 for closing </s>
        total = 1 + sum(
            len(line_ids) + 1 for line_ids in encodings["input_ids"]
        )
        token_counts.append(total)

    return token_counts


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def validate_token_lengths(
    paragraphs: list[Paragraph],
    backbone_name: str,
    raise_on_hard_fail: bool = True,
) -> TokenLengthSummary:
    """Run the §1.6 gate on a list of paragraphs in memory.

    Tokenizes via the backbone's fast tokenizer, computes the per-paragraph
    token-length distribution, applies HARD FAIL and SUPERVISOR_FLAG
    thresholds, and returns the validated summary.
    """
    token_counts = tokenize_paragraph_token_counts(paragraphs, backbone_name)
    return _build_summary_from_token_counts(
        token_counts, backbone_name, raise_on_hard_fail=raise_on_hard_fail
    )


def run_gate(
    paragraphs_pkl_path: Path | str,
    backbone_name: str,
    update_preparation_summary: bool = True,
    raise_on_hard_fail: bool = True,
) -> TokenLengthSummary:
    """Load paragraphs.pkl, run the gate, and (optionally) update the
    sibling preparation_summary.json with the new token_length_distribution
    field.

    If preparation_summary.json doesn't exist next to paragraphs.pkl, a
    standalone token_length_summary.json is written instead.
    """
    paragraphs_pkl_path = Path(paragraphs_pkl_path)
    with paragraphs_pkl_path.open("rb") as f:
        paragraphs = pickle.load(f)

    summary = validate_token_lengths(
        paragraphs, backbone_name, raise_on_hard_fail=raise_on_hard_fail
    )

    if update_preparation_summary:
        out_dir = paragraphs_pkl_path.parent
        prep_summary_path = out_dir / "preparation_summary.json"
        if prep_summary_path.exists():
            data = load_json(prep_summary_path)
            data["token_length_distribution"] = asdict(summary)
            save_json(data, prep_summary_path)
            print(f"Updated {prep_summary_path}")
        else:
            standalone_path = out_dir / "token_length_summary.json"
            save_json(summary, standalone_path)
            print(f"Wrote standalone {standalone_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Token-length validation gate per spec §1.6 (v1.3)"
    )
    parser.add_argument(
        "--paragraphs-pkl",
        type=Path,
        required=True,
        help="Path to paragraphs.pkl from prep output",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="Backbone name (e.g., 'roberta-base' or 'allenai/longformer-base-4096')",
    )
    parser.add_argument(
        "--no-update-summary",
        action="store_true",
        help="Do not update sibling preparation_summary.json (write standalone instead)",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Report HARD FAIL as a warning instead of raising. Useful for exploratory runs.",
    )
    args = parser.parse_args()

    summary = run_gate(
        paragraphs_pkl_path=args.paragraphs_pkl,
        backbone_name=args.backbone,
        update_preparation_summary=not args.no_update_summary,
        raise_on_hard_fail=not args.warn_only,
    )

    d = summary.distribution
    print()
    print(f"Token-length gate complete (backbone: {summary.tokenizer}).")
    print(f"  Token length min/p50/p80/p95/p99/max:")
    print(
        f"    {d.min:,} / {d.p50:,.0f} / {d.p80:,.0f} / "
        f"{d.p95:,.0f} / {d.p99:,.0f} / {d.max:,}"
    )
    print(
        f"  Truncation rate at backbone limit: "
        f"{summary.truncation_rate_at_backbone_limit * 100:.2f}%"
    )
