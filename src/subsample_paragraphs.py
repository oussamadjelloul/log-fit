"""Stratified subsample of paragraphs.pkl to paper L21 budgets — Component 1.6.

Reads paragraphs.pkl, samples down to (max_normal, max_anomaly) with
STRATIFIED RANDOM sampling (separate per-label RNGs), writes a new
paragraphs.pkl plus subsample_summary.json.

PARADIGM:
- Stratified: normals and anomalies sampled INDEPENDENTLY (each with its
  own RNG). Changing `max_normal` does NOT change which anomalies are kept
  (and vice versa).
- If input has fewer paragraphs than the cap for a label, all are passed
  through unchanged for that label.
- Order preservation: output keeps the original list ordering of retained
  paragraphs (relevant for BGL/TB which are time-sorted).

WHEN TO USE:
- Full datasets (HDFS ~558k normals, BGL/TB millions of windows) need to
  match paper §IV-A L21 allocation (25k normal + 2k anomaly) before splits.
- Subsets where the prep happened to overshoot L21 (e.g., your current TB
  subset with 77k normals + 45k anomalies).

WHEN NOT TO USE:
- Inputs already at or below L21 caps for both labels (e.g., HDFS subset
  with 14.9k normals + 698 anomalies): the script will run but is a no-op.

Reference: logfit-repro-spec-v1.5.md Component 1.6 (NEW v1.5.1); paper
§IV-A L21.
"""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from src.types import Paragraph


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_NORMAL = 25_000   # Paper §IV-A L21
DEFAULT_MAX_ANOMALY = 2_000   # Paper §IV-A L21
DEFAULT_SEED = 42

# Per-label seed offsets so the two sampling streams are decoupled.
# Same principle as splits.py D_NEW6: changing one cap MUST NOT shift the
# other label's retained IDs for fixed (seed, input).
NORMAL_SEED_OFFSET = 0
ANOMALY_SEED_OFFSET = 1_000


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SubsampleSummary:
    """Persisted record of the subsampling operation."""

    input_path: str
    output_path: str
    seed: int
    max_normal: int
    max_anomaly: int
    input_normal: int
    input_anomaly: int
    output_normal: int
    output_anomaly: int
    normal_was_capped: bool
    anomaly_was_capped: bool


# ---------------------------------------------------------------------------
# Subsampling core
# ---------------------------------------------------------------------------


def subsample_paragraphs(
    paragraphs: list[Paragraph],
    max_normal: int = DEFAULT_MAX_NORMAL,
    max_anomaly: int = DEFAULT_MAX_ANOMALY,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Paragraph], SubsampleSummary]:
    """Stratified random subsample of paragraphs.

    Args:
        paragraphs: full list from prep output (any order).
        max_normal: cap on retained normals. If input has fewer normals,
            all are retained.
        max_anomaly: cap on retained anomalies. Same passthrough rule.
        seed: base RNG seed. Normal stream uses `seed + NORMAL_SEED_OFFSET`,
            anomaly stream uses `seed + ANOMALY_SEED_OFFSET`.

    Returns:
        (output_paragraphs, summary). Output preserves the input list order
        of retained paragraphs.

    Raises:
        ValueError: if max_normal or max_anomaly is negative.
    """
    if max_normal < 0 or max_anomaly < 0:
        raise ValueError(
            f"caps must be non-negative; "
            f"got max_normal={max_normal}, max_anomaly={max_anomaly}"
        )

    normal_indices = [i for i, p in enumerate(paragraphs) if p.label == 0]
    anomaly_indices = [i for i, p in enumerate(paragraphs) if p.label == 1]

    input_normal = len(normal_indices)
    input_anomaly = len(anomaly_indices)

    normal_was_capped = input_normal > max_normal
    anomaly_was_capped = input_anomaly > max_anomaly

    # Normal stream
    if normal_was_capped:
        rng_n = random.Random(seed + NORMAL_SEED_OFFSET)
        normal_indices_kept = rng_n.sample(normal_indices, max_normal)
    else:
        normal_indices_kept = normal_indices

    # Anomaly stream (decoupled RNG)
    if anomaly_was_capped:
        rng_a = random.Random(seed + ANOMALY_SEED_OFFSET)
        anomaly_indices_kept = rng_a.sample(anomaly_indices, max_anomaly)
    else:
        anomaly_indices_kept = anomaly_indices

    indices_to_keep = set(normal_indices_kept) | set(anomaly_indices_kept)

    # Preserve input order for retained paragraphs.
    output_paragraphs = [
        p for i, p in enumerate(paragraphs) if i in indices_to_keep
    ]

    summary = SubsampleSummary(
        input_path="",   # filled in by main()
        output_path="",
        seed=seed,
        max_normal=max_normal,
        max_anomaly=max_anomaly,
        input_normal=input_normal,
        input_anomaly=input_anomaly,
        output_normal=min(input_normal, max_normal),
        output_anomaly=min(input_anomaly, max_anomaly),
        normal_was_capped=normal_was_capped,
        anomaly_was_capped=anomaly_was_capped,
    )

    return output_paragraphs, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Stratified subsample of paragraphs.pkl to paper §IV-A L21 "
            "budgets (default 25k normal + 2k anomaly)."
        )
    )
    parser.add_argument(
        "--input-pkl",
        type=Path,
        required=True,
        help="Path to source paragraphs.pkl (from prep output)",
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        required=True,
        help="Path to write subsampled paragraphs.pkl",
    )
    parser.add_argument(
        "--max-normal",
        type=int,
        default=DEFAULT_MAX_NORMAL,
        help=f"Cap on normal paragraphs (default {DEFAULT_MAX_NORMAL})",
    )
    parser.add_argument(
        "--max-anomaly",
        type=int,
        default=DEFAULT_MAX_ANOMALY,
        help=f"Cap on anomaly paragraphs (default {DEFAULT_MAX_ANOMALY})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    with args.input_pkl.open("rb") as f:
        paragraphs = pickle.load(f)

    out_paragraphs, summary = subsample_paragraphs(
        paragraphs,
        max_normal=args.max_normal,
        max_anomaly=args.max_anomaly,
        seed=args.seed,
    )

    summary.input_path = str(args.input_pkl)
    summary.output_path = str(args.output_pkl)

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as f:
        pickle.dump(out_paragraphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary_path = args.output_pkl.parent / "subsample_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    print(f"\nSubsample complete (paper L21 caps).")
    print(
        f"  Input:  normal={summary.input_normal:,}, "
        f"anomaly={summary.input_anomaly:,}"
    )
    print(
        f"  Caps:   normal={args.max_normal:,}, "
        f"anomaly={args.max_anomaly:,}"
    )
    print(
        f"  Output: normal={summary.output_normal:,}, "
        f"anomaly={summary.output_anomaly:,}"
    )
    print(f"  Normal capped:  {summary.normal_was_capped}")
    print(f"  Anomaly capped: {summary.anomaly_was_capped}")
    print(f"  Wrote {args.output_pkl}")
    print(f"  Wrote {summary_path}")


if __name__ == "__main__":
    main()
