#!/usr/bin/env python3
"""Diagnostic: paragraph length + vocabulary distribution per dataset.

Helps explain why MLM-based anomaly detection works well on some datasets
(e.g. TB) but not others (e.g. HDFS). Compares the normal vs anomaly
distributions in terms of paragraph length, vocabulary size, and the
lexical overlap between the two classes.

Hypothesis to test:
  - HDFS has high Jaccard(normal, anomaly) — both classes use the same
    ~30 log templates with different argument values, so MLM can't
    discriminate them by vocabulary alone.
  - TB has lower Jaccard — anomaly paragraphs contain tokens that don't
    appear in normals (kernel panics, segfaults, etc.), giving MLM a
    clear discriminative signal.

Usage:
    python scripts/diagnose_distribution.py \\
        --paragraphs data/processed/hdfs_subset/paragraphs.pkl \\
        --dataset-name HDFS

    python scripts/diagnose_distribution.py \\
        --paragraphs data/processed/tbird_30s_subset/paragraphs.pkl \\
        --dataset-name TB

Run once per dataset and compare the section [4] Jaccard + [5] anomaly-only
tokens side by side.

Outputs (all to stdout):
  [1] Paragraph counts by label
  [2] Lines per paragraph (mean, p50, p95)
  [3] Words per paragraph (mean, p50, p95)
  [4] Vocabulary stats — normal vocab, anomaly vocab, Jaccard
  [5] Top anomaly-only tokens — what signal anomalies provide
  [6] Random sample paragraphs from each class
"""

from __future__ import annotations

import argparse
import pickle
import random
import statistics
from collections import Counter
from pathlib import Path

# Required for pickle deserialization — pickle needs the class to be
# importable. Must be run from project root so `src` is on the import path.
from src.types import Paragraph  # noqa: F401  pickle dependency


def length_stats(values: list[int], label: str) -> None:
    if not values:
        print(f"  {label}: (empty)")
        return
    sorted_v = sorted(values)
    p50 = sorted_v[len(sorted_v) // 2]
    p95 = sorted_v[min(len(sorted_v) - 1, int(len(sorted_v) * 0.95))]
    print(
        f"  {label}: n={len(values):,}  mean={statistics.mean(values):.1f}  "
        f"p50={p50}  p95={p95}  min={min(values)}  max={max(values)}"
    )


def vocab(paragraphs) -> Counter:
    """Build a {token: count} Counter via whitespace tokenization."""
    v: Counter = Counter()
    for p in paragraphs:
        for line in p.lines:
            for word in line.split():
                v[word] += 1
    return v


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--paragraphs", type=Path, required=True,
        help="Path to paragraphs.pkl (output of prepare_*.py or subsample).",
    )
    parser.add_argument(
        "--dataset-name", type=str, required=True,
        help="Display name (e.g. 'HDFS', 'BGL', 'TB') — for output headers.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=2,
        help="Number of sample paragraphs to show per class (default 2).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for sample selection (deterministic; default 42).",
    )
    parser.add_argument(
        "--top-tokens", type=int, default=15,
        help="Number of anomaly-only tokens to list (default 15).",
    )
    args = parser.parse_args()

    with open(args.paragraphs, "rb") as f:
        paragraphs = pickle.load(f)

    print("=" * 72)
    print(f"DISTRIBUTION DIAGNOSTIC: {args.dataset_name}")
    print(f"Source: {args.paragraphs}")
    print("=" * 72)

    normal = [p for p in paragraphs if p.label == 0]
    anomaly = [p for p in paragraphs if p.label == 1]

    if not normal or not anomaly:
        print("\nERROR: both classes must be non-empty.")
        print(f"  Normals: {len(normal)}, Anomalies: {len(anomaly)}")
        return

    # ---- [1] counts ----
    print("\n[1] Paragraph counts")
    print(f"  Total:    {len(paragraphs):,}")
    print(
        f"  Normal:   {len(normal):,}  "
        f"({100 * len(normal) / len(paragraphs):.1f}%)"
    )
    print(
        f"  Anomaly:  {len(anomaly):,}  "
        f"({100 * len(anomaly) / len(paragraphs):.1f}%)"
    )

    # ---- [2] lines per paragraph ----
    print("\n[2] Lines per paragraph")
    length_stats([len(p.lines) for p in normal], "Normal ")
    length_stats([len(p.lines) for p in anomaly], "Anomaly")

    # ---- [3] words per paragraph ----
    print("\n[3] Words per paragraph (whitespace-split, summed across lines)")
    def word_count(p) -> int:
        return sum(len(line.split()) for line in p.lines)
    length_stats([word_count(p) for p in normal], "Normal ")
    length_stats([word_count(p) for p in anomaly], "Anomaly")

    # ---- [4] vocabulary stats ----
    print("\n[4] Vocabulary (whitespace tokens, exact match)")
    vocab_n = vocab(normal)
    vocab_a = vocab(anomaly)
    set_n, set_a = set(vocab_n), set(vocab_a)
    shared = set_n & set_a
    only_n = set_n - set_a
    only_a = set_a - set_n
    union = set_n | set_a
    jaccard = len(shared) / len(union) if union else 0.0

    print(f"  Normal vocab size:        {len(set_n):,}")
    print(f"  Anomaly vocab size:       {len(set_a):,}")
    print(f"  Shared tokens:            {len(shared):,}")
    print(f"  Normal-only tokens:       {len(only_n):,}")
    print(f"  Anomaly-only tokens:      {len(only_a):,}")
    print(
        f"  Jaccard(N, A):            {jaccard:.4f}   "
        f"(higher = more lexical overlap; 1.0 = identical vocabs)"
    )

    # ---- [5] top anomaly-only tokens ----
    if only_a:
        print(
            f"\n[5] Top {args.top_tokens} ANOMALY-ONLY tokens "
            "(potential anomaly markers):"
        )
        # Precompute per-paragraph token set for the coverage stat
        anomaly_token_sets = [
            set(word for line in p.lines for word in line.split())
            for p in anomaly
        ]
        anomaly_only_sorted = sorted(
            ((vocab_a[w], w) for w in only_a), reverse=True
        )
        print(
            f"  {'count':>10s}  {'%anom':>6s}  token"
        )
        for count, word in anomaly_only_sorted[: args.top_tokens]:
            pct_in_anomalies = (
                sum(1 for ts in anomaly_token_sets if word in ts)
                / len(anomaly) * 100
            )
            # Truncate word for display
            display_word = word if len(word) <= 60 else word[:57] + "..."
            print(f"  {count:>10,d}  {pct_in_anomalies:>5.1f}%  '{display_word}'")
    else:
        print(
            "\n[5] No anomaly-only tokens — anomaly vocabulary is a strict "
            "subset of normal vocabulary. MLM has no lexical signal for "
            "discriminating anomaly from normal."
        )

    # ---- [6] sample paragraphs ----
    rng = random.Random(args.seed)
    print(f"\n[6] Random sample paragraphs (seed={args.seed})")
    print(f"\n  --- NORMAL samples ({args.n_samples}) ---")
    for i, p in enumerate(rng.sample(normal, min(args.n_samples, len(normal)))):
        print(
            f"\n  [normal {i+1}] paragraph_id={p.paragraph_id}, "
            f"lines={len(p.lines)}"
        )
        for line in p.lines[:5]:
            print(f"    > {line[:150]}")
        if len(p.lines) > 5:
            print(f"    ... ({len(p.lines) - 5} more lines)")

    print(f"\n  --- ANOMALY samples ({args.n_samples}) ---")
    for i, p in enumerate(rng.sample(anomaly, min(args.n_samples, len(anomaly)))):
        print(
            f"\n  [anomaly {i+1}] paragraph_id={p.paragraph_id}, "
            f"lines={len(p.lines)}"
        )
        for line in p.lines[:5]:
            print(f"    > {line[:150]}")
        if len(p.lines) > 5:
            print(f"    ... ({len(p.lines) - 5} more lines)")

    print()


if __name__ == "__main__":
    main()
