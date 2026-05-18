"""5-fold cross-validation splits per spec D14.

Per D14:
- 5-fold cross-validation with stratified test folds (preserving the
  normal/anomaly ratio across folds).
- Per-fold training-set sampling from the 4-fold train pool with
  fold-specific seeds → cross-fold anomaly overlap, E[overlap] ~ 500 at
  HDFS scale (pool size ~ 8K anomalies).
- Per-fold budget: train_normal_per_fold + train_anomaly_per_fold
  (default 25,000 + 2,000 per L21). If the train pool for a fold is
  smaller than the budget, use the full pool (graceful degradation for
  small datasets like BGL subsets).

Test sets are NOT sampled: each paragraph appears in exactly one test fold.
This makes per-fold test metrics comparable and supports standard 5-fold CV
F1/AUROC aggregation.

Output: splits.json with one record per fold containing four ID lists.

Reference: logfit-repro-spec-v1.3.md D14; logfit-repro-decisions-v1.4.md
D_NEW3 (Phase 1 paper-faithful).
"""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.types import Paragraph


DEFAULT_N_FOLDS = 5
DEFAULT_TRAIN_NORMAL_PER_FOLD = 25_000
DEFAULT_TRAIN_ANOMALY_PER_FOLD = 2_000
DEFAULT_SEED = 42


@dataclass
class FoldSplit:
    """One fold of a k-fold cross-validation split."""

    fold_id: int  # 0-indexed
    train_normal_ids: list[str] = field(default_factory=list)
    train_anomaly_ids: list[str] = field(default_factory=list)
    test_normal_ids: list[str] = field(default_factory=list)
    test_anomaly_ids: list[str] = field(default_factory=list)
    seed: int = DEFAULT_SEED

    def train_total(self) -> int:
        return len(self.train_normal_ids) + len(self.train_anomaly_ids)

    def test_total(self) -> int:
        return len(self.test_normal_ids) + len(self.test_anomaly_ids)

    def train_anomaly_rate(self) -> float:
        total = self.train_total()
        return len(self.train_anomaly_ids) / total if total > 0 else 0.0

    def test_anomaly_rate(self) -> float:
        total = self.test_total()
        return len(self.test_anomaly_ids) / total if total > 0 else 0.0


@dataclass
class SplitsArtifact:
    """Persistable record of all folds plus the config that produced them."""

    n_folds: int
    train_normal_per_fold: int
    train_anomaly_per_fold: int
    seed: int
    total_paragraphs: int
    total_normal: int
    total_anomaly: int
    folds: list[FoldSplit] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stratified fold assignment
# ---------------------------------------------------------------------------


def stratified_fold_assignment(
    paragraphs: list[Paragraph],
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """Return a list of length len(paragraphs) where each entry is the
    fold index (0..n_folds-1) for the corresponding paragraph.

    The assignment is stratified by `label`: normal and anomaly paragraphs
    are independently shuffled (using `seed`) and round-robin assigned to
    folds, so each fold has approximately the same normal/anomaly ratio.

    Deterministic given the same seed and the same paragraph order.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if not paragraphs:
        return []

    normal_idxs = [i for i, p in enumerate(paragraphs) if p.label == 0]
    anomaly_idxs = [i for i, p in enumerate(paragraphs) if p.label == 1]

    rng = random.Random(seed)
    rng.shuffle(normal_idxs)
    rng.shuffle(anomaly_idxs)

    assignment = [-1] * len(paragraphs)
    for slot, idx in enumerate(normal_idxs):
        assignment[idx] = slot % n_folds
    for slot, idx in enumerate(anomaly_idxs):
        assignment[idx] = slot % n_folds

    # Sanity check: no -1 left
    if -1 in assignment:
        missing = sum(1 for a in assignment if a == -1)
        raise ValueError(
            f"BUG: {missing} paragraphs not assigned to any fold. "
            f"Check that all paragraphs have label in {{0, 1}}."
        )

    return assignment


# ---------------------------------------------------------------------------
# Per-fold sampling
# ---------------------------------------------------------------------------


def _sample_or_full(
    items: list[Paragraph],
    n: int,
    seed: int,
) -> list[Paragraph]:
    """Return up to `n` items sampled deterministically from `items`.

    If `len(items) <= n`, returns all items unchanged (no sampling).
    Otherwise samples without replacement using a seeded RNG.
    """
    if len(items) <= n:
        return items
    rng = random.Random(seed)
    return rng.sample(items, n)


# ---------------------------------------------------------------------------
# Splits construction
# ---------------------------------------------------------------------------


def create_splits(
    paragraphs: list[Paragraph],
    n_folds: int = DEFAULT_N_FOLDS,
    train_normal_per_fold: int = DEFAULT_TRAIN_NORMAL_PER_FOLD,
    train_anomaly_per_fold: int = DEFAULT_TRAIN_ANOMALY_PER_FOLD,
    seed: int = DEFAULT_SEED,
) -> SplitsArtifact:
    """Build n_folds stratified train/test splits.

    Algorithm:
    1. Stratified fold assignment: each paragraph belongs to exactly one
       test fold (preserving anomaly rate per fold).
    2. For each fold k:
       - test_set := paragraphs assigned to fold k (full set, no sampling)
       - train_pool := paragraphs assigned to any other fold
       - train_normal := sample up to `train_normal_per_fold` from the
         normal subset of train_pool, with seed = seed + k
       - train_anomaly := sample up to `train_anomaly_per_fold` from the
         anomaly subset of train_pool, with seed = seed + k + 1000

    The per-fold seeds for normal vs anomaly sampling are offset (`k` vs
    `k + 1000`) so the anomaly stream is not entangled with the normal
    stream when one of them gets re-budgeted in a future config change.
    """
    if not paragraphs:
        raise ValueError("paragraphs is empty")

    assignment = stratified_fold_assignment(paragraphs, n_folds, seed)

    # Pre-group paragraphs by fold and label for efficient lookup
    paragraphs_by_fold: list[list[Paragraph]] = [[] for _ in range(n_folds)]
    for i, p in enumerate(paragraphs):
        paragraphs_by_fold[assignment[i]].append(p)

    folds: list[FoldSplit] = []
    for k in range(n_folds):
        test_set = paragraphs_by_fold[k]
        test_normal = [p for p in test_set if p.label == 0]
        test_anomaly = [p for p in test_set if p.label == 1]

        train_pool: list[Paragraph] = []
        for other_k in range(n_folds):
            if other_k == k:
                continue
            train_pool.extend(paragraphs_by_fold[other_k])

        train_normal_pool = [p for p in train_pool if p.label == 0]
        train_anomaly_pool = [p for p in train_pool if p.label == 1]

        # D14: per-fold seed offset for the anomaly stream creates the
        # cross-fold anomaly overlap pattern (E[overlap] ~ 500 at HDFS scale).
        train_normal_sample = _sample_or_full(
            train_normal_pool, train_normal_per_fold, seed=seed + k
        )
        train_anomaly_sample = _sample_or_full(
            train_anomaly_pool, train_anomaly_per_fold, seed=seed + k + 1000
        )

        folds.append(
            FoldSplit(
                fold_id=k,
                train_normal_ids=[p.paragraph_id for p in train_normal_sample],
                train_anomaly_ids=[p.paragraph_id for p in train_anomaly_sample],
                test_normal_ids=[p.paragraph_id for p in test_normal],
                test_anomaly_ids=[p.paragraph_id for p in test_anomaly],
                seed=seed + k,
            )
        )

    return SplitsArtifact(
        n_folds=n_folds,
        train_normal_per_fold=train_normal_per_fold,
        train_anomaly_per_fold=train_anomaly_per_fold,
        seed=seed,
        total_paragraphs=len(paragraphs),
        total_normal=sum(1 for p in paragraphs if p.label == 0),
        total_anomaly=sum(1 for p in paragraphs if p.label == 1),
        folds=folds,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_splits(artifact: SplitsArtifact, path: Path | str) -> None:
    """Write the SplitsArtifact to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(artifact), f, indent=2)


def load_splits(path: Path | str) -> SplitsArtifact:
    """Load a SplitsArtifact from JSON. Reconstructs FoldSplit dataclasses."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    folds = [FoldSplit(**fold_data) for fold_data in data.pop("folds", [])]
    return SplitsArtifact(folds=folds, **data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build 5-fold stratified CV splits per spec D14."
    )
    parser.add_argument(
        "--paragraphs-pkl",
        type=Path,
        required=True,
        help="Path to paragraphs.pkl from prep output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write splits.json (default: alongside paragraphs.pkl)",
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument(
        "--train-normal-per-fold",
        type=int,
        default=DEFAULT_TRAIN_NORMAL_PER_FOLD,
    )
    parser.add_argument(
        "--train-anomaly-per-fold",
        type=int,
        default=DEFAULT_TRAIN_ANOMALY_PER_FOLD,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    with args.paragraphs_pkl.open("rb") as f:
        paragraphs = pickle.load(f)

    artifact = create_splits(
        paragraphs,
        n_folds=args.n_folds,
        train_normal_per_fold=args.train_normal_per_fold,
        train_anomaly_per_fold=args.train_anomaly_per_fold,
        seed=args.seed,
    )

    output_path = (
        args.output
        if args.output is not None
        else args.paragraphs_pkl.parent / "splits.json"
    )
    save_splits(artifact, output_path)

    print(f"\nSplits created ({artifact.n_folds}-fold CV).")
    print(
        f"  Total paragraphs: {artifact.total_paragraphs:,} "
        f"(normal={artifact.total_normal:,}, "
        f"anomaly={artifact.total_anomaly:,})"
    )
    for fold in artifact.folds:
        print(
            f"  Fold {fold.fold_id}: "
            f"train={fold.train_total():,} "
            f"(N={len(fold.train_normal_ids):,}, "
            f"A={len(fold.train_anomaly_ids):,}, "
            f"rate={fold.train_anomaly_rate()*100:.2f}%) / "
            f"test={fold.test_total():,} "
            f"(N={len(fold.test_normal_ids):,}, "
            f"A={len(fold.test_anomaly_ids):,}, "
            f"rate={fold.test_anomaly_rate()*100:.2f}%)"
        )
    print(f"  Wrote {output_path}")


if __name__ == "__main__":
    main()
