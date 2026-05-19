"""5-fold cross-validation splits — paper-faithful (v1.5 D_NEW6).

PARADIGM (v1.5 — replaces v1.4 Paradigm A):
- NORMALS partitioned across N folds via stratified shuffle + round-robin.
- ANOMALIES kept as a SINGLE pool, non-partitioned. Each fold samples
  independently from the same pool with per-fold seeds → cross-fold anomaly
  overlap E[overlap] ≈ 500 = 50% at the 2k pool / 1k budget scale (matches
  D14 documentation, which v1.4's code failed to deliver).
- Per-fold sampling uses four DISTINCT seed streams (offsets 0/2000/3000/
  4000) so changing one budget does NOT ripple through the others. Tests
  pin this invariant.

PER-FOLD SCHEMA (5 ID lists, was 4 in v1.4):
- train_normal_ids   ← 5,000 per L23 (was 25,000 default in v1.4)
- tune_normal_ids    ← 1,000 per L24 (NEW v1.5)
- tune_anomaly_ids   ← 1,000 per L24 (NEW v1.5)
- test_normal_ids    ← fold's normal partition (5,000 at L21 scale)
- test_anomaly_ids   ← 1,000 per L25 (was: fold's anomaly partition ~400)

REMOVED from v1.4: `train_anomaly_ids`. LogFiT trains on normals only;
the field was unused by train.py and misleading. Per D_NEW6.

CROSS-FOLD PROPERTIES (verified by tests):
- test_normal:   disjoint  (normal partition)
- train_normal:  partial overlap (E[overlap] ≈ 938 at HDFS scale)
- tune_normal:   partial overlap
- tune_anomaly:  per-fold-shuffled subset of pool → E[overlap] ≈ 500
- test_anomaly:  per-fold-shuffled subset of pool \\ tune_anomaly →
                 E[overlap] ≈ 500

WITHIN-FOLD DISJOINTNESS (enforced and tested):
- train_normal ⊥ tune_normal     (tune drawn from pool \\ train_ids)
- tune_anomaly ⊥ test_anomaly    (test drawn from pool \\ tune_ids)
- train_normal ⊥ test_normal     (different partitions)
- tune_normal ⊥ test_normal      (different partitions; tune from train_pool)

GRACEFUL DEGRADATION (small pools — HDFS subset case, D_NEW6.1):
PROPORTIONAL split. When pool < sum of budgets, BOTH streams shrink by
the budget ratio so neither is starved:
- Normals (pool < train + tune):
    effective_train = pool * train_budget // (train_budget + tune_budget)
    effective_tune  = pool - effective_train
- Anomalies (pool < tune + test):
    effective_tune = pool * tune_budget // (tune_budget + test_budget)
    effective_test = pool - effective_tune
Initial v1.5 used stream-priority (train-priority for normals, tune-
priority for anomalies); this left HDFS subset (698 anomalies < 1k tune
budget) with zero test_anomaly per fold. Changed to proportional in
v1.5.1 so both streams stay non-empty whenever both budgets > 0 and
pool > 1.

Reference: logfit-repro-decisions-v1.5.md D_NEW6; logfit-repro-spec-v1.5.md
Component 1.7.
"""

from __future__ import annotations

import json
import math
import pickle
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.types import Paragraph


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N_FOLDS = 5

# Paper §IV-A L23–L25 per-fold budgets at HDFS L21 allocation scale.
DEFAULT_TRAIN_NORMAL_PER_FOLD = 5_000   # L23
DEFAULT_TUNE_NORMAL_PER_FOLD = 1_000    # L24
DEFAULT_TUNE_ANOMALY_PER_FOLD = 1_000   # L24
DEFAULT_TEST_ANOMALY_PER_FOLD = 1_000   # L25
DEFAULT_SEED = 42

# Per-fold seed offsets keep the four sampling streams decoupled. Distinct
# offsets are an explicit design choice (D_NEW6): changing one budget must
# NOT shift another stream's IDs for fixed (seed, fold_id). Pinned by test.
TRAIN_NORMAL_SEED_OFFSET = 0
TUNE_NORMAL_SEED_OFFSET = 2_000
TUNE_ANOMALY_SEED_OFFSET = 3_000
TEST_ANOMALY_SEED_OFFSET = 4_000


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FoldSplit:
    """One fold of a k-fold cross-validation split (v1.5 schema, 5 ID lists)."""

    fold_id: int                                                 # 0-indexed
    seed: int = DEFAULT_SEED
    train_normal_ids: list[str] = field(default_factory=list)
    tune_normal_ids: list[str] = field(default_factory=list)
    tune_anomaly_ids: list[str] = field(default_factory=list)
    test_normal_ids: list[str] = field(default_factory=list)
    test_anomaly_ids: list[str] = field(default_factory=list)

    def train_total(self) -> int:
        return len(self.train_normal_ids)

    def tune_total(self) -> int:
        return len(self.tune_normal_ids) + len(self.tune_anomaly_ids)

    def test_total(self) -> int:
        return len(self.test_normal_ids) + len(self.test_anomaly_ids)

    def tune_anomaly_rate(self) -> float:
        total = self.tune_total()
        return len(self.tune_anomaly_ids) / total if total > 0 else 0.0

    def test_anomaly_rate(self) -> float:
        total = self.test_total()
        return len(self.test_anomaly_ids) / total if total > 0 else 0.0


@dataclass
class SplitsArtifact:
    """Persistable record of all folds plus the config that produced them."""

    n_folds: int
    train_normal_per_fold: int
    tune_normal_per_fold: int
    tune_anomaly_per_fold: int
    test_anomaly_per_fold: int
    seed: int
    total_paragraphs: int
    total_normal: int
    total_anomaly: int
    folds: list[FoldSplit] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stratified fold assignment — NORMALS ONLY (Paradigm B)
# ---------------------------------------------------------------------------


def stratified_normal_fold_assignment(
    paragraphs: list[Paragraph],
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
) -> dict[int, int]:
    """Return a dict mapping `paragraph-list-index -> fold_id` for NORMALS ONLY.

    Anomaly indices are NOT in the returned dict — the anomaly pool stays
    whole and is sampled per-fold downstream (Paradigm B / D14).

    Deterministic given the same seed and paragraph order.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if not paragraphs:
        return {}

    normal_idxs = [i for i, p in enumerate(paragraphs) if p.label == 0]

    rng = random.Random(seed)
    rng.shuffle(normal_idxs)

    assignment: dict[int, int] = {}
    for slot, idx in enumerate(normal_idxs):
        assignment[idx] = slot % n_folds
    return assignment


# ---------------------------------------------------------------------------
# Sampling primitives
# ---------------------------------------------------------------------------


def _shuffle_with_seed(items: list, seed: int) -> list:
    """Return a NEW list, shuffled deterministically. Doesn't mutate input."""
    out = list(items)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def _carve_train_and_tune_normals(
    pool: list[Paragraph],
    train_budget: int,
    tune_budget: int,
    seed: int,
) -> tuple[list[Paragraph], list[Paragraph]]:
    """Sample disjoint train and tune normal sets from a single normal pool.

    Seed-offset invariant (D_NEW6, holds in non-degraded regime): train uses
    `seed + TRAIN_NORMAL_SEED_OFFSET`, tune uses `seed + TUNE_NORMAL_SEED_OFFSET`.
    When `pool >= train_budget + tune_budget`, changing tune_budget MUST NOT
    change which IDs end up in train_sample for a given (seed, pool).

    Disjointness within fold: tune is drawn from a fresh shuffle of the pool,
    then filters out any IDs already in train_sample BEFORE taking its share.

    Graceful degradation (D_NEW6.1): when `pool < train + tune`, BOTH
    streams scale down by their budget ratio:
        effective_train = pool * train_budget // (train_budget + tune_budget)
        effective_tune  = pool - effective_train
    In the degraded regime the seed-offset invariant no longer holds — both
    sample sizes depend on both budgets — but the underlying shuffles still
    use independent seeds and the result is deterministic in
    (seed, pool, budgets).
    """
    if train_budget < 0 or tune_budget < 0:
        raise ValueError(
            f"budgets must be non-negative; "
            f"got train={train_budget}, tune={tune_budget}"
        )

    # Proportional degradation when pool < total budget (D_NEW6.1).
    pool_size = len(pool)
    effective_train = train_budget
    effective_tune = tune_budget
    total_budget = train_budget + tune_budget
    if pool_size < total_budget and total_budget > 0:
        effective_train = pool_size * train_budget // total_budget
        effective_tune = pool_size - effective_train

    # Train: shuffle with train-specific seed, take first effective_train.
    train_shuffled = _shuffle_with_seed(pool, seed + TRAIN_NORMAL_SEED_OFFSET)
    train_sample = train_shuffled[:effective_train]
    train_ids = {p.paragraph_id for p in train_sample}

    # Tune: shuffle independently, exclude train IDs, take first effective_tune.
    tune_shuffled = _shuffle_with_seed(pool, seed + TUNE_NORMAL_SEED_OFFSET)
    tune_candidates = [p for p in tune_shuffled if p.paragraph_id not in train_ids]
    tune_sample = tune_candidates[:effective_tune]

    return train_sample, tune_sample


def _carve_tune_and_test_anomalies(
    pool: list[Paragraph],
    tune_budget: int,
    test_budget: int,
    seed: int,
) -> tuple[list[Paragraph], list[Paragraph]]:
    """Sample disjoint tune_anomaly and test_anomaly sets from the anomaly pool.

    Pool is NOT partitioned — every fold samples from the same 2k pool with
    its own seed, producing cross-fold overlap E[overlap] ≈ K²/|pool|
    (e.g., 500 at the 2k pool / 1k budget scale per D14).

    Seed-offset invariant (D_NEW6, holds in non-degraded regime): tune uses
    TUNE_ANOMALY_SEED_OFFSET, test uses TEST_ANOMALY_SEED_OFFSET. When
    `pool >= tune + test`, changing one budget doesn't perturb the other.

    Disjointness within fold: test filters out IDs already in this fold's
    tune_anomaly.

    Graceful degradation (D_NEW6.1): when `pool < tune + test`, BOTH
    streams scale down by their budget ratio:
        effective_tune = pool * tune_budget // (tune_budget + test_budget)
        effective_test = pool - effective_tune
    In the degraded regime the seed-offset invariant no longer holds —
    both sample sizes depend on both budgets — but the underlying shuffles
    still use independent seeds and the result is deterministic in
    (seed, pool, budgets).
    """
    if tune_budget < 0 or test_budget < 0:
        raise ValueError(
            f"budgets must be non-negative; "
            f"got tune={tune_budget}, test={test_budget}"
        )

    # Proportional degradation when pool < total budget (D_NEW6.1).
    pool_size = len(pool)
    effective_tune = tune_budget
    effective_test = test_budget
    total_budget = tune_budget + test_budget
    if pool_size < total_budget and total_budget > 0:
        effective_tune = pool_size * tune_budget // total_budget
        effective_test = pool_size - effective_tune

    tune_shuffled = _shuffle_with_seed(pool, seed + TUNE_ANOMALY_SEED_OFFSET)
    tune_sample = tune_shuffled[:effective_tune]
    tune_ids = {p.paragraph_id for p in tune_sample}

    test_shuffled = _shuffle_with_seed(pool, seed + TEST_ANOMALY_SEED_OFFSET)
    test_candidates = [p for p in test_shuffled if p.paragraph_id not in tune_ids]
    test_sample = test_candidates[:effective_test]

    return tune_sample, test_sample


# ---------------------------------------------------------------------------
# Splits construction
# ---------------------------------------------------------------------------


def create_splits(
    paragraphs: list[Paragraph],
    n_folds: int = DEFAULT_N_FOLDS,
    train_normal_per_fold: int = DEFAULT_TRAIN_NORMAL_PER_FOLD,
    tune_normal_per_fold: int = DEFAULT_TUNE_NORMAL_PER_FOLD,
    tune_anomaly_per_fold: int = DEFAULT_TUNE_ANOMALY_PER_FOLD,
    test_anomaly_per_fold: int = DEFAULT_TEST_ANOMALY_PER_FOLD,
    seed: int = DEFAULT_SEED,
) -> SplitsArtifact:
    """Build n_folds stratified train/tune/test splits — Paradigm B (D_NEW6).

    Algorithm:
    1. Partition NORMALS into n_folds via stratified round-robin (seed).
    2. Anomalies stay as a single pool.
    3. For each fold k (with `fold_seed = seed + k`):
       - `test_normal := fold_k's normal partition` (deterministic, full set)
       - From `train_pool := union of other folds' normal partitions`:
           - sample `train_normal` (≤ train_budget) using TRAIN seed offset
           - sample `tune_normal`  (≤ tune_budget)  using TUNE_NORMAL offset,
             excluding train_normal IDs
       - From the anomaly pool:
           - sample `tune_anomaly` (≤ tune_budget) using TUNE_ANOMALY offset
           - sample `test_anomaly` (≤ test_budget) using TEST_ANOMALY offset,
             excluding tune_anomaly IDs

    The four distinct seed offsets keep streams independent (test:
    `test_tune_carve_does_not_perturb_train_carve`).
    """
    if not paragraphs:
        raise ValueError("paragraphs is empty")

    normal_assignment = stratified_normal_fold_assignment(paragraphs, n_folds, seed)

    normals_by_fold: list[list[Paragraph]] = [[] for _ in range(n_folds)]
    for i, p in enumerate(paragraphs):
        if p.label == 0:
            normals_by_fold[normal_assignment[i]].append(p)

    anomaly_pool: list[Paragraph] = [p for p in paragraphs if p.label == 1]

    folds: list[FoldSplit] = []
    for k in range(n_folds):
        fold_seed = seed + k

        # Normals
        test_normal_paragraphs = normals_by_fold[k]
        train_pool_normal: list[Paragraph] = []
        for other_k in range(n_folds):
            if other_k == k:
                continue
            train_pool_normal.extend(normals_by_fold[other_k])

        train_normal_sample, tune_normal_sample = _carve_train_and_tune_normals(
            pool=train_pool_normal,
            train_budget=train_normal_per_fold,
            tune_budget=tune_normal_per_fold,
            seed=fold_seed,
        )

        # Anomalies (non-partitioned pool, per-fold sampling)
        tune_anomaly_sample, test_anomaly_sample = _carve_tune_and_test_anomalies(
            pool=anomaly_pool,
            tune_budget=tune_anomaly_per_fold,
            test_budget=test_anomaly_per_fold,
            seed=fold_seed,
        )

        folds.append(
            FoldSplit(
                fold_id=k,
                seed=fold_seed,
                train_normal_ids=[p.paragraph_id for p in train_normal_sample],
                tune_normal_ids=[p.paragraph_id for p in tune_normal_sample],
                tune_anomaly_ids=[p.paragraph_id for p in tune_anomaly_sample],
                test_normal_ids=[p.paragraph_id for p in test_normal_paragraphs],
                test_anomaly_ids=[p.paragraph_id for p in test_anomaly_sample],
            )
        )

    return SplitsArtifact(
        n_folds=n_folds,
        train_normal_per_fold=train_normal_per_fold,
        tune_normal_per_fold=tune_normal_per_fold,
        tune_anomaly_per_fold=tune_anomaly_per_fold,
        test_anomaly_per_fold=test_anomaly_per_fold,
        seed=seed,
        total_paragraphs=len(paragraphs),
        total_normal=sum(1 for p in paragraphs if p.label == 0),
        total_anomaly=len(anomaly_pool),
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
        description="Build 5-fold stratified CV splits per paper §IV-A (v1.5 D_NEW6)."
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
        "--tune-normal-per-fold",
        type=int,
        default=DEFAULT_TUNE_NORMAL_PER_FOLD,
    )
    parser.add_argument(
        "--tune-anomaly-per-fold",
        type=int,
        default=DEFAULT_TUNE_ANOMALY_PER_FOLD,
    )
    parser.add_argument(
        "--test-anomaly-per-fold",
        type=int,
        default=DEFAULT_TEST_ANOMALY_PER_FOLD,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    with args.paragraphs_pkl.open("rb") as f:
        paragraphs = pickle.load(f)

    artifact = create_splits(
        paragraphs,
        n_folds=args.n_folds,
        train_normal_per_fold=args.train_normal_per_fold,
        tune_normal_per_fold=args.tune_normal_per_fold,
        tune_anomaly_per_fold=args.tune_anomaly_per_fold,
        test_anomaly_per_fold=args.test_anomaly_per_fold,
        seed=args.seed,
    )

    output_path = (
        args.output
        if args.output is not None
        else args.paragraphs_pkl.parent / "splits.json"
    )
    save_splits(artifact, output_path)

    print(f"\nSplits created ({artifact.n_folds}-fold CV, Paradigm B / v1.5).")
    print(
        f"  Total paragraphs: {artifact.total_paragraphs:,} "
        f"(normal={artifact.total_normal:,}, "
        f"anomaly={artifact.total_anomaly:,})"
    )
    print(
        f"  Per-fold budgets: train_normal={artifact.train_normal_per_fold:,}, "
        f"tune_normal={artifact.tune_normal_per_fold:,}, "
        f"tune_anomaly={artifact.tune_anomaly_per_fold:,}, "
        f"test_anomaly={artifact.test_anomaly_per_fold:,}"
    )
    for fold in artifact.folds:
        print(
            f"  Fold {fold.fold_id}: "
            f"train_N={len(fold.train_normal_ids):,}  "
            f"tune_N={len(fold.tune_normal_ids):,}  "
            f"tune_A={len(fold.tune_anomaly_ids):,}  "
            f"test_N={len(fold.test_normal_ids):,}  "
            f"test_A={len(fold.test_anomaly_ids):,}"
        )
    print(f"  Wrote {output_path}")


if __name__ == "__main__":
    main()
