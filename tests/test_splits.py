"""Tests for src/splits.py — 5-fold CV splits per paper §IV-A (v1.5 D_NEW6).

Paradigm B (v1.5 — paper-faithful, replaces v1.4 Paradigm A):
- NORMALS partitioned across folds.
- ANOMALIES kept as a single pool, sampled per-fold (with cross-fold overlap).
- 5 ID lists per fold: train_normal, tune_normal, tune_anomaly,
  test_normal, test_anomaly. `train_anomaly_ids` REMOVED — training is
  normals-only in LogFiT.

Mapping to spec-v1.5 Component 1.7 / decisions-v1.5 D_NEW6:
- test_normals_partitioned                 — Paradigm B normal stratification
- test_anomalies_pooled_not_partitioned    — Paradigm B anomaly pool
- test_cross_fold_anomaly_overlap_*        — D14 / E[overlap] ≈ K²/|pool|
- test_within_fold_disjoint_*              — invariant
- test_seed_offsets_independent_*          — D_NEW6 invariant (changing
                                              one budget MUST NOT shift
                                              another stream's IDs)
- test_graceful_degradation_*              — D_NEW6.1 proportional split
- test_hdfs_subset_regression_v1_5_1       — regression guard for the
                                              tune-priority bug (test_A=0)
- test_persistence_roundtrip               — JSON schema
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.splits import (
    DEFAULT_N_FOLDS,
    DEFAULT_SEED,
    DEFAULT_TEST_ANOMALY_PER_FOLD,
    DEFAULT_TRAIN_NORMAL_PER_FOLD,
    DEFAULT_TUNE_ANOMALY_PER_FOLD,
    DEFAULT_TUNE_NORMAL_PER_FOLD,
    TEST_ANOMALY_SEED_OFFSET,
    TRAIN_NORMAL_SEED_OFFSET,
    TUNE_ANOMALY_SEED_OFFSET,
    TUNE_NORMAL_SEED_OFFSET,
    FoldSplit,
    SplitsArtifact,
    _carve_train_and_tune_normals,
    _carve_tune_and_test_anomalies,
    _shuffle_with_seed,
    create_splits,
    load_splits,
    save_splits,
    stratified_normal_fold_assignment,
)
from src.types import Paragraph


def _make_paragraphs(n_normal: int, n_anomaly: int) -> list[Paragraph]:
    """Synthetic paragraph list. IDs: 'n_0' ... 'n_{N-1}', 'a_0' ... 'a_{M-1}'."""
    paragraphs = [
        Paragraph(paragraph_id=f"n_{i}", lines=["normal line"], label=0)
        for i in range(n_normal)
    ]
    paragraphs.extend(
        Paragraph(paragraph_id=f"a_{i}", lines=["anomaly line"], label=1)
        for i in range(n_anomaly)
    )
    return paragraphs


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_paper_faithful_defaults(self):
        """Defaults match LogFiT paper §IV-A L23–L25 per-fold budgets."""
        assert DEFAULT_TRAIN_NORMAL_PER_FOLD == 5_000   # L23
        assert DEFAULT_TUNE_NORMAL_PER_FOLD == 1_000    # L24
        assert DEFAULT_TUNE_ANOMALY_PER_FOLD == 1_000   # L24
        assert DEFAULT_TEST_ANOMALY_PER_FOLD == 1_000   # L25
        assert DEFAULT_N_FOLDS == 5
        assert DEFAULT_SEED == 42

    def test_seed_offsets_are_distinct(self):
        """Four streams must use four distinct offsets so changing one
        budget doesn't shift another's IDs (D_NEW6)."""
        offsets = {
            TRAIN_NORMAL_SEED_OFFSET,
            TUNE_NORMAL_SEED_OFFSET,
            TUNE_ANOMALY_SEED_OFFSET,
            TEST_ANOMALY_SEED_OFFSET,
        }
        assert len(offsets) == 4


# ---------------------------------------------------------------------------
# stratified_normal_fold_assignment
# ---------------------------------------------------------------------------


class TestStratifiedNormalFoldAssignment:
    def test_normals_only_in_assignment(self):
        """Anomaly indices NOT present in the assignment dict (Paradigm B)."""
        ps = _make_paragraphs(20, 5)
        asn = stratified_normal_fold_assignment(ps, n_folds=5)
        # 20 normals → 20 keys; the 5 anomaly indices (20..24) are absent
        assert len(asn) == 20
        anomaly_indices = {i for i, p in enumerate(ps) if p.label == 1}
        assert anomaly_indices.isdisjoint(asn.keys())

    def test_all_folds_in_valid_range(self):
        ps = _make_paragraphs(100, 20)
        asn = stratified_normal_fold_assignment(ps, n_folds=5)
        assert all(0 <= v < 5 for v in asn.values())

    def test_round_robin_partition_balanced(self):
        """Normals split roughly evenly across folds."""
        ps = _make_paragraphs(500, 100)
        asn = stratified_normal_fold_assignment(ps, n_folds=5)
        per_fold_counts = [0] * 5
        for fold_id in asn.values():
            per_fold_counts[fold_id] += 1
        # 500/5 = 100 each
        assert all(98 <= c <= 102 for c in per_fold_counts), per_fold_counts

    def test_deterministic_with_same_seed(self):
        ps = _make_paragraphs(100, 20)
        a1 = stratified_normal_fold_assignment(ps, n_folds=5, seed=42)
        a2 = stratified_normal_fold_assignment(ps, n_folds=5, seed=42)
        assert a1 == a2

    def test_different_seeds_give_different_assignments(self):
        ps = _make_paragraphs(100, 20)
        a1 = stratified_normal_fold_assignment(ps, n_folds=5, seed=42)
        a2 = stratified_normal_fold_assignment(ps, n_folds=5, seed=43)
        assert a1 != a2

    def test_empty_returns_empty_dict(self):
        assert stratified_normal_fold_assignment([], n_folds=5) == {}

    def test_n_folds_below_2_raises(self):
        ps = _make_paragraphs(10, 2)
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            stratified_normal_fold_assignment(ps, n_folds=1)

    def test_all_anomaly_input_yields_empty_assignment(self):
        """If no normals exist, no fold assignment (anomalies aren't partitioned)."""
        ps = _make_paragraphs(0, 20)
        asn = stratified_normal_fold_assignment(ps, n_folds=5)
        assert asn == {}


# ---------------------------------------------------------------------------
# _shuffle_with_seed
# ---------------------------------------------------------------------------


class TestShuffleWithSeed:
    def test_returns_new_list_doesnt_mutate(self):
        items = [1, 2, 3, 4, 5]
        original = list(items)
        _shuffle_with_seed(items, seed=42)
        assert items == original  # input untouched

    def test_deterministic_with_same_seed(self):
        items = list(range(20))
        s1 = _shuffle_with_seed(items, seed=42)
        s2 = _shuffle_with_seed(items, seed=42)
        assert s1 == s2

    def test_different_seeds_give_different_orders(self):
        items = list(range(20))
        s1 = _shuffle_with_seed(items, seed=42)
        s2 = _shuffle_with_seed(items, seed=43)
        assert s1 != s2

    def test_preserves_items(self):
        items = list(range(20))
        out = _shuffle_with_seed(items, seed=42)
        assert sorted(out) == items


# ---------------------------------------------------------------------------
# _carve_train_and_tune_normals
# ---------------------------------------------------------------------------


class TestCarveTrainAndTuneNormals:
    def test_train_and_tune_disjoint(self):
        pool = [Paragraph(f"x_{i}", ["a"], 0) for i in range(100)]
        train, tune = _carve_train_and_tune_normals(
            pool=pool, train_budget=40, tune_budget=20, seed=42
        )
        train_ids = {p.paragraph_id for p in train}
        tune_ids = {p.paragraph_id for p in tune}
        assert train_ids.isdisjoint(tune_ids)

    def test_budgets_respected_when_pool_sufficient(self):
        pool = [Paragraph(f"x_{i}", ["a"], 0) for i in range(100)]
        train, tune = _carve_train_and_tune_normals(
            pool=pool, train_budget=40, tune_budget=20, seed=42
        )
        assert len(train) == 40
        assert len(tune) == 20

    def test_graceful_degradation_proportional_normals(self):
        """D_NEW6.1: when pool < train + tune, scale BOTH proportionally.
        Pool=50, budgets 40+20 (ratio 2:1) → train=33, tune=17 (full pool used)."""
        pool = [Paragraph(f"x_{i}", ["a"], 0) for i in range(50)]
        train, tune = _carve_train_and_tune_normals(
            pool=pool, train_budget=40, tune_budget=20, seed=42
        )
        # 50 * 40 // 60 = 33; 50 - 33 = 17
        assert len(train) == 33
        assert len(tune) == 17
        # Full pool consumed
        assert len(train) + len(tune) == 50

    def test_pool_equals_train_budget_still_proportional(self):
        """Even when pool == train_budget exactly, proportional split
        still applies because pool < train + tune. Pool=40, budgets 40+20
        → train=26, tune=14 (not train=40, tune=0)."""
        pool = [Paragraph(f"x_{i}", ["a"], 0) for i in range(40)]
        train, tune = _carve_train_and_tune_normals(
            pool=pool, train_budget=40, tune_budget=20, seed=42
        )
        # 40 * 40 // 60 = 26; 40 - 26 = 14
        assert len(train) == 26
        assert len(tune) == 14

    def test_train_budget_exceeds_pool_proportional(self):
        """train_budget > pool size → proportional split still applies.
        Pool=30, budgets 40+20 (ratio 2:1) → train=20, tune=10."""
        pool = [Paragraph(f"x_{i}", ["a"], 0) for i in range(30)]
        train, tune = _carve_train_and_tune_normals(
            pool=pool, train_budget=40, tune_budget=20, seed=42
        )
        # 30 * 40 // 60 = 20; 30 - 20 = 10
        assert len(train) == 20
        assert len(tune) == 10
        assert len(train) + len(tune) == 30

    def test_zero_budgets_return_empty(self):
        pool = [Paragraph(f"x_{i}", ["a"], 0) for i in range(20)]
        train, tune = _carve_train_and_tune_normals(
            pool=pool, train_budget=0, tune_budget=0, seed=42
        )
        assert train == []
        assert tune == []

    def test_negative_budget_raises(self):
        pool = [Paragraph(f"x_{i}", ["a"], 0) for i in range(20)]
        with pytest.raises(ValueError, match="non-negative"):
            _carve_train_and_tune_normals(
                pool=pool, train_budget=-1, tune_budget=5, seed=42
            )

    def test_deterministic_with_same_seed(self):
        pool = [Paragraph(f"x_{i}", ["a"], 0) for i in range(100)]
        t1, u1 = _carve_train_and_tune_normals(pool, 30, 10, seed=42)
        t2, u2 = _carve_train_and_tune_normals(pool, 30, 10, seed=42)
        assert [p.paragraph_id for p in t1] == [p.paragraph_id for p in t2]
        assert [p.paragraph_id for p in u1] == [p.paragraph_id for p in u2]


# ---------------------------------------------------------------------------
# _carve_tune_and_test_anomalies
# ---------------------------------------------------------------------------


class TestCarveTuneAndTestAnomalies:
    def test_tune_and_test_disjoint(self):
        pool = [Paragraph(f"a_{i}", ["a"], 1) for i in range(100)]
        tune, test = _carve_tune_and_test_anomalies(
            pool=pool, tune_budget=30, test_budget=30, seed=42
        )
        tune_ids = {p.paragraph_id for p in tune}
        test_ids = {p.paragraph_id for p in test}
        assert tune_ids.isdisjoint(test_ids)

    def test_budgets_respected_when_pool_sufficient(self):
        pool = [Paragraph(f"a_{i}", ["a"], 1) for i in range(100)]
        tune, test = _carve_tune_and_test_anomalies(
            pool=pool, tune_budget=30, test_budget=40, seed=42
        )
        assert len(tune) == 30
        assert len(test) == 40

    def test_graceful_degradation_proportional_anomalies(self):
        """D_NEW6.1: when pool < tune + test, scale BOTH proportionally.
        Pool=40, budgets 30+30 (equal) → tune=20, test=20 (50/50 split)."""
        pool = [Paragraph(f"a_{i}", ["a"], 1) for i in range(40)]
        tune, test = _carve_tune_and_test_anomalies(
            pool=pool, tune_budget=30, test_budget=30, seed=42
        )
        # 40 * 30 // 60 = 20; 40 - 20 = 20
        assert len(tune) == 20
        assert len(test) == 20
        # Full pool used
        assert len(tune) + len(test) == 40

    def test_pool_equals_tune_budget_still_proportional(self):
        """Pool=30, budgets 30+30 → tune=15, test=15 (not tune=30, test=0)."""
        pool = [Paragraph(f"a_{i}", ["a"], 1) for i in range(30)]
        tune, test = _carve_tune_and_test_anomalies(
            pool=pool, tune_budget=30, test_budget=30, seed=42
        )
        # 30 * 30 // 60 = 15; 30 - 15 = 15
        assert len(tune) == 15
        assert len(test) == 15

    def test_zero_budgets_return_empty(self):
        pool = [Paragraph(f"a_{i}", ["a"], 1) for i in range(20)]
        tune, test = _carve_tune_and_test_anomalies(
            pool=pool, tune_budget=0, test_budget=0, seed=42
        )
        assert tune == []
        assert test == []

    def test_negative_budget_raises(self):
        pool = [Paragraph(f"a_{i}", ["a"], 1) for i in range(20)]
        with pytest.raises(ValueError, match="non-negative"):
            _carve_tune_and_test_anomalies(
                pool=pool, tune_budget=5, test_budget=-1, seed=42
            )


# ---------------------------------------------------------------------------
# create_splits — main API at HDFS-like scale
# ---------------------------------------------------------------------------


class TestCreateSplitsBasic:
    def test_returns_correct_number_of_folds(self):
        ps = _make_paragraphs(100, 20)
        art = create_splits(ps, n_folds=5)
        assert len(art.folds) == 5
        assert art.n_folds == 5

    def test_metadata_fields_populated(self):
        ps = _make_paragraphs(100, 20)
        art = create_splits(ps, n_folds=5, seed=42)
        assert art.total_paragraphs == 120
        assert art.total_normal == 100
        assert art.total_anomaly == 20
        assert art.seed == 42
        assert art.n_folds == 5
        # New v1.5 metadata
        assert art.tune_normal_per_fold == DEFAULT_TUNE_NORMAL_PER_FOLD
        assert art.tune_anomaly_per_fold == DEFAULT_TUNE_ANOMALY_PER_FOLD
        assert art.test_anomaly_per_fold == DEFAULT_TEST_ANOMALY_PER_FOLD

    def test_metadata_records_explicit_budgets(self):
        ps = _make_paragraphs(500, 100)
        art = create_splits(
            ps,
            n_folds=5,
            train_normal_per_fold=50,
            tune_normal_per_fold=17,
            tune_anomaly_per_fold=23,
            test_anomaly_per_fold=29,
        )
        assert art.train_normal_per_fold == 50
        assert art.tune_normal_per_fold == 17
        assert art.tune_anomaly_per_fold == 23
        assert art.test_anomaly_per_fold == 29

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            create_splits([])

    def test_no_train_anomaly_ids_field_v1_5(self):
        """v1.5 D_NEW6: train_anomaly_ids field removed — training is
        normals-only. Regression guard."""
        ps = _make_paragraphs(500, 100)
        art = create_splits(ps, n_folds=5)
        for fold in art.folds:
            assert not hasattr(fold, "train_anomaly_ids"), (
                f"Fold {fold.fold_id} has train_anomaly_ids — removed in v1.5"
            )

    def test_hdfs_scale_paper_faithful_budgets(self):
        """At L21 scale (25k normal + 2k anomaly) all 5 paper budgets satisfied
        exactly per fold."""
        ps = _make_paragraphs(25_000, 2_000)
        art = create_splits(ps)
        for fold in art.folds:
            assert len(fold.train_normal_ids) == 5_000
            assert len(fold.tune_normal_ids) == 1_000
            assert len(fold.tune_anomaly_ids) == 1_000
            assert len(fold.test_normal_ids) == 5_000
            assert len(fold.test_anomaly_ids) == 1_000


# ---------------------------------------------------------------------------
# Paradigm B properties — partition vs. pool
# ---------------------------------------------------------------------------


class TestParadigmB:
    def test_test_normals_partitioned_across_folds(self):
        """Each normal paragraph appears in exactly ONE fold's test_normal."""
        ps = _make_paragraphs(500, 100)
        art = create_splits(ps, n_folds=5)
        all_test_normal_ids: list[str] = []
        for fold in art.folds:
            all_test_normal_ids.extend(fold.test_normal_ids)
        # 500 unique normal IDs, each in exactly one fold's test_normal
        assert len(all_test_normal_ids) == 500
        assert len(set(all_test_normal_ids)) == 500

    def test_anomalies_pooled_not_partitioned(self):
        """Anomalies are NOT partitioned: total across folds' test_anomaly_ids
        > total anomaly count (because of cross-fold overlap)."""
        # Pool = 200 anomalies, budget = 100/fold * 5 folds = 500 samples
        ps = _make_paragraphs(2_500, 200)
        art = create_splits(
            ps,
            train_normal_per_fold=200,
            tune_normal_per_fold=50,
            tune_anomaly_per_fold=50,
            test_anomaly_per_fold=100,
        )
        # If pooled: 500 total samples drawn from 200-anomaly pool → many
        # appear in multiple folds.
        total_test_anomaly_samples = sum(
            len(f.test_anomaly_ids) for f in art.folds
        )
        unique_test_anomaly_ids = set()
        for f in art.folds:
            unique_test_anomaly_ids.update(f.test_anomaly_ids)
        # Total samples > unique IDs → overlap exists (Paradigm B signature).
        assert total_test_anomaly_samples > len(unique_test_anomaly_ids)
        # And all sampled IDs are from the pool (no fabrication).
        anomaly_pool_ids = {f"a_{i}" for i in range(200)}
        assert unique_test_anomaly_ids.issubset(anomaly_pool_ids)

    def test_cross_fold_anomaly_overlap_within_expected_range(self):
        """D14 / paper-faithful: E[overlap of test_anomaly between any 2 folds]
        ≈ K² / |pool|, where K = test_anomaly_per_fold."""
        # Pool 2000, budget 1000 per fold → E[overlap] = 1000² / 2000 = 500.
        ps = _make_paragraphs(5_000, 2_000)
        art = create_splits(ps)
        f0 = set(art.folds[0].test_anomaly_ids)
        f1 = set(art.folds[1].test_anomaly_ids)
        overlap = len(f0 & f1)
        # Allow ±150 absolute deviation (sampling variance ~ sqrt(K))
        assert 350 < overlap < 650, f"Overlap {overlap} far from E≈500"

    def test_cross_fold_tune_anomaly_also_overlaps(self):
        """Same overlap property for tune_anomaly stream."""
        ps = _make_paragraphs(5_000, 2_000)
        art = create_splits(ps)
        f0 = set(art.folds[0].tune_anomaly_ids)
        f1 = set(art.folds[1].tune_anomaly_ids)
        overlap = len(f0 & f1)
        assert 350 < overlap < 650, f"tune_anomaly overlap {overlap} far from E≈500"


# ---------------------------------------------------------------------------
# Within-fold disjointness invariants
# ---------------------------------------------------------------------------


class TestWithinFoldDisjointness:
    @pytest.fixture
    def artifact(self):
        ps = _make_paragraphs(1_000, 200)
        return create_splits(
            ps,
            n_folds=5,
            train_normal_per_fold=100,
            tune_normal_per_fold=50,
            tune_anomaly_per_fold=30,
            test_anomaly_per_fold=30,
        )

    def test_train_normal_disjoint_from_tune_normal(self, artifact):
        for fold in artifact.folds:
            assert set(fold.train_normal_ids).isdisjoint(set(fold.tune_normal_ids))

    def test_tune_anomaly_disjoint_from_test_anomaly(self, artifact):
        for fold in artifact.folds:
            assert set(fold.tune_anomaly_ids).isdisjoint(set(fold.test_anomaly_ids))

    def test_train_normal_disjoint_from_test_normal(self, artifact):
        """Train sampled from other-fold normals; test is fold's own partition."""
        for fold in artifact.folds:
            assert set(fold.train_normal_ids).isdisjoint(set(fold.test_normal_ids))

    def test_tune_normal_disjoint_from_test_normal(self, artifact):
        """Tune drawn from train pool (other folds' normals); test is own partition."""
        for fold in artifact.folds:
            assert set(fold.tune_normal_ids).isdisjoint(set(fold.test_normal_ids))

    def test_tune_normal_and_tune_anomaly_disjoint(self, artifact):
        """Disjoint by construction (different label pools)."""
        for fold in artifact.folds:
            assert set(fold.tune_normal_ids).isdisjoint(set(fold.tune_anomaly_ids))


# ---------------------------------------------------------------------------
# D_NEW6 seed-offset invariant
# ---------------------------------------------------------------------------


class TestSeedOffsetInvariant:
    """Changing one stream's budget MUST NOT shift another stream's IDs for
    fixed (seed, fold_id). The four distinct offsets guarantee this in the
    NON-degraded regime; in degraded regime budgets become coupled by the
    proportional split (D_NEW6.1)."""

    def test_tune_change_does_not_perturb_train_normal(self):
        ps = _make_paragraphs(1_000, 200)
        a_with_tune = create_splits(
            ps,
            train_normal_per_fold=100,
            tune_normal_per_fold=50,
            tune_anomaly_per_fold=30,
            test_anomaly_per_fold=30,
            seed=42,
        )
        a_no_tune = create_splits(
            ps,
            train_normal_per_fold=100,
            tune_normal_per_fold=0,         # changed
            tune_anomaly_per_fold=30,
            test_anomaly_per_fold=30,
            seed=42,
        )
        for fa, fb in zip(a_with_tune.folds, a_no_tune.folds):
            assert fa.train_normal_ids == fb.train_normal_ids, (
                f"Fold {fa.fold_id}: tune_normal budget perturbed train_normal "
                "— seed offsets not independent."
            )

    def test_test_anomaly_change_does_not_perturb_tune_anomaly(self):
        ps = _make_paragraphs(1_000, 200)
        a1 = create_splits(
            ps, tune_anomaly_per_fold=50, test_anomaly_per_fold=50, seed=42
        )
        a2 = create_splits(
            ps, tune_anomaly_per_fold=50, test_anomaly_per_fold=10, seed=42
        )
        for fa, fb in zip(a1.folds, a2.folds):
            assert fa.tune_anomaly_ids == fb.tune_anomaly_ids, (
                f"Fold {fa.fold_id}: test_anomaly budget perturbed tune_anomaly"
            )

    def test_train_change_does_not_perturb_tune_anomaly(self):
        """Cross-stream independence: train_normal budget changes anything
        about train, but tune_anomaly is unaffected (different seed offset)."""
        ps = _make_paragraphs(1_000, 200)
        a1 = create_splits(
            ps, train_normal_per_fold=100, tune_anomaly_per_fold=30, seed=42
        )
        a2 = create_splits(
            ps, train_normal_per_fold=80, tune_anomaly_per_fold=30, seed=42
        )
        for fa, fb in zip(a1.folds, a2.folds):
            assert fa.tune_anomaly_ids == fb.tune_anomaly_ids


# ---------------------------------------------------------------------------
# Graceful degradation (small-pool subset case) — D_NEW6.1 proportional
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_small_normal_pool_proportional(self):
        """D_NEW6.1: 600 normal / 5 folds → 480 normal per train pool.
        Train_budget=5000, tune_budget=1000 (default). Proportional split:
        train = 480*5000//6000 = 400; tune = 480-400 = 80."""
        ps = _make_paragraphs(600, 100)
        art = create_splits(ps)  # defaults: train=5k, tune_n=1k
        for fold in art.folds:
            assert len(fold.train_normal_ids) == 400
            assert len(fold.tune_normal_ids) == 80
            # Full pool used
            assert len(fold.train_normal_ids) + len(fold.tune_normal_ids) == 480

    def test_small_anomaly_pool_proportional(self):
        """D_NEW6.1: 100 anomaly pool, budgets 1k+1k → tune=50, test=50
        (proportional 50/50, not tune=100/test=0)."""
        ps = _make_paragraphs(600, 100)
        art = create_splits(ps)
        for fold in art.folds:
            assert len(fold.tune_anomaly_ids) == 50
            assert len(fold.test_anomaly_ids) == 50
            assert len(fold.tune_anomaly_ids) + len(fold.test_anomaly_ids) == 100

    def test_hdfs_subset_regression_v1_5_1(self):
        """Regression guard: HDFS subset shape (14941 normal + 698 anomaly).
        Under initial v1.5 tune-priority, this gave tune_A=698, test_A=0 —
        broken evaluation. v1.5.1 proportional: tune=349, test=349."""
        ps = _make_paragraphs(14_941, 698)
        art = create_splits(ps)  # all defaults
        for fold in art.folds:
            assert len(fold.tune_anomaly_ids) == 349
            assert len(fold.test_anomaly_ids) == 349
            # Full pool used (no leakage, no fabrication)
            assert (
                len(fold.tune_anomaly_ids) + len(fold.test_anomaly_ids) == 698
            )
            # Within-fold disjoint
            assert set(fold.tune_anomaly_ids).isdisjoint(set(fold.test_anomaly_ids))

    def test_partial_pool_proportional_normals(self):
        """Pool ~50, budgets 40+20 → train≈33, tune≈17 (proportional)."""
        # 5 folds × 12-13 normals → 4-fold train pool is 49 or 50.
        ps = _make_paragraphs(62, 10)
        art = create_splits(
            ps,
            n_folds=5,
            train_normal_per_fold=40,
            tune_normal_per_fold=20,
            tune_anomaly_per_fold=5,
            test_anomaly_per_fold=5,
        )
        for fold in art.folds:
            assert len(fold.train_normal_ids) <= 40
            # Sum equals pool size (49 or 50)
            assert len(fold.train_normal_ids) + len(fold.tune_normal_ids) <= 50

    def test_zero_anomaly_input_test_anomaly_empty(self):
        """All-normal input → all anomaly streams empty, no exception."""
        ps = _make_paragraphs(100, 0)
        art = create_splits(ps, n_folds=5)
        for fold in art.folds:
            assert fold.tune_anomaly_ids == []
            assert fold.test_anomaly_ids == []


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_produces_identical_artifact(self):
        ps = _make_paragraphs(500, 100)
        a1 = create_splits(ps, seed=42)
        a2 = create_splits(ps, seed=42)
        for f1, f2 in zip(a1.folds, a2.folds):
            assert f1.fold_id == f2.fold_id
            assert f1.seed == f2.seed
            assert f1.train_normal_ids == f2.train_normal_ids
            assert f1.tune_normal_ids == f2.tune_normal_ids
            assert f1.tune_anomaly_ids == f2.tune_anomaly_ids
            assert f1.test_normal_ids == f2.test_normal_ids
            assert f1.test_anomaly_ids == f2.test_anomaly_ids

    def test_different_seeds_produce_different_artifacts(self):
        ps = _make_paragraphs(500, 100)
        a1 = create_splits(ps, seed=42)
        a2 = create_splits(ps, seed=43)
        # At least one fold differs in at least one stream
        differs = any(
            f1.train_normal_ids != f2.train_normal_ids
            or f1.tune_anomaly_ids != f2.tune_anomaly_ids
            for f1, f2 in zip(a1.folds, a2.folds)
        )
        assert differs

    def test_per_fold_seed_recorded(self):
        ps = _make_paragraphs(100, 20)
        art = create_splits(ps, seed=42, n_folds=5)
        for k, fold in enumerate(art.folds):
            assert fold.seed == 42 + k


# ---------------------------------------------------------------------------
# Label correctness
# ---------------------------------------------------------------------------


class TestLabelCorrectness:
    def test_all_normal_streams_have_label_0(self):
        ps = _make_paragraphs(500, 100)
        art = create_splits(ps, n_folds=5)
        lookup = {p.paragraph_id: p.label for p in ps}
        for fold in art.folds:
            for pid in fold.train_normal_ids + fold.tune_normal_ids + fold.test_normal_ids:
                assert lookup[pid] == 0, f"{pid} in normal stream has label != 0"

    def test_all_anomaly_streams_have_label_1(self):
        ps = _make_paragraphs(500, 100)
        art = create_splits(ps, n_folds=5)
        lookup = {p.paragraph_id: p.label for p in ps}
        for fold in art.folds:
            for pid in fold.tune_anomaly_ids + fold.test_anomaly_ids:
                assert lookup[pid] == 1, f"{pid} in anomaly stream has label != 1"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_round_trip_save_load(self, tmp_path: Path):
        ps = _make_paragraphs(500, 100)
        original = create_splits(ps, n_folds=5, seed=42)
        path = tmp_path / "splits.json"
        save_splits(original, path)
        loaded = load_splits(path)

        assert loaded.n_folds == original.n_folds
        assert loaded.total_paragraphs == original.total_paragraphs
        assert loaded.tune_normal_per_fold == original.tune_normal_per_fold
        assert loaded.tune_anomaly_per_fold == original.tune_anomaly_per_fold
        assert loaded.test_anomaly_per_fold == original.test_anomaly_per_fold
        for f_orig, f_load in zip(original.folds, loaded.folds):
            assert f_orig.fold_id == f_load.fold_id
            assert f_orig.seed == f_load.seed
            assert f_orig.train_normal_ids == f_load.train_normal_ids
            assert f_orig.tune_normal_ids == f_load.tune_normal_ids
            assert f_orig.tune_anomaly_ids == f_load.tune_anomaly_ids
            assert f_orig.test_normal_ids == f_load.test_normal_ids
            assert f_orig.test_anomaly_ids == f_load.test_anomaly_ids

    def test_save_creates_parent_directories(self, tmp_path: Path):
        ps = _make_paragraphs(20, 4)
        art = create_splits(ps, n_folds=5, seed=42)
        nested = tmp_path / "nested" / "deep" / "splits.json"
        save_splits(art, nested)
        assert nested.exists()

    def test_persisted_json_has_v1_5_schema(self, tmp_path: Path):
        """JSON file shape: per-fold has 5 ID lists, no train_anomaly_ids;
        artifact has tune/test budget metadata."""
        ps = _make_paragraphs(20, 4)
        art = create_splits(ps, n_folds=5, seed=42)
        path = tmp_path / "splits.json"
        save_splits(art, path)
        with path.open("r") as f:
            data = json.load(f)

        assert "n_folds" in data
        assert "tune_normal_per_fold" in data
        assert "tune_anomaly_per_fold" in data
        assert "test_anomaly_per_fold" in data
        assert "folds" in data
        assert len(data["folds"]) == 5

        f0 = data["folds"][0]
        # 5 ID lists present
        assert "train_normal_ids" in f0
        assert "tune_normal_ids" in f0
        assert "tune_anomaly_ids" in f0
        assert "test_normal_ids" in f0
        assert "test_anomaly_ids" in f0
        # train_anomaly_ids absent (v1.5 schema)
        assert "train_anomaly_ids" not in f0


# ---------------------------------------------------------------------------
# FoldSplit convenience methods
# ---------------------------------------------------------------------------


class TestFoldSplitMethods:
    def test_train_total(self):
        fold = FoldSplit(fold_id=0, train_normal_ids=["a", "b", "c"])
        assert fold.train_total() == 3

    def test_tune_total(self):
        fold = FoldSplit(
            fold_id=0,
            tune_normal_ids=["n_1", "n_2", "n_3"],
            tune_anomaly_ids=["a_1", "a_2"],
        )
        assert fold.tune_total() == 5

    def test_test_total(self):
        fold = FoldSplit(
            fold_id=0,
            test_normal_ids=["n_1", "n_2"],
            test_anomaly_ids=["a_1", "a_2", "a_3"],
        )
        assert fold.test_total() == 5

    def test_tune_anomaly_rate(self):
        fold = FoldSplit(
            fold_id=0,
            tune_normal_ids=["n"] * 80,
            tune_anomaly_ids=["a"] * 20,
        )
        assert fold.tune_anomaly_rate() == pytest.approx(0.20)

    def test_test_anomaly_rate(self):
        fold = FoldSplit(
            fold_id=0,
            test_normal_ids=["n"] * 90,
            test_anomaly_ids=["a"] * 10,
        )
        assert fold.test_anomaly_rate() == pytest.approx(0.10)

    def test_rates_zero_when_empty(self):
        fold = FoldSplit(fold_id=0)
        assert fold.tune_anomaly_rate() == 0.0
        assert fold.test_anomaly_rate() == 0.0
