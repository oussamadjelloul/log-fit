"""Tests for src/splits.py — 5-fold CV splits per spec D14."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from src.splits import (
    DEFAULT_N_FOLDS,
    DEFAULT_SEED,
    DEFAULT_TRAIN_ANOMALY_PER_FOLD,
    DEFAULT_TRAIN_NORMAL_PER_FOLD,
    FoldSplit,
    SplitsArtifact,
    _sample_or_full,
    create_splits,
    load_splits,
    save_splits,
    stratified_fold_assignment,
)
from src.types import Paragraph


def _make_paragraphs(n_normal: int, n_anomaly: int) -> list[Paragraph]:
    """Build a synthetic paragraph list. IDs are 'n_0', 'n_1', ..., 'a_0', 'a_1', ..."""
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
# Stratified fold assignment
# ---------------------------------------------------------------------------


class TestStratifiedFoldAssignment:
    def test_assignment_length_matches_input(self):
        paragraphs = _make_paragraphs(100, 20)
        assignment = stratified_fold_assignment(paragraphs, n_folds=5)
        assert len(assignment) == 120

    def test_all_in_valid_range(self):
        paragraphs = _make_paragraphs(100, 20)
        assignment = stratified_fold_assignment(paragraphs, n_folds=5)
        assert all(0 <= a < 5 for a in assignment)
        assert -1 not in assignment

    def test_stratification_preserves_anomaly_rate(self):
        """Each fold should have ~same anomaly rate as the overall dataset."""
        paragraphs = _make_paragraphs(500, 100)  # 16.7% anomaly rate
        assignment = stratified_fold_assignment(paragraphs, n_folds=5)

        per_fold_counts = {
            fold: {"normal": 0, "anomaly": 0} for fold in range(5)
        }
        for idx, fold in enumerate(assignment):
            label = paragraphs[idx].label
            key = "anomaly" if label == 1 else "normal"
            per_fold_counts[fold][key] += 1

        # Each fold should have exactly ~20 anomalies (100/5)
        for fold, counts in per_fold_counts.items():
            assert 18 <= counts["anomaly"] <= 22, (
                f"Fold {fold} has {counts['anomaly']} anomalies, "
                f"expected ~20"
            )
            assert 95 <= counts["normal"] <= 105

    def test_deterministic_with_same_seed(self):
        paragraphs = _make_paragraphs(100, 20)
        a1 = stratified_fold_assignment(paragraphs, n_folds=5, seed=42)
        a2 = stratified_fold_assignment(paragraphs, n_folds=5, seed=42)
        assert a1 == a2

    def test_different_seeds_give_different_assignments(self):
        paragraphs = _make_paragraphs(100, 20)
        a1 = stratified_fold_assignment(paragraphs, n_folds=5, seed=42)
        a2 = stratified_fold_assignment(paragraphs, n_folds=5, seed=43)
        assert a1 != a2

    def test_empty_paragraphs_returns_empty(self):
        assert stratified_fold_assignment([], n_folds=5) == []

    def test_n_folds_below_2_raises(self):
        paragraphs = _make_paragraphs(10, 2)
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            stratified_fold_assignment(paragraphs, n_folds=1)

    def test_single_class_only_normal_works(self):
        paragraphs = _make_paragraphs(20, 0)
        assignment = stratified_fold_assignment(paragraphs, n_folds=5)
        assert len(assignment) == 20
        assert -1 not in assignment

    def test_single_class_only_anomaly_works(self):
        paragraphs = _make_paragraphs(0, 20)
        assignment = stratified_fold_assignment(paragraphs, n_folds=5)
        assert len(assignment) == 20
        assert -1 not in assignment


# ---------------------------------------------------------------------------
# _sample_or_full helper
# ---------------------------------------------------------------------------


class TestSampleOrFull:
    def test_returns_full_when_smaller_than_n(self):
        items = [Paragraph(paragraph_id=f"x_{i}", lines=[], label=0)
                 for i in range(5)]
        result = _sample_or_full(items, n=10, seed=42)
        assert result == items  # exact same list

    def test_samples_when_larger_than_n(self):
        items = [Paragraph(paragraph_id=f"x_{i}", lines=[], label=0)
                 for i in range(20)]
        result = _sample_or_full(items, n=5, seed=42)
        assert len(result) == 5
        # All sampled items are from the original
        ids_sampled = {p.paragraph_id for p in result}
        ids_original = {p.paragraph_id for p in items}
        assert ids_sampled.issubset(ids_original)

    def test_deterministic_with_same_seed(self):
        items = [Paragraph(paragraph_id=f"x_{i}", lines=[], label=0)
                 for i in range(20)]
        r1 = _sample_or_full(items, n=5, seed=42)
        r2 = _sample_or_full(items, n=5, seed=42)
        assert [p.paragraph_id for p in r1] == [p.paragraph_id for p in r2]

    def test_different_seeds_produce_different_samples(self):
        items = [Paragraph(paragraph_id=f"x_{i}", lines=[], label=0)
                 for i in range(20)]
        r1 = _sample_or_full(items, n=5, seed=42)
        r2 = _sample_or_full(items, n=5, seed=43)
        assert [p.paragraph_id for p in r1] != [p.paragraph_id for p in r2]


# ---------------------------------------------------------------------------
# create_splits — main API
# ---------------------------------------------------------------------------


class TestCreateSplits:
    def test_returns_correct_number_of_folds(self):
        paragraphs = _make_paragraphs(100, 20)
        artifact = create_splits(paragraphs, n_folds=5)
        assert len(artifact.folds) == 5
        assert artifact.n_folds == 5

    def test_test_sets_partition_input(self):
        """Each paragraph appears in exactly one test set."""
        paragraphs = _make_paragraphs(100, 20)
        artifact = create_splits(paragraphs, n_folds=5)

        all_test_ids = []
        for fold in artifact.folds:
            all_test_ids.extend(fold.test_normal_ids)
            all_test_ids.extend(fold.test_anomaly_ids)

        # Every paragraph appears once
        assert len(all_test_ids) == 120
        assert len(set(all_test_ids)) == 120

    def test_train_test_disjoint_within_fold(self):
        """For each fold, no paragraph_id appears in both train and test."""
        paragraphs = _make_paragraphs(100, 20)
        artifact = create_splits(paragraphs, n_folds=5)

        for fold in artifact.folds:
            train_ids = set(fold.train_normal_ids) | set(fold.train_anomaly_ids)
            test_ids = set(fold.test_normal_ids) | set(fold.test_anomaly_ids)
            assert train_ids.isdisjoint(test_ids), (
                f"Fold {fold.fold_id} has overlap between train and test"
            )

    def test_train_cap_respected_when_pool_larger(self):
        """When pool > cap, train sample is exactly cap-sized."""
        # 1000 normal, 100 anomaly. 5 folds → train pool ~800N + 80A.
        # Cap 50N + 10A → train should be exactly 50+10 per fold.
        paragraphs = _make_paragraphs(1000, 100)
        artifact = create_splits(
            paragraphs,
            n_folds=5,
            train_normal_per_fold=50,
            train_anomaly_per_fold=10,
        )
        for fold in artifact.folds:
            assert len(fold.train_normal_ids) == 50
            assert len(fold.train_anomaly_ids) == 10

    def test_full_pool_used_when_pool_smaller_than_cap(self):
        """BGL-subset case: when pool < cap, use everything."""
        # 50 normal, 10 anomaly. 5 folds. Train pool ~40N + 8A.
        # Cap = 25000+2000 → train should be 40N + 8A (full pool).
        paragraphs = _make_paragraphs(50, 10)
        artifact = create_splits(paragraphs, n_folds=5)
        for fold in artifact.folds:
            # Pool is 40 normal (4 folds × 10) and 8 anomaly (4 folds × 2)
            assert len(fold.train_normal_ids) == 40
            assert len(fold.train_anomaly_ids) == 8

    def test_deterministic_with_same_seed(self):
        paragraphs = _make_paragraphs(100, 20)
        a1 = create_splits(paragraphs, n_folds=5, seed=42)
        a2 = create_splits(paragraphs, n_folds=5, seed=42)
        for f1, f2 in zip(a1.folds, a2.folds):
            assert f1.train_normal_ids == f2.train_normal_ids
            assert f1.train_anomaly_ids == f2.train_anomaly_ids
            assert f1.test_normal_ids == f2.test_normal_ids
            assert f1.test_anomaly_ids == f2.test_anomaly_ids

    def test_metadata_fields_populated(self):
        paragraphs = _make_paragraphs(100, 20)
        artifact = create_splits(paragraphs, n_folds=5, seed=42)
        assert artifact.total_paragraphs == 120
        assert artifact.total_normal == 100
        assert artifact.total_anomaly == 20
        assert artifact.seed == 42
        assert artifact.n_folds == 5

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            create_splits([])


# ---------------------------------------------------------------------------
# D14 cross-fold anomaly overlap behavior
# ---------------------------------------------------------------------------


class TestCrossFoldAnomalyOverlap:
    """D14 spec: anomaly samples shuffled per-fold; cross-fold overlap exists.

    With a finite anomaly pool sampled `train_anomaly_per_fold` times across
    n_folds folds, expected overlap between any two folds is approximately
    `train_anomaly_per_fold^2 / pool_size`.
    """

    def test_some_overlap_exists_between_folds(self):
        # Pool size 200 anomalies, sample 40 per fold (out of 4-fold train pool
        # which has 160 anomalies → expected per-fold sample ~40 from 160 pool).
        # E[overlap] between two folds ≈ 40 × 40 / 160 = 10 anomalies.
        paragraphs = _make_paragraphs(1000, 200)
        artifact = create_splits(
            paragraphs,
            n_folds=5,
            train_normal_per_fold=200,
            train_anomaly_per_fold=40,
            seed=42,
        )

        fold0_anomaly = set(artifact.folds[0].train_anomaly_ids)
        fold1_anomaly = set(artifact.folds[1].train_anomaly_ids)
        overlap = fold0_anomaly & fold1_anomaly
        # Some overlap expected; not zero
        assert len(overlap) > 0
        # But also not full overlap (they're sampled independently)
        assert len(overlap) < len(fold0_anomaly)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip_save_load(self, tmp_path: Path):
        paragraphs = _make_paragraphs(50, 10)
        original = create_splits(paragraphs, n_folds=5, seed=42)

        path = tmp_path / "splits.json"
        save_splits(original, path)
        loaded = load_splits(path)

        assert loaded.n_folds == original.n_folds
        assert loaded.total_paragraphs == original.total_paragraphs
        assert loaded.total_normal == original.total_normal
        assert loaded.total_anomaly == original.total_anomaly
        assert len(loaded.folds) == len(original.folds)
        for orig, ld in zip(original.folds, loaded.folds):
            assert orig.fold_id == ld.fold_id
            assert orig.train_normal_ids == ld.train_normal_ids
            assert orig.train_anomaly_ids == ld.train_anomaly_ids
            assert orig.test_normal_ids == ld.test_normal_ids
            assert orig.test_anomaly_ids == ld.test_anomaly_ids
            assert orig.seed == ld.seed

    def test_save_creates_parent_directories(self, tmp_path: Path):
        paragraphs = _make_paragraphs(20, 4)
        artifact = create_splits(paragraphs, n_folds=5, seed=42)
        nested_path = tmp_path / "nested" / "deep" / "splits.json"
        save_splits(artifact, nested_path)
        assert nested_path.exists()

    def test_json_is_human_readable(self, tmp_path: Path):
        paragraphs = _make_paragraphs(20, 4)
        artifact = create_splits(paragraphs, n_folds=5, seed=42)
        path = tmp_path / "splits.json"
        save_splits(artifact, path)
        with path.open("r") as f:
            data = json.load(f)
        assert "n_folds" in data
        assert "folds" in data
        assert len(data["folds"]) == 5
        # First fold has the expected keys
        f0 = data["folds"][0]
        assert "train_normal_ids" in f0
        assert "test_anomaly_ids" in f0


# ---------------------------------------------------------------------------
# FoldSplit convenience methods
# ---------------------------------------------------------------------------


class TestFoldSplitMethods:
    def test_train_total(self):
        fold = FoldSplit(
            fold_id=0,
            train_normal_ids=["a", "b", "c"],
            train_anomaly_ids=["x"],
        )
        assert fold.train_total() == 4

    def test_test_total(self):
        fold = FoldSplit(
            fold_id=0,
            test_normal_ids=["a", "b"],
            test_anomaly_ids=["x", "y", "z"],
        )
        assert fold.test_total() == 5

    def test_anomaly_rate(self):
        fold = FoldSplit(
            fold_id=0,
            train_normal_ids=["a"] * 90,
            train_anomaly_ids=["x"] * 10,
        )
        assert fold.train_anomaly_rate() == pytest.approx(0.10)

    def test_anomaly_rate_zero_when_empty(self):
        fold = FoldSplit(fold_id=0)
        assert fold.train_anomaly_rate() == 0.0
        assert fold.test_anomaly_rate() == 0.0
