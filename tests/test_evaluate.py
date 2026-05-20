"""Tests for src/evaluate.py — cross-fold test-set evaluation (paper §IV-A).

Strategy:
- Pure-logic helpers (`evaluate_fold`, `_safe_std`, `aggregate_folds`)
  tested with synthetic FoldMetrics + SplitScores.
- `load_tuning_grid` tested with on-disk JSON fixtures.
- `evaluate_all_folds` end-to-end tested by building 3 fake fold directories
  with realistic tuning_grid.json + scores_test.json fixtures, then asserting
  the aggregated DatasetResults.
- `_safe_std` verified against a known-value test using ddof=1 sample std.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path

import pytest

from src.evaluate import (
    _resolve_output_path,
    _safe_std,
    aggregate_folds,
    evaluate_all_folds,
    evaluate_fold,
    load_tuning_grid,
)
from src.types import (
    DatasetResults,
    FoldMetrics,
    ParagraphScore,
    SplitScores,
    TopKAccuracyRecord,
    TuningCell,
    TuningGrid,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic fixtures
# ---------------------------------------------------------------------------


def _ps(
    pid: str,
    label: int,
    accs: dict[int, float],
    n_masked_total: int = 10,
    n_passes: int = 1,
) -> ParagraphScore:
    records = [
        TopKAccuracyRecord(top_k=k, accuracy=a) for k, a in sorted(accs.items())
    ]
    return ParagraphScore(
        paragraph_id=pid,
        label=label,
        topk_accuracies=records,
        n_masked_total=n_masked_total,
        n_passes=n_passes,
    )


def _make_test_scores(
    n_normal: int = 5,
    n_anomaly: int = 5,
    base_acc_normal: float = 0.9,
    base_acc_anomaly: float = 0.4,
) -> SplitScores:
    """Fixture: paper-default K=5,9,12 grid with separable normal/anomaly."""
    scores: list[ParagraphScore] = []
    for i in range(n_normal):
        scores.append(_ps(
            f"n_{i}", 0,
            {5: base_acc_normal, 9: base_acc_normal + 0.03, 12: base_acc_normal + 0.05},
        ))
    for i in range(n_anomaly):
        scores.append(_ps(
            f"a_{i}", 1,
            {5: base_acc_anomaly, 9: base_acc_anomaly + 0.03, 12: base_acc_anomaly + 0.05},
        ))
    return SplitScores(
        split_name="test", topk_grid=[5, 9, 12], scores=scores, n_passes=1
    )


def _make_fold_metrics(
    fold_idx: int, precision: float, recall: float, f1: float, specificity: float,
    top_k: int = 5, threshold: float = 0.5,
) -> FoldMetrics:
    """FoldMetrics with placeholder confusion counts derived from rates."""
    return FoldMetrics(
        fold_idx=fold_idx,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        tp=10, fp=2, tn=20, fn=3,   # placeholder counts
        top_k=top_k,
        threshold=threshold,
    )


def _write_tuning_grid_json(
    path: Path,
    best_top_k: int = 5,
    best_threshold: float = 0.5,
    train_top1_acc: float = 0.85,
) -> None:
    """Write a minimal tuning_grid.json for fixture purposes."""
    cell = TuningCell(
        top_k=best_top_k, threshold=best_threshold,
        precision=1.0, recall=1.0, f1=1.0, specificity=1.0,
        tp=5, fp=0, tn=5, fn=0,
    )
    grid = TuningGrid(
        cells=[cell],
        best_top_k=best_top_k,
        best_threshold=best_threshold,
        train_top1_acc=train_top1_acc,
    )
    path.write_text(json.dumps(asdict(grid)))


def _write_scores_test_json(
    path: Path,
    test_scores: SplitScores | None = None,
) -> None:
    if test_scores is None:
        test_scores = _make_test_scores()
    path.write_text(json.dumps(asdict(test_scores)))


# ---------------------------------------------------------------------------
# load_tuning_grid
# ---------------------------------------------------------------------------


class TestLoadTuningGrid:
    def test_load_canonical_v1_5_artifact(self, tmp_path: Path):
        path = tmp_path / "tuning_grid.json"
        _write_tuning_grid_json(path, best_top_k=9, best_threshold=0.82)
        grid = load_tuning_grid(path)
        assert grid.best_top_k == 9
        assert grid.best_threshold == pytest.approx(0.82)
        assert grid.train_top1_acc == pytest.approx(0.85)
        assert len(grid.cells) == 1
        # tp/fp/tn/fn round-trip correctly
        assert grid.cells[0].tp == 5
        assert grid.cells[0].fp == 0

    def test_missing_required_field_raises(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "cells": [],
            "best_top_k": 5,
            # missing best_threshold and train_top1_acc
        }))
        with pytest.raises(ValueError, match="missing required keys"):
            load_tuning_grid(path)

    def test_legacy_cells_without_tp_fp_tn_fn_load_with_zeros(
        self, tmp_path: Path
    ):
        """Pre-v1.5 tuning_grid.json without confusion fields should load
        with zeros — backward-compat path."""
        legacy_grid = {
            "cells": [
                {
                    "top_k": 5, "threshold": 0.5,
                    "precision": 0.8, "recall": 0.7, "f1": 0.75,
                    "specificity": 0.9,
                    # no tp/fp/tn/fn
                },
            ],
            "best_top_k": 5,
            "best_threshold": 0.5,
            "train_top1_acc": 0.85,
        }
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy_grid))
        grid = load_tuning_grid(path)
        assert grid.cells[0].tp == 0
        assert grid.cells[0].fp == 0
        assert grid.cells[0].precision == pytest.approx(0.8)

    def test_multiple_cells_preserved(self, tmp_path: Path):
        grid_obj = TuningGrid(
            cells=[
                TuningCell(5, 0.5, 0.8, 0.7, 0.75, 0.9, tp=5, fp=1, tn=5, fn=1),
                TuningCell(9, 0.6, 0.85, 0.75, 0.8, 0.92, tp=6, fp=1, tn=4, fn=2),
                TuningCell(12, 0.7, 0.9, 0.8, 0.85, 0.95, tp=7, fp=1, tn=3, fn=2),
            ],
            best_top_k=12,
            best_threshold=0.7,
            train_top1_acc=0.85,
        )
        path = tmp_path / "grid.json"
        path.write_text(json.dumps(asdict(grid_obj)))
        loaded = load_tuning_grid(path)
        assert len(loaded.cells) == 3
        assert loaded.cells[2].top_k == 12
        assert loaded.cells[2].threshold == pytest.approx(0.7)
        assert loaded.best_top_k == 12


# ---------------------------------------------------------------------------
# evaluate_fold
# ---------------------------------------------------------------------------


class TestEvaluateFold:
    def test_perfect_separation_at_threshold(self):
        """Clean separation: normals at 0.9, anomalies at 0.4, θ=0.5."""
        scores = _make_test_scores(
            n_normal=5, n_anomaly=5, base_acc_normal=0.9, base_acc_anomaly=0.4
        )
        m = evaluate_fold(
            test_scores=scores, best_top_k=5, best_threshold=0.5, fold_idx=0
        )
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)
        assert m.specificity == pytest.approx(1.0)
        assert m.tp == 5 and m.fp == 0 and m.tn == 5 and m.fn == 0

    def test_overlapping_distributions(self):
        """One anomaly accidentally above threshold (FN), one normal below (FP)."""
        scores = SplitScores(
            split_name="test", topk_grid=[5], n_passes=1,
            scores=[
                _ps("a_0", 1, {5: 0.3}),  # below 0.5 → TP
                _ps("a_1", 1, {5: 0.7}),  # above 0.5 → FN
                _ps("n_0", 0, {5: 0.9}),  # above 0.5 → TN
                _ps("n_1", 0, {5: 0.4}),  # below 0.5 → FP
            ],
        )
        m = evaluate_fold(scores, best_top_k=5, best_threshold=0.5, fold_idx=2)
        assert m.tp == 1 and m.fp == 1 and m.tn == 1 and m.fn == 1
        assert m.f1 == pytest.approx(0.5)

    def test_zero_mask_paragraphs_predicted_normal(self):
        """Zero-mask paragraphs (accuracy=1.0) are always predicted normal,
        regardless of label — matches D_NEW8 'pass by default' convention."""
        scores = SplitScores(
            split_name="test", topk_grid=[5], n_passes=1,
            scores=[
                _ps("zero_a", 1, {5: 1.0}, n_masked_total=0),
                _ps("zero_n", 0, {5: 1.0}, n_masked_total=0),
            ],
        )
        m = evaluate_fold(scores, best_top_k=5, best_threshold=0.5, fold_idx=0)
        # Both predicted normal; the labels distribute as fn=1 (anomaly→normal)
        # and tn=1 (normal→normal).
        assert m.tp == 0 and m.fp == 0
        assert m.tn == 1 and m.fn == 1

    def test_fold_idx_preserved_in_result(self):
        scores = _make_test_scores()
        m = evaluate_fold(scores, best_top_k=5, best_threshold=0.5, fold_idx=3)
        assert m.fold_idx == 3

    def test_operating_point_recorded(self):
        scores = _make_test_scores()
        m = evaluate_fold(scores, best_top_k=12, best_threshold=0.77, fold_idx=0)
        assert m.top_k == 12
        assert m.threshold == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# _safe_std
# ---------------------------------------------------------------------------


class TestSafeStd:
    def test_empty_returns_zero(self):
        assert _safe_std([]) == 0.0

    def test_single_value_returns_zero(self):
        """statistics.stdev raises for n<2; we return 0.0 instead."""
        assert _safe_std([0.5]) == 0.0

    def test_two_values_matches_sample_std(self):
        # mean=0.5, variance (ddof=1): ((0.3-0.5)^2 + (0.7-0.5)^2) / (2-1) = 0.08
        # std = sqrt(0.08) ≈ 0.2828
        assert _safe_std([0.3, 0.7]) == pytest.approx(statistics.stdev([0.3, 0.7]))

    def test_uses_ddof_1_sample_std_against_known_values(self):
        """For [0.8, 0.85, 0.82, 0.9, 0.78]:
        mean = 0.83, variance (ddof=1) = sum((x - 0.83)^2) / (5-1)
             = (0.0009 + 0.0004 + 0.0001 + 0.0049 + 0.0025) / 4
             = 0.0088 / 4 = 0.0022
        std = sqrt(0.0022) ≈ 0.0469
        """
        values = [0.8, 0.85, 0.82, 0.9, 0.78]
        expected = statistics.stdev(values)  # ddof=1 by default
        assert _safe_std(values) == pytest.approx(expected)
        # Manually verify it's NOT population std (ddof=0)
        pop_std = (sum((v - 0.83) ** 2 for v in values) / 5) ** 0.5
        assert _safe_std(values) > pop_std  # sample std > pop std for same data


# ---------------------------------------------------------------------------
# aggregate_folds
# ---------------------------------------------------------------------------


class TestAggregateFolds:
    def test_basic_means(self):
        folds = [
            _make_fold_metrics(0, 0.8, 0.7, 0.75, 0.9),
            _make_fold_metrics(1, 0.85, 0.75, 0.8, 0.92),
            _make_fold_metrics(2, 0.9, 0.8, 0.85, 0.94),
        ]
        res = aggregate_folds(folds, dataset="hdfs", window_seconds=None, backbone="roberta-base")
        assert res.p_mean == pytest.approx(0.85)
        assert res.r_mean == pytest.approx(0.75)
        assert res.f1_mean == pytest.approx(0.8)
        assert res.spec_mean == pytest.approx(0.92)

    def test_uses_sample_std_ddof_1(self):
        """Three folds with P=[0.8, 0.85, 0.9]: mean=0.85, std(ddof=1)=0.05."""
        folds = [
            _make_fold_metrics(0, 0.8, 0.7, 0.75, 0.9),
            _make_fold_metrics(1, 0.85, 0.75, 0.8, 0.92),
            _make_fold_metrics(2, 0.9, 0.8, 0.85, 0.94),
        ]
        res = aggregate_folds(folds, dataset="hdfs", window_seconds=None, backbone="roberta-base")
        assert res.p_std == pytest.approx(statistics.stdev([0.8, 0.85, 0.9]))
        assert res.p_std == pytest.approx(0.05)

    def test_per_fold_values_populated(self):
        folds = [
            _make_fold_metrics(0, 0.8, 0.7, 0.75, 0.9),
            _make_fold_metrics(1, 0.85, 0.75, 0.8, 0.92),
        ]
        res = aggregate_folds(folds, dataset="bgl", window_seconds=30, backbone="allenai/longformer-base-4096")
        assert res.per_fold_values["precision"] == [0.8, 0.85]
        assert res.per_fold_values["recall"] == [0.7, 0.75]
        assert res.per_fold_values["f1"] == [0.75, 0.8]
        assert res.per_fold_values["specificity"] == [0.9, 0.92]

    def test_per_fold_operating_points_populated(self):
        folds = [
            _make_fold_metrics(0, 0.8, 0.7, 0.75, 0.9, top_k=5, threshold=0.81),
            _make_fold_metrics(1, 0.85, 0.75, 0.8, 0.92, top_k=9, threshold=0.79),
        ]
        res = aggregate_folds(folds, dataset="hdfs", window_seconds=None, backbone="roberta-base")
        assert len(res.per_fold_operating_points) == 2
        assert res.per_fold_operating_points[0]["fold_idx"] == 0
        assert res.per_fold_operating_points[0]["top_k"] == 5
        assert res.per_fold_operating_points[0]["threshold"] == pytest.approx(0.81)
        assert res.per_fold_operating_points[1]["top_k"] == 9
        assert res.per_fold_operating_points[1]["tp"] == 10  # from placeholder
        assert res.per_fold_operating_points[1]["fp"] == 2

    def test_dataset_window_backbone_recorded(self):
        folds = [_make_fold_metrics(0, 0.8, 0.7, 0.75, 0.9)]
        res = aggregate_folds(
            folds, dataset="tbird", window_seconds=30,
            backbone="allenai/longformer-base-4096",
        )
        assert res.dataset == "tbird"
        assert res.window_seconds == 30
        assert res.backbone == "allenai/longformer-base-4096"

    def test_single_fold_std_zero(self):
        folds = [_make_fold_metrics(0, 0.8, 0.7, 0.75, 0.9)]
        res = aggregate_folds(folds, dataset="hdfs", window_seconds=None, backbone="roberta-base")
        assert res.p_mean == pytest.approx(0.8)
        assert res.p_std == 0.0
        assert res.r_std == 0.0
        assert res.f1_std == 0.0
        assert res.spec_std == 0.0

    def test_empty_folds_raises(self):
        with pytest.raises(ValueError, match="at least one FoldMetrics"):
            aggregate_folds(
                [], dataset="hdfs", window_seconds=None, backbone="roberta-base"
            )

    def test_window_seconds_none_persists_as_none(self):
        folds = [_make_fold_metrics(0, 0.8, 0.7, 0.75, 0.9)]
        res = aggregate_folds(folds, dataset="hdfs", window_seconds=None, backbone="roberta-base")
        assert res.window_seconds is None


# ---------------------------------------------------------------------------
# evaluate_all_folds (end-to-end)
# ---------------------------------------------------------------------------


class TestEvaluateAllFolds:
    def _build_3_fold_directory(self, tmp_path: Path) -> Path:
        """Create scores_root with 3 fold directories, each having
        tuning_grid.json + scores_test.json. Folds use K from the paper grid."""
        scores_root = tmp_path / "scores"
        fold_ks = [5, 9, 12]   # paper default grid
        fold_thetas = [0.5, 0.55, 0.6]
        for k_idx in range(3):
            fold_dir = scores_root / f"fold_{k_idx}"
            fold_dir.mkdir(parents=True)
            _write_tuning_grid_json(
                fold_dir / "tuning_grid.json",
                best_top_k=fold_ks[k_idx],
                best_threshold=fold_thetas[k_idx],
            )
            _write_scores_test_json(fold_dir / "scores_test.json")
        return scores_root

    def test_end_to_end_three_folds(self, tmp_path: Path):
        scores_root = self._build_3_fold_directory(tmp_path)
        res = evaluate_all_folds(
            scores_root=scores_root,
            n_folds=3,
            dataset="hdfs",
            window_seconds=None,
            backbone="roberta-base",
        )
        assert isinstance(res, DatasetResults)
        assert res.dataset == "hdfs"
        # All three folds have perfect separation (synthetic) — F1=1.0 mean, 0 std
        assert res.f1_mean == pytest.approx(1.0)
        assert res.f1_std == 0.0
        # Per-fold operating points captured the different (K, θ) used
        ks = [op["top_k"] for op in res.per_fold_operating_points]
        assert ks == [5, 9, 12]
        thetas = [op["threshold"] for op in res.per_fold_operating_points]
        assert thetas == pytest.approx([0.5, 0.55, 0.6])

    def test_n_folds_zero_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="n_folds must be >= 1"):
            evaluate_all_folds(
                scores_root=tmp_path, n_folds=0,
                dataset="hdfs", window_seconds=None, backbone="roberta-base",
            )

    def test_missing_fold_directory_raises(self, tmp_path: Path):
        # Build only 2 folds but request 3
        scores_root = tmp_path / "scores"
        for k in range(2):
            fold_dir = scores_root / f"fold_{k}"
            fold_dir.mkdir(parents=True)
            _write_tuning_grid_json(fold_dir / "tuning_grid.json")
            _write_scores_test_json(fold_dir / "scores_test.json")
        with pytest.raises(FileNotFoundError, match=r"Fold directory does not exist"):
            evaluate_all_folds(
                scores_root=scores_root, n_folds=3,
                dataset="hdfs", window_seconds=None, backbone="roberta-base",
            )

    def test_missing_tuning_grid_raises(self, tmp_path: Path):
        scores_root = tmp_path / "scores"
        fold_dir = scores_root / "fold_0"
        fold_dir.mkdir(parents=True)
        _write_scores_test_json(fold_dir / "scores_test.json")
        # No tuning_grid.json
        with pytest.raises(FileNotFoundError, match=r"Missing tuning_grid.json"):
            evaluate_all_folds(
                scores_root=scores_root, n_folds=1,
                dataset="hdfs", window_seconds=None, backbone="roberta-base",
            )

    def test_missing_scores_test_raises(self, tmp_path: Path):
        scores_root = tmp_path / "scores"
        fold_dir = scores_root / "fold_0"
        fold_dir.mkdir(parents=True)
        _write_tuning_grid_json(fold_dir / "tuning_grid.json")
        # No scores_test.json
        with pytest.raises(FileNotFoundError, match=r"Missing scores_test.json"):
            evaluate_all_folds(
                scores_root=scores_root, n_folds=1,
                dataset="hdfs", window_seconds=None, backbone="roberta-base",
            )

    def test_wrong_split_name_raises(self, tmp_path: Path):
        """scores_test.json file must have split_name=='test' — guards against
        accidentally pointing at scores_tune.json."""
        scores_root = tmp_path / "scores"
        fold_dir = scores_root / "fold_0"
        fold_dir.mkdir(parents=True)
        _write_tuning_grid_json(fold_dir / "tuning_grid.json")
        # Write a tune-named scores under the test filename
        tune_scores = _make_test_scores()
        tune_scores.split_name = "tune"
        (fold_dir / "scores_test.json").write_text(json.dumps(asdict(tune_scores)))
        with pytest.raises(ValueError, match=r"split_name='tune'"):
            evaluate_all_folds(
                scores_root=scores_root, n_folds=1,
                dataset="hdfs", window_seconds=None, backbone="roberta-base",
            )


# ---------------------------------------------------------------------------
# _resolve_output_path
# ---------------------------------------------------------------------------


class TestResolveOutputPath:
    def test_directory_appends_filename(self, tmp_path: Path):
        out = _resolve_output_path(tmp_path)
        assert out == tmp_path / "dataset_results.json"

    def test_file_path_used_as_is(self, tmp_path: Path):
        target = tmp_path / "custom_name.json"
        assert _resolve_output_path(target) == target

    def test_nonexistent_no_suffix_treated_as_dir(self, tmp_path: Path):
        target = tmp_path / "newdir"
        assert _resolve_output_path(target) == target / "dataset_results.json"
