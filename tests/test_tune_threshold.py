"""Tests for src/tune_threshold.py — per-fold threshold tuning (paper §III-C).

Strategy:
- Pure-logic helpers (`_build_threshold_grid`, `_topk_accuracy_for_paragraph`,
  `_classify_paragraph`, `_compute_metrics`, `_evaluate_cell`,
  `_find_best_cell`) tested with synthetic Paragraph/Score objects.
- `tune_threshold` end-to-end tested with constructed SplitScores.
- `load_split_scores` tested with on-disk JSON fixtures, including the
  v1.5 D_NEW7 schema-violation hard-fail.
- CLI helpers (`_parse_topk_grid`, `_resolve_output_path`) tested directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from src.tune_threshold import (
    DEFAULT_N_THRESHOLDS,
    DEFAULT_THRESHOLD_DELTA,
    DEFAULT_TOPK_GRID,
    _build_threshold_grid,
    _classify_paragraph,
    _compute_metrics,
    _evaluate_cell,
    _find_best_cell,
    _parse_topk_grid,
    _resolve_output_path,
    _topk_accuracy_for_paragraph,
    load_split_scores,
    tune_threshold,
)
from src.types import (
    ParagraphScore,
    SplitScores,
    TopKAccuracyRecord,
    TuningCell,
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
    """Construct a ParagraphScore from a {top_k: accuracy} dict."""
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


def _make_synthetic_tune_scores(
    n_normal: int = 5, n_anomaly: int = 5, base_acc_normal: float = 0.9,
    base_acc_anomaly: float = 0.4,
) -> SplitScores:
    """Construct a SplitScores with cleanly separable normal/anomaly scores."""
    scores: list[ParagraphScore] = []
    # Normal: high accuracy
    for i in range(n_normal):
        scores.append(
            _ps(
                f"n_{i}", 0,
                {5: base_acc_normal, 9: base_acc_normal + 0.03, 12: base_acc_normal + 0.05},
            )
        )
    # Anomaly: low accuracy
    for i in range(n_anomaly):
        scores.append(
            _ps(
                f"a_{i}", 1,
                {5: base_acc_anomaly, 9: base_acc_anomaly + 0.03, 12: base_acc_anomaly + 0.05},
            )
        )
    return SplitScores(
        split_name="tune",
        topk_grid=[5, 9, 12],
        scores=scores,
        n_passes=1,
    )


# ---------------------------------------------------------------------------
# _build_threshold_grid
# ---------------------------------------------------------------------------


class TestBuildThresholdGrid:
    def test_basic_three_thresholds_evenly_spaced(self):
        grid = _build_threshold_grid(train_top1_acc=0.86)
        assert len(grid) == 3
        # linspace(0.76, 0.86, 3) -> [0.76, 0.81, 0.86]
        assert grid[0] == pytest.approx(0.76)
        assert grid[1] == pytest.approx(0.81)
        assert grid[2] == pytest.approx(0.86)

    def test_clamped_at_zero_when_top1_below_delta(self):
        """train_top1_acc=0.05, delta=0.1 -> lower clamped to 0."""
        grid = _build_threshold_grid(train_top1_acc=0.05)
        assert grid[0] == pytest.approx(0.0)
        assert grid[-1] == pytest.approx(0.05)

    def test_n_thresholds_one_returns_single_value(self):
        grid = _build_threshold_grid(train_top1_acc=0.7, n_thresholds=1)
        assert grid == [0.7]

    def test_n_thresholds_zero_raises(self):
        with pytest.raises(ValueError, match="n_thresholds must be >= 1"):
            _build_threshold_grid(train_top1_acc=0.7, n_thresholds=0)

    def test_n_thresholds_negative_raises(self):
        with pytest.raises(ValueError, match="n_thresholds must be >= 1"):
            _build_threshold_grid(train_top1_acc=0.7, n_thresholds=-1)

    def test_train_top1_acc_zero_returns_all_zero(self):
        grid = _build_threshold_grid(train_top1_acc=0.0)
        assert all(t == 0.0 for t in grid)

    def test_train_top1_acc_one_returns_proper_grid(self):
        grid = _build_threshold_grid(train_top1_acc=1.0)
        assert grid[0] == pytest.approx(0.9)
        assert grid[-1] == pytest.approx(1.0)

    def test_train_top1_acc_negative_raises(self):
        with pytest.raises(ValueError, match=r"train_top1_acc must be in \[0, 1\]"):
            _build_threshold_grid(train_top1_acc=-0.1)

    def test_train_top1_acc_above_one_raises(self):
        with pytest.raises(ValueError, match=r"train_top1_acc must be in \[0, 1\]"):
            _build_threshold_grid(train_top1_acc=1.5)

    def test_negative_delta_raises(self):
        with pytest.raises(ValueError, match="delta must be >= 0"):
            _build_threshold_grid(train_top1_acc=0.7, delta=-0.1)

    def test_custom_delta(self):
        grid = _build_threshold_grid(train_top1_acc=0.8, delta=0.2)
        # linspace(0.6, 0.8, 3) -> [0.6, 0.7, 0.8]
        assert grid[0] == pytest.approx(0.6)
        assert grid[1] == pytest.approx(0.7)
        assert grid[2] == pytest.approx(0.8)

    def test_custom_n_thresholds_five(self):
        grid = _build_threshold_grid(
            train_top1_acc=1.0, n_thresholds=5, delta=0.4
        )
        # linspace(0.6, 1.0, 5) -> [0.6, 0.7, 0.8, 0.9, 1.0]
        expected = [0.6, 0.7, 0.8, 0.9, 1.0]
        for g, e in zip(grid, expected):
            assert g == pytest.approx(e)


# ---------------------------------------------------------------------------
# _topk_accuracy_for_paragraph
# ---------------------------------------------------------------------------


class TestTopkAccuracyLookup:
    def test_finds_record_for_k(self):
        s = _ps("p1", 0, {5: 0.9, 9: 0.95, 12: 0.97})
        assert _topk_accuracy_for_paragraph(s, 5) == pytest.approx(0.9)
        assert _topk_accuracy_for_paragraph(s, 9) == pytest.approx(0.95)
        assert _topk_accuracy_for_paragraph(s, 12) == pytest.approx(0.97)

    def test_missing_k_raises_with_available_list(self):
        s = _ps("p1", 0, {5: 0.9, 9: 0.95})
        with pytest.raises(ValueError, match=r"top_k=7 not found"):
            _topk_accuracy_for_paragraph(s, 7)

    def test_error_lists_available_top_ks(self):
        s = _ps("p1", 0, {5: 0.9, 12: 0.97})
        with pytest.raises(ValueError, match=r"\[5, 12\]"):
            _topk_accuracy_for_paragraph(s, 9)


# ---------------------------------------------------------------------------
# _classify_paragraph
# ---------------------------------------------------------------------------


class TestClassifyParagraph:
    def test_below_threshold_predicts_anomaly(self):
        s = _ps("p1", 0, {5: 0.4})
        assert _classify_paragraph(s, 5, threshold=0.5) == 1

    def test_above_threshold_predicts_normal(self):
        s = _ps("p1", 0, {5: 0.9})
        assert _classify_paragraph(s, 5, threshold=0.5) == 0

    def test_exactly_at_threshold_predicts_normal(self):
        """Boundary: accuracy == threshold → normal (strict < for anomaly)."""
        s = _ps("p1", 0, {5: 0.5})
        assert _classify_paragraph(s, 5, threshold=0.5) == 0

    def test_zero_mask_paragraph_predicts_normal(self):
        """D_NEW8 zero-mask paragraphs (accuracy=1.0) are predicted normal
        for any threshold ≤ 1.0 — paper's 'pass by default' convention."""
        s = _ps("zero_mask", 0, {5: 1.0}, n_masked_total=0)
        for theta in [0.1, 0.5, 0.9, 1.0]:
            assert _classify_paragraph(s, 5, theta) == 0

    def test_threshold_above_one_classifies_zero_mask_as_anomaly(self):
        """Pathological: threshold > 1.0 would flip zero-mask paragraphs
        to anomaly. Documents the boundary; not expected in practice."""
        s = _ps("zero_mask", 0, {5: 1.0}, n_masked_total=0)
        assert _classify_paragraph(s, 5, threshold=1.01) == 1


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions(self):
        m = _compute_metrics([1, 1, 0, 0], [1, 1, 0, 0])
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)
        assert m["tp"] == 2
        assert m["fp"] == 0
        assert m["tn"] == 2
        assert m["fn"] == 0

    def test_all_wrong(self):
        m = _compute_metrics([1, 1, 0, 0], [0, 0, 1, 1])
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)
        assert m["f1"] == pytest.approx(0.0)
        assert m["specificity"] == pytest.approx(0.0)
        assert m["tp"] == 0
        assert m["fp"] == 2
        assert m["tn"] == 0
        assert m["fn"] == 2

    def test_balanced_mixed(self):
        """tp=1, fp=1, tn=1, fn=1 → P=0.5, R=0.5, F1=0.5, Spec=0.5."""
        m = _compute_metrics([1, 1, 0, 0], [1, 0, 0, 1])
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(0.5)
        assert m["f1"] == pytest.approx(0.5)
        assert m["specificity"] == pytest.approx(0.5)

    def test_no_positive_predictions(self):
        """All predictions=0: P=0 (no positive preds), R=0 (no tp), F1=0."""
        m = _compute_metrics([0, 0, 0, 0], [1, 1, 0, 0])
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)
        assert m["f1"] == pytest.approx(0.0)
        # All anomalies are FN; all normals are TN
        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["tn"] == 2
        assert m["fn"] == 2

    def test_no_negative_predictions(self):
        """All predictions=1: P=fraction-true-anomaly, R=1, F1 follows."""
        m = _compute_metrics([1, 1, 1, 1], [1, 1, 0, 0])
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(2 / 3)
        assert m["specificity"] == pytest.approx(0.0)

    def test_all_actual_anomalies(self):
        """All labels=1, mixed predictions. Specificity=0 (no negatives)."""
        m = _compute_metrics([1, 0], [1, 1])
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(0.5)
        assert m["specificity"] == pytest.approx(0.0)  # no actual negatives

    def test_all_actual_normals(self):
        """All labels=0, mixed predictions. Recall=0 (no positives)."""
        m = _compute_metrics([1, 0], [0, 0])
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)
        assert m["specificity"] == pytest.approx(0.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must have the same length"):
            _compute_metrics([0, 1, 0], [1, 0])

    def test_empty_inputs(self):
        """Edge case: empty predictions → all-zero counts → all-zero metrics."""
        m = _compute_metrics([], [])
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["specificity"] == 0.0


# ---------------------------------------------------------------------------
# _evaluate_cell
# ---------------------------------------------------------------------------


class TestEvaluateCell:
    def test_perfect_separation(self):
        """Anomalies at 0.3, normals at 0.9; threshold at 0.5 → perfect."""
        scores = [
            _ps("a_0", 1, {5: 0.3}),
            _ps("a_1", 1, {5: 0.3}),
            _ps("n_0", 0, {5: 0.9}),
            _ps("n_1", 0, {5: 0.9}),
        ]
        cell = _evaluate_cell(scores, top_k=5, threshold=0.5)
        assert cell.top_k == 5
        assert cell.threshold == 0.5
        assert cell.f1 == pytest.approx(1.0)
        assert cell.tp == 2 and cell.fp == 0 and cell.tn == 2 and cell.fn == 0

    def test_overlapping_distributions(self):
        """Threshold splits one anomaly correctly, one wrong."""
        scores = [
            _ps("a_0", 1, {5: 0.3}),  # below 0.5 → TP
            _ps("a_1", 1, {5: 0.7}),  # above 0.5 → FN
            _ps("n_0", 0, {5: 0.9}),  # above 0.5 → TN
            _ps("n_1", 0, {5: 0.4}),  # below 0.5 → FP
        ]
        cell = _evaluate_cell(scores, top_k=5, threshold=0.5)
        assert cell.tp == 1 and cell.fp == 1 and cell.tn == 1 and cell.fn == 1
        assert cell.f1 == pytest.approx(0.5)

    def test_threshold_zero_all_normal(self):
        """θ=0 means accuracy < 0 never holds → all predicted normal."""
        scores = [
            _ps("a_0", 1, {5: 0.3}),
            _ps("n_0", 0, {5: 0.9}),
        ]
        cell = _evaluate_cell(scores, top_k=5, threshold=0.0)
        assert cell.tp == 0 and cell.fp == 0
        assert cell.fn == 1 and cell.tn == 1
        assert cell.recall == 0.0


# ---------------------------------------------------------------------------
# _find_best_cell
# ---------------------------------------------------------------------------


def _cell(top_k: int, threshold: float, f1: float) -> TuningCell:
    """Build a TuningCell with placeholder P/R/Spec values for tie-break tests."""
    return TuningCell(
        top_k=top_k, threshold=threshold,
        precision=f1, recall=f1, f1=f1, specificity=f1,
    )


class TestFindBestCell:
    def test_unique_f1_max_wins(self):
        cells = [_cell(5, 0.7, 0.8), _cell(9, 0.7, 0.9), _cell(12, 0.7, 0.85)]
        k, theta = _find_best_cell(cells)
        assert k == 9
        assert theta == pytest.approx(0.7)

    def test_tie_break_smaller_k(self):
        """Equal F1 → smaller K wins."""
        cells = [_cell(12, 0.7, 0.9), _cell(5, 0.7, 0.9), _cell(9, 0.7, 0.9)]
        k, theta = _find_best_cell(cells)
        assert k == 5

    def test_tie_break_larger_threshold(self):
        """Equal F1 and K → larger threshold wins."""
        cells = [_cell(5, 0.7, 0.9), _cell(5, 0.8, 0.9), _cell(5, 0.6, 0.9)]
        k, theta = _find_best_cell(cells)
        assert k == 5
        assert theta == pytest.approx(0.8)

    def test_full_tie_break_chain(self):
        """Different F1s, mixed Ks and θs. Verify each tier of tie-break."""
        cells = [
            _cell(9, 0.7, 0.85),   # not the best F1
            _cell(5, 0.6, 0.9),    # F1=0.9, K=5, θ=0.6
            _cell(5, 0.8, 0.9),    # F1=0.9, K=5, θ=0.8  ← winner
            _cell(12, 0.6, 0.9),   # F1=0.9, K=12
            _cell(5, 0.5, 0.9),    # F1=0.9, K=5, θ=0.5
        ]
        k, theta = _find_best_cell(cells)
        assert k == 5
        assert theta == pytest.approx(0.8)

    def test_empty_cells_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            _find_best_cell([])

    def test_single_cell(self):
        k, theta = _find_best_cell([_cell(9, 0.5, 0.42)])
        assert k == 9
        assert theta == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# tune_threshold — end-to-end
# ---------------------------------------------------------------------------


class TestTuneThreshold:
    def test_separable_data_picks_perfect_cell(self):
        """Clean separation between normal and anomaly scores → F1=1.0."""
        # normals at 0.9, anomalies at 0.4 → any threshold in (0.4, 0.9] separates
        tune = _make_synthetic_tune_scores(
            n_normal=10, n_anomaly=10,
            base_acc_normal=0.9, base_acc_anomaly=0.4,
        )
        # train_top1_acc=0.85 → threshold grid [0.75, 0.8, 0.85] — all separate
        grid = tune_threshold(tune, train_top1_acc=0.85)
        # 3 K * 3 thresholds = 9 cells
        assert len(grid.cells) == 9
        # All cells with threshold in separable range should have F1=1.0
        # Best: tie-break to smallest K (5), then largest θ (0.85)
        assert grid.best_top_k == 5
        assert grid.best_threshold == pytest.approx(0.85)
        assert grid.train_top1_acc == pytest.approx(0.85)

    def test_default_topk_grid(self):
        tune = _make_synthetic_tune_scores()
        grid = tune_threshold(tune, train_top1_acc=0.85)
        ks_evaluated = sorted({c.top_k for c in grid.cells})
        assert ks_evaluated == DEFAULT_TOPK_GRID

    def test_custom_topk_grid(self):
        tune = _make_synthetic_tune_scores()
        # Need score records for K=7 too — add them
        for s in tune.scores:
            s.topk_accuracies.append(TopKAccuracyRecord(top_k=7, accuracy=s.topk_accuracies[0].accuracy))
        grid = tune_threshold(tune, train_top1_acc=0.7, topk_grid=[7])
        assert all(c.top_k == 7 for c in grid.cells)
        assert len({c.top_k for c in grid.cells}) == 1

    def test_empty_topk_grid_raises(self):
        tune = _make_synthetic_tune_scores()
        with pytest.raises(ValueError, match="at least one K"):
            tune_threshold(tune, train_top1_acc=0.85, topk_grid=[])

    def test_paper_defaults_produce_9_cells(self):
        tune = _make_synthetic_tune_scores()
        grid = tune_threshold(tune, train_top1_acc=0.85)
        assert len(grid.cells) == 3 * 3   # 3 K * 3 thresholds

    def test_all_normal_tune_set_f1_zero(self):
        """No anomalies in tune → recall=0 → F1=0 everywhere."""
        scores = [
            _ps(f"n_{i}", 0, {5: 0.9, 9: 0.9, 12: 0.9}) for i in range(5)
        ]
        tune = SplitScores(
            split_name="tune", topk_grid=[5, 9, 12], scores=scores, n_passes=1
        )
        grid = tune_threshold(tune, train_top1_acc=0.85)
        assert all(c.f1 == 0.0 for c in grid.cells)


# ---------------------------------------------------------------------------
# load_split_scores — schema validation
# ---------------------------------------------------------------------------


class TestLoadSplitScores:
    def test_load_canonical_v1_5_records_format(self, tmp_path: Path):
        scores_data = {
            "split_name": "tune",
            "topk_grid": [5, 9, 12],
            "n_passes": 1,
            "scores": [
                {
                    "paragraph_id": "p1",
                    "label": 0,
                    "topk_accuracies": [
                        {"top_k": 5, "accuracy": 0.9},
                        {"top_k": 9, "accuracy": 0.95},
                        {"top_k": 12, "accuracy": 0.97},
                    ],
                    "n_masked_total": 40,
                    "n_passes": 1,
                },
            ],
        }
        path = tmp_path / "scores_tune.json"
        path.write_text(json.dumps(scores_data))
        loaded = load_split_scores(path)
        assert loaded.split_name == "tune"
        assert loaded.topk_grid == [5, 9, 12]
        assert len(loaded.scores) == 1
        assert loaded.scores[0].topk_accuracies[0].top_k == 5
        assert loaded.scores[0].topk_accuracies[0].accuracy == 0.9

    def test_legacy_dict_form_raises_with_migration_message(
        self, tmp_path: Path
    ):
        """Pre-v1.5 dict[int, float] form must be rejected with a clear msg."""
        scores_data = {
            "split_name": "tune",
            "topk_grid": [5, 9, 12],
            "scores": [
                {
                    "paragraph_id": "p1",
                    "label": 0,
                    # LEGACY: dict instead of list
                    "topk_accuracies": {"5": 0.9, "9": 0.95, "12": 0.97},
                    "n_masked_total": 40,
                },
            ],
        }
        path = tmp_path / "scores_legacy.json"
        path.write_text(json.dumps(scores_data))
        with pytest.raises(ValueError, match="legacy dict-form"):
            load_split_scores(path)

    def test_malformed_topk_accuracies_raises(self, tmp_path: Path):
        scores_data = {
            "split_name": "tune",
            "topk_grid": [5, 9, 12],
            "scores": [
                {
                    "paragraph_id": "p1",
                    "label": 0,
                    "topk_accuracies": "not a list",
                    "n_masked_total": 40,
                },
            ],
        }
        path = tmp_path / "scores_bad.json"
        path.write_text(json.dumps(scores_data))
        with pytest.raises(ValueError, match="malformed topk_accuracies"):
            load_split_scores(path)

    def test_missing_scores_field_raises(self, tmp_path: Path):
        path = tmp_path / "scores_missing.json"
        path.write_text(json.dumps({"split_name": "tune", "topk_grid": []}))
        with pytest.raises(ValueError, match="no 'scores' field"):
            load_split_scores(path)

    def test_empty_scores_list_loads(self, tmp_path: Path):
        """Empty scores list is degenerate but valid (no records to inspect)."""
        path = tmp_path / "scores_empty.json"
        path.write_text(json.dumps({
            "split_name": "tune", "topk_grid": [5], "scores": [], "n_passes": 1,
        }))
        loaded = load_split_scores(path)
        assert loaded.scores == []


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


class TestParseTopkGrid:
    def test_basic(self):
        assert _parse_topk_grid("5,9,12") == [5, 9, 12]

    def test_whitespace(self):
        assert _parse_topk_grid(" 5 , 9 , 12 ") == [5, 9, 12]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one integer"):
            _parse_topk_grid("")

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            _parse_topk_grid("5,0,9")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            _parse_topk_grid("5,-3,9")


class TestResolveOutputPath:
    def test_directory_appends_filename(self, tmp_path: Path):
        out = _resolve_output_path(tmp_path)
        assert out == tmp_path / "tuning_grid.json"

    def test_file_path_used_as_is(self, tmp_path: Path):
        target = tmp_path / "custom.json"
        out = _resolve_output_path(target)
        assert out == target

    def test_nonexistent_no_suffix_treated_as_dir(self, tmp_path: Path):
        target = tmp_path / "newdir"
        out = _resolve_output_path(target)
        assert out == target / "tuning_grid.json"
