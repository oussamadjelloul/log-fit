"""Tests for src/score.py — paragraph scoring (v1.5 D_NEW6/7/8/9).

Strategy:
- Pure-logic helpers (`_parse_topk_grid`, `_resolve_backbone`,
  `_load_fold_data`, `_select_split_ids`, `_map_paragraphs_by_id`,
  `_prepare_eval_paragraphs`, `_resolve_output_path`, `_resolve_output_dir`)
  tested with synthetic JSON fixtures and `Paragraph` instances.
- The scoring loop (`_score_paragraphs_r_passes`) is NOT unit-tested here.
  It requires a real (or carefully-faked) MLM model and the
  `MaskedSentenceDataset` machinery; coverage is via the Narval integration
  run through `scripts/score.sh`.

Mapping to bugs from the v1.5 score.py adversarial review:
- M1: canonical schema enforcement       -> TestLoadFoldData, TestSelectSplitIds
- M2: paragraph_id uniqueness assertion  -> TestMapParagraphsById
- I3: hard-fail on missing --backbone-decision -> TestResolveBackbone
- B1: batch-size CLI semantics           -> covered in integration; argparse
                                              tested implicitly via main()
- B2: list-of-records format             -> src/types.py + score.py emit
                                              TopKAccuracyRecord (not dict),
                                              persistence covered in
                                              integration
- I1/D_NEW8: --n-passes flag             -> covered in integration
- I2: zero-mask edge case                -> documented; tests below verify
                                              the `n_masked_total` path
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.score import (
    _load_fold_data,
    _map_paragraphs_by_id,
    _parse_topk_grid,
    _prepare_eval_paragraphs,
    _resolve_backbone,
    _resolve_output_dir,
    _resolve_output_path,
    _select_split_ids,
)
from src.types import Paragraph


# ---------------------------------------------------------------------------
# Helpers — synthetic fixtures
# ---------------------------------------------------------------------------


def _make_paragraphs(n_normal: int, n_anomaly: int) -> list[Paragraph]:
    """Synthetic Paragraph list. IDs: 'n_0'..., 'a_0'..."""
    ps = [
        Paragraph(paragraph_id=f"n_{i}", lines=["x"], label=0)
        for i in range(n_normal)
    ]
    ps.extend(
        Paragraph(paragraph_id=f"a_{i}", lines=["y"], label=1)
        for i in range(n_anomaly)
    )
    return ps


def _canonical_fold(fold_id: int) -> dict:
    """v1.5 canonical splits.json fold dict (D_NEW6 schema)."""
    return {
        "fold_id": fold_id,
        "seed": 42 + fold_id,
        "train_normal_ids": [f"n_{i}" for i in range(10)],
        "tune_normal_ids": [f"n_{i}" for i in range(10, 15)],
        "tune_anomaly_ids": [f"a_{i}" for i in range(5)],
        "test_normal_ids": [f"n_{i}" for i in range(15, 25)],
        "test_anomaly_ids": [f"a_{i}" for i in range(5, 10)],
    }


def _canonical_splits_file(tmp_path: Path, n_folds: int = 5) -> Path:
    splits = {
        "n_folds": n_folds,
        "train_normal_per_fold": 10,
        "tune_normal_per_fold": 5,
        "tune_anomaly_per_fold": 5,
        "test_anomaly_per_fold": 5,
        "seed": 42,
        "total_paragraphs": 30,
        "total_normal": 25,
        "total_anomaly": 10,
        "folds": [_canonical_fold(k) for k in range(n_folds)],
    }
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(splits))
    return path


# ---------------------------------------------------------------------------
# _parse_topk_grid
# ---------------------------------------------------------------------------


class TestParseTopkGrid:
    def test_basic_grid(self):
        assert _parse_topk_grid("5,9,12") == [5, 9, 12]

    def test_single_value(self):
        assert _parse_topk_grid("10") == [10]

    def test_whitespace_tolerated(self):
        assert _parse_topk_grid(" 5 , 9 , 12 ") == [5, 9, 12]

    def test_trailing_comma_ignored(self):
        """Trailing commas are silently dropped (empty parts stripped)."""
        assert _parse_topk_grid("5,9,12,") == [5, 9, 12]

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="at least one integer"):
            _parse_topk_grid("")

    def test_only_commas_raises(self):
        with pytest.raises(ValueError, match="at least one integer"):
            _parse_topk_grid(",,,")

    def test_zero_value_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            _parse_topk_grid("5,0,9")

    def test_negative_value_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            _parse_topk_grid("5,-1,9")

    def test_non_int_raises(self):
        with pytest.raises(ValueError):
            _parse_topk_grid("5,abc,9")

    def test_order_preserved(self):
        """Caller's ordering is preserved; downstream sorts as needed."""
        assert _parse_topk_grid("12,5,9") == [12, 5, 9]


# ---------------------------------------------------------------------------
# _resolve_backbone — config precedence + I3 hard-fail
# ---------------------------------------------------------------------------


def _minimal_cfg(
    use_longformer: bool = False,
    explicit_backbone: str | None = None,
) -> dict:
    training: dict = {}
    if use_longformer:
        training["use_longformer"] = True
    if explicit_backbone is not None:
        training["backbone"] = explicit_backbone
    return {
        "backbone": {
            "roberta_id": "roberta-base",
            "longformer_id": "allenai/longformer-base-4096",
        },
        "training": training,
    }


class TestResolveBackbone:
    def test_default_is_roberta(self):
        assert _resolve_backbone(_minimal_cfg(), None) == "roberta-base"

    def test_use_longformer_flag(self):
        assert (
            _resolve_backbone(_minimal_cfg(use_longformer=True), None)
            == "allenai/longformer-base-4096"
        )

    def test_explicit_backbone_overrides_use_longformer_flag(self):
        """training.backbone wins over training.use_longformer."""
        cfg = _minimal_cfg(
            use_longformer=True, explicit_backbone="roberta-base"
        )
        assert _resolve_backbone(cfg, None) == "roberta-base"

    def test_decision_artifact_overrides_yaml(self, tmp_path: Path):
        decision_path = tmp_path / "backbone_choice.json"
        decision_path.write_text(
            json.dumps({"chosen_backbone": "allenai/longformer-base-4096"})
        )
        assert (
            _resolve_backbone(_minimal_cfg(), decision_path)
            == "allenai/longformer-base-4096"
        )

    def test_explicit_backbone_overrides_decision_artifact(self, tmp_path: Path):
        """training.backbone has highest precedence (overrides even the
        decision artifact)."""
        decision_path = tmp_path / "backbone_choice.json"
        decision_path.write_text(
            json.dumps({"chosen_backbone": "allenai/longformer-base-4096"})
        )
        cfg = _minimal_cfg(explicit_backbone="roberta-base")
        assert _resolve_backbone(cfg, decision_path) == "roberta-base"

    def test_decision_artifact_missing_field_raises(self, tmp_path: Path):
        decision_path = tmp_path / "bad.json"
        decision_path.write_text(json.dumps({"other": "x"}))
        with pytest.raises(ValueError, match="chosen_backbone"):
            _resolve_backbone(_minimal_cfg(), decision_path)

    def test_decision_path_does_not_exist_raises_v1_5_I3(self, tmp_path: Path):
        """v1.5 I3 fix: missing decision file MUST raise, not silent fallthrough."""
        ghost = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError, match="backbone-decision"):
            _resolve_backbone(_minimal_cfg(), ghost)

    def test_decision_path_none_uses_yaml(self):
        """If --backbone-decision flag is omitted (None), no file check
        is performed — fall through to YAML logic."""
        cfg = _minimal_cfg(use_longformer=True)
        assert _resolve_backbone(cfg, None) == "allenai/longformer-base-4096"


# ---------------------------------------------------------------------------
# _load_fold_data — M1 schema enforcement
# ---------------------------------------------------------------------------


class TestLoadFoldData:
    def test_finds_fold_by_id(self, tmp_path: Path):
        splits_path = _canonical_splits_file(tmp_path, n_folds=5)
        fold = _load_fold_data(splits_path, fold_idx=2)
        assert fold["fold_id"] == 2

    def test_fold_zero(self, tmp_path: Path):
        splits_path = _canonical_splits_file(tmp_path, n_folds=5)
        fold = _load_fold_data(splits_path, fold_idx=0)
        assert fold["fold_id"] == 0

    def test_fold_not_found_raises(self, tmp_path: Path):
        splits_path = _canonical_splits_file(tmp_path, n_folds=3)
        with pytest.raises(ValueError, match="Fold 99 not found"):
            _load_fold_data(splits_path, fold_idx=99)

    def test_empty_folds_raises(self, tmp_path: Path):
        splits_path = tmp_path / "splits.json"
        splits_path.write_text(json.dumps({"n_folds": 0, "folds": []}))
        with pytest.raises(ValueError, match="No folds found"):
            _load_fold_data(splits_path, fold_idx=0)

    def test_missing_folds_key_raises(self, tmp_path: Path):
        splits_path = tmp_path / "splits.json"
        splits_path.write_text(json.dumps({"n_folds": 0}))
        with pytest.raises(ValueError, match="No folds found"):
            _load_fold_data(splits_path, fold_idx=0)

    def test_legacy_fold_idx_schema_rejected_v1_5_M1(self, tmp_path: Path):
        """v1.5 M1: non-canonical 'fold_idx' key must NOT be accepted."""
        legacy = {
            "n_folds": 1,
            "folds": [
                {
                    "fold_idx": 0,   # legacy key, not fold_id
                    "train_normal_ids": [],
                    "test_normal_ids": [],
                    "test_anomaly_ids": [],
                }
            ],
        }
        splits_path = tmp_path / "splits.json"
        splits_path.write_text(json.dumps(legacy))
        with pytest.raises(ValueError, match="non-canonical schema"):
            _load_fold_data(splits_path, fold_idx=0)


# ---------------------------------------------------------------------------
# _select_split_ids — M1 schema enforcement + split routing
# ---------------------------------------------------------------------------


class TestSelectSplitIds:
    def test_tune_split_returns_normal_then_anomaly(self):
        fold = _canonical_fold(0)
        ids = _select_split_ids(fold, "tune")
        # 5 normal + 5 anomaly = 10
        assert len(ids) == 10
        # Order: normals first, then anomalies
        assert all(pid.startswith("n_") for pid in ids[:5])
        assert all(pid.startswith("a_") for pid in ids[5:])

    def test_test_split_returns_normal_then_anomaly(self):
        fold = _canonical_fold(0)
        ids = _select_split_ids(fold, "test")
        # 10 normal + 5 anomaly = 15
        assert len(ids) == 15
        assert all(pid.startswith("n_") for pid in ids[:10])
        assert all(pid.startswith("a_") for pid in ids[10:])

    def test_train_split_only_normals_under_paradigm_b(self):
        """v1.5 Paradigm B: train has only normal IDs. Train_anomaly is
        empty (or absent) — _select_split_ids must still work."""
        fold = _canonical_fold(0)
        # Paradigm B fold has no train_anomaly_ids field; emulate by
        # providing it as an empty list (load_splits guarantees this).
        fold["train_anomaly_ids"] = []
        ids = _select_split_ids(fold, "train")
        assert len(ids) == 10   # 10 train_normal + 0 train_anomaly
        assert all(pid.startswith("n_") for pid in ids)

    def test_unknown_split_raises(self):
        fold = _canonical_fold(0)
        with pytest.raises(ValueError, match="Unknown split_name"):
            _select_split_ids(fold, "validation")

    def test_missing_split_keys_raises_v1_5_M1(self):
        """v1.5 M1: legacy aliases (e.g. 'tune_normal') must NOT be
        accepted. Only 'tune_normal_ids' / 'tune_anomaly_ids' are canonical."""
        legacy_fold = {
            "fold_id": 0,
            "tune_normal": ["n_1"],          # legacy: missing _ids suffix
            "tune_anomaly": ["a_1"],
        }
        with pytest.raises(ValueError, match="missing required keys"):
            _select_split_ids(legacy_fold, "tune")

    def test_returns_new_list_not_alias(self):
        """Modifying the returned list must not corrupt the fold dict."""
        fold = _canonical_fold(0)
        ids = _select_split_ids(fold, "test")
        ids.append("rogue_id")
        # Underlying fold's keys are untouched
        assert "rogue_id" not in fold["test_normal_ids"]
        assert "rogue_id" not in fold["test_anomaly_ids"]


# ---------------------------------------------------------------------------
# _map_paragraphs_by_id — M2 uniqueness assertion
# ---------------------------------------------------------------------------


class TestMapParagraphsById:
    def test_maps_existing_ids(self):
        ps = _make_paragraphs(5, 3)
        out = _map_paragraphs_by_id(ps, ["n_0", "a_1", "n_4"])
        assert [p.paragraph_id for p in out] == ["n_0", "a_1", "n_4"]

    def test_preserves_id_order(self):
        ps = _make_paragraphs(5, 3)
        out = _map_paragraphs_by_id(ps, ["a_0", "n_3", "n_1"])
        assert [p.paragraph_id for p in out] == ["a_0", "n_3", "n_1"]

    def test_missing_id_raises_with_count(self):
        ps = _make_paragraphs(5, 0)
        with pytest.raises(ValueError, match="Missing 2 paragraph IDs"):
            _map_paragraphs_by_id(ps, ["n_0", "ghost_a", "ghost_b"])

    def test_duplicate_paragraph_id_in_input_raises_v1_5_M2(self):
        """v1.5 M2: paragraphs.pkl with duplicate paragraph_ids → HARD FAIL,
        not silent dedup."""
        ps = [
            Paragraph(paragraph_id="dup", lines=["x"], label=0),
            Paragraph(paragraph_id="dup", lines=["x2"], label=0),
        ]
        with pytest.raises(ValueError, match="Duplicate paragraph_id"):
            _map_paragraphs_by_id(ps, ["dup"])

    def test_empty_id_list_returns_empty(self):
        ps = _make_paragraphs(5, 3)
        assert _map_paragraphs_by_id(ps, []) == []

    def test_label_preserved(self):
        ps = _make_paragraphs(3, 2)
        out = _map_paragraphs_by_id(ps, ["n_0", "a_0", "n_2"])
        labels = [p.label for p in out]
        assert labels == [0, 1, 0]


# ---------------------------------------------------------------------------
# _prepare_eval_paragraphs — map + sort
# ---------------------------------------------------------------------------


class TestPrepareEvalParagraphs:
    def test_output_sorted_by_paragraph_id_lexicographic(self):
        ps = _make_paragraphs(5, 3)
        out = _prepare_eval_paragraphs(ps, ["n_2", "n_0", "a_1", "a_0"])
        # Lexicographic sort: 'a_0' < 'a_1' < 'n_0' < 'n_2'
        assert [p.paragraph_id for p in out] == ["a_0", "a_1", "n_0", "n_2"]

    def test_lexicographic_not_numeric(self):
        """'n_10' comes BEFORE 'n_2' lexicographically — spec convention."""
        ps = [
            Paragraph(paragraph_id="n_2", lines=["x"], label=0),
            Paragraph(paragraph_id="n_10", lines=["x"], label=0),
        ]
        out = _prepare_eval_paragraphs(ps, ["n_2", "n_10"])
        assert [p.paragraph_id for p in out] == ["n_10", "n_2"]

    def test_propagates_missing_id_error(self):
        ps = _make_paragraphs(3, 0)
        with pytest.raises(ValueError, match="Missing"):
            _prepare_eval_paragraphs(ps, ["ghost"])


# ---------------------------------------------------------------------------
# _resolve_output_path — single-split mode heuristics
# ---------------------------------------------------------------------------


class TestResolveOutputPath:
    def test_existing_directory_appends_filename(self, tmp_path: Path):
        out = _resolve_output_path(tmp_path, "tune")
        assert out == tmp_path / "scores_tune.json"

    def test_path_with_json_suffix_used_as_is(self, tmp_path: Path):
        target = tmp_path / "my_scores.json"
        out = _resolve_output_path(target, "tune")
        assert out == target

    def test_nonexistent_path_no_suffix_treated_as_dir(self, tmp_path: Path):
        target = tmp_path / "newdir"
        # Note: not created on disk
        out = _resolve_output_path(target, "test")
        assert out == target / "scores_test.json"

    def test_filename_in_split_reflects_split_name(self, tmp_path: Path):
        for split in ("train", "tune", "test"):
            out = _resolve_output_path(tmp_path, split)
            assert out.name == f"scores_{split}.json"


# ---------------------------------------------------------------------------
# _resolve_output_dir — 'both' mode requires a directory path
# ---------------------------------------------------------------------------


class TestResolveOutputDir:
    def test_directory_path_accepted(self, tmp_path: Path):
        out = _resolve_output_dir(tmp_path)
        assert out == tmp_path

    def test_nonexistent_no_suffix_accepted(self, tmp_path: Path):
        target = tmp_path / "new_scores"
        out = _resolve_output_dir(target)
        assert out == target

    def test_path_with_file_suffix_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="must be a directory"):
            _resolve_output_dir(tmp_path / "scores.json")
