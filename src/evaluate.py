"""Cross-fold test-set evaluation per LogFiT paper §IV-A.

For each fold:
  1. Load tuning_grid.json -> (best_top_k, best_threshold)
  2. Load scores_test.json
  3. Apply the operating point to test paragraphs -> confusion matrix
  4. Record per-fold P / R / F1 / Specificity

Across folds:
  - Cross-fold mean and SAMPLE standard deviation (ddof=1, paper convention)
  - Persist per-fold raw values + operating points for audit

Reuses classification + metrics logic from `src.tune_threshold` so the
test-set rule is bit-identical to the tune-set rule (single source of
truth: `accuracy < threshold -> anomaly`).

Reference: LogFiT paper §IV-A Table I;
docs/Logfit-repro-spec-v1.2.md Component 7;
docs/logfit-repro-decisions-v1.5.md D_NEW7.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.tune_threshold import (
    _classify_paragraph,
    _compute_metrics,
    load_split_scores,
)
from src.types import DatasetResults, FoldMetrics, SplitScores, TuningCell, TuningGrid
from src.utils.io import load_json, save_json


# ---------------------------------------------------------------------------
# Tuning-grid deserialization (JSON -> dataclasses)
# ---------------------------------------------------------------------------


def load_tuning_grid(path: Path | str) -> TuningGrid:
    """Load and validate a tuning_grid.json file written by `src.tune_threshold`.

    Backward-compatible with pre-v1.5 cells that lack tp/fp/tn/fn fields:
    those default to 0. The four required top-level keys are
    `cells`, `best_top_k`, `best_threshold`, `train_top1_acc`.
    """
    data = load_json(path)
    required = ["cells", "best_top_k", "best_threshold", "train_top1_acc"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"{path} is missing required keys {missing}. "
            f"Not a tuning_grid.json artifact. Regenerate with "
            f"src.tune_threshold (v1.5+)."
        )
    cells: list[TuningCell] = []
    for c in data["cells"]:
        cells.append(
            TuningCell(
                top_k=int(c["top_k"]),
                threshold=float(c["threshold"]),
                precision=float(c["precision"]),
                recall=float(c["recall"]),
                f1=float(c["f1"]),
                specificity=float(c["specificity"]),
                tp=int(c.get("tp", 0)),
                fp=int(c.get("fp", 0)),
                tn=int(c.get("tn", 0)),
                fn=int(c.get("fn", 0)),
            )
        )
    return TuningGrid(
        cells=cells,
        best_top_k=int(data["best_top_k"]),
        best_threshold=float(data["best_threshold"]),
        train_top1_acc=float(data["train_top1_acc"]),
    )


# ---------------------------------------------------------------------------
# Pure-logic helpers
# ---------------------------------------------------------------------------


def evaluate_fold(
    test_scores: SplitScores,
    best_top_k: int,
    best_threshold: float,
    fold_idx: int,
) -> FoldMetrics:
    """Apply (best_top_k, best_threshold) to test scores -> FoldMetrics.

    The classification rule and metric computations are imported from
    `src.tune_threshold` so the test-set and tune-set behavior cannot
    drift apart.

    `fold_idx` is recorded in the returned `FoldMetrics.fold_idx` for
    audit. Callers should pass the fold ID from the splits artifact.
    """
    predictions = [
        _classify_paragraph(s, best_top_k, best_threshold)
        for s in test_scores.scores
    ]
    labels = [s.label for s in test_scores.scores]
    m = _compute_metrics(predictions, labels)
    return FoldMetrics(
        fold_idx=int(fold_idx),
        precision=float(m["precision"]),
        recall=float(m["recall"]),
        f1=float(m["f1"]),
        specificity=float(m["specificity"]),
        tp=int(m["tp"]),
        fp=int(m["fp"]),
        tn=int(m["tn"]),
        fn=int(m["fn"]),
        top_k=int(best_top_k),
        threshold=float(best_threshold),
    )


def _safe_std(values: list[float]) -> float:
    """Sample standard deviation (ddof=1), 0.0 for n < 2.

    `statistics.stdev` requires at least 2 data points and uses Bessel's
    correction (ddof=1) by default — matches the paper's convention.
    For n < 2 (e.g. a single-fold run or empty list), returns 0.0 rather
    than raising, so the aggregation report doesn't crash on degenerate
    runs.
    """
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def aggregate_folds(
    fold_metrics: list[FoldMetrics],
    dataset: str,
    window_seconds: int | None,
    backbone: str,
) -> DatasetResults:
    """Compute cross-fold mean and sample std (ddof=1) for P / R / F1 / Spec.

    Also packs the per-fold raw values and operating points into
    `DatasetResults.per_fold_values` and `per_fold_operating_points` for
    downstream audit (so the dissertation can show: "fold 3 had K=12, θ=0.79,
    F1=0.81").

    Args:
        fold_metrics: at least one FoldMetrics; std is 0 for n < 2.
        dataset: dataset name (e.g. "hdfs"); recorded as-is.
        window_seconds: window size in seconds, or None for HDFS.
        backbone: backbone model name (e.g. "roberta-base"); recorded as-is.

    Raises:
        ValueError if fold_metrics is empty.
    """
    if not fold_metrics:
        raise ValueError(
            "aggregate_folds requires at least one FoldMetrics."
        )

    p_values = [m.precision for m in fold_metrics]
    r_values = [m.recall for m in fold_metrics]
    f1_values = [m.f1 for m in fold_metrics]
    spec_values = [m.specificity for m in fold_metrics]

    return DatasetResults(
        dataset=dataset,
        window_seconds=window_seconds,
        backbone=backbone,
        p_mean=statistics.mean(p_values),
        p_std=_safe_std(p_values),
        r_mean=statistics.mean(r_values),
        r_std=_safe_std(r_values),
        f1_mean=statistics.mean(f1_values),
        f1_std=_safe_std(f1_values),
        spec_mean=statistics.mean(spec_values),
        spec_std=_safe_std(spec_values),
        per_fold_values={
            "precision": list(p_values),
            "recall": list(r_values),
            "f1": list(f1_values),
            "specificity": list(spec_values),
        },
        per_fold_operating_points=[
            {
                "fold_idx": m.fold_idx,
                "top_k": m.top_k,
                "threshold": m.threshold,
                "tp": m.tp,
                "fp": m.fp,
                "tn": m.tn,
                "fn": m.fn,
            }
            for m in fold_metrics
        ],
    )


# ---------------------------------------------------------------------------
# Full pipeline (per-fold inputs -> aggregated result)
# ---------------------------------------------------------------------------


def evaluate_all_folds(
    scores_root: Path,
    n_folds: int,
    dataset: str,
    window_seconds: int | None,
    backbone: str,
    fold_dir_template: str = "fold_{fold_idx}",
    tuning_grid_filename: str = "tuning_grid.json",
    scores_test_filename: str = "scores_test.json",
) -> DatasetResults:
    """Process all N folds and aggregate. Each fold directory must contain
    both `tuning_grid.json` (from tune_threshold.py) and `scores_test.json`
    (from score.py --split test|both).

    fold_dir_template: layout under scores_root. Default 'fold_{fold_idx}'
    matches score.sh's output layout.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")

    scores_root = Path(scores_root)
    fold_metrics_list: list[FoldMetrics] = []

    for k in range(n_folds):
        fold_dir = scores_root / fold_dir_template.format(fold_idx=k)
        if not fold_dir.exists():
            raise FileNotFoundError(
                f"Fold directory does not exist: {fold_dir}. "
                f"Expected {n_folds} folds under {scores_root}."
            )

        tuning_path = fold_dir / tuning_grid_filename
        scores_path = fold_dir / scores_test_filename
        if not tuning_path.exists():
            raise FileNotFoundError(
                f"Missing {tuning_grid_filename} in {fold_dir}. "
                f"Run src.tune_threshold for fold {k} first."
            )
        if not scores_path.exists():
            raise FileNotFoundError(
                f"Missing {scores_test_filename} in {fold_dir}. "
                f"Run src.score --split test (or both) for fold {k} first."
            )

        tuning_grid = load_tuning_grid(tuning_path)
        test_scores = load_split_scores(scores_path)

        if test_scores.split_name != "test":
            raise ValueError(
                f"{scores_path} has split_name={test_scores.split_name!r}, "
                f"expected 'test'. evaluate.py operates only on test scores."
            )

        fm = evaluate_fold(
            test_scores=test_scores,
            best_top_k=tuning_grid.best_top_k,
            best_threshold=tuning_grid.best_threshold,
            fold_idx=k,
        )
        fold_metrics_list.append(fm)

    return aggregate_folds(
        fold_metrics_list,
        dataset=dataset,
        window_seconds=window_seconds,
        backbone=backbone,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _resolve_output_path(output: Path) -> Path:
    """Append `dataset_results.json` if output is a directory."""
    if output.exists() and output.is_dir():
        return output / "dataset_results.json"
    if output.suffix:
        return output
    return output / "dataset_results.json"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description=(
            "Cross-fold test-set evaluation per LogFiT paper §IV-A. "
            "Applies each fold's (best_top_k, best_threshold) from "
            "tuning_grid.json to the corresponding scores_test.json, "
            "aggregates with sample std (ddof=1)."
        ),
    )
    parser.add_argument(
        "--scores-root",
        type=Path,
        required=True,
        help="Root directory containing fold_{0..N-1}/ subdirectories, "
             "each with tuning_grid.json and scores_test.json.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hdfs", "bgl", "tbird"],
        help="Dataset name (for the DatasetResults record).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="Backbone name (e.g. 'roberta-base', 'allenai/longformer-base-4096').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write dataset_results.json (or directory; "
             "auto-appends 'dataset_results.json' in that case).",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=None,
        help="Window size in seconds. Omit for HDFS (session-based). "
             "BGL/TB use 30 per paper §IV-A.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds to aggregate (default 5, paper §IV-A).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    results = evaluate_all_folds(
        scores_root=args.scores_root,
        n_folds=args.n_folds,
        dataset=args.dataset,
        window_seconds=args.window_seconds,
        backbone=args.backbone,
    )

    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(asdict(results), output_path)

    # Console summary — matches the dissertation Table I format
    print(f"[evaluate.py] Wrote {output_path}")
    window_str = (
        f"{results.window_seconds}s" if results.window_seconds else "session"
    )
    print(
        f"  Dataset:    {results.dataset} "
        f"(backbone={results.backbone}, window={window_str})"
    )
    print(f"  Folds:      {args.n_folds}")
    print(
        f"  Precision:  {results.p_mean:.4f} ± {results.p_std:.4f}"
    )
    print(f"  Recall:     {results.r_mean:.4f} ± {results.r_std:.4f}")
    print(f"  F1:         {results.f1_mean:.4f} ± {results.f1_std:.4f}")
    print(f"  Spec:       {results.spec_mean:.4f} ± {results.spec_std:.4f}")
    print("  Operating points (K, θ) per fold:")
    for op in results.per_fold_operating_points:
        print(
            f"    fold {op['fold_idx']}: K={op['top_k']}, "
            f"θ={op['threshold']:.4f}  "
            f"(tp={op['tp']}, fp={op['fp']}, tn={op['tn']}, fn={op['fn']})"
        )


if __name__ == "__main__":
    main()
