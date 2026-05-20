"""Per-fold threshold tuning per LogFiT paper §III-C.

For each fold, search a 3 × 3 grid of (top_k, threshold) on the tune split
and pick the F1-best (K, θ) operating point. Tie-break: smaller K first,
then larger θ.

Classification rule (paper §III-C):
    predict ANOMALY if topk_accuracy(paragraph) < threshold
    predict NORMAL otherwise

Threshold grid (paper §III-C):
    linspace(train_top1_acc - 0.1, train_top1_acc, 3)
    Clamped to [0, 1]. With n=3 thresholds and Δ=0.1, the grid is evenly
    spaced at [top1-0.1, top1-0.05, top1].

Top-k grid: {5, 9, 12} by default (paper §III-C).

Reads scores_tune.json in v1.5 D_NEW7 records format. Writes a TuningGrid
artifact that evaluate.py consumes to apply (best_K, best_θ) to the test
split.

Reference: LogFiT paper Almodovar et al. 2024 §III-C;
docs/logfit-repro-decisions-v1.5.md D_NEW7.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.types import (
    ParagraphScore,
    SplitScores,
    TopKAccuracyRecord,
    TuningCell,
    TuningGrid,
)
from src.utils.io import ensure_dir, load_json, save_json


# ---------------------------------------------------------------------------
# Defaults (paper §III-C)
# ---------------------------------------------------------------------------

DEFAULT_TOPK_GRID: list[int] = [5, 9, 12]
DEFAULT_N_THRESHOLDS: int = 3
DEFAULT_THRESHOLD_DELTA: float = 0.1


# ---------------------------------------------------------------------------
# Score deserialization (JSON -> dataclasses)
# ---------------------------------------------------------------------------


def _deserialize_paragraph_score(d: dict[str, Any]) -> ParagraphScore:
    """Reconstruct a ParagraphScore from its asdict() JSON shape."""
    records = [
        TopKAccuracyRecord(top_k=int(r["top_k"]), accuracy=float(r["accuracy"]))
        for r in d["topk_accuracies"]
    ]
    return ParagraphScore(
        paragraph_id=str(d["paragraph_id"]),
        label=int(d["label"]),
        topk_accuracies=records,
        n_masked_total=int(d.get("n_masked_total", 0)),
        n_passes=int(d.get("n_passes", 1)),
    )


def load_split_scores(path: Path | str) -> SplitScores:
    """Load and validate a scores_*.json file written by `src.score`.

    Asserts the canonical v1.5 D_NEW7 records-format shape:
      - top-level `split_name`, `topk_grid`, `scores`, `n_passes`
      - each `scores[i].topk_accuracies` is a list of `{top_k, accuracy}`

    Raises ValueError with a clear migration message if the legacy
    `dict[int, float]` form is detected.
    """
    data = load_json(path)
    if "scores" not in data:
        raise ValueError(
            f"{path} has no 'scores' field — not a scores_*.json artifact."
        )

    # Sanity: detect legacy dict form on the first record
    if data["scores"]:
        first = data["scores"][0]
        topk_obj = first.get("topk_accuracies")
        if isinstance(topk_obj, dict):
            raise ValueError(
                f"{path} uses legacy dict-form topk_accuracies. "
                f"Regenerate with src.score (v1.5+) to get the list-of-records "
                f"format (D_NEW7)."
            )
        if not isinstance(topk_obj, list):
            raise ValueError(
                f"{path} has malformed topk_accuracies on first record: "
                f"{type(topk_obj).__name__} (expected list)."
            )

    scores = [_deserialize_paragraph_score(s) for s in data["scores"]]
    return SplitScores(
        split_name=str(data["split_name"]),
        topk_grid=[int(k) for k in data["topk_grid"]],
        scores=scores,
        n_passes=int(data.get("n_passes", 1)),
    )


# ---------------------------------------------------------------------------
# Pure-logic helpers
# ---------------------------------------------------------------------------


def _build_threshold_grid(
    train_top1_acc: float,
    n_thresholds: int = DEFAULT_N_THRESHOLDS,
    delta: float = DEFAULT_THRESHOLD_DELTA,
) -> list[float]:
    """Construct an evenly-spaced threshold grid per paper §III-C.

    Returns linspace(max(0, top1 - delta), top1, n_thresholds), clamped at
    [0, 1]. For n=1, returns [top1].

    Why clamp at 0: train_top1_acc < delta is rare but possible (early
    training, broken models). Without clamping, the grid would include
    negative thresholds, which are degenerate (every paragraph predicted
    normal — equivalent to threshold=0).
    """
    if n_thresholds < 1:
        raise ValueError(
            f"n_thresholds must be >= 1, got {n_thresholds}"
        )
    if not (0.0 <= train_top1_acc <= 1.0):
        raise ValueError(
            f"train_top1_acc must be in [0, 1], got {train_top1_acc}"
        )
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")

    upper = train_top1_acc
    lower = max(0.0, train_top1_acc - delta)

    if n_thresholds == 1:
        return [upper]

    step = (upper - lower) / (n_thresholds - 1)
    return [lower + step * i for i in range(n_thresholds)]


def _topk_accuracy_for_paragraph(
    score: ParagraphScore, top_k: int
) -> float:
    """Look up the accuracy record matching `top_k`.

    Raises ValueError if no record matches — score.py guarantees every
    requested K is scored, so a missing K here means the scoring grid and
    the tuning grid are inconsistent (a configuration error).
    """
    for record in score.topk_accuracies:
        if record.top_k == top_k:
            return record.accuracy
    available = [r.top_k for r in score.topk_accuracies]
    raise ValueError(
        f"top_k={top_k} not found for paragraph {score.paragraph_id} "
        f"(available: {available}). The scoring grid and the tuning grid "
        f"must overlap; re-run score.py with --topk-grid covering "
        f"{top_k}, or tune over the available K values."
    )


def _classify_paragraph(
    score: ParagraphScore, top_k: int, threshold: float
) -> int:
    """Classify per paper §III-C: 1 (anomaly) iff topk_accuracy < threshold.

    Strict `<` (not `<=`) — the boundary case (accuracy == threshold) is
    classified as normal. Zero-mask paragraphs (n_masked_total == 0,
    accuracy=1.0 by D_NEW8 convention) are predicted normal for any
    threshold ≤ 1.0, which matches the paper's "pass by default" treatment.
    """
    accuracy = _topk_accuracy_for_paragraph(score, top_k)
    return 1 if accuracy < threshold else 0


def _compute_metrics(
    predictions: list[int], labels: list[int]
) -> dict[str, float | int]:
    """Confusion-matrix-derived P, R, F1, Specificity.

    Division-by-zero conventions:
      - precision = 0 if (tp + fp) == 0 (no positive predictions)
      - recall = 0 if (tp + fn) == 0 (no actual positives — degenerate eval)
      - f1 = 0 if (precision + recall) == 0
      - specificity = 0 if (tn + fp) == 0 (no actual negatives)

    Returns a dict with `precision`, `recall`, `f1`, `specificity`,
    `tp`, `fp`, `tn`, `fn`.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"predictions ({len(predictions)}) and labels ({len(labels)}) "
            f"must have the same length."
        )

    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _evaluate_cell(
    scores: list[ParagraphScore], top_k: int, threshold: float
) -> TuningCell:
    """Evaluate one (top_k, threshold) cell on the given scores list."""
    predictions = [_classify_paragraph(s, top_k, threshold) for s in scores]
    labels = [s.label for s in scores]
    m = _compute_metrics(predictions, labels)
    return TuningCell(
        top_k=int(top_k),
        threshold=float(threshold),
        precision=float(m["precision"]),
        recall=float(m["recall"]),
        f1=float(m["f1"]),
        specificity=float(m["specificity"]),
        tp=int(m["tp"]),
        fp=int(m["fp"]),
        tn=int(m["tn"]),
        fn=int(m["fn"]),
    )


def _find_best_cell(cells: list[TuningCell]) -> tuple[int, float]:
    """Pick the (top_k, threshold) cell with the best F1.

    Tie-break (in order):
      1. higher F1
      2. smaller top_k (paper §III-C — prefer the lower-K operating point)
      3. larger threshold (more aggressive — prefer the larger θ when
         F1 and K are tied, biasing toward higher recall)

    Equivalent sort key: (-f1, top_k, -threshold), then take the first.
    """
    if not cells:
        raise ValueError("Cannot find best cell from an empty list.")
    sorted_cells = sorted(
        cells, key=lambda c: (-c.f1, c.top_k, -c.threshold)
    )
    best = sorted_cells[0]
    return int(best.top_k), float(best.threshold)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def tune_threshold(
    tune_scores: SplitScores,
    train_top1_acc: float,
    topk_grid: list[int] | None = None,
    n_thresholds: int = DEFAULT_N_THRESHOLDS,
    threshold_delta: float = DEFAULT_THRESHOLD_DELTA,
) -> TuningGrid:
    """Search the (top_k, threshold) grid for the F1-best operating point.

    Args:
        tune_scores: SplitScores from `score.py --split tune`.
        train_top1_acc: Last-epoch top-1 accuracy from train_log.json.
        topk_grid: K values to search (default [5, 9, 12]).
        n_thresholds: number of thresholds in the linspace (default 3).
        threshold_delta: bandwidth below train_top1_acc (default 0.1).

    Returns:
        TuningGrid with all evaluated cells + the best (top_k, threshold).
    """
    if topk_grid is None:
        topk_grid = DEFAULT_TOPK_GRID
    if not topk_grid:
        raise ValueError("topk_grid must contain at least one K value.")

    threshold_grid = _build_threshold_grid(
        train_top1_acc=train_top1_acc,
        n_thresholds=n_thresholds,
        delta=threshold_delta,
    )

    cells: list[TuningCell] = []
    for k in topk_grid:
        for theta in threshold_grid:
            cells.append(_evaluate_cell(tune_scores.scores, k, theta))

    best_k, best_theta = _find_best_cell(cells)
    return TuningGrid(
        cells=cells,
        best_top_k=best_k,
        best_threshold=best_theta,
        train_top1_acc=float(train_top1_acc),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_topk_grid(value: str) -> list[int]:
    """Parse '5,9,12' -> [5, 9, 12]. Same convention as score.py."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("topk_grid must contain at least one integer")
    grid: list[int] = []
    for p in parts:
        k = int(p)
        if k <= 0:
            raise ValueError(f"topk values must be positive, got {k}")
        grid.append(k)
    return grid


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.tune_threshold",
        description=(
            "Per-fold threshold tuning per LogFiT paper §III-C. "
            "Searches a (top_k × threshold) grid on the tune set and picks "
            "the F1-best cell. Outputs tuning_grid.json for evaluate.py."
        ),
    )
    parser.add_argument(
        "--scores-tune",
        type=Path,
        required=True,
        help="Path to scores_tune.json (v1.5 D_NEW7 records format).",
    )
    parser.add_argument(
        "--train-log",
        type=Path,
        required=True,
        help="Path to train_log.json for the same fold (provides "
             "train_top1_acc, the anchor for the threshold grid).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write tuning_grid.json (or directory; "
             "auto-appends 'tuning_grid.json' in that case).",
    )
    parser.add_argument(
        "--topk-grid",
        type=str,
        default=None,
        help="Comma-separated K values (e.g. '5,9,12'). "
             f"Default: {','.join(str(k) for k in DEFAULT_TOPK_GRID)}.",
    )
    parser.add_argument(
        "--n-thresholds",
        type=int,
        default=DEFAULT_N_THRESHOLDS,
        help=f"Number of thresholds in the grid (default {DEFAULT_N_THRESHOLDS}).",
    )
    parser.add_argument(
        "--threshold-delta",
        type=float,
        default=DEFAULT_THRESHOLD_DELTA,
        help=f"Bandwidth below train_top1_acc (default {DEFAULT_THRESHOLD_DELTA}).",
    )
    return parser


def _resolve_output_path(output: Path) -> Path:
    """Resolve --output: append tuning_grid.json if output is a directory."""
    if output.exists() and output.is_dir():
        return output / "tuning_grid.json"
    if output.suffix:
        return output
    return output / "tuning_grid.json"


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load train_top1_acc from train_log.json
    train_log = load_json(args.train_log)
    if "train_top1_acc" not in train_log:
        raise ValueError(
            f"{args.train_log} has no 'train_top1_acc' field. "
            f"Regenerate train_log.json from training, or pass a valid file."
        )
    train_top1_acc = float(train_log["train_top1_acc"])

    # Load tune scores
    tune_scores = load_split_scores(args.scores_tune)
    if tune_scores.split_name != "tune":
        raise ValueError(
            f"{args.scores_tune} has split_name={tune_scores.split_name!r}, "
            f"expected 'tune'. tune_threshold.py operates only on tune scores."
        )

    # Top-k grid: CLI override > score's own topk_grid > default
    if args.topk_grid is not None:
        topk_grid = _parse_topk_grid(args.topk_grid)
    else:
        topk_grid = list(tune_scores.topk_grid) if tune_scores.topk_grid else DEFAULT_TOPK_GRID

    # Verify every tuning K is present in the scoring output
    scored_ks = set(tune_scores.topk_grid)
    missing_ks = [k for k in topk_grid if k not in scored_ks]
    if missing_ks:
        raise ValueError(
            f"Tuning K values {missing_ks} are not in the scoring output's "
            f"topk_grid {sorted(scored_ks)}. Re-run score.py with --topk-grid "
            f"covering all values you want to tune over."
        )

    # Run tuning
    grid = tune_threshold(
        tune_scores=tune_scores,
        train_top1_acc=train_top1_acc,
        topk_grid=topk_grid,
        n_thresholds=args.n_thresholds,
        threshold_delta=args.threshold_delta,
    )

    # Persist
    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(asdict(grid), output_path)

    # Console summary for the operator
    print(f"[tune_threshold.py] Wrote {output_path}")
    print(f"  train_top1_acc:   {grid.train_top1_acc:.4f}")
    print(f"  best (K, θ):      ({grid.best_top_k}, {grid.best_threshold:.4f})")
    best_cell = next(
        c for c in grid.cells
        if c.top_k == grid.best_top_k and c.threshold == grid.best_threshold
    )
    print(
        f"  tune metrics:     P={best_cell.precision:.4f} "
        f"R={best_cell.recall:.4f} F1={best_cell.f1:.4f} "
        f"Spec={best_cell.specificity:.4f}"
    )
    print(
        f"  tune confusion:   tp={best_cell.tp} fp={best_cell.fp} "
        f"tn={best_cell.tn} fn={best_cell.fn}"
    )


if __name__ == "__main__":
    main()
