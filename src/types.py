"""Data classes shared across the pipeline.

Pure data containers — no methodology decisions are embedded here. Every
methodology decision lives in src/* modules that produce or consume these
types. The dataclasses themselves are versioned with the spec.

Reference: logfit-repro-spec-v1.3.md Component 0 (notation) and Component 12 (API);
decisions-v1.5 D_NEW7/D_NEW8 for scoring schema additions.

v1.5 changes:
- TopKAccuracyRecord NEW (records-format scoring per D_NEW7).
- ParagraphScore.topk_accuracies: dict[int, float] -> list[TopKAccuracyRecord].
- ParagraphScore gains `n_masked_total` and `n_passes` audit fields.
- SplitScores gains `n_passes` (mirror at split level for downstream consumers).

v1.3 changes:
- DropCounters: split `singleton_window` into `singleton_window_normal` and
  `singleton_window_anomaly` per R2 F5; added `lines_missing_content` per
  D4 v1.3 / R2 F6; added `encoding_replacements_seen` per D20 NEW;
  documented `duplicate_blockid` semantics per D19 NEW; documented
  `lines_with_multiple_blockids` distinct-id semantics per D16 v1.3.
- PreparationSummary: added `encoding_offending_line_numbers` per D20 NEW.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

BackboneName = Literal["roberta-base", "allenai/longformer-base-4096"]
DatasetName = Literal["hdfs", "bgl", "tbird"]


@dataclass(frozen=True)
class Paragraph:
    """A log paragraph — one grouping unit (HDFS block_id session or
    BGL/TB time window)."""

    paragraph_id: str
    lines: list[str]
    label: int                              # 0 = normal, 1 = anomaly
    source_blockid: str | None = None       # HDFS only
    source_window_id: int | None = None      # BGL/TB only
    start_timestamp: float | None = None     # BGL/TB only


@dataclass
class DropCounters:
    """Surfaced in preparation_summary.json. v1.3 schema.

    Counters are organized by dataset origin. Non-zero values get explained
    or escalated per the disclosure standard.
    """

    # ----- HDFS counters -----
    no_blockid: int = 0
    missing_label_assertion_fired: int = 0    # must remain 0; non-zero = HARD FAIL
    duplicate_blockid: int = 0                # [D19 v1.3] CSV duplicate rows; conflicting raises
    lines_with_multiple_blockids: int = 0     # [D16 v1.3] distinct-id semantics; flagged at 5%

    # ----- BGL/TB counters -----
    unparseable_timestamp: int = 0            # HARD FAIL if rate > 0.1%
    empty_window: int = 0                     # vestigial; cannot occur with defaultdict logic
    singleton_window_normal: int = 0          # [R2 F5 v1.3] split from singleton_window
    singleton_window_anomaly: int = 0         # [R2 F5 v1.3] split — fires SUPERVISOR_FLAG if > 0

    # ----- Shared counters (NEW v1.3) -----
    lines_missing_content: int = 0            # [D4 v1.3 / R2 F6] 2-token lines rejected
    encoding_replacements_seen: int = 0       # [D20 NEW] \ufffd occurrences


@dataclass
class LengthDistribution:
    """Percentile summary of paragraph length (in words or in tokens)."""

    min: int
    max: int
    p50: float
    p80: float
    p95: float
    p99: float


@dataclass
class TokenLengthSummary:
    """Output of §1.6 token-length validation gate."""

    tokenizer: str
    distribution: LengthDistribution
    truncation_rate_at_backbone_limit: float


@dataclass
class PreparationSummary:
    """Persisted as preparation_summary.json per dataset (x window).

    v1.3: added `encoding_offending_line_numbers` for D20 audit trail.
    """

    dataset: DatasetName
    window_seconds: int | None
    total_lines_read: int
    total_paragraphs: int
    normal_paragraphs: int
    anomaly_paragraphs: int
    anomaly_rate: float
    drop_counters: DropCounters
    word_length_distribution: LengthDistribution
    token_length_distribution: TokenLengthSummary | None = None
    seed: int = 42
    # NEW v1.3 [D20] — capped at first 100 captured; defaults to empty
    encoding_offending_line_numbers: list[int] = field(default_factory=list)


@dataclass
class AllocationManifest:
    """The 25k normal + 2k anomaly sample budget per dataset [L21]."""

    dataset: DatasetName
    window_seconds: int | None
    seed: int
    normal_paragraph_ids: list[str]
    anomaly_paragraph_ids: list[str]


@dataclass
class FoldSpec:
    """Per-fold allocation per D14 (v1.2)."""

    fold_idx: int
    seed: int
    test_normal: list[str]
    train_normal: list[str]
    tune_normal: list[str]
    tune_anomaly: list[str]
    test_anomaly: list[str]


@dataclass
class TopKAccuracyRecord:
    """One top-k accuracy row for a paragraph.

    v1.5 D_NEW7: scoring output uses a list of these records instead of
    `dict[int, float]` to avoid JSON int-key coercion fragility. When the
    dict form was persisted, keys became strings on disk; downstream
    consumers indexing by `scores[K]` (int) hit KeyError on reload.

    A `top_k` is always a positive int; `accuracy` is in [0.0, 1.0].
    Records within a paragraph are ordered ascending by `top_k`.
    """

    top_k: int
    accuracy: float


@dataclass
class ParagraphScore:
    """Per-paragraph score artifact persisted by `src.score`.

    v1.5 changes (D_NEW7, D_NEW8):
    - `topk_accuracies` is now a list of TopKAccuracyRecord (records format).
    - `n_masked_total` exposes the total number of masked positions scored
      across all passes (sum across passes when `n_passes > 1`). A zero
      value means stochastic masking produced no scorable positions for
      this paragraph; per spec convention `topk_accuracies` records each
      carry `accuracy=1.0` in that degenerate case.
    - `n_passes` mirrors the run-level config for self-contained audits.
    """

    paragraph_id: str
    label: int
    topk_accuracies: list[TopKAccuracyRecord]
    n_masked_total: int = 0
    n_passes: int = 1


@dataclass
class SplitScores:
    """Wraps the scored output of one (fold, split) pair.

    v1.5: `n_passes` added at the split level so downstream consumers
    (tune_threshold, evaluate) don't have to peek inside `scores[0]` to
    learn the run configuration.
    """

    split_name: str
    topk_grid: list[int]
    scores: list[ParagraphScore]
    n_passes: int = 1


@dataclass
class TuningCell:
    top_k: int
    threshold: float
    precision: float
    recall: float
    f1: float
    specificity: float


@dataclass
class TuningGrid:
    cells: list[TuningCell]
    best_top_k: int
    best_threshold: float
    train_top1_acc: float


@dataclass
class FoldMetrics:
    fold_idx: int
    precision: float
    recall: float
    f1: float
    specificity: float
    tp: int
    fp: int
    tn: int
    fn: int
    top_k: int
    threshold: float


@dataclass
class DatasetResults:
    dataset: DatasetName
    window_seconds: int | None
    backbone: BackboneName
    p_mean: float
    p_std: float
    r_mean: float
    r_std: float
    f1_mean: float
    f1_std: float
    spec_mean: float
    spec_std: float
    per_fold_values: dict[str, list[float]]
    per_fold_operating_points: list[dict[str, float | int]]


@dataclass
class TrainingConfig:
    backbone: BackboneName
    epochs: int
    max_lr: float
    effective_batch_size: int
    physical_batch_size: int
    gradient_accumulation_steps: int

    one_cycle_pct_start: float = 0.3
    one_cycle_anneal_strategy: str = "cos"
    one_cycle_div_factor: float = 25.0
    one_cycle_final_div_factor: float = 1e4

    epoch_1_unfrozen_layers: tuple[int, ...] = (10, 11)
    epoch_2_unfrozen_layers: tuple[int, ...] = (8, 9, 10, 11)
    embeddings_frozen_until_epoch: int = 2

    seed: int = 42
    full_determinism: bool = True
    fp16: bool = True
    fp16_weight_divergence_threshold: float = 1e-4


@dataclass
class TrainedFold:
    fold_idx: int
    model_path: Path
    train_top1_acc: float
    model_weight_sha256: str
    epoch_losses: list[float]
    deterministic_warnings: list[str]
