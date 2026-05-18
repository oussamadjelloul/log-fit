"""Data classes shared across the pipeline.

These are pure data containers — no methodology decisions are embedded here.
Every methodology decision lives in src/* modules that produce or consume
these types. The dataclasses themselves are versioned with the spec.

Reference: logfit-repro-spec-v1.2.md Component 0 (notation) and Component 12 (API).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Backbone type — Literal restricts to the two paper-supported choices [L1]
BackboneName = Literal["roberta-base", "allenai/longformer-base-4096"]

# Dataset name [paper Section IV-A]
DatasetName = Literal["hdfs", "bgl", "tbird"]


@dataclass(frozen=True)
class Paragraph:
    """A log paragraph — one grouping unit (HDFS block_id session or
    BGL/TB time window).

    `lines` carries the raw log lines as published (no field stripping per D4
    default; conditional empirical guard handled at preparation time).
    """

    paragraph_id: str               # globally unique within (dataset, window)
    lines: list[str]                # raw log lines, full content [D4]
    label: int                      # 0 = normal, 1 = anomaly
    source_blockid: str | None = None       # HDFS only
    source_window_id: int | None = None      # BGL/TB only
    start_timestamp: float | None = None     # BGL/TB only, Unix epoch


@dataclass
class DropCounters:
    """Surfaced in preparation_summary.json. Non-zero values get explained
    or escalated per the disclosure standard."""

    # HDFS counters
    no_blockid: int = 0
    missing_label_assertion_fired: int = 0   # must remain 0; non-zero = hard fail
    duplicate_blockid: int = 0
    lines_with_multiple_blockids: int = 0    # [D16] flagged if rate > 5%

    # BGL/TB counters
    unparseable_timestamp: int = 0           # hard fail if rate > 0.1%
    empty_window: int = 0
    singleton_window: int = 0


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

    tokenizer: str                           # backbone name used for tokenization
    distribution: LengthDistribution
    truncation_rate_at_backbone_limit: float


@dataclass
class PreparationSummary:
    """Persisted as preparation_summary.json per dataset (x window)."""

    dataset: DatasetName
    window_seconds: int | None
    total_lines_read: int
    total_paragraphs: int
    normal_paragraphs: int
    anomaly_paragraphs: int
    anomaly_rate: float
    drop_counters: DropCounters
    word_length_distribution: LengthDistribution
    token_length_distribution: TokenLengthSummary | None = None    # populated by §1.6
    seed: int = 42


@dataclass
class AllocationManifest:
    """The 25k normal + 2k anomaly sample budget per dataset [L21]."""

    dataset: DatasetName
    window_seconds: int | None
    seed: int
    normal_paragraph_ids: list[str]          # length 25,000 [L21]
    anomaly_paragraph_ids: list[str]         # length 2,000 [L21]


@dataclass
class FoldSpec:
    """Per-fold allocation per D14 (v1.2). Normals k-fold-partitioned,
    anomalies non-partitioned and per-fold shuffled."""

    fold_idx: int                            # 0..4
    seed: int                                # 42 + fold_idx [D14]
    test_normal: list[str]                   # 5,000 [L25]
    train_normal: list[str]                  # 5,000 [L23]
    tune_normal: list[str]                   # 1,000 [L24]
    tune_anomaly: list[str]                  # 1,000 [L24]
    test_anomaly: list[str]                  # 1,000 [L25]


@dataclass
class TuningCell:
    """One cell of the (top_k, threshold) grid search [L15, L16]."""

    top_k: int                               # in {5, 9, 12}
    threshold: float                         # from linspace(top1-0.1, top1, 3)
    precision: float
    recall: float
    f1: float
    specificity: float


@dataclass
class TuningGrid:
    """Full 9-cell grid + selected (K, theta) per Component 6."""

    cells: list[TuningCell]
    best_top_k: int
    best_threshold: float
    train_top1_acc: float                    # used to build the threshold grid


@dataclass
class FoldMetrics:
    """Per-fold test-set metrics [L26]."""

    fold_idx: int
    precision: float
    recall: float
    f1: float
    specificity: float
    tp: int
    fp: int
    tn: int
    fn: int
    top_k: int                               # operating point used
    threshold: float                         # operating point used


@dataclass
class DatasetResults:
    """Aggregated across 5 folds per (dataset x window) per Component 7."""

    dataset: DatasetName
    window_seconds: int | None
    backbone: BackboneName
    p_mean: float
    p_std: float                             # sample std, ddof=1
    r_mean: float
    r_std: float
    f1_mean: float
    f1_std: float
    spec_mean: float
    spec_std: float
    per_fold_values: dict[str, list[float]]  # keys: "P", "R", "F1", "Spec"
    per_fold_operating_points: list[dict[str, float | int]]


@dataclass
class TrainingConfig:
    """Resolved training configuration after D2 lr-grid selection."""

    backbone: BackboneName
    epochs: int                              # [D1] 3 for HDFS, 5 for BGL/TB
    max_lr: float                            # [D2] from grid search
    effective_batch_size: int                # [D3]
    physical_batch_size: int                 # depends on backbone + GPU
    gradient_accumulation_steps: int

    # OneCycleLR config [T1, D15 v1.2]
    one_cycle_pct_start: float = 0.3
    one_cycle_anneal_strategy: str = "cos"
    one_cycle_div_factor: float = 25.0
    one_cycle_final_div_factor: float = 1e4

    # Gradual unfreezing schedule [D15]
    epoch_1_unfrozen_layers: tuple[int, ...] = (10, 11)
    epoch_2_unfrozen_layers: tuple[int, ...] = (8, 9, 10, 11)
    embeddings_frozen_until_epoch: int = 2   # unfrozen at epoch 3+ [T3]

    # Determinism [D8 strengthened v1.2]
    seed: int = 42
    full_determinism: bool = True
    fp16: bool = True
    fp16_weight_divergence_threshold: float = 1e-4   # [Q7]


@dataclass
class TrainedFold:
    """Artifacts produced by training one fold."""

    fold_idx: int
    model_path: Path
    train_top1_acc: float                    # used by Component 6 [D10 gate]
    model_weight_sha256: str                 # [T7] cross-run determinism audit
    epoch_losses: list[float]
    deterministic_warnings: list[str]         # non-deterministic kernels noted [D8]
