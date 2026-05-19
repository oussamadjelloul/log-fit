"""Tests for src/train.py — training orchestration per spec Component 4.

Strategy:
- Pure-logic helpers (`_resolve_batch_sizes`, `_compute_steps_per_epoch`,
  `_extract_epoch_losses`, `_extract_last_top1_acc`) tested with synthetic
  inputs. No HF model load needed.
- `GradualUnfreezingCallback` tested with a minimal nn.Module mock mimicking
  RoBERTa's `.encoder.layer` + `.embeddings` interface — no model download.
- `build_training_config` tested with synthetic YAML fixtures.
- End-to-end `train_fold` is NOT unit-tested here; it runs as an integration
  test on Narval via `scripts/train.sh` (model download + GPU required).

Mapping to spec-v1.2 Component 13:
- test_unfreezing_schedule_transitions (spec-named, covers T3 fix)
- test_resolve_batch_sizes_* (divisibility contract)
- test_compute_steps_per_epoch_* (B4 regression coverage)
- test_extract_epoch_losses_*
- test_build_training_config_* (config schema round-trip)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn

from src.dataset import MaskedSentenceCollator, MaskedSentenceDataset
from src.train import (
    GradualUnfreezingCallback,
    TopKAccuracyLogger,
    TrainingRunConfig,
    _compute_steps_per_epoch,
    _extract_epoch_losses,
    _extract_last_top1_acc,
    _resolve_batch_sizes,
    _resolve_physical_batch_size,
    build_training_config,
)
from src.types import Paragraph
from tests.test_mask import FakeTokenizer


# ---------------------------------------------------------------------------
# Fake RoBERTa / Longformer-shaped model for callback tests
#
# The callback duck-types on `.roberta.encoder.layer` / `.roberta.embeddings`
# (or `.longformer.*`). Real nn.Linear / nn.Embedding modules are used so
# that `.parameters()` returns real nn.Parameter instances whose
# `requires_grad` flag the callback actually toggles.
# ---------------------------------------------------------------------------


class _LayerStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(8, 8)


class _EncoderStub(nn.Module):
    def __init__(self, n_layers: int = 12):
        super().__init__()
        self.layer = nn.ModuleList([_LayerStub() for _ in range(n_layers)])


class _EmbeddingsStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(100, 8)
        self.position_embeddings = nn.Embedding(128, 8)


class _BackboneStub(nn.Module):
    def __init__(self, n_layers: int = 12):
        super().__init__()
        self.encoder = _EncoderStub(n_layers)
        self.embeddings = _EmbeddingsStub()


class FakeRobertaModel(nn.Module):
    def __init__(self, n_layers: int = 12):
        super().__init__()
        self.roberta = _BackboneStub(n_layers)


class FakeLongformerModel(nn.Module):
    def __init__(self, n_layers: int = 12):
        super().__init__()
        self.longformer = _BackboneStub(n_layers)


def _layer_unfrozen_flags(layers: nn.ModuleList) -> list[bool]:
    """For each layer, True iff ALL its parameters require grad."""
    return [
        all(p.requires_grad for p in layer.parameters())
        for layer in layers
    ]


def _embeddings_unfrozen(embeddings: nn.Module) -> bool:
    return all(p.requires_grad for p in embeddings.parameters())


def _make_callback(
    model: nn.Module,
    epoch_3_plus="all",
) -> GradualUnfreezingCallback:
    return GradualUnfreezingCallback(
        model=model,
        epoch_1_unfrozen_layers=(10, 11),
        epoch_2_unfrozen_layers=(8, 9, 10, 11),
        epoch_3_plus_unfrozen_layers=epoch_3_plus,
        embeddings_frozen_until_epoch=2,
    )


# ---------------------------------------------------------------------------
# GradualUnfreezingCallback
# ---------------------------------------------------------------------------


class TestCallbackConstruction:
    def test_accepts_roberta_model(self):
        cb = _make_callback(FakeRobertaModel())
        assert cb._epoch_count == 0
        assert len(cb.layers) == 12

    def test_accepts_longformer_model(self):
        cb = _make_callback(FakeLongformerModel())
        assert len(cb.layers) == 12

    def test_rejects_unsupported_backbone(self):
        class FakeUnknown(nn.Module):
            pass
        with pytest.raises(ValueError, match="Unsupported backbone"):
            _make_callback(FakeUnknown())

    def test_rejects_non_12_layers(self):
        with pytest.raises(ValueError, match="Expected 12 encoder layers"):
            _make_callback(FakeRobertaModel(n_layers=10))


class TestUnfreezingScheduleTransitions:
    """Spec-mandated test (Component 13). Verifies T3, T5 fixes."""

    def test_epoch_1_unfreezes_only_layers_10_11(self):
        cb = _make_callback(FakeRobertaModel())
        cb.on_epoch_begin(None, None, None)  # epoch 1
        assert _layer_unfrozen_flags(cb.layers) == [False] * 10 + [True, True]
        assert _embeddings_unfrozen(cb.embeddings) is False

    def test_epoch_2_unfreezes_layers_8_through_11(self):
        cb = _make_callback(FakeRobertaModel())
        cb.on_epoch_begin(None, None, None)  # 1
        cb.on_epoch_begin(None, None, None)  # 2
        assert (
            _layer_unfrozen_flags(cb.layers)
            == [False] * 8 + [True, True, True, True]
        )
        assert _embeddings_unfrozen(cb.embeddings) is False

    def test_epoch_3_unfreezes_all_layers_and_embeddings(self):
        """T3 fix: embeddings join the unfrozen set at epoch 3, not stay frozen."""
        cb = _make_callback(FakeRobertaModel())
        for _ in range(3):
            cb.on_epoch_begin(None, None, None)
        assert _layer_unfrozen_flags(cb.layers) == [True] * 12
        assert _embeddings_unfrozen(cb.embeddings) is True

    def test_epoch_4_and_beyond_remains_all_unfrozen(self):
        cb = _make_callback(FakeRobertaModel())
        for _ in range(5):
            cb.on_epoch_begin(None, None, None)
        assert _layer_unfrozen_flags(cb.layers) == [True] * 12
        assert _embeddings_unfrozen(cb.embeddings) is True

    def test_internal_counter_increments_independent_of_state_object(self):
        """T5 fix: counter advances on each on_epoch_begin call, regardless
        of what the (now ignored) TrainerState says."""
        cb = _make_callback(FakeRobertaModel())
        for expected in (1, 2, 3, 4, 5):
            cb.on_epoch_begin(None, None, None)
            assert cb._epoch_count == expected

    def test_epoch_3_plus_explicit_layer_list(self):
        """If epoch_3_plus_unfrozen_layers is a tuple, only those layers +
        embeddings unfreeze. ('all' is a special-case shorthand.)"""
        cb = _make_callback(
            FakeRobertaModel(),
            epoch_3_plus=(6, 7, 8, 9, 10, 11),
        )
        for _ in range(3):
            cb.on_epoch_begin(None, None, None)
        assert _layer_unfrozen_flags(cb.layers) == [False] * 6 + [True] * 6
        assert _embeddings_unfrozen(cb.embeddings) is True

    def test_longformer_branch_works(self):
        """Both `model.roberta` and `model.longformer` paths must apply
        identical schedule."""
        cb = _make_callback(FakeLongformerModel())
        for _ in range(3):
            cb.on_epoch_begin(None, None, None)
        assert _layer_unfrozen_flags(cb.layers) == [True] * 12
        assert _embeddings_unfrozen(cb.embeddings) is True


# ---------------------------------------------------------------------------
# TopKAccuracyLogger — flag behavior (constructor surface only)
#
# The end-to-end behavior (running eval on real logits) requires a real model
# + Trainer and is covered by the Narval integration run, not here. These
# tests just lock down the public API of the reset_rng_each_epoch flag so
# accidental regressions in the default or attribute name are caught.
# ---------------------------------------------------------------------------


def _make_top1_logger(**kwargs) -> TopKAccuracyLogger:
    tokenizer = FakeTokenizer()
    paragraphs = [
        Paragraph(paragraph_id=f"p_{i}", lines=["a b c d"], label=0)
        for i in range(3)
    ]
    eval_ds = MaskedSentenceDataset(
        paragraphs=paragraphs,
        tokenizer=tokenizer,
        backbone_token_limit=512,
        seed=42,
    )
    collator = MaskedSentenceCollator(tokenizer)
    return TopKAccuracyLogger(
        eval_dataset=eval_ds,
        collator=collator,
        batch_size=2,
        **kwargs,
    )


class TestTopKAccuracyLoggerFlag:
    def test_reset_default_is_true(self):
        """Default behavior: reset eval RNG each epoch so the learning
        curve is directly comparable across epochs."""
        logger = _make_top1_logger()
        assert logger._reset_rng_each_epoch is True

    def test_reset_can_be_disabled(self):
        """Pass reset_rng_each_epoch=False to get fresh masks per epoch
        (mirrors training-time stochasticity)."""
        logger = _make_top1_logger(reset_rng_each_epoch=False)
        assert logger._reset_rng_each_epoch is False

    def test_eval_dataset_reset_rng_restores_initial_state(self):
        """Sanity: MaskedSentenceDataset.reset_rng() actually restores the
        construction-time RNG state — the property the flag relies on."""
        logger = _make_top1_logger()
        ds = logger._eval_dataset

        # Advance the RNG by drawing a few masks
        before = [ds[i] for i in range(len(ds))]
        # Reset and re-walk
        ds.reset_rng()
        after = [ds[i] for i in range(len(ds))]
        assert before == after


# ---------------------------------------------------------------------------
# _resolve_physical_batch_size
# ---------------------------------------------------------------------------


class TestResolvePhysicalBatchSize:
    def test_roberta_gets_8(self):
        assert _resolve_physical_batch_size("roberta-base") == 8

    def test_longformer_gets_2(self):
        assert (
            _resolve_physical_batch_size("allenai/longformer-base-4096") == 2
        )

    def test_case_insensitive(self):
        assert _resolve_physical_batch_size("RoBERTa-Base") == 8
        assert _resolve_physical_batch_size("LONGFORMER-X") == 2

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            _resolve_physical_batch_size("gpt-2")


# ---------------------------------------------------------------------------
# _resolve_batch_sizes
# ---------------------------------------------------------------------------


class TestResolveBatchSizes:
    def test_exact_match_gives_accum_1(self):
        physical, accum = _resolve_batch_sizes(
            effective_batch_size=8, physical_batch_size=8
        )
        assert physical == 8
        assert accum == 1

    def test_longformer_4x_accumulation(self):
        """Longformer effective=8, physical=2 → accum=4. Matches D3."""
        physical, accum = _resolve_batch_sizes(
            effective_batch_size=8, physical_batch_size=2
        )
        assert physical == 2
        assert accum == 4

    def test_indivisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            _resolve_batch_sizes(
                effective_batch_size=10, physical_batch_size=3
            )

    def test_zero_physical_raises(self):
        with pytest.raises(ValueError, match="physical_batch_size"):
            _resolve_batch_sizes(
                effective_batch_size=8, physical_batch_size=0
            )

    def test_negative_physical_raises(self):
        with pytest.raises(ValueError, match="physical_batch_size"):
            _resolve_batch_sizes(
                effective_batch_size=8, physical_batch_size=-2
            )

    def test_zero_effective_raises(self):
        with pytest.raises(ValueError, match="effective_batch_size"):
            _resolve_batch_sizes(
                effective_batch_size=0, physical_batch_size=8
            )


# ---------------------------------------------------------------------------
# _compute_steps_per_epoch — B4 regression coverage
# ---------------------------------------------------------------------------


class TestComputeStepsPerEpoch:
    """Verifies the helper matches HF Trainer 4.45's actual per-epoch update
    count: ``ceil(N / (bs * accum))``. HF's `do_sync_step` flushes the partial
    accumulation group at end-of-epoch, so the count is ceil, not floor.
    See _compute_steps_per_epoch docstring + HF issue #36297."""

    def test_accum_1_matches_dataloader_length(self):
        # ceil(100/8) = 13
        assert _compute_steps_per_epoch(100, 8, 1) == 13
        assert _compute_steps_per_epoch(1000, 8, 1) == 125

    def test_partial_accumulation_at_end_of_epoch_counted(self):
        """Core HF behavior: when len_dataloader doesn't divide evenly by
        accum, the remainder triggers an end-of-epoch sync. We count it.

        N=100, bs=8, accum=2:
          len_dataloader = ceil(100/8) = 13
          full groups    = 13 // 2 = 6
          end-of-epoch flush adds 1 → total = 7
          equivalently:  ceil(100 / 16) = 7
        """
        assert _compute_steps_per_epoch(100, 8, 2) == 7

    def test_longformer_realistic_case(self):
        """N=10000, bs=8, accum=4. HF's end-of-epoch flush takes us from
        floor(1250/4)=312 to ceil(1250/4)=313."""
        assert _compute_steps_per_epoch(10000, 8, 4) == 313

    def test_perfect_division_unchanged(self):
        """When len_dataloader is divisible by accum, no partial group,
        so floor == ceil. The two formulas agree."""
        # N=200, bs=8, accum=5: ceil(200/8)=25, 25/5=5 exactly
        assert _compute_steps_per_epoch(200, 8, 5) == 5
        # N=80, bs=8, accum=2: ceil(80/8)=10, 10/2=5 exactly
        assert _compute_steps_per_epoch(80, 8, 2) == 5

    def test_zero_samples_returns_1(self):
        """max(1, ...) floor ensures at least 1 step per epoch even on
        degenerate empty datasets."""
        assert _compute_steps_per_epoch(0, 8, 1) == 1

    def test_n_smaller_than_bs_returns_1(self):
        # ceil(3 / (8*1)) = 1
        assert _compute_steps_per_epoch(3, 8, 1) == 1
        assert _compute_steps_per_epoch(7, 8, 1) == 1

    def test_n_equals_bs_returns_1(self):
        assert _compute_steps_per_epoch(8, 8, 1) == 1

    def test_accum_larger_than_dataloader_returns_1(self):
        """Even when accum is huge, ceil floors to 1 (single end-of-epoch sync)."""
        # N=10, bs=8, accum=100: ceil(10/800) = 1
        assert _compute_steps_per_epoch(10, 8, 100) == 1

    def test_equivalence_of_two_ceil_forms(self):
        """Sanity: ceil(N/(bs*accum)) == ceil(ceil(N/bs)/accum) for our inputs.
        Documenting the identity used in the docstring."""
        import math
        for n in (1, 7, 8, 100, 10000):
            for bs in (1, 8, 32):
                for accum in (1, 2, 4, 8):
                    expected = math.ceil(math.ceil(n / bs) / accum)
                    actual = math.ceil(n / (bs * accum))
                    assert expected == actual, (
                        f"Mismatch at N={n}, bs={bs}, accum={accum}"
                    )


# ---------------------------------------------------------------------------
# _extract_epoch_losses
# ---------------------------------------------------------------------------


class TestExtractEpochLosses:
    def test_empty_history_returns_empty(self):
        assert _extract_epoch_losses([]) == []

    def test_history_without_loss_returns_empty(self):
        history = [{"epoch": 1.0, "train_top1_acc": 0.8}]
        assert _extract_epoch_losses(history) == []

    def test_single_entry_per_epoch(self):
        history = [
            {"epoch": 1.0, "loss": 2.3},
            {"epoch": 2.0, "loss": 1.5},
            {"epoch": 3.0, "loss": 1.1},
        ]
        assert _extract_epoch_losses(history) == [2.3, 1.5, 1.1]

    def test_same_epoch_multiple_entries_keeps_last(self):
        """HF logs every `logging_steps` batches. Per integer epoch the
        LAST entry's loss is retained — gives end-of-epoch loss."""
        history = [
            {"epoch": 1.0, "loss": 2.5},
            {"epoch": 1.0, "loss": 2.3},   # later entry overrides
            {"epoch": 2.0, "loss": 1.5},
        ]
        assert _extract_epoch_losses(history) == [2.3, 1.5]

    def test_ignores_entries_missing_epoch_field(self):
        history = [
            {"loss": 2.3},                # no epoch
            {"epoch": 1.0, "loss": 1.5},
        ]
        assert _extract_epoch_losses(history) == [1.5]

    def test_ignores_entries_missing_loss_field(self):
        history = [
            {"epoch": 1.0, "train_top1_acc": 0.7},  # no loss
            {"epoch": 1.0, "loss": 2.0},
        ]
        assert _extract_epoch_losses(history) == [2.0]

    def test_returns_losses_sorted_by_epoch(self):
        """Even if HF logs out of order, output is ordered by epoch_idx."""
        history = [
            {"epoch": 3.0, "loss": 1.1},
            {"epoch": 1.0, "loss": 2.3},
            {"epoch": 2.0, "loss": 1.5},
        ]
        assert _extract_epoch_losses(history) == [2.3, 1.5, 1.1]


# ---------------------------------------------------------------------------
# _extract_last_top1_acc
# ---------------------------------------------------------------------------


class TestExtractLastTop1Acc:
    def test_empty_history_returns_zero(self):
        assert _extract_last_top1_acc([]) == 0.0

    def test_returns_last_top1_acc_in_chronological_order(self):
        history = [
            {"epoch": 1, "train_top1_acc": 0.60},
            {"epoch": 2, "train_top1_acc": 0.80},
            {"epoch": 3, "train_top1_acc": 0.85},
        ]
        assert _extract_last_top1_acc(history) == 0.85

    def test_no_top1_acc_returns_zero(self):
        history = [{"epoch": 1, "loss": 2.3}]
        assert _extract_last_top1_acc(history) == 0.0

    def test_skips_entries_without_top1_acc(self):
        history = [
            {"epoch": 1, "train_top1_acc": 0.6},
            {"epoch": 1.5, "loss": 1.5},          # no top1_acc — skipped
        ]
        assert _extract_last_top1_acc(history) == 0.6


# ---------------------------------------------------------------------------
# build_training_config — YAML schema round-trip
# ---------------------------------------------------------------------------


_MINIMAL_CONFIG_YAML = """
dataset: hdfs
window_seconds: null
seed: 42

backbone:
  roberta_id: roberta-base
  longformer_id: allenai/longformer-base-4096

masking:
  sentence_ratio: 0.5
  token_ratio: 0.8

training:
  epochs: 3
  max_lr: null
  lr_grid: [1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4]
  effective_batch_size: 8
  betas: [0.9, 0.99]
  eps: 1.0e-5
  weight_decay: 0.01
  lr_scheduler:
    pct_start: 0.3
    anneal_strategy: cos
    div_factor: 25.0
    final_div_factor: 1.0e+4
    three_phase: false
  gradual_unfreezing_schedule:
    epoch_1_unfrozen_layers: [10, 11]
    epoch_2_unfrozen_layers: [8, 9, 10, 11]
    epoch_3_plus_unfrozen_layers: all
    embeddings_frozen_until_epoch: 2
  fp16: true

determinism:
  hf_full_determinism: true
  fp16_weight_divergence_threshold: 1.0e-4
"""


@pytest.fixture
def minimal_config_path(tmp_path: Path) -> Path:
    p = tmp_path / "minimal.yaml"
    p.write_text(_MINIMAL_CONFIG_YAML)
    return p


class TestBuildTrainingConfig:
    def test_returns_training_run_config(self, minimal_config_path):
        cfg = build_training_config(minimal_config_path)
        assert isinstance(cfg, TrainingRunConfig)

    def test_top_level_fields_parsed(self, minimal_config_path):
        cfg = build_training_config(minimal_config_path)
        assert cfg.dataset == "hdfs"
        assert cfg.window_seconds is None
        assert cfg.seed == 42

    def test_training_block_parsed(self, minimal_config_path):
        cfg = build_training_config(minimal_config_path)
        assert cfg.epochs == 3
        assert cfg.max_lr is None
        assert cfg.lr_grid == [1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4]
        assert cfg.effective_batch_size == 8

    def test_resolves_roberta_physical_batch_size(self, minimal_config_path):
        cfg = build_training_config(minimal_config_path)
        assert cfg.backbone == "roberta-base"
        assert cfg.physical_batch_size == 8
        assert cfg.gradient_accumulation_steps == 1

    def test_one_cycle_params_parsed(self, minimal_config_path):
        cfg = build_training_config(minimal_config_path)
        assert cfg.one_cycle_pct_start == 0.3
        assert cfg.one_cycle_anneal_strategy == "cos"
        assert cfg.one_cycle_div_factor == 25.0
        assert cfg.one_cycle_final_div_factor == 1.0e4
        assert cfg.one_cycle_three_phase is False

    def test_unfreezing_schedule_parsed(self, minimal_config_path):
        cfg = build_training_config(minimal_config_path)
        assert cfg.epoch_1_unfrozen_layers == (10, 11)
        assert cfg.epoch_2_unfrozen_layers == (8, 9, 10, 11)
        assert cfg.epoch_3_plus_unfrozen_layers == "all"
        assert cfg.embeddings_frozen_until_epoch == 2

    def test_masking_params_parsed(self, minimal_config_path):
        cfg = build_training_config(minimal_config_path)
        assert cfg.masking_sentence_ratio == 0.5
        assert cfg.masking_token_ratio == 0.8

    def test_determinism_params_parsed(self, minimal_config_path):
        cfg = build_training_config(minimal_config_path)
        assert cfg.full_determinism is True
        assert cfg.fp16 is True
        assert cfg.fp16_weight_divergence_threshold == 1e-4

    def test_betas_parsed_as_tuple(self, minimal_config_path):
        cfg = build_training_config(minimal_config_path)
        assert cfg.betas == (0.9, 0.99)
        assert isinstance(cfg.betas, tuple)

    def test_backbone_override_via_training_block(self, tmp_path):
        """training.backbone overrides the default RoBERTa selection."""
        yaml_text = _MINIMAL_CONFIG_YAML.replace(
            "  fp16: true",
            "  backbone: allenai/longformer-base-4096\n  fp16: true",
        )
        p = tmp_path / "override.yaml"
        p.write_text(yaml_text)
        cfg = build_training_config(p)
        assert cfg.backbone == "allenai/longformer-base-4096"
        assert cfg.physical_batch_size == 2
        assert cfg.gradient_accumulation_steps == 4

    def test_backbone_decision_artifact_override(self, tmp_path):
        """A backbone-decision JSON (from src.select_backbone) overrides
        the YAML default when its path is provided."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        decision_path = tmp_path / "backbone_choice.json"
        decision_path.write_text(
            '{"chosen_backbone": "allenai/longformer-base-4096"}'
        )

        cfg = build_training_config(
            cfg_path, backbone_decision_path=decision_path
        )
        assert cfg.backbone == "allenai/longformer-base-4096"

    def test_backbone_decision_missing_field_raises(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        decision_path = tmp_path / "bad_choice.json"
        decision_path.write_text('{"other_field": "x"}')

        with pytest.raises(ValueError, match="chosen_backbone"):
            build_training_config(
                cfg_path, backbone_decision_path=decision_path
            )

    def test_nonexistent_decision_path_silently_skipped(self, tmp_path):
        """If a decision path is supplied but the file doesn't exist,
        fall through to YAML-based selection. Permits 'try the decision
        artifact, else default' patterns in scripts/train.sh."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        cfg = build_training_config(
            cfg_path,
            backbone_decision_path=tmp_path / "does_not_exist.json",
        )
        assert cfg.backbone == "roberta-base"
