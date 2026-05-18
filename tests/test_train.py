"""Tests for src/train.py backbone resolution precedence."""

from __future__ import annotations

import json
from pathlib import Path

from src.train import build_training_config


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _base_yaml() -> str:
    return """dataset: hdfs
window_seconds: null
seed: 42

backbone:
  selection_quantile: 0.8
  threshold_words: 512
  roberta_id: roberta-base
  longformer_id: allenai/longformer-base-4096

masking:
  sentence_ratio: 0.5
  token_ratio: 0.8

training:
  framework: huggingface_trainer
  transformers_version: 4.45.0
  epochs: 3
  max_lr: null
  lr_grid: [1.0e-5]
  effective_batch_size: 8
  optimizer: adamw
  betas: [0.9, 0.99]
  eps: 1.0e-5
  weight_decay: 0.01
  lr_scheduler:
    type: torch_one_cycle_lr
    pct_start: 0.3
    anneal_strategy: cos
    div_factor: 25.0
    final_div_factor: 1.0e4
    three_phase: false
  gradual_unfreezing_schedule:
    epoch_1_unfrozen_layers: [10, 11]
    epoch_2_unfrozen_layers: [8, 9, 10, 11]
    epoch_3_plus_unfrozen_layers: all
    embeddings_frozen_until_epoch: 2
  fp16: true

determinism:
  python_hash_seed: 0
  torch_deterministic_algorithms: true
  cublas_workspace_config: ":4096:8"
  hf_full_determinism: true
  fp16_weight_divergence_threshold: 1.0e-4
"""


def _base_yaml_with_training_backbone(backbone: str) -> str:
    base = _base_yaml().splitlines()
    out: list[str] = []
    inserted = False
    for line in base:
        out.append(line)
        if line.strip() == "training:":
            out.append(f"  backbone: {backbone}")
            inserted = True
    if not inserted:
        raise RuntimeError("Failed to insert training.backbone")
    return "\n".join(out) + "\n"


def test_backbone_default_is_roberta(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _base_yaml())

    cfg = build_training_config(config_path)
    assert cfg.backbone == "roberta-base"


def test_backbone_decision_overrides_default(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _base_yaml())

    decision_path = tmp_path / "backbone_decision.json"
    decision_path.write_text(
        json.dumps({"chosen_backbone": "allenai/longformer-base-4096"}),
        encoding="utf-8",
    )

    cfg = build_training_config(config_path, decision_path)
    assert cfg.backbone == "allenai/longformer-base-4096"


def test_training_backbone_override_has_priority(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(
        config_path, _base_yaml_with_training_backbone("roberta-base")
    )

    decision_path = tmp_path / "backbone_decision.json"
    decision_path.write_text(
        json.dumps({"chosen_backbone": "allenai/longformer-base-4096"}),
        encoding="utf-8",
    )

    cfg = build_training_config(config_path, decision_path)
    assert cfg.backbone == "roberta-base"


def test_missing_chosen_backbone_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _base_yaml())

    decision_path = tmp_path / "backbone_decision.json"
    decision_path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

    try:
        build_training_config(config_path, decision_path)
    except ValueError as exc:
        assert "chosen_backbone" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing chosen_backbone")
