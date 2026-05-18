"""Tests for configs/*.yaml — schema and value validation.

Addresses R2 F14 v1.3 — verifies the v1.3 spec contract for each
(dataset x window) config. Cheap regression protection against silent
YAML edits that would violate paper §IV-A.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Repo root is two parents up from this test file
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


def _load(name: str) -> dict:
    """Load a config YAML by basename."""
    path = CONFIGS_DIR / name
    assert path.exists(), f"Missing config file: {path}"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Names of all run configs the spec requires
ALL_CONFIGS = [
    "hdfs.yaml",
    "bgl_10s.yaml",
    "bgl_30s.yaml",
    "bgl_60s.yaml",
    "tbird_10s.yaml",
    "tbird_30s.yaml",
    "tbird_60s.yaml",
]


class TestAllConfigsLoadable:
    """Sanity: every required YAML exists and parses."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_loads(self, name: str):
        cfg = _load(name)
        assert isinstance(cfg, dict)
        assert "dataset" in cfg
        assert "seed" in cfg


class TestThunderbirdCap:
    """[R2 F14 / L19] Verify Thunderbird configs set max_input_lines=20M."""

    @pytest.mark.parametrize(
        "name", ["tbird_10s.yaml", "tbird_30s.yaml", "tbird_60s.yaml"]
    )
    def test_tbird_max_input_lines_is_20m(self, name: str):
        cfg = _load(name)
        assert "max_input_lines" in cfg, (
            f"{name} must declare max_input_lines per L19"
        )
        assert cfg["max_input_lines"] == 20_000_000, (
            f"{name} max_input_lines must be 20,000,000 per L19, "
            f"got {cfg['max_input_lines']}"
        )

    @pytest.mark.parametrize(
        "name", ["bgl_10s.yaml", "bgl_30s.yaml", "bgl_60s.yaml", "hdfs.yaml"]
    )
    def test_non_tbird_configs_no_cap_required(self, name: str):
        """BGL and HDFS configs don't need max_input_lines (read full file)."""
        cfg = _load(name)
        # Field is optional for non-Thunderbird; if present, must not constrain
        if "max_input_lines" in cfg:
            assert cfg["max_input_lines"] is None or cfg["max_input_lines"] > 0


class TestConsistentSchema:
    """Every config must declare the same set of top-level sections."""

    REQUIRED_TOP_LEVEL = {
        "dataset",
        "seed",
        "sample_budget",
        "cv",
        "backbone",
        "token_length_gate",
        "masking",
        "training",
        "determinism",
        "inference",
        "variability",
    }

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_has_required_top_level_keys(self, name: str):
        cfg = _load(name)
        missing = self.REQUIRED_TOP_LEVEL - set(cfg.keys())
        assert not missing, f"{name} missing required keys: {missing}"


class TestCriticalValuesLocked:
    """Spot-check that critical L# and D# values are present and correct."""

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_sample_budget_is_25k_2k(self, name: str):
        cfg = _load(name)
        assert cfg["sample_budget"]["normal"] == 25_000      # [L21]
        assert cfg["sample_budget"]["anomaly"] == 2_000       # [L21]

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_cv_folds_is_5(self, name: str):
        cfg = _load(name)
        assert cfg["cv"]["folds"] == 5                        # [L22]

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_anomaly_partition_strategy(self, name: str):
        cfg = _load(name)
        assert cfg["cv"]["anomaly_partition_strategy"] == "per_fold_shuffle"  # [D14]

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_masking_ratios(self, name: str):
        cfg = _load(name)
        assert cfg["masking"]["sentence_ratio"] == 0.5        # [L5]
        assert cfg["masking"]["token_ratio"] == 0.8           # [L6]

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_lr_scheduler_is_one_cycle(self, name: str):
        cfg = _load(name)
        assert cfg["training"]["lr_scheduler"]["type"] == "torch_one_cycle_lr"  # [D15 v1.2 / T1]

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_transformers_version_pinned(self, name: str):
        cfg = _load(name)
        assert cfg["training"]["transformers_version"] == "4.45.0"  # [T5]

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_full_determinism_enabled(self, name: str):
        cfg = _load(name)
        assert cfg["determinism"]["hf_full_determinism"] is True   # [T7]

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_inference_topk_grid(self, name: str):
        cfg = _load(name)
        assert cfg["inference"]["topk_grid"] == [5, 9, 12]    # [L13]


class TestEpochsPerDataset:
    """[D1] HDFS uses 3 epochs, BGL/TB use 5."""

    def test_hdfs_epochs_3(self):
        assert _load("hdfs.yaml")["training"]["epochs"] == 3

    @pytest.mark.parametrize(
        "name",
        ["bgl_10s.yaml", "bgl_30s.yaml", "bgl_60s.yaml",
         "tbird_10s.yaml", "tbird_30s.yaml", "tbird_60s.yaml"],
    )
    def test_bgl_tbird_epochs_5(self, name: str):
        assert _load(name)["training"]["epochs"] == 5


class TestVariabilityEnabledForBgl:
    """Variability test runs only on BGL (paper Table VII)."""

    def test_bgl_30s_variability_enabled(self):
        cfg = _load("bgl_30s.yaml")
        assert cfg["variability"]["enabled"] is True

    def test_hdfs_variability_disabled(self):
        cfg = _load("hdfs.yaml")
        assert cfg["variability"]["enabled"] is False

    @pytest.mark.parametrize(
        "name", ["tbird_10s.yaml", "tbird_30s.yaml", "tbird_60s.yaml"]
    )
    def test_tbird_variability_disabled(self, name: str):
        cfg = _load(name)
        assert cfg["variability"]["enabled"] is False

    def test_bgl_variability_uses_v13_lemma_strategy(self):
        """[T6 / D17 v1.2] multi-word filter for WordNet lemmas."""
        cfg = _load("bgl_30s.yaml")
        assert (
            cfg["variability"]["lemma_strategy"]
            == "first_synset_first_single_word_lemma"
        )


class TestDatasetWindowMatch:
    """The dataset/window fields must match the filename convention."""

    def test_hdfs_no_window(self):
        cfg = _load("hdfs.yaml")
        assert cfg["dataset"] == "hdfs"
        assert cfg["window_seconds"] is None

    @pytest.mark.parametrize(
        "name,dataset,window",
        [
            ("bgl_10s.yaml", "bgl", 10),
            ("bgl_30s.yaml", "bgl", 30),
            ("bgl_60s.yaml", "bgl", 60),
            ("tbird_10s.yaml", "tbird", 10),
            ("tbird_30s.yaml", "tbird", 30),
            ("tbird_60s.yaml", "tbird", 60),
        ],
    )
    def test_dataset_and_window_match_filename(
        self, name: str, dataset: str, window: int
    ):
        cfg = _load(name)
        assert cfg["dataset"] == dataset
        assert cfg["window_seconds"] == window
