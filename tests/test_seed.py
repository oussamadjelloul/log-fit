"""Tests for src/utils/seed.py — global determinism setup.

Pattern reference for the rest of the test suite. Each test should map back
to a specific decision (D#) or finding (T#) in docs/logfit-repro-decisions-v1.2.md.
"""

from __future__ import annotations

import random
import numpy as np
import pytest

from src.utils.seed import (
    check_determinism_env,
    compute_state_dict_sha256,
    compute_weight_divergence,
    set_all_seeds,
    stable_paragraph_seed,
)


class TestSetAllSeeds:
    """[D8] Global seed must reach Python random, NumPy, and PyTorch."""

    def test_python_random_reproducible(self):
        set_all_seeds(42)
        a = random.random()
        set_all_seeds(42)
        b = random.random()
        assert a == b, "Python random.random() must be deterministic under set_all_seeds"

    def test_numpy_reproducible(self):
        set_all_seeds(42)
        a = np.random.randn(10)
        set_all_seeds(42)
        b = np.random.randn(10)
        assert np.array_equal(a, b), "numpy.random must be deterministic under set_all_seeds"

    def test_different_seeds_diverge(self):
        set_all_seeds(42)
        a = np.random.randn(10)
        set_all_seeds(43)
        b = np.random.randn(10)
        assert not np.array_equal(a, b), "different seeds must produce different streams"


class TestCheckDeterminismEnv:
    """[D8] Snapshot of env state for persistence in train_log.json."""

    def test_returns_required_keys(self):
        set_all_seeds(42)
        env = check_determinism_env()
        required = {
            "PYTHONHASHSEED",
            "CUBLAS_WORKSPACE_CONFIG",
            "torch_deterministic_algorithms",
            "cudnn_deterministic",
            "torch_version",
        }
        assert required.issubset(env.keys()), f"Missing keys: {required - env.keys()}"


class TestStateDictSha256:
    """[T7] Cross-run model weight verification under fp16 non-determinism."""

    def test_same_dict_same_hash(self):
        import torch
        sd = {"layer.weight": torch.randn(10, 10), "layer.bias": torch.randn(10)}
        h1 = compute_state_dict_sha256(sd)
        h2 = compute_state_dict_sha256(sd)
        assert h1 == h2, "Same state dict must hash identically"

    def test_different_dicts_different_hash(self):
        import torch
        sd_a = {"layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        sd_b = {"layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.1]])}
        assert compute_state_dict_sha256(sd_a) != compute_state_dict_sha256(sd_b)

    def test_key_order_independent(self):
        """Hash must be deterministic regardless of dict insertion order."""
        import torch
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        sd_a = {"a": t1, "b": t2}
        sd_b = {"b": t2, "a": t1}
        assert compute_state_dict_sha256(sd_a) == compute_state_dict_sha256(sd_b)


class TestWeightDivergence:
    """[Q7] Per-layer L-infinity divergence for fp16 audit (1e-4 threshold)."""

    def test_identical_dicts_zero_divergence(self):
        import torch
        sd = {"layer.weight": torch.randn(5, 5)}
        div = compute_weight_divergence(sd, sd)
        assert div["layer.weight"] == pytest.approx(0.0)

    def test_known_divergence(self):
        import torch
        sd_a = {"w": torch.tensor([1.0, 2.0, 3.0])}
        sd_b = {"w": torch.tensor([1.0, 2.5, 3.0])}
        div = compute_weight_divergence(sd_a, sd_b)
        assert div["w"] == pytest.approx(0.5)

    def test_mismatched_keys_raises(self):
        import torch
        sd_a = {"a": torch.tensor([1.0])}
        sd_b = {"b": torch.tensor([1.0])}
        with pytest.raises(ValueError, match="State dict keys differ"):
            compute_weight_divergence(sd_a, sd_b)


class TestStableParagraphSeedDeprecated:
    """[T5 / B2] Per-paragraph seeding was removed in v1.2.

    This test guards against accidental reintroduction of the v1.0 design.
    """

    def test_raises_on_call(self):
        with pytest.raises(RuntimeError, match="removed in v1.2"):
            stable_paragraph_seed("any_id")
