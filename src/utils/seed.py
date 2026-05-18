"""Global determinism setup per decisions-v1.2 D8 (strengthened post-T7 finding).

Call set_all_seeds(42) at the start of every script. The PEP 456 / cuBLAS env
vars (PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG) must be set BEFORE Python starts
— configured in scripts/*.sh, not here.

Reference: logfit-repro-decisions-v1.2.md D8; logfit-repro-spec-v1.2.md Component 14.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """Set every RNG we use to the given seed, and enable deterministic algorithms.

    NOTE: This is necessary but not sufficient — `TrainingArguments(full_determinism=True)`
    must also be set in Trainer config. See D8 v1.2.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # transformers also has its own set_seed; lazy-import so we don't pull
    # transformers in for non-training utilities
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass  # transformers not required for non-training utilities

    # Deterministic algorithm flags — defense in depth alongside
    # TrainingArguments(full_determinism=True).
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Verify env vars set by SLURM script (defense in depth)
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if "PYTHONHASHSEED" not in os.environ:
        # Note: setting this here is too late for hash() of strings in this process,
        # but ensures child processes inherit it. SLURM script should set it before
        # Python starts.
        os.environ["PYTHONHASHSEED"] = "0"


def check_determinism_env() -> dict[str, str]:
    """Return a snapshot of determinism-relevant env vars and torch flags.

    Persist this in train_log.json alongside model_weight_sha256 so cross-run
    audits can spot env drift.
    """
    return {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>"),
        "torch_deterministic_algorithms": str(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": str(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": str(torch.backends.cudnn.benchmark),
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device_count": str(torch.cuda.device_count()),
    }


def compute_state_dict_sha256(state_dict: dict) -> str:
    """SHA256 of model weights, layer-ordered for determinism.

    Used per Component 14 [T7] to verify cross-run weight stability under fp16.
    Two runs with the same seed on the same hardware should produce identical
    hashes; if they don't, log per-layer L-infinity divergence and escalate
    if any layer exceeds Q7 threshold (1e-4).
    """
    h = hashlib.sha256()
    # Sort keys for deterministic iteration order
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        # Convert to numpy for stable byte representation
        arr = tensor.detach().numpy() if hasattr(tensor, "detach") else tensor
        h.update(key.encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def compute_weight_divergence(
    state_dict_a: dict, state_dict_b: dict
) -> dict[str, float]:
    """Per-layer L-infinity divergence between two state dicts.

    Returns {layer_name: max_absolute_difference}. Used to diagnose fp16
    non-determinism per Q7 threshold (1e-4).
    """
    divergences: dict[str, float] = {}
    keys_a = set(state_dict_a.keys())
    keys_b = set(state_dict_b.keys())

    if keys_a != keys_b:
        raise ValueError(
            f"State dict keys differ: only in A: {keys_a - keys_b}, "
            f"only in B: {keys_b - keys_a}"
        )

    for key in sorted(keys_a):
        a = state_dict_a[key]
        b = state_dict_b[key]
        if hasattr(a, "cpu"):
            a = a.cpu()
        if hasattr(b, "cpu"):
            b = b.cpu()
        diff = (a.float() - b.float()).abs().max().item()
        divergences[key] = diff

    return divergences


def stable_paragraph_seed(paragraph_id: str) -> int:
    """DEPRECATED in v1.2 — kept only for reference.

    v1.0 spec used per-paragraph hash seeding for inference masking. v1.1
    review found this caused two bugs (PEP 456 non-determinism, permanent
    FN trap on unlucky paragraphs). v1.2 removed per-paragraph seeding
    entirely; inference now uses a single global RNG progressing across
    paragraphs in sorted paragraph_id order. See spec Component 5.

    DO NOT call this from production code. It exists only to document
    what the removed design looked like.
    """
    raise RuntimeError(
        "Per-paragraph seeding was removed in v1.2 (T5 / B2 finding). "
        "Use a single global numpy.random.RandomState across the eval loop "
        "and iterate paragraphs in sorted(key=paragraph_id) order. "
        "See spec-v1.2 Component 5."
    )
