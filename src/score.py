"""Paragraph-level top-k scoring for LogFiT models (spec Component 5 + v1.5 D_NEW6-9).

Implements:
- D_NEW6 contract: reads canonical splits.json schema (fold_id, *_ids).
- D_NEW7: per-paragraph topk accuracies persisted as a list of records
  ({top_k, accuracy}, ...), NOT dict[int, float]. JSON int-key coercion
  used to silently break downstream consumers.
- D_NEW8: --n-passes flag (default 1 = paper-faithful). R passes draw R
  independent masks per paragraph; per-paragraph accuracy is the POOLED
  ratio (sum_correct / sum_total across passes), not mean-of-fractions.
- D_NEW9: inference batch size defaults to physical_batch_size (8 for
  RoBERTa, 2 for Longformer) — the safe-by-construction value. --batch-size
  overrides for empirical optimization.

Reference: docs/Logfit-repro-spec-v1.2.md Component 5;
docs/logfit-repro-decisions-v1.5.md D_NEW6/D_NEW7/D_NEW8/D_NEW9.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.dataset import MaskedSentenceCollator, MaskedSentenceDataset
from src.select_backbone import backbone_token_limit
from src.train import _resolve_physical_batch_size, load_paragraphs
from src.types import Paragraph, ParagraphScore, SplitScores, TopKAccuracyRecord
from src.utils.io import ensure_dir, load_json, load_yaml, save_json


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_topk_grid(value: str) -> list[int]:
    """Parse a comma-separated grid (e.g. '5,9,12') into a list of ints."""
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


# ---------------------------------------------------------------------------
# Backbone resolution
# ---------------------------------------------------------------------------


def _resolve_backbone(
    cfg: dict[str, Any],
    backbone_decision_path: Path | None,
) -> str:
    """Determine which backbone to load.

    Precedence (highest wins):
    1. training.backbone (explicit YAML override)
    2. backbone_decision_path file (artifact from src.select_backbone)
    3. training.use_longformer flag (YAML)
    4. backbone.roberta_id (default)

    v1.5 I3 fix: if backbone_decision_path is provided but the file does
    not exist, raise FileNotFoundError rather than silently fall through.
    """
    backbone_cfg = cfg["backbone"]
    training = cfg["training"]

    backbone_name = backbone_cfg["roberta_id"]
    if training.get("use_longformer", False):
        backbone_name = backbone_cfg["longformer_id"]

    if backbone_decision_path is not None:
        decision_path = Path(backbone_decision_path)
        if not decision_path.exists():
            raise FileNotFoundError(
                f"--backbone-decision was provided but file does not exist: "
                f"{decision_path}. Either remove the flag or fix the path."
            )
        decision_data = load_json(decision_path)
        chosen = decision_data.get("chosen_backbone")
        if chosen is None:
            raise ValueError(
                f"Backbone decision artifact is missing 'chosen_backbone': "
                f"{decision_path}"
            )
        backbone_name = chosen

    if training.get("backbone"):
        backbone_name = training["backbone"]

    return backbone_name


# ---------------------------------------------------------------------------
# Splits + paragraph lookups (v1.5 canonical schema only)
# ---------------------------------------------------------------------------


def _load_fold_data(splits_path: Path, fold_idx: int) -> dict[str, Any]:
    """Find the fold dict with ``fold_id == fold_idx`` in splits.json.

    v1.5 M1: requires canonical ``fold_id`` key. Legacy aliases (``fold_idx``)
    are not supported — splits artifacts from before v1.5 must be regenerated.
    """
    splits = load_json(splits_path)
    folds = splits.get("folds")
    if not folds:
        raise ValueError(f"No folds found in splits file: {splits_path}")
    for fold in folds:
        if "fold_id" not in fold:
            raise ValueError(
                f"Splits file uses non-canonical schema (missing 'fold_id'): "
                f"{splits_path}. Regenerate with `python -m src.splits` (v1.5+)."
            )
        if fold["fold_id"] == fold_idx:
            return fold
    raise ValueError(f"Fold {fold_idx} not found in splits file: {splits_path}")


def _select_split_ids(fold_data: dict[str, Any], split_name: str) -> list[str]:
    """Return paragraph IDs for the requested split, in (normal + anomaly) order.

    v1.5 D_NEW6 + M1: uses canonical ``{split}_normal_ids`` / ``{split}_anomaly_ids``
    keys only. ``--split train`` is supported for ablation/diagnostic scoring of
    the training subset; LogFiT training itself uses normal-only, but scoring
    train at inference time is a legitimate sanity check.
    """
    if split_name not in {"train", "tune", "test"}:
        raise ValueError(
            f"Unknown split_name: {split_name!r}. Expected one of "
            "{'train', 'tune', 'test'}."
        )
    key_normal = f"{split_name}_normal_ids"
    key_anomaly = f"{split_name}_anomaly_ids"
    if key_normal not in fold_data or key_anomaly not in fold_data:
        raise ValueError(
            f"Split {split_name!r} is missing required keys "
            f"({key_normal}, {key_anomaly}) in the splits artifact. "
            "Regenerate splits.json with `python -m src.splits` (v1.5+)."
        )
    return list(fold_data[key_normal]) + list(fold_data[key_anomaly])


def _map_paragraphs_by_id(
    paragraphs: list[Paragraph], ids: list[str]
) -> list[Paragraph]:
    """Map ID list to Paragraph objects.

    v1.5 M2: asserts paragraph_id uniqueness in the input. Silent dedup via
    dict-comprehension would have masked upstream prep bugs.
    """
    lookup: dict[str, Paragraph] = {}
    duplicates: list[str] = []
    for p in paragraphs:
        if p.paragraph_id in lookup:
            duplicates.append(p.paragraph_id)
        else:
            lookup[p.paragraph_id] = p
    if duplicates:
        raise ValueError(
            f"Duplicate paragraph_id values in paragraphs.pkl "
            f"(first 5: {duplicates[:5]}). All paragraph IDs must be unique."
        )

    missing = [pid for pid in ids if pid not in lookup]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} paragraph IDs in paragraphs.pkl "
            f"(first 5: {missing[:5]})."
        )
    return [lookup[pid] for pid in ids]


def _prepare_eval_paragraphs(
    paragraphs: list[Paragraph],
    ids: list[str],
) -> list[Paragraph]:
    """Map IDs to paragraphs and sort by paragraph_id.

    Per spec v1.2 Component 5: sort is lexicographic on the paragraph_id
    string (NOT numeric on the window index). Intentional and deterministic;
    downstream analyses should sort by start_timestamp if chronological
    iteration is needed.
    """
    eval_paragraphs = _map_paragraphs_by_id(paragraphs, ids)
    return sorted(eval_paragraphs, key=lambda p: p.paragraph_id)


# ---------------------------------------------------------------------------
# Scoring loop (R-passes, pooled accumulation, records format)
# ---------------------------------------------------------------------------


def _score_paragraphs_r_passes(
    eval_paragraphs: list[Paragraph],
    model: torch.nn.Module,
    tokenizer: Any,
    backbone_token_limit_value: int,
    sentence_ratio: float,
    token_ratio: float,
    topk_grid: list[int],
    n_passes: int,
    batch_size: int,
    base_seed: int,
) -> list[ParagraphScore]:
    """Compute pooled top-k accuracy per paragraph over ``n_passes`` mask draws.

    Algorithm (v1.5 D_NEW8):
    1. For each pass p in 0..n_passes-1:
         - Build a fresh ``MaskedSentenceDataset`` with seed = base_seed + p.
         - Iterate the dataloader; per masked position, count whether the
           ground-truth token appears in the top-K predictions for each K.
    2. Accumulate per-paragraph (correct_in_top_K, total_masked) across passes.
    3. Compute pooled per-K accuracy:
           accuracy(K) = sum_correct_in_top_K / sum_total_masked
       Pooled (numerator + denominator summed separately) rather than
       averaged-of-fractions: more robust to passes with different mask
       cardinalities.

    Zero-mask edge case (I2):
        If ``sum_total_masked == 0`` for a paragraph (no sentence sampled
        across any pass), all K-accuracies are set to 1.0 by convention.
        ``n_masked_total = 0`` carries the diagnostic forward so downstream
        consumers can filter or flag if desired.

    Output (v1.5 D_NEW7):
        list[ParagraphScore] with ``topk_accuracies`` as list of
        TopKAccuracyRecord (ascending top_k), plus ``n_masked_total`` and
        ``n_passes`` audit fields.

    Determinism:
        Per-pass seed = ``base_seed + pass_idx``. Distinct masks across
        passes, identical sequence of masks across runs.
    """
    if n_passes < 1:
        raise ValueError(f"n_passes must be >= 1, got {n_passes}")
    if not topk_grid:
        raise ValueError("topk_grid must contain at least one value")

    sorted_topk = sorted(topk_grid)
    max_k = sorted_topk[-1]
    device = model.device

    correct_by_para: list[dict[int, int]] = [
        {k: 0 for k in sorted_topk} for _ in eval_paragraphs
    ]
    total_by_para: list[int] = [0] * len(eval_paragraphs)

    collator = MaskedSentenceCollator(tokenizer)

    with torch.no_grad():
        for pass_idx in range(n_passes):
            pass_seed = base_seed + pass_idx
            dataset = MaskedSentenceDataset(
                paragraphs=eval_paragraphs,
                tokenizer=tokenizer,
                backbone_token_limit=backbone_token_limit_value,
                r_sent=sentence_ratio,
                r_tok=token_ratio,
                seed=pass_seed,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collator,
            )

            offset = 0
            for batch in dataloader:
                bs = int(batch["input_ids"].shape[0])
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                top_preds = logits.topk(max_k, dim=-1).indices
                labels = batch["labels"]

                for i in range(bs):
                    para_idx = offset + i
                    mask = labels[i] != -100
                    n_masked = int(mask.sum().item())
                    if n_masked > 0:
                        top_for_mask = top_preds[i][mask]
                        label_mask = labels[i][mask]
                        match = top_for_mask.eq(label_mask.unsqueeze(-1))
                        total_by_para[para_idx] += n_masked
                        for k in sorted_topk:
                            correct_by_para[para_idx][k] += int(
                                match[:, :k].any(dim=-1).sum().item()
                            )
                offset += bs

    scores: list[ParagraphScore] = []
    for idx, para in enumerate(eval_paragraphs):
        if total_by_para[idx] > 0:
            records = [
                TopKAccuracyRecord(
                    top_k=k,
                    accuracy=correct_by_para[idx][k] / total_by_para[idx],
                )
                for k in sorted_topk
            ]
        else:
            # I2: zero-mask edge case. Paper convention: no positions
            # challenged -> "passes" by default at every K.
            records = [
                TopKAccuracyRecord(top_k=k, accuracy=1.0)
                for k in sorted_topk
            ]
        scores.append(
            ParagraphScore(
                paragraph_id=para.paragraph_id,
                label=para.label,
                topk_accuracies=records,
                n_masked_total=total_by_para[idx],
                n_passes=n_passes,
            )
        )
    return scores


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------


def _resolve_output_path(output: Path, split_name: str) -> Path:
    """Resolve --output for single-split mode.

    Heuristics:
    - If the path exists as a directory -> append scores_{split_name}.json.
    - If the path has a file suffix (.json) -> use it as-is.
    - Otherwise (non-existent path with no suffix) -> treat as a directory
      and append scores_{split_name}.json. Parent created in the caller.
    """
    if output.exists() and output.is_dir():
        return output / f"scores_{split_name}.json"
    if output.suffix:
        return output
    return output / f"scores_{split_name}.json"


def _resolve_output_dir(output: Path) -> Path:
    """Resolve --output for ``--split both`` mode (must be a directory)."""
    if output.suffix:
        raise ValueError(
            f"When --split=both, --output must be a directory path, "
            f"got {output} (looks like a file)."
        )
    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.score",
        description=(
            "Score paragraphs with a trained LogFiT model. "
            "v1.5 D_NEW6-9: records format, --n-passes, --batch-size."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paragraphs", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold-idx", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["train", "tune", "test", "both"],
        default="tune",
        help="Which split to score (default: tune). 'both' = tune + test.",
    )
    parser.add_argument(
        "--backbone-decision",
        type=Path,
        default=None,
        help="Optional backbone_choice.json from src.select_backbone. "
             "Raises if the file does not exist (v1.5 I3 fix).",
    )
    parser.add_argument(
        "--topk-grid",
        type=str,
        default=None,
        help="Comma-separated override for inference.topk_grid (e.g. '5,9,12').",
    )
    parser.add_argument(
        "--n-passes",
        type=int,
        default=1,
        help="v1.5 D_NEW8: number of independent mask draws per paragraph. "
             "Default 1 = paper-faithful. R>=2 reduces per-paragraph score "
             "variance at R*compute cost. Cross-run determinism preserved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="v1.5 D_NEW9: override inference batch size. Default = "
             "physical_batch_size from config (Longformer=2, RoBERTa=8). "
             "The default is safe-by-construction; raise only if you've "
             "verified the larger batch fits on your GPU.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override inference.global_seed from config (D_NEW8 base seed).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    backbone_name = _resolve_backbone(cfg, args.backbone_decision)
    backbone_limit = backbone_token_limit(backbone_name)

    masking = cfg["masking"]
    training = cfg["training"]
    inference = cfg.get("inference", {})

    sentence_ratio = float(masking["sentence_ratio"])
    token_ratio = float(masking["token_ratio"])

    # D_NEW9: batch size resolution. Default = physical_batch_size (safe);
    # CLI flag overrides for users who've empirically verified a larger size.
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError(
                f"--batch-size must be positive, got {args.batch_size}"
            )
        batch_size = int(args.batch_size)
    else:
        physical_bs = training.get("physical_batch_size")
        if physical_bs is None:
            physical_bs = _resolve_physical_batch_size(backbone_name)
        batch_size = int(physical_bs)

    # D_NEW8: n_passes (default 1 = paper-faithful).
    n_passes = int(args.n_passes)
    if n_passes < 1:
        raise ValueError(f"--n-passes must be >= 1, got {n_passes}")

    # Seed (CLI override > inference.global_seed > top-level seed > 42).
    if args.seed is not None:
        seed = int(args.seed)
    else:
        seed = int(inference.get("global_seed", cfg.get("seed", 42)))

    # Top-k grid (CLI override > inference.topk_grid > [5, 9, 12]).
    if args.topk_grid is None:
        topk_grid = list(inference.get("topk_grid", [5, 9, 12]))
    else:
        topk_grid = _parse_topk_grid(args.topk_grid)

    paragraphs = load_paragraphs(args.paragraphs)
    fold_data = _load_fold_data(args.splits, args.fold_idx)

    tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def run_split(split_name: str) -> SplitScores:
        ids = _select_split_ids(fold_data, split_name)
        eval_paragraphs = _prepare_eval_paragraphs(paragraphs, ids)
        scores = _score_paragraphs_r_passes(
            eval_paragraphs=eval_paragraphs,
            model=model,
            tokenizer=tokenizer,
            backbone_token_limit_value=backbone_limit,
            sentence_ratio=sentence_ratio,
            token_ratio=token_ratio,
            topk_grid=topk_grid,
            n_passes=n_passes,
            batch_size=batch_size,
            base_seed=seed,
        )
        return SplitScores(
            split_name=split_name,
            topk_grid=sorted(topk_grid),
            scores=scores,
            n_passes=n_passes,
        )

    if args.split == "both":
        output_dir = ensure_dir(_resolve_output_dir(args.output))
        for split_name in ["tune", "test"]:
            split_scores = run_split(split_name)
            output_path = output_dir / f"scores_{split_name}.json"
            save_json(asdict(split_scores), output_path)
            print(f"[score.py] Wrote {output_path}")
    else:
        split_scores = run_split(args.split)
        output_path = _resolve_output_path(args.output, args.split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(asdict(split_scores), output_path)
        print(f"[score.py] Wrote {output_path}")


if __name__ == "__main__":
    main()
