"""Paragraph-level top-k scoring for LogFiT models (Component 5)."""

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
from src.train import load_paragraphs
from src.types import Paragraph, ParagraphScore
from src.utils.io import ensure_dir, load_json, load_yaml, save_json


def _parse_topk_grid(value: str) -> list[int]:
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


def _resolve_backbone(
    cfg: dict[str, Any],
    backbone_decision_path: Path | None,
) -> str:
    backbone_cfg = cfg["backbone"]
    training = cfg["training"]

    backbone_name = backbone_cfg["roberta_id"]
    if training.get("use_longformer", False):
        backbone_name = backbone_cfg["longformer_id"]

    if backbone_decision_path is not None:
        decision_path = Path(backbone_decision_path)
        if decision_path.exists():
            decision_data = load_json(decision_path)
            chosen = decision_data.get("chosen_backbone")
            if chosen is None:
                raise ValueError(
                    "Backbone decision artifact is missing 'chosen_backbone': "
                    f"{decision_path}"
                )
            backbone_name = chosen

    if training.get("backbone"):
        backbone_name = training["backbone"]

    return backbone_name


def _load_fold_data(splits_path: Path, fold_idx: int) -> dict[str, Any]:
    splits = load_json(splits_path)
    folds = splits.get("folds")
    if not folds:
        raise ValueError(f"No folds found in splits file: {splits_path}")
    for fold in folds:
        fold_id = fold.get("fold_id", fold.get("fold_idx"))
        if fold_id == fold_idx:
            return fold
    raise ValueError(f"Fold {fold_idx} not found in splits file: {splits_path}")


def _get_id_list(fold_data: dict[str, Any], keys: list[str]) -> list[str] | None:
    for key in keys:
        if key in fold_data:
            return list(fold_data[key])
    return None


def _select_split_ids(fold_data: dict[str, Any], split_name: str) -> list[str]:
    if split_name == "tune":
        normal = _get_id_list(fold_data, ["tune_normal", "tune_normal_ids"])
        anomaly = _get_id_list(
            fold_data, ["tune_anomaly", "tune_anomaly_ids"]
        )
    elif split_name == "test":
        normal = _get_id_list(fold_data, ["test_normal", "test_normal_ids"])
        anomaly = _get_id_list(
            fold_data, ["test_anomaly", "test_anomaly_ids"]
        )
    else:
        raise ValueError(f"Unknown split_name: {split_name}")

    if normal is None or anomaly is None:
        raise ValueError(
            f"Split '{split_name}' missing normal/anomaly IDs in splits file"
        )
    return list(normal) + list(anomaly)


def _map_paragraphs_by_id(
    paragraphs: list[Paragraph], ids: list[str]
) -> list[Paragraph]:
    lookup = {p.paragraph_id: p for p in paragraphs}
    missing = [pid for pid in ids if pid not in lookup]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} paragraph IDs in paragraphs.pkl"
        )
    return [lookup[pid] for pid in ids]


def _prepare_eval_paragraphs(
    paragraphs: list[Paragraph],
    ids: list[str],
) -> list[Paragraph]:
    eval_paragraphs = _map_paragraphs_by_id(paragraphs, ids)
    return sorted(eval_paragraphs, key=lambda p: p.paragraph_id)


def _score_paragraphs(
    eval_paragraphs: list[Paragraph],
    model: torch.nn.Module,
    dataloader: DataLoader,
    topk_grid: list[int],
) -> list[ParagraphScore]:
    scores: list[ParagraphScore] = []
    max_k = max(topk_grid)
    device = model.device
    offset = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_size = int(batch["input_ids"].shape[0])
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            top_preds = logits.topk(max_k, dim=-1).indices
            labels = batch["labels"]

            for i in range(batch_size):
                para = eval_paragraphs[offset + i]
                mask = labels[i] != -100
                if bool(mask.any().item()):
                    top_for_mask = top_preds[i][mask]
                    label_mask = labels[i][mask]
                    match = top_for_mask.eq(label_mask.unsqueeze(-1))
                    total = int(label_mask.numel())
                    topk_accuracies = {}
                    for k in topk_grid:
                        correct = match[:, :k].any(dim=-1).sum().item()
                        topk_accuracies[k] = correct / total
                else:
                    topk_accuracies = {k: 1.0 for k in topk_grid}

                scores.append(
                    ParagraphScore(
                        paragraph_id=para.paragraph_id,
                        label=para.label,
                        topk_accuracies=topk_accuracies,
                    )
                )

            offset += batch_size

    return scores


def _resolve_output_path(output: Path, split_name: str) -> Path:
    if output.exists() and output.is_dir():
        return output / f"scores_{split_name}.json"
    if output.suffix:
        return output
    return output / f"scores_{split_name}.json"


def _resolve_output_dir(output: Path) -> Path:
    if output.suffix:
        raise ValueError(
            "When scoring both splits, output must be a directory path"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score paragraphs with a trained LogFiT model."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paragraphs", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold-idx", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["tune", "test", "both"],
        default="tune",
        help="Which split to score",
    )
    parser.add_argument(
        "--backbone-decision",
        type=Path,
        default=None,
        help="Optional backbone_decision.json to override config backbone",
    )
    parser.add_argument(
        "--topk-grid",
        type=str,
        default=None,
        help="Comma-separated override for inference.topk_grid (e.g. 5,9,12)",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    backbone_name = _resolve_backbone(cfg, args.backbone_decision)
    backbone_limit = backbone_token_limit(backbone_name)

    masking = cfg["masking"]
    training = cfg["training"]
    inference = cfg.get("inference", {})

    sentence_ratio = float(masking["sentence_ratio"])
    token_ratio = float(masking["token_ratio"])
    batch_size = int(training["effective_batch_size"])
    seed = int(inference.get("global_seed", cfg.get("seed", 42)))

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

    def run_split(split_name: str) -> list[ParagraphScore]:
        ids = _select_split_ids(fold_data, split_name)
        eval_paragraphs = _prepare_eval_paragraphs(paragraphs, ids)
        dataset = MaskedSentenceDataset(
            paragraphs=eval_paragraphs,
            tokenizer=tokenizer,
            backbone_token_limit=backbone_limit,
            r_sent=sentence_ratio,
            r_tok=token_ratio,
            seed=seed,
        )
        collator = MaskedSentenceCollator(tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        return _score_paragraphs(eval_paragraphs, model, dataloader, topk_grid)

    if args.split == "both":
        output_dir = ensure_dir(_resolve_output_dir(args.output))
        for split_name in ["tune", "test"]:
            scores = run_split(split_name)
            output_path = output_dir / f"scores_{split_name}.json"
            save_json([asdict(s) for s in scores], output_path)
            print(f"Wrote {output_path}")
    else:
        scores = run_split(args.split)
        output_path = _resolve_output_path(args.output, args.split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json([asdict(s) for s in scores], output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
