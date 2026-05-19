"""Training loop and checkpointing per spec Component 4.

Implements:
- OneCycleLR wired outside Trainer (D15/T1)
- Gradual unfreezing callback (D15/T3/T5)
- LR grid search (D2)
- Longformer smoke test (Component 1.6)
- Determinism logging + fp16 verification hooks (D8/T7/Q7)

Backbone selection can be driven by the decision artifact written by
src/select_backbone.py via build_training_config(..., backbone_decision_path=...).
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.dataset import MaskedSentenceCollator, MaskedSentenceDataset
from src.splits import FoldSplit, load_splits
from src.token_length_gate import get_backbone_limit
from src.types import Paragraph, TrainedFold
from src.utils.io import (
    ensure_dir,
    get_git_commit_hash,
    load_json,
    load_yaml,
    save_json,
)
from src.utils.seed import (
    check_determinism_env,
    compute_state_dict_sha256,
    compute_weight_divergence,
    set_all_seeds,
)


@dataclass
class TrainingRunConfig:
    dataset: str
    window_seconds: int | None
    backbone: str
    epochs: int
    max_lr: float | None
    lr_grid: list[float]
    effective_batch_size: int
    physical_batch_size: int
    gradient_accumulation_steps: int
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    one_cycle_pct_start: float
    one_cycle_anneal_strategy: str
    one_cycle_div_factor: float
    one_cycle_final_div_factor: float
    one_cycle_three_phase: bool
    epoch_1_unfrozen_layers: tuple[int, ...]
    epoch_2_unfrozen_layers: tuple[int, ...]
    epoch_3_plus_unfrozen_layers: str | tuple[int, ...]
    embeddings_frozen_until_epoch: int
    seed: int
    full_determinism: bool
    fp16: bool
    fp16_weight_divergence_threshold: float
    masking_sentence_ratio: float
    masking_token_ratio: float


class GradualUnfreezingCallback(TrainerCallback):
    """Gradual unfreezing schedule per D15 (v1.2).

    Uses an internal epoch counter to avoid TrainerState.epoch semantics.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epoch_1_unfrozen_layers: Iterable[int],
        epoch_2_unfrozen_layers: Iterable[int],
        epoch_3_plus_unfrozen_layers: str | Iterable[int],
        embeddings_frozen_until_epoch: int,
    ) -> None:
        self.model = model
        self._epoch_count = 0
        self.epoch_1_unfrozen_layers = tuple(epoch_1_unfrozen_layers)
        self.epoch_2_unfrozen_layers = tuple(epoch_2_unfrozen_layers)
        self.epoch_3_plus_unfrozen_layers = epoch_3_plus_unfrozen_layers
        self.embeddings_frozen_until_epoch = embeddings_frozen_until_epoch

        if hasattr(model, "roberta"):
            self.layers = model.roberta.encoder.layer
            self.embeddings = model.roberta.embeddings
        elif hasattr(model, "longformer"):
            self.layers = model.longformer.encoder.layer
            self.embeddings = model.longformer.embeddings
        else:
            raise ValueError(
                f"Unsupported backbone: {type(model).__name__}"
            )

        if len(self.layers) != 12:
            raise ValueError(
                f"Expected 12 encoder layers, got {len(self.layers)}"
            )

    def _set_layer_requires_grad(self, layer_idx: int, requires_grad: bool) -> None:
        for p in self.layers[layer_idx].parameters():
            p.requires_grad = requires_grad

    def _set_embeddings_requires_grad(self, requires_grad: bool) -> None:
        for p in self.embeddings.parameters():
            p.requires_grad = requires_grad

    def _freeze_all_layers(self) -> None:
        for i in range(len(self.layers)):
            self._set_layer_requires_grad(i, False)

    def _unfreeze_layers(self, layers: Iterable[int]) -> None:
        for i in layers:
            self._set_layer_requires_grad(i, True)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_count += 1
        epoch_num = self._epoch_count

        if epoch_num <= self.embeddings_frozen_until_epoch:
            self._set_embeddings_requires_grad(False)
        else:
            self._set_embeddings_requires_grad(True)

        self._freeze_all_layers()

        if epoch_num == 1:
            self._unfreeze_layers(self.epoch_1_unfrozen_layers)
        elif epoch_num == 2:
            self._unfreeze_layers(self.epoch_2_unfrozen_layers)
        else:
            if self.epoch_3_plus_unfrozen_layers == "all":
                self._unfreeze_layers(range(len(self.layers)))
            else:
                self._unfreeze_layers(self.epoch_3_plus_unfrozen_layers)

        return control


class TopKAccuracyLogger(TrainerCallback):
    """Compute top-1 masked-token accuracy each epoch using a SEPARATE eval
    Dataset, so the training Dataset's RNG state is not perturbed.

    B1 fix: previously this callback iterated ``trainer.get_train_dataloader()``,
    which advanced the training Dataset's internal RNG by N calls per epoch.
    Adding/removing this callback (or changing eval frequency) silently
    altered the training mask sequence and final weight SHA256. Now we use
    a dedicated eval Dataset with its own RNG instance — eval cannot
    perturb training.

    The ``reset_rng_each_epoch`` flag (default True) controls whether the
    eval Dataset's RNG is reset to its construction seed at the start of
    every epoch's eval pass:

    - True  (default): eval mask sequence is IDENTICAL every epoch, so the
      logged top-1 accuracy across epochs is directly comparable — any
      delta is attributable to model weight changes alone. Best for
      diagnostic learning curves.
    - False: RNG advances naturally per epoch, so each epoch's top-1 is
      measured on a FRESH mask sample from the training distribution.
      Mirrors training-time stochasticity; the metric measures robustness
      across mask perturbations rather than weight progress alone.

    Note: with default True, only the FINAL epoch's top-1 is used by
    Component 6's threshold-tuning grid, so the practical effect of this
    flag is mainly on intra-training diagnostics.
    """

    def __init__(
        self,
        eval_dataset: MaskedSentenceDataset,
        collator: MaskedSentenceCollator,
        batch_size: int,
        reset_rng_each_epoch: bool = True,
    ) -> None:
        self._eval_dataset = eval_dataset
        self._collator = collator
        self._batch_size = batch_size
        self._reset_rng_each_epoch = reset_rng_each_epoch
        self._trainer: Trainer | None = None

    def attach_trainer(self, trainer: Trainer) -> None:
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._trainer is None:
            return control

        # Reset eval Dataset RNG so every epoch evaluates on the same fixed
        # mask set — makes the top-1 learning curve cleanly comparable
        # across epochs. Disable via reset_rng_each_epoch=False to get
        # fresh masks per epoch instead.
        if self._reset_rng_each_epoch:
            self._eval_dataset.reset_rng()

        model = self._trainer.model
        model_was_training = model.training
        model.eval()

        dataloader = torch.utils.data.DataLoader(
            self._eval_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            collate_fn=self._collator,
            num_workers=0,
        )

        correct = 0
        total = 0
        device = model.device

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                labels = batch["labels"]
                mask = labels != -100
                if mask.any():
                    correct += (preds[mask] == labels[mask]).sum().item()
                    total += mask.sum().item()

        if model_was_training:
            model.train()

        top1_acc = (correct / total) if total > 0 else 0.0
        self._trainer.log({"train_top1_acc": top1_acc})
        return control


def _resolve_physical_batch_size(backbone_name: str) -> int:
    name = backbone_name.lower()
    if "longformer" in name:
        return 2
    if "roberta" in name:
        return 8
    raise ValueError(
        f"Unknown backbone {backbone_name!r}; expected roberta or longformer"
    )


def _resolve_batch_sizes(
    effective_batch_size: int, physical_batch_size: int
) -> tuple[int, int]:
    if physical_batch_size <= 0:
        raise ValueError(
            f"physical_batch_size must be positive, got {physical_batch_size}"
        )
    if effective_batch_size <= 0:
        raise ValueError(
            f"effective_batch_size must be positive, got {effective_batch_size}"
        )
    if effective_batch_size % physical_batch_size != 0:
        raise ValueError(
            "effective_batch_size must be divisible by physical_batch_size "
            f"(got {effective_batch_size} vs {physical_batch_size})"
        )
    return physical_batch_size, effective_batch_size // physical_batch_size


def _compute_steps_per_epoch(
    n_samples: int,
    physical_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Optimizer-update steps per epoch, matching HF Trainer 4.45's behavior.

    HF's `_inner_training_loop` triggers `optimizer.step() + scheduler.step()`
    whenever ``(step + 1) % accum == 0 OR (step + 1) == steps_in_epoch``. The
    second clause flushes the partial accumulation group at the end of every
    epoch, so the per-epoch update count is::

        ceil(len_dataloader / accum) = ceil(N / (bs * accum))

    (The two forms are mathematically equivalent for positive integers.)

    Earlier revisions of this helper used ``floor(len_dataloader / accum)``
    on the (incorrect) assumption that HF dropped the remainder. That
    undercounts by 1 whenever there's a partial group at end-of-epoch and
    causes ``OneCycleLR`` to raise ``ValueError: Tried to step X times``
    when HF calls scheduler.step() once more than total_steps allows.

    References: HF trainer.py do_sync_step logic; issue #36297.
    """
    return max(
        1, math.ceil(n_samples / (physical_batch_size * gradient_accumulation_steps))
    )


def load_paragraphs(paragraphs_pkl_path: Path | str) -> list[Paragraph]:
    paragraphs_pkl_path = Path(paragraphs_pkl_path)
    with paragraphs_pkl_path.open("rb") as f:
        return pickle.load(f)


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


def _find_longest_paragraph_index(
    paragraphs: list[Paragraph], tokenizer: Any
) -> int:
    max_len = -1
    max_idx = 0
    for i, p in enumerate(paragraphs):
        total = 2  # BOS + EOS
        for line in p.lines:
            total += len(tokenizer.encode(line, add_special_tokens=False))
        if total > max_len:
            max_len = total
            max_idx = i
    return max_idx


def _longformer_smoke_test(
    model: torch.nn.Module,
    dataset: MaskedSentenceDataset,
    collator: MaskedSentenceCollator,
    paragraph_idx: int,
) -> None:
    dummy_batch = collator([dataset[paragraph_idx]])
    device = model.device
    dummy_batch = {k: v.to(device) for k, v in dummy_batch.items()}
    with torch.no_grad():
        model(**dummy_batch)


def _build_training_args(
    output_dir: Path,
    config: TrainingRunConfig,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.physical_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay,
        adam_beta1=config.betas[0],
        adam_beta2=config.betas[1],
        adam_epsilon=config.eps,
        logging_steps=50,
        save_strategy="epoch",
        seed=config.seed,
        data_seed=config.seed,
        full_determinism=config.full_determinism,
        fp16=config.fp16,
        report_to=[],
        dataloader_num_workers=0,
    )


def _build_one_cycle_lr(
    optimizer: torch.optim.Optimizer,
    max_lr: float,
    total_steps: int,
    config: TrainingRunConfig,
) -> torch.optim.lr_scheduler.OneCycleLR:
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=config.one_cycle_pct_start,
        anneal_strategy=config.one_cycle_anneal_strategy,
        div_factor=config.one_cycle_div_factor,
        final_div_factor=config.one_cycle_final_div_factor,
        three_phase=config.one_cycle_three_phase,
    )


def _extract_epoch_losses(log_history: list[dict[str, Any]]) -> list[float]:
    last_loss_by_epoch: dict[int, float] = {}
    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            epoch_idx = int(round(entry["epoch"]))
            last_loss_by_epoch[epoch_idx] = float(entry["loss"])
    return [last_loss_by_epoch[k] for k in sorted(last_loss_by_epoch.keys())]


def _extract_last_top1_acc(log_history: list[dict[str, Any]]) -> float:
    for entry in reversed(log_history):
        if "train_top1_acc" in entry:
            return float(entry["train_top1_acc"])
    return 0.0


def _lr_grid_search(
    train_paragraphs: list[Paragraph],
    tokenizer: Any,
    config: TrainingRunConfig,
    output_dir: Path,
    backbone_token_limit: int,
) -> tuple[float, list[dict[str, float]]]:
    subset = train_paragraphs[:500] if len(train_paragraphs) > 500 else train_paragraphs
    results: list[dict[str, float]] = []

    for lr in config.lr_grid:
        model = AutoModelForMaskedLM.from_pretrained(config.backbone)
        dataset = MaskedSentenceDataset(
            paragraphs=subset,
            tokenizer=tokenizer,
            backbone_token_limit=backbone_token_limit,
            r_sent=config.masking_sentence_ratio,
            r_tok=config.masking_token_ratio,
            seed=config.seed,
        )
        collator = MaskedSentenceCollator(tokenizer)

        steps_per_epoch = _compute_steps_per_epoch(
            n_samples=len(dataset),
            physical_batch_size=config.physical_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        total_steps = steps_per_epoch  # 1 epoch for grid search

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        scheduler = _build_one_cycle_lr(optimizer, lr, total_steps, config)

        args = TrainingArguments(
            output_dir=str(output_dir / f"lr_search_{lr:g}"),
            num_train_epochs=1,
            per_device_train_batch_size=config.physical_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            adam_beta1=config.betas[0],
            adam_beta2=config.betas[1],
            adam_epsilon=config.eps,
            logging_steps=50,
            save_strategy="no",
            seed=config.seed,
            data_seed=config.seed,
            full_determinism=config.full_determinism,
            fp16=config.fp16,
            report_to=[],
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=collator,
            optimizers=(optimizer, scheduler),
        )
        train_output = trainer.train()
        results.append({"lr": float(lr), "loss": float(train_output.training_loss)})

    best = min(results, key=lambda r: r["loss"])
    return best["lr"], results


def train_fold(
    paragraphs: list[Paragraph],
    fold: FoldSplit,
    config: TrainingRunConfig,
    runs_root: Path | str,
    resume_from_checkpoint: str | None = None,
    return_model: bool = False,
) -> tuple[TrainedFold, torch.nn.Module | None]:
    """Train a single fold on normal paragraphs only."""
    runs_root = Path(runs_root)
    run_dir = ensure_dir(
        runs_root
        / f"{config.dataset}_{config.window_seconds or 'hdfs'}"
        / f"fold_{fold.fold_id}"
    )

    set_all_seeds(config.seed)

    train_paragraphs = _map_paragraphs_by_id(
        paragraphs, fold.train_normal_ids
    )

    tokenizer = AutoTokenizer.from_pretrained(config.backbone, use_fast=True)
    backbone_token_limit = get_backbone_limit(config.backbone)

    dataset = MaskedSentenceDataset(
        paragraphs=train_paragraphs,
        tokenizer=tokenizer,
        backbone_token_limit=backbone_token_limit,
        r_sent=config.masking_sentence_ratio,
        r_tok=config.masking_token_ratio,
        seed=config.seed,
    )
    collator = MaskedSentenceCollator(tokenizer)

    max_lr = config.max_lr
    lr_grid_results: list[dict[str, float]] = []
    if max_lr is None:
        max_lr, lr_grid_results = _lr_grid_search(
            train_paragraphs=train_paragraphs,
            tokenizer=tokenizer,
            config=config,
            output_dir=run_dir,
            backbone_token_limit=backbone_token_limit,
        )
        # B2: grid search trained 4 models, advancing the global torch RNG.
        # Re-seed so main training is identical whether max_lr was searched
        # or provided directly. Preserves the T7 weight-SHA256 audit.
        set_all_seeds(config.seed)

    model = AutoModelForMaskedLM.from_pretrained(config.backbone)

    steps_per_epoch = _compute_steps_per_epoch(
        n_samples=len(dataset),
        physical_batch_size=config.physical_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    total_steps = config.epochs * steps_per_epoch

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_one_cycle_lr(optimizer, max_lr, total_steps, config)

    args = _build_training_args(run_dir, config)

    # B1: separate eval Dataset for the top-1 logger so eval doesn't perturb
    # the training Dataset's RNG. seed+1 keeps the eval mask stream decoupled
    # from any single training epoch's mask stream.
    eval_dataset = MaskedSentenceDataset(
        paragraphs=train_paragraphs,
        tokenizer=tokenizer,
        backbone_token_limit=backbone_token_limit,
        r_sent=config.masking_sentence_ratio,
        r_tok=config.masking_token_ratio,
        seed=config.seed + 1,
    )

    unfreeze_cb = GradualUnfreezingCallback(
        model=model,
        epoch_1_unfrozen_layers=config.epoch_1_unfrozen_layers,
        epoch_2_unfrozen_layers=config.epoch_2_unfrozen_layers,
        epoch_3_plus_unfrozen_layers=config.epoch_3_plus_unfrozen_layers,
        embeddings_frozen_until_epoch=config.embeddings_frozen_until_epoch,
    )
    top1_cb = TopKAccuracyLogger(
        eval_dataset=eval_dataset,
        collator=collator,
        batch_size=config.physical_batch_size,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        optimizers=(optimizer, scheduler),
        callbacks=[unfreeze_cb, top1_cb],
    )
    top1_cb.attach_trainer(trainer)

    if "longformer" in config.backbone.lower():
        idx = _find_longest_paragraph_index(train_paragraphs, tokenizer)
        try:
            _longformer_smoke_test(model, dataset, collator, idx)
        except Exception as e:
            raise RuntimeError(
                f"Longformer smoke test failed: {e}"
            ) from e

    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(run_dir / "final_model"))

    model_weight_sha256 = compute_state_dict_sha256(model.state_dict())
    log_history = trainer.state.log_history
    train_top1_acc = _extract_last_top1_acc(log_history)
    epoch_losses = _extract_epoch_losses(log_history)

    training_failed = train_top1_acc < 0.5

    # B3: write the audit log BEFORE raising on failure. Otherwise diagnostic
    # fields (epoch_losses, max_lr, lr_grid_results, determinism_env) are lost
    # on the exact runs where they're most needed.
    train_log = {
        "dataset": config.dataset,
        "window_seconds": config.window_seconds,
        "fold_idx": fold.fold_id,
        "backbone": config.backbone,
        "git_commit": get_git_commit_hash(),
        "seed": config.seed,
        "determinism_env": check_determinism_env(),
        "max_lr": max_lr,
        "lr_grid_results": lr_grid_results,
        "training_loss": float(train_output.training_loss),
        "epoch_losses": epoch_losses,
        "train_top1_acc": train_top1_acc,
        "model_weight_sha256": model_weight_sha256,
        "training_failed": training_failed,
    }
    save_json(train_log, run_dir / "train_log.json")

    if training_failed:
        raise RuntimeError(
            f"Training failed: train_top1_acc={train_top1_acc:.3f} < 0.5 "
            f"(diagnostics: {run_dir / 'train_log.json'})"
        )

    trained = TrainedFold(
        fold_idx=fold.fold_id,
        model_path=run_dir / "final_model",
        train_top1_acc=train_top1_acc,
        model_weight_sha256=model_weight_sha256,
        epoch_losses=epoch_losses,
        deterministic_warnings=[],
    )
    return (trained, model) if return_model else (trained, None)


def run_fp16_verification(
    paragraphs: list[Paragraph],
    fold: FoldSplit,
    config: TrainingRunConfig,
    runs_root: Path | str,
) -> None:
    """Run the same fold twice and compare fp16 determinism."""
    if not config.fp16:
        return

    set_all_seeds(config.seed)
    trained_a, model_a = train_fold(
        paragraphs=paragraphs,
        fold=fold,
        config=config,
        runs_root=Path(runs_root) / "fp16_verify_a",
        return_model=True,
    )
    if model_a is None:
        return
    hash_a = compute_state_dict_sha256(model_a.state_dict())

    set_all_seeds(config.seed)
    trained_b, model_b = train_fold(
        paragraphs=paragraphs,
        fold=fold,
        config=config,
        runs_root=Path(runs_root) / "fp16_verify_b",
        return_model=True,
    )
    if model_b is None:
        return
    hash_b = compute_state_dict_sha256(model_b.state_dict())

    if hash_a != hash_b:
        divergences = compute_weight_divergence(
            model_a.state_dict(), model_b.state_dict()
        )
        max_div = max(divergences.values()) if divergences else 0.0
        if max_div > config.fp16_weight_divergence_threshold:
            raise RuntimeError(
                "FP16 verification failed: max divergence "
                f"{max_div:.6f} exceeds threshold "
                f"{config.fp16_weight_divergence_threshold}"
            )


def build_training_config(
    config_path: Path | str,
    backbone_decision_path: Path | str | None = None,
) -> TrainingRunConfig:
    cfg = load_yaml(config_path)

    dataset = cfg["dataset"]
    window_seconds = cfg.get("window_seconds")
    backbone_cfg = cfg["backbone"]
    training = cfg["training"]
    determinism = cfg["determinism"]
    masking = cfg["masking"]

    backbone_name = backbone_cfg["roberta_id"]
    if training.get("use_longformer", False):
        backbone_name = backbone_cfg["longformer_id"]

    if backbone_decision_path is not None:
        decision_path = Path(backbone_decision_path)
        if decision_path.exists():
            decision_data = load_json(decision_path)
            if "chosen_backbone" not in decision_data:
                raise ValueError(
                    "Backbone decision artifact is missing 'chosen_backbone': "
                    f"{decision_path}"
                )
            backbone_name = decision_data["chosen_backbone"]

    if training.get("backbone"):
        backbone_name = training["backbone"]

    physical_bs = training.get("physical_batch_size")
    if physical_bs is None:
        physical_bs = _resolve_physical_batch_size(backbone_name)

    physical_bs, accum_steps = _resolve_batch_sizes(
        training["effective_batch_size"], physical_bs
    )

    # Defensive numeric coercion. PyYAML's safe_load follows YAML 1.1 float
    # semantics: values like `1.0e4` (no sign on exponent) silently parse as
    # STRINGS, not floats. Earlier this bit `final_div_factor` and only
    # surfaced inside OneCycleLR's first step() call. Coercing here means a
    # mis-typed config raises ValueError at build time, with a clear traceback.
    max_lr_val = training.get("max_lr")

    return TrainingRunConfig(
        dataset=dataset,
        window_seconds=window_seconds,
        backbone=backbone_name,
        epochs=training["epochs"],
        max_lr=float(max_lr_val) if max_lr_val is not None else None,
        lr_grid=[float(x) for x in training["lr_grid"]],
        effective_batch_size=training["effective_batch_size"],
        physical_batch_size=physical_bs,
        gradient_accumulation_steps=accum_steps,
        betas=(float(training["betas"][0]), float(training["betas"][1])),
        eps=float(training["eps"]),
        weight_decay=float(training["weight_decay"]),
        one_cycle_pct_start=float(training["lr_scheduler"]["pct_start"]),
        one_cycle_anneal_strategy=training["lr_scheduler"]["anneal_strategy"],
        one_cycle_div_factor=float(training["lr_scheduler"]["div_factor"]),
        one_cycle_final_div_factor=float(
            training["lr_scheduler"]["final_div_factor"]
        ),
        one_cycle_three_phase=training["lr_scheduler"].get("three_phase", False),
        epoch_1_unfrozen_layers=tuple(
            training["gradual_unfreezing_schedule"]["epoch_1_unfrozen_layers"]
        ),
        epoch_2_unfrozen_layers=tuple(
            training["gradual_unfreezing_schedule"]["epoch_2_unfrozen_layers"]
        ),
        epoch_3_plus_unfrozen_layers=training["gradual_unfreezing_schedule"][
            "epoch_3_plus_unfrozen_layers"
        ],
        embeddings_frozen_until_epoch=training["gradual_unfreezing_schedule"][
            "embeddings_frozen_until_epoch"
        ],
        seed=cfg.get("seed", 42),
        full_determinism=determinism["hf_full_determinism"],
        fp16=training["fp16"],
        fp16_weight_divergence_threshold=float(
            determinism["fp16_weight_divergence_threshold"]
        ),
        masking_sentence_ratio=float(masking["sentence_ratio"]),
        masking_token_ratio=float(masking["token_ratio"]),
    )


def train_fold_from_paths(
    config_path: Path | str,
    paragraphs_pkl_path: Path | str,
    splits_path: Path | str,
    fold_idx: int,
    backbone_decision_path: Path | str | None = None,
    runs_root: Path | str = "runs",
    resume_from_checkpoint: str | None = None,
    fp16_verification: bool = False,
) -> TrainedFold:
    config = build_training_config(config_path, backbone_decision_path)
    paragraphs = load_paragraphs(paragraphs_pkl_path)
    splits = load_splits(splits_path)
    fold = next(f for f in splits.folds if f.fold_id == fold_idx)

    if fp16_verification:
        run_fp16_verification(paragraphs, fold, config, runs_root)

    trained, _ = train_fold(
        paragraphs=paragraphs,
        fold=fold,
        config=config,
        runs_root=runs_root,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return trained


def _build_arg_parser():
    """CLI for one-fold training, invoked by scripts/train.sh."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Train one fold of the LogFiT reproduction (spec §4).",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Dataset config YAML (e.g., configs/hdfs.yaml).",
    )
    parser.add_argument(
        "--paragraphs",
        required=True,
        type=Path,
        help="Path to paragraphs.pkl produced by src.prepare_*.",
    )
    parser.add_argument(
        "--splits",
        required=True,
        type=Path,
        help="Path to splits.json produced by src.splits.",
    )
    parser.add_argument(
        "--fold-idx",
        required=True,
        type=int,
        help="Fold index (0..N_FOLDS-1). On SLURM array jobs, pass $SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--backbone-decision",
        type=Path,
        default=None,
        help="Optional backbone_choice.json from src.select_backbone. "
             "If the file doesn't exist, falls through to YAML default.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory for run outputs (default: ./runs).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume from (HF Trainer convention).",
    )
    parser.add_argument(
        "--fp16-verify",
        action="store_true",
        help="Run T7 fp16 verification: trains the same fold TWICE and "
             "compares weight SHA256. Doubles training cost — use sparingly.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    trained = train_fold_from_paths(
        config_path=args.config,
        paragraphs_pkl_path=args.paragraphs,
        splits_path=args.splits,
        fold_idx=args.fold_idx,
        backbone_decision_path=args.backbone_decision,
        runs_root=args.runs_root,
        resume_from_checkpoint=args.resume,
        fp16_verification=args.fp16_verify,
    )
    print(
        f"[train.py] Fold {trained.fold_idx} complete: "
        f"top1_acc={trained.train_top1_acc:.4f}, "
        f"sha256={trained.model_weight_sha256[:16]}..."
    )
    print(f"[train.py] Model saved to: {trained.model_path}")


if __name__ == "__main__":
    main()
