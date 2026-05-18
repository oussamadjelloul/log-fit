# LogFiT Reproduction — Technique & Methodology Specification

**Version:** 1.2 (post-second-pass-adversarial-review)
**Status:** Draft for supervisor sign-off (paired with `logfit-repro-decisions-v1.2.md`)
**Author:** Oussama
**Source paper:** Almodovar et al., IEEE TNSM 21(2):1715–1723, 2024. DOI: 10.1109/TNSM.2024.3358730
**Companion doc:** `logfit-repro-decisions-v1.2.md`
**Scope:** This document is _how the reproduction works_. All choices traced via tags `[L#]` (locked from paper) and `[D#]` (decision by reproducer).

## Changelog v1.1 → v1.2

Post second-pass tri-LLM adversarial review. v1.1 fixes for v1.0 BLOCKING items judged real; v1.1 introduced new bugs in Components 4 and 8 (the largest v1.1 rewrites). v1.2 closes the 2 convergent BLOCKING + 4 convergent IMPORTANT findings + 4 MINOR polish items.

- **§1.6 (Token-Length Validation Gate)** — Rationale paragraph added per reviewer F9 (R2). Longformer attention-window padding note added per F11.
- **Component 4.1 (REWRITTEN v1.2):** LR scheduler resolved per **T1**. `torch.optim.lr_scheduler.OneCycleLR` instantiated outside `Trainer`, passed via `optimizers=(optimizer, scheduler)`. `full_determinism=True` added per **T7**.
- **Component 4.2 (REWRITTEN v1.2):** Callback uses `self._epoch_count` internal counter per **T5** (eliminates `TrainerState.epoch` version-sensitivity). Embeddings unfrozen in epoch 3+ branch per **T3** (corrects v1.1 permanent freeze).
- **Component 5:** Lex-sort comment added per F8 (R2) — sort is deterministic and intentional, not numeric.
- **Component 8 (REVISED v1.2):**
  - `strip_trailing_punct` function explicitly defined per **T2**.
  - Whitespace preservation via `re.split(r'(\s+)', line)` per **T4** (fixes destructive tab collapse).
  - Multi-word WordNet lemma filter added per **T6** (skips `open_up`, `carry_out`, etc.).
- **Component 10 (YAML):** Updated for OneCycleLR config + `full_determinism`.
- **Component 13 (test suite):** 4 new tests — `test_strip_trailing_punct`, `test_variability_no_multiword_lemmas`, `test_unfreezing_schedule_transitions`, `test_variability_whitespace_preserved`.
- **Component 14 (reproducibility checklist):** Added per-fold model-weight SHA256 verification step per **T7** fp16 mitigation.

---

## Component 0: Notation and Definitions (Unchanged)

See v1.1 Component 0. Symbols: `line`, `P`, `S`, `T`, `x`, `M`, `r_sent=0.5`, `r_tok=0.8`, `K ∈ {5,9,12}`, `θ`, `acc_topk(P)`, `y_P`, `f_θ`, `q_0.8`.

---

## Component 1: Data Preparation

### 1.1 HDFS Pipeline (Unchanged from v1.1)

Multi-block-id handling per **D16** — extract all `blk_-?\d+` matches per line, duplicate line into each block's paragraph, surface `lines_with_multiple_blockids` counter. Hard fail on missing label assertion. First-appearance offset sort.

### 1.2 BGL / Thunderbird Pipeline (Unchanged from v1.1)

Time-window grouping (10s, 30s, 60s). Hard fail if `unparseable_timestamp / total_lines > 0.001`.

### 1.3 Sample Allocation (Unchanged from v1.1)

25,000 normal + 2,000 anomaly per dataset, seed=42, persisted to `allocated_indices.json`.

### 1.4 5-Fold Split Construction (Unchanged from v1.1 — D14)

Normals k-fold-partitioned. Anomalies non-partitioned, per-fold shuffled with `seed=42+fold_idx`. Within-fold disjointness invariants verified by tests. Cross-fold anomaly overlap acknowledged (now quantified in decisions-v1.2 §2.3).

### 1.5 Persistence Schema (Unchanged from v1.1)

See v1.1 §1.5. `paragraphs.pkl` records + `preparation_summary.json` with drop counters, word-length distribution, token-length distribution.

### 1.6 Token-Length Validation Gate (REVISED v1.2 — rationale + Longformer note)

**Trigger:** Once per (dataset × window) after §1.3 allocation, before §1.4 fold construction.

**Algorithm:** Unchanged from v1.1 — tokenize per-line per Component 3.1 protocol; compute p50/p80/p95/p99/max; check thresholds.

**Thresholds and rationale (NEW v1.2):**

- **HARD FAIL trigger:** `max > 5 × backbone_limit`. **Rationale:** Right-truncating a paragraph at 5× the backbone limit discards 1 − 1/5 = **80% of paragraph content**. Below this point, the paragraph reaching the model is more "first 20% of the window" than "the window", and the anomaly signal (which can occur anywhere in the window) is systematically biased toward early-window events. 5× is the threshold at which "right-truncation is incidental" transitions to "right-truncation defines what's evaluated."
- **SUPERVISOR_FLAG trigger:** `p95 > backbone_limit`. **Rationale:** p95 > limit means ≥5% of paragraphs are truncated. Standard practice in MLM literature treats >5% truncation as a confound requiring disclosure rather than silent acceptance.
- **Alternative considered:** HARD FAIL on `p99 > 5×limit` instead of `max > 5×limit` (more robust to single-paragraph outliers). Rejected because a single 50,000-token paragraph in BGL/TB indicates a malformed window that warrants pipeline review, not silent acceptance.

**Longformer attention-window padding (NEW v1.2 — reviewer F11):**

Longformer requires input sequences padded to a multiple of the attention window size (default 512). The token-length gate's `total_tokens` calculation does not account for this padding. Practical consequence:

- A paragraph with 4,095 tokens passes the gate (4,095 ≤ 4,096) but Longformer's forward pass will pad it to the next multiple of 512 → 4,608 tokens, which exceeds `max_position_embeddings=4,098`. The forward pass either truncates (losing the EOS positioning) or fails.
- **Mitigation:** Treat the effective Longformer limit as `4,096 - (4,096 % attention_window_size) = 4,096 - 0 = 4,096` for clean cases, but verify empirically before the first batched forward pass. Add a one-time empirical check during smoke testing: tokenize the longest paragraph, run a single forward pass at the chosen `per_device_train_batch_size`, confirm no OOM and no shape mismatch.
- This empirical verification is part of the §8 execution sequence step 7 (single-fold smoke test).

---

## Component 2: Backbone Selection [L1, L2] (Unchanged)

`q_0.8(word_counts) ≤ 512 → roberta-base`, else `longformer-base-4096`.

---

## Component 3: Masking (Unchanged from v1.1)

§3.1 tokenize-per-line concatenation produces clean `sent_boundaries`. §3.2 masking algorithm: 50% sentences → 80% tokens within each, special tokens never masked, on-the-fly per epoch.

---

## Component 4: Training Procedure (REWRITTEN v1.2 — T1 + T3 + T5 + T7)

### 4.1 Training Configuration (REVISED v1.2 — OneCycleLR + full_determinism)

```python
import os
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# T7: Determinism setup (D8) — env vars set in SLURM script, not here
# CUBLAS_WORKSPACE_CONFIG=:4096:8
# PYTHONHASHSEED=0
set_all_seeds(42)   # Python random, NumPy, torch, torch.cuda, transformers

# T7: full_determinism flag (canonical HF determinism flag)
training_args = TrainingArguments(
    output_dir=f"runs/{dataset}_{window}/fold_{fold_idx}",
    num_train_epochs=epochs,                  # 3 for HDFS, 5 for BGL/TB [D1]
    per_device_train_batch_size=physical_bs,  # 8 for RoBERTa, 2 for Longformer [D3]
    gradient_accumulation_steps=accum_steps,  # reach effective_batch_size
    weight_decay=0.01,                        # [L10]
    adam_beta1=0.9,                            # [L10]
    adam_beta2=0.99,                           # [L10]
    adam_epsilon=1e-5,                         # [L10]
    # NOTE v1.2: learning_rate, lr_scheduler_type, warmup_ratio REMOVED.
    # OneCycleLR built manually below and passed via Trainer(optimizers=...).
    logging_steps=50,
    save_strategy="epoch",
    seed=42,
    data_seed=42,
    full_determinism=True,                    # NEW v1.2 [T7] — HF canonical determinism flag
    fp16=True,                                 # accepted residual non-determinism, see D8 fp16 disclosure
    report_to=[],
    dataloader_num_workers=0,
)

model = AutoModelForMaskedLM.from_pretrained(backbone_name)
tokenizer = AutoTokenizer.from_pretrained(backbone_name)

# T1: Build OneCycleLR per D15
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=max_lr,                                 # placeholder; OneCycleLR overrides
    betas=(0.9, 0.99),
    eps=1e-5,
    weight_decay=0.01,
)

steps_per_epoch = math.ceil(len(train_dataset) / (physical_bs * accum_steps))
total_steps = epochs * steps_per_epoch

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,                             # from D2 grid search
    total_steps=total_steps,
    pct_start=0.3,                              # PyTorch default; Smith & Topin recommend early peak
    anneal_strategy='cos',                      # PyTorch default
    div_factor=25.0,                            # initial_lr = max_lr / 25
    final_div_factor=1e4,                       # final_lr = max_lr / 25 / 1e4
    three_phase=False,                          # two-phase: ramp-up + decay
)
# OneCycleLR steps once per training batch. Trainer calls scheduler.step()
# automatically when scheduler is passed via `optimizers` arg.

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=MaskedSentenceDataset(train_paragraphs, tokenizer, r_sent=0.5, r_tok=0.8),
    data_collator=MaskedSentenceCollator(tokenizer),    # implements Component 3
    optimizers=(optimizer, scheduler),         # NEW v1.2 [T1] — bypass HF's scheduler creation
    callbacks=[GradualUnfreezingCallback(model), TopKAccuracyLogger()],
)
```

### 4.2 GradualUnfreezingCallback (REVISED v1.2 — internal counter + embedding unfreeze)

```python
from transformers import TrainerCallback

class GradualUnfreezingCallback(TrainerCallback):
    """
    Schedule (per D15, v1.2 revised):
    - Epoch 1: LM head + last 2 encoder layers (10, 11) unfrozen.
                Embeddings + layers 0-9 frozen.
    - Epoch 2: LM head + last 4 encoder layers (8, 9, 10, 11) unfrozen.
                Embeddings + layers 0-7 frozen.
    - Epoch 3+: ALL parameters unfrozen, including embeddings.

    T5 [v1.2]: schedule is indexed by an internal counter (self._epoch_count),
    not TrainerState.epoch, to eliminate version-sensitivity to transformers internals.
    """

    def __init__(self, model):
        self.model = model
        self._epoch_count = 0                  # NEW v1.2 [T5]

        # Identify encoder layers — works for RoBERTa and Longformer (both 12 layers)
        if hasattr(model, "roberta"):
            self.layers = model.roberta.encoder.layer
            self.embeddings = model.roberta.embeddings
        elif hasattr(model, "longformer"):
            self.layers = model.longformer.encoder.layer
            self.embeddings = model.longformer.embeddings
        else:
            raise ValueError(f"Unsupported backbone: {type(model).__name__}")

        assert len(self.layers) == 12, f"Expected 12 layers, got {len(self.layers)}"

    def _set_layer_requires_grad(self, layer_idx, requires_grad):
        for p in self.layers[layer_idx].parameters():
            p.requires_grad = requires_grad

    def _set_embeddings_requires_grad(self, requires_grad):
        for p in self.embeddings.parameters():
            p.requires_grad = requires_grad

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_count += 1                  # T5: increment internal counter
        epoch_num = self._epoch_count            # 1-indexed: 1, 2, 3, ...

        if epoch_num == 1:
            # Freeze all layers + embeddings; unfreeze last 2 layers
            self._set_embeddings_requires_grad(False)
            for i in range(12):
                self._set_layer_requires_grad(i, False)
            for i in [10, 11]:
                self._set_layer_requires_grad(i, True)
        elif epoch_num == 2:
            # Freeze embeddings; unfreeze last 4 layers
            self._set_embeddings_requires_grad(False)
            for i in range(12):
                self._set_layer_requires_grad(i, False)
            for i in [8, 9, 10, 11]:
                self._set_layer_requires_grad(i, True)
        else:  # epoch_num >= 3
            # T3 [v1.2]: Unfreeze EVERYTHING including embeddings
            # (corrects v1.1 permanent embedding freeze; matches paper's "all parameters" intent)
            self._set_embeddings_requires_grad(True)
            for i in range(12):
                self._set_layer_requires_grad(i, True)

        # LM head is never explicitly frozen (default = trainable)
```

### 4.3 Top-1 Accuracy Logger (Unchanged from v1.1)

`TopKAccuracyLogger` callback evaluates top-1 masked-token accuracy on training set after each epoch. Result used by Component 6's threshold grid.

### 4.4 Training Loop and Failure Handling (Unchanged from v1.1)

```python
trainer.train()
final_top1_acc = trainer.state.log_history[-1]["train_top1_acc"]
if final_top1_acc < 0.5:
    raise RuntimeError(f"Training failed: top1_acc={final_top1_acc:.3f} < 0.5 [D10]")
trainer.save_model(f"runs/{dataset}_{window}/fold_{fold_idx}/final_model")

# T7 [v1.2]: Persist model weight SHA256 for cross-run determinism audit (D8 fp16 mitigation)
weight_hash = compute_state_dict_sha256(model.state_dict())
log_to_json("train_log.json", "model_weight_sha256", weight_hash)
```

**Resume protocol:** HuggingFace `Trainer` natively supports `--resume_from_checkpoint`. Full optimizer + scheduler state restored. Note: OneCycleLR state is part of the saved scheduler state, so resume preserves the current LR position in the cycle.

---

## Component 5: Anomaly Scoring (REVISED v1.2 — lex-sort comment)

**Algorithm unchanged from v1.1.** Per-paragraph hash seeding removed; global RNG used; paragraphs processed in `sorted(eval_paragraphs, key=lambda p: p.paragraph_id)` order.

**NEW v1.2 — sort semantics clarification (reviewer F8):**

The sort key is **lexicographic on the `paragraph_id` string**, not numeric on the window index. Concretely, for BGL 30s windows, paragraph IDs sort as `"bgl_30s_w0", "bgl_30s_w1", "bgl_30s_w10", "bgl_30s_w11", ..., "bgl_30s_w2", "bgl_30s_w20", ...`. Window 10 sorts before window 2.

**This is deterministic and intentional.** The reproducibility contract holds — same code, same data, same model → same RNG progression → same scores. Downstream analyses should **not** assume window-order iteration. If chronological analysis is needed later, sort by `start_timestamp` field explicitly.

---

## Component 6: Threshold Tuning (Unchanged from v1.1)

Grid search over `topk ∈ {5, 9, 12}` × `threshold = linspace(top1_acc - 0.1, top1_acc, 3)`. Pick best F1 on tune set. Tie-break: smaller K, then larger θ.

---

## Component 7: Evaluation (Unchanged from v1.1)

P/R/F1/Specificity per fold. Aggregation across folds: mean ± sample std (ddof=1). Per-fold `(K, θ)` persisted. For BGL/TB, additional aggregation across the three window sizes.

**Disclosure standard (preserved from v1.1):** Per-fold means may average across operating points that differ per fold. `per_fold_operating_points` persisted in `dataset_results.json`. Reported sample std is a lower bound on true between-fold variance (per decisions-v1.2 §2.3).

---

## Component 8: Variability Test — BGL only (REVISED v1.2 — T2 + T4 + T6)

**Inputs:** Trained BGL model `f_θ`, original BGL test set, best `(K, θ)`.
**Outputs:** `variability_metrics.json`, `wordnet_substitutions.json`.

### 8.1 Helper: `strip_trailing_punct` (NEW v1.2 — T2)

```python
import re

_TRAILING_PUNCT_RE = re.compile(r'^(.*?)([.,;:!?)\]\}\'"]*)$')

def strip_trailing_punct(tok: str) -> tuple[str, str]:
    """Split a whitespace-delimited token into (content, trailing_punct).

    Leading punctuation (opening quotes, brackets, parens) is treated as part of
    the content and will block matching — by design: '"receive' should not match
    the verb "receive" because the leading quote is meaningful in log content.

    Examples:
        strip_trailing_punct("receive.")    -> ("receive", ".")
        strip_trailing_punct("receive")     -> ("receive", "")
        strip_trailing_punct("DataNode.")   -> ("DataNode", ".")
        strip_trailing_punct('"receive')    -> ('"receive', "")
        strip_trailing_punct("done!")        -> ("done", "!")
        strip_trailing_punct("(start)")     -> ("(start", ")")
    """
    m = _TRAILING_PUNCT_RE.match(tok)
    return (m.group(1), m.group(2)) if m else (tok, "")
```

### 8.2 Build Replacement Map (REVISED v1.2 — T6 multi-word filter)

```python
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from collections import Counter

def build_replacement_map(test_set: list, top_n: int = 10) -> dict[str, str]:
    # Collect verb tokens across all test paragraphs
    corpus = ' '.join(line for P in test_set for line in P.lines)
    tagged = pos_tag(word_tokenize(corpus.lower()))
    verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    verbs = [tok for tok, tag in tagged if tag in verb_tags]

    freq = Counter(verbs)

    # Filter to verbs with at least one verb-POS synset and rank
    candidates = []
    for verb, count in freq.most_common():
        if wordnet.synsets(verb, pos='v'):
            candidates.append(verb)
        if len(candidates) >= top_n:
            break

    replacement_map = {}
    for V in candidates:
        replacement = None
        for synset in wordnet.synsets(V, pos='v'):
            for lemma in synset.lemmas():
                name = lemma.name()
                # NEW v1.2 [T6]: skip multi-word lemmas to preserve 1-to-1 substitution
                if '_' in name:
                    continue
                if name.lower() != V.lower():
                    replacement = name
                    break
            if replacement is not None:
                break
        if replacement is not None:
            replacement_map[V] = replacement
        # If no single-word non-self synonym found, skip this verb

    return replacement_map
```

### 8.3 Apply Variability (REVISED v1.2 — T4 whitespace preservation)

```python
def apply_variability(test_set: list, replacement_map: dict[str, str]) -> list:
    """Apply WordNet substitution to test set lines, preserving original whitespace.

    T4 [v1.2]: Uses re.split with capture group to preserve whitespace tokens
    (tabs, multi-spaces) as separate list elements. The previous v1.1 approach
    (line.split() + ' '.join()) collapsed all whitespace to single spaces, which
    is destructive for tab-separated BGL/TB log fields.
    """
    modified_test_set = []
    for P in test_set:
        modified_lines = []
        for line in P.lines:
            # Split preserving whitespace as separate tokens
            parts = re.split(r'(\s+)', line)   # captures whitespace; non-whitespace at even indices
            new_parts = []
            for part in parts:
                if not part or part.isspace():
                    # Whitespace token — preserve verbatim
                    new_parts.append(part)
                    continue
                # Non-whitespace token — attempt substitution
                stripped, trailing = strip_trailing_punct(part)
                if stripped.lower() in replacement_map:
                    replacement = replacement_map[stripped.lower()]
                    # Case-preserving replacement of EXACT stripped form
                    if stripped.istitle():
                        replacement = replacement.title()
                    elif stripped.isupper():
                        replacement = replacement.upper()
                    # Mixed-case (camelCase / iSCSI / DataNode) does not match — falls through to else
                    new_parts.append(replacement + trailing)
                else:
                    new_parts.append(part)
            modified_lines.append(''.join(new_parts))
        modified_P = Paragraph(
            paragraph_id=P.paragraph_id,
            lines=modified_lines,
            label=P.label,
            source_window_id=P.source_window_id,
            start_timestamp=P.start_timestamp,
        )
        modified_test_set.append(modified_P)
    return modified_test_set
```

### 8.4 Run Variability Evaluation (Unchanged)

```python
replacement_map = build_replacement_map(test_set, top_n=10)
modified_test_set = apply_variability(test_set, replacement_map)
variability_metrics = evaluate_fold(model, modified_test_set, K_best, theta_best, rng=...)

# Persist audit trail
save_json("wordnet_substitutions.json", {
    "replacement_map": replacement_map,
    "per_verb": [
        {"verb": v, "replacement": r, "is_single_word": '_' not in r}  # always True per T6 filter
        for v, r in replacement_map.items()
    ],
})
```

**Critical invariants (enforced by tests):**

- Training paragraphs are never modified.
- Substring matches inside camelCase / snake_case / file paths are not replaced.
- All replacement values are single-word (multi-word lemmas filtered per T6).
- Whitespace structure of original lines preserved (tab-separated fields stay tab-separated).

---

## Component 9: Throughput Measurement (Unchanged from v1.1)

Samples/sec on Narval A100 40GB, batch=1, hardware caveat in disclosure.

---

## Component 10: Run Configuration Schema (REVISED v1.2 — OneCycleLR config)

```yaml
# Example: configs/bgl_30s.yaml
dataset: bgl
window_seconds: 30
seed: 42

sample_budget:
  normal: 25000
  anomaly: 2000

cv:
  folds: 5
  per_fold_train_normal: 5000
  per_fold_tune_normal: 1000
  per_fold_tune_anomaly: 1000
  per_fold_test_normal: 5000
  per_fold_test_anomaly: 1000
  anomaly_partition_strategy: 'per_fold_shuffle' # [D14]

backbone:
  selection_quantile: 0.8
  threshold_words: 512
  roberta_id: 'roberta-base'
  longformer_id: 'allenai/longformer-base-4096'

token_length_gate:
  flag_threshold_p95_over_limit: true
  hard_fail_max_over_5x_limit: true
  longformer_attention_window_check: true # NEW v1.2 — smoke-test forward pass

masking:
  sentence_ratio: 0.5
  token_ratio: 0.8

training:
  framework: 'huggingface_trainer' # [D15]
  transformers_version: '4.45.0' # NEW v1.2 [T5] — pinned
  epochs: 5 # 3 for HDFS, 5 for BGL/TB [D1]
  max_lr: null # populated by D2 grid search
  lr_grid: [1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4]
  effective_batch_size: 8 # 8 for RoBERTa, 2 for Longformer [D3]
  optimizer: adamw
  betas: [0.9, 0.99]
  eps: 1.0e-5
  weight_decay: 0.01

  # REVISED v1.2 [T1] — OneCycleLR replaces v1.1 cosine
  lr_scheduler:
    type: 'torch_one_cycle_lr'
    pct_start: 0.3
    anneal_strategy: 'cos'
    div_factor: 25.0
    final_div_factor: 1.0e4
    three_phase: false

  gradual_unfreezing_schedule: # [D15]
    epoch_1_unfrozen_layers: [10, 11]
    epoch_2_unfrozen_layers: [8, 9, 10, 11]
    epoch_3_plus_unfrozen_layers: 'all'
    embeddings_frozen_until_epoch: 2 # REVISED v1.2 [T3] — unfrozen at epoch 3+

  fp16: true # accepted residual non-determinism [D8]

determinism: # [D8 strengthened v1.2]
  python_hash_seed: 0
  torch_deterministic_algorithms: true
  cublas_workspace_config: ':4096:8'
  hf_full_determinism: true # NEW v1.2 [T7]
  fp16_weight_divergence_threshold: 1.0e-4 # NEW v1.2 [Q7] — escalate if exceeded

inference:
  topk_grid: [5, 9, 12]
  threshold_grid_size: 3
  global_seed: 42

variability:
  enabled: false # true for BGL only
  top_n_verbs: 10 # [D13]
  lemma_strategy: 'first_synset_first_single_word_lemma' # REVISED v1.2 [T6, D17]
  substitution_scope: 'exact_token_match' # [D18]
  preserve_whitespace: true # NEW v1.2 [T4]
```

---

## Component 11: Output Artifacts (REVISED v1.2 — weight SHA256 added)

```
runs/{dataset}_{window}/fold_{idx}/
├── config_resolved.yaml
├── train_log.json                   # NEW v1.2 fields: model_weight_sha256, full_determinism_warnings
├── final_model/
├── tuning_results.json
├── fold_metrics.json
├── variability_metrics.json         # BGL only
└── wordnet_substitutions.json       # BGL only (now includes is_single_word audit field)
```

---

## Component 12: API Surface (REVISED v1.2 — new helpers)

In addition to v1.1 API:

```python
# src/variability_bgl.py — new helper
def strip_trailing_punct(tok: str) -> tuple[str, str]:
    ...

# src/utils/io.py — new helper for T7
def compute_state_dict_sha256(state_dict: dict) -> str:
    """SHA256 hash of model weights, layer-ordered for determinism."""
    ...
```

---

## Component 13: Test Suite (REVISED v1.2 — 4 new tests)

All v1.1 tests preserved. New tests:

| Test                                               | Asserts                                                                                                                                                                                                                                         | Module                 |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `test_strip_trailing_punct` (NEW v1.2)             | `"receive." → ("receive", ".")`, `"DataNode." → ("DataNode", ".")`, `'"receive' → ('"receive', "")`, `"(start)" → ("(start", ")")`                                                                                                              | `variability_bgl.py`   |
| `test_variability_no_multiword_lemmas` (NEW v1.2)  | All values in `replacement_map` contain no `_` and no whitespace                                                                                                                                                                                | `variability_bgl.py`   |
| `test_unfreezing_schedule_transitions` (NEW v1.2)  | After epoch 1: only layers {10, 11} have `requires_grad=True` among encoder layers; embeddings frozen. After epoch 2: layers {8, 9, 10, 11} unfrozen, embeddings still frozen. After epoch 3: all layers + embeddings have `requires_grad=True` | `unfreeze_callback.py` |
| `test_variability_whitespace_preserved` (NEW v1.2) | Line with tab-separated fields `"INFO\tdfs.DataNode\treceive blk_X"` after substitution still has tab separators (not collapsed to single space)                                                                                                | `variability_bgl.py`   |

Test integrity check: every BLOCKING/IMPORTANT finding from the second-pass review maps to at least one regression test.

---

## Component 14: Reproducibility Checklist (REVISED v1.2 — SHA256 verification)

- [ ] `seed=42` set in Python `random`, NumPy, PyTorch (CPU + CUDA), HuggingFace
- [ ] `TrainingArguments(full_determinism=True, ...)` set in `src/train.py`
- [ ] `torch.use_deterministic_algorithms(True)` enabled in `src/utils/seed.py` (defense-in-depth alongside `full_determinism=True`)
- [ ] `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported in all SLURM scripts
- [ ] `PYTHONHASHSEED=0` exported in all SLURM scripts
- [ ] `transformers==4.45.0` pinned in `pyproject.toml` (T5)
- [ ] No per-paragraph hash-based seeding anywhere (v1.0 deviation removed)
- [ ] All `paragraphs.pkl`, `allocated_indices.json`, `fold_*.json` checksummed (SHA256)
- [ ] **NEW v1.2 — Per-fold model-weight SHA256:** After training each fold, compute SHA256 of `model.state_dict()` and log to `train_log.json`. Run same fold twice with identical config; verify SHA256 match. If mismatch detected, compute per-layer L∞ weight divergence and log; escalate to supervisor if any layer exceeds `1.0e-4` (Q7 threshold).
- [ ] **NEW v1.2 — Longformer smoke test:** For Longformer-selected datasets, before launching full training, run a single forward pass on the longest training paragraph at the chosen batch size; verify no OOM and no shape mismatch from attention-window padding.
- [ ] Backbone choice + max_lr persisted to `config_resolved.yaml`
- [ ] Drop counters in `preparation_summary.json` non-zero values explained
- [ ] `lines_with_multiple_blockids` rate logged for HDFS
- [ ] `token_length_distribution` in `preparation_summary.json` populated; gate condition checked
- [ ] Per-fold metrics persisted, not just means
- [ ] Per-fold operating points `(K, θ)` persisted
- [ ] `wordnet_substitutions.json` shows actual replacements applied; `is_single_word` audit field always True
- [ ] HuggingFace `transformers==4.45.0` / PyTorch / CUDA / Python versions pinned
- [ ] Final `dataset_results.json` includes git commit hash of code version
- [ ] `full_determinism=True` warnings about non-deterministic kernels logged but run allowed to proceed

---

_LogFiT Reproduction Technique & Methodology Specification v1.2_
_Post-second-pass adversarial review of v1.1; companion to logfit-repro-decisions-v1.2.md_
_Oussama — Université Laval Cybersecurity Research Lab, May 2026_
