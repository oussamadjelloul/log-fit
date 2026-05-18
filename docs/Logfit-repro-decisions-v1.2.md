# LogFiT Reproduction — Methodology Decisions

**Version:** 1.2 (post-second-pass-adversarial-review)
**Status:** Draft for supervisor sign-off (gate artifact — no code written before sign-off)
**Author:** Oussama
**Source paper:** Almodovar, C., Sabrina, F., Karimi, S., Azad, S. (2024). _LogFiT: Log Anomaly Detection Using Fine-Tuned Language Models._ IEEE Transactions on Network and Service Management, 21(2), 1715–1723. DOI: [10.1109/TNSM.2024.3358730](https://doi.org/10.1109/TNSM.2024.3358730)
**Reproduction repo:** `~/logfit-repro/` (separate from `~/sdd-v5/`)
**Reference implementation availability:** None located. Paper §IV-A states _"The source code implementing the LogFiT model, datasets and model checkpoints will be made available online"_ — no public GitHub repo found as of search date.

## Changelog v1.1 → v1.2

Driven by second-pass tri-LLM adversarial review (both reviewers APPROVE-WITH-CONDITIONS, 2 convergent BLOCKING + 4 convergent IMPORTANT findings). v1.1 fixes for v1.0 BLOCKING/IMPORTANT items were judged real, but v1.1 itself introduced new bugs in Component 4 (HF Trainer migration) and Component 8 (WordNet substitution).

**Convergent BLOCKING findings (both reviewers):**

- **T1 — LR scheduler.** v1.1 D15 prose mentioned `get_linear_schedule_with_warmup`, code used `cosine`, neither implements true 1-cycle, and `transformers` has no `get_one_cycle_schedule_with_warmup`. **v1.2 resolution:** Adopt `torch.optim.lr_scheduler.OneCycleLR` (PyTorch canonical 1-cycle) integrated via `Trainer(optimizers=(optimizer, scheduler))`. See D15.
- **T2 — `strip_trailing_punct` undefined.** Called in Component 8 step 9 but never defined. **v1.2 resolution:** Explicit regex-based definition adopted from reviewer 2's recommended code. See spec Component 8.

**Convergent IMPORTANT findings:**

- **T3 — Embeddings permanently frozen.** v1.1 callback never unfroze embeddings; paper's "all parameters unfrozen" intent violated. **v1.2 resolution:** Unfreeze embeddings in `epoch >= 2` branch. Three-confound D15 disclosure now reduces to one (stage compression only).
- **T4 — Destructive whitespace normalization** (R1 only). `line.split() + ' '.join(...)` collapsed tab-separated fields. **v1.2 resolution:** `re.split(r'(\s+)', line)` preserving whitespace tokens.
- **T5 — `state.epoch` version-sensitivity** (R2 only). `TrainerState.epoch` semantics differ across `transformers` versions. **v1.2 resolution:** Switch to callback-internal `self._epoch_count` counter; pin `transformers==4.45.0`.
- **T6 — Multi-word WordNet lemmas** (R2 only). `wordnet.synsets("open", "v")[0].lemmas()` can return `"open_up"`, breaking 1-to-1 token substitution. **v1.2 resolution:** Skip multi-word lemmas in D17 algorithm.
- **T7 — `full_determinism=True` + `fp16` non-determinism** (R2 only). v1.1 set manual flags but missed HF's canonical `full_determinism=True`. **v1.2 resolution:** Strengthen D8; add SHA256 weight-hash verification to reproducibility checklist.
- **T8 — 50% anomaly overlap under-disclosed.** v1.1 D14 noted cross-fold overlap qualitatively; v1.2 quantifies (E[overlap] = 500/1000 = 50%) and adds §2.3 Protocol Risk Disclosure.

**MINOR polish (R2):** Lex-sort on `paragraph_id` annotated; §1.6 token-length thresholds get explicit rationale; Longformer attention-window padding flagged for empirical verification.

---

## 1. Reproduction Scope (Unchanged from v1.0)

**Faithful reproduction only.** Random 25k normal + 2k anomaly per dataset, 5-fold cross-validation, exactly as paper §IV-A specifies. No chronological-split variant in this reproduction.

---

## 2. Protocol Risk Disclosure (REVISED v1.2 — §2.3 added)

### 2.1 Random-sample 5-fold CV on temporally ordered data (paper-inherited)

The reproduced protocol carries a known data-leakage risk acknowledged in the paper itself (§IV-A Observations paragraph):

> _"We note that k-fold cross-validation may lead the models to peek into the future, by processing log samples that only occur at later times leading to the model's effectiveness improvement. This has been shown in the past in time-based datasets [50]."_

Same protocol family as LLM-LADE (Zhang KBS 2025). Disclosure-explicit in LogFiT. The dissertation must state explicitly that SDD's chronological-split numbers and reproduced LogFiT's random-sample numbers are not under the same protocol.

### 2.2 Paper-internal scoring inconsistency (LogBERT adoption claim)

Paper §III-C states LogFiT _"uses a technique adopted from LogBERT"_ but operationalizes the score as top-k accuracy (fraction of correctly predicted masked tokens), whereas LogBERT (Guo et al., IJCNN 2021) §III.B uses a misprediction count threshold. Reproduction follows the paper's operationalization (accuracy-based), not its claimed source.

### 2.3 Cross-fold anomaly leakage from per-fold pool sampling (NEW v1.2)

Under D14's per-fold shuffle of the 2,000-anomaly pool:

- Each anomaly has independent probability **0.5** of landing in any given fold's test set (1,000 of 2,000 drawn per fold).
- Across two distinct folds k ≠ k′, the events are independent: **P(anomaly X in both test_k and test_k′) = 0.25**.
- Expected size of `test_k ∩ test_k′` = **2,000 × 0.25 = 500 paragraphs** — i.e., **50% of each fold's 1,000 test-anomaly sample overlaps with any other fold's test-anomaly sample**.

**Consequence:** Cross-fold test metrics are not statistically independent. A model that overfits to specific anomaly paragraphs during fold k's tuning may receive partially-inflated metrics in fold k′ when those paragraphs reappear in fold k′'s test set. The cross-fold sample std of reproduced F1 is therefore an **under-estimate of true between-fold variance**.

**Dissertation reporting requirement:** When reporting reproduced LogFiT 5-fold means in the SDD chapter, explicitly state that fold-level metric independence is partially compromised by anomaly reuse forced by the paper's 2k pool budget (D14). Do not present the sample std (ddof=1) as a confidence interval — it is a lower bound.

---

## 3. Locked Specifications (Unchanged from v1.1)

All L1–L27 items from v1.1 carry forward unchanged.

---

## 4. Decisions Made Where Paper Is Silent (REVISED v1.2)

### D1. Number of training epochs (Unchanged)

3 epochs for HDFS, 5 epochs for BGL/Thunderbird.

### D2. Maximum learning rate (Unchanged from v1.1)

Per-dataset grid search over `{1e-5, 5e-5, 1e-4, 5e-4}` on a 500-paragraph subset, 1 epoch each, pick lowest masked-token cross-entropy.

### D3. Batch size (Unchanged)

8 for RoBERTa, 2 for Longformer on A100 40GB, gradient accumulation as needed.

### D4. Raw log line content (Unchanged from v1.1)

Default "as published" with §1.6 token-length empirical guard.

### D5. BGL/Thunderbird window labeling (Unchanged)

Window is anomaly if any constituent log line has label_field != "-".

### D6. Sentence boundary definition (Unchanged — confirmed by reviewer audits)

One log line = one sentence.

### D7. Truncation strategy (Unchanged)

Right-truncate at backbone's max sequence length.

### D8. Random seed and CUDA determinism (STRENGTHENED v1.2 — full_determinism + fp16 disclosure)

- **Paper says:** Unspecified.
- **Decision (v1.2):** Global `seed=42` across Python `random`, NumPy, PyTorch (CPU + CUDA), HuggingFace. Additionally:
  - `transformers.TrainingArguments(full_determinism=True, ...)` — HF canonical determinism flag (sets internal seeds, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`).
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported in all SLURM scripts.
  - `PYTHONHASHSEED=0` exported in all SLURM scripts.
  - Operations emitting "no deterministic implementation available" warnings are logged; run proceeds with caveat.
- **`fp16=True` disclosure (NEW v1.2):** Mixed-precision (fp16) can produce per-kernel non-determinism on some operations even with `full_determinism=True`. Decision: keep `fp16=True` for A100 throughput (~2× speedup); accept residual non-determinism on a small number of kernels. Mitigation: run the same fold twice on the same GPU, compute SHA256 of final-model weights, document magnitude of any divergence in `train_log.json`. If weight divergence exceeds 1e-4 in any layer, escalate.
- **Persistence:** Seed value, `full_determinism` flag, CUDA config, and per-run model-weight SHA256 logged in `train_log.json`.
- **Rationale:** Reviewer F8 (R2) caught that v1.1's manual flag-setting was insufficient without `full_determinism=True`. Single-seed reproduction acceptable given 5-fold CV; multi-seed sweep deferred.

### D9. Whether to mask `<s>` / `</s>` / pad tokens (Unchanged)

Never mask special tokens.

### D10. Threshold lower-bound clipping (Unchanged)

Clip to `max(0.0, top1_acc - 0.1)`. If `train_top1_acc < 0.5`, run failed.

### D11. UMAP for ablation Figure 4 (Unchanged — out of scope)

### D12. Throughput measurement (Unchanged — secondary, hardware caveat)

### D13. WordNet verb count: top-10 (Unchanged from v1.1)

Top-10 verbs per paper §V, not "top 10%" per paper §IV-A (paper-internal contradiction; §V matches Table VII numerics).

### D14. Anomaly pool partitioning across folds (REVISED v1.2 — quantification added, overstatement removed)

- **Paper says:** Per fold, 1,000 anomalies for tuning and 1,000 anomalies for test, out of a 2,000-anomaly budget. Partitioning convention unspecified.
- **Decision:** The 2,000-anomaly pool is **not k-fold-partitioned**. For each fold `f ∈ {0..4}`:
  ```
  rng = numpy.random.RandomState(seed + f)        # seed = 42 globally
  shuffled = rng.permutation(anomaly_pool_ids)    # length 2000
  tune_anomaly = shuffled[:1000]
  test_anomaly = shuffled[1000:2000]
  ```
- **Acknowledged consequence (REVISED v1.2 — quantified):** Expected cross-fold test-anomaly overlap is **500 paragraphs (50%)** between any two folds (derivation in §2.3). This means cross-fold metrics are not statistically independent; the reported sample std is a lower bound on true between-fold variance.
- **Rationale:** v1.0 attempted k-fold partition of anomalies (arithmetically impossible). Alternatives considered:
  - **(A) Static 50/50 split** (same 1k tune + 1k test for all folds): rejected — test anomalies identical across folds, weakens cross-fold independence further than (B).
  - **(B) Per-fold random shuffle** (chosen): preserves **per-fold cardinality contract** of the paper (1k+1k); cross-fold anomaly independence is sacrificed by necessity given the 2k pool budget. _(v1.2 note: v1.1 rationale incorrectly claimed this "preserves CV's intent of independent samples per fold" — that overstatement is retracted.)_
- **Disclosure standard:** Per §2.3, the dissertation must pre-register that cross-fold test independence is partially compromised by anomaly reuse.

### D15. Training stack: HuggingFace `Trainer` + OneCycleLR (REVISED v1.2 — scheduler resolved)

- **Paper says:** Fine-tuning uses "super-convergence techniques [41] implemented in the FastAI/ULMFiT framework [42], [43]" (§III-A). Reference [41] = Smith & Topin 2019.
- **Decision (v1.2):** HuggingFace `transformers.Trainer` with:
  - **Optimizer:** AdamW with `betas=(0.9, 0.99)`, `eps=1e-5`, `weight_decay=0.01` [L10].
  - **LR schedule (REVISED v1.2):** `torch.optim.lr_scheduler.OneCycleLR` — canonical PyTorch implementation of Smith & Topin 2019 1-cycle (paper reference [41]). Configuration:
    - `max_lr = D2 selected value`
    - `total_steps = num_epochs × steps_per_epoch` (computed once at Trainer init)
    - `pct_start = 0.3` (PyTorch default; Smith & Topin recommend early-peak)
    - `anneal_strategy = 'cos'` (PyTorch default)
    - `div_factor = 25` (initial_lr = max_lr / 25)
    - `final_div_factor = 1e4` (final_lr = max_lr / 25 / 1e4)
    - `three_phase = False` (two-phase: ramp-up, decay)
  - **Integration:** Scheduler instantiated outside `Trainer`, passed via `Trainer(optimizers=(optimizer, scheduler), ...)`. `Trainer` then calls `scheduler.step()` per training batch (canonical pattern; no override required).
  - **Gradual unfreezing (REVISED v1.2 — internal counter + embedding unfreeze):** Manual via per-epoch `param.requires_grad` updates in a custom `TrainerCallback`:
    - Epoch 1: Unfreeze LM head + last 2 encoder layers (10, 11). Embeddings + layers 0–9 frozen.
    - Epoch 2: Additionally unfreeze layers 8, 9 (last 4 unfrozen). Embeddings still frozen.
    - Epoch 3+: **All parameters unfrozen, including embeddings** (matches paper's "all parameters" intent; corrects v1.1 over-freezing per reviewer F4/F3).
    - Schedule indexed by a **callback-internal counter** `self._epoch_count`, not `TrainerState.epoch` (per reviewer F2 / T5 — eliminates version-sensitivity to `transformers` internals).
- **Acknowledged confound (v1.2 reduced from three to one):** The 3-stage compressed unfreezing schedule deviates from canonical ULMFiT (Howard & Ruder 2018), which unfreezes one layer per epoch over ~12 epochs. v1.2 fits the schedule into the paper's 3-epoch HDFS / 5-epoch BGL+TB budget. If the paper used a different schedule (untestable from text), reproduced F1 may diverge. **Scheduler-shape and embedding-handling confounds from v1.1 are now resolved** (OneCycleLR matches paper's cited Smith & Topin; embeddings unfreeze in final stage matches "all parameters").
- **Version pinning (NEW v1.2):** `transformers==4.45.0` pinned in `pyproject.toml` per reviewer F2 (R2). Pinned to prevent silent change in `Trainer` callback semantics.
- **Sanity-check obligation:** Compare per-epoch training loss to canonical 1-cycle's super-convergence shape (rapid early decrease, plateau in late phase). Significant deviation → revisit.

### D16. HDFS lines mentioning multiple block_ids (Unchanged from v1.1)

Extract all `blk_-?\d+` matches per line; attribute line to each referenced block (duplicate across paragraphs). Surface counter `lines_with_multiple_blockids`.

### D17. WordNet "lemma" operationalization (REVISED v1.2 — multi-word filter added)

- **Decision (v1.2):** For each top-10 verb V:
  ```python
  for synset in wordnet.synsets(V, pos='v'):
      for lemma in synset.lemmas():
          name = lemma.name()
          if '_' in name:
              continue                       # NEW v1.2: skip multi-word lemmas (e.g., "open_up")
          if name.lower() != V.lower():
              replacement = name             # single-word synonym
              break
      if replacement is not None:
          break
  if replacement is None:
      continue                                # rare: no single-word non-self synonym found; skip verb
  ```
- **Rationale for multi-word filter (NEW v1.2):** Multi-word lemmas (e.g., `open_up`, `carry_out`, `make_up`) would produce 2-token replacements substituted into 1-token slots, changing paragraph token counts in the eval set and shifting `sent_boundaries` relative to training paragraphs. This violates the variability test's premise of lexical-only substitution. Reviewer F5 (R2).
- **Acknowledged residual confound:** Reproduced variability F1 (paper Table VII: 89.38) may not be exactly reproducible — original authors' lemma-selection convention is unknown. Tolerance Q3 applies.

### D18. Replacement substitution scope — exact-token-match only (Unchanged from v1.1)

Replacement applies only to whitespace-delimited tokens whose lowercased form exactly equals the source verb. Substring matches inside camelCase / snake_case / file paths / class names are excluded.

---

## 5. Open Questions for Supervisor (REVISED v1.2)

| #             | Question                                                               | Default if no input                   |
| ------------- | ---------------------------------------------------------------------- | ------------------------------------- |
| Q1            | Confirm scope: HDFS + BGL + TB, all three windows?                     | All three datasets, all three windows |
| Q2            | Throughput: report or skip?                                            | Report with hardware caveat           |
| Q3            | F1 tolerance vs paper Tables III/IV/V?                                 | ±0.02 F1 absolute                     |
| Q4            | Anomaly pool partitioning (D14): per-fold shuffle confirmed?           | Per-fold shuffle (D14)                |
| Q5            | Training stack: HF Trainer + OneCycleLR (D15) confirmed?               | HF Trainer + OneCycleLR               |
| Q6            | If HDFS F1 misses paper by >Q3: run timestamp-strip sensitivity?       | Yes                                   |
| Q7 (NEW v1.2) | Acceptable fp16 weight-divergence threshold across two same-seed runs? | 1e-4 absolute on any layer            |

---

## 6. Validation Criteria (Unchanged stub — paper numbers transcription pending)

> **Action item before supervisor sign-off:** Oussama transcribes paper Tables III/IV/V/VII numeric values into §6 so the tolerance contract (Q3) becomes concrete.

Failure-mode checklist preserved from v1.1:

- F1 > 0.99 → suspect data leakage in 5-fold split
- F1 < 0.50 → suspect training failure, threshold miscalibration, or label inversion
- BGL variability F1 collapse → suspect tokenizer mismatch or WordNet replacement leaking into training

---

## 7. Filesystem Layout (Unchanged from v1.1)

See v1.1 §7. `~/logfit-repro/` with the same `src/`, `scripts/`, `tests/`, `configs/`, `docs/` structure. `pyproject.toml` now pins `transformers==4.45.0` (v1.2 change).

---

## 8. Execution Sequence (Unchanged from v1.1)

14 steps from decisions sign-off → preprocessing → token-length gate → backbone selection → fold construction → single-fold smoke test → full runs → validation → variability test → throughput → writeup.

---

## 9. Disclosure Standard (Unchanged from v1.0)

- All decisions in §4 flagged as **decisions by the reproducer**, not paper specifications.
- Per-fold results persisted, not just means.
- All hyperparameter sweeps logged with full grid + selected value.
- Training failures (D10) reported, not silently re-run with different seeds.
- Tolerance for reproduction success (Q3) declared before runs.
- Cross-fold anomaly leakage (§2.3) disclosed in dissertation, not hidden.

---

_LogFiT Reproduction Methodology Decisions v1.2_
_Post-second-pass adversarial review of v1.1 (2 convergent BLOCKING + 4 convergent IMPORTANT findings resolved)_
_Companion: logfit-repro-spec-v1.2.md_
_Oussama — Université Laval Cybersecurity Research Lab, May 2026_
