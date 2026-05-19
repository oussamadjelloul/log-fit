# LogFiT Reproduction — v1.5 Decisions

**Date:** 2026-05-19
**Status:** Locked + implementation complete for splits/score/subsample. Reflects
the Paradigm B choice (not the Path 3 plan in the prior v1.5 draft) and the
D_NEW6.1 degradation amendment surfaced by the HDFS-subset retrain.
**Supersedes:** v1.4 decisions where conflicts arise (explicitly flagged below).
**Author:** Oussama Djoussa, Université Laval Cybersecurity Lab

---

## 1. Context — what triggered v1.5

`src/score.py` was implemented against a tune split that `src/splits.py` did not
produce. Investigation surfaced a deeper methodology gap: the LogFiT paper §IV-A
(rows L21–L25) specifies a per-fold **tune carve** of 1,000 normal + 1,000
anomaly samples, drawn from the train pool and disjoint from the training
sample. v1.4 splits.py implemented a 4-list schema (`train_normal_ids`,
`train_anomaly_ids`, `test_normal_ids`, `test_anomaly_ids`) — no tune set.
Consequence: `score.py --split tune` would have raised on every real splits
artifact, and `tune_threshold.py` would have had no tune data to operate on.

Closing this gap required reopening a second methodology question: the v1.4
code partitioned BOTH normals and anomalies across folds (paradigm A), but
paper §IV-A L25 specifies a fixed-budget test_anomaly per fold (1,000) which
can only be sampled from a shared 2k anomaly pool with cross-fold overlap
(paradigm B). v1.5 locks paradigm B.

A subsequent triage uncovered that the initial v1.5 degradation policy
(tune-priority) starved the HDFS-subset test set of anomalies. D_NEW6.1
(proportional degradation) fixes this.

Five locked decisions follow, plus one new pipeline component.

---

## 2. Locked decisions

### D_NEW6 — Paradigm B splits: per-fold tune carve, anomaly pooling

**Supersedes:** v1.4 D14 (which partitioned both normals and anomalies); also
supersedes the Path-3 draft of D_NEW6 in the prior v1.5 plan, which kept the
v1.4 partition of anomalies and added tune carve on top.

**Source:** LogFiT paper §IV-A K-fold Cross-validation (rows L21–L25):
- L21 allocation: 25,000 normal + 2,000 anomaly paragraphs total per dataset
- L23 per-fold train: 5,000 normal (from 20k train split)
- L24 per-fold tune: **1,000 normal (from train split) + 1,000 anomaly**
- L25 per-fold test: 5,000 normal (from test split) + 1,000 anomaly

The paper's `test_anomaly = 1,000 per fold` over 5 folds (= 5,000 anomaly test
samples in total) against a 2,000-anomaly pool implies cross-fold sampling
with overlap, not a partition. v1.5 implements that explicitly.

**Schema (`FoldSplit` v1.5, 5 ID lists):**
```python
@dataclass
class FoldSplit:
    fold_id: int
    seed: int
    train_normal_ids: list[str]    # 5,000 per L23
    tune_normal_ids: list[str]     # 1,000 per L24 (NEW v1.5)
    tune_anomaly_ids: list[str]    # 1,000 per L24 (NEW v1.5)
    test_normal_ids: list[str]     # fold's normal partition (5,000 at L21 scale)
    test_anomaly_ids: list[str]    # 1,000 per L25 (was: fold's anomaly partition ~400 in v1.4)
```

**REMOVED from v1.4:** `train_anomaly_ids`. LogFiT trains on normals only; the
field was unused by `train.py` and misleading. Tests in `test_splits.py` assert
the field's absence as a regression guard.

**Carve algorithm:**

1. **Partition normals only** across N folds via stratified shuffle + round-robin
   (`stratified_normal_fold_assignment`).
2. **Anomalies stay as a single pool** (no partitioning).
3. For each fold k with `fold_seed = base_seed + k`:
   - `test_normal_ids := fold_k's normal partition` (deterministic, full set).
   - From `train_pool = ⋃_{j ≠ k} fold_j's normal partition` (~20k at L21 scale):
     - sample `train_normal_ids` (≤ train_budget) using `seed + TRAIN_NORMAL_SEED_OFFSET (=0)`
     - sample `tune_normal_ids` (≤ tune_budget) using `seed + TUNE_NORMAL_SEED_OFFSET (=2000)`,
       excluding train_normal IDs
   - From the anomaly pool:
     - sample `tune_anomaly_ids` (≤ tune_budget) using `seed + TUNE_ANOMALY_SEED_OFFSET (=3000)`
     - sample `test_anomaly_ids` (≤ test_budget) using `seed + TEST_ANOMALY_SEED_OFFSET (=4000)`,
       excluding tune_anomaly IDs

The four distinct seed offsets are the key invariant: **changing one stream's
budget MUST NOT shift another stream's IDs** for fixed `(seed, fold_id)`. Tested
via `TestSeedOffsetInvariant` (3 cases).

**Defaults (locked, paper-faithful):**
- `train_normal_per_fold = 5_000` (paper L23)
- `tune_normal_per_fold = 1_000` (paper L24)
- `tune_anomaly_per_fold = 1_000` (paper L24)
- `test_anomaly_per_fold = 1_000` (paper L25)

**Disjointness invariants (test-locked):**
- Within-fold:
  - `set(train_normal_ids) ∩ set(tune_normal_ids) == ∅` (tune carved from pool \ train_ids)
  - `set(tune_anomaly_ids) ∩ set(test_anomaly_ids) == ∅` (test carved from pool \ tune_ids)
  - `set(train_normal_ids) ∩ set(test_normal_ids) == ∅` (different partitions)
  - `set(tune_normal_ids) ∩ set(test_normal_ids) == ∅` (tune from train pool)
- Cross-fold:
  - `test_normal_ids`: **disjoint** across folds (normal partition guarantee)
  - `train_normal_ids` / `tune_normal_ids`: partial overlap (E[overlap] ≈ 938 at HDFS scale)
  - `tune_anomaly_ids` / `test_anomaly_ids`: per-fold shuffle of shared pool →
    **E[overlap] ≈ K² / |pool| ≈ 500** at the paper-faithful 1k budget / 2k pool scale

**Cross-fold anomaly overlap test:** `test_cross_fold_anomaly_overlap_within_expected_range`
asserts `350 < overlap < 650` for any two folds at HDFS L21 scale, matching the
theoretical expectation.

**Migration:** existing `splits.json` artifacts from v1.4 are NOT
backward-compatible (different schema, different anomaly partitioning). Re-run
`python -m src.splits` to regenerate before running `score.py` or
`tune_threshold.py`.

### D_NEW6.1 — Proportional degradation when pool < total budget

**Supersedes:** D_NEW6's initial degradation policy (train-priority for
normals, tune-priority for anomalies, both stream-priority).

**Trigger:** Pipeline-validation smoke test on HDFS subset (698 anomalies < 1k
tune_anomaly budget) revealed the initial policy starved the test stream:
tune_anomaly took its full 1k budget capped at the pool size (698), leaving
zero anomalies for `test_anomaly`. Result: F1/precision/recall unmeasurable.

**Resolution:** When `pool < tune + test` (or `pool < train + tune` for normals),
allocate **proportionally**:

```
# Normals (in _carve_train_and_tune_normals):
if pool < train_budget + tune_budget:
    effective_train = pool * train_budget // (train_budget + tune_budget)
    effective_tune  = pool - effective_train

# Anomalies (in _carve_tune_and_test_anomalies):
if pool < tune_budget + test_budget:
    effective_tune = pool * tune_budget // (tune_budget + test_budget)
    effective_test = pool - effective_tune
```

Both streams stay non-empty whenever both budgets > 0 and pool > 1.

**Trade-off — seed-offset invariant scope:**
- **Non-degraded regime** (pool ≥ total budget): D_NEW6 invariant holds —
  changing one stream's budget does not shift another stream's IDs.
- **Degraded regime** (pool < total budget): budgets are coupled by
  construction; changing tune_budget shifts effective_train. The underlying
  shuffles still use independent seeds, so the result is deterministic in
  `(seed, pool, budgets)`.

Tests in `TestSeedOffsetInvariant` use parameters that DO NOT trigger
degradation, so they assert the strict invariant. A degenerate-regime
determinism test in `TestDeterminism` covers reproducibility under degradation.

**Empirical verification on three subsets:**

| Dataset | Anomaly pool | Result |
|---|---|---|
| HDFS (post-subsample) | 2,000 | tune=1,000, test=1,000 (paper-faithful) |
| BGL subset | 2,000 (post-subsample) | tune=1,000, test=1,000 (paper-faithful) |
| TB subset | 2,000 (post-subsample) | tune=1,000, test=1,000 (paper-faithful) |
| HDFS subset BEFORE subsample | 698 | tune=349, test=349 (proportional) |

**Disclosure for dissertation:** Smoke-test runs at subset scale that hit the
degraded regime must declare the actual (tune, test) sizes. Final runs on
subsampled L21 data hit the non-degraded regime cleanly.

**Test-locked:**
- `test_graceful_degradation_proportional_normals` — pool=50, budgets 40+20 → 33+17
- `test_graceful_degradation_proportional_anomalies` — pool=40, budgets 30+30 → 20+20
- `test_small_normal_pool_proportional` — full create_splits trace on 600+100 input
- `test_small_anomaly_pool_proportional` — same input, anomaly carve
- `test_hdfs_subset_regression_v1_5_1` — regression guard for the 698-anomaly bug

### D_NEW7 — Score persistence as list-of-records, not int-keyed dict

**Supersedes:** v1.4 implicit assumption that `dict[int, float]` survives
JSON round-trip.

**Problem:** `ParagraphScore.topk_accuracies: dict[int, float]` becomes
`{"5": 0.85, "9": 0.90, "12": 0.95}` after `json.dump` (Python int keys are
coerced to strings by the JSON spec). Downstream `tune_threshold.py` accessing
`score["topk_accuracies"][5]` raises `KeyError: 5`.

**Resolution:** Persist as a flat list of records.

```python
@dataclass
class TopKAccuracyRecord:
    top_k: int
    accuracy: float

@dataclass
class ParagraphScore:
    paragraph_id: str
    label: int
    topk_accuracies: list[TopKAccuracyRecord]   # CHANGED v1.5 (was dict[int, float])
    n_masked_total: int                          # NEW v1.5 — total masked positions across all passes
    n_passes: int                                # NEW v1.5 — for R-passes audit (see D_NEW8)
```

**Why list-of-records:**
- JSON-native: no key-type coercion fragility.
- Order is preserved (numeric order of `top_k` enforced by `score.py`).
- `tune_threshold.py` iterates records and filters by `top_k`, never indexes
  by integer key.

**Test-locked:** `tests/test_score.py` enforces the canonical schema and the
list-of-records output shape via integration with `src/types.py`.

### D_NEW8 — `--n-passes` scoring flag, default 1 (paper-faithful)

**Supersedes:** v1.4 implicit single-pass scoring (spec v1.2 §5).

**Motivation:** With `r_sent=0.5 × r_tok=0.8`, single-pass scoring masks ~40%
of tokens per paragraph. Per-paragraph topk-accuracy is therefore noisy at
small `n_masked_total`. The paper itself is silent on the number of passes;
the v1.2 spec inherits this silence as "single pass."

For paper-faithful Phase 1 we hold to single-pass scoring. For Phase 2
ablations and noise-reduction sensitivity checks, R-passes scoring is needed.

**Resolution:** `--n-passes N` CLI flag in `src/score.py`:
- `--n-passes 1` (default) — paper-faithful single pass.
- `--n-passes R ≥ 2` — R independent mask draws per paragraph; per-paragraph
  topk accuracies are computed as the **pooled** ratio across passes:
  ```
  topk_accuracy(K) = sum(correct_in_top_K across passes) / sum(n_masked across passes)
  ```
  Pooled (numerator + denominator separately summed) rather than averaged
  (mean-of-fractions) — robust to passes with different mask cardinalities.

**Zero-mask edge case:** If `sum(n_masked) == 0` for a paragraph across all
passes (stochastic masking produced no scorable positions), all K-accuracies
are set to **1.0** by convention; `n_masked_total = 0` carries the diagnostic
forward so downstream consumers (`tune_threshold.py`, `evaluate.py`) can filter
or flag.

**Determinism:**
- Each pass uses `seed = base_seed + pass_idx`.
- Cross-run reproducibility: same `--seed` + same `--n-passes` produces
  bit-identical scores.

**Reporting:** `--n-passes` is persisted at the `SplitScores` level (same for
every paragraph in a run). Per-paragraph `n_masked_total` is the sum across
all passes — sufficient to reconstruct per-pass count if needed.

### D_NEW9 — Inference batch size defaults to `physical_batch_size`

**Supersedes:** v1.4 implicit defaulting to `effective_batch_size`.

**Problem:** Training uses `physical_batch_size` (with gradient accumulation to
reach `effective_batch_size`). Inference at `batch_size=effective_batch_size`
runs forward-only with no gradient buffers but proportionally more activation
memory. Net memory usage at inference exceeds training. Longformer-BGL scoring
would OOM at the v1.4 default.

**Resolution:**
- `src/score.py` defaults inference `batch_size` to `physical_batch_size` from
  config (Longformer=2, RoBERTa=8).
- `--batch-size N` CLI flag overrides for empirical optimization (some
  inference setups can fit larger batches than training when fp16 + no_grad).

**Rationale:** "What fits during training" is a strict lower bound on what
fits during forward-only inference with no_grad — at the same batch size.
Defaulting to the known-safe value protects against the OOM showstopper case.

---

## 3. New pipeline component

### Component 1.6 — `src/subsample_paragraphs.py` (paper L21 caps)

**NEW v1.5.** Bridges the gap between prep output and the paper's allocation
budget.

**Problem:** Paper §IV-A L21 specifies 25k normal + 2k anomaly per dataset.
Your prep scripts (`prepare_hdfs.py`, `prepare_bgl_tbird.py`) output **all**
paragraphs (or all up to `--max-input-lines`), which can be:
- **Below** L21: HDFS subset has 14.9k normal + 698 anomaly. Cannot be
  upsampled; flagged as a paper deviation.
- **Above** L21: TB subset has 77.3k normal + 45.1k anomaly. Needs subsampling
  to match L21.

`splits.py` could be patched to add a label-stratified subsample step, but
that conflates two concerns (allocation vs. partitioning). Keeping them as
separate pipeline stages is cleaner.

**Algorithm (`subsample_paragraphs`):**

1. Separate input into normal and anomaly index lists.
2. Apply per-label stratified random sampling:
   - **Normals**: if `count > max_normal`, sample `max_normal` items using
     `random.sample(normal_indices, max_normal)` with `seed + NORMAL_SEED_OFFSET (=0)`.
   - **Anomalies**: same with `seed + ANOMALY_SEED_OFFSET (=1000)`.
3. **Cap-only semantics**: inputs below cap pass through unchanged for that label.
4. **Order preservation**: retained paragraphs appear in output in input order
   (matters for BGL/TB time-sorted prep output).
5. Persist `subsample_summary.json` alongside the new `paragraphs.pkl` with
   audit fields: input/output counts per label, was_capped flags, seed, caps.

**Seed-offset decoupling:** changing `max_normal` does NOT affect which
anomalies are retained, and vice versa. Same principle as D_NEW6's four-stream
invariant. Test-locked via `TestSeedOffsetDecoupling` (2 cases).

**Defaults (paper L21):**
- `max_normal = 25_000`
- `max_anomaly = 2_000`
- `seed = 42`

**Pipeline placement (correct order):**
```
prepare_*.py            (output: paragraphs.pkl)
        ↓
subsample_paragraphs.py (cap to L21 budgets)
        ↓
splits.py               (5-fold CV per D_NEW6)
        ↓
train.py
```

**Empirical results on three subsets (post-subsample):**

| Dataset | Input | Output | Effect |
|---|---|---|---|
| HDFS subset | 14,941 + 698 | 14,941 + 698 | no-op (both below caps) |
| BGL subset | 24,070 + 2,555 | 24,070 + 2,000 | anomaly capped only |
| TB subset | 77,347 + 45,106 | 25,000 + 2,000 | both capped |
| HDFS full (re-prepped) | 558,223 + 16,838 | 25,000 + 2,000 | both capped — **paper-exact** |

**Final per-fold budgets after subsample → splits:**

| Dataset | train_N / tune_N / tune_A / test_N / test_A | vs paper |
|---|---|---|
| HDFS (full → L21) | 5,000 / 1,000 / 1,000 / 5,000 / 1,000 | **paper-exact** ✓ |
| TB | 5,000 / 1,000 / 1,000 / 5,000 / 1,000 | **paper-exact** ✓ |
| BGL | 5,000 / 1,000 / 1,000 / 4,814 / 1,000 | test_N at 96% of paper L25 |

**Disclosure for BGL:** test_N=4,814 vs paper's 5,000 because BGL subset prep
output only 24,070 normals (4% below 25k L21). Within rounding tolerance for
the dissertation; could be fixed by re-prepping BGL with more raw input lines.

---

## 4. What v1.5 does NOT change from v1.4

All v1.4 decisions remain in force:
- D_NEW1, D_NEW2, D_NEW3 (Phase 1 paper-faithful), D_NEW4 (Phase 2 sliding-window deferred), D_NEW5 (results disclosure)
- All v1.3 carryovers: D1, D2, D3, D4 v1.3, D5, D8 v1.2, D13, D15 v1.2, D16 v1.3, D17 v1.2, D19 v1.3, D20 v1.3

**D14 is superseded** by D_NEW6 (Paradigm B replaces the original both-axis
partition).

The L2 backbone selection (RoBERTa-base 512 / Longformer-base-4096 4096) and
the empirical evidence base (E1–E4) are unchanged.

---

## 5. Implementation status (as shipped)

| Module | v1.5 work | Status | Tests |
|---|---|---|---|
| `src/types.py` | Add `TopKAccuracyRecord`; modify `ParagraphScore` (records + 2 audit fields) | ✓ Done | covered indirectly |
| `src/splits.py` | Paradigm B rewrite, 5 ID lists, four seed offsets, proportional degradation | ✓ Done | 65 / 65 pass |
| `tests/test_splits.py` | Full rewrite for Paradigm B + D_NEW6.1 regression guards | ✓ Done | 65 tests |
| `src/subsample_paragraphs.py` | NEW — L21 stratified subsample | ✓ Done | 27 / 27 pass |
| `tests/test_subsample_paragraphs.py` | NEW | ✓ Done | 27 tests |
| `src/score.py` | Records format (B2), `--n-passes` (D_NEW8), `--batch-size` (D_NEW9), I3 hard-fail, M1 canonical schema, M2 uniqueness | ✓ Done | unit-tested via test_score.py |
| `tests/test_score.py` | NEW — pure-logic helpers | ✓ Done | 46 / 46 pass |
| `scripts/score.sh` | NEW — SLURM array wrapper | ✓ Done | n/a (integration) |
| `src/tune_threshold.py` | Build against D_NEW7 records format + tune split | Not started | — |
| `src/evaluate.py` | Build against fold metrics aggregation per spec §7 | Not started | — |
| `tests/test_tune_threshold.py` / `tests/test_evaluate.py` | NEW | Not started | — |
| `scripts/evaluate.sh` | NEW — CPU-only SLURM wrapper | Not started | — |

**Test totals as of v1.5 shipped:** 65 (splits) + 27 (subsample) + 46 (score) =
**138 tests** covering the v1.5 changes alone.

---

## 6. Decision summary table

| ID | Decision | Status | Source |
|---|---|---|---|
| D_NEW6 | Paradigm B splits — normals partitioned, anomalies pooled, 5 ID lists, four seed offsets | Locked + shipped | Paper §IV-A L21–L25; supersedes Path-3 plan |
| D_NEW6.1 | Proportional degradation when pool < total budget | Locked + shipped | HDFS-subset (698 anomalies) integration triage |
| D_NEW7 | Score persistence as list of `{top_k, accuracy}` records | Locked + shipped | score.py review B2 (JSON int-key coercion) |
| D_NEW8 | `--n-passes` flag, default 1 (paper-faithful); pooled accumulation | Locked + shipped | score.py review I1 (paragraph-level noise) |
| D_NEW9 | Inference `batch_size` defaults to `physical_batch_size` | Locked + shipped | score.py review B1 (Longformer OOM risk) |
| Component 1.6 | `subsample_paragraphs.py` — stratified L21 cap | Locked + shipped | Paper §IV-A L21 allocation requirement |

---

## 7. Open items for v1.6 (if/when needed)

- `src/tune_threshold.py` build (paper §III-C grid: `linspace(train_top1_acc - 0.1, train_top1_acc, 3)` × `topk ∈ {5, 9, 12}`, F1 best with smaller K then larger θ tie-break).
- `src/evaluate.py` build (P/R/F1/Spec per fold + mean ± std with `ddof=1`; persist per-fold operating points).
- BGL re-prep with larger `--max-input-lines` to reach paper-exact 25k normals (currently 24,070 → 96% of L25 test_N).
- Cosmetic `train.py` path bug: `hdfs_hdfs` fallback when `window_seconds is None`. Non-blocking.

---

## 8. References

- **v1.4 decisions:** `docs/logfit-repro-decisions-v1.4.md`
- **v1.3 spec:** `docs/logfit-repro-spec-v1.3.md` (Components 5, 6, 7 deferred to v1.2)
- **v1.2 spec §5 (Anomaly Scoring):** `docs/Logfit-repro-spec-v1.2.md`
- **Source paper:** Almodovar, C., Sabrina, F., Karimi, S., & Azad, S. (2024).
  LogFiT: Log Anomaly Detection using Fine-Tuned Language Models.
  *IEEE Transactions on Network and Service Management*, 21(2):1715–1723.
  DOI: 10.1109/TNSM.2024.3358730.
