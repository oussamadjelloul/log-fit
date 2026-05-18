# LogFiT Reproduction — v1.4 Decisions

**Date:** 2026-05-18
**Status:** Draft. Pending supervisor sign-off on Q_NEW1–Q_NEW3.
**Supersedes:** v1.3 decisions where conflicts arise (explicitly flagged).
**Author:** Oussama Djoussa, Université Laval Cybersecurity Lab

---

## 1. Context — what changed since v1.3

v1.3 locked the preparation-layer methodology and was test-verified (167/167 unit tests passing). Running the v1.3 pipeline on real-data subsets surfaced empirical findings that require methodology adjustments:

1. **L2 backbone selection rule is broken for log data.** Uses word-count quantile but compares against token-count threshold. Inconsistent units inside the rule itself.
2. **§1.6 HARD FAIL threshold rests on an unverified assumption** (signal sparsity). Full-BGL anomaly-distribution evidence shows that assumption is empirically wrong, in a direction that makes right-truncation *more* defensible than v1.3 assumed.
3. **Paper does not report truncation rate or token-length statistics.** Reproducibility requires we document what we measure, since the paper does not.

v1.4 integrates four pieces of evidence (token-length measurements, Gemini adversarial review, paper-text verification, anomaly temporal distribution) into a coherent, defensible methodology.

---

## 2. Empirical evidence (locked)

### E1 — Token-length gate measurements

Ran `src/token_length_gate.py` on real-data subsets after v1.3 prep:

| Dataset | Backbone | min | p50 | p80 | p95 | p99 | max | Trunc rate |
|---|---|---|---|---|---|---|---|---|
| HDFS (200k lines, 15,639 paragraphs) | roberta-base (512) | 62 | 693 | 705 | 733 | 784 | 13,211 | 95.63% |
| BGL 30s (100k lines, 240 paragraphs) | longformer-base-4096 (4096) | 152 | 13,232 | 64,333 | 77,258 | 80,600 | 97,419 | 77.50% |

Both subsets HARD FAIL §1.6 (max > 5× backbone limit).

### E2 — Word-to-token expansion ratio on log data

| Dataset | p50 ratio | p80 ratio | max ratio |
|---|---|---|---|
| HDFS | 693/165 = **4.20×** | 705/165 = **4.27×** | 13,211/2,857 = **4.62×** |
| BGL 30s | 13,232/2,388 = **5.54×** | 64,333/9,764 = **6.59×** | 97,419/16,005 = **6.09×** |

For natural English prose, the typical BPE ratio is ~1.3×. Log data exhibits **4-6× expansion** due to BPE fragmentation of structured artifacts: block IDs (`blk_-1608999687919862906` → ~12-15 tokens), IP+port pairs (`/10.251.107.50:54106` → ~10 tokens), timestamps, hex addresses, FQ class names. This is a known phenomenon in the log-anomaly literature (see LogBERT's custom WordPiece vocabulary); the LogFiT paper's parsing-free design with default RoBERTa vocab inherits the issue.

### E3 — Anomaly temporal distribution on full BGL.log

Ran `scripts/anomaly_temporal_distribution.py` on the full BGL.log (4.7M lines, 2,555 anomaly windows analyzed under 30s windowing):

**First-anomaly position within window (normalized 0=start, 1=end):**
- p25 = 0.000, p50 = 0.000, p75 = 0.000, mean = 0.071
- **88.4% of anomaly windows have first-anomaly in first 10% of window**
- 90.5% in first 25%
- Histogram concentrated almost entirely in the first decile (2248 of 2555 windows)

**Anomaly density per anomaly window:** mean = 0.732, **median = 1.000**.
Most anomaly windows are entirely anomalous (every line non-"-"). The "all-anomaly position distribution uniform" pattern observed in the subset is structural — when every line is anomalous, every position contributes equally — not informative about clustering.

**Window length (anomaly windows):** mean 182.8 lines, p50 175 lines.

**Subset-vs-full validation:** the 100k-line subset showed p50 first-anomaly = 0.041 and median density = 0.016. Full file shows p50 = 0.000 and median density = 1.000. The subset under-stated the strength of the early-clustering pattern and miss-stated the density pattern. **Full-file numbers govern v1.4 decisions.**

### E4 — Paper text verification (Almodovar et al. 2024)

Three statements verified verbatim against the paper PDF:

**§III.A (page 5), Heuristic paragraph:**

> "LogFiT tool also includes a heuristic to automatically select between RoBERTa and Longformer based on the 0.8-quantile length **(in words)** of the training log samples, found during preprocessing. LogFiT selects RoBERTa for datasets with log samples containing no more than **512 tokens**."

The measurement is in **words**; the threshold is expressed in **tokens**; no conversion factor is provided. The rule is internally inconsistent within a single sentence.

**§IV-A (page 6), Experimental Setup:** Covers datasets, log paragraphs, k-fold CV, log content variability, baselines, implementation details, evaluation metrics, hyperparameter tuning. **No mention of truncation rate, no quantification of paragraphs exceeding context window.**

**Table II caption:** `PER-PARAGRAPH WORD AND SENTENCE STATISTICS FOR THE DATASETS`. Columns: Dataset, Avg Word Count, Avg Sentence Count, #Unique Words. **No token-length statistics.**

---

## 3. Methodology adjustments

### D_NEW1 — L2 backbone selection rule: token-based fix

**Supersedes v1.3 D_L2.**

**Old rule:** Compute 0.8-quantile of *word count* per paragraph. If ≤ 512, RoBERTa; else Longformer.

**New rule:**
1. After prep produces `paragraphs.pkl`, tokenize every paragraph using `roberta-base` tokenizer (RoBERTa and Longformer share the same Byte-Level BPE vocab, size 50,265; Longformer was initialized from RoBERTa weights — tokenization is identical, only position embeddings differ).
2. Compute `p80` of the resulting per-paragraph token-count distribution.
3. **If `p80 ≤ 512`, use RoBERTa-base. Else, use Longformer-base-4096.**
4. Persist the token-count distribution in `preparation_summary.json` under `token_length_distribution`.

**Empirical consequence:** under v1.3 rule (word-based), HDFS p80=165 → RoBERTa. Under v1.4 rule (token-based), HDFS p80=705 → would route to **Longformer**. This deviates from the paper, which uses RoBERTa for HDFS.

**Resolution for paper-faithful Phase 1:** Use the paper's backbone assignments **as published** (RoBERTa for HDFS; Longformer for BGL/TB 30s/60s). Document the v1.4 rule and report what it *would* have selected as a methodological observation. The v1.4 rule applies to any new dataset added to the pipeline (e.g., dissertation experiments beyond reproduction).

### D_NEW2 — §1.6 HARD FAIL threshold: revised philosophy

**Supersedes v1.3 §1.6 HARD FAIL trigger.**

**Old rationale:** `max > 5 × backbone_limit` HARD FAILs because right-truncation would discard > 80% of paragraph content, biasing the anomaly signal toward early-window events.

**New rationale, evidence-based:**
- E3 shows BGL anomaly events are **frontloaded** (88.4% first-anomaly in first 10% of window).
- E3 shows median BGL anomaly window has **100% anomaly density** (every line is non-"-"); right-truncation at any fraction sees at least *some* anomalous content for these windows.
- The "loss of anomaly signal under truncation" concern is therefore **empirically weak for time-windowed configs on BGL**, contradicting v1.3's defensive 5× threshold.

**New rule for v1.4:**
- HARD FAIL retained for `max > 5 × backbone_limit` only when the empirical anomaly-position distribution shows p50 first-anomaly > 0.30 (i.e., truncation is *not* signal-preserving). This must be verified per-dataset via `scripts/anomaly_temporal_distribution.py` before gate runs.
- For datasets where p50 first-anomaly < 0.30: HARD FAIL **demoted to SUPERVISOR_FLAG** with explicit truncation-rate disclosure.
- The p95 > backbone_limit SUPERVISOR_FLAG retained unchanged.

**Per-dataset application:**
- **BGL 30s** (E3 evidence): p50 first-anomaly = 0.000 → HARD FAIL demoted; SUPERVISOR_FLAG with disclosed truncation rate.
- **HDFS**: temporal-distribution analysis not yet run (HDFS is block-grouped, not time-windowed; analysis script needs adaptation). For Phase 1, treat HDFS the same way (demote HARD FAIL) on the basis that paper-faithful reproduction is the goal and the paper accepts right-truncation.
- **Thunderbird 30s/60s**: same as BGL pending temporal-distribution confirmation.

### D_NEW3 — Truncation policy: paper-faithful Phase 1, optional Phase 2 sensitivity check

**Resolves Q_NEW (v1.4) on truncation strategy.**

**Phase 1 (primary, paper-faithful):**
- Right-truncate paragraphs at `backbone_limit` tokens during training and inference.
- Use the paper's backbone-to-dataset assignments (RoBERTa-HDFS, Longformer-BGL/TB).
- Report F1 / AUROC as headline numbers.
- Disclose truncation rate + estimated FN inflation in the results table (see D_NEW5).
- **Rationale:** the goal is to reproduce the paper's claims under the paper's methodology. E3 evidence indicates the methodology is more defensible than initially feared.

**Phase 2 (optional, sensitivity check):**
- Implement sliding-window inference (see D_NEW4) without retraining.
- Run on the same trained models as Phase 1.
- Report Phase 1 vs Phase 2 deltas in the discussion.
- **Goal:** quantify the ~5-10% F1 inflation expected from recovering the windows where first-anomaly fell outside the truncation cutoff. This is a defensible sensitivity analysis, not a methodology replacement.
- **Trigger to execute Phase 2:** discretionary, based on whether Phase 1 F1 closely matches the paper's published numbers. If Phase 1 matches → Phase 2 strengthens the dissertation. If Phase 1 diverges → debug Phase 1 first; Phase 2 only after.

**Rejected alternatives (with reasoning):**
- **Option B (shorter windows for BGL/TB):** Would cut 4 of 6 paper configurations. Not paper-faithful.
- **Option C (Longformer for HDFS):** Deviates from paper's backbone choice. Phase 2 may explore this incidentally if HDFS gets re-tokenized for sliding-window scoring, but not in Phase 1.
- **Option D (sub-line replacement):** Materially changes input distribution. Methodology innovation, not reproduction.
- **Option E (reject affected configs):** Useless for reproduction.
- **Option F (random subsampling):** Destroys temporal order. Wrong for cascade-pattern data per E3.

### D_NEW4 — Phase 2 sliding-window protocol (deferred specification)

When Phase 2 is executed, the protocol is:

1. For each paragraph with `token_count > backbone_limit`, split into overlapping segments of `backbone_limit` tokens with stride `backbone_limit / 2` (50% overlap).
2. Each segment inherits the parent paragraph's label.
3. Run inference on every segment, producing per-segment anomaly scores.
4. Aggregate per-paragraph score via **max-pooling** over segment scores (any segment scoring above threshold flags the parent paragraph as anomaly). Justification: matches the anomaly-detection semantics (one anomalous region anywhere in the paragraph → anomalous paragraph).
5. Compute F1/AUROC over per-paragraph aggregated scores; compare to Phase 1.

Implementation deferred to `src/sliding_window_score.py` (not blocking Phase 1).

### D_NEW5 — Results disclosure requirements

Every results table in the dissertation must accompany F1/AUROC numbers with:

1. **Backbone used** (RoBERTa-base or Longformer-base-4096) per dataset.
2. **Truncation rate at backbone limit** (% of paragraphs with token_count > limit).
3. **Estimated FN inflation from right-truncation** (from temporal-distribution analysis; ~9.5% for BGL 30s).
4. **Whether sliding-window Phase 2 was run** and, if so, Phase 1 vs Phase 2 F1 delta.

Rationale: paper omitted these (E4); reproduction documents what reproduction discloses, restoring the transparency the original paper lacked.

---

## 4. Paper methodology gaps (informational; for SLR / dissertation, not blocking reproduction)

### M1 — Word/token unit conflation in §III.A

Per E4, the L2 heuristic measures word-count quantile but compares against token-count threshold within a single sentence, with no conversion factor. This is a definitional error. Combined with E2 (4-6× expansion for log data), the paper's RoBERTa-for-HDFS choice is unjustifiable as written: word p80=165 → "no more than 512 tokens" → RoBERTa, but actual token p80=705. The rule, taken literally, should have routed HDFS to Longformer.

This is a useful precedent for the SLR: prior work in the LogFiT lineage has shipped with internally inconsistent preprocessing rules; SemLog-X taxonomy can cite this as a category of methodology-transparency gap.

### M2 — Truncation rate undocumented

Per E4, §IV-A doesn't quantify the truncation rate. Table II reports word and sentence statistics, no token statistics. A reader cannot independently verify whether the paper's reported F1 numbers represent "anomaly detection performance" or "anomaly detection performance on the first 5% of each window."

E3 partially rehabilitates the paper: empirically, the first 5-30% of BGL windows contains ~90% of anomaly trigger signal. So the paper's F1 numbers are likely meaningful, not degenerate. But the **paper itself does not establish this**; the reader has to do the analysis our pipeline did.

### M3 — Word-to-token ratio assumption never validated

The paper's heuristic implicitly assumes word count ≈ token count (or some fixed ratio close to 1). E2 shows this assumption is empirically violated by a factor of 4-6 on log data. The paper never validates or even acknowledges this. This is an SLR-relevant gap: a recurring failure mode of "LM for log analysis" papers is treating tokenization as a transparent step rather than a methodology-sensitive one.

---

## 5. Open questions requiring supervisor sign-off

### Q_NEW1 — Approve two-phase plan (Phase 1 paper-faithful, Phase 2 sensitivity)

D_NEW3 lays out the plan. Supervisor sign-off required because: Phase 2 adds ~30% to the engineering scope (sliding-window inference + result aggregation), but pays off by separating "did SDD beat LogFiT-as-published" from "did SDD beat LogFiT-with-corrected-methodology." Recommend approval; alternative is Phase 1 only with M1-M3 as future-work notes.

### Q_NEW2 — Methodology critique framing for thesis

Section 4 (M1-M3) constitutes a substantive critique of the LogFiT paper's preprocessing methodology. Two framings possible:

**(a) Light:** Single paragraph in the experimental discussion noting "LogFiT's word-count heuristic and undocumented truncation rate make exact reproduction methodology-sensitive; we adopt the paper's published backbone assignments and disclose truncation rates throughout."

**(b) Strong:** Dedicated subsection in the related-work / SLR analysis chapter using the LogFiT case as a worked example of a methodology-transparency gap in LM-based log anomaly detection. Connects to SemLog-X's design principle (disclosure of detection-evidence).

Recommend (b) — it elevates an empirical finding into a thesis-level contribution and motivates SDD's design philosophy. (a) is the fallback if scope concerns dominate.

### Q_NEW3 — BGL/TB 30s/60s configs: keep, drop, or shorten?

Even under Phase 1 paper-faithful settings with D_NEW2's relaxed HARD FAIL, the BGL 30s subset still shows 77.5% truncation rate. Three sub-options:

- **(i)** Keep all 6 paper configurations (3 BGL × {10,30,60}s + 3 TB × same). Accept high truncation for 30s/60s. Disclose per D_NEW5. (Current plan under D_NEW3.)
- **(ii)** Keep all 6, but elevate sliding-window Phase 2 to mandatory for 30s/60s configs only. Smaller dissertation scope adjustment.
- **(iii)** Drop 30s/60s configs entirely, run only 10s. Cuts the F1 comparison surface but eliminates the truncation question entirely.

Recommend (i) for Phase 1, with (ii) as the cleanest path if Phase 2 is approved per Q_NEW1.

---

## 6. What v1.4 does NOT change from v1.3

These v1.3 decisions remain in force:
- D1: 3 epochs HDFS, 5 epochs BGL/TB
- D2: max_lr grid {1e-5...5e-4}, 500-paragraph subset for selection
- D3: batch 8 RoBERTa / 2 Longformer
- D4 v1.3: 3-field minimum, label-only stripping, encoding audit
- D5: window anomaly if any line label_field != "-"
- D8 v1.2: seed=42, full_determinism=True, fp16 with 1e-4 threshold
- D13: top-10 verbs
- D14: per-fold anomaly shuffle, cross-fold overlap E[overlap]=500=50%
- D15 v1.2: HF Trainer + OneCycleLR (transformers 5.x compatibility audit still pending — see implementation status)
- D16 v1.3: distinct-id semantics for multi-blockid counter
- D17 v1.2: WordNet first-synset-first-single-word-lemma
- D19 v1.3: duplicate-block_id HARD FAIL on conflict
- D20 v1.3: encoding violations tracked

The L1 backbone selection (RoBERTa-base 512 / Longformer-base-4096 4096) is unchanged; only the *selection rule* (D_L2 → D_NEW1) is revised.

---

## 7. Implementation status as of v1.4 draft

**Prep layer (Component 1):** ✅ Complete, 167/167 unit tests passing on Narval sdd_env. v1.3 code stands; no v1.4 changes needed.

**Token-length gate (Component 1.6):** ✅ Implemented (`src/token_length_gate.py`), 22/22 tests passing. v1.4 adjustment needed: demote HARD FAIL to SUPERVISOR_FLAG for time-windowed datasets per D_NEW2. ~5 lines of code change.

**Anomaly temporal distribution analysis:** ✅ Standalone script (`scripts/anomaly_temporal_distribution.py`) implemented and run on full BGL.log. Re-usable for HDFS (after adaptation for block-grouped paragraphs) and Thunderbird.

**Backbone selection (D_NEW1):** ❌ Not yet implemented. New module `src/select_backbone.py` needed. ~80 lines. Reads `preparation_summary.json` token-length distribution, applies v1.4 rule, writes `backbone_choice.json`. For paper-faithful Phase 1, this module can be bypassed via config override (specify backbone explicitly).

**Splits (Component 1.7):** ❌ Not yet implemented. New module `src/splits.py` needed. ~150 lines. Unchanged from v1.3 D14 plan.

**Training (Component 4):** ❌ Not yet implemented. Will need transformers 5.x compatibility verification per v1.3 implementation status. Phase 1 path: standard right-truncation via `tokenizer(text, truncation=True, max_length=backbone_limit)`.

**Scoring + evaluation (Components 5, 7):** ❌ Not yet implemented.

**Sliding-window Phase 2 (D_NEW4):** ❌ Deferred. To be specced and built only after Phase 1 results are in and supervisor signs off on Q_NEW1.

---

## 8. Decision summary table

| ID | Decision | Status | Source |
|---|---|---|---|
| D_NEW1 | L2 rule operates on token counts, not word counts | Locked, pending Phase-1 bypass for paper-faithful run | E1, E2, E4 (§III.A unit conflation), Gemini Q5 |
| D_NEW2 | §1.6 HARD FAIL demoted to SUPERVISOR_FLAG for datasets with empirical p50 first-anomaly < 0.30 | Locked for BGL; HDFS/TB pending temporal analysis | E3 |
| D_NEW3 | Paper-faithful Phase 1 (right-truncation); Phase 2 sliding-window as optional sensitivity check | **Pending supervisor approval (Q_NEW1)** | E3, Gemini Q7+Q8 |
| D_NEW4 | Phase 2 protocol: 50%-overlap sliding window, max-pool aggregation | Spec locked, implementation deferred | Gemini Q4 (Option G) |
| D_NEW5 | Disclose backbone, truncation rate, FN inflation estimate, sliding-window deltas in every results table | Locked | E4 (paper omitted these), M2 |

---

## 9. References

- **v1.3 decisions doc:** `docs/logfit-repro-decisions-v1.3.md`
- **v1.3 spec doc:** `docs/logfit-repro-spec-v1.3.md`
- **Gemini adversarial review:** `docs/logfit-gemini-v14-token-length-validation.md` (prompt) + Gemini response (this session)
- **Empirical artifacts:**
  - `~/scratch/log-fit/data/hdfs_subset/preparation_summary.json` (E1 HDFS)
  - `~/scratch/log-fit/data/bgl_subset/preparation_summary.json` (E1 BGL subset)
  - `~/scratch/log-fit/data/bgl_subset/anomaly_temporal_distribution.json` (E3 subset)
  - `~/scratch/log-fit/data/bgl_full/anomaly_temporal_full_bgl_30s.json` (E3 full file — authoritative)
- **Source paper:** Almodovar, C., Sabrina, F., Karimi, S., & Azad, S. (2024). LogFiT: Log Anomaly Detection using Fine-Tuned Language Models. *IEEE Transactions on Network and Service Management*, 21(2).
