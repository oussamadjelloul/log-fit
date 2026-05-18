# LogFiT Reproduction — Methodology Decisions

**Version:** 1.3 (post-code-review of preparation modules)
**Status:** Draft for supervisor sign-off (gate artifact)
**Author:** Oussama
**Source paper:** Almodovar, C., Sabrina, F., Karimi, S., Azad, S. (2024). *LogFiT: Log Anomaly Detection Using Fine-Tuned Language Models.* IEEE Transactions on Network and Service Management, 21(2), 1715–1723. DOI: [10.1109/TNSM.2024.3358730](https://doi.org/10.1109/TNSM.2024.3358730)
**Reproduction repo:** `~/logfit-repro/`

## Changelog v1.2 → v1.3

Driven by tri-LLM adversarial review of `src/prepare_hdfs.py` and `src/prepare_bgl_tbird.py`. Two reviewers (Gemini + parallel Claude), 20 findings, 2 convergent. v1.2 methodology contract is preserved; v1.3 adds disclosure decisions and corrects implementation precision issues caught by the code review.

**Convergent findings → corrections:**

- **`lines_with_multiple_blockids` counter semantics** (R1 F1 + R2 F3) — counter was occurrence-based; both reviewers flagged it should be distinct-id-based, since D16's intent ("duplicate into multiple paragraphs") only applies when distinct blocks are referenced. **D16 clarified in v1.3.**
- **Cross-module import** of `compute_length_distribution` (R1 F3 + R2 F8) — extracted to `src/utils/stats.py` in v1.3.

**R2-only IMPORTANT findings → corrections:**

- **R2 F1** — Duplicate `block_id` in label CSV silently overwrites. **D19 (NEW)** added.
- **R2 F2** — `errors="replace"` produces no audit trail. **D20 (NEW)** added.

**R2-only MINOR findings → corrections:**

- **R2 F4** — block_id regex lacks word boundary. Pattern updated to `r"\bblk_-?\d+\b"`. **D16 expanded.**
- **R2 F5** — singleton windows don't distinguish anomaly drops. Counter split in v1.3.
- **R2 F6** — 2-token lines accepted as valid paragraphs. Now rejected; **D4 expanded.**
- **R2 F7** — D4 silent on delimiter whitespace handling. **D4 expanded** with explicit specification.
- **R2 F13** — label CSV header case-sensitivity. Now documented requirement: exact-case `BlockId,Label`.

Test gaps **R2 F9–F12, F14** addressed in spec Component 13.

---

## 1. Reproduction Scope (Unchanged from v1.0)

**Faithful reproduction only.** Random 25k normal + 2k anomaly per dataset, 5-fold CV, exactly as paper §IV-A specifies.

---

## 2. Protocol Risk Disclosure (REVISED v1.3 — §2.4 added)

### 2.1 Random-sample 5-fold CV on temporally ordered data (Unchanged from v1.1)
### 2.2 Paper-internal scoring inconsistency: LogBERT adoption claim (Unchanged from v1.1)
### 2.3 Cross-fold anomaly leakage from per-fold pool sampling (Unchanged from v1.2)

### 2.4 Encoding-replacement silent data loss (NEW v1.3)

Per **D20**, files are read with `errors="replace"`, which substitutes invalid UTF-8 bytes with `U+FFFD`. A corrupted byte inside a block_id (HDFS) or timestamp (BGL/TB) causes the regex or `int()` parse to silently fail. v1.3 mitigates with:

- An explicit `encoding_replacements_seen` counter in `DropCounters`.
- The first 100 offending line numbers persisted to `preparation_summary.json.encoding_offending_line_numbers` for triage.
- A supervisor warning if rate exceeds 0.01% of total lines.

The dissertation chapter must disclose that some lines may have been silently substituted under `errors="replace"` and that the audit trail captures the first 100 offending positions.

---

## 3. Locked Specifications (Unchanged from v1.2)

All L1–L27 carry forward.

---

## 4. Decisions Made Where Paper Is Silent (REVISED v1.3)

### D1–D3 (Unchanged)
### D4. Raw log line content + delimiter whitespace (REVISED v1.3)

- **Paper says:** "raw logs" (§III-A, §IV-A); HDFS Fig. 1 shows full lines retained.
- **Decision (v1.3, retained from v1.2 + new clarification):**
  1. Default: keep the full raw line including timestamp, level, component, message.
  2. Token-length conditional guard via §1.6.
  3. **Whitespace policy (NEW v1.3 — R2 F7):** Leading whitespace between the label field and subsequent content is treated as a *delimiter* and stripped. All other whitespace inside content (tabs, multiple spaces between non-label fields) is preserved verbatim. Operationally: `content = stripped[len(label_field):].lstrip()` strips the delimiter only.
  4. **Empty-content reject (NEW v1.3 — R2 F6):** Lines with exactly 2 whitespace-delimited tokens (label + timestamp, no content) are rejected as parse failures. Surfaced via `lines_missing_content` counter, not `unparseable_timestamp` (which is reserved for non-integer timestamps).

### D5. BGL/Thunderbird window labeling (Unchanged)
### D6. Sentence boundary definition (Unchanged)
### D7. Truncation strategy (Unchanged)
### D8. Random seed and CUDA determinism (Unchanged from v1.2)
### D9. Whether to mask special tokens (Unchanged)
### D10. Threshold lower-bound clipping (Unchanged)
### D11. UMAP for ablation Figure 4 (Unchanged — out of scope)
### D12. Throughput measurement (Unchanged)
### D13. WordNet verb count: top-10 (Unchanged from v1.1)
### D14. Anomaly pool partitioning (Unchanged from v1.2)
### D15. Training stack: HuggingFace Trainer + OneCycleLR (Unchanged from v1.2)

### D16. HDFS multi-block-id line handling (REVISED v1.3)

- **Paper says:** Unspecified.
- **Decision (v1.3):**
  1. **Regex:** `re.compile(r"\bblk_-?\d+\b")` — word-bounded, rejects substring matches like `prefixblk_123` (NEW v1.3 — R2 F4).
  2. **Extraction:** All matches per line via `re.findall`.
  3. **Within-line deduplication:** `unique_block_ids = list(dict.fromkeys(block_ids))` — preserves first-mention order.
  4. **Duplication:** Line is appended to each *distinct* block's paragraph (DeepLog convention).
  5. **Counter semantics (CORRECTED v1.3 — convergent R1 F1 / R2 F3):** `lines_with_multiple_blockids` increments only when a line references **two or more distinct block_ids**, not when the same block_id appears multiple times in one line. Operationally: `if len(unique_block_ids) > 1: counter += 1`. The verbose-log case (same block referenced N times) is not a duplication event and does not increment this counter.
- **Rationale (CORRECTED v1.3):** The counter measures D16's operative semantics — lines that cause cross-block duplication. Counting repeated mentions of the same block would inflate the metric and spuriously trigger the 5% SUPERVISOR_FLAG threshold on verbose-log datasets, with no methodological meaning.

### D17. WordNet "lemma" operationalization (Unchanged from v1.2)
### D18. Replacement substitution scope (Unchanged from v1.2)

### D19. Duplicate `block_id` in `anomaly_label.csv` (NEW v1.3)

- **Paper says:** Unspecified.
- **Decision:** `load_label_dict` HARD FAILs on **conflicting** duplicate rows (same block_id, different labels). **Exact duplicate rows** (same block_id, identical label) are accepted with a warning log entry. Either type of duplicate is surfaced via `drop_counters.duplicate_blockid`.
- **Rationale:** BUG-3 (strict label assertion on missing entries) and D19 (strict on conflicting duplicates) together enforce that the label CSV is treated as ground truth: any ambiguity in the CSV blocks the run, never silently flips downstream labels. The LOGHUB-distributed `anomaly_label.csv` is expected clean; D19 is defensive for adversarial / manually edited CSVs.
- **Verification command** (run once on the LOGHUB-distributed file before first use):
  ```python
  import pandas as pd
  df = pd.read_csv("anomaly_label.csv")
  assert not df.duplicated(subset=["BlockId"]).any(), "Duplicate block_ids in CSV"
  ```

### D20. Encoding violation handling (NEW v1.3)

- **Paper says:** Unspecified.
- **Decision:** Files are read with `errors="replace"` (preserved from v1.2 behaviour). v1.3 adds:
  1. A counter `encoding_replacements_seen` that increments any time a line contains `\ufffd` (the UTF-8 replacement character).
  2. The first 100 offending line numbers persisted to `preparation_summary.json.encoding_offending_line_numbers`.
  3. A supervisor warning printed if `encoding_replacements_seen / total_lines > 1e-4` (i.e., > 0.01% of lines).
- **Rationale:** v1.2 silently dropped corrupted lines into the existing `no_blockid` / `unparseable_timestamp` counters, with no way to triage encoding-induced vs legitimate parse failures. v1.3 makes encoding corruption auditable. Per §2.4 dissertation disclosure.
- **Alternative considered:** Switch to `errors="strict"` and HARD FAIL on any invalid UTF-8 byte. Rejected because real published log datasets occasionally contain non-UTF-8 bytes (e.g., from kernel-level message buffers); HARD FAIL would block runs on datasets the original paper presumably processed.

---

## 5. Open Questions for Supervisor (REVISED v1.3 — Q8 added)

| # | Question | Default |
|---|---|---|
| Q1 | Confirm scope: HDFS + BGL + TB, all three windows? | All three |
| Q2 | Throughput: report or skip? | Report with hardware caveat |
| Q3 | F1 tolerance vs paper Tables III/IV/V? | ±0.02 F1 absolute |
| Q4 | Anomaly pool partitioning (D14): per-fold shuffle confirmed? | Per-fold shuffle |
| Q5 | Training stack: HF Trainer + OneCycleLR (D15) confirmed? | Confirmed |
| Q6 | HDFS F1 miss → run timestamp-strip sensitivity? | Yes |
| Q7 | Acceptable fp16 weight-divergence threshold? | 1e-4 |
| Q8 (NEW v1.3) | D20: errors="replace" with audit trail, or strict UTF-8 HARD FAIL? | replace + audit trail |

---

## 6. Validation Criteria (Unchanged stub — paper numbers transcription pending)

---

## 7. Filesystem Layout (REVISED v1.3 — `src/utils/stats.py` added)

```
~/logfit-repro/
├── docs/
│   ├── logfit-repro-decisions-v1.3.md     # this file
│   └── logfit-repro-spec-v1.3.md
├── src/
│   ├── types.py                            # REVISED v1.3 — new DropCounters fields
│   ├── utils/
│   │   ├── seed.py
│   │   ├── io.py
│   │   └── stats.py                        # NEW v1.3 [extracted from prepare_hdfs.py]
│   ├── prepare_hdfs.py                     # REVISED v1.3
│   └── prepare_bgl_tbird.py                # REVISED v1.3 [imports from stats.py]
├── tests/
│   ├── test_prepare_hdfs.py                # REVISED v1.3
│   ├── test_prepare_bgl_tbird.py           # REVISED v1.3
│   └── test_configs.py                     # NEW v1.3
├── configs/                                # unchanged
├── scripts/                                # unchanged
└── pyproject.toml                          # unchanged
```

---

## 8. Execution Sequence (Unchanged from v1.1)

---

## 9. Disclosure Standard (Unchanged from v1.0)

---

_LogFiT Reproduction Methodology Decisions v1.3_
_Post-code-review of preparation modules (2 convergent + 4 unique IMPORTANT/MINOR findings resolved)_
_Companion: logfit-repro-spec-v1.3.md_
_Oussama — Université Laval Cybersecurity Research Lab, May 2026_
