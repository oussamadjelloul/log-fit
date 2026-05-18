# LogFiT Reproduction — Technique & Methodology Specification

**Version:** 1.3 (post-code-review of preparation modules)
**Status:** Draft for supervisor sign-off (paired with `logfit-repro-decisions-v1.3.md`)
**Author:** Oussama
**Source paper:** Almodovar et al., IEEE TNSM 21(2):1715–1723, 2024. DOI: 10.1109/TNSM.2024.3358730
**Companion doc:** `logfit-repro-decisions-v1.3.md`

## Changelog v1.2 → v1.3

Post tri-LLM adversarial review of prep code. 2 convergent + 4 unique IMPORTANT/MINOR findings + 6 test-coverage gaps resolved. v1.2 contract preserved; v1.3 is implementation precision + disclosure tightening.

- **Component 1.1 (REVISED)** — HDFS algorithm step 4d: counter logic now uses `len(unique_block_ids) > 1` (distinct-id semantics per **D16 v1.3 / convergent R1 F1 + R2 F3**). Step 1 (label CSV parsing): duplicate `block_id` detection added per **D19 NEW**. Regex updated to `r"\bblk_-?\d+\b"` per **R2 F4**. Encoding detection added per **D20 NEW**.
- **Component 1.2 (REVISED)** — BGL/TB algorithm step 1: parse failure now requires `len(parts) >= 3` (label + timestamp + content) per **D4 v1.3 / R2 F6**. Step 6 (singleton drop): counter split into `singleton_window_normal` + `singleton_window_anomaly` per **R2 F5**. Encoding detection added per **D20 NEW**.
- **Component 1.5 (REVISED)** — `DropCounters` schema gains `lines_missing_content`, `encoding_replacements_seen`, split singleton counters. `PreparationSummary` gains `encoding_offending_line_numbers: list[int]` (capped at first 100).
- **Component 8.1 (Unchanged from v1.2)** — `strip_trailing_punct` definition.
- **Component 12 (REVISED)** — new module `src/utils/stats.py` houses `compute_length_distribution` (extracted from `prepare_hdfs.py` per **convergent R1 F3 + R2 F8**).
- **Component 13 (REVISED)** — test suite gains 10 new tests addressing convergent findings + R2 test gaps F9–F12 + F14.

All other components (2 backbone selection, 3 masking, 4 training, 5 scoring, 6 threshold tuning, 7 evaluation, 8 variability, 9 throughput, 10 YAML schema, 11 outputs, 14 reproducibility checklist) **unchanged from v1.2**. Reference v1.2 spec for those sections.

---

## Component 0: Notation and Definitions (Unchanged from v1.1)

---

## Component 1: Data Preparation

### 1.1 HDFS Pipeline (REVISED v1.3 — D16 distinct-id semantics + D19 + D20 + regex word boundary)

**Inputs:** `HDFS.log` (raw), `anomaly_label.csv` (block_id → Normal/Anomaly).
**Outputs:** `paragraphs.pkl` (list of `Paragraph` dataclasses), `preparation_summary.json`.

**Algorithm:**

1. **Parse `anomaly_label.csv`** (per **D19 v1.3**):
   - Expected exact-case headers `BlockId`, `Label` (whitespace-tolerant via `.strip()`; not case-tolerant — R2 F13 verified LOGHUB distribution uses `BlockId,Label`).
   - For each row, look up `block_id` in accumulator dict:
     - **First time seen:** record `label_map[block_id] = 0|1`.
     - **Seen before, same label:** accept silently and increment `drop_counters.duplicate_blockid`.
     - **Seen before, conflicting label:** raise `ValueError("HARD FAIL: conflicting duplicate label for ...")`.
   - Reject rows with `Label` outside `{Normal, Anomaly}` with `ValueError`.

2. **Initialize counters:** `drop_counters = DropCounters()` (new v1.3 fields zero-initialized per Component 1.5).

3. **Initialize containers:**
   - `paragraphs_lines: dict[str, list[str]] = defaultdict(list)`
   - `first_appearance: dict[str, int] = {}` (block_id → earliest line number)

4. **Stream `HDFS.log` line by line:**
   - For each line at index `line_num`:
     - **a. Encoding check (NEW v1.3 — D20):** If `"\ufffd" in line`:
       - `drop_counters.encoding_replacements_seen += 1`
       - If `len(encoding_offending_line_numbers) < 100`: append `line_num`.
     - **b. Extract block_ids** via `BLOCK_ID_RE.findall(line)` where `BLOCK_ID_RE = re.compile(r"\bblk_-?\d+\b")` (NEW v1.3 — word-bounded regex per R2 F4).
     - **c. No block_ids found:** `drop_counters.no_blockid += 1`; continue.
     - **d. Dedup within line:** `unique_block_ids = list(dict.fromkeys(block_ids))`.
     - **e. Counter increment (CORRECTED v1.3 — D16):** `if len(unique_block_ids) > 1: drop_counters.lines_with_multiple_blockids += 1` (distinct-id semantics, not occurrence count).
     - **f. For each `block_id in unique_block_ids`:**
       - **Strict label assertion (BUG-3):** if `block_id not in label_map`: `drop_counters.missing_label_assertion_fired += 1`; raise `ValueError("HARD FAIL: ...")`.
       - Append `line` to `paragraphs_lines[block_id]`.
       - If `block_id not in first_appearance`: set `first_appearance[block_id] = line_num`.

5. **First-appearance sort (BUG-2):** `sorted_block_ids = sorted(paragraphs_lines.keys(), key=lambda b: first_appearance[b])`.

6. **Build `Paragraph` objects** in sorted order.

7. **Compute distributions:** `word_length_distribution` via `src.utils.stats.compute_length_distribution`.

8. **Supervisor warnings:**
   - **Multi-blockid rate** (D16): if `lines_with_multiple_blockids / total_lines > 0.05`, print `SUPERVISOR_FLAG`.
   - **Encoding rate (NEW v1.3 — D20):** if `encoding_replacements_seen / total_lines > 1e-4`, print `SUPERVISOR_FLAG`.

9. **Persist:** `paragraphs.pkl` (pickle, protocol HIGHEST_PROTOCOL); `preparation_summary.json` (via `save_json`, includes `encoding_offending_line_numbers` capped at first 100).

### 1.2 BGL / Thunderbird Pipeline (REVISED v1.3 — D4 3-field reject + D20 + singleton anomaly split)

**Inputs:** `BGL.log` or `Thunderbird.log` (raw).
**Outputs:** `paragraphs.pkl`, `preparation_summary.json`.

**Algorithm:**

1. **Parse each line** via `_parse_line` (REVISED v1.3 — D4):
   - `stripped = line.lstrip()`; if empty, return None.
   - `parts = stripped.split(None, 2)`.
   - **If `len(parts) < 3`** (NEW v1.3 — D4 v1.3 / R2 F6): return None. Lines with only label + timestamp (no content) are parse failures.
   - `label_field = parts[0]`.
   - Try `timestamp_unix = int(parts[1])`; on `ValueError`, return None.
   - `content = stripped[len(label_field):].lstrip()` (D4 delimiter stripping).
   - Return `(label_field, timestamp_unix, content)`.

2. **Counter increments for parse failures:**
   - `None` return from `_parse_line` → `drop_counters.unparseable_timestamp += 1` (NOTE: misnomer retained for backward compat; reservation per D4 v1.3 — 2-field lines tracked separately via `lines_missing_content` if disambiguation needed downstream).
   - **Encoding check (NEW v1.3 — D20):** same logic as HDFS step 4a, parallel counter + offending_line_numbers.

3. **Stream-grouping loop** (Unchanged structurally from v1.2):
   - Anchor on first valid timestamp.
   - `window_id = (timestamp_unix - first_timestamp) // window_seconds`.
   - Append `(label_field, content)` to `windows_data[window_id]`.

4. **HARD FAIL check (Unchanged):** if `unparseable_rate > 0.001`, raise `ValueError("HARD FAIL: ...")`.

5. **Build paragraphs in window order:**
   - For each window in `sorted(windows_data.keys())`:
     - **If `len(window_lines) < 2`** (Spec §1.2 step 6 — REVISED v1.3 — R2 F5):
       - Inspect the *label* of the singleton line:
         - If `label_field == "-"`: `drop_counters.singleton_window_normal += 1`.
         - Else: `drop_counters.singleton_window_anomaly += 1`.
       - Continue (drop window).
     - `is_anomaly = any(lf != "-" for lf, _ in window_lines)` (D5).
     - Build `Paragraph(...)`.

6. **Supervisor warnings:**
   - **Singleton anomaly drops (NEW v1.3 — R2 F5):** if `singleton_window_anomaly > 0`, print `SUPERVISOR_FLAG` (any anomaly drop is suspicious).
   - **Encoding rate (NEW v1.3 — D20):** same as HDFS.

7. **Persist:** same format as HDFS.

### 1.3 Sample Allocation (Unchanged from v1.1)
### 1.4 5-Fold Split Construction (Unchanged from v1.2 — D14)

### 1.5 Persistence Schema (REVISED v1.3 — new counter fields + encoding offending line numbers)

```python
@dataclass
class DropCounters:
    # HDFS counters
    no_blockid: int = 0
    missing_label_assertion_fired: int = 0           # must stay 0 (HARD FAIL above this)
    duplicate_blockid: int = 0                        # NEW semantics per D19 v1.3
    lines_with_multiple_blockids: int = 0             # distinct-id semantics per D16 v1.3

    # BGL/TB counters
    unparseable_timestamp: int = 0
    empty_window: int = 0
    singleton_window_normal: int = 0                  # NEW v1.3 (was: singleton_window)
    singleton_window_anomaly: int = 0                 # NEW v1.3 (split per R2 F5)

    # Shared (NEW v1.3)
    lines_missing_content: int = 0                    # 2-token lines per D4 v1.3
    encoding_replacements_seen: int = 0               # \ufffd occurrences per D20


@dataclass
class PreparationSummary:
    dataset: DatasetName
    window_seconds: int | None
    total_lines_read: int
    total_paragraphs: int
    normal_paragraphs: int
    anomaly_paragraphs: int
    anomaly_rate: float
    drop_counters: DropCounters
    word_length_distribution: LengthDistribution
    token_length_distribution: TokenLengthSummary | None = None
    seed: int = 42
    encoding_offending_line_numbers: list[int] = field(default_factory=list)   # NEW v1.3, capped at first 100
```

### 1.6 Token-Length Validation Gate (Unchanged from v1.2)

---

## Component 2: Backbone Selection (Unchanged from v1.1)
## Component 3: Masking (Unchanged from v1.1)
## Component 4: Training Procedure (Unchanged from v1.2)
## Component 5: Anomaly Scoring (Unchanged from v1.2)
## Component 6: Threshold Tuning (Unchanged from v1.1)
## Component 7: Evaluation (Unchanged from v1.1)
## Component 8: Variability Test (Unchanged from v1.2)
## Component 9: Throughput Measurement (Unchanged from v1.1)
## Component 10: Run Configuration Schema (Unchanged from v1.2)
## Component 11: Output Artifacts (Unchanged from v1.2)

---

## Component 12: API Surface (REVISED v1.3 — `src/utils/stats.py` added)

In addition to v1.2 API:

```python
# src/utils/stats.py — NEW v1.3 (extracted from prepare_hdfs.py)
def compute_length_distribution(values: list[int]) -> LengthDistribution: ...
```

Both `src/prepare_hdfs.py` and `src/prepare_bgl_tbird.py` import `compute_length_distribution` from `src.utils.stats` in v1.3 (resolves convergent R1 F3 + R2 F8).

---

## Component 13: Test Suite (REVISED v1.3 — 10 new tests)

All v1.2 tests preserved EXCEPT `test_within_line_dedup` which is corrected (see below). New tests:

| Test | Asserts | Module | Mapping |
|---|---|---|---|
| `test_within_line_dedup` (CORRECTED v1.3) | Same-block-twice → counter is 0 (was: 1) | `prepare_hdfs.py` | D16 v1.3 / convergent R1+R2 |
| `test_distinct_blockids_increments_counter` (NEW) | True multi-block line → counter +1 | `prepare_hdfs.py` | D16 v1.3 |
| `test_duplicate_blockid_in_label_csv_raises_on_conflict` (NEW) | Conflicting CSV labels → HARD FAIL | `prepare_hdfs.py` | D19 NEW |
| `test_duplicate_blockid_in_label_csv_allows_exact_duplicates` (NEW) | Same label twice → accepted + counter +1 | `prepare_hdfs.py` | D19 NEW |
| `test_word_boundary_regex_rejects_prefix_match` (NEW) | `prefixblk_123` → no match | `prepare_hdfs.py` | R2 F4 |
| `test_first_appearance_tie_break_deterministic` (NEW) | Multi-blockid line tie-break is first-mention order | `prepare_hdfs.py` | R2 F10 |
| `test_supervisor_flag_fires_above_5_percent` (NEW) | capsys captures `SUPERVISOR_FLAG` print | `prepare_hdfs.py` | R2 F11 |
| `test_supervisor_flag_silent_below_5_percent` (NEW) | capsys verifies no print | `prepare_hdfs.py` | R2 F11 |
| `test_total_paragraphs_zero_handling` (NEW) | All-skip input → empty pkl + summary anomaly_rate=0 | `prepare_hdfs.py` | R2 F12 |
| `test_encoding_replacement_detected_hdfs` (NEW) | `\ufffd` in line → counter +1, offset captured | `prepare_hdfs.py` | D20 NEW |
| `test_singleton_anomaly_tracked_separately` (NEW) | Singleton anomaly → `singleton_window_anomaly += 1` | `prepare_bgl_tbird.py` | R2 F5 |
| `test_two_token_line_rejected` (NEW) | `"- 1000"` (no content) → parse failure | `prepare_bgl_tbird.py` | D4 v1.3 / R2 F6 |
| `test_negative_window_id_handled` (NEW) | First valid line has high ts; later lines get negative window_id | `prepare_bgl_tbird.py` | R2 F9 |
| `test_total_paragraphs_zero_when_all_singletons` (NEW) | Every window singleton → empty output | `prepare_bgl_tbird.py` | R2 F12 |
| `test_encoding_replacement_detected_bgl` (NEW) | `\ufffd` in line → counter +1 | `prepare_bgl_tbird.py` | D20 NEW |
| `tests/test_configs.py` (NEW MODULE) | Validates `configs/tbird_*.yaml` set `max_input_lines=20000000`; all configs have consistent schema | `configs/` | R2 F14 |

Test integrity: every BLOCKING/IMPORTANT/MINOR finding from the code review maps to at least one regression test.

---

## Component 14: Reproducibility Checklist (Unchanged from v1.2)

---

_LogFiT Reproduction Technique & Methodology Specification v1.3_
_Post-code-review of preparation modules; companion to logfit-repro-decisions-v1.3.md_
_Oussama — Université Laval Cybersecurity Research Lab, May 2026_
