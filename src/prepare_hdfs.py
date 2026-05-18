"""HDFS preprocessing per spec Component 1.1.

Reads HDFS.log + anomaly_label.csv, groups lines by block_id, produces
paragraphs.pkl + preparation_summary.json.

Decisions implemented:
- [D4]    Raw lines kept verbatim (no field stripping). Token-length guard
          deferred to src/token_length_gate.py.
- [D16 v1.3]  Multi-block-id lines duplicated into each DISTINCT block's
              paragraph. Counter `lines_with_multiple_blockids` uses
              distinct-id semantics (post-dedup count > 1), not occurrence
              count. Word-bounded regex `\\bblk_-?\\d+\\b` rejects substring
              matches like `prefixblk_123`.
- [D19 NEW v1.3]  Duplicate `block_id` in label CSV: HARD FAIL on conflicting
                   labels; accept exact duplicates with `duplicate_blockid`
                   counter increment.
- [D20 NEW v1.3]  Encoding violations (`\\ufffd` in line) tracked via
                   `encoding_replacements_seen` counter + first 100 line
                   numbers in `encoding_offending_line_numbers`. Supervisor
                   warning fires above 0.01% rate.
- [BUG-2]  First-appearance offset sort, not lexicographic.
- [BUG-3]  Strict label assertion: missing block_id → HARD FAIL.

Reference: logfit-repro-spec-v1.3.md Component 1.1.
"""

from __future__ import annotations

import csv
import pickle
import re
from collections import defaultdict
from pathlib import Path

from src.types import (
    DropCounters,
    Paragraph,
    PreparationSummary,
)
from src.utils.io import save_json
from src.utils.stats import compute_length_distribution

# [D16 v1.3 / R2 F4] Word-bounded regex; `prefixblk_123` no longer matches.
BLOCK_ID_RE = re.compile(r"\bblk_-?\d+\b")

# [D16] SUPERVISOR_FLAG threshold for distinct-id multi-blockid rate.
LINES_WITH_MULTIPLE_BLOCKIDS_FLAG_THRESHOLD = 0.05

# [D20 NEW v1.3] SUPERVISOR_FLAG threshold for encoding replacement rate.
ENCODING_REPLACEMENT_FLAG_THRESHOLD = 1e-4

# [D20 NEW v1.3] Cap for captured offending line numbers (audit trail).
ENCODING_OFFENDING_LINES_CAP = 100

# UTF-8 replacement character used by errors="replace" mode.
UTF8_REPLACEMENT_CHAR = "\ufffd"


def load_label_dict(label_csv_path: Path | str) -> tuple[dict[str, int], int]:
    """Parse anomaly_label.csv into ({block_id: 0|1}, duplicate_count).

    Expected exact-case headers: `BlockId`, `Label` (whitespace-tolerant via
    `.strip()`; not case-tolerant — LOGHUB-distributed file uses exact case).

    [D19 v1.3] Duplicate `block_id` handling:
    - First occurrence: recorded.
    - Subsequent occurrence with same label: accepted, counter incremented.
    - Subsequent occurrence with conflicting label: HARD FAIL.

    Returns
    -------
    (label_map, duplicate_count)
        label_map: {block_id: 0 if Normal else 1}
        duplicate_count: number of exact-duplicate rows accepted

    Raises
    ------
    ValueError
        - Missing/malformed columns.
        - Label value outside {Normal, Anomaly}.
        - Conflicting duplicate rows.
    """
    label_csv_path = Path(label_csv_path)
    label_map: dict[str, int] = {}
    duplicate_count = 0

    with label_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{label_csv_path} appears to have no header row")

        block_id_field = next(
            (fn for fn in reader.fieldnames if fn.strip() == "BlockId"), None
        )
        label_field = next(
            (fn for fn in reader.fieldnames if fn.strip() == "Label"), None
        )

        if block_id_field is None or label_field is None:
            raise ValueError(
                f"Expected columns 'BlockId' and 'Label' in {label_csv_path}, "
                f"got {reader.fieldnames}"
            )

        for row_num, row in enumerate(reader, start=2):  # row 1 is header
            block_id = row[block_id_field].strip()
            label_str = row[label_field].strip()
            if label_str not in ("Normal", "Anomaly"):
                raise ValueError(
                    f"Unexpected label value {label_str!r} for {block_id} "
                    f"at row {row_num} in {label_csv_path}. "
                    f"Expected 'Normal' or 'Anomaly'."
                )
            new_label = 0 if label_str == "Normal" else 1

            # [D19 v1.3] Duplicate detection
            if block_id in label_map:
                existing_label = label_map[block_id]
                if existing_label == new_label:
                    duplicate_count += 1
                    print(
                        f"WARNING: duplicate row for block_id {block_id!r} "
                        f"at row {row_num} of {label_csv_path.name} "
                        f"(same label {label_str!r}). Accepted."
                    )
                else:
                    raise ValueError(
                        f"HARD FAIL: conflicting duplicate label for "
                        f"block_id {block_id!r} in {label_csv_path.name}. "
                        f"Existing label: "
                        f"{'Normal' if existing_label == 0 else 'Anomaly'}; "
                        f"new label at row {row_num}: {label_str!r}. "
                        f"Per decisions-v1.3 D19."
                    )
            else:
                label_map[block_id] = new_label

    return label_map, duplicate_count


def prepare_hdfs(
    raw_log_path: Path | str,
    label_csv_path: Path | str,
    output_dir: Path | str,
    seed: int = 42,
) -> PreparationSummary:
    """Group HDFS log lines into paragraphs by block_id (spec Component 1.1)."""
    raw_log_path = Path(raw_log_path)
    label_csv_path = Path(label_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map, duplicate_count = load_label_dict(label_csv_path)

    drop_counters = DropCounters()
    drop_counters.duplicate_blockid = duplicate_count  # [D19]

    paragraphs_lines: dict[str, list[str]] = defaultdict(list)
    first_appearance: dict[str, int] = {}
    encoding_offending_line_numbers: list[int] = []
    total_lines = 0

    with raw_log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, raw_line in enumerate(f):
            total_lines += 1
            line = raw_line.rstrip("\r\n")

            # [D20 NEW v1.3] Encoding audit — record any line containing \ufffd
            if UTF8_REPLACEMENT_CHAR in line:
                drop_counters.encoding_replacements_seen += 1
                if len(encoding_offending_line_numbers) < ENCODING_OFFENDING_LINES_CAP:
                    encoding_offending_line_numbers.append(line_num + 1)  # 1-indexed

            block_ids = BLOCK_ID_RE.findall(line)

            if not block_ids:
                drop_counters.no_blockid += 1
                continue

            # [D16 v1.3] Dedup BEFORE counter increment (distinct-id semantics)
            unique_block_ids = list(dict.fromkeys(block_ids))

            if len(unique_block_ids) > 1:
                drop_counters.lines_with_multiple_blockids += 1

            for block_id in unique_block_ids:
                # [BUG-3] Strict label assertion
                if block_id not in label_map:
                    drop_counters.missing_label_assertion_fired += 1
                    raise ValueError(
                        f"HARD FAIL: block_id {block_id!r} on line "
                        f"{line_num + 1} of {raw_log_path.name} has no "
                        f"entry in {label_csv_path.name}. Strict label "
                        f"assertion per spec Section 1.1 step 4f."
                    )

                paragraphs_lines[block_id].append(line)
                if block_id not in first_appearance:
                    first_appearance[block_id] = line_num

    # [BUG-2] First-appearance sort
    sorted_block_ids = sorted(
        paragraphs_lines.keys(), key=lambda b: first_appearance[b]
    )

    paragraphs: list[Paragraph] = [
        Paragraph(
            paragraph_id=block_id,
            lines=paragraphs_lines[block_id],
            label=label_map[block_id],
            source_blockid=block_id,
        )
        for block_id in sorted_block_ids
    ]

    word_lengths = [
        sum(len(line.split()) for line in p.lines) for p in paragraphs
    ]
    word_length_dist = compute_length_distribution(word_lengths)

    normal_count = sum(1 for p in paragraphs if p.label == 0)
    anomaly_count = sum(1 for p in paragraphs if p.label == 1)

    # [D16] Multi-blockid SUPERVISOR_FLAG
    if total_lines > 0:
        multi_rate = drop_counters.lines_with_multiple_blockids / total_lines
        if multi_rate > LINES_WITH_MULTIPLE_BLOCKIDS_FLAG_THRESHOLD:
            print(
                f"SUPERVISOR_FLAG: HDFS multi-blockid rate is "
                f"{multi_rate * 100:.2f}% (threshold "
                f"{LINES_WITH_MULTIPLE_BLOCKIDS_FLAG_THRESHOLD * 100:.0f}%). "
                f"See decisions-v1.3 D16."
            )

        # [D20 NEW v1.3] Encoding SUPERVISOR_FLAG
        encoding_rate = drop_counters.encoding_replacements_seen / total_lines
        if encoding_rate > ENCODING_REPLACEMENT_FLAG_THRESHOLD:
            print(
                f"SUPERVISOR_FLAG: HDFS encoding replacement rate is "
                f"{encoding_rate * 100:.4f}% (threshold "
                f"{ENCODING_REPLACEMENT_FLAG_THRESHOLD * 100:.2f}%). "
                f"See decisions-v1.3 D20."
            )

    summary = PreparationSummary(
        dataset="hdfs",
        window_seconds=None,
        total_lines_read=total_lines,
        total_paragraphs=len(paragraphs),
        normal_paragraphs=normal_count,
        anomaly_paragraphs=anomaly_count,
        anomaly_rate=(anomaly_count / len(paragraphs)) if paragraphs else 0.0,
        drop_counters=drop_counters,
        word_length_distribution=word_length_dist,
        token_length_distribution=None,
        seed=seed,
        encoding_offending_line_numbers=encoding_offending_line_numbers,
    )

    paragraphs_path = output_dir / "paragraphs.pkl"
    with paragraphs_path.open("wb") as f:
        pickle.dump(paragraphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_json(summary, output_dir / "preparation_summary.json")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HDFS preprocessing per spec Section 1.1 (v1.3)"
    )
    parser.add_argument("--raw-log", type=Path, required=True)
    parser.add_argument("--label-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = prepare_hdfs(
        raw_log_path=args.raw_log,
        label_csv_path=args.label_csv,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    wld = summary.word_length_distribution
    print("\nHDFS preprocessing complete (v1.3).")
    print(f"  Total lines read:        {summary.total_lines_read:,}")
    print(f"  Total paragraphs:        {summary.total_paragraphs:,}")
    print(f"  Normal paragraphs:       {summary.normal_paragraphs:,}")
    print(f"  Anomaly paragraphs:      {summary.anomaly_paragraphs:,}")
    print(f"  Anomaly rate:            {summary.anomaly_rate * 100:.2f}%")
    print(f"  No-blockid lines:        {summary.drop_counters.no_blockid:,}")
    print(
        f"  Multi-blockid lines:     "
        f"{summary.drop_counters.lines_with_multiple_blockids:,} "
        f"(distinct-id semantics, D16 v1.3)"
    )
    print(
        f"  Duplicate CSV rows:      "
        f"{summary.drop_counters.duplicate_blockid:,} (exact dupes accepted, D19)"
    )
    print(
        f"  Encoding replacements:   "
        f"{summary.drop_counters.encoding_replacements_seen:,} "
        f"(first {len(summary.encoding_offending_line_numbers)} offsets captured)"
    )
    print(
        f"  Word length p50/p80/p95/p99: "
        f"{wld.p50:.0f} / {wld.p80:.0f} / {wld.p95:.0f} / {wld.p99:.0f}"
    )
