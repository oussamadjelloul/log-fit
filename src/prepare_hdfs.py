"""HDFS preprocessing per spec Component 1.1.

Reads HDFS.log + anomaly_label.csv, groups lines by block_id, produces
paragraphs.pkl + preparation_summary.json.

Decisions implemented here:
- [D16]   Lines with multiple block_ids are duplicated into each referenced
          block's paragraph (DeepLog convention). Rate surfaced via counter;
          SUPERVISOR_FLAG printed if rate exceeds 5%.
- [D4]    Raw lines kept as-published (no field stripping). Token-length
          guard is applied later by src/token_length_gate.py, not here.
- [BUG-2] First-appearance offset sort, not lexicographic on block_id strings.
- [BUG-3] Strict label assertion: a block_id with no entry in the CSV is
          a HARD FAIL, never silent default-to-NORMAL.

Reference: logfit-repro-spec-v1.2.md Component 1.1; logfit-repro-decisions-v1.2.md D16.
"""

from __future__ import annotations

import csv
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.types import (
    DropCounters,
    LengthDistribution,
    Paragraph,
    PreparationSummary,
)
from src.utils.io import save_json


# Regex matches negative and positive block IDs.
# HDFS log format: "...block blk_-1608999687919862906 src:..." or "blk_1234567"
BLOCK_ID_RE = re.compile(r"blk_-?\d+")

# Threshold above which multi-blockid rate triggers SUPERVISOR_FLAG (see D16).
LINES_WITH_MULTIPLE_BLOCKIDS_FLAG_THRESHOLD = 0.05


def load_label_dict(label_csv_path: Path | str) -> dict[str, int]:
    """Parse anomaly_label.csv into {block_id: 0|1}.

    Expected CSV columns: BlockId, Label (Label in {"Normal", "Anomaly"}).
    Whitespace in column headers is tolerated; unexpected label values raise.
    """
    label_csv_path = Path(label_csv_path)
    label_map: dict[str, int] = {}

    with label_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"{label_csv_path} appears to have no header row"
            )

        # Defensive column lookup (handle stray whitespace in header)
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

        for row in reader:
            block_id = row[block_id_field].strip()
            label_str = row[label_field].strip()
            if label_str not in ("Normal", "Anomaly"):
                raise ValueError(
                    f"Unexpected label value {label_str!r} for {block_id} "
                    f"in {label_csv_path}. Expected 'Normal' or 'Anomaly'."
                )
            label_map[block_id] = 0 if label_str == "Normal" else 1

    return label_map


def compute_length_distribution(values: list[int]) -> LengthDistribution:
    """Percentile summary of integer lengths.

    Returns all zeros when the input is empty (guards against
    np.percentile on an empty array, which warns).
    """
    if not values:
        return LengthDistribution(min=0, max=0, p50=0.0, p80=0.0, p95=0.0, p99=0.0)
    arr = np.asarray(values)
    return LengthDistribution(
        min=int(arr.min()),
        max=int(arr.max()),
        p50=float(np.percentile(arr, 50)),
        p80=float(np.percentile(arr, 80)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


def prepare_hdfs(
    raw_log_path: Path | str,
    label_csv_path: Path | str,
    output_dir: Path | str,
    seed: int = 42,
) -> PreparationSummary:
    """Group HDFS log lines into paragraphs by block_id.

    Implements spec Component 1.1 algorithm. Returns the PreparationSummary
    and writes paragraphs.pkl + preparation_summary.json to output_dir.

    Memory note: HDFS.log is ~11M lines. This implementation holds all
    grouped lines in memory simultaneously (~2-3 GB peak). On systems with
    less than 8 GB free RAM, subset the log first or stream-and-persist
    (not implemented in v1.2).
    """
    raw_log_path = Path(raw_log_path)
    label_csv_path = Path(label_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_dict(label_csv_path)
    drop_counters = DropCounters()
    paragraphs_lines: dict[str, list[str]] = defaultdict(list)
    first_appearance: dict[str, int] = {}  # block_id -> earliest line number

    total_lines = 0

    with raw_log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, raw_line in enumerate(f):
            total_lines += 1
            line = raw_line.rstrip("\r\n")

            block_ids = BLOCK_ID_RE.findall(line)

            if not block_ids:
                drop_counters.no_blockid += 1
                continue

            if len(block_ids) > 1:
                drop_counters.lines_with_multiple_blockids += 1

            # Deduplicate within a single line (preserves first-mention order).
            # A line that mentions blk_X twice should be added to blk_X once.
            unique_block_ids = list(dict.fromkeys(block_ids))

            for block_id in unique_block_ids:
                # [BUG-3] Strict label assertion — no silent fallback to NORMAL.
                if block_id not in label_map:
                    drop_counters.missing_label_assertion_fired += 1
                    raise ValueError(
                        f"HARD FAIL: block_id {block_id!r} on line "
                        f"{line_num + 1} of {raw_log_path.name} has no "
                        f"entry in {label_csv_path.name}. Strict label "
                        f"assertion per spec Section 1.1 step 4c."
                    )

                paragraphs_lines[block_id].append(line)
                if block_id not in first_appearance:
                    first_appearance[block_id] = line_num

    # [BUG-2] First-appearance sort, not lexicographic on block_id strings.
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

    # [D16] Surface high multi-blockid rate as a supervisor concern.
    if total_lines > 0:
        multi_rate = drop_counters.lines_with_multiple_blockids / total_lines
        if multi_rate > LINES_WITH_MULTIPLE_BLOCKIDS_FLAG_THRESHOLD:
            print(
                f"SUPERVISOR_FLAG: HDFS multi-blockid rate is "
                f"{multi_rate * 100:.2f}% (threshold "
                f"{LINES_WITH_MULTIPLE_BLOCKIDS_FLAG_THRESHOLD * 100:.0f}%). "
                f"See decisions-v1.2 D16 disclosure."
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
        token_length_distribution=None,  # populated later by §1.6 gate
        seed=seed,
    )

    # Persist
    paragraphs_path = output_dir / "paragraphs.pkl"
    with paragraphs_path.open("wb") as f:
        pickle.dump(paragraphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_json(summary, output_dir / "preparation_summary.json")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HDFS preprocessing per spec Section 1.1 (D16 multi-blockid)"
    )
    parser.add_argument(
        "--raw-log",
        type=Path,
        required=True,
        help="Path to HDFS.log",
    )
    parser.add_argument(
        "--label-csv",
        type=Path,
        required=True,
        help="Path to anomaly_label.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for paragraphs.pkl + preparation_summary.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    summary = prepare_hdfs(
        raw_log_path=args.raw_log,
        label_csv_path=args.label_csv,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    wld = summary.word_length_distribution
    print("\nHDFS preprocessing complete.")
    print(f"  Total lines read:        {summary.total_lines_read:,}")
    print(f"  Total paragraphs:        {summary.total_paragraphs:,}")
    print(f"  Normal paragraphs:       {summary.normal_paragraphs:,}")
    print(f"  Anomaly paragraphs:      {summary.anomaly_paragraphs:,}")
    print(f"  Anomaly rate:            {summary.anomaly_rate * 100:.2f}%")
    print(f"  Lines without block_id:  {summary.drop_counters.no_blockid:,}")
    print(
        f"  Lines with multi-blockid: "
        f"{summary.drop_counters.lines_with_multiple_blockids:,}"
    )
    print(
        f"  Word length p50/p80/p95/p99: "
        f"{wld.p50:.0f} / {wld.p80:.0f} / {wld.p95:.0f} / {wld.p99:.0f}"
    )
