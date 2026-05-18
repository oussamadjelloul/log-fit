"""BGL and Thunderbird preprocessing per spec Component 1.2.

Reads BGL.log or Thunderbird.log, groups lines into non-overlapping time
windows, produces paragraphs.pkl + preparation_summary.json.

Decisions implemented (v1.3):
- [D4 v1.3]   Raw line content stripped of leading label only; timestamp and
              remainder retained verbatim. Lines with < 3 whitespace-delimited
              tokens (label + timestamp + content required) rejected. Surfaced
              via `lines_missing_content` counter (R2 F6).
- [D5]        Window labeled anomaly iff ANY constituent line has
              label_field != "-".
- [L19]       Thunderbird input limited via max_input_lines parameter
              (configs/tbird_*.yaml sets 20,000,000).
- [D20 NEW v1.3]  Encoding violations (\\ufffd in line) tracked via
                   `encoding_replacements_seen` + first 100 line numbers in
                   `encoding_offending_line_numbers`.
- [R2 F5 / v1.3]  Singleton windows split by label.
- [T6 / R2 F15]   Timestamp parse failure rate > 0.1% triggers HARD FAIL.
- [Spec section 1.2 step 6]   Windows with < 2 lines dropped.

Reference: logfit-repro-spec-v1.3.md Component 1.2.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

from src.types import (
    DatasetName,
    DropCounters,
    Paragraph,
    PreparationSummary,
)
from src.utils.io import save_json
from src.utils.stats import compute_length_distribution

UNPARSEABLE_TIMESTAMP_FAIL_THRESHOLD = 0.001
MIN_LINES_PER_WINDOW = 2
NORMAL_LABEL_FIELD = "-"
ENCODING_REPLACEMENT_FLAG_THRESHOLD = 1e-4
ENCODING_OFFENDING_LINES_CAP = 100
UTF8_REPLACEMENT_CHAR = "\ufffd"


def _parse_line(line: str) -> tuple[str, int, str] | None:
    """Parse a BGL/TB line into (label_field, timestamp_unix, content).

    Returns None on parse failure. v1.3 [D4 / R2 F6]: requires
    `len(parts) >= 3` (label + timestamp + content).

    Content [D4 v1.3]: line with leading label and post-label whitespace
    delimiter stripped; everything else preserved verbatim.
    """
    stripped = line.lstrip()
    if not stripped:
        return None

    parts = stripped.split(None, 2)
    if len(parts) < 3:
        return None

    label_field = parts[0]
    try:
        timestamp_unix = int(parts[1])
    except ValueError:
        return None

    content = stripped[len(label_field):].lstrip()
    return label_field, timestamp_unix, content


def prepare_bgl_tbird(
    raw_log_path: Path | str,
    dataset: DatasetName,
    window_seconds: int,
    output_dir: Path | str,
    max_input_lines: int | None = None,
    seed: int = 42,
) -> PreparationSummary:
    """Group BGL/Thunderbird log lines into non-overlapping time windows."""
    if dataset not in ("bgl", "tbird"):
        raise ValueError(f"dataset must be 'bgl' or 'tbird', got {dataset!r}")
    if window_seconds <= 0:
        raise ValueError(
            f"window_seconds must be positive, got {window_seconds}"
        )
    if window_seconds not in (10, 30, 60):
        print(
            f"WARNING: window_seconds={window_seconds} is not in "
            f"{{10, 30, 60}} from paper section IV-A. Proceeding anyway."
        )

    raw_log_path = Path(raw_log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drop_counters = DropCounters()
    windows_data: dict[int, list[tuple[str, str]]] = defaultdict(list)
    window_start_ts: dict[int, int] = {}
    encoding_offending_line_numbers: list[int] = []
    first_timestamp: int | None = None
    total_lines_read = 0

    with raw_log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, raw_line in enumerate(f):
            if max_input_lines is not None and total_lines_read >= max_input_lines:
                break
            total_lines_read += 1
            line = raw_line.rstrip("\r\n")

            if UTF8_REPLACEMENT_CHAR in line:
                drop_counters.encoding_replacements_seen += 1
                if len(encoding_offending_line_numbers) < ENCODING_OFFENDING_LINES_CAP:
                    encoding_offending_line_numbers.append(line_num + 1)

            parsed = _parse_line(line)
            if parsed is None:
                stripped = line.lstrip()
                parts = stripped.split(None, 2) if stripped else []
                if len(parts) == 2:
                    drop_counters.lines_missing_content += 1
                else:
                    drop_counters.unparseable_timestamp += 1
                continue

            label_field, timestamp_unix, content = parsed

            if first_timestamp is None:
                first_timestamp = timestamp_unix

            window_id = (timestamp_unix - first_timestamp) // window_seconds
            windows_data[window_id].append((label_field, content))

            if window_id not in window_start_ts:
                window_start_ts[window_id] = timestamp_unix
            elif timestamp_unix < window_start_ts[window_id]:
                window_start_ts[window_id] = timestamp_unix

    if total_lines_read > 0:
        unparseable_rate = (
            drop_counters.unparseable_timestamp + drop_counters.lines_missing_content
        ) / total_lines_read
        if unparseable_rate > UNPARSEABLE_TIMESTAMP_FAIL_THRESHOLD:
            raise ValueError(
                f"HARD FAIL: line parse failure rate is "
                f"{unparseable_rate * 100:.3f}% "
                f"({drop_counters.unparseable_timestamp} bad-timestamp + "
                f"{drop_counters.lines_missing_content} missing-content "
                f"of {total_lines_read}), exceeds "
                f"{UNPARSEABLE_TIMESTAMP_FAIL_THRESHOLD * 100:.1f}% "
                f"threshold in {raw_log_path.name}. Spec section 1.2."
            )

    paragraphs: list[Paragraph] = []
    sorted_window_ids = sorted(windows_data.keys())

    for window_id in sorted_window_ids:
        window_lines = windows_data[window_id]

        if len(window_lines) < MIN_LINES_PER_WINDOW:
            singleton_is_anomaly = any(
                lf != NORMAL_LABEL_FIELD for lf, _ in window_lines
            )
            if singleton_is_anomaly:
                drop_counters.singleton_window_anomaly += 1
            else:
                drop_counters.singleton_window_normal += 1
            continue

        is_anomaly = any(lf != NORMAL_LABEL_FIELD for lf, _ in window_lines)

        paragraphs.append(
            Paragraph(
                paragraph_id=f"{dataset}_{window_seconds}s_w{window_id}",
                lines=[content for _, content in window_lines],
                label=1 if is_anomaly else 0,
                source_window_id=int(window_id),
                start_timestamp=float(window_start_ts[window_id]),
            )
        )

    word_lengths = [
        sum(len(line.split()) for line in p.lines) for p in paragraphs
    ]
    word_length_dist = compute_length_distribution(word_lengths)

    normal_count = sum(1 for p in paragraphs if p.label == 0)
    anomaly_count = sum(1 for p in paragraphs if p.label == 1)

    if drop_counters.singleton_window_anomaly > 0:
        print(
            f"SUPERVISOR_FLAG: {drop_counters.singleton_window_anomaly} "
            f"anomalous singleton window(s) dropped. Anomaly count may be "
            f"under-reported. See decisions-v1.3 R2 F5."
        )

    if total_lines_read > 0:
        encoding_rate = drop_counters.encoding_replacements_seen / total_lines_read
        if encoding_rate > ENCODING_REPLACEMENT_FLAG_THRESHOLD:
            print(
                f"SUPERVISOR_FLAG: {dataset.upper()} encoding replacement rate "
                f"is {encoding_rate * 100:.4f}% "
                f"(threshold {ENCODING_REPLACEMENT_FLAG_THRESHOLD * 100:.2f}%). "
                f"See decisions-v1.3 D20."
            )

    summary = PreparationSummary(
        dataset=dataset,
        window_seconds=window_seconds,
        total_lines_read=total_lines_read,
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
        description="BGL/Thunderbird preprocessing per spec Component 1.2 (v1.3)"
    )
    parser.add_argument("--raw-log", type=Path, required=True)
    parser.add_argument("--dataset", choices=["bgl", "tbird"], required=True)
    parser.add_argument("--window-seconds", type=int, required=True, choices=[10, 30, 60])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-input-lines", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = prepare_bgl_tbird(
        raw_log_path=args.raw_log,
        dataset=args.dataset,
        window_seconds=args.window_seconds,
        output_dir=args.output_dir,
        max_input_lines=args.max_input_lines,
        seed=args.seed,
    )

    wld = summary.word_length_distribution
    print(f"\n{args.dataset.upper()} {args.window_seconds}s preprocessing complete (v1.3).")
    print(f"  Total lines read:           {summary.total_lines_read:,}")
    print(f"  Total paragraphs:           {summary.total_paragraphs:,}")
    print(f"  Normal paragraphs:          {summary.normal_paragraphs:,}")
    print(f"  Anomaly paragraphs:         {summary.anomaly_paragraphs:,}")
    print(f"  Anomaly rate:               {summary.anomaly_rate * 100:.2f}%")
    print(f"  Unparseable timestamps:     {summary.drop_counters.unparseable_timestamp:,}")
    print(f"  Lines missing content:      {summary.drop_counters.lines_missing_content:,}")
    print(
        f"  Singleton normal dropped:   "
        f"{summary.drop_counters.singleton_window_normal:,}"
    )
    print(
        f"  Singleton anomaly dropped:  "
        f"{summary.drop_counters.singleton_window_anomaly:,} (flagged if > 0)"
    )
    print(
        f"  Encoding replacements:      "
        f"{summary.drop_counters.encoding_replacements_seen:,} "
        f"(first {len(summary.encoding_offending_line_numbers)} offsets captured)"
    )
    print(
        f"  Word length p50/p80/p95/p99: "
        f"{wld.p50:.0f} / {wld.p80:.0f} / {wld.p95:.0f} / {wld.p99:.0f}"
    )
