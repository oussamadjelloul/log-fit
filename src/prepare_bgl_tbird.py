"""BGL and Thunderbird preprocessing per spec Component 1.2.

Reads BGL.log or Thunderbird.log, groups lines into non-overlapping
time windows of configurable size, produces paragraphs.pkl +
preparation_summary.json.

Decisions implemented here:
- [D4]   Lines stripped of leading label field only; timestamp and remaining
         content kept verbatim (no field stripping for tokenization).
- [D5]   Window labeled anomaly if ANY constituent line has
         label_field != "-", otherwise normal. Standard log-anomaly convention
         (DeepLog, LogBERT, NeuralLog all use this).
- [L19]  Thunderbird input limited via max_input_lines parameter
         (configs/tbird_*.yaml sets 20,000,000).
- [T6 / R2 F15]  Timestamp parse failure rate > 0.1% triggers HARD FAIL.
- [§1.2 step 6]  Windows with < 2 lines dropped.

BGL/TB raw line format (first two fields are the only ones we parse):

    <label_field> <timestamp_unix> <date_str> <node> ... <message>

where label_field is "-" for normal, otherwise an alert category like
"APPSEV", "KERNDTLB", or similar. Everything after the leading label is
preserved verbatim as paragraph content.

Reference: logfit-repro-spec-v1.2.md Component 1.2; logfit-repro-decisions-v1.2.md D5.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

from src.prepare_hdfs import compute_length_distribution  # noqa: F401 — TODO: extract to src/utils/stats.py if a third prep module is added
from src.types import (
    DatasetName,
    DropCounters,
    Paragraph,
    PreparationSummary,
)
from src.utils.io import save_json


# [R2 F15 / T6] HARD FAIL if unparseable rate exceeds this fraction.
UNPARSEABLE_TIMESTAMP_FAIL_THRESHOLD = 0.001

# [Spec §1.2 step 6] Drop windows shorter than this number of lines.
MIN_LINES_PER_WINDOW = 2

# Normal label marker per BGL/TB convention. Any other value indicates an
# anomalous line; D5 aggregates the window label as OR of line labels.
NORMAL_LABEL_FIELD = "-"


def _parse_line(line: str) -> tuple[str, int, str] | None:
    """Parse a single BGL/TB line into (label_field, timestamp_unix, content).

    Returns None on parse failure (corrupt, empty, or non-conforming line).
    Expected format: `<label> <timestamp> <... rest ...>`

    Content [D4]: the line with the leading label_field and the whitespace
    immediately following it stripped. All other whitespace and structure
    is preserved verbatim. This means the timestamp is the first token
    of the returned content string.
    """
    stripped = line.lstrip()
    if not stripped:
        return None

    # Split limited to 2 separators so the third part keeps internal whitespace.
    parts = stripped.split(None, 2)
    if len(parts) < 2:
        return None

    label_field = parts[0]
    try:
        timestamp_unix = int(parts[1])
    except ValueError:
        return None

    # [D4] content = stripped line minus the label and its trailing whitespace.
    # We use slice-by-length rather than reconstructing via parts to preserve
    # the original tab/space structure between timestamp and subsequent fields.
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
    """Group BGL/Thunderbird log lines into non-overlapping time windows.

    Parameters
    ----------
    raw_log_path : path to raw BGL.log or Thunderbird.log
    dataset : "bgl" or "tbird"
    window_seconds : 10, 30, or 60 per paper §IV-A (others accepted with warning)
    output_dir : output directory for paragraphs.pkl + preparation_summary.json
    max_input_lines : cap on lines read from the file. Pass 20_000_000 for
        Thunderbird per L19. Pass None to read the entire file (BGL default).
    seed : carried through to PreparationSummary for downstream RNG anchoring

    Returns
    -------
    PreparationSummary with drop counters and length distributions populated.
    Note: token_length_distribution is None; populated later by §1.6 gate
    once the backbone is selected.

    Raises
    ------
    ValueError
        - dataset is not "bgl" or "tbird"
        - timestamp parse failure rate > 0.1% (spec §1.2 step 3, R2 F15)
    """
    if dataset not in ("bgl", "tbird"):
        raise ValueError(f"dataset must be 'bgl' or 'tbird', got {dataset!r}")

    if window_seconds not in (10, 30, 60):
        # Paper specifies 10/30/60 in §IV-A. Soft warning to allow exploratory
        # window sizes without blocking the run.
        print(
            f"WARNING: window_seconds={window_seconds} is not one of "
            f"{{10, 30, 60}} from paper §IV-A. Proceeding anyway."
        )

    if window_seconds <= 0:
        raise ValueError(
            f"window_seconds must be positive, got {window_seconds}"
        )

    raw_log_path = Path(raw_log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drop_counters = DropCounters()

    # window_id -> list of (label_field, content) tuples
    # Using a dict keyed by window_id (rather than sorted insertion) means
    # out-of-order timestamps still land in the correct window.
    windows_data: dict[int, list[tuple[str, str]]] = defaultdict(list)

    # window_id -> earliest timestamp seen in that window (used for metadata)
    window_start_ts: dict[int, int] = {}

    first_timestamp: int | None = None  # anchors all window IDs
    total_lines_read = 0

    with raw_log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            # [L19] enforce input cap (Thunderbird subset = 20M)
            if max_input_lines is not None and total_lines_read >= max_input_lines:
                break
            total_lines_read += 1

            line = raw_line.rstrip("\r\n")
            parsed = _parse_line(line)
            if parsed is None:
                drop_counters.unparseable_timestamp += 1
                continue

            label_field, timestamp_unix, content = parsed

            # Anchor on first valid timestamp; all subsequent windows
            # are computed relative to this.
            if first_timestamp is None:
                first_timestamp = timestamp_unix

            window_id = (timestamp_unix - first_timestamp) // window_seconds

            # Negative window_id can occur if a later-in-file line has an
            # EARLIER timestamp than first_timestamp. We allow this — the
            # algorithm is timestamp-anchored, not line-order-anchored.
            windows_data[window_id].append((label_field, content))

            if window_id not in window_start_ts:
                window_start_ts[window_id] = timestamp_unix
            elif timestamp_unix < window_start_ts[window_id]:
                # Track the EARLIEST timestamp seen in this window
                window_start_ts[window_id] = timestamp_unix

    # [Spec §1.2 step 3, R2 F15] HARD FAIL on excessive parse failures
    if total_lines_read > 0:
        unparseable_rate = drop_counters.unparseable_timestamp / total_lines_read
        if unparseable_rate > UNPARSEABLE_TIMESTAMP_FAIL_THRESHOLD:
            raise ValueError(
                f"HARD FAIL: timestamp parse failure rate is "
                f"{unparseable_rate * 100:.3f}% "
                f"({drop_counters.unparseable_timestamp} of {total_lines_read} "
                f"lines), exceeds {UNPARSEABLE_TIMESTAMP_FAIL_THRESHOLD * 100:.1f}% "
                f"threshold in {raw_log_path.name}. Spec §1.2 step 3."
            )

    # Build paragraphs in window_id order. window_id is timestamp-derived,
    # so this gives chronological ordering.
    paragraphs: list[Paragraph] = []
    sorted_window_ids = sorted(windows_data.keys())

    for window_id in sorted_window_ids:
        window_lines = windows_data[window_id]

        # [Spec §1.2 step 6] Drop windows with too few lines
        if len(window_lines) < MIN_LINES_PER_WINDOW:
            drop_counters.singleton_window += 1
            continue

        # [D5] Window is anomaly iff ANY constituent line has non-"-" label
        is_anomaly = any(
            lf != NORMAL_LABEL_FIELD for lf, _ in window_lines
        )

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
        token_length_distribution=None,  # populated later by §1.6 gate
        seed=seed,
    )

    paragraphs_path = output_dir / "paragraphs.pkl"
    with paragraphs_path.open("wb") as f:
        pickle.dump(paragraphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_json(summary, output_dir / "preparation_summary.json")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BGL/Thunderbird preprocessing per spec Component 1.2"
    )
    parser.add_argument(
        "--raw-log",
        type=Path,
        required=True,
        help="Path to BGL.log or Thunderbird.log",
    )
    parser.add_argument(
        "--dataset",
        choices=["bgl", "tbird"],
        required=True,
        help="Dataset identifier",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        required=True,
        choices=[10, 30, 60],
        help="Time window size in seconds (10, 30, or 60 per paper §IV-A)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for paragraphs.pkl + preparation_summary.json",
    )
    parser.add_argument(
        "--max-input-lines",
        type=int,
        default=None,
        help="Cap on lines read. Pass 20000000 for Thunderbird per L19.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
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
    print(f"\n{args.dataset.upper()} {args.window_seconds}s preprocessing complete.")
    print(f"  Total lines read:        {summary.total_lines_read:,}")
    print(f"  Total paragraphs:        {summary.total_paragraphs:,}")
    print(f"  Normal paragraphs:       {summary.normal_paragraphs:,}")
    print(f"  Anomaly paragraphs:      {summary.anomaly_paragraphs:,}")
    print(f"  Anomaly rate:            {summary.anomaly_rate * 100:.2f}%")
    print(
        f"  Unparseable timestamp:   "
        f"{summary.drop_counters.unparseable_timestamp:,}"
    )
    print(
        f"  Singleton windows dropped: "
        f"{summary.drop_counters.singleton_window:,}"
    )
    print(
        f"  Word length p50/p80/p95/p99: "
        f"{wld.p50:.0f} / {wld.p80:.0f} / {wld.p95:.0f} / {wld.p99:.0f}"
    )
