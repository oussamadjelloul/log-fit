"""HDFS canonicalized preprocessing — companion to prepare_hdfs.py.

Builds a CANONICALIZED paragraphs.pkl from the full HDFS_v1 line-level parquet
(logfit-project/HDFS_v1 on Hugging Face), replicating the volatile-field
normalization the LogFiT authors used in their published `hdfsv1-grouped-labeled`
dataset. This is the *canon arm* of the canon-vs-raw ablation; the raw arm is
the untouched src/prepare_hdfs.py.

Single-variable contract: every decision EXCEPT the per-line text representation
is inherited from the raw pipeline by IMPORT, not copy:
  - block grouping : src.prepare_hdfs.BLOCK_ID_RE (same word-bounded regex)
  - label source   : src.prepare_hdfs.load_label_dict (anomaly_label.csv)
  - schema         : src.types.Paragraph / PreparationSummary / DropCounters
  - stats / IO     : src.utils.stats / src.utils.io
The ONLY change is that each Paragraph.line is
  "<level> <component> <canonicalized content>"   (header dropped)
instead of the verbatim raw line.

Label cross-check [user decision]: the CSV label for every block must equal the
block-level max-pool of HDFS_v1's line-level `anomaly`. Mismatch => HARD FAIL
(override with --allow-label-mismatch: keeps the CSV label and warns).

Input is the LOCAL HDFS_v1 parquet (a directory of shards, a glob, or a file).
Download once on a Narval LOGIN node (compute nodes are offline):
  hf download logfit-project/HDFS_v1 --repo-type dataset --local-dir $SCRATCH/log-fit/data/raw/HDFS_v1_hf

Reference: prepare_hdfs.py (raw arm); src/utils/canonicalize.py (validated
recipe, 3500/3500 byte-exact vs the authors' oracle).
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from glob import glob
from pathlib import Path

import pandas as pd

from src.prepare_hdfs import BLOCK_ID_RE, load_label_dict
from src.types import DropCounters, Paragraph, PreparationSummary
from src.utils.canonicalize import canonicalize_content
from src.utils.io import save_json
from src.utils.stats import compute_length_distribution

_PARQUET_COLUMNS = ["line_number", "level", "component", "content", "anomaly"]


def _resolve_parquet_files(path: Path | str) -> list[Path]:
    """Return a sorted list of parquet files from a dir, glob, or single file."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.rglob("*.parquet"))
    elif any(ch in str(p) for ch in "*?["):
        files = sorted(Path(x) for x in glob(str(p)))
    elif p.is_file():
        files = [p]
    else:
        files = []
    if not files:
        raise FileNotFoundError(f"No parquet files found at {path!r}")
    return files


def prepare_hdfs_canon(
    hdfs_v1_parquet: Path | str,
    label_csv_path: Path | str,
    output_dir: Path | str,
    seed: int = 42,
    fail_on_label_mismatch: bool = True,
) -> PreparationSummary:
    """Group canonicalized HDFS_v1 lines into paragraphs by block_id."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map, duplicate_count = load_label_dict(label_csv_path)

    drop = DropCounters()
    drop.duplicate_blockid = duplicate_count

    block_lines: dict[str, list[tuple[int, str]]] = defaultdict(list)
    first_appearance: dict[str, int] = {}
    block_parquet_label: dict[str, int] = {}
    # content -> (block_ids, canonicalized_content); content repeats massively.
    content_cache: dict[str, tuple[list[str], str]] = {}
    total_lines = 0

    for shard in _resolve_parquet_files(hdfs_v1_parquet):
        df = pd.read_parquet(shard, columns=_PARQUET_COLUMNS)
        for row in df.itertuples(index=False):
            total_lines += 1
            content = "" if row.content is None else str(row.content)

            cached = content_cache.get(content)
            if cached is None:
                bids = BLOCK_ID_RE.findall(content)
                cached = (bids, canonicalize_content(content))
                content_cache[content] = cached
            block_ids, ccontent = cached

            if not block_ids:
                drop.no_blockid += 1
                continue
            unique_block_ids = list(dict.fromkeys(block_ids))
            if len(unique_block_ids) > 1:
                drop.lines_with_multiple_blockids += 1

            cline = f"{row.level} {row.component} {ccontent}"
            ln_i = int(row.line_number)
            a = int(row.anomaly)
            for block_id in unique_block_ids:
                if block_id not in label_map:
                    drop.missing_label_assertion_fired += 1
                    raise ValueError(
                        f"HARD FAIL: block_id {block_id!r} (line {ln_i}) has no "
                        f"entry in {Path(label_csv_path).name}. Strict label "
                        f"assertion (mirrors prepare_hdfs BUG-3)."
                    )
                block_lines[block_id].append((ln_i, cline))
                prev = first_appearance.get(block_id)
                if prev is None or ln_i < prev:
                    first_appearance[block_id] = ln_i
                if a > block_parquet_label.get(block_id, 0):
                    block_parquet_label[block_id] = a
        del df

    # ----- label cross-check: CSV vs parquet max-pool [user decision] -----
    mismatches = [
        (b, label_map[b], block_parquet_label.get(b, 0))
        for b in block_lines
        if label_map[b] != block_parquet_label.get(b, 0)
    ]
    n_mismatch = len(mismatches)
    if mismatches:
        head = ", ".join(f"{b}(csv={c},parquet={p})" for b, c, p in mismatches[:10])
        msg = (
            f"{n_mismatch} block(s) where anomaly_label.csv disagrees with "
            f"HDFS_v1 parquet (block = max-pool of line-level anomaly). "
            f"First 10: {head}"
        )
        if fail_on_label_mismatch:
            raise ValueError("HARD FAIL: " + msg)
        print("WARNING: " + msg + " — CSV label kept (--allow-label-mismatch).")

    # ----- build paragraphs in first-appearance order (mirrors prepare_hdfs) -
    sorted_block_ids = sorted(block_lines, key=lambda b: first_appearance[b])
    paragraphs: list[Paragraph] = []
    for block_id in sorted_block_ids:
        lines = [cl for _, cl in sorted(block_lines[block_id], key=lambda t: t[0])]
        paragraphs.append(
            Paragraph(
                paragraph_id=block_id,
                lines=lines,
                label=label_map[block_id],
                source_blockid=block_id,
            )
        )

    word_lengths = [sum(len(l.split()) for l in p.lines) for p in paragraphs]
    word_length_dist = compute_length_distribution(word_lengths)
    normal_count = sum(1 for p in paragraphs if p.label == 0)
    anomaly_count = sum(1 for p in paragraphs if p.label == 1)

    summary = PreparationSummary(
        dataset="hdfs",
        window_seconds=None,
        total_lines_read=total_lines,
        total_paragraphs=len(paragraphs),
        normal_paragraphs=normal_count,
        anomaly_paragraphs=anomaly_count,
        anomaly_rate=(anomaly_count / len(paragraphs)) if paragraphs else 0.0,
        drop_counters=drop,
        word_length_distribution=word_length_dist,
        token_length_distribution=None,
        seed=seed,
        encoding_offending_line_numbers=[],
    )

    with (output_dir / "paragraphs.pkl").open("wb") as f:
        pickle.dump(paragraphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_json(summary, output_dir / "preparation_summary.json")

    summary._n_label_mismatch = n_mismatch  # type: ignore[attr-defined]
    return summary


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Canonicalized HDFS preprocessing (canon arm; replicates the "
        "logfit-project canonicalization on full HDFS_v1)."
    )
    ap.add_argument("--hdfs-v1-parquet", type=Path, required=True,
                    help="Local HDFS_v1 parquet: directory of shards, glob, or file.")
    ap.add_argument("--label-csv", type=Path, required=True,
                    help="anomaly_label.csv (same file the raw arm uses).")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-label-mismatch", action="store_true",
                    help="Downgrade CSV-vs-parquet label mismatch from HARD FAIL "
                         "to a warning (CSV label kept). Default: hard fail.")
    args = ap.parse_args()

    summary = prepare_hdfs_canon(
        hdfs_v1_parquet=args.hdfs_v1_parquet,
        label_csv_path=args.label_csv,
        output_dir=args.output_dir,
        seed=args.seed,
        fail_on_label_mismatch=not args.allow_label_mismatch,
    )

    wld = summary.word_length_distribution
    n_mm = getattr(summary, "_n_label_mismatch", 0)
    print("\nHDFS canonicalized preprocessing complete.")
    print(f"  Total lines read:        {summary.total_lines_read:,}")
    print(f"  Total paragraphs:        {summary.total_paragraphs:,}")
    print(f"  Normal paragraphs:       {summary.normal_paragraphs:,}")
    print(f"  Anomaly paragraphs:      {summary.anomaly_paragraphs:,}")
    print(f"  Anomaly rate:            {summary.anomaly_rate * 100:.2f}%")
    print(f"  No-blockid lines:        {summary.drop_counters.no_blockid:,}")
    print(f"  Multi-blockid lines:     {summary.drop_counters.lines_with_multiple_blockids:,}")
    print(f"  Duplicate CSV rows:      {summary.drop_counters.duplicate_blockid:,}")
    print(f"  CSV<->parquet mismatches: {n_mm:,} (0 = sources agree)")
    print(f"  Word length p50/p80/p95/p99: "
          f"{wld.p50:.0f} / {wld.p80:.0f} / {wld.p95:.0f} / {wld.p99:.0f}")


if __name__ == "__main__":
    _main()
