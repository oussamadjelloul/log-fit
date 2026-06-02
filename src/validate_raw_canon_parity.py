"""Confirm the raw and canon arms differ ONLY in per-line text.

Loads both paragraphs.pkl files and asserts:
  - identical block_id SET (same population),
  - identical label for every block,
  - identical per-block LINE COUNT (canonicalization is per-line 1:1: each raw
    line maps to exactly one canonical line, header dropped but no lines added
    or removed).
If all three hold, the canon-vs-raw F1 delta isolates the text representation.

Run from the project root:
  python -m src.validate_raw_canon_parity \
      --raw-paragraphs   data/processed/hdfs_raw_full/paragraphs.pkl \
      --canon-paragraphs data/processed/hdfs_canon_full/paragraphs.pkl

Exit 0 = arms are parity-equal (single-variable); exit 1 = any difference.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def load_block_index(path: Path | str) -> dict[str, tuple[int, int]]:
    """Return {block_id: (label, n_lines)} from a paragraphs.pkl."""
    with open(path, "rb") as f:
        paragraphs = pickle.load(f)
    return {p.paragraph_id: (int(p.label), len(p.lines)) for p in paragraphs}


def compare_parity(raw: dict, canon: dict) -> dict:
    """Pure comparison. Returns a report dict."""
    raw_ids = set(raw)
    canon_ids = set(canon)
    shared = sorted(raw_ids & canon_ids)
    return {
        "n_raw": len(raw_ids),
        "n_canon": len(canon_ids),
        "raw_only": sorted(raw_ids - canon_ids),
        "canon_only": sorted(canon_ids - raw_ids),
        "label_mismatch": [b for b in shared if raw[b][0] != canon[b][0]],
        "linecount_mismatch": [
            (b, raw[b][1], canon[b][1]) for b in shared if raw[b][1] != canon[b][1]
        ],
    }


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Confirm raw and canon arms differ ONLY in text "
        "(same blocks, labels, and per-block line counts)."
    )
    ap.add_argument("--raw-paragraphs", type=Path, required=True)
    ap.add_argument("--canon-paragraphs", type=Path, required=True)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    raw = load_block_index(args.raw_paragraphs)
    canon = load_block_index(args.canon_paragraphs)
    r = compare_parity(raw, canon)

    print("=== raw vs canon parity (single-variable check) ===")
    print(f"  raw blocks:            {r['n_raw']:,}")
    print(f"  canon blocks:          {r['n_canon']:,}")
    print(f"  raw-only blocks:       {len(r['raw_only']):,}  (must be 0)")
    print(f"  canon-only blocks:     {len(r['canon_only']):,}  (must be 0)")
    print(f"  label mismatches:      {len(r['label_mismatch']):,}  (must be 0)")
    print(f"  line-count mismatches: {len(r['linecount_mismatch']):,}  (must be 0)")

    for b in r["raw_only"][: args.show]:
        print(f"  RAW-ONLY:   {b}")
    for b in r["canon_only"][: args.show]:
        print(f"  CANON-ONLY: {b}")
    for b in r["label_mismatch"][: args.show]:
        print(f"  LABEL MM:   {b} raw={raw[b][0]} canon={canon[b][0]}")
    for b, rn, cn in r["linecount_mismatch"][: args.show]:
        print(f"  LINES MM:   {b} raw={rn} canon={cn}")

    ok = not (r["raw_only"] or r["canon_only"] or r["label_mismatch"] or r["linecount_mismatch"])
    print("\nRESULT:", "PASS — arms differ only in per-line text; comparison is single-variable."
          if ok else "FAIL — see differences above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _main()
