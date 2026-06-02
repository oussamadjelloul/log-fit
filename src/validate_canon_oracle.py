"""Validate canon paragraphs against the authors' grouped-labeled oracle.

Proves our canonicalization matches logfit-project/hdfsv1-grouped-labeled
BYTE-FOR-BYTE on every shared block_id (the full-set version of the sandbox
3500/3500 check), and cross-checks labels on shared blocks.

Run from the project root (needs src.types to unpickle Paragraph):
  python -m src.validate_canon_oracle \
      --canon-paragraphs data/processed/hdfs_canon_full/paragraphs.pkl \
      --oracle-parquet data/raw/hdfsv1_grouped_labeled_hf

Oracle download (LOGIN node, offline mode disabled for the command only):
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download \
      logfit-project/hdfsv1-grouped-labeled --repo-type dataset \
      --revision refs/convert/parquet \
      --local-dir $SCRATCH/log-fit/data/raw/hdfsv1_grouped_labeled_hf

Exit 0 = all shared blocks byte-exact AND labels agree AND every oracle block is
present in canon; exit 1 = any mismatch.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def load_canon_paragraphs(path: Path | str):
    """Return (text_by_block, label_by_block) from a canon paragraphs.pkl.

    `text` is the newline-join of Paragraph.lines — the representation that
    matches the oracle's `text` column.
    """
    with open(path, "rb") as f:
        paragraphs = pickle.load(f)
    text = {p.paragraph_id: "\n".join(p.lines) for p in paragraphs}
    label = {p.paragraph_id: int(p.label) for p in paragraphs}
    return text, label


def load_oracle(parquet_path: Path | str):
    """Return (text_by_block, label_by_block) from the grouped-labeled parquet."""
    import pandas as pd

    from src.prepare_hdfs_canon import _resolve_parquet_files

    text: dict[str, str] = {}
    label: dict[str, int] = {}
    for shard in _resolve_parquet_files(parquet_path):
        df = pd.read_parquet(shard, columns=["block_id", "text", "anomaly"])
        for bid, t, a in zip(df["block_id"], df["text"], df["anomaly"]):
            text[str(bid)] = str(t)
            label[str(bid)] = int(a)
    return text, label


def compare_to_oracle(canon_text, canon_label, oracle_text, oracle_label) -> dict:
    """Pure comparison over shared block_ids. Returns a report dict."""
    canon_ids = set(canon_text)
    oracle_ids = set(oracle_text)
    shared = sorted(canon_ids & oracle_ids)
    return {
        "n_canon": len(canon_ids),
        "n_oracle": len(oracle_ids),
        "n_shared": len(shared),
        "oracle_not_in_canon": sorted(oracle_ids - canon_ids),
        "text_mismatch": [b for b in shared if canon_text[b] != oracle_text[b]],
        "label_mismatch": [b for b in shared if canon_label[b] != oracle_label[b]],
    }


def _first_diff_line(a: str, b: str):
    al, bl = a.split("\n"), b.split("\n")
    for i, (x, y) in enumerate(zip(al, bl)):
        if x != y:
            return i, x, y
    if len(al) != len(bl):
        return min(len(al), len(bl)), f"<{len(al)} lines>", f"<{len(bl)} lines>"
    return -1, "", ""


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate canon paragraphs vs the authors' grouped-labeled "
        "oracle (byte-exact text + labels)."
    )
    ap.add_argument("--canon-paragraphs", type=Path, required=True)
    ap.add_argument("--oracle-parquet", type=Path, required=True,
                    help="logfit-project/hdfsv1-grouped-labeled parquet (dir/glob/file).")
    ap.add_argument("--show", type=int, default=3, help="Mismatch examples to print.")
    args = ap.parse_args()

    canon_text, canon_label = load_canon_paragraphs(args.canon_paragraphs)
    oracle_text, oracle_label = load_oracle(args.oracle_parquet)
    r = compare_to_oracle(canon_text, canon_label, oracle_text, oracle_label)

    print("=== canon vs grouped-labeled oracle ===")
    print(f"  canon blocks:          {r['n_canon']:,}")
    print(f"  oracle blocks:         {r['n_oracle']:,}")
    print(f"  shared blocks:         {r['n_shared']:,}")
    print(f"  oracle NOT in canon:   {len(r['oracle_not_in_canon']):,}  (must be 0)")
    print(f"  text mismatches:       {len(r['text_mismatch']):,}  (must be 0)")
    print(f"  label mismatches:      {len(r['label_mismatch']):,}  (must be 0)")

    for bid in r["text_mismatch"][: args.show]:
        i, x, y = _first_diff_line(canon_text[bid], oracle_text[bid])
        print(f"\n  TEXT MISMATCH {bid} @ line {i}")
        print(f"    canon : {x[:160]}")
        print(f"    oracle: {y[:160]}")
    for bid in r["label_mismatch"][: args.show]:
        print(f"  LABEL MISMATCH {bid}: canon={canon_label[bid]} oracle={oracle_label[bid]}")
    for bid in r["oracle_not_in_canon"][: args.show]:
        print(f"  ORACLE-ONLY block (missing from canon): {bid}")

    ok = not (r["text_mismatch"] or r["label_mismatch"] or r["oracle_not_in_canon"])
    print("\nRESULT:", "PASS — canon provably matches the authors on all shared blocks."
          if ok else "FAIL — see mismatches above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _main()
