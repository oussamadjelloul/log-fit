"""Anomaly Temporal Distribution Experiment.

For BGL / Thunderbird time-windowed logs, this analyzes WHERE within
each window the anomaly lines appear. Answers: under right-truncation
at the backbone's token limit, what fraction of anomaly signal is
discarded vs preserved?

Standalone — re-reads raw BGL.log with the same windowing logic as
prepare_bgl_tbird.py, tracking per-line label positions (which prep
discards since it stores only paragraph-level labels).

Reference: logfit-repro-decisions-v1.4 Q_NEW (truncation policy).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_line(line):
    """Parse BGL/TB line -> (label_field, timestamp_unix). None on failure."""
    stripped = line.lstrip()
    if not stripped:
        return None
    parts = stripped.split(None, 2)
    if len(parts) < 3:
        return None
    try:
        ts = int(parts[1])
    except ValueError:
        return None
    return parts[0], ts


def collect_windows(raw_log_path, window_seconds, max_lines=None):
    """Group lines by window. Returns window_id -> [(line_idx_in_window, label), ...]."""
    windows = defaultdict(list)
    first_ts = None
    lines_so_far = defaultdict(int)

    with raw_log_path.open("r", encoding="utf-8", errors="replace") as f:
        for i, raw_line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            parsed = parse_line(raw_line)
            if parsed is None:
                continue
            label, ts = parsed
            if first_ts is None:
                first_ts = ts
            wid = (ts - first_ts) // window_seconds
            windows[wid].append((lines_so_far[wid], label))
            lines_so_far[wid] += 1

    return windows


def analyze(windows):
    first_pos = []
    all_pos = []
    density = []
    lengths = []
    NORMAL = "-"
    MIN_LINES = 2

    for wid, lines in windows.items():
        n = len(lines)
        if n < MIN_LINES:
            continue
        anom_idxs = [li for li, lbl in lines if lbl != NORMAL]
        if not anom_idxs:
            continue
        first_pos.append(anom_idxs[0] / n)
        for idx in anom_idxs:
            all_pos.append(idx / n)
        density.append(len(anom_idxs) / n)
        lengths.append(n)

    def hist(positions, n_bins=10):
        b = [0] * n_bins
        for p in positions:
            b[min(int(p * n_bins), n_bins - 1)] += 1
        return b

    def pct_under(positions, threshold):
        if not positions:
            return 0.0
        return sum(1 for p in positions if p <= threshold) / len(positions)

    def stats(positions):
        if not positions:
            return {"p25": 0.0, "p50": 0.0, "p75": 0.0, "mean": 0.0}
        return {
            "p25": float(np.percentile(positions, 25)),
            "p50": float(np.percentile(positions, 50)),
            "p75": float(np.percentile(positions, 75)),
            "mean": float(np.mean(positions)),
        }

    return {
        "anomaly_windows_analyzed": len(first_pos),
        "total_anomaly_lines": len(all_pos),
        "first_anomaly_position": {
            **stats(first_pos),
            "hist_10bins_0to1": hist(first_pos),
            "pct_in_first_10pct": pct_under(first_pos, 0.10),
            "pct_in_first_25pct": pct_under(first_pos, 0.25),
        },
        "all_anomaly_positions": {
            **stats(all_pos),
            "hist_10bins_0to1": hist(all_pos),
            "pct_in_first_10pct": pct_under(all_pos, 0.10),
            "pct_in_first_25pct": pct_under(all_pos, 0.25),
        },
        "anomaly_density_per_window": {
            "mean": float(np.mean(density)) if density else 0.0,
            "p50": float(np.median(density)) if density else 0.0,
        },
        "anomaly_window_length_lines": {
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "p50": float(np.median(lengths)) if lengths else 0.0,
            "max": int(max(lengths)) if lengths else 0,
        },
    }


def print_hist(bins, label, w=40):
    mx = max(bins) if bins else 1
    print(f"\n  {label}:")
    for i, c in enumerate(bins):
        lo, hi = i / 10, (i + 1) / 10
        bar = "#" * int(w * c / mx) if mx > 0 else ""
        print(f"  [{lo:.1f}-{hi:.1f}]  {c:6d}  {bar}")


def print_summary(r, ws):
    print(f"\n=== Anomaly Temporal Distribution ({ws}s windows) ===")
    print(f"\nAnomaly windows analyzed:  {r['anomaly_windows_analyzed']:,}")
    print(f"Total anomaly lines:       {r['total_anomaly_lines']:,}")

    fap = r["first_anomaly_position"]
    print(f"\n--- FIRST anomaly position per window ---")
    print(f"  p25={fap['p25']:.3f}  p50={fap['p50']:.3f}  "
          f"p75={fap['p75']:.3f}  mean={fap['mean']:.3f}")
    print(f"  First-anomaly in first 10% of window:  {fap['pct_in_first_10pct']*100:.1f}%")
    print(f"  First-anomaly in first 25% of window:  {fap['pct_in_first_25pct']*100:.1f}%")
    print_hist(fap["hist_10bins_0to1"], "First-anomaly distribution (0=start, 1=end)")

    aap = r["all_anomaly_positions"]
    print(f"\n--- ALL anomaly positions ---")
    print(f"  p25={aap['p25']:.3f}  p50={aap['p50']:.3f}  "
          f"p75={aap['p75']:.3f}  mean={aap['mean']:.3f}")
    print(f"  All anomalies in first 10% of window:  {aap['pct_in_first_10pct']*100:.1f}%")
    print(f"  All anomalies in first 25% of window:  {aap['pct_in_first_25pct']*100:.1f}%")
    print_hist(aap["hist_10bins_0to1"], "All-anomaly distribution")

    den = r["anomaly_density_per_window"]
    wl = r["anomaly_window_length_lines"]
    print(f"\n--- Context ---")
    print(f"  Anomaly density: mean={den['mean']:.3f}, p50={den['p50']:.3f}")
    print(f"  Window length (lines): mean={wl['mean']:.1f}, p50={wl['p50']:.1f}, max={wl['max']}")

    p50 = fap["p50"]
    print(f"\n--- Interpretation for §1.6 truncation policy ---")
    if p50 < 0.10:
        v = "STRONGLY EARLY"
        impl = ("Right-truncation likely defensible — most signal in first 10% of window, "
                "preserved under truncation to ~5-30% of content.")
        rec = "Option A (paper-faithful) is acceptable."
    elif p50 < 0.30:
        v = "EARLY"
        impl = ("Right-truncation preserves SUBSTANTIAL signal but loses some. "
                "Impact depends on how aggressively each paragraph is truncated.")
        rec = "Option A partially defensible. Consider Option G sensitivity check."
    elif p50 < 0.70:
        v = "DISTRIBUTED / MIDDLE"
        impl = ("Right-truncation systematically discards anomaly signal. "
                "Gemini's mathematical concern (Q7) is empirically grounded.")
        rec = "Option A is methodologically weak. Strong recommendation for Option G."
    else:
        v = "LATE"
        impl = ("Right-truncation discards almost ALL anomaly signal. Model evaluates "
                "only the 'normal preamble' of each anomalous window. Degenerate task.")
        rec = "Option A is indefensible. Option G is the only valid path."

    print(f"  Verdict:        Anomalies cluster {v}")
    print(f"  Implication:    {impl}")
    print(f"  Recommendation: {rec}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-log", type=Path, required=True)
    p.add_argument("--window-seconds", type=int, default=30)
    p.add_argument("--max-lines", type=int, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    a = p.parse_args()

    windows = collect_windows(a.raw_log, a.window_seconds, a.max_lines)
    r = analyze(windows)
    r["_meta"] = {
        "raw_log": str(a.raw_log),
        "window_seconds": a.window_seconds,
        "max_lines": a.max_lines,
    }

    if a.output_json:
        a.output_json.parent.mkdir(parents=True, exist_ok=True)
        with a.output_json.open("w") as f:
            json.dump(r, f, indent=2)
        print(f"Wrote {a.output_json}")

    print_summary(r, a.window_seconds)


if __name__ == "__main__":
    main()
