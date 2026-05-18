"""Tests for src/prepare_bgl_tbird.py — BGL/Thunderbird preprocessing.

Mapping to v1.2 spec Component 13:
- test_bgl_window_alignment              → windows non-overlapping, span exactly W seconds
- test_bgl_label_aggregation              → [D5] window anomaly iff ≥1 anomalous line
- test_bgl_timestamp_failure_threshold   → [R2 F15] >0.1% unparseable → HARD FAIL

Additional coverage:
- test_parse_line                        → _parse_line edge cases
- test_content_excludes_label             → [D4] label stripped, rest preserved
- test_thunderbird_max_lines_cap          → [L19] L19 cap enforced
- test_singleton_windows_dropped          → spec §1.2 step 6
- test_out_of_order_timestamps            → line lands in correct window
- test_invalid_dataset_raises             → guard on dataset param
- test_paragraph_id_format                → format is "{dataset}_{W}s_w{idx}"
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from src.prepare_bgl_tbird import (
    MIN_LINES_PER_WINDOW,
    NORMAL_LABEL_FIELD,
    UNPARSEABLE_TIMESTAMP_FAIL_THRESHOLD,
    _parse_line,
    prepare_bgl_tbird,
)
from src.types import Paragraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bgl_normal(ts: int, msg: str = "normal kernel message") -> str:
    """Build a BGL-style normal line."""
    return f"- {ts} 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL INFO {msg}"


def _bgl_anomaly(ts: int, category: str = "APPSEV", msg: str = "anomaly detected") -> str:
    """Build a BGL-style anomaly line."""
    return f"{category} {ts} 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL ERROR {msg}"


# ---------------------------------------------------------------------------
# _parse_line tests
# ---------------------------------------------------------------------------


class TestParseLine:
    def test_normal_line(self):
        line = _bgl_normal(1000)
        result = _parse_line(line)
        assert result is not None
        label, ts, content = result
        assert label == "-"
        assert ts == 1000
        assert content.startswith("1000 ")  # [D4] timestamp is first token of content

    def test_anomaly_line(self):
        line = _bgl_anomaly(1005, "APPSEV", "severe error")
        result = _parse_line(line)
        assert result is not None
        label, ts, content = result
        assert label == "APPSEV"
        assert ts == 1005
        assert content.startswith("1005 ")

    def test_empty_line_returns_none(self):
        assert _parse_line("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_line("   \t  ") is None

    def test_single_token_returns_none(self):
        assert _parse_line("-") is None

    def test_non_integer_timestamp_returns_none(self):
        assert _parse_line("- abc rest of line") is None

    def test_leading_whitespace_tolerated(self):
        result = _parse_line("  - 1000 message")
        assert result is not None
        label, ts, content = result
        assert label == "-"
        assert ts == 1000
        assert content.startswith("1000")

    def test_content_excludes_only_label(self):
        """[D4] Label field stripped; everything else preserved."""
        line = "APPSEV 1234 2005.06.03 NODE RAS KERNEL ERROR something happened"
        result = _parse_line(line)
        assert result is not None
        _, _, content = result
        # The label "APPSEV" is removed, everything else preserved
        assert content == "1234 2005.06.03 NODE RAS KERNEL ERROR something happened"
        assert "APPSEV" not in content


# ---------------------------------------------------------------------------
# Window alignment tests
# ---------------------------------------------------------------------------


class TestWindowAlignment:
    """[Spec §1.2 step 4] Windows are non-overlapping and span exactly W seconds.

    With W=10 and first_timestamp=1000:
      window 0: [1000, 1010)
      window 1: [1010, 1020)
      window 2: [1020, 1030)
    """

    def test_window_boundaries_10s(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        # Pack 3 lines per window for windows 0, 1, 2
        lines = (
            [_bgl_normal(ts) for ts in [1000, 1003, 1009]]   # window 0
            + [_bgl_normal(ts) for ts in [1010, 1015, 1019]]  # window 1
            + [_bgl_normal(ts) for ts in [1020, 1025, 1029]]  # window 2
        )
        _write_log(log_path, lines)

        summary = prepare_bgl_tbird(
            raw_log_path=log_path,
            dataset="bgl",
            window_seconds=10,
            output_dir=out_path,
        )

        assert summary.total_paragraphs == 3
        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs: list[Paragraph] = pickle.load(f)

        # Each window has 3 lines
        for p in paragraphs:
            assert len(p.lines) == 3

        # Window IDs are 0, 1, 2
        assert [p.source_window_id for p in paragraphs] == [0, 1, 2]

    def test_window_boundary_is_exclusive_upper(self, tmp_path: Path):
        """A line at timestamp 1010 belongs to window 1, not window 0,
        when W=10 and first_timestamp=1000."""
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        lines = [
            _bgl_normal(1000),
            _bgl_normal(1005),
            _bgl_normal(1009),   # last line of window 0
            _bgl_normal(1010),   # first line of window 1
            _bgl_normal(1015),
        ]
        _write_log(log_path, lines)

        prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)

        # Window 0 has 3 lines, window 1 has 2 lines
        windows = {p.source_window_id: p for p in paragraphs}
        assert len(windows[0].lines) == 3
        assert len(windows[1].lines) == 2


# ---------------------------------------------------------------------------
# Label aggregation tests [D5]
# ---------------------------------------------------------------------------


class TestLabelAggregationD5:
    """[D5] Window is labeled anomaly iff ANY constituent line has
    label_field != '-'."""

    def test_all_normal_window_is_normal(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        lines = [_bgl_normal(ts) for ts in [1000, 1005, 1009]]
        _write_log(log_path, lines)

        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.normal_paragraphs == 1
        assert summary.anomaly_paragraphs == 0

    def test_single_anomaly_makes_window_anomaly(self, tmp_path: Path):
        """[D5] Even ONE anomalous line flips the window to anomaly."""
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        lines = [
            _bgl_normal(1000),
            _bgl_normal(1003),
            _bgl_anomaly(1007, "APPSEV"),  # one anomaly among normals
            _bgl_normal(1009),
        ]
        _write_log(log_path, lines)

        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.anomaly_paragraphs == 1
        assert summary.normal_paragraphs == 0

    def test_all_anomaly_window_is_anomaly(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        lines = [
            _bgl_anomaly(1000, "APPSEV"),
            _bgl_anomaly(1005, "KERNDTLB"),
            _bgl_anomaly(1009, "APPSEV"),
        ]
        _write_log(log_path, lines)

        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.anomaly_paragraphs == 1
        assert summary.normal_paragraphs == 0

    def test_mixed_windows(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        lines = (
            [_bgl_normal(ts) for ts in [1000, 1003, 1009]]              # normal window
            + [_bgl_anomaly(ts, "APPSEV") for ts in [1010, 1015, 1019]]  # anomaly window
            + [_bgl_normal(1020), _bgl_anomaly(1025), _bgl_normal(1029)]  # mixed -> anomaly
        )
        _write_log(log_path, lines)

        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.normal_paragraphs == 1
        assert summary.anomaly_paragraphs == 2


# ---------------------------------------------------------------------------
# Timestamp failure threshold [R2 F15 / T6]
# ---------------------------------------------------------------------------


class TestTimestampFailureThreshold:
    """[Spec §1.2 step 3, R2 F15] HARD FAIL when unparseable rate > 0.1%."""

    def test_below_threshold_no_failure(self, tmp_path: Path):
        """1 unparseable in 2000 lines = 0.05%, below 0.1% → no failure."""
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        # 1999 valid lines + 1 unparseable
        valid = [_bgl_normal(1000 + i // 100) for i in range(1999)]
        invalid = ["GARBAGE_LINE no timestamp here"]
        _write_log(log_path, valid + invalid)

        # Should succeed (0.05% < 0.1%)
        summary = prepare_bgl_tbird(log_path, "bgl", 60, out_path)
        assert summary.drop_counters.unparseable_timestamp == 1

    def test_above_threshold_hard_fails(self, tmp_path: Path):
        """11 unparseable in 1000 lines = 1.1%, above 0.1% → HARD FAIL."""
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        valid = [_bgl_normal(1000 + i // 10) for i in range(989)]
        invalid = ["GARBAGE no timestamp"] * 11
        _write_log(log_path, valid + invalid)

        with pytest.raises(ValueError, match="HARD FAIL.*timestamp parse"):
            prepare_bgl_tbird(log_path, "bgl", 60, out_path)

    def test_failure_threshold_value_matches_spec(self):
        """Sanity: threshold constant matches the spec's 0.1% value."""
        assert UNPARSEABLE_TIMESTAMP_FAIL_THRESHOLD == 0.001


# ---------------------------------------------------------------------------
# Singleton window handling [Spec §1.2 step 6]
# ---------------------------------------------------------------------------


class TestSingletonWindowsDropped:
    """[Spec §1.2 step 6] Windows with < MIN_LINES_PER_WINDOW lines are dropped."""

    def test_min_lines_per_window_value(self):
        assert MIN_LINES_PER_WINDOW == 2

    def test_singleton_window_dropped(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        lines = [
            _bgl_normal(1000),
            _bgl_normal(1005),
            _bgl_normal(1009),
            # Window 1 has only 1 line — should be dropped
            _bgl_normal(1015),
            # Window 2 has 2 lines — kept
            _bgl_normal(1020),
            _bgl_normal(1025),
        ]
        _write_log(log_path, lines)

        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.total_paragraphs == 2
        assert summary.drop_counters.singleton_window == 1

    def test_two_line_window_kept(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        # Exactly 2 lines — should be kept (at MIN threshold)
        lines = [_bgl_normal(1000), _bgl_normal(1005)]
        _write_log(log_path, lines)

        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.total_paragraphs == 1
        assert summary.drop_counters.singleton_window == 0


# ---------------------------------------------------------------------------
# Thunderbird L19 cap
# ---------------------------------------------------------------------------


class TestThunderbirdMaxLinesCap:
    """[L19] Thunderbird input capped at max_input_lines (configs set 20M)."""

    def test_cap_enforced(self, tmp_path: Path):
        log_path = tmp_path / "Thunderbird.log"
        out_path = tmp_path / "out"

        # Write 100 lines but cap at 50
        lines = [_bgl_normal(1000 + i // 10) for i in range(100)]
        _write_log(log_path, lines)

        summary = prepare_bgl_tbird(
            raw_log_path=log_path,
            dataset="tbird",
            window_seconds=10,
            output_dir=out_path,
            max_input_lines=50,
        )

        assert summary.total_lines_read == 50

    def test_no_cap_reads_all(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        lines = [_bgl_normal(1000 + i // 10) for i in range(100)]
        _write_log(log_path, lines)

        summary = prepare_bgl_tbird(
            raw_log_path=log_path,
            dataset="bgl",
            window_seconds=10,
            output_dir=out_path,
            max_input_lines=None,
        )

        assert summary.total_lines_read == 100


# ---------------------------------------------------------------------------
# Edge cases and parameter validation
# ---------------------------------------------------------------------------


class TestParameterValidation:
    def test_invalid_dataset_raises(self, tmp_path: Path):
        log_path = tmp_path / "log.txt"
        _write_log(log_path, [_bgl_normal(1000), _bgl_normal(1005)])

        with pytest.raises(ValueError, match="dataset must be"):
            prepare_bgl_tbird(
                raw_log_path=log_path,
                dataset="invalid",  # type: ignore[arg-type]
                window_seconds=10,
                output_dir=tmp_path / "out",
            )

    def test_zero_window_seconds_raises(self, tmp_path: Path):
        log_path = tmp_path / "log.txt"
        _write_log(log_path, [_bgl_normal(1000), _bgl_normal(1005)])

        with pytest.raises(ValueError, match="window_seconds must be positive"):
            prepare_bgl_tbird(
                raw_log_path=log_path,
                dataset="bgl",
                window_seconds=0,
                output_dir=tmp_path / "out",
            )

    def test_non_standard_window_warns_but_proceeds(self, tmp_path: Path, capsys):
        log_path = tmp_path / "log.txt"
        # Need enough lines to land in a single non-singleton window
        _write_log(log_path, [_bgl_normal(1000 + i) for i in range(5)])

        # window_seconds=15 is not 10/30/60; should warn but not raise
        summary = prepare_bgl_tbird(
            raw_log_path=log_path,
            dataset="bgl",
            window_seconds=15,
            output_dir=tmp_path / "out",
        )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert summary.total_paragraphs >= 1


# ---------------------------------------------------------------------------
# Out-of-order timestamps
# ---------------------------------------------------------------------------


class TestOutOfOrderTimestamps:
    """A line with an earlier timestamp than the previous line still lands
    in the correct (timestamp-anchored) window."""

    def test_late_line_routed_to_earlier_window(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        # first_timestamp = 1000 (from first line)
        # Window 0: [1000, 1010)  -> ts 1000, 1005
        # Window 1: [1010, 1020)  -> ts 1015
        # Out-of-order: a ts=1003 appears AFTER ts=1015 in the file
        lines = [
            _bgl_normal(1000),
            _bgl_normal(1005),
            _bgl_normal(1015),
            _bgl_normal(1003),  # later in file but earlier in time
            _bgl_normal(1019),
        ]
        _write_log(log_path, lines)

        prepare_bgl_tbird(log_path, "bgl", 10, out_path)

        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)

        windows = {p.source_window_id: p for p in paragraphs}
        # Window 0 has 3 lines (1000, 1005, 1003)
        assert len(windows[0].lines) == 3
        # Window 1 has 2 lines (1015, 1019)
        assert len(windows[1].lines) == 2


# ---------------------------------------------------------------------------
# Paragraph ID format
# ---------------------------------------------------------------------------


class TestParagraphIdFormat:
    """paragraph_id format: '{dataset}_{W}s_w{window_id}'."""

    def test_bgl_30s_format(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        lines = [_bgl_normal(1000), _bgl_normal(1005)]
        _write_log(log_path, lines)

        prepare_bgl_tbird(log_path, "bgl", 30, out_path)

        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)

        assert paragraphs[0].paragraph_id == "bgl_30s_w0"

    def test_tbird_60s_format(self, tmp_path: Path):
        log_path = tmp_path / "Thunderbird.log"
        out_path = tmp_path / "out"
        lines = [_bgl_normal(1000), _bgl_normal(1059)]
        _write_log(log_path, lines)

        prepare_bgl_tbird(log_path, "tbird", 60, out_path)

        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)

        assert paragraphs[0].paragraph_id == "tbird_60s_w0"


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


class TestSummaryJsonWritten:
    def test_dataset_and_window_in_json(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        _write_log(log_path, [_bgl_normal(1000), _bgl_normal(1005)])

        prepare_bgl_tbird(log_path, "bgl", 10, out_path)

        with (out_path / "preparation_summary.json").open("r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["dataset"] == "bgl"
        assert data["window_seconds"] == 10
        assert data["drop_counters"]["unparseable_timestamp"] == 0


# ---------------------------------------------------------------------------
# Normal label sentinel
# ---------------------------------------------------------------------------


class TestNormalLabelSentinel:
    """The normal-label sentinel is '-' per BGL/TB convention."""

    def test_sentinel_value(self):
        assert NORMAL_LABEL_FIELD == "-"

    def test_any_non_dash_treated_as_anomaly(self, tmp_path: Path):
        """Even unusual non-dash labels (e.g., 'KERNEL', 'ALERT') count
        as anomalies per D5's 'label_field != "-"' rule."""
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"

        # Use a non-standard but non-dash label
        lines = [
            _bgl_normal(1000),
            _bgl_normal(1005),
            f"KERNEL 1009 some unusual label content here",
        ]
        _write_log(log_path, lines)

        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        # The window with "KERNEL" line is an anomaly
        assert summary.anomaly_paragraphs == 1
