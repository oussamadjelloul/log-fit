"""Tests for src/prepare_bgl_tbird.py — v1.3 BGL/Thunderbird preprocessing.

Mapping to v1.3 spec Component 13:
- TestParseLine                       → _parse_line edge cases (3-field min per D4 v1.3)
- TestWindowAlignment                  → spec §1.2 step 3 boundaries
- TestLabelAggregationD5               → [D5] window anomaly iff ≥1 anomalous line
- TestTimestampFailureThreshold       → [R2 F15] >0.1% unparseable → HARD FAIL
- TestSingletonWindowsDropped          → [R2 F5 v1.3] CORRECTED: split by label
- TestSingletonAnomalyFlagged          → [R2 F5 v1.3] NEW: anomaly singleton fires flag
- TestTwoTokenLineRejected             → [D4 v1.3 / R2 F6] NEW
- TestNegativeWindowId                 → [R2 F9 v1.3] NEW
- TestEncodingReplacement              → [D20 NEW v1.3]
- TestTotalParagraphsZero              → [R2 F12 v1.3]
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
    UTF8_REPLACEMENT_CHAR,
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
    return f"- {ts} 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL INFO {msg}"


def _bgl_anomaly(
    ts: int, category: str = "APPSEV", msg: str = "anomaly detected"
) -> str:
    return f"{category} {ts} 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL ERROR {msg}"


# ---------------------------------------------------------------------------
# _parse_line — v1.3 with 3-field requirement
# ---------------------------------------------------------------------------


class TestParseLine:
    def test_normal_line(self):
        line = _bgl_normal(1000)
        result = _parse_line(line)
        assert result is not None
        label, ts, content = result
        assert label == "-"
        assert ts == 1000
        assert content.startswith("1000 ")

    def test_anomaly_line(self):
        line = _bgl_anomaly(1005, "APPSEV", "severe error")
        result = _parse_line(line)
        assert result is not None
        label, ts, content = result
        assert label == "APPSEV"
        assert ts == 1005

    def test_empty_line_returns_none(self):
        assert _parse_line("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_line("   \t  ") is None

    def test_single_token_returns_none(self):
        assert _parse_line("-") is None

    def test_two_token_line_rejected(self):
        """[D4 v1.3 / R2 F6 NEW] Lines with only label + timestamp (no content)
        rejected as parse failure."""
        assert _parse_line("- 1000") is None
        assert _parse_line("APPSEV 1234") is None

    def test_three_field_line_minimal(self):
        """3-field line is the minimum acceptable."""
        result = _parse_line("- 1000 minimal_content")
        assert result is not None
        label, ts, content = result
        assert label == "-"
        assert ts == 1000
        assert content == "1000 minimal_content"

    def test_non_integer_timestamp_returns_none(self):
        assert _parse_line("- abc rest of line") is None

    def test_leading_whitespace_tolerated(self):
        result = _parse_line("  - 1000 message_content")
        assert result is not None
        label, ts, content = result
        assert label == "-"
        assert ts == 1000

    def test_content_excludes_only_label(self):
        line = "APPSEV 1234 2005.06.03 NODE RAS KERNEL ERROR something"
        result = _parse_line(line)
        assert result is not None
        _, _, content = result
        assert content == "1234 2005.06.03 NODE RAS KERNEL ERROR something"
        assert "APPSEV" not in content


# ---------------------------------------------------------------------------
# Window alignment (Unchanged from v1.2 verification)
# ---------------------------------------------------------------------------


class TestWindowAlignment:
    def test_window_boundaries_10s(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        lines = (
            [_bgl_normal(ts) for ts in [1000, 1003, 1009]]
            + [_bgl_normal(ts) for ts in [1010, 1015, 1019]]
            + [_bgl_normal(ts) for ts in [1020, 1025, 1029]]
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
            paragraphs = pickle.load(f)
        for p in paragraphs:
            assert len(p.lines) == 3
        assert [p.source_window_id for p in paragraphs] == [0, 1, 2]

    def test_window_boundary_exclusive_upper(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        lines = [
            _bgl_normal(1000),
            _bgl_normal(1005),
            _bgl_normal(1009),
            _bgl_normal(1010),
            _bgl_normal(1015),
        ]
        _write_log(log_path, lines)
        prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)
        windows = {p.source_window_id: p for p in paragraphs}
        assert len(windows[0].lines) == 3
        assert len(windows[1].lines) == 2


# ---------------------------------------------------------------------------
# Label aggregation D5
# ---------------------------------------------------------------------------


class TestLabelAggregationD5:
    def test_all_normal_window(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        _write_log(log_path, [_bgl_normal(ts) for ts in [1000, 1005, 1009]])
        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.normal_paragraphs == 1
        assert summary.anomaly_paragraphs == 0

    def test_single_anomaly_makes_window_anomaly(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        _write_log(
            log_path,
            [
                _bgl_normal(1000),
                _bgl_normal(1003),
                _bgl_anomaly(1007),
                _bgl_normal(1009),
            ],
        )
        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.anomaly_paragraphs == 1
        assert summary.normal_paragraphs == 0


# ---------------------------------------------------------------------------
# Timestamp failure threshold
# ---------------------------------------------------------------------------


class TestTimestampFailureThreshold:
    def test_threshold_constant(self):
        assert UNPARSEABLE_TIMESTAMP_FAIL_THRESHOLD == 0.001

    def test_below_threshold_no_failure(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        # 1999 valid + 1 garbage = 0.05% < 0.1%
        valid = [_bgl_normal(1000 + i // 100) for i in range(1999)]
        invalid = ["GARBAGE_LINE no timestamp here at all"]
        _write_log(log_path, valid + invalid)
        summary = prepare_bgl_tbird(log_path, "bgl", 60, out_path)
        # Either unparseable_timestamp or lines_missing_content counted
        total_fails = (
            summary.drop_counters.unparseable_timestamp
            + summary.drop_counters.lines_missing_content
        )
        assert total_fails == 1

    def test_above_threshold_hard_fails(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        # 989 valid + 11 garbage = 1.1% > 0.1%
        valid = [_bgl_normal(1000 + i // 10) for i in range(989)]
        invalid = ["GARBAGE no timestamp at all"] * 11
        _write_log(log_path, valid + invalid)
        with pytest.raises(ValueError, match="HARD FAIL"):
            prepare_bgl_tbird(log_path, "bgl", 60, out_path)


# ---------------------------------------------------------------------------
# Singleton windows — v1.3 split by label
# ---------------------------------------------------------------------------


class TestSingletonWindowsDropped:
    """[R2 F5 v1.3] CORRECTED — counter split into normal/anomaly subcounts."""

    def test_min_lines_per_window_value(self):
        assert MIN_LINES_PER_WINDOW == 2

    def test_normal_singleton_dropped(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        lines = [
            _bgl_normal(1000),
            _bgl_normal(1005),
            _bgl_normal(1009),
            _bgl_normal(1015),  # alone in window 1
            _bgl_normal(1020),
            _bgl_normal(1025),
        ]
        _write_log(log_path, lines)
        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.total_paragraphs == 2
        assert summary.drop_counters.singleton_window_normal == 1
        assert summary.drop_counters.singleton_window_anomaly == 0

    def test_anomaly_singleton_tracked_separately(
        self, tmp_path: Path, capsys
    ):
        """[R2 F5 v1.3 NEW] Singleton anomaly drop tracked in its own counter
        and fires a SUPERVISOR_FLAG."""
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        lines = [
            _bgl_normal(1000),
            _bgl_normal(1005),
            _bgl_normal(1009),
            _bgl_anomaly(1015),    # ALONE in window 1 — anomaly singleton
            _bgl_normal(1020),
            _bgl_normal(1025),
        ]
        _write_log(log_path, lines)
        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.drop_counters.singleton_window_anomaly == 1
        assert summary.drop_counters.singleton_window_normal == 0

        captured = capsys.readouterr()
        assert "SUPERVISOR_FLAG" in captured.out
        assert "anomalous singleton" in captured.out

    def test_two_line_window_kept(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        _write_log(log_path, [_bgl_normal(1000), _bgl_normal(1005)])
        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.total_paragraphs == 1


# ---------------------------------------------------------------------------
# Thunderbird L19 cap
# ---------------------------------------------------------------------------


class TestThunderbirdMaxLinesCap:
    def test_cap_enforced(self, tmp_path: Path):
        log_path = tmp_path / "Thunderbird.log"
        out_path = tmp_path / "out"
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
# Parameter validation
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


# ---------------------------------------------------------------------------
# Out-of-order timestamps
# ---------------------------------------------------------------------------


class TestOutOfOrderTimestamps:
    def test_late_line_routed_to_earlier_window(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        lines = [
            _bgl_normal(1000),
            _bgl_normal(1005),
            _bgl_normal(1015),
            _bgl_normal(1003),   # out of order
            _bgl_normal(1019),
        ]
        _write_log(log_path, lines)
        prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)
        windows = {p.source_window_id: p for p in paragraphs}
        assert len(windows[0].lines) == 3
        assert len(windows[1].lines) == 2

    def test_negative_window_id_handled(self, tmp_path: Path):
        """[R2 F9 v1.3 NEW] When the first valid line has a *later* timestamp
        than subsequent lines, the subsequent lines get *negative* window IDs.
        The code accepts this; the test codifies the behavior."""
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        # First line ts=2000 anchors; subsequent earlier-ts lines get neg windows
        lines = [
            _bgl_normal(2000),
            _bgl_normal(2005),   # window 0
            _bgl_normal(1000),   # window -100
            _bgl_normal(1005),   # window -100
        ]
        _write_log(log_path, lines)
        prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)
        window_ids = [p.source_window_id for p in paragraphs]
        assert -100 in window_ids
        # Negative windows come first (sorted ascending)
        assert window_ids == sorted(window_ids)


# ---------------------------------------------------------------------------
# Paragraph ID format
# ---------------------------------------------------------------------------


class TestParagraphIdFormat:
    def test_bgl_30s_format(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        _write_log(log_path, [_bgl_normal(1000), _bgl_normal(1005)])
        prepare_bgl_tbird(log_path, "bgl", 30, out_path)
        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)
        assert paragraphs[0].paragraph_id == "bgl_30s_w0"

    def test_tbird_60s_format(self, tmp_path: Path):
        log_path = tmp_path / "Thunderbird.log"
        out_path = tmp_path / "out"
        _write_log(log_path, [_bgl_normal(1000), _bgl_normal(1059)])
        prepare_bgl_tbird(log_path, "tbird", 60, out_path)
        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)
        assert paragraphs[0].paragraph_id == "tbird_60s_w0"


# ---------------------------------------------------------------------------
# Summary JSON — v1.3 new fields
# ---------------------------------------------------------------------------


class TestSummaryJsonWritten:
    def test_v13_fields_present(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        _write_log(log_path, [_bgl_normal(1000), _bgl_normal(1005)])
        prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        with (out_path / "preparation_summary.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["dataset"] == "bgl"
        assert data["window_seconds"] == 10
        # v1.3 new counter fields
        assert "singleton_window_normal" in data["drop_counters"]
        assert "singleton_window_anomaly" in data["drop_counters"]
        assert "lines_missing_content" in data["drop_counters"]
        assert "encoding_replacements_seen" in data["drop_counters"]
        # v1.3 new top-level field
        assert "encoding_offending_line_numbers" in data


# ---------------------------------------------------------------------------
# Encoding replacement [D20 NEW v1.3]
# ---------------------------------------------------------------------------


class TestEncodingReplacement:
    def test_replacement_char_in_line_increments_counter(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        lines = [
            _bgl_normal(1000),
            f"- 1005 normal message with {UTF8_REPLACEMENT_CHAR} char",
            _bgl_normal(1009),
        ]
        _write_log(log_path, lines)
        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.drop_counters.encoding_replacements_seen == 1
        assert summary.encoding_offending_line_numbers == [2]

    def test_offending_offsets_capped(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        # 150 lines all with the replacement char in content (but valid otherwise)
        lines = [
            f"- {1000 + i} message {UTF8_REPLACEMENT_CHAR} continues"
            for i in range(150)
        ]
        _write_log(log_path, lines)
        summary = prepare_bgl_tbird(log_path, "bgl", 60, out_path)
        assert summary.drop_counters.encoding_replacements_seen == 150
        assert len(summary.encoding_offending_line_numbers) == 100


# ---------------------------------------------------------------------------
# Total paragraphs zero [R2 F12 v1.3]
# ---------------------------------------------------------------------------


class TestTotalParagraphsZero:
    def test_all_singletons_yields_empty_output(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        # Each line is in its own window (timestamps 100 seconds apart, W=10)
        lines = [
            _bgl_normal(1000),
            _bgl_normal(1100),
            _bgl_normal(1200),
            _bgl_normal(1300),
        ]
        _write_log(log_path, lines)
        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.total_paragraphs == 0
        assert summary.anomaly_rate == 0.0
        assert summary.drop_counters.singleton_window_normal == 4


# ---------------------------------------------------------------------------
# Normal label sentinel
# ---------------------------------------------------------------------------


class TestNormalLabelSentinel:
    def test_sentinel_value(self):
        assert NORMAL_LABEL_FIELD == "-"

    def test_non_dash_is_anomaly(self, tmp_path: Path):
        log_path = tmp_path / "BGL.log"
        out_path = tmp_path / "out"
        lines = [
            _bgl_normal(1000),
            _bgl_normal(1005),
            "KERNEL 1009 some unusual label content here",
        ]
        _write_log(log_path, lines)
        summary = prepare_bgl_tbird(log_path, "bgl", 10, out_path)
        assert summary.anomaly_paragraphs == 1
