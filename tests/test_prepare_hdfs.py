"""Tests for src/prepare_hdfs.py — v1.3 HDFS preprocessing.

Mapping to v1.3 spec Component 13:
- test_hdfs_blockid_extraction      → BLOCK_ID_RE (v1.3: word-bounded)
- test_within_line_dedup             → [D16 v1.3] CORRECTED: same-block-twice → counter is 0
- test_distinct_blockids_increments  → [D16 v1.3] NEW: true multi-block line → counter +1
- test_hdfs_first_appearance_sort   → [BUG-2] sort order
- test_duplicate_blockid_in_csv_*   → [D19 NEW v1.3] duplicate row handling
- test_word_boundary_regex_*         → [R2 F4 v1.3] prefix-match rejection
- test_first_appearance_tie_break    → [R2 F10 v1.3] determinism
- test_supervisor_flag_*             → [R2 F11 v1.3] capsys-based
- test_total_paragraphs_zero         → [R2 F12 v1.3] empty output
- test_encoding_replacement          → [D20 NEW v1.3] \ufffd audit
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import pytest

from src.prepare_hdfs import (
    BLOCK_ID_RE,
    LINES_WITH_MULTIPLE_BLOCKIDS_FLAG_THRESHOLD,
    UTF8_REPLACEMENT_CHAR,
    load_label_dict,
    prepare_hdfs,
)
from src.types import Paragraph
from src.utils.stats import compute_length_distribution

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_hdfs_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_label_csv(path: Path, labels: dict[str, str]) -> None:
    """Simple label CSV. For duplicate-row tests, use _write_label_csv_raw."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["BlockId", "Label"])
        for block_id, label in labels.items():
            writer.writerow([block_id, label])


def _write_label_csv_raw(path: Path, rows: list[tuple[str, str]]) -> None:
    """Label CSV that allows duplicate rows (for D19 tests)."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["BlockId", "Label"])
        for block_id, label in rows:
            writer.writerow([block_id, label])


@pytest.fixture
def hdfs_minimal(tmp_path: Path) -> tuple[Path, Path, Path]:
    log_path = tmp_path / "HDFS.log"
    csv_path = tmp_path / "anomaly_label.csv"
    out_path = tmp_path / "out"

    log_lines = [
        "081109 203518 143 INFO Receiving block "
        "blk_-1608999687919862906 src: /10.0.0.1",
        "081109 203518 35 INFO BLOCK* allocateBlock: "
        "blk_-9999999999999999999",
        "081109 203519 143 INFO Receiving block "
        "blk_-1608999687919862906 src: /10.0.0.1",
        "081109 203520 12 INFO Some normal log without block reference",
        "081109 203521 99 INFO PacketResponder 1 for block "
        "blk_1234567 terminating",
    ]
    _write_hdfs_log(log_path, log_lines)
    _write_label_csv(
        csv_path,
        {
            "blk_-1608999687919862906": "Normal",
            "blk_-9999999999999999999": "Anomaly",
            "blk_1234567": "Normal",
        },
    )
    return log_path, csv_path, out_path


# ---------------------------------------------------------------------------
# Regex tests (v1.3 word-bounded)
# ---------------------------------------------------------------------------


class TestBlockIdRegex:
    def test_extracts_negative_blockid(self):
        assert BLOCK_ID_RE.findall(
            "Receiving block blk_-1608999687919862906 src:"
        ) == ["blk_-1608999687919862906"]

    def test_extracts_positive_blockid(self):
        assert BLOCK_ID_RE.findall(
            "block blk_1234567 src:"
        ) == ["blk_1234567"]

    def test_extracts_multiple_blockids_in_order(self):
        line = "Replicate block blk_-100 src:/10.0.0.1 to blk_200 done"
        assert BLOCK_ID_RE.findall(line) == ["blk_-100", "blk_200"]

    def test_no_blockid_returns_empty(self):
        assert BLOCK_ID_RE.findall("This line has no block reference") == []

    def test_word_boundary_rejects_prefix_match(self):
        """[R2 F4 v1.3] Regex must not match inside larger tokens.

        Without word boundaries, `r"blk_-?\\d+"` would match `prefixblk_123`.
        With `r"\\bblk_-?\\d+\\b"` it does not.
        """
        assert BLOCK_ID_RE.findall("prefixblk_123 word") == []
        assert BLOCK_ID_RE.findall("fooblk_456bar") == []

    def test_word_boundary_accepts_legitimate_blockid_at_eol(self):
        """End of line is a valid word boundary."""
        assert BLOCK_ID_RE.findall("block blk_123") == ["blk_123"]

    def test_does_not_match_blk_with_no_digits(self):
        assert BLOCK_ID_RE.findall("This mentions blk_abc but no digits") == []


# ---------------------------------------------------------------------------
# load_label_dict — v1.3 with D19 duplicate handling
# ---------------------------------------------------------------------------


class TestLoadLabelDict:
    def test_parses_normal_and_anomaly(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_label_csv(csv_path, {"blk_1": "Normal", "blk_2": "Anomaly"})
        labels, dup_count = load_label_dict(csv_path)
        assert labels == {"blk_1": 0, "blk_2": 1}
        assert dup_count == 0

    def test_unexpected_label_raises(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_label_csv(csv_path, {"blk_1": "Suspicious"})
        with pytest.raises(ValueError, match="Unexpected label"):
            load_label_dict(csv_path)

    def test_missing_columns_raises(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["WrongColumn", "OtherColumn"])
            writer.writerow(["blk_1", "Normal"])
        with pytest.raises(ValueError, match="Expected columns"):
            load_label_dict(csv_path)

    def test_tolerates_whitespace_in_headers(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([" BlockId ", " Label "])
            writer.writerow(["blk_1", "Normal"])
        labels, _ = load_label_dict(csv_path)
        assert labels == {"blk_1": 0}


class TestDuplicateBlockIdInCsv:
    """[D19 NEW v1.3] Duplicate row handling in label CSV."""

    def test_exact_duplicate_accepted(self, tmp_path: Path, capsys):
        csv_path = tmp_path / "labels.csv"
        _write_label_csv_raw(
            csv_path,
            [
                ("blk_1", "Normal"),
                ("blk_1", "Normal"),   # exact duplicate
                ("blk_2", "Anomaly"),
            ],
        )
        labels, dup_count = load_label_dict(csv_path)
        assert labels == {"blk_1": 0, "blk_2": 1}
        assert dup_count == 1
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "blk_1" in captured.out

    def test_conflicting_duplicate_hard_fails(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_label_csv_raw(
            csv_path,
            [
                ("blk_1", "Normal"),
                ("blk_1", "Anomaly"),   # CONFLICT
            ],
        )
        with pytest.raises(ValueError, match="HARD FAIL.*conflicting"):
            load_label_dict(csv_path)

    def test_conflicting_duplicate_error_includes_block_id(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_label_csv_raw(
            csv_path,
            [
                ("blk_42", "Anomaly"),
                ("blk_42", "Normal"),
            ],
        )
        with pytest.raises(ValueError) as exc_info:
            load_label_dict(csv_path)
        assert "blk_42" in str(exc_info.value)


# ---------------------------------------------------------------------------
# compute_length_distribution (moved to src/utils/stats.py in v1.3)
# ---------------------------------------------------------------------------


class TestComputeLengthDistribution:
    def test_empty_returns_zeros(self):
        result = compute_length_distribution([])
        assert result.min == 0
        assert result.max == 0
        assert result.p50 == 0.0

    def test_single_value(self):
        result = compute_length_distribution([42])
        assert result.min == 42
        assert result.max == 42

    def test_percentile_calculation(self):
        result = compute_length_distribution(list(range(1, 101)))
        assert result.p50 == pytest.approx(50.5)
        assert result.min == 1
        assert result.max == 100


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestPrepareHdfsMinimal:
    def test_paragraph_count(self, hdfs_minimal):
        log_path, csv_path, out_path = hdfs_minimal
        summary = prepare_hdfs(log_path, csv_path, out_path)
        assert summary.total_paragraphs == 3
        assert summary.normal_paragraphs == 2
        assert summary.anomaly_paragraphs == 1

    def test_anomaly_rate(self, hdfs_minimal):
        log_path, csv_path, out_path = hdfs_minimal
        summary = prepare_hdfs(log_path, csv_path, out_path)
        assert summary.anomaly_rate == pytest.approx(1 / 3)

    def test_no_blockid_lines_dropped(self, hdfs_minimal):
        log_path, csv_path, out_path = hdfs_minimal
        summary = prepare_hdfs(log_path, csv_path, out_path)
        assert summary.drop_counters.no_blockid == 1

    def test_paragraphs_pkl_loadable(self, hdfs_minimal):
        log_path, csv_path, out_path = hdfs_minimal
        prepare_hdfs(log_path, csv_path, out_path)
        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)
        assert len(paragraphs) == 3
        assert all(isinstance(p, Paragraph) for p in paragraphs)


class TestFirstAppearanceSort:
    """[BUG-2] Sort by first-appearance offset, not lex."""

    def test_lex_differs_from_first_appearance(self, tmp_path: Path):
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        log_lines = [
            "081109 INFO event for blk_9",
            "081109 INFO event for blk_5",
            "081109 INFO event for blk_1",
        ]
        _write_hdfs_log(log_path, log_lines)
        _write_label_csv(
            csv_path,
            {"blk_9": "Normal", "blk_5": "Normal", "blk_1": "Normal"},
        )
        prepare_hdfs(log_path, csv_path, out_path)

        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)

        actual_order = [p.paragraph_id for p in paragraphs]
        assert actual_order == ["blk_9", "blk_5", "blk_1"]
        assert actual_order != sorted(actual_order)

    def test_multi_blockid_first_appearance_tie_break_deterministic(
        self, tmp_path: Path
    ):
        """[R2 F10 v1.3] When two blocks first appear on the same line,
        tie-break is first-mention order within that line. Determinism
        relies on Python 3.7+ dict insertion order and stable sort."""
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        log_lines = [
            "081109 INFO event for blk_OLD",
            "081109 INFO replicate from blk_A to blk_B then blk_C",
        ]
        _write_hdfs_log(log_path, log_lines)
        _write_label_csv(
            csv_path,
            {
                "blk_OLD": "Normal",
                "blk_A": "Normal",
                "blk_B": "Normal",
                "blk_C": "Normal",
            },
        )
        prepare_hdfs(log_path, csv_path, out_path)

        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)
        assert [p.paragraph_id for p in paragraphs] == [
            "blk_OLD",
            "blk_A",
            "blk_B",
            "blk_C",
        ]


class TestMultiBlockidD16:
    """[D16 v1.3] CORRECTED counter semantics — distinct-id based."""

    def test_within_line_dedup_no_counter_increment(self, tmp_path: Path):
        """[D16 v1.3 / convergent R1 F1 + R2 F3] CORRECTED v1.3.

        A line mentioning the same block_id multiple times produces NO
        duplication event — only the same paragraph is appended to. Counter
        must NOT increment for this case (was: incremented in v1.2)."""
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        log_lines = [
            "081109 INFO event involving blk_X twice: blk_X and blk_X again",
        ]
        _write_hdfs_log(log_path, log_lines)
        _write_label_csv(csv_path, {"blk_X": "Normal"})

        summary = prepare_hdfs(log_path, csv_path, out_path)
        # CORRECTED v1.3: same-block-twice should NOT increment counter
        assert summary.drop_counters.lines_with_multiple_blockids == 0

        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)
        assert len(paragraphs) == 1
        assert len(paragraphs[0].lines) == 1

    def test_distinct_blockids_increments_counter(self, tmp_path: Path):
        """[D16 v1.3 NEW] True multi-block line → counter +1."""
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        log_lines = [
            "081109 INFO event for blk_A",
            "081109 INFO event for blk_B",
            "081109 INFO Replicate from blk_A to blk_B done",
        ]
        _write_hdfs_log(log_path, log_lines)
        _write_label_csv(csv_path, {"blk_A": "Normal", "blk_B": "Normal"})

        summary = prepare_hdfs(log_path, csv_path, out_path)
        assert summary.drop_counters.lines_with_multiple_blockids == 1

        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)
        blocks = {p.paragraph_id: p for p in paragraphs}
        # D16 duplication: replicate line appears in both
        assert any("Replicate" in line for line in blocks["blk_A"].lines)
        assert any("Replicate" in line for line in blocks["blk_B"].lines)

    def test_three_distinct_blocks_in_one_line(self, tmp_path: Path):
        """A line referencing 3 distinct blocks still increments counter by 1
        (per-line, not per-distinct-block)."""
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        _write_hdfs_log(log_path, ["INFO ref blk_A and blk_B and blk_C"])
        _write_label_csv(
            csv_path,
            {"blk_A": "Normal", "blk_B": "Normal", "blk_C": "Normal"},
        )

        summary = prepare_hdfs(log_path, csv_path, out_path)
        assert summary.drop_counters.lines_with_multiple_blockids == 1
        assert summary.total_paragraphs == 3


class TestStrictLabelAssertion:
    """[BUG-3] HARD FAIL on missing block_id in label CSV."""

    def test_missing_label_raises(self, tmp_path: Path):
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"
        _write_hdfs_log(log_path, ["INFO event for blk_UNLABELED"])
        _write_label_csv(csv_path, {"blk_OTHER": "Normal"})
        with pytest.raises(ValueError, match="HARD FAIL"):
            prepare_hdfs(log_path, csv_path, out_path)

    def test_error_mentions_line_number(self, tmp_path: Path):
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"
        _write_hdfs_log(
            log_path,
            [
                "INFO event for blk_KNOWN",
                "INFO event for blk_UNLABELED",
            ],
        )
        _write_label_csv(csv_path, {"blk_KNOWN": "Normal"})
        with pytest.raises(ValueError) as exc_info:
            prepare_hdfs(log_path, csv_path, out_path)
        msg = str(exc_info.value)
        assert "blk_UNLABELED" in msg
        assert "line 2" in msg


class TestSupervisorFlag:
    """[R2 F11 v1.3] capsys-based tests for SUPERVISOR_FLAG printing."""

    def test_supervisor_flag_fires_above_5_percent(
        self, tmp_path: Path, capsys
    ):
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        # 10 lines total, 6 are multi-blockid → 60% rate (well above 5%)
        log_lines = [
            "INFO single for blk_A",
            "INFO single for blk_B",
            "INFO single for blk_C",
            "INFO single for blk_D",
        ] + [
            f"INFO replicate from blk_A to blk_B at step {i}"
            for i in range(6)
        ]
        _write_hdfs_log(log_path, log_lines)
        _write_label_csv(
            csv_path,
            {
                "blk_A": "Normal",
                "blk_B": "Normal",
                "blk_C": "Normal",
                "blk_D": "Normal",
            },
        )
        prepare_hdfs(log_path, csv_path, out_path)
        captured = capsys.readouterr()
        assert "SUPERVISOR_FLAG" in captured.out
        assert "multi-blockid" in captured.out

    def test_supervisor_flag_silent_below_5_percent(
        self, tmp_path: Path, capsys
    ):
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        # 100 lines, 0 multi-blockid → 0% rate
        log_lines = [
            f"INFO event for blk_{i % 5}" for i in range(100)
        ]
        _write_hdfs_log(log_path, log_lines)
        _write_label_csv(
            csv_path, {f"blk_{i}": "Normal" for i in range(5)}
        )
        prepare_hdfs(log_path, csv_path, out_path)
        captured = capsys.readouterr()
        assert "multi-blockid" not in captured.out


class TestTotalParagraphsZero:
    """[R2 F12 v1.3] Empty output handling."""

    def test_no_parseable_blockids_yields_empty_output(self, tmp_path: Path):
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        _write_hdfs_log(
            log_path,
            [
                "INFO line with no block reference",
                "INFO another line without one",
            ],
        )
        _write_label_csv(csv_path, {"blk_NEVER_USED": "Normal"})

        summary = prepare_hdfs(log_path, csv_path, out_path)
        assert summary.total_paragraphs == 0
        assert summary.anomaly_rate == 0.0

        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)
        assert paragraphs == []


class TestEncodingReplacement:
    """[D20 NEW v1.3] \\ufffd audit trail."""

    def test_replacement_char_in_line_increments_counter(self, tmp_path: Path):
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        # Embed a literal U+FFFD in a line (simulating errors="replace" output)
        log_lines = [
            "INFO event for blk_1 normal line",
            f"INFO event for blk_2 with {UTF8_REPLACEMENT_CHAR} corruption",
            "INFO event for blk_3 another normal",
        ]
        _write_hdfs_log(log_path, log_lines)
        _write_label_csv(
            csv_path,
            {"blk_1": "Normal", "blk_2": "Normal", "blk_3": "Normal"},
        )
        summary = prepare_hdfs(log_path, csv_path, out_path)
        assert summary.drop_counters.encoding_replacements_seen == 1
        # The offending line is line 2 (1-indexed)
        assert summary.encoding_offending_line_numbers == [2]

    def test_offending_line_numbers_capped_at_100(self, tmp_path: Path):
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        # Write 150 lines, all containing the replacement char
        log_lines = [
            f"INFO event for blk_{i} with {UTF8_REPLACEMENT_CHAR} char"
            for i in range(150)
        ]
        _write_hdfs_log(log_path, log_lines)
        _write_label_csv(
            csv_path, {f"blk_{i}": "Normal" for i in range(150)}
        )
        summary = prepare_hdfs(log_path, csv_path, out_path)
        assert summary.drop_counters.encoding_replacements_seen == 150
        # Captured offsets capped at first 100
        assert len(summary.encoding_offending_line_numbers) == 100


class TestDropCountersSurfaced:
    def test_no_blockid_counter_in_json(self, hdfs_minimal):
        log_path, csv_path, out_path = hdfs_minimal
        prepare_hdfs(log_path, csv_path, out_path)
        with (out_path / "preparation_summary.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["drop_counters"]["no_blockid"] == 1
        assert data["drop_counters"]["lines_with_multiple_blockids"] == 0
        # v1.3 new fields present
        assert "encoding_replacements_seen" in data["drop_counters"]
        assert "duplicate_blockid" in data["drop_counters"]


class TestThresholdConstants:
    def test_flag_threshold_matches_spec(self):
        assert LINES_WITH_MULTIPLE_BLOCKIDS_FLAG_THRESHOLD == 0.05
