"""Tests for src/prepare_hdfs.py — HDFS preprocessing.

Mapping to v1.2 spec Component 13:
- test_hdfs_blockid_extraction      → BLOCK_ID_RE regex behaviour
- test_hdfs_multi_blockid            → [D16] line with 2 block_ids duplicated
- test_hdfs_first_appearance_sort   → [BUG-2] sort order matches first-appearance offset

Additional coverage:
- test_load_label_dict               → CSV parsing edge cases
- test_strict_label_assertion        → [BUG-3] HARD FAIL on missing label
- test_drop_counters_surfaced        → preparation_summary.json counters
- test_word_length_distribution      → percentile computation
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import pytest

from src.prepare_hdfs import (
    BLOCK_ID_RE,
    compute_length_distribution,
    load_label_dict,
    prepare_hdfs,
)
from src.types import LengthDistribution, Paragraph


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _write_hdfs_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_label_csv(path: Path, labels: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["BlockId", "Label"])
        for block_id, label in labels.items():
            writer.writerow([block_id, label])


@pytest.fixture
def hdfs_minimal(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Three blocks, one no-blockid line, no multi-blockid lines."""
    log_path = tmp_path / "HDFS.log"
    csv_path = tmp_path / "anomaly_label.csv"
    out_path = tmp_path / "out"

    log_lines = [
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block "
        "blk_-1608999687919862906 src: /10.250.19.102 dest: /10.250.19.102",
        "081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: "
        "/mnt/hadoop/mapred/system/job_200811092030_0001/job.split "
        "blk_-9999999999999999999",
        "081109 203519 143 INFO dfs.DataNode$DataXceiver: Receiving block "
        "blk_-1608999687919862906 src: /10.250.19.102 dest: /10.250.19.102",
        "081109 203520 12 INFO dfs.DataNode: Some normal log without block reference",
        "081109 203521 99 INFO dfs.DataNode$PacketResponder: "
        "PacketResponder 1 for block blk_1234567 terminating",
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
# Regex tests
# ---------------------------------------------------------------------------


class TestBlockIdRegex:
    """Block ID regex extraction (used by D16 multi-blockid logic)."""

    def test_extracts_negative_blockid(self):
        assert BLOCK_ID_RE.findall(
            "Receiving block blk_-1608999687919862906 src:"
        ) == ["blk_-1608999687919862906"]

    def test_extracts_positive_blockid(self):
        assert BLOCK_ID_RE.findall(
            "Receiving block blk_1234567 src:"
        ) == ["blk_1234567"]

    def test_extracts_multiple_blockids_in_order(self):
        """[D16] Line with multiple block_ids should yield all of them
        in source order."""
        line = "Replicate block blk_-100 src:/10.0.0.1 to blk_200 done"
        assert BLOCK_ID_RE.findall(line) == ["blk_-100", "blk_200"]

    def test_no_blockid_returns_empty(self):
        assert BLOCK_ID_RE.findall("This line has no block reference") == []

    def test_does_not_match_partial_words(self):
        """blk_ must be followed by an optional minus and digits — not
        random alphanumerics."""
        # 'blk_abc' should not match (no digits after underscore)
        assert BLOCK_ID_RE.findall("This mentions blk_abc but not real") == []


# ---------------------------------------------------------------------------
# CSV parsing tests
# ---------------------------------------------------------------------------


class TestLoadLabelDict:
    def test_parses_normal_and_anomaly(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_label_csv(csv_path, {"blk_1": "Normal", "blk_2": "Anomaly"})
        labels = load_label_dict(csv_path)
        assert labels == {"blk_1": 0, "blk_2": 1}

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
        labels = load_label_dict(csv_path)
        assert labels == {"blk_1": 0}


# ---------------------------------------------------------------------------
# LengthDistribution tests
# ---------------------------------------------------------------------------


class TestComputeLengthDistribution:
    def test_empty_returns_zeros(self):
        result = compute_length_distribution([])
        assert result == LengthDistribution(
            min=0, max=0, p50=0.0, p80=0.0, p95=0.0, p99=0.0
        )

    def test_single_value(self):
        result = compute_length_distribution([42])
        assert result.min == 42
        assert result.max == 42
        assert result.p50 == 42.0

    def test_percentile_calculation(self):
        result = compute_length_distribution(list(range(1, 101)))
        # p50 of [1..100] is 50.5 under numpy default linear interpolation
        assert result.p50 == pytest.approx(50.5)
        assert result.p99 == pytest.approx(99.01, abs=0.5)
        assert result.min == 1
        assert result.max == 100


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestPrepareHdfsMinimal:
    """End-to-end on the minimal fixture."""

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
        # The fixture has exactly one no-blockid line.
        assert summary.drop_counters.no_blockid == 1

    def test_paragraphs_pkl_written_and_loadable(self, hdfs_minimal):
        log_path, csv_path, out_path = hdfs_minimal
        prepare_hdfs(log_path, csv_path, out_path)
        paragraphs_path = out_path / "paragraphs.pkl"
        assert paragraphs_path.exists()
        with paragraphs_path.open("rb") as f:
            paragraphs = pickle.load(f)
        assert len(paragraphs) == 3
        assert all(isinstance(p, Paragraph) for p in paragraphs)

    def test_preparation_summary_json_written(self, hdfs_minimal):
        log_path, csv_path, out_path = hdfs_minimal
        prepare_hdfs(log_path, csv_path, out_path)
        summary_path = out_path / "preparation_summary.json"
        assert summary_path.exists()
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["dataset"] == "hdfs"
        assert data["total_paragraphs"] == 3


class TestFirstAppearanceSort:
    """[BUG-2] Paragraphs must be ordered by first-appearance offset of
    block_id in the raw log, NOT by lexicographic order of block_id strings."""

    def test_lex_sort_differs_from_first_appearance(self, tmp_path: Path):
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        # First-appearance order: blk_9, blk_5, blk_1
        # Lexicographic order would be:  blk_1, blk_5, blk_9
        log_lines = [
            "081109 203518 143 INFO event for blk_9",
            "081109 203519 143 INFO event for blk_5",
            "081109 203520 143 INFO event for blk_1",
        ]
        _write_hdfs_log(log_path, log_lines)
        _write_label_csv(
            csv_path, {"blk_9": "Normal", "blk_5": "Normal", "blk_1": "Normal"}
        )

        prepare_hdfs(log_path, csv_path, out_path)

        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)

        actual_order = [p.paragraph_id for p in paragraphs]
        assert actual_order == ["blk_9", "blk_5", "blk_1"]
        # Sanity: lex sort would have produced a different order
        assert actual_order != sorted(actual_order)


class TestMultiBlockidD16:
    """[D16] Lines with multiple block_ids are duplicated into each
    referenced block's paragraph (DeepLog convention)."""

    def test_line_with_two_blockids_appears_in_both(self, tmp_path: Path):
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        log_lines = [
            "081109 203518 143 INFO event for blk_A",
            "081109 203519 143 INFO event for blk_B",
            "081109 203520 143 INFO Replicate from blk_A to blk_B done",
        ]
        _write_hdfs_log(log_path, log_lines)
        _write_label_csv(csv_path, {"blk_A": "Normal", "blk_B": "Normal"})

        summary = prepare_hdfs(log_path, csv_path, out_path)

        # Counter incremented exactly once (one line had multiple block_ids)
        assert summary.drop_counters.lines_with_multiple_blockids == 1

        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)

        blocks = {p.paragraph_id: p for p in paragraphs}
        # Each block has 2 lines: its own creation line + the shared replicate line
        assert len(blocks["blk_A"].lines) == 2
        assert len(blocks["blk_B"].lines) == 2
        # The replicate line should appear in BOTH paragraphs (D16 duplication)
        assert any("Replicate" in line for line in blocks["blk_A"].lines)
        assert any("Replicate" in line for line in blocks["blk_B"].lines)

    def test_within_line_dedup(self, tmp_path: Path):
        """If the same block_id appears twice in one line, the line is
        added to that block exactly once (not twice)."""
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        log_lines = [
            "081109 203518 143 INFO event involving blk_X twice: blk_X and blk_X again",
        ]
        _write_hdfs_log(log_path, log_lines)
        _write_label_csv(csv_path, {"blk_X": "Normal"})

        summary = prepare_hdfs(log_path, csv_path, out_path)

        # 3 mentions of the same block_id → counts as multi-blockid line
        assert summary.drop_counters.lines_with_multiple_blockids == 1

        with (out_path / "paragraphs.pkl").open("rb") as f:
            paragraphs = pickle.load(f)

        # But the line is only stored ONCE in blk_X's paragraph
        assert len(paragraphs) == 1
        assert len(paragraphs[0].lines) == 1


class TestStrictLabelAssertion:
    """[BUG-3] Block ID with no entry in the label CSV must HARD FAIL,
    not silently default to NORMAL."""

    def test_missing_label_raises_hard_fail(self, tmp_path: Path):
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        _write_hdfs_log(
            log_path,
            ["081109 203518 143 INFO event for blk_UNLABELED"],
        )
        # blk_UNLABELED is NOT in the label CSV
        _write_label_csv(csv_path, {"blk_OTHER": "Normal"})

        with pytest.raises(ValueError, match="HARD FAIL"):
            prepare_hdfs(log_path, csv_path, out_path)

    def test_missing_label_error_mentions_line_number(self, tmp_path: Path):
        """Error message includes line number and block_id for triage."""
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        _write_hdfs_log(
            log_path,
            [
                "081109 203518 143 INFO event for blk_KNOWN",
                "081109 203519 143 INFO event for blk_UNLABELED",
            ],
        )
        _write_label_csv(csv_path, {"blk_KNOWN": "Normal"})

        with pytest.raises(ValueError) as exc_info:
            prepare_hdfs(log_path, csv_path, out_path)

        msg = str(exc_info.value)
        assert "blk_UNLABELED" in msg
        assert "line 2" in msg


class TestDropCountersSurfaced:
    """preparation_summary.json must surface non-zero drop counters."""

    def test_no_blockid_counter_in_json(self, hdfs_minimal):
        log_path, csv_path, out_path = hdfs_minimal
        prepare_hdfs(log_path, csv_path, out_path)
        with (out_path / "preparation_summary.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["drop_counters"]["no_blockid"] == 1
        assert data["drop_counters"]["lines_with_multiple_blockids"] == 0


class TestWordLengthDistribution:
    """Word length stats computed correctly across paragraphs."""

    def test_word_length_per_paragraph(self, tmp_path: Path):
        log_path = tmp_path / "HDFS.log"
        csv_path = tmp_path / "labels.csv"
        out_path = tmp_path / "out"

        # blk_A: 2 lines of 5 words each = 10 words total
        # blk_B: 1 line of 3 words
        log_lines = [
            "one two three four blk_A",     # 5 words
            "five six seven eight blk_A",   # 5 words
            "nine ten blk_B",                # 3 words
        ]
        _write_hdfs_log(log_path, log_lines)
        _write_label_csv(csv_path, {"blk_A": "Normal", "blk_B": "Normal"})

        summary = prepare_hdfs(log_path, csv_path, out_path)
        wld = summary.word_length_distribution

        assert wld.min == 3
        assert wld.max == 10
