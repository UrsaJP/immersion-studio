"""Unit tests for core.subtitle: parse_srt and write_srt."""

import pytest

from core.subtitle import parse_srt, write_srt


class TestParseSrt:
    def test_parses_three_entries(self, fake_srt):
        entries = parse_srt(fake_srt)
        assert len(entries) == 3

    def test_entry_structure(self, fake_srt):
        """Each entry is a (index_str, timing_str, text_str) tuple."""
        entries = parse_srt(fake_srt)
        idx, timing, text = entries[0]
        assert idx == "1"
        assert "-->" in timing
        assert text == "おはようございます"

    def test_second_entry_text(self, fake_srt):
        entries = parse_srt(fake_srt)
        _, _, text = entries[1]
        assert text == "今日はいい天気ですね"

    def test_missing_file_returns_empty(self, tmp_path):
        result = parse_srt(str(tmp_path / "nonexistent.srt"))
        assert result == []

    def test_empty_path_returns_empty(self):
        assert parse_srt("") == []

    def test_bom_stripped(self, tmp_path):
        """UTF-8 BOM written by some tools must not leak into parsed index."""
        srt_file = tmp_path / "bom.srt"
        # Encode a plain string with utf-8-sig to produce a BOM-prefixed file
        content = "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n\n"
        srt_file.write_bytes(content.encode("utf-8-sig"))
        entries = parse_srt(str(srt_file))
        assert len(entries) == 1
        assert entries[0][0] == "1"  # index without BOM


class TestWriteSrt:
    def test_write_and_reparse(self, tmp_path, fake_srt):
        """write_srt output must be parseable by parse_srt."""
        original = parse_srt(fake_srt)
        out = str(tmp_path / "out.srt")
        write_srt(original, out)
        reparsed = parse_srt(out)
        assert len(reparsed) == len(original)

    def test_renumbers_from_one(self, tmp_path):
        """write_srt always renumbers entries sequentially from 1."""
        entries = [("99", "00:00:01,000 --> 00:00:02,000", "テスト")]
        out = str(tmp_path / "renumbered.srt")
        write_srt(entries, out)
        result = parse_srt(out)
        assert result[0][0] == "1"

    def test_text_preserved(self, tmp_path, fake_srt):
        original = parse_srt(fake_srt)
        out = str(tmp_path / "text_check.srt")
        write_srt(original, out)
        reparsed = parse_srt(out)
        for (_, _, orig_text), (_, _, new_text) in zip(original, reparsed):
            assert orig_text == new_text

    def test_empty_entries_writes_empty_file(self, tmp_path):
        out = str(tmp_path / "empty.srt")
        write_srt([], out)
        result = parse_srt(out)
        assert result == []
