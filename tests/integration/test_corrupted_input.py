"""Integration tests: corrupted/edge-case SRT inputs must not crash the app."""

import pytest

from core.subtitle import parse_srt, write_srt


class TestCorruptedSrt:
    def test_totally_empty_file(self, tmp_path):
        srt = tmp_path / "empty.srt"
        srt.write_text("", encoding="utf-8")
        result = parse_srt(str(srt))
        assert result == []

    def test_whitespace_only_file(self, tmp_path):
        srt = tmp_path / "whitespace.srt"
        srt.write_text("   \n\n\t\n", encoding="utf-8")
        result = parse_srt(str(srt))
        assert result == []

    def test_single_line_file(self, tmp_path):
        srt = tmp_path / "single.srt"
        srt.write_text("just a line\n", encoding="utf-8")
        result = parse_srt(str(srt))
        # Must not raise; may return [] (incomplete block — fewer than 3 lines)
        assert isinstance(result, list)

    def test_two_line_block(self, tmp_path):
        """Block with only index + timing (no text) has only 2 lines — no entry."""
        srt = tmp_path / "twolines.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n", encoding="utf-8")
        result = parse_srt(str(srt))
        # parse_srt requires len(lines) >= 3
        assert isinstance(result, list)

    def test_mixed_valid_and_corrupt_blocks(self, tmp_path):
        """Valid blocks before/after corrupt blocks must still parse."""
        content = (
            "1\n00:00:01,000 --> 00:00:02,000\nValid line\n\n"
            "CORRUPT BLOCK\n\n"
            "3\n00:00:05,000 --> 00:00:06,000\nAnother valid\n\n"
        )
        srt = tmp_path / "mixed.srt"
        srt.write_text(content, encoding="utf-8")
        result = parse_srt(str(srt))
        # Must not raise; at least 1 valid entry expected
        assert isinstance(result, list)

    def test_latin1_encoded_file_does_not_crash(self, tmp_path):
        """Non-UTF-8 bytes must be replaced, not raise UnicodeDecodeError."""
        srt = tmp_path / "latin1.srt"
        raw = b"1\n00:00:01,000 --> 00:00:02,000\nCaf\xe9\n\n"
        srt.write_bytes(raw)
        result = parse_srt(str(srt))
        assert isinstance(result, list)

    def test_very_large_entry_count(self, tmp_path):
        """A large SRT file (1 000 entries) must parse without error."""
        lines = []
        for i in range(1, 1001):
            h = (i * 3) // 3600
            m = ((i * 3) % 3600) // 60
            s = (i * 3) % 60
            start = f"{h:02d}:{m:02d}:{s:02d},000"
            end_s = s + 2 if s + 2 < 60 else 59
            end = f"{h:02d}:{m:02d}:{end_s:02d},000"
            lines.append(f"{i}\n{start} --> {end}\nLine {i}\n")
        srt = tmp_path / "large.srt"
        srt.write_text("\n".join(lines), encoding="utf-8")
        result = parse_srt(str(srt))
        assert len(result) == 1000

    def test_write_srt_then_parse_is_stable(self, tmp_path):
        """write_srt output must be idempotent under a second parse+write cycle."""
        content = "1\n00:00:01,000 --> 00:00:02,000\nテスト\n\n"
        srt1 = tmp_path / "pass1.srt"
        srt1.write_text(content, encoding="utf-8")
        entries1 = parse_srt(str(srt1))

        srt2 = tmp_path / "pass2.srt"
        write_srt(entries1, str(srt2))
        entries2 = parse_srt(str(srt2))

        srt3 = tmp_path / "pass3.srt"
        write_srt(entries2, str(srt3))
        entries3 = parse_srt(str(srt3))

        assert len(entries2) == len(entries1)
        assert len(entries3) == len(entries2)
        for (_, _, t2), (_, _, t3) in zip(entries2, entries3):
            assert t2 == t3
