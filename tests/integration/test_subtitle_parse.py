"""Integration tests: SRT parse → translate_srt mock → write_srt roundtrip.

Requires PySide6 (skipped automatically when not installed).
"""

import pytest
pytest.importorskip("PySide6", reason="PySide6 not installed — skipping Qt-dependent tests")

from pathlib import Path
from unittest.mock import patch

from core.subtitle import parse_srt, write_srt
from core.translation import translate_srt, TRANSLATION_PROMPT


class TestSrtParseRoundtrip:
    def test_full_parse_write_reparse(self, fake_srt, tmp_path):
        """parse_srt → write_srt → parse_srt must preserve all entries."""
        original = parse_srt(fake_srt)
        out = str(tmp_path / "roundtrip.srt")
        write_srt(original, out)
        reparsed = parse_srt(out)

        assert len(reparsed) == len(original)
        for (_, _, orig_text), (_, _, new_text) in zip(original, reparsed):
            assert orig_text == new_text

    def test_timing_preserved_after_roundtrip(self, fake_srt, tmp_path):
        original = parse_srt(fake_srt)
        out = str(tmp_path / "timing.srt")
        write_srt(original, out)
        reparsed = parse_srt(out)

        for (_, orig_timing, _), (_, new_timing, _) in zip(original, reparsed):
            # Timings must both contain '-->'
            assert "-->" in orig_timing
            assert "-->" in new_timing


class TestTranslateSrtMocked:
    """translate_srt with AI provider mocked — verifies pipeline wiring."""

    def test_translates_all_entries(self, fake_srt, tmp_path):
        with patch("core.translation.call_ai_provider", return_value="translation") as mock_ai:
            result = translate_srt(
                Path(fake_srt),
                provider_id="ollama",
                model="qwen2.5:7b",
            )
        entries = parse_srt(str(result))
        assert len(entries) == 3  # all 3 source entries translated

    def test_translated_text_written_to_file(self, fake_srt, tmp_path):
        with patch("core.translation.call_ai_provider", return_value="good morning"):
            result = translate_srt(
                Path(fake_srt),
                provider_id="ollama",
                model="qwen2.5:7b",
            )
        entries = parse_srt(str(result))
        texts = [t for _, _, t in entries]
        assert all("good morning" in t for t in texts)

    def test_progress_callback_called_for_each_entry(self, fake_srt):
        calls = []
        def _cb(cur, tot):
            calls.append((cur, tot))

        with patch("core.translation.call_ai_provider", return_value="text"):
            translate_srt(
                Path(fake_srt),
                provider_id="ollama",
                model="qwen2.5:7b",
                progress_cb=_cb,
            )
        assert len(calls) == 3
        # Final call: current == total
        last_cur, last_tot = calls[-1]
        assert last_cur == last_tot

    def test_output_path_ends_with_en_srt(self, fake_srt):
        with patch("core.translation.call_ai_provider", return_value="x"):
            result = translate_srt(
                Path(fake_srt),
                provider_id="ollama",
                model="qwen2.5:7b",
            )
        assert str(result).endswith(".en.srt")

    def test_ai_provider_called_with_prompt(self, fake_srt):
        with patch("core.translation.call_ai_provider", return_value="y") as mock_ai:
            translate_srt(
                Path(fake_srt),
                provider_id="anthropic",
                model="claude-haiku-4-5-20251001",
                api_key="test-key",
            )
        # Provider and model must be forwarded to call_ai_provider
        assert mock_ai.call_count == 3
        for call in mock_ai.call_args_list:
            args, kwargs = call
            assert args[0] == "anthropic"
            assert args[1] == "claude-haiku-4-5-20251001"
            assert kwargs.get("api_key") == "test-key"
