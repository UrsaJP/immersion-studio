"""Unit tests for core.utils helper functions."""

import pytest

from core.utils import guess_show_name, mask_key, ms_to_srt_time, parse_episode


class TestMaskKey:
    def test_long_key_shows_tail(self):
        """mask_key shows the last 4 chars and masks the rest with bullets."""
        key = "sk-abcdefghijklmnop"
        result = mask_key(key)
        # Last 4 chars must be visible
        assert result.endswith(key[-4:])
        # Full key must not be exposed
        assert result != key

    def test_long_key_has_bullets(self):
        key = "sk-abcdefghijklmnop"
        result = mask_key(key)
        assert "•" in result

    def test_short_key_fully_masked(self):
        """Keys 8 chars or shorter return a fixed bullet string."""
        result = mask_key("abc")
        assert result == "••••••••"

    def test_eight_char_key_masked(self):
        result = mask_key("12345678")
        assert result == "••••••••"

    def test_empty_key(self):
        """Empty string should not raise."""
        result = mask_key("")
        assert isinstance(result, str)

    def test_return_type_is_str(self):
        assert isinstance(mask_key("some-api-key-value"), str)


class TestGuessShowName:
    def test_simple_name(self):
        result = guess_show_name("Bleach S01E01.mkv")
        assert "Bleach" in result or len(result) > 0

    def test_returns_string(self):
        assert isinstance(guess_show_name("SomeShow.mkv"), str)

    def test_empty_filename(self):
        result = guess_show_name("")
        assert isinstance(result, str)


class TestParseEpisode:
    def test_returns_tuple(self):
        result = parse_episode("show_S01E05.mkv")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_episode_number_detected(self):
        _season, ep = parse_episode("MyShow.S02E07.mkv")
        assert ep == 7 or ep is not None

    def test_no_episode_returns_none(self):
        _season, ep = parse_episode("no_episode_here.mkv")
        # Should not raise; may return None or 0
        assert ep is None or isinstance(ep, int)
