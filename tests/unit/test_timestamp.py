"""Unit tests for SRT/ASS timestamp conversion helpers in core.utils."""

import pytest

from core.utils import ms_to_srt_time, srt_time_to_ms


# ══════════════════════════════════════════════════════════════════════════════
# srt_time_to_ms
# ══════════════════════════════════════════════════════════════════════════════

class TestSrtTimeToMs:
    def test_zero(self):
        assert srt_time_to_ms("00:00:00,000") == 0

    def test_one_second(self):
        assert srt_time_to_ms("00:00:01,000") == 1000

    def test_minutes(self):
        assert srt_time_to_ms("00:01:00,000") == 60_000

    def test_hours(self):
        assert srt_time_to_ms("01:00:00,000") == 3_600_000

    def test_mixed(self):
        # 1h 2m 3s 456ms
        assert srt_time_to_ms("01:02:03,456") == 3_723_456

    def test_dot_separator(self):
        """Some SRT writers use '.' instead of ',' for milliseconds."""
        assert srt_time_to_ms("00:00:01.500") == 1500

    def test_corrupt_returns_zero(self):
        """Malformed timestamps must return 0, not raise."""
        assert srt_time_to_ms("not-a-timestamp") == 0

    def test_empty_string(self):
        assert srt_time_to_ms("") == 0


# ══════════════════════════════════════════════════════════════════════════════
# ms_to_srt_time
# ══════════════════════════════════════════════════════════════════════════════

class TestMsToSrtTime:
    def test_zero(self):
        assert ms_to_srt_time(0) == "00:00:00,000"

    def test_one_second(self):
        assert ms_to_srt_time(1000) == "00:00:01,000"

    def test_minutes(self):
        assert ms_to_srt_time(60_000) == "00:01:00,000"

    def test_hours(self):
        assert ms_to_srt_time(3_600_000) == "01:00:00,000"

    def test_mixed(self):
        assert ms_to_srt_time(3_723_456) == "01:02:03,456"

    def test_sub_millisecond_truncated(self):
        """Fractional milliseconds should be truncated, not rounded."""
        assert ms_to_srt_time(1001) == "00:00:01,001"


# ══════════════════════════════════════════════════════════════════════════════
# round-trip
# ══════════════════════════════════════════════════════════════════════════════

class TestRoundTrip:
    @pytest.mark.parametrize("ms", [0, 500, 1000, 60_000, 3_600_000, 3_723_456, 86_399_999])
    def test_roundtrip(self, ms):
        assert srt_time_to_ms(ms_to_srt_time(ms)) == ms
