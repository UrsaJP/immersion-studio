"""Unit tests for core.config model cache helpers: _is_cache_fresh."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from core.config import _is_cache_fresh, MODEL_CACHE_TTL


class TestIsCacheFresh:
    def _make_ts(self, delta: timedelta) -> str:
        return (datetime.now() + delta).isoformat()

    def test_fresh_ollama_cache_is_always_stale(self):
        """Ollama TTL is 0, meaning always fetch — never use cache."""
        ts = self._make_ts(timedelta(minutes=-1))
        result = _is_cache_fresh("ollama", ts)
        assert result is False, "Ollama TTL=0 must always return False (always fetch)"

    def test_fresh_openai_cache(self):
        """OpenAI/cloud providers: recent fetched_at should be considered fresh."""
        ts = self._make_ts(timedelta(hours=-1))
        result = _is_cache_fresh("openai", ts)
        assert result is True

    def test_stale_openai_cache(self):
        """A fetched_at far in the past must be stale."""
        ts = self._make_ts(timedelta(days=-30))
        result = _is_cache_fresh("openai", ts)
        assert result is False

    def test_corrupt_timestamp_returns_false(self):
        """A malformed fetched_at should not raise and should return False."""
        result = _is_cache_fresh("openai", "not-a-valid-iso-date")
        assert result is False

    def test_empty_timestamp_returns_false(self):
        result = _is_cache_fresh("openai", "")
        assert result is False

    def test_model_cache_ttl_dict_exists(self):
        """MODEL_CACHE_TTL must be a dict with at least ollama=0."""
        assert isinstance(MODEL_CACHE_TTL, dict)
        assert MODEL_CACHE_TTL.get("ollama") == 0

    def test_unknown_provider_uses_default_ttl(self):
        """Providers not in MODEL_CACHE_TTL default to 14 days."""
        ts_fresh = self._make_ts(timedelta(hours=-1))
        # Should be fresh (within 14d default)
        result = _is_cache_fresh("totally_unknown_provider", ts_fresh)
        assert result is True
