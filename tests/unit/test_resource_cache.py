"""Unit tests for ResourceCache (core/config.py Phase 2 Step 4)."""

import threading
import pytest
from core.config import ResourceCache


class TestResourceCache:
    def setup_method(self):
        self.cache = ResourceCache()

    # ── Basic get / populate ──────────────────────────────────────────────────

    def test_populate_called_on_first_get(self):
        calls = []
        def populate(k):
            calls.append(k)
            return "value_" + k
        result = self.cache.get("key1", populate)
        assert result == "value_key1"
        assert calls == ["key1"]

    def test_populate_not_called_on_second_get(self):
        calls = []
        def populate(k):
            calls.append(k)
            return 42
        self.cache.get("k", populate)
        self.cache.get("k", populate)
        assert len(calls) == 1

    def test_different_keys_populate_independently(self):
        populate = lambda k: k + "_val"
        assert self.cache.get("a", populate) == "a_val"
        assert self.cache.get("b", populate) == "b_val"

    # ── Invalidate / clear ────────────────────────────────────────────────────

    def test_invalidate_causes_repopulation(self):
        counter = [0]
        def populate(k):
            counter[0] += 1
            return counter[0]
        self.cache.get("k", populate)
        self.cache.invalidate("k")
        result = self.cache.get("k", populate)
        assert result == 2
        assert counter[0] == 2

    def test_invalidate_nonexistent_key_is_safe(self):
        self.cache.invalidate("does_not_exist")  # must not raise

    def test_clear_removes_all_entries(self):
        populate = lambda k: "v"
        self.cache.get("a", populate)
        self.cache.get("b", populate)
        self.cache.clear()
        assert len(self.cache) == 0

    # ── __contains__ / __len__ ────────────────────────────────────────────────

    def test_contains_true_after_get(self):
        self.cache.get("x", lambda k: 1)
        assert "x" in self.cache

    def test_contains_false_before_get(self):
        assert "missing" not in self.cache

    def test_len(self):
        populate = lambda k: 0
        self.cache.get("a", populate)
        self.cache.get("b", populate)
        assert len(self.cache) == 2

    # ── Thread safety ─────────────────────────────────────────────────────────

    def test_populate_called_once_under_concurrency(self):
        """Even under race conditions, populate() must be called exactly once."""
        call_count = [0]
        import time

        def slow_populate(k):
            call_count[0] += 1
            time.sleep(0.01)   # simulate work
            return "done"

        threads = [
            threading.Thread(target=lambda: self.cache.get("shared", slow_populate))
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count[0] == 1
        assert self.cache.get("shared", slow_populate) == "done"

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_none_value_is_cached(self):
        calls = [0]
        def populate(k):
            calls[0] += 1
            return None
        self.cache.get("null_key", populate)
        self.cache.get("null_key", populate)
        assert calls[0] == 1

    def test_exception_in_populate_propagates(self):
        def bad_populate(k):
            raise ValueError("populate error")
        with pytest.raises(ValueError, match="populate error"):
            self.cache.get("bad", bad_populate)
        # key should NOT be stored after failed populate
        assert "bad" not in self.cache
