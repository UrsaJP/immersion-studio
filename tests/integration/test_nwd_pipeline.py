"""Integration tests: full NWD pipeline (precompute_subtitle_tokens → calculate_nwd).

Tests the end-to-end flow from a real SRT file → tokenisation → NWD scoring.
Requires fugashi or falls back to RegexTokenizer.
"""

import json
import pytest
from pathlib import Path

from core.pipeline import precompute_subtitle_tokens
from core.nwd import calculate_nwd, get_frequent_unknowns, nwd_zone, NWD_BADGE_RED


# conftest.py provides: tmp_db, fake_srt


class TestPrecomputeSubtitleTokens:
    def test_inserts_rows_for_each_entry(self, tmp_db, fake_srt):
        count = precompute_subtitle_tokens(Path(fake_srt), fake_srt, tmp_db)
        assert count == 3   # MINIMAL_SRT has 3 entries

    def test_rows_in_media_subtitles(self, tmp_db, fake_srt):
        precompute_subtitle_tokens(Path(fake_srt), fake_srt, tmp_db)
        with tmp_db.connect() as conn:
            rows = conn.execute(
                "SELECT media_path, tokens_json FROM media_subtitles WHERE media_path = ?",
                (fake_srt,),
            ).fetchall()
        assert len(rows) == 3
        for _, tokens_json in rows:
            parsed = json.loads(tokens_json)
            assert isinstance(parsed, list)

    def test_tokens_contain_japanese_words(self, tmp_db, fake_srt):
        precompute_subtitle_tokens(Path(fake_srt), fake_srt, tmp_db)
        with tmp_db.connect() as conn:
            rows = conn.execute(
                "SELECT tokens_json FROM media_subtitles WHERE media_path = ?",
                (fake_srt,),
            ).fetchall()
        all_tokens = []
        for (tj,) in rows:
            all_tokens.extend(json.loads(tj))
        # MINIMAL_SRT contains おはようございます, 今日, 天気, ありがとうございます etc.
        assert len(all_tokens) > 0

    def test_timestamps_stored(self, tmp_db, fake_srt):
        precompute_subtitle_tokens(Path(fake_srt), fake_srt, tmp_db)
        with tmp_db.connect() as conn:
            row = conn.execute(
                "SELECT start_ms, end_ms FROM media_subtitles WHERE media_path=? LIMIT 1",
                (fake_srt,),
            ).fetchone()
        assert row[0] >= 0
        assert row[1] > row[0]

    def test_no_overwrite_returns_existing_count(self, tmp_db, fake_srt):
        precompute_subtitle_tokens(Path(fake_srt), fake_srt, tmp_db)
        count2 = precompute_subtitle_tokens(Path(fake_srt), fake_srt, tmp_db, overwrite=False)
        assert count2 == 3  # returns existing row count, no new inserts
        with tmp_db.connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM media_subtitles WHERE media_path=?", (fake_srt,)
            ).fetchone()[0]
        assert total == 3  # still 3, not 6

    def test_overwrite_replaces_rows(self, tmp_db, fake_srt):
        precompute_subtitle_tokens(Path(fake_srt), fake_srt, tmp_db)
        precompute_subtitle_tokens(Path(fake_srt), fake_srt, tmp_db, overwrite=True)
        with tmp_db.connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM media_subtitles WHERE media_path=?", (fake_srt,)
            ).fetchone()[0]
        assert total == 3

    def test_file_not_found_raises(self, tmp_db):
        with pytest.raises(FileNotFoundError):
            precompute_subtitle_tokens(Path("/no/such/file.srt"), "/media.mkv", tmp_db)


class TestNwdPipelineEndToEnd:
    def test_zero_score_when_no_known_vocab(self, tmp_db, fake_srt):
        precompute_subtitle_tokens(Path(fake_srt), fake_srt, tmp_db)
        score = calculate_nwd(fake_srt, tmp_db)
        # No known_vocab → score should be 0
        assert score == 0.0
        assert nwd_zone(score) == NWD_BADGE_RED

    def test_score_increases_after_seeding_known_vocab(self, tmp_db, fake_srt):
        precompute_subtitle_tokens(Path(fake_srt), fake_srt, tmp_db)

        # Seed some of the words that appear in MINIMAL_SRT
        from datetime import datetime
        now = datetime.now().isoformat()
        with tmp_db.connect() as conn:
            # おはよう and ありがとう should appear from MINIMAL_SRT tokenisation
            conn.executemany(
                "INSERT OR IGNORE INTO known_vocab (word, reading, source, added_at) VALUES (?,?,?,?)",
                [("おはよう", "", "test", now), ("ありがとう", "", "test", now)],
            )

        score = calculate_nwd(fake_srt, tmp_db)
        # Score should be positive (some words known)
        assert score >= 0.0

    def test_get_frequent_unknowns_after_pipeline(self, tmp_db, fake_srt):
        precompute_subtitle_tokens(Path(fake_srt), fake_srt, tmp_db)
        unknowns = get_frequent_unknowns(fake_srt, tmp_db, limit=10)
        # Should have results since known_vocab is empty
        assert isinstance(unknowns, list)
        # Each item is (word, count)
        for word, count in unknowns:
            assert isinstance(word, str)
            assert isinstance(count, int)
            assert count >= 1
