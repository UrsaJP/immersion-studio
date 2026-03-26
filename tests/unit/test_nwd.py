"""Unit tests for core/nwd.py.

Tests: nwd_zone, calculate_nwd, get_frequent_unknowns,
       import_ankimorph_vocab, import_seed_vocab.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime

from core.nwd import (
    NWD_BADGE_GREEN,
    NWD_BADGE_RED,
    NWD_BADGE_YELLOW,
    NWD_ZONE_GREEN_THRESHOLD,
    NWD_ZONE_YELLOW_THRESHOLD,
    calculate_nwd,
    get_frequent_unknowns,
    import_ankimorph_vocab,
    import_seed_vocab,
    nwd_zone,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

# conftest.py provides: tmp_db


# ══════════════════════════════════════════════════════════════════════════════
# nwd_zone
# ══════════════════════════════════════════════════════════════════════════════

class TestNwdZone:
    def test_green_threshold(self):
        assert nwd_zone(NWD_ZONE_GREEN_THRESHOLD) == NWD_BADGE_GREEN

    def test_perfect_score_green(self):
        assert nwd_zone(1.0) == NWD_BADGE_GREEN

    def test_yellow_threshold(self):
        assert nwd_zone(NWD_ZONE_YELLOW_THRESHOLD) == NWD_BADGE_YELLOW

    def test_just_below_green_is_yellow(self):
        assert nwd_zone(NWD_ZONE_GREEN_THRESHOLD - 0.001) == NWD_BADGE_YELLOW

    def test_zero_is_red(self):
        assert nwd_zone(0.0) == NWD_BADGE_RED

    def test_just_below_yellow_is_red(self):
        assert nwd_zone(NWD_ZONE_YELLOW_THRESHOLD - 0.001) == NWD_BADGE_RED


# ══════════════════════════════════════════════════════════════════════════════
# calculate_nwd
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculateNwd:
    def _insert_tokens(self, db, media_path, tokens_list):
        """Helper: insert rows into media_subtitles (creates media_info row first)."""
        with db.connect() as conn:
            conn.execute(
                "INSERT INTO media_info (path) VALUES (?) ON CONFLICT(path) DO NOTHING",
                (media_path,),
            )
            conn.executemany(
                "INSERT INTO media_subtitles (media_path, start_ms, end_ms, text, tokens_json) VALUES (?,?,?,?,?)",
                [(media_path, i * 1000, i * 1000 + 500, f"text{i}", json.dumps(tokens))
                 for i, tokens in enumerate(tokens_list)],
            )

    def _insert_known(self, db, words):
        now = datetime.now().isoformat()
        with db.connect() as conn:
            conn.executemany(
                "INSERT INTO known_vocab (word, reading, source, added_at) VALUES (?,?,?,?)",
                [(w, "", "test", now) for w in words],
            )

    def test_no_tokens_returns_zero(self, tmp_db):
        score = calculate_nwd("/test/video.mkv", tmp_db)
        assert score == 0.0

    def test_all_known_returns_one(self, tmp_db):
        media = "/test/video.mkv"
        self._insert_tokens(tmp_db, media, [["食べる", "飲む"], ["走る"]])
        self._insert_known(tmp_db, ["食べる", "飲む", "走る"])
        score = calculate_nwd(media, tmp_db)
        assert score == 1.0

    def test_none_known_returns_zero(self, tmp_db):
        media = "/test/video.mkv"
        self._insert_tokens(tmp_db, media, [["食べる", "飲む"]])
        score = calculate_nwd(media, tmp_db)
        assert score == 0.0

    def test_half_known(self, tmp_db):
        media = "/test/video.mkv"
        self._insert_tokens(tmp_db, media, [["食べる", "飲む", "走る", "見る"]])
        self._insert_known(tmp_db, ["食べる", "飲む"])
        score = calculate_nwd(media, tmp_db)
        assert 0.49 <= score <= 0.51

    def test_persists_to_media_info(self, tmp_db):
        media = "/test/video.mkv"
        self._insert_tokens(tmp_db, media, [["食べる"]])
        self._insert_known(tmp_db, ["食べる"])
        calculate_nwd(media, tmp_db)
        with tmp_db.connect() as conn:
            row = conn.execute(
                "SELECT nwd_score, nwd_zone FROM media_info WHERE path = ?", (media,)
            ).fetchone()
        assert row is not None
        assert row[0] == 1.0
        assert row[1] == NWD_BADGE_GREEN

    def test_non_japanese_tokens_excluded(self, tmp_db):
        media = "/test/video.mkv"
        # Only ASCII tokens — should count as no JP words
        self._insert_tokens(tmp_db, media, [["hello", "world"]])
        score = calculate_nwd(media, tmp_db)
        assert score == 0.0

    def test_upsert_updates_existing_media_info(self, tmp_db):
        media = "/test/video.mkv"
        self._insert_tokens(tmp_db, media, [["食べる"]])
        calculate_nwd(media, tmp_db)  # first score: 0.0 (no known vocab)
        self._insert_known(tmp_db, ["食べる"])
        score = calculate_nwd(media, tmp_db)  # re-score: should be 1.0
        assert score == 1.0

    def test_malformed_tokens_json_skipped(self, tmp_db):
        media = "/test/video.mkv"
        with tmp_db.connect() as conn:
            conn.execute(
                "INSERT INTO media_info (path) VALUES (?) ON CONFLICT(path) DO NOTHING",
                (media,),
            )
            conn.execute(
                "INSERT INTO media_subtitles (media_path, start_ms, end_ms, text, tokens_json) VALUES (?,?,?,?,?)",
                (media, 0, 500, "text", "NOT_VALID_JSON"),
            )
        score = calculate_nwd(media, tmp_db)
        assert score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# get_frequent_unknowns
# ══════════════════════════════════════════════════════════════════════════════

class TestGetFrequentUnknowns:
    def _insert_tokens(self, db, media_path, tokens_list):
        with db.connect() as conn:
            conn.execute(
                "INSERT INTO media_info (path) VALUES (?) ON CONFLICT(path) DO NOTHING",
                (media_path,),
            )
            conn.executemany(
                "INSERT INTO media_subtitles (media_path, start_ms, end_ms, text, tokens_json) VALUES (?,?,?,?,?)",
                [(media_path, i * 1000, i * 1000 + 500, "", json.dumps(t))
                 for i, t in enumerate(tokens_list)],
            )

    def _insert_known(self, db, words):
        now = datetime.now().isoformat()
        with db.connect() as conn:
            conn.executemany(
                "INSERT INTO known_vocab (word, reading, source, added_at) VALUES (?,?,?,?)",
                [(w, "", "test", now) for w in words],
            )

    def test_returns_empty_list_when_no_data(self, tmp_db):
        result = get_frequent_unknowns("/nope.mkv", tmp_db)
        assert result == []

    def test_known_words_excluded(self, tmp_db):
        media = "/test/video.mkv"
        self._insert_tokens(tmp_db, media, [["食べる", "飲む"], ["食べる"]])
        self._insert_known(tmp_db, ["食べる"])
        unknowns = get_frequent_unknowns(media, tmp_db)
        words = [w for w, _ in unknowns]
        assert "食べる" not in words
        assert "飲む" in words

    def test_sorted_by_frequency(self, tmp_db):
        media = "/test/video.mkv"
        self._insert_tokens(tmp_db, media, [
            ["走る", "飲む", "走る"],
            ["走る", "見る"],
        ])
        unknowns = get_frequent_unknowns(media, tmp_db)
        assert unknowns[0][0] == "走る"
        assert unknowns[0][1] == 3

    def test_limit_respected(self, tmp_db):
        media = "/test/video.mkv"
        words = [f"単語{i}" for i in range(20)]
        self._insert_tokens(tmp_db, media, [words])
        unknowns = get_frequent_unknowns(media, tmp_db, limit=5)
        assert len(unknowns) <= 5


# ══════════════════════════════════════════════════════════════════════════════
# import_ankimorph_vocab
# ══════════════════════════════════════════════════════════════════════════════

class TestImportAnkimorphVocab:
    def _make_tsv(self, tmp_path, content):
        p = tmp_path / "morph_export.tsv"
        p.write_text(content, encoding="utf-8")
        return str(p)

    def test_file_not_found(self, tmp_db):
        with pytest.raises(FileNotFoundError):
            import_ankimorph_vocab("/no/such/file.tsv", tmp_db)

    def test_basic_import(self, tmp_path, tmp_db):
        tsv = self._make_tsv(tmp_path, "食べる\t食べる\n飲む\t飲む\n")
        count = import_ankimorph_vocab(tsv, tmp_db)
        assert count == 2
        with tmp_db.connect() as conn:
            rows = conn.execute("SELECT word FROM known_vocab ORDER BY word").fetchall()
        words = {r[0] for r in rows}
        assert "食べる" in words
        assert "飲む" in words

    def test_header_row_skipped(self, tmp_path, tmp_db):
        tsv = self._make_tsv(tmp_path, "morph\treading\n食べる\t食べる\n")
        count = import_ankimorph_vocab(tsv, tmp_db)
        assert count == 1

    def test_comment_lines_skipped(self, tmp_path, tmp_db):
        tsv = self._make_tsv(tmp_path, "# comment\n食べる\n")
        count = import_ankimorph_vocab(tsv, tmp_db)
        assert count == 1

    def test_upsert_updates_existing(self, tmp_path, tmp_db):
        tsv = self._make_tsv(tmp_path, "食べる\t食べる\n")
        import_ankimorph_vocab(tsv, tmp_db)
        tsv2 = self._make_tsv(tmp_path, "食べる\tたべる\n")
        count = import_ankimorph_vocab(tsv2, tmp_db)
        assert count == 1
        with tmp_db.connect() as conn:
            row = conn.execute("SELECT reading FROM known_vocab WHERE word='食べる'").fetchone()
        assert row[0] == "たべる"

    def test_empty_file_returns_zero(self, tmp_path, tmp_db):
        tsv = self._make_tsv(tmp_path, "")
        count = import_ankimorph_vocab(tsv, tmp_db)
        assert count == 0

    def test_source_set_to_ankimorph(self, tmp_path, tmp_db):
        tsv = self._make_tsv(tmp_path, "食べる\n")
        import_ankimorph_vocab(tsv, tmp_db)
        with tmp_db.connect() as conn:
            row = conn.execute("SELECT source FROM known_vocab WHERE word='食べる'").fetchone()
        assert row[0] == "ankimorph"


# ══════════════════════════════════════════════════════════════════════════════
# import_seed_vocab
# ══════════════════════════════════════════════════════════════════════════════

class TestImportSeedVocab:
    def _make_seed(self, tmp_path, content):
        p = tmp_path / "jlpt_n5.txt"
        p.write_text(content, encoding="utf-8")
        return str(p)

    def test_file_not_found(self, tmp_db):
        with pytest.raises(FileNotFoundError):
            import_seed_vocab("/no/such/file.txt", tmp_db)

    def test_basic_one_per_line(self, tmp_path, tmp_db):
        seed = self._make_seed(tmp_path, "食べる\n飲む\n走る\n")
        count = import_seed_vocab(seed, tmp_db)
        assert count == 3

    def test_tsv_two_columns(self, tmp_path, tmp_db):
        seed = self._make_seed(tmp_path, "食べる\tたべる\n飲む\tのむ\n")
        import_seed_vocab(seed, tmp_db)
        with tmp_db.connect() as conn:
            row = conn.execute("SELECT reading FROM known_vocab WHERE word='食べる'").fetchone()
        assert row[0] == "たべる"

    def test_custom_source_label(self, tmp_path, tmp_db):
        seed = self._make_seed(tmp_path, "食べる\n")
        import_seed_vocab(seed, tmp_db, source="custom_seed")
        with tmp_db.connect() as conn:
            row = conn.execute("SELECT source FROM known_vocab WHERE word='食べる'").fetchone()
        assert row[0] == "custom_seed"

    def test_comment_lines_skipped(self, tmp_path, tmp_db):
        seed = self._make_seed(tmp_path, "# N5 vocab\n食べる\n")
        count = import_seed_vocab(seed, tmp_db)
        assert count == 1

    def test_empty_lines_skipped(self, tmp_path, tmp_db):
        seed = self._make_seed(tmp_path, "食べる\n\n\n飲む\n")
        count = import_seed_vocab(seed, tmp_db)
        assert count == 2

    def test_empty_file_returns_zero(self, tmp_path, tmp_db):
        seed = self._make_seed(tmp_path, "")
        count = import_seed_vocab(seed, tmp_db)
        assert count == 0

    def test_upsert_does_not_duplicate(self, tmp_path, tmp_db):
        seed = self._make_seed(tmp_path, "食べる\n")
        import_seed_vocab(seed, tmp_db)
        import_seed_vocab(seed, tmp_db)
        with tmp_db.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM known_vocab WHERE word='食べる'").fetchone()[0]
        assert count == 1
