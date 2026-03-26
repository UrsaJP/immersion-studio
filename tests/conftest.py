"""Shared pytest fixtures for Immersion Studio Phase 1 tests.

Provides:
    tmp_db          — DatabaseManager pointing at a fresh in-memory (or tmp) DB
                      with V1 schema initialised.
    fake_srt        — Path to a minimal 3-entry SRT file in a tmp directory.
    fake_mkv_path   — A string path inside tmp_path (file does NOT exist on disk;
                      used to test path-construction logic without ffprobe).
"""

import pytest

from core.db import DatabaseManager, init_schema

# ── Minimal SRT content used by multiple test modules ─────────────────────────
MINIMAL_SRT = """\
1
00:00:01,000 --> 00:00:03,000
おはようございます

2
00:00:04,500 --> 00:00:06,000
今日はいい天気ですね

3
00:00:07,000 --> 00:00:09,000
ありがとうございます
"""


@pytest.fixture
def tmp_db(tmp_path):
    """Return a DatabaseManager for an isolated SQLite file with V1 schema.

    The database is created fresh for each test function.  Using a real file
    (not :memory:) ensures WAL mode, foreign key pragma, and busy_timeout are
    exercised the same way as in production.

    Yields:
        DatabaseManager: Ready-to-use db manager with all V1 tables created.
    """
    db_file = tmp_path / "test_immersion.db"
    db = DatabaseManager(db_file)
    init_schema(db)
    yield db


@pytest.fixture
def fake_srt(tmp_path):
    """Write a minimal 3-entry SRT file and return its path as a string.

    Returns:
        str: Absolute path to the SRT file.
    """
    srt_file = tmp_path / "test.srt"
    srt_file.write_text(MINIMAL_SRT, encoding="utf-8")
    return str(srt_file)


@pytest.fixture
def fake_mkv_path(tmp_path):
    """Return a string path for a hypothetical MKV file inside tmp_path.

    The file is NOT created on disk.  Use this fixture only for tests that
    inspect path-construction or error-handling without calling ffprobe.

    Returns:
        str: Path string ending in '.mkv'.
    """
    return str(tmp_path / "sample_ep01.mkv")
