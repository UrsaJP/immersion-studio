"""AIST SQLite backend: history, immersion log, settings persistence.

Single responsibility: SQLite read/write for AIST history, immersion log,
translation memory, settings, and V1 Immersion Studio schema tables.
Used by core/pipeline.py, core/translation.py, and main.py (schema init).

STATUS: extended
DIVERGES_FROM_AIST: True
Changes: Added DatabaseManager context-manager class, HistoryEntry dataclass,
         init_schema() function with all V1 tables and indexes.

# DECISION: imports updated from `aist_config` → `.config` (relative within core
# package). No logic changed. Without this change the module cannot be imported
# at all from outside the AIST bundle.
"""

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: extended
# DIVERGES_FROM_AIST: True
# Changes: Added DatabaseManager, HistoryEntry dataclass, init_schema()
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import os
import csv
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import CONFIG_DIR, CONFIG_FILE, DB_FILE, RESUME_FILE, DEFAULT_SETTINGS

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
IS_APP_DB_PATH: Path = (
    Path.home() / "Library" / "Application Support" / "ImmersionStudio" / "immersion_studio.db"
)
SQLITE_WAL_PRAGMA: str = "PRAGMA journal_mode=WAL"
SQLITE_FK_PRAGMA: str = "PRAGMA foreign_keys=ON"
SQLITE_BUSY_PRAGMA: str = "PRAGMA busy_timeout=5000"


# ── Typed data models ─────────────────────────────────────────────────────────

@dataclass
class HistoryEntry:
    """Typed wrapper for a history table row."""
    path: str
    title: str
    nwd_score: float | None
    processed_at: str | None


# ══════════════════════════════════════════════════════════════════════════════
# DatabaseManager — context-manager pattern
# ══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """Single-responsibility: manage SQLite connection lifecycle.

    Always use as context manager — never call connect() and hold the connection.
    WAL mode enabled: concurrent reads do not block writes (needed for session
    timer + UI reads).

    Usage:
        db = DatabaseManager()
        with db.connect() as conn:
            conn.execute(...)
    """

    def __init__(self, path: Path = IS_APP_DB_PATH):
        """Initialize DatabaseManager.

        Args:
            path: Path to the SQLite database file. Parent directories are
                  created automatically.
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self):
        """Yield an open, WAL-mode SQLite connection. Auto-commits or rolls back.

        Yields:
            sqlite3.Connection: Open database connection.

        Raises:
            Exception: Any sqlite3 exception — connection is rolled back and closed.
        """
        conn = sqlite3.connect(str(self._path))
        conn.execute(SQLITE_WAL_PRAGMA)
        conn.execute(SQLITE_FK_PRAGMA)
        conn.execute(SQLITE_BUSY_PRAGMA)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# V1 SCHEMA INIT
# ══════════════════════════════════════════════════════════════════════════════

def init_schema(db: DatabaseManager) -> None:
    """Initialize all V1 Immersion Studio tables and indexes.

    Uses CREATE TABLE IF NOT EXISTS and CREATE INDEX IF NOT EXISTS everywhere —
    safe to call on every app launch (idempotent).

    Args:
        db: DatabaseManager instance pointing at the app database.

    Raises:
        sqlite3.Error: If schema creation fails (e.g. disk full).
    """
    try:
        with db.connect() as conn:
            # ── Per-session tracking ───────────────────────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    media_type TEXT,
                    title TEXT,
                    source TEXT,
                    language TEXT DEFAULT 'ja',
                    lines_seen INTEGER DEFAULT 0,
                    new_cards_added INTEGER DEFAULT 0,
                    active_minutes REAL DEFAULT 0,
                    passive_minutes REAL DEFAULT 0,
                    nwd_zone TEXT
                )
            """)
            # ── Known vocabulary (sourced from AnkiMorphs export) ────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS known_vocab (
                    word TEXT PRIMARY KEY,
                    reading TEXT,
                    source TEXT,
                    added_at TEXT
                )
            """)
            # ── Per-file NWD comprehension score ──────────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS media_info (
                    path TEXT PRIMARY KEY,
                    title TEXT,
                    nwd_score REAL,
                    nwd_zone TEXT,
                    total_unique_words INTEGER,
                    unknown_word_count INTEGER,
                    nwd_calculated_at TEXT
                )
            """)
            # ── Vocabulary encountered during sessions ────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vocab_encountered (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER REFERENCES sessions(id),
                    word TEXT,
                    reading TEXT,
                    known INTEGER DEFAULT 0
                )
            """)
            # ── LLM concept definition cache ──────────────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS concept_defs (
                    word TEXT PRIMARY KEY,
                    definition TEXT,
                    generated_at TEXT
                )
            """)
            # ── Grammar explanation cache (opt-in only) ───────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS grammar_explanations (
                    sentence_hash TEXT PRIMARY KEY,
                    explanation TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            # ── Pre-tokenized subtitles (NWD mining query) ────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS media_subtitles (
                    id INTEGER PRIMARY KEY,
                    media_path TEXT REFERENCES media_info(path),
                    start_ms INTEGER,
                    end_ms INTEGER,
                    text TEXT,
                    tokens_json TEXT
                )
            """)
            # ── Translation job log ────────────────────────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS translation_jobs (
                    id INTEGER PRIMARY KEY,
                    input_path TEXT,
                    output_path TEXT,
                    provider TEXT,
                    model TEXT,
                    line_count INTEGER,
                    started_at TEXT,
                    completed_at TEXT,
                    status TEXT
                )
            """)
            # ── Chorusing practice sessions ────────────────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chorusing_sessions (
                    id INTEGER PRIMARY KEY,
                    sentence TEXT,
                    source_file TEXT,
                    timestamp_start REAL,
                    practice_date TEXT,
                    overall_score REAL,
                    pitch_score REAL,
                    timing_score REAL,
                    mfcc_score REAL,
                    pitch_weight REAL DEFAULT 0.6,
                    timing_weight REAL DEFAULT 0.25,
                    mfcc_weight REAL DEFAULT 0.15,
                    recording_path TEXT
                )
            """)

            # ── Indexes ────────────────────────────────────────────────────────
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_media_subtitles_time
                ON media_subtitles(media_path, start_ms, end_ms)
            """)
            # NOTE: idx_subtitles_token references a 'token' column which does
            # not exist in media_subtitles (tokens are stored as JSON blob in
            # tokens_json). This index is defined in the plan spec but cannot be
            # created as-is. DECISION: create it without the column reference so
            # the schema is otherwise complete; Phase 2 must add a token column
            # or use a virtual table / FTS approach for frequent-unknowns mining.
            # conn.execute(
            #     "CREATE INDEX IF NOT EXISTS idx_subtitles_token ON media_subtitles(token)"
            # )
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vocab_session
                ON vocab_encountered(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vocab_word
                ON vocab_encountered(word)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_start
                ON sessions(start_time)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chorusing_practice_date
                ON chorusing_sessions(practice_date)
            """)
        logger.info("init_schema complete")
    except Exception as e:
        logger.error("init_schema failed: %s", e, exc_info=True)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# SQLITE BACKEND  (replaces history.json — FIX #1 from v3 limitations)
# ══════════════════════════════════════════════════════════════════════════════

def db_init():
    con = sqlite3.connect(DB_FILE)
    con.execute("""CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, file TEXT, status TEXT,
        lines INTEGER, duration_s INTEGER, flags TEXT,
        mkv_out TEXT, srt_out TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS immersion_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, file TEXT, audio_seconds REAL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS translation_memory (
        jp_hash TEXT PRIMARY KEY,
        jp_text TEXT NOT NULL,
        en_text TEXT NOT NULL,
        show_name TEXT DEFAULT '',
        updated_at TEXT DEFAULT ''
    )""")
    con.commit()
    con.close()
    _migrate_json_history()


# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATION MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def _jp_hash(text: str) -> str:
    import hashlib
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def tm_lookup(jp_texts: list) -> dict:
    """Look up a list of JP texts in translation memory.
    Returns {jp_text: en_text} for cache hits."""
    if not jp_texts:
        return {}
    try:
        con = sqlite3.connect(DB_FILE)
        result = {}
        for jp in jp_texts:
            row = con.execute(
                "SELECT en_text FROM translation_memory WHERE jp_hash=?",
                (_jp_hash(jp),)
            ).fetchone()
            if row:
                result[jp] = row[0]
        con.close()
        return result
    except Exception:
        return {}


def tm_store(pairs: list, show_name: str = ""):
    """Store [(jp_text, en_text), ...] pairs in translation memory."""
    if not pairs:
        return
    try:
        con = sqlite3.connect(DB_FILE)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for jp, en in pairs:
            if not jp or not en or not jp.strip() or not en.strip():
                continue
            con.execute(
                "INSERT OR REPLACE INTO translation_memory "
                "(jp_hash, jp_text, en_text, show_name, updated_at) VALUES (?,?,?,?,?)",
                (_jp_hash(jp), jp.strip(), en.strip(), show_name, now)
            )
        con.commit()
        con.close()
    except Exception:
        pass


def tm_stats() -> dict:
    """Return translation memory statistics."""
    try:
        con = sqlite3.connect(DB_FILE)
        total = con.execute("SELECT COUNT(*) FROM translation_memory").fetchone()[0]
        shows = con.execute(
            "SELECT COUNT(DISTINCT show_name) FROM translation_memory WHERE show_name != ''"
        ).fetchone()[0]
        con.close()
        return {"total": total, "shows": shows}
    except Exception:
        return {"total": 0, "shows": 0}


def tm_clear(show_name: str = ""):
    """Clear translation memory — all or for one show."""
    try:
        con = sqlite3.connect(DB_FILE)
        if show_name:
            con.execute("DELETE FROM translation_memory WHERE show_name=?", (show_name,))
        else:
            con.execute("DELETE FROM translation_memory")
        con.commit()
        con.close()
    except Exception:
        pass


def _migrate_json_history():
    old = CONFIG_DIR / "history.json"
    if not old.exists():
        return
    try:
        with open(old) as f:
            entries = json.load(f)
        con = sqlite3.connect(DB_FILE)
        for e in entries:
            con.execute(
                "INSERT OR IGNORE INTO history (date,file,status,lines,duration_s,flags) "
                "VALUES (?,?,?,?,?,?)",
                (e.get("date",""), e.get("file",""), e.get("status",""),
                 e.get("lines",0), 0, e.get("flags",""))
            )
        con.commit()
        con.close()
        old.rename(old.with_suffix(".json.bak"))
    except Exception:
        pass


def db_add_history(date, file, status, lines, duration_s, flags, mkv_out="", srt_out=""):
    try:
        con = sqlite3.connect(DB_FILE)
        con.execute(
            "INSERT INTO history (date,file,status,lines,duration_s,flags,mkv_out,srt_out) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (date, file, status, lines, duration_s, flags, mkv_out, srt_out)
        )
        con.commit()
        con.close()
    except Exception:
        pass


def db_get_history(limit=500) -> list:
    try:
        con = sqlite3.connect(DB_FILE)
        rows = con.execute(
            "SELECT date,file,status,lines,duration_s,flags FROM history "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        con.close()
        return [{"date":r[0],"file":r[1],"status":r[2],"lines":r[3],
                 "time":f"{r[4]}s","flags":r[5]} for r in rows]
    except Exception:
        return []


def db_log_immersion(file: str, audio_seconds: float):
    try:
        con = sqlite3.connect(DB_FILE)
        con.execute("INSERT INTO immersion_log (date,file,audio_seconds) VALUES (?,?,?)",
                    (datetime.now().strftime("%Y-%m-%d"), file, audio_seconds))
        con.commit()
        con.close()
    except Exception:
        pass


def db_get_immersion_hours() -> float:
    try:
        con = sqlite3.connect(DB_FILE)
        r = con.execute("SELECT SUM(audio_seconds) FROM immersion_log").fetchone()
        con.close()
        return (r[0] or 0) / 3600
    except Exception:
        return 0.0


def db_get_immersion_by_day(days: int = 90) -> dict:
    """Return {date_str: hours} for the last N days with any immersion logged."""
    try:
        con = sqlite3.connect(DB_FILE)
        rows = con.execute(
            "SELECT date, SUM(audio_seconds) FROM immersion_log "
            "GROUP BY date ORDER BY date DESC LIMIT ?",
            (days,)
        ).fetchall()
        con.close()
        return {r[0]: r[1] / 3600 for r in rows}
    except Exception:
        return {}


def db_export_csv(path: str):
    rows = db_get_history(limit=10000)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date","file","status","lines","time","flags"])
        writer.writeheader()
        writer.writerows(rows)


def load_resume() -> set:
    try:
        with open(RESUME_FILE) as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_resume(completed: set):
    with open(RESUME_FILE, "w") as f:
        json.dump(list(completed), f)


def clear_resume():
    try:
        RESUME_FILE.unlink()
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def load_settings() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            s = json.load(f)
        for k, v in DEFAULT_SETTINGS.items():
            s.setdefault(k, v)
        return s
    except Exception:
        return dict(DEFAULT_SETTINGS)


def save_settings(s: dict):
    # Atomic write: write to a temp file first, then replace, so a mid-write
    # crash never leaves settings.json in a corrupted/truncated state.
    tmp = CONFIG_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, CONFIG_FILE)
