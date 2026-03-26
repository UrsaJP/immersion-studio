"""Immersion Studio — NWD (N+1 Word Density) scoring and vocabulary management.

Single responsibility: calculate per-media comprehension scores, surface the
most-frequent unknown words, and import vocabulary from AnkiMorphs TSV or
plain-text seed files into the ``known_vocab`` table.

NWD score = known_unique_words / total_unique_words (0.0 – 1.0)
Zone badges:
    🟢  >= NWD_ZONE_GREEN_THRESHOLD  (85 %)
    🟡  >= NWD_ZONE_YELLOW_THRESHOLD (60 %)
    🔴  <  NWD_ZONE_YELLOW_THRESHOLD

Data sources:
    media_subtitles.tokens_json  — pre-tokenised by core/pipeline.py
    known_vocab                  — imported via import_ankimorph_vocab() /
                                   import_seed_vocab()

Used by: ui/widgets/tracker_widget.py, core/pipeline.py.

STATUS: new
DIVERGES_FROM_AIST: True
Changes: written from scratch per Phase 2 Step 5 spec.
"""

from __future__ import annotations

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: new
# DIVERGES_FROM_AIST: True
# Changes: written from scratch.
# ─────────────────────────────────────────────────────────────────────────────

import csv
import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import DatabaseManager

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
NWD_ZONE_GREEN_THRESHOLD:  float = 0.85   # >= green badge
NWD_ZONE_YELLOW_THRESHOLD: float = 0.60   # >= yellow badge; < red badge

NWD_BADGE_GREEN:  str = "🟢"
NWD_BADGE_YELLOW: str = "🟡"
NWD_BADGE_RED:    str = "🔴"

# AnkiMorphs TSV column indices (0-based)
ANKIMORPH_WORD_COL:    int = 0
ANKIMORPH_READING_COL: int = 1   # may be absent in some exports

# Minimum length for a token to be treated as a vocabulary word
MIN_TOKEN_LEN: int = 1

# Japanese kanji/kana filter — only count real JP tokens
_JP_RE: re.Pattern = re.compile(r"[぀-ヿ一-鿿]")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def nwd_zone(score: float) -> str:
    """Return the zone badge string for a given NWD *score*.

    Args:
        score: NWD score in [0.0, 1.0].

    Returns:
        One of NWD_BADGE_GREEN / NWD_BADGE_YELLOW / NWD_BADGE_RED.
    """
    if score >= NWD_ZONE_GREEN_THRESHOLD:
        return NWD_BADGE_GREEN
    if score >= NWD_ZONE_YELLOW_THRESHOLD:
        return NWD_BADGE_YELLOW
    return NWD_BADGE_RED


def calculate_nwd(media_path: str, db: DatabaseManager) -> float:
    """Compute the NWD score for *media_path* and persist it to ``media_info``.

    Reads pre-tokenised subtitles from ``media_subtitles``, looks up each
    unique lemma in ``known_vocab``, computes the ratio of known unique words
    to total unique words, then writes the result back to ``media_info``.

    Args:
        media_path: Absolute path of the media file (FK into ``media_info``).
        db:         Open ``DatabaseManager`` instance.

    Returns:
        NWD score as a float in [0.0, 1.0].  Returns 0.0 if no subtitle tokens
        are found for the file.
    """
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT tokens_json FROM media_subtitles WHERE media_path = ?",
            (media_path,),
        ).fetchall()

    if not rows:
        logger.warning("calculate_nwd: no subtitle tokens found for %s", media_path)
        return 0.0

    unique_words: set[str] = set()
    for (tokens_json,) in rows:
        try:
            tokens: list[str] = json.loads(tokens_json) if tokens_json else []
        except (json.JSONDecodeError, TypeError):
            logger.warning("calculate_nwd: malformed tokens_json for %s", media_path)
            continue
        for token in tokens:
            if token and len(token) >= MIN_TOKEN_LEN and _JP_RE.search(token):
                unique_words.add(token)

    known_count = 0
    if not unique_words:
        logger.info("calculate_nwd: no Japanese tokens in subtitles for %s", media_path)
        score = 0.0
    else:
        with db.connect() as conn:
            # Use a single IN query with up to SQLITE_MAX_VARIABLE_NUMBER tokens.
            # For safety chunk into batches of 900 if the vocab is huge.
            known_count = _count_known(conn, unique_words)

        total = len(unique_words)
        score = round(known_count / total, 4)
        logger.info(
            "calculate_nwd: %s → score=%.4f (%d/%d unique words known)",
            media_path, score, known_count, total,
        )

    zone = nwd_zone(score)
    now  = datetime.now().isoformat()
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO media_info (path, nwd_score, nwd_zone, total_unique_words,
                                    unknown_word_count, nwd_calculated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                nwd_score            = excluded.nwd_score,
                nwd_zone             = excluded.nwd_zone,
                total_unique_words   = excluded.total_unique_words,
                unknown_word_count   = excluded.unknown_word_count,
                nwd_calculated_at    = excluded.nwd_calculated_at
            """,
            (
                media_path,
                score,
                zone,
                len(unique_words),
                len(unique_words) - known_count,
                now,
            ),
        )
    return score


def get_frequent_unknowns(
    media_path: str,
    db: DatabaseManager,
    limit: int = 10,
) -> list[tuple[str, int]]:
    """Return the most-frequent unknown words for *media_path*.

    A word is "unknown" if it is present in ``media_subtitles.tokens_json``
    but absent from ``known_vocab``.

    Args:
        media_path: Absolute path of the media file.
        db:         Open ``DatabaseManager`` instance.
        limit:      Maximum number of words to return.

    Returns:
        List of (word, occurrence_count) sorted by frequency descending.
        Empty list if no data is available.
    """
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT tokens_json FROM media_subtitles WHERE media_path = ?",
            (media_path,),
        ).fetchall()

    counter: Counter = Counter()
    for (tokens_json,) in rows:
        try:
            tokens = json.loads(tokens_json) if tokens_json else []
        except (json.JSONDecodeError, TypeError):
            continue
        for token in tokens:
            if token and _JP_RE.search(token):
                counter[token] += 1

    if not counter:
        return []

    candidates = list(counter.keys())
    with db.connect() as conn:
        known = _fetch_known_set(conn, set(candidates))

    result = [
        (word, count)
        for word, count in counter.most_common()
        if word not in known
    ]
    return result[:limit]


def import_ankimorph_vocab(tsv_path: str | Path, db: DatabaseManager) -> int:
    """Import a AnkiMorphs TSV export into ``known_vocab``.

    AnkiMorphs exports a TSV where:
        column 0 — the base/dictionary form (lemma)
        column 1 — kana reading (optional)

    Lines starting with ``#`` and the header row (``Morph`` / ``Word``) are
    skipped automatically.  Existing words are updated (UPSERT).

    Args:
        tsv_path: Path to the AnkiMorphs TSV export file.
        db:       Open ``DatabaseManager`` instance.

    Returns:
        Number of words upserted.

    Raises:
        FileNotFoundError: If *tsv_path* does not exist.
        RuntimeError:      If the file cannot be parsed.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"AnkiMorphs TSV not found: {tsv_path}")

    rows: list[tuple[str, str, str, str]] = []
    now = datetime.now().isoformat()

    try:
        with open(tsv_path, encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                if not line:
                    continue
                word = line[ANKIMORPH_WORD_COL].strip()
                if not word or word.startswith("#"):
                    continue
                # Skip typical header rows
                if word.lower() in ("morph", "word", "lemma", "base"):
                    continue
                reading = (
                    line[ANKIMORPH_READING_COL].strip()
                    if len(line) > ANKIMORPH_READING_COL
                    else ""
                )
                rows.append((word, reading, "ankimorph", now))
    except Exception as exc:
        raise RuntimeError(f"import_ankimorph_vocab: failed to read {tsv_path}: {exc}") from exc

    if not rows:
        logger.warning("import_ankimorph_vocab: no rows found in %s", tsv_path)
        return 0

    with db.connect() as conn:
        conn.executemany(
            """
            INSERT INTO known_vocab (word, reading, source, added_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(word) DO UPDATE SET
                reading  = excluded.reading,
                source   = excluded.source,
                added_at = excluded.added_at
            """,
            rows,
        )

    logger.info("import_ankimorph_vocab: upserted %d words from %s", len(rows), tsv_path)
    return len(rows)


# ── AnkiMorphs DB path helpers ────────────────────────────────────────────────

ANKIMORPH_DB_FILENAME: str = "ankimorphs.db"
ANKI2_BASE: Path = Path.home() / "Library" / "Application Support" / "Anki2"


def find_ankimorph_db() -> Path | None:
    """Search for the AnkiMorphs database under ``~/Library/Application Support/Anki2``.

    Scans all profile subdirectories for ``ankimorphs.db``.  Returns the first
    match found (alphabetically by profile name), or ``None`` if not found.

    Returns:
        Path to ``ankimorphs.db``, or ``None``.
    """
    if not ANKI2_BASE.exists():
        return None
    for candidate in sorted(ANKI2_BASE.iterdir()):
        db = candidate / ANKIMORPH_DB_FILENAME
        if db.exists():
            return db
    return None


def import_ankimorph_db(
    db: DatabaseManager,
    ankimorph_db_path: str | Path | None = None,
    min_interval: int = 0,
) -> int:
    """Import known morphs directly from the AnkiMorphs SQLite database.

    Reads the ``Morphs`` table from the AnkiMorphs database and upserts each
    morph lemma into ``known_vocab``.  Both the lemma and its inflection are
    stored so NWD scoring can match conjugated surface forms.

    Auto-detects the database path if *ankimorph_db_path* is not given by
    calling :func:`find_ankimorph_db`.

    Schema read:
        ``Morphs(lemma TEXT PK, inflection TEXT PK,
                 highest_lemma_learning_interval INTEGER,
                 highest_inflection_learning_interval INTEGER)``

    Args:
        db:                 Open ``DatabaseManager`` instance.
        ankimorph_db_path:  Explicit path to ``ankimorphs.db``.  If ``None``,
                            auto-detected via :func:`find_ankimorph_db`.
        min_interval:       Only import morphs whose
                            ``highest_lemma_learning_interval`` is at least
                            this value.  ``0`` imports all morphs (any seen
                            card).  Use ``21`` for mature-only.

    Returns:
        Number of words upserted into ``known_vocab``.

    Raises:
        FileNotFoundError: If the AnkiMorphs DB cannot be found.
        RuntimeError:      If the DB cannot be read.
    """
    import sqlite3 as _sqlite3

    if ankimorph_db_path is None:
        ankimorph_db_path = find_ankimorph_db()
    if ankimorph_db_path is None:
        raise FileNotFoundError(
            f"AnkiMorphs database not found under {ANKI2_BASE}. "
            "Run AnkiMorphs recalc in Anki first."
        )

    ankimorph_db_path = Path(ankimorph_db_path)
    if not ankimorph_db_path.exists():
        raise FileNotFoundError(f"AnkiMorphs database not found: {ankimorph_db_path}")

    now = datetime.now().isoformat()
    rows: list[tuple[str, str, str, str]] = []

    try:
        conn = _sqlite3.connect(str(ankimorph_db_path))
        try:
            cursor = conn.execute(
                """
                SELECT lemma, inflection
                FROM Morphs
                WHERE highest_lemma_learning_interval >= ?
                ORDER BY lemma
                """,
                (min_interval,),
            )
            for lemma, inflection in cursor.fetchall():
                lemma     = (lemma     or "").strip()
                inflection = (inflection or "").strip()
                if lemma:
                    rows.append((lemma, inflection, "ankimorph", now))
        finally:
            conn.close()
    except Exception as exc:
        raise RuntimeError(
            f"import_ankimorph_db: failed to read {ankimorph_db_path}: {exc}"
        ) from exc

    if not rows:
        logger.warning(
            "import_ankimorph_db: no morphs found in %s (min_interval=%d). "
            "Run AnkiMorphs recalc in Anki to populate the database.",
            ankimorph_db_path, min_interval,
        )
        return 0

    with db.connect() as conn:
        conn.executemany(
            """
            INSERT INTO known_vocab (word, reading, source, added_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(word) DO UPDATE SET
                reading  = excluded.reading,
                source   = excluded.source,
                added_at = excluded.added_at
            """,
            rows,
        )

    logger.info(
        "import_ankimorph_db: upserted %d morphs from %s",
        len(rows), ankimorph_db_path,
    )
    return len(rows)


def import_seed_vocab(
    path: str | Path,
    db: DatabaseManager,
    source: str = "jlpt_seed",
) -> int:
    """Import a plain-text or TSV seed vocabulary file into ``known_vocab``.

    Each non-empty, non-comment line is treated as one word (first whitespace-
    delimited token is used so TSV files with a reading column work too).

    Args:
        path:   Path to the seed file.
        db:     Open ``DatabaseManager`` instance.
        source: Source label stored in ``known_vocab.source``.

    Returns:
        Number of words upserted.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError:      If the file cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Seed vocab file not found: {path}")

    rows: list[tuple[str, str, str, str]] = []
    now = datetime.now().isoformat()

    try:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                word    = parts[0].strip()
                reading = parts[1].strip() if len(parts) > 1 else ""
                if word:
                    rows.append((word, reading, source, now))
    except Exception as exc:
        raise RuntimeError(f"import_seed_vocab: failed to read {path}: {exc}") from exc

    if not rows:
        logger.warning("import_seed_vocab: no rows found in %s", path)
        return 0

    with db.connect() as conn:
        conn.executemany(
            """
            INSERT INTO known_vocab (word, reading, source, added_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(word) DO UPDATE SET
                reading  = excluded.reading,
                source   = excluded.source,
                added_at = excluded.added_at
            """,
            rows,
        )

    logger.info("import_seed_vocab: upserted %d words from %s (source=%s)", len(rows), path, source)
    return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_SQLITE_MAX_PARAMS: int = 900  # safe SQLite variable limit (actual: 999)


def _count_known(conn, words: set[str]) -> int:
    """Count how many words in *words* exist in ``known_vocab``.

    Chunks the query to stay below SQLite's variable limit.

    Args:
        conn:  Active SQLite connection.
        words: Set of word strings to look up.

    Returns:
        Integer count of known words.
    """
    word_list = list(words)
    known = 0
    for i in range(0, len(word_list), _SQLITE_MAX_PARAMS):
        chunk = word_list[i : i + _SQLITE_MAX_PARAMS]
        placeholders = ",".join("?" * len(chunk))
        row = conn.execute(
            f"SELECT COUNT(*) FROM known_vocab WHERE word IN ({placeholders})",
            chunk,
        ).fetchone()
        known += row[0] if row else 0
    return known


def _fetch_known_set(conn, words: set[str]) -> set[str]:
    """Return the subset of *words* that exist in ``known_vocab``.

    Args:
        conn:  Active SQLite connection.
        words: Candidate word strings.

    Returns:
        Set of known word strings.
    """
    word_list = list(words)
    known: set[str] = set()
    for i in range(0, len(word_list), _SQLITE_MAX_PARAMS):
        chunk = word_list[i : i + _SQLITE_MAX_PARAMS]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT word FROM known_vocab WHERE word IN ({placeholders})",
            chunk,
        ).fetchall()
        for (w,) in rows:
            known.add(w)
    return known
