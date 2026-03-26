"""Immersion Studio pipeline: NWD tokenization pass, optional Whisper transcription.

Single responsibility: precompute_subtitle_tokens() tokenizes an SRT file and
stores per-entry token JSON in the media_subtitles table for NWD scoring queries.
Used by: core/nwd.py (Phase 2). Reads from: core/subtitle.py, core/japanese.py,
core/db.py.

STATUS: extended
DIVERGES_FROM_AIST: True
Changes: implemented precompute_subtitle_tokens() and transcribe_with_whisper()
         per Phase 2 Step 6 spec. Uses SRTFormat().parse() → FugashiTokenizer
         (with RegexTokenizer fallback) → media_subtitles UPSERT.
"""

from __future__ import annotations

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: extended
# DIVERGES_FROM_AIST: True
# Changes: implemented Phase 2 Step 6 spec.
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import DatabaseManager

from .subtitle import SRTFormat
from .japanese import JapaneseNLP

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
WHISPER_DEFAULT_MODEL:  str = "base"
PIPELINE_VERSION:       str = "2.0.0"
WHISPER_LANGUAGE:       str = "ja"
WHISPER_OUTPUT_FORMAT:  str = "srt"


# ══════════════════════════════════════════════════════════════════════════════
# NWD TOKENIZATION PASS
# ══════════════════════════════════════════════════════════════════════════════

# Module-level NLP instance — shared across calls, engine auto-detected once.
_nlp: JapaneseNLP | None = None


def _get_nlp() -> JapaneseNLP:
    global _nlp
    if _nlp is None:
        _nlp = JapaneseNLP()
        logger.info("pipeline: morphological engine = %s", _nlp.engine_name)
    return _nlp


def precompute_subtitle_tokens(
    srt_path: Path,
    media_path: str,
    db: DatabaseManager,
    *,
    overwrite: bool = False,
) -> int:
    """Tokenize all subtitle entries in an SRT and persist to ``media_subtitles``.

    For each subtitle entry the full token dict list (surface, lemma, pos,
    pos2, reading) is stored as JSON in ``media_subtitles.tokens_json``.
    Only the lemma strings are used by the NWD scoring query, but the full
    token data is stored for potential future use.

    Args:
        srt_path:   Path to the SRT file to tokenize.
        media_path: Canonical media file path — used as FK in ``media_subtitles``
                    and as PK in ``media_info``.
        db:         ``DatabaseManager`` instance (from ``core/db.py``).
        overwrite:  If ``True``, existing rows for *media_path* are deleted
                    before insertion.  If ``False`` and rows already exist,
                    returns the existing row count without re-inserting.

    Returns:
        Number of subtitle entries tokenized and inserted.

    Raises:
        FileNotFoundError: If *srt_path* does not exist.
        RuntimeError:      If SRT parsing fails.
    """
    srt_path = Path(srt_path)
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    # Short-circuit if already tokenised and overwrite not requested
    if not overwrite:
        with db.connect() as conn:
            (existing,) = conn.execute(
                "SELECT COUNT(*) FROM media_subtitles WHERE media_path = ?",
                (media_path,),
            ).fetchone()
        if existing:
            logger.debug(
                "precompute_subtitle_tokens: %s already has %d rows, skipping",
                media_path, existing,
            )
            return existing

    entries = SRTFormat().parse(str(srt_path))
    if not entries:
        logger.warning("precompute_subtitle_tokens: no entries parsed from %s", srt_path)
        return 0

    nlp = _get_nlp()
    rows: list[tuple] = []
    for entry in entries:
        tokens = nlp.tokenize(entry.text)
        tokens_json = json.dumps(
            [t["lemma"] for t in tokens],   # store lemmas only — compact
            ensure_ascii=False,
        )
        rows.append((
            media_path,
            entry.start.ms,
            entry.end.ms,
            entry.text,
            tokens_json,
        ))

    with db.connect() as conn:
        # Ensure a media_info row exists to satisfy the FK constraint
        conn.execute(
            """
            INSERT INTO media_info (path) VALUES (?)
            ON CONFLICT(path) DO NOTHING
            """,
            (media_path,),
        )
        if overwrite:
            conn.execute(
                "DELETE FROM media_subtitles WHERE media_path = ?",
                (media_path,),
            )
        conn.executemany(
            """
            INSERT INTO media_subtitles (media_path, start_ms, end_ms, text, tokens_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

    logger.info(
        "precompute_subtitle_tokens: inserted %d entries for %s (engine=%s)",
        len(rows), media_path, nlp.engine_name,
    )
    return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
# WHISPER TRANSCRIPTION  (optional path — only needed when no SRT exists)
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_with_whisper(
    media_path: str,
    output_dir: Path,
    model: str = WHISPER_DEFAULT_MODEL,
) -> Path:
    """Run OpenAI Whisper on a media file and return the path to the output SRT.

    Whisper is invoked as a subprocess (``whisper`` CLI).  The caller is
    responsible for checking whether the SRT already exists before calling this
    function.

    Args:
        media_path: Path to the media file (``.mkv``, ``.mp4``, ``.mp3``, etc.).
        output_dir: Directory where Whisper writes its output files.
        model:      Whisper model name (``tiny`` / ``base`` / ``small`` / ``medium`` / ``large``).

    Returns:
        Path to the generated ``.srt`` file.

    Raises:
        RuntimeError: If the ``whisper`` CLI is not found in PATH or exits non-zero.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not shutil.which("whisper"):
        raise RuntimeError(
            "whisper CLI not found in PATH. "
            "Install with: pip install openai-whisper"
        )

    cmd = [
        "whisper", str(media_path),
        "--model", model,
        "--output_format", WHISPER_OUTPUT_FORMAT,
        "--output_dir", str(output_dir),
        "--language", WHISPER_LANGUAGE,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as exc:
        raise RuntimeError(f"whisper subprocess error: {exc}") from exc

    if result.returncode != 0:
        logger.error("whisper failed (exit %d): %s", result.returncode, result.stderr[-2000:])
        raise RuntimeError(f"whisper failed:\n{result.stderr[-2000:]}")

    srt_out = output_dir / (Path(media_path).stem + ".srt")
    if not srt_out.exists():
        raise RuntimeError(
            f"whisper ran successfully but expected output not found: {srt_out}"
        )

    logger.info("transcribe_with_whisper: wrote %s", srt_out)
    return srt_out
