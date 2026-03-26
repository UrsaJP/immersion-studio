"""Immersion Studio batch MKV subtitle translation pipeline.

Single responsibility: extract subtitle tracks from MKV files, AI-translate
each entry, mux the result into a new MKV copy. The original file is never
modified. Uses ffprobe/ffmpeg for media I/O and a configurable AI provider
for translation.

Imports from: core/config.py (PROVIDERS, load_api_key), core/subtitle.py
              (parse_srt/write_srt).
Used by: ui/widgets/translation_widget.py.

STATUS: extracted
DIVERGES_FROM_AIST: True
Changes: extracted tokenization pass only, stripped UI and player,
         Whisper integration is optional path
"""

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: extracted
# DIVERGES_FROM_AIST: True
# Changes: written from scratch per Feature 4 spec; call_ai_provider extracted
#          from aist_app._call_ai (lines ~5319–5504), stripped self/stream/action
#          override; all UI updates go through Qt signals.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import weakref
from pathlib import Path
from typing import Callable

import requests

from PySide6.QtCore import QObject, QRunnable, Qt, Signal

from .config import get_provider
from .subtitle import parse_srt, write_srt

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TRANSLATION_PROMPT: str = (
    "Translate the following Japanese subtitle line to English. "
    "Output ONLY the translated text — no explanations, no romanization.\n\n{text}"
)
TRANSLATION_OUTPUT_SUFFIX: str = "_translated"
FFPROBE_TIMEOUT_SEC: int = 30
FFMPEG_TIMEOUT_SEC: int = 0  # 0 = no timeout for large file mux operations
OLLAMA_CONNECT_TIMEOUT_SEC: int = 10
OLLAMA_READ_TIMEOUT_SEC: int = 600
OPENAI_COMPAT_TIMEOUT_SEC: int = 120
ANTHROPIC_TIMEOUT_SEC: int = 120
GEMINI_TIMEOUT_SEC: int = 120
OPENROUTER_HTTP_REFERER: str = "https://github.com/UrsaJP/aist"
OPENROUTER_X_TITLE: str = "ImmersionStudio"


# ══════════════════════════════════════════════════════════════════════════════
# FFPROBE — SUBTITLE TRACK DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def list_subtitle_tracks(mkv_path: str) -> list[dict]:
    """Return subtitle stream info from an MKV.

    Args:
        mkv_path: Path to the MKV file.

    Returns:
        List of dicts, each with keys:
            'index' (int): stream index within the file.
            'codec' (str): codec name (e.g. 'subrip', 'ass').
            'language' (str): BCP-47 language tag or 'und'.
            'title' (str): stream title metadata, empty string if absent.

    Raises:
        RuntimeError: If ffprobe exits non-zero or is not in PATH.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "s", mkv_path],
            capture_output=True, text=True, check=True,
            timeout=FFPROBE_TIMEOUT_SEC,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe not found in PATH. Install with: brew install ffmpeg"
        )
    except subprocess.CalledProcessError as e:
        logger.error("ffprobe failed for %s: %s", mkv_path, e.stderr, exc_info=True)
        raise RuntimeError(f"ffprobe failed:\n{e.stderr[-2000:]}")
    streams = json.loads(result.stdout).get("streams", [])
    return [
        {
            "index":    s.get("index"),
            "codec":    s.get("codec_name", ""),
            "language": s.get("tags", {}).get("language", "und"),
            "title":    s.get("tags", {}).get("title", ""),
        }
        for s in streams
    ]


# ══════════════════════════════════════════════════════════════════════════════
# AI PROVIDER ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def call_ai_provider(
    provider_id: str,
    model: str,
    prompt: str,
    api_key: str | None = None,
) -> str:
    """Route a prompt to the configured AI provider and return the response text.

    Extracted from aist_app.py _call_ai (lines ~5319–5504). Streaming path
    removed — translation worker calls this synchronously per entry.

    Args:
        provider_id: Provider ID from PROVIDERS dict (e.g. 'ollama', 'anthropic').
        model: Model name string (e.g. 'qwen2.5:7b', 'claude-haiku-4-5-20251001').
        prompt: The full prompt string to send.
        api_key: API key. If None, the caller should have loaded it from Keychain
                 before calling this function.

    Returns:
        The response text from the AI provider.

    Raises:
        RuntimeError: If the provider returns a non-200 status or malformed response.
    """
    prov = get_provider(provider_id)
    base_url = prov["base_url"]

    # ── Ollama (local) ────────────────────────────────────────────────────────
    if base_url is None:
        import urllib3
        session = requests.Session()
        session.mount("http://", requests.adapters.HTTPAdapter(
            max_retries=urllib3.util.Retry(
                total=2, backoff_factor=2,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["POST"],
            )
        ))
        resp = session.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "keep_alive": -1,
                  "options": {"temperature": 0.3}},
            timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, OLLAMA_READ_TIMEOUT_SEC),
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    # ── Anthropic (non-OpenAI format) ─────────────────────────────────────────
    if base_url == "anthropic":
        if not api_key:
            raise RuntimeError(
                "No API key set for Anthropic — add it in Settings → AI Providers"
            )
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 3000,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=ANTHROPIC_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Anthropic response malformed: %s, keys=%s", e, list(data.keys()), exc_info=True)
            raise RuntimeError(
                f"Anthropic response missing expected content: {e}. "
                f"Keys: {list(data.keys())}"
            )

    # ── Gemini (non-OpenAI format) ────────────────────────────────────────────
    if base_url == "gemini":
        if not api_key:
            raise RuntimeError(
                "No API key set for Gemini — add it in Settings → AI Providers"
            )
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            f":generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 3000, "temperature": 0.3},
            },
            timeout=GEMINI_TIMEOUT_SEC,
        )
        if resp.status_code == 429:
            raise RuntimeError("429 Too Many Requests — Gemini free tier rate limit hit")
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Gemini response malformed: %s, keys=%s", e, list(data.keys()), exc_info=True)
            raise RuntimeError(
                f"Gemini response missing expected content: {e}. "
                f"Keys: {list(data.keys())}"
            )

    # ── OpenAI-compatible (all other providers) ───────────────────────────────
    if not api_key:
        raise RuntimeError(
            f"No API key set for {prov['name']} — add it in Settings → AI Providers"
        )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # OpenRouter requires these headers — without them it returns 401 even with a valid key.
    if "openrouter.ai" in base_url:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
        headers["X-Title"] = OPENROUTER_X_TITLE
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 3000,
            "temperature": 0.3,
        },
        timeout=OPENAI_COMPAT_TIMEOUT_SEC,
    )
    if resp.status_code == 429:
        raise RuntimeError(f"429 Too Many Requests — {prov['name']} rate limit hit")
    if resp.status_code == 401:
        raise RuntimeError(
            f"401 Unauthorized — {prov['name']} rejected the API key. "
            f"Go to Settings → AI Providers and update your key for {prov['name']}."
        )
    if resp.status_code == 403:
        raise RuntimeError(
            f"403 Forbidden — {prov['name']} blocked the request. "
            f"Check that your API key has access to model '{model}'."
        )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        logger.error("%s response malformed: %s, keys=%s", prov["name"], e, list(data.keys()), exc_info=True)
        raise RuntimeError(
            f"{prov['name']} response missing expected content: {e}. "
            f"Keys: {list(data.keys())}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SRT TRANSLATION
# ══════════════════════════════════════════════════════════════════════════════

def translate_srt(
    srt_path: Path,
    provider_id: str,
    model: str,
    progress_cb: Callable[[int, int], None] | None = None,
    api_key: str | None = None,
) -> Path:
    """Translate every subtitle entry in an SRT. Returns path to translated SRT.

    Args:
        srt_path: Path to the source SRT file.
        provider_id: AI provider ID (e.g. 'ollama', 'anthropic').
        model: Model name string.
        progress_cb: Optional callback(current, total) called after each entry.
                     Caller is responsible for threading — this function is
                     synchronous and must be called from a worker thread.
        api_key: API key for the provider (None for Ollama/LM Studio).

    Returns:
        Path to the translated SRT file (same directory, .en.srt suffix).

    Raises:
        RuntimeError: If AI provider call fails or SRT cannot be parsed.
    """
    entries = parse_srt(str(srt_path))
    total = len(entries)
    translated_entries = []
    for i, (idx, timing, text) in enumerate(entries):
        stripped = text.strip()
        if not stripped:
            translated_entries.append((idx, timing, text))
        else:
            result = call_ai_provider(
                provider_id, model,
                TRANSLATION_PROMPT.format(text=stripped),
                api_key=api_key,
            )
            translated_entries.append((idx, timing, result.strip()))
        if progress_cb:
            progress_cb(i + 1, total)
    out_path = srt_path.with_suffix(".en.srt")
    write_srt(translated_entries, str(out_path))
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# MKV MUX
# ══════════════════════════════════════════════════════════════════════════════

def mux_subtitle_into_mkv(mkv_path: str, srt_path: str, output_path: str) -> None:
    """Mux a translated SRT into a new MKV copy. Original file is never modified.

    Args:
        mkv_path: Path to the original MKV.
        srt_path: Path to the translated SRT file.
        output_path: Path for the new MKV output file.

    Returns:
        None

    Raises:
        RuntimeError: If ffmpeg exits non-zero or is not in PATH.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", mkv_path,
        "-i", srt_path,
        "-c", "copy",
        "-map", "0",
        "-map", "1",
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=FFMPEG_TIMEOUT_SEC if FFMPEG_TIMEOUT_SEC > 0 else None,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install with: brew install ffmpeg"
        )
    if result.returncode != 0:
        logger.error("ffmpeg mux failed (exit %d): %s", result.returncode, result.stderr[-2000:])
        raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr[-2000:]}")


# ══════════════════════════════════════════════════════════════════════════════
# Qt WORKER — THREADING
# ══════════════════════════════════════════════════════════════════════════════

class TranslationSignals(QObject):
    """Signals emitted by TranslationWorker.

    Emits:
        progress(current_line: int, total_lines: int): After each subtitle entry.
        finished(output_path: str): When mux completes successfully.
        error(message: str): On any exception in the worker.
    """
    progress = Signal(int, int)   # (current_line, total_lines)
    finished = Signal(str)        # output_path
    error    = Signal(str)


class TranslationWorker(QRunnable):
    """Batch MKV subtitle translation worker.

    Runs on QThreadPool. Extracts subtitle track, translates line-by-line
    via AI provider, and muxes the result into a new MKV file.

    # ⚠️ WORKER THREAD — runs on QThreadPool. Never touch Qt widgets directly.
    # Emit self.signals.finished(result) and let Qt dispatch to main thread.

    Signals consumed by TranslationWidget via weakref to prevent accessing
    a destroyed widget.

    Args:
        mkv_path: Path to the source MKV file.
        track_index: Subtitle stream index within the MKV (from list_subtitle_tracks).
        provider_id: AI provider ID.
        model: Model name string.
        api_key: API key (None for local providers).
    """

    def __init__(
        self,
        mkv_path: str,
        track_index: int,
        provider_id: str,
        model: str,
        api_key: str | None = None,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.signals      = TranslationSignals()
        self.mkv_path     = mkv_path
        self.track_index  = track_index
        self.provider_id  = provider_id
        self.model        = model
        self.api_key      = api_key

    # ⚠️ WORKER THREAD — runs on QThreadPool, never touch Qt widgets directly,
    # emit self.signals.finished(result) and let Qt dispatch to main thread.
    def run(self) -> None:
        """Execute full pipeline: extract SRT → translate → mux MKV."""
        try:
            with tempfile.TemporaryDirectory() as tmp:
                srt_tmp = os.path.join(tmp, "extracted.srt")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", self.mkv_path,
                     "-map", f"0:s:{self.track_index}", srt_tmp],
                    check=True, capture_output=True,
                )
                out_srt = translate_srt(
                    Path(srt_tmp),
                    self.provider_id,
                    self.model,
                    progress_cb=lambda c, t: self.signals.progress.emit(c, t),
                    api_key=self.api_key,
                )
                stem = Path(self.mkv_path).stem
                output_path = str(
                    Path(self.mkv_path).parent / f"{stem}{TRANSLATION_OUTPUT_SUFFIX}.mkv"
                )
                mux_subtitle_into_mkv(self.mkv_path, str(out_srt), output_path)
            self.signals.finished.emit(output_path)
        except Exception as e:
            logger.error("TranslationWorker failed: %s", e, exc_info=True)
            self.signals.error.emit(str(e))
