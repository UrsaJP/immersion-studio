"""AIST AnkiConnect integration: add notes, store media, query fields.

Single responsibility: communicate with Anki via AnkiConnect HTTP API.
Used by core/pipeline.py for concept-def caching.

STATUS: extended
DIVERGES_FROM_AIST: True
Changes: added AnkiConnectionError, AnkiLogicError, AnkiNote dataclass,
         AnkiConnector class with exponential-backoff retry (Phase 2 Step 3).
         All legacy module-level functions preserved unchanged.
"""

from __future__ import annotations

# ── STATUS ────────────────────────────────────────────────────────────────────
# STATUS: extended
# DIVERGES_FROM_AIST: True
# Changes: see module docstring.
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
ANKICONNECT_DEFAULT_URL:    str = "http://127.0.0.1:8765"
ANKICONNECT_VERSION:        int = 6
ANKICONNECT_TIMEOUT_SEC:    int = 10
ANKICONNECT_RETRY_COUNT:    int = 3          # total attempts
ANKICONNECT_RETRY_BACKOFF:  float = 1.5      # seconds, doubled each retry


# ══════════════════════════════════════════════════════════════════════════════
# TYPED EXCEPTIONS  (Phase 2 additions)
# ══════════════════════════════════════════════════════════════════════════════

class AnkiError(Exception):
    """Base exception for all AnkiConnect errors."""


class AnkiConnectionError(AnkiError):
    """Raised when the AnkiConnect HTTP server cannot be reached.

    This typically means Anki is not running or the port is wrong.
    """


class AnkiLogicError(AnkiError):
    """Raised when AnkiConnect returns a non-null error field in its response.

    The ``error`` attribute contains the raw AnkiConnect error string.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.error: str = message


# ══════════════════════════════════════════════════════════════════════════════
# TYPED DATA MODEL  (Phase 2 additions)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnkiNote:
    """Represents a single Anki note to be added via AnkiConnect.

    Args:
        deck:       Target deck name (e.g. ``"Japanese::Mining"``).
        note_type:  Model/note-type name (e.g. ``"Japanese sentence"``).
        fields:     Mapping of field name → field value.
        tags:       Optional list of tags to attach to the note.
        allow_duplicate: Whether to allow adding a note whose fields already exist.
    """
    deck:            str
    note_type:       str
    fields:          dict[str, str]
    tags:            list[str]           = field(default_factory=list)
    allow_duplicate: bool                = False

    def to_api_dict(self) -> dict:
        """Serialise to the dict shape AnkiConnect expects in ``addNote``."""
        return {
            "deckName":  self.deck,
            "modelName": self.note_type,
            "fields":    self.fields,
            "options":   {"allowDuplicate": self.allow_duplicate},
            "tags":      self.tags,
        }


# ══════════════════════════════════════════════════════════════════════════════
# AnkiConnector  (Phase 2 additions)
# ══════════════════════════════════════════════════════════════════════════════

class AnkiConnector:
    """High-level, retry-capable interface to the AnkiConnect HTTP API.

    Wraps the raw ``anki_request`` helper with:
    * Typed ``AnkiConnectionError`` / ``AnkiLogicError`` exceptions so callers
      can distinguish "Anki not running" from "duplicate note" etc.
    * Exponential-backoff retry for transient connection failures.
    * Convenience methods for the operations IS uses: add_note, store_media,
      get_field_names, is_alive.

    Args:
        url:     AnkiConnect base URL.  Defaults to ``http://127.0.0.1:8765``.
        retries: Maximum number of request attempts before giving up.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        url:     str = ANKICONNECT_DEFAULT_URL,
        retries: int = ANKICONNECT_RETRY_COUNT,
        timeout: int = ANKICONNECT_TIMEOUT_SEC,
    ) -> None:
        self.url     = url
        self.retries = retries
        self.timeout = timeout

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _post(self, action: str, **params) -> object:
        """Send one AnkiConnect action with exponential-backoff retry.

        Args:
            action: AnkiConnect action name (e.g. ``"addNote"``).
            **params: Action parameters forwarded verbatim.

        Returns:
            The ``result`` field from the AnkiConnect JSON response.

        Raises:
            AnkiConnectionError: Network failure or Anki not running.
            AnkiLogicError:      AnkiConnect returned a non-null ``error`` field.
        """
        payload = {"action": action, "version": ANKICONNECT_VERSION, "params": params}
        delay   = ANKICONNECT_RETRY_BACKOFF
        last_exc: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(self.url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if data.get("error"):
                    raise AnkiLogicError(str(data["error"]))
                return data.get("result")
            except AnkiLogicError:
                raise           # logic errors are not retried
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "AnkiConnector._post: attempt %d/%d failed for action=%s: %s",
                    attempt, self.retries, action, exc,
                )
                if attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2

        raise AnkiConnectionError(
            f"AnkiConnect unreachable at {self.url} after {self.retries} attempts: {last_exc}"
        ) from last_exc

    # ── Public API ────────────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        """Return True if AnkiConnect responds, False otherwise.

        Never raises.
        """
        try:
            self._post("version")
            return True
        except Exception:
            return False

    def add_note(self, note: AnkiNote) -> int:
        """Add *note* to Anki and return the new note ID.

        Args:
            note: An ``AnkiNote`` instance.

        Returns:
            The integer ID of the newly created note.

        Raises:
            AnkiConnectionError: Anki not running.
            AnkiLogicError:      Duplicate note or invalid deck/model.
        """
        note_id = self._post("addNote", note=note.to_api_dict())
        logger.info("AnkiConnector.add_note: created note_id=%s in deck=%s", note_id, note.deck)
        return int(note_id)

    def store_media(self, filename: str, data_b64: str) -> None:
        """Store a base-64 encoded media file in Anki's media folder.

        Args:
            filename: Target filename inside Anki's ``collection.media`` folder.
            data_b64: Base-64 encoded file contents.

        Raises:
            AnkiConnectionError: Anki not running.
            AnkiLogicError:      AnkiConnect rejected the request.
        """
        self._post("storeMediaFile", filename=filename, data=data_b64)
        logger.debug("AnkiConnector.store_media: stored %s", filename)

    def get_field_names(self, note_type: str) -> list[str]:
        """Return field names for *note_type*.

        Args:
            note_type: Model/note-type name (e.g. ``"Japanese sentence"``).

        Returns:
            Ordered list of field names.  Empty list if not found.

        Raises:
            AnkiConnectionError: Anki not running.
            AnkiLogicError:      Unknown note type.
        """
        result = self._post("modelFieldNames", modelName=note_type)
        return list(result or [])

    def find_notes(self, query: str) -> list[int]:
        """Run an Anki search query and return matching note IDs.

        Args:
            query: Anki search query string (same syntax as the browser).

        Returns:
            List of integer note IDs.

        Raises:
            AnkiConnectionError: Anki not running.
            AnkiLogicError:      Malformed query.
        """
        result = self._post("findNotes", query=query)
        return [int(nid) for nid in (result or [])]

    def notes_info(self, note_ids: list[int]) -> list[dict]:
        """Fetch full note data for *note_ids*.

        Args:
            note_ids: List of note IDs (as returned by ``find_notes``).

        Returns:
            List of note-info dicts as returned by AnkiConnect.

        Raises:
            AnkiConnectionError: Anki not running.
            AnkiLogicError:      Invalid note IDs.
        """
        result = self._post("notesInfo", notes=note_ids)
        return list(result or [])

# ══════════════════════════════════════════════════════════════════════════════
# ANKICONNECT
# ══════════════════════════════════════════════════════════════════════════════

def anki_request(url: str, action: str, **params):
    r = requests.post(url, json={"action": action, "version": 6, "params": params}, timeout=10)
    r.raise_for_status()
    result = r.json()
    if result.get("error"):
        raise RuntimeError(f"AnkiConnect: {result['error']}")
    return result.get("result")


def anki_add_note(url, deck, note_type, fields: dict, tags: list = None) -> bool:
    try:
        anki_request(url, "addNote", note={
            "deckName": deck, "modelName": note_type,
            "fields": fields,
            "options": {"allowDuplicate": False},
            "tags": tags or ["aist"],
        })
        return True
    except Exception:
        return False


def anki_store_media(url: str, filename: str, data_b64: str):
    anki_request(url, "storeMediaFile", filename=filename, data=data_b64)


def anki_query_morphs(url: str, morph_deck: str, morph_field: str) -> set:
    """Query AnkiMorphs cards to build a live known-words set."""
    try:
        note_ids = anki_request(url, "findNotes",
                                 query=f'deck:"{morph_deck}" tag:AnkiMorphs*')
        if not note_ids:
            return set()
        notes = anki_request(url, "notesInfo", notes=note_ids[:2000])
        words = set()
        for note in (notes or []):
            flds = note.get("fields", {})
            if morph_field in flds:
                words.add(flds[morph_field]["value"].strip())
        return words
    except Exception:
        return set()


def anki_get_field_names(url: str, note_type: str) -> list:
    try:
        return anki_request(url, "modelFieldNames", modelName=note_type) or []
    except Exception:
        return []
