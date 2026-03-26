"""Unit tests for core/anki.py Phase 2 additions.

Tests: AnkiNote, AnkiConnectionError, AnkiLogicError, AnkiConnector.
"""

import pytest
from unittest.mock import MagicMock, patch

import requests

from core.anki import (
    AnkiConnectionError,
    AnkiError,
    AnkiLogicError,
    AnkiNote,
    AnkiConnector,
    ANKICONNECT_DEFAULT_URL,
)


# ══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ══════════════════════════════════════════════════════════════════════════════

class TestExceptions:
    def test_anki_connection_error_is_anki_error(self):
        exc = AnkiConnectionError("no connection")
        assert isinstance(exc, AnkiError)

    def test_anki_logic_error_is_anki_error(self):
        exc = AnkiLogicError("duplicate")
        assert isinstance(exc, AnkiError)
        assert exc.error == "duplicate"

    def test_anki_logic_error_str(self):
        exc = AnkiLogicError("bad deck")
        assert "bad deck" in str(exc)


# ══════════════════════════════════════════════════════════════════════════════
# AnkiNote
# ══════════════════════════════════════════════════════════════════════════════

class TestAnkiNote:
    def test_to_api_dict_structure(self):
        note = AnkiNote(
            deck="Mining",
            note_type="JP Sentence",
            fields={"Front": "食べる", "Back": "to eat"},
            tags=["mining"],
        )
        d = note.to_api_dict()
        assert d["deckName"]  == "Mining"
        assert d["modelName"] == "JP Sentence"
        assert d["fields"]    == {"Front": "食べる", "Back": "to eat"}
        assert d["tags"]      == ["mining"]

    def test_allow_duplicate_default_false(self):
        note = AnkiNote("D", "M", {})
        assert note.to_api_dict()["options"]["allowDuplicate"] is False

    def test_allow_duplicate_true(self):
        note = AnkiNote("D", "M", {}, allow_duplicate=True)
        assert note.to_api_dict()["options"]["allowDuplicate"] is True

    def test_default_tags_empty(self):
        note = AnkiNote("D", "M", {})
        assert note.tags == []


# ══════════════════════════════════════════════════════════════════════════════
# AnkiConnector._post — unit tests via mock
# ══════════════════════════════════════════════════════════════════════════════

def _make_response(result=None, error=None, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"result": result, "error": error}
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


class TestAnkiConnectorPost:
    def setup_method(self):
        self.ac = AnkiConnector(retries=1)

    @patch("core.anki.requests.post")
    def test_successful_post_returns_result(self, mock_post):
        mock_post.return_value = _make_response(result=42)
        result = self.ac._post("version")
        assert result == 42

    @patch("core.anki.requests.post")
    def test_logic_error_raises_anki_logic_error(self, mock_post):
        mock_post.return_value = _make_response(result=None, error="duplicate note")
        with pytest.raises(AnkiLogicError) as exc_info:
            self.ac._post("addNote")
        assert "duplicate" in str(exc_info.value)

    @patch("core.anki.requests.post", side_effect=ConnectionError("refused"))
    def test_connection_error_raises_anki_connection_error(self, _):
        with pytest.raises(AnkiConnectionError):
            self.ac._post("version")

    @patch("core.anki.requests.post")
    def test_retry_on_connection_error(self, mock_post):
        ac = AnkiConnector(retries=3)
        mock_post.side_effect = [
            ConnectionError("err"),
            ConnectionError("err"),
            _make_response(result="ok"),
        ]
        with patch("core.anki.time.sleep"):
            result = ac._post("version")
        assert result == "ok"
        assert mock_post.call_count == 3

    @patch("core.anki.requests.post")
    def test_is_alive_returns_true(self, mock_post):
        mock_post.return_value = _make_response(result=6)
        assert self.ac.is_alive() is True

    @patch("core.anki.requests.post", side_effect=ConnectionError())
    def test_is_alive_returns_false_on_error(self, _):
        assert self.ac.is_alive() is False


class TestAnkiConnectorMethods:
    @patch("core.anki.requests.post")
    def test_add_note(self, mock_post):
        mock_post.return_value = _make_response(result=1234567890)
        ac = AnkiConnector(retries=1)
        note = AnkiNote("Mining", "JP", {"Front": "x"}, tags=["test"])
        note_id = ac.add_note(note)
        assert note_id == 1234567890

    @patch("core.anki.requests.post")
    def test_get_field_names(self, mock_post):
        mock_post.return_value = _make_response(result=["Front", "Back"])
        ac = AnkiConnector(retries=1)
        fields = ac.get_field_names("JP Sentence")
        assert fields == ["Front", "Back"]

    @patch("core.anki.requests.post")
    def test_find_notes(self, mock_post):
        mock_post.return_value = _make_response(result=[1, 2, 3])
        ac = AnkiConnector(retries=1)
        ids = ac.find_notes("deck:Mining")
        assert ids == [1, 2, 3]

    @patch("core.anki.requests.post")
    def test_find_notes_empty(self, mock_post):
        mock_post.return_value = _make_response(result=[])
        ac = AnkiConnector(retries=1)
        assert ac.find_notes("deck:Empty") == []

    @patch("core.anki.requests.post")
    def test_notes_info(self, mock_post):
        data = [{"noteId": 1, "fields": {}}]
        mock_post.return_value = _make_response(result=data)
        ac = AnkiConnector(retries=1)
        result = ac.notes_info([1])
        assert result == data

    def test_default_url(self):
        ac = AnkiConnector()
        assert ac.url == ANKICONNECT_DEFAULT_URL
