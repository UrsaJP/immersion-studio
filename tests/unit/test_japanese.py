"""Unit tests for core/japanese.py Phase 2 additions.

Tests: MorphologicalEngine ABC, FugashiTokenizer, RegexTokenizer, JapaneseNLP.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.japanese import (
    MorphologicalEngine,
    FugashiTokenizer,
    RegexTokenizer,
    JapaneseNLP,
)


# ══════════════════════════════════════════════════════════════════════════════
# RegexTokenizer  (no external deps — always available)
# ══════════════════════════════════════════════════════════════════════════════

class TestRegexTokenizer:
    def setup_method(self):
        self.t = RegexTokenizer()

    def test_name(self):
        assert self.t.name == "regex"

    def test_tokenize_returns_list_of_dicts(self):
        tokens = self.t.tokenize("おはよう")
        assert isinstance(tokens, list)
        assert len(tokens) == 1
        tok = tokens[0]
        assert set(tok.keys()) >= {"surface", "lemma", "pos", "pos2", "reading"}

    def test_surface_equals_lemma(self):
        tokens = self.t.tokenize("食べる")
        assert tokens[0]["surface"] == tokens[0]["lemma"] == "食べる"

    def test_ascii_excluded(self):
        tokens = self.t.tokenize("hello world")
        assert tokens == []

    def test_mixed_text(self):
        tokens = self.t.tokenize("today is 今日 a good day")
        assert len(tokens) == 1
        assert tokens[0]["surface"] == "今日"

    def test_multiple_tokens(self):
        tokens = self.t.tokenize("今日はいい天気")
        assert len(tokens) >= 1

    def test_get_lemmas_filters_non_jp(self):
        lemmas = self.t.get_lemmas("hello 日本語 world")
        assert "日本語" in lemmas
        assert "hello" not in lemmas
        assert "world" not in lemmas

    def test_empty_string(self):
        assert self.t.tokenize("") == []

    def test_pos_is_noun_placeholder(self):
        tokens = self.t.tokenize("食べる")
        assert tokens[0]["pos"] == "名詞"

    def test_reading_is_empty(self):
        tokens = self.t.tokenize("食べる")
        assert tokens[0]["reading"] == ""

    def test_is_subclass_of_abstract(self):
        assert isinstance(self.t, MorphologicalEngine)


# ══════════════════════════════════════════════════════════════════════════════
# FugashiTokenizer  (mocked so tests pass without MeCab installed)
# ══════════════════════════════════════════════════════════════════════════════

class TestFugashiTokenizer:
    def test_name(self):
        ft = FugashiTokenizer()
        assert ft.name == "fugashi"

    def test_is_subclass_of_abstract(self):
        assert isinstance(FugashiTokenizer(), MorphologicalEngine)

    def test_tokenize_with_mock_tagger(self):
        ft = FugashiTokenizer()
        # Simulate a fugashi word object
        mock_word = MagicMock()
        mock_word.surface = "食べ"
        mock_word.feature = "動詞,自立,*,*,五段・カ行促音便,連用形,食べる,タベ,タベ"
        ft._tagger = MagicMock(return_value=[mock_word])

        tokens = ft.tokenize("食べる")
        assert len(tokens) == 1
        tok = tokens[0]
        assert tok["surface"] == "食べ"
        assert tok["pos"] == "動詞"

    def test_tagger_error_returns_empty(self):
        ft = FugashiTokenizer()
        ft._tagger = MagicMock(side_effect=RuntimeError("mecab error"))
        tokens = ft.tokenize("日本語")
        assert tokens == []

    def test_ensure_tagger_raises_import_error(self):
        ft = FugashiTokenizer()
        with patch.dict("sys.modules", {"fugashi": None}):
            with pytest.raises(Exception):
                ft._ensure_tagger()

    def test_get_lemmas_uses_lemma_field(self):
        ft = FugashiTokenizer()
        mock_word = MagicMock()
        mock_word.surface = "食べ"
        mock_word.feature = "動詞,自立,*,*,*,*,食べる,タベル,タベル"
        ft._tagger = MagicMock(return_value=[mock_word])
        lemmas = ft.get_lemmas("食べる")
        assert "食べる" in lemmas


# ══════════════════════════════════════════════════════════════════════════════
# JapaneseNLP
# ══════════════════════════════════════════════════════════════════════════════

class TestJapaneseNLP:
    def test_explicit_engine_is_used(self):
        engine = RegexTokenizer()
        nlp = JapaneseNLP(engine=engine)
        assert nlp.engine is engine

    def test_engine_name_from_regex(self):
        nlp = JapaneseNLP(engine=RegexTokenizer())
        assert nlp.engine_name == "regex"

    def test_tokenize_delegates_to_engine(self):
        engine = RegexTokenizer()
        nlp = JapaneseNLP(engine=engine)
        tokens = nlp.tokenize("日本語")
        assert isinstance(tokens, list)
        assert tokens[0]["surface"] == "日本語"

    def test_get_lemmas_delegates_to_engine(self):
        nlp = JapaneseNLP(engine=RegexTokenizer())
        lemmas = nlp.get_lemmas("日本語")
        assert "日本語" in lemmas

    def test_auto_detect_falls_back_to_regex(self):
        with patch("core.japanese.FugashiTokenizer._ensure_tagger", side_effect=Exception("no mecab")):
            nlp = JapaneseNLP()
        assert nlp.engine_name == "regex"

    def test_auto_detect_uses_fugashi_when_available(self):
        mock_ft = MagicMock(spec=FugashiTokenizer)
        mock_ft.name = "fugashi"
        with patch("core.japanese.FugashiTokenizer", return_value=mock_ft):
            mock_ft._ensure_tagger = MagicMock()
            nlp = JapaneseNLP._auto_detect()
        # If fugashi succeeded, engine is FugashiTokenizer-like
        assert nlp is mock_ft
