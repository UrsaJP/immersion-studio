"""Integration tests: settings.json and model_cache.json atomic write/read roundtrip.

Monkeypatches IS_SETTINGS_PATH and IS_MODEL_CACHE_PATH to isolated tmp dirs
so tests never touch ~/Library/Application Support/ImmersionStudio/.
"""

import json
import os
from pathlib import Path

import pytest

import core.config as config_mod
from core.config import load_is_settings, load_model_cache, save_is_settings, save_model_cache


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Redirect all settings I/O to a tmp directory for every test in this module."""
    fake_dir = tmp_path / "ImmersionStudio"
    fake_dir.mkdir()
    monkeypatch.setattr(config_mod, "IS_APP_SUPPORT_DIR", fake_dir)
    monkeypatch.setattr(config_mod, "IS_SETTINGS_PATH", fake_dir / "settings.json")
    monkeypatch.setattr(config_mod, "IS_MODEL_CACHE_PATH", fake_dir / "model_cache.json")


class TestSettingsRoundtrip:
    def test_save_and_load(self):
        data = {"subtitle_translation_provider": "anthropic", "theme": "dark"}
        save_is_settings(data)
        loaded = load_is_settings()
        assert loaded == data

    def test_empty_dict_roundtrip(self):
        save_is_settings({})
        loaded = load_is_settings()
        assert loaded == {}

    def test_load_returns_empty_when_no_file(self):
        # File does not exist — must return {} not raise
        result = load_is_settings()
        assert result == {}

    def test_atomic_write_no_tmp_left_behind(self, tmp_path):
        fake_dir = Path(config_mod.IS_SETTINGS_PATH).parent
        save_is_settings({"key": "value"})
        tmp_files = list(fake_dir.glob("*.tmp"))
        assert not tmp_files, f"Temp file(s) left behind: {tmp_files}"

    def test_unicode_roundtrip(self):
        data = {"title": "テスト", "description": "日本語テキスト"}
        save_is_settings(data)
        loaded = load_is_settings()
        assert loaded["title"] == "テスト"
        assert loaded["description"] == "日本語テキスト"

    def test_nested_dict_roundtrip(self):
        data = {
            "overrides": {
                "subtitle_translation": {"provider": "ollama", "model": "qwen2.5:7b"}
            }
        }
        save_is_settings(data)
        loaded = load_is_settings()
        assert loaded["overrides"]["subtitle_translation"]["model"] == "qwen2.5:7b"

    def test_overwrite_existing_file(self):
        save_is_settings({"version": 1})
        save_is_settings({"version": 2})
        loaded = load_is_settings()
        assert loaded["version"] == 2


class TestModelCacheRoundtrip:
    def test_save_and_load(self):
        cache = {
            "openai": ["gpt-4o-mini", "gpt-4o"],
            "anthropic": ["claude-haiku-4-5-20251001"],
        }
        save_model_cache(cache)
        loaded = load_model_cache()
        assert loaded == cache

    def test_load_returns_empty_when_no_file(self):
        result = load_model_cache()
        assert result == {}

    def test_atomic_write_no_tmp_left_behind(self):
        fake_dir = Path(config_mod.IS_MODEL_CACHE_PATH).parent
        save_model_cache({"ollama": []})
        tmp_files = list(fake_dir.glob("*.tmp"))
        assert not tmp_files

    def test_empty_provider_models_list(self):
        cache = {"groq": []}
        save_model_cache(cache)
        loaded = load_model_cache()
        assert loaded["groq"] == []

    def test_overwrite_preserves_all_providers(self):
        save_model_cache({"openai": ["gpt-4o"]})
        save_model_cache({"openai": ["gpt-4o", "gpt-4o-mini"], "gemini": ["gemini-2.0-flash"]})
        loaded = load_model_cache()
        assert "gemini" in loaded
        assert "gpt-4o-mini" in loaded["openai"]
