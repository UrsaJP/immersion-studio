"""Unit tests for core.config: get_provider, AI_PROVIDERS registry."""

import pytest

from core.config import AI_PROVIDERS, PROVIDERS, get_provider

# All 18 expected provider IDs from the plan.
# AI_PROVIDERS (AIST-origin list) has 14; PROVIDERS dict (IS-specific) has all 18.
EXPECTED_PROVIDER_IDS = {
    "ollama", "openai", "anthropic", "gemini", "groq", "deepseek",
    "mistral", "kimi", "together", "openrouter", "perplexity", "cohere",
    "xai", "cerebras", "fireworks", "lmstudio", "lepton", "anyscale",
}


class TestAiProvidersRegistry:
    def test_all_18_providers_present(self):
        # PROVIDERS dict is the authoritative 18-entry registry
        ids = set(PROVIDERS.keys())
        missing = EXPECTED_PROVIDER_IDS - ids
        assert not missing, f"Missing providers: {missing}"

    def test_each_provider_has_id_and_name(self):
        for entry in AI_PROVIDERS:
            pid, pname, *_ = entry
            assert isinstance(pid, str) and pid, "Provider ID must be non-empty string"
            assert isinstance(pname, str) and pname, "Provider name must be non-empty string"

    def test_ollama_base_url_is_none(self):
        """Ollama uses a custom handler signalled by base_url=None."""
        for pid, _pname, base_url, *_ in AI_PROVIDERS:
            if pid == "ollama":
                assert base_url is None
                return
        pytest.fail("ollama not found in AI_PROVIDERS")

    def test_anthropic_base_url_is_sentinel(self):
        for pid, _pname, base_url, *_ in AI_PROVIDERS:
            if pid == "anthropic":
                assert base_url == "anthropic"
                return
        pytest.fail("anthropic not found in AI_PROVIDERS")

    def test_gemini_base_url_is_sentinel(self):
        for pid, _pname, base_url, *_ in AI_PROVIDERS:
            if pid == "gemini":
                assert base_url == "gemini"
                return
        pytest.fail("gemini not found in AI_PROVIDERS")

    def test_openai_compat_providers_have_https_base_url(self):
        skip = {"ollama", "anthropic", "gemini"}
        for pid, _pname, base_url, *_ in AI_PROVIDERS:
            if pid in skip:
                continue
            assert base_url and base_url.startswith("http"), (
                f"{pid} should have an http(s) base_url, got: {base_url!r}"
            )

    def test_no_duplicate_ids(self):
        ids = [p[0] for p in AI_PROVIDERS]
        assert len(ids) == len(set(ids)), "Duplicate provider IDs detected"


class TestGetProvider:
    def test_known_provider_returns_dict(self):
        prov = get_provider("openai")
        assert prov["id"] == "openai"
        assert "name" in prov
        assert "base_url" in prov
        assert "default_model" in prov
        assert "models" in prov

    def test_unknown_provider_falls_back_to_ollama(self):
        prov = get_provider("nonexistent_provider_xyz")
        assert prov["id"] == "ollama"

    def test_ollama_default_model_set(self):
        prov = get_provider("ollama")
        assert prov["default_model"], "Ollama must have a default model"

    def test_models_is_list(self):
        prov = get_provider("anthropic")
        assert isinstance(prov["models"], list)
        assert len(prov["models"]) >= 1
