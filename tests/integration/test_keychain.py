"""Integration tests: Keychain helpers (save/load/delete_api_key).

core.config uses `import keyring` inside each helper function.  Because
`keyring` may not be installed in the test environment, we inject a
MagicMock into sys.modules before each test so the inline import resolves
to our mock.

Tests verify the KEYCHAIN_SERVICE namespace and error-handling contract.
"""

import sys
import types
from unittest.mock import MagicMock, call

import pytest

from core.config import (
    KEYCHAIN_SERVICE,
    delete_api_key,
    load_api_key,
    save_api_key,
)


@pytest.fixture(autouse=True)
def _mock_keyring():
    """Inject a fresh MagicMock as the `keyring` module for every test.

    Restores the original sys.modules entry (or removes it) on teardown
    so tests are fully isolated.
    """
    original = sys.modules.get("keyring", None)
    mock_mod = MagicMock()
    sys.modules["keyring"] = mock_mod
    yield mock_mod
    if original is None:
        sys.modules.pop("keyring", None)
    else:
        sys.modules["keyring"] = original


class TestKeychainService:
    def test_service_name_is_immersion_studio(self):
        assert KEYCHAIN_SERVICE == "ImmersionStudio"


class TestSaveApiKey:
    def test_calls_set_password(self, _mock_keyring):
        save_api_key("openai", "sk-test-key")
        _mock_keyring.set_password.assert_called_once_with(
            KEYCHAIN_SERVICE, "openai", "sk-test-key"
        )

    def test_uses_provider_id_as_username(self, _mock_keyring):
        save_api_key("anthropic", "ant-key")
        _svc, username, _pw = _mock_keyring.set_password.call_args.args
        assert username == "anthropic"


class TestLoadApiKey:
    def test_returns_stored_key(self, _mock_keyring):
        _mock_keyring.get_password.return_value = "sk-stored-key"
        result = load_api_key("openai")
        assert result == "sk-stored-key"

    def test_returns_none_when_not_set(self, _mock_keyring):
        _mock_keyring.get_password.return_value = None
        result = load_api_key("openai")
        assert result is None

    def test_returns_none_on_keyring_error(self, _mock_keyring):
        """load_api_key must swallow Keychain errors and return None."""
        _mock_keyring.get_password.side_effect = Exception("Keychain locked")
        result = load_api_key("openai")
        assert result is None

    def test_calls_get_password_with_correct_service_and_id(self, _mock_keyring):
        _mock_keyring.get_password.return_value = None
        load_api_key("gemini")
        _mock_keyring.get_password.assert_called_once_with(KEYCHAIN_SERVICE, "gemini")


class TestDeleteApiKey:
    def test_calls_delete_password(self, _mock_keyring):
        delete_api_key("openai")
        _mock_keyring.delete_password.assert_called_once_with(KEYCHAIN_SERVICE, "openai")

    def test_swallows_exception_on_missing_key(self, _mock_keyring):
        """Deleting a non-existent key must not raise."""
        _mock_keyring.delete_password.side_effect = Exception("Item not found")
        delete_api_key("nonexistent")  # must not raise


class TestKeychainRoundtrip:
    def test_save_then_load_returns_same_key(self, _mock_keyring):
        store = {}

        def fake_set(service, username, password):
            store[(service, username)] = password

        def fake_get(service, username):
            return store.get((service, username))

        _mock_keyring.set_password.side_effect = fake_set
        _mock_keyring.get_password.side_effect = fake_get

        save_api_key("groq", "gsk-mykey")
        result = load_api_key("groq")
        assert result == "gsk-mykey"

    def test_save_then_delete_then_load_returns_none(self, _mock_keyring):
        store = {}

        def fake_set(service, username, password):
            store[(service, username)] = password

        def fake_get(service, username):
            return store.get((service, username))

        def fake_delete(service, username):
            store.pop((service, username), None)

        _mock_keyring.set_password.side_effect = fake_set
        _mock_keyring.get_password.side_effect = fake_get
        _mock_keyring.delete_password.side_effect = fake_delete

        save_api_key("groq", "gsk-mykey")
        delete_api_key("groq")
        result = load_api_key("groq")
        assert result is None
