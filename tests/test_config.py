"""Tests for environment-based settings."""

import pytest

from app.config.settings import (
    DEFAULT_GEMINI_MODEL,
    ConfigurationError,
    load_settings,
)


def test_loads_defaults_without_an_api_key() -> None:
    settings = load_settings({}, load_dotenv_file=False)

    assert settings.model == DEFAULT_GEMINI_MODEL
    assert settings.temperature == 0.2
    assert settings.api_key is None


def test_loads_custom_settings() -> None:
    settings = load_settings(
        {
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "gemini-test-model",
            "TEMPERATURE": "1.25",
        },
        load_dotenv_file=False,
    )

    assert settings.api_key == "test-key"
    assert settings.model == "gemini-test-model"
    assert settings.temperature == 1.25


def test_missing_api_key_fails_only_when_required() -> None:
    settings = load_settings({}, load_dotenv_file=False)

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        settings.require_api_key()


@pytest.mark.parametrize("temperature", ["-0.01", "2.01", "not-a-number"])
def test_rejects_invalid_temperature(temperature: str) -> None:
    with pytest.raises(ConfigurationError):
        load_settings({"TEMPERATURE": temperature}, load_dotenv_file=False)


def test_settings_repr_does_not_expose_api_key() -> None:
    settings = load_settings(
        {"GEMINI_API_KEY": "secret-test-key"},
        load_dotenv_file=False,
    )

    assert "secret-test-key" not in repr(settings)
