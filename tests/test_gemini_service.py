"""Tests for Gemini SDK integration using mocked responses."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from google.genai import types

from app.config.settings import ConfigurationError, Settings, load_settings
from app.services.gemini_service import GeminiService, GeminiServiceError


def make_settings(
    api_key: str | None = "test-api-key",
    *,
    temperature: str = "0.7",
) -> Settings:
    environment = {"GEMINI_MODEL": "gemini-test-model", "TEMPERATURE": temperature}
    if api_key is not None:
        environment["GEMINI_API_KEY"] = api_key
    return load_settings(environment, load_dotenv_file=False)


def make_response(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(text=text, prompt_feedback=None, candidates=[])


def test_initializes_official_client_with_key_and_timeout() -> None:
    settings = make_settings()

    with patch("app.services.gemini_service.genai.Client") as client_factory:
        service = GeminiService(settings=settings)

    client_factory.assert_called_once()
    assert client_factory.call_args.kwargs["api_key"] == "test-api-key"
    assert client_factory.call_args.kwargs["http_options"].timeout == 60_000
    assert service is not None


def test_missing_key_prevents_client_initialization() -> None:
    settings = make_settings(api_key=None)
    mocked_client = Mock()

    with patch("app.services.gemini_service.genai.Client") as client_factory:
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
            GeminiService(settings=settings, client=mocked_client)

    client_factory.assert_not_called()


def test_generate_answer_formats_history_and_generation_config() -> None:
    client = Mock()
    client.models.generate_content.return_value = make_response("  Hello there.  ")
    service = GeminiService(settings=make_settings(), client=client)
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Can you help?"},
    ]

    answer = service.generate_answer(history)

    assert answer == "Hello there."
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == "gemini-test-model"
    assert [content.role for content in call["contents"]] == [
        "user",
        "model",
        "user",
    ]
    assert [
        content.parts[0].text for content in call["contents"]
    ] == ["Hi", "Hello", "Can you help?"]
    assert isinstance(call["config"], types.GenerateContentConfig)
    assert call["config"].temperature == 0.7
    assert call["config"].system_instruction


def test_returns_safe_message_for_empty_response() -> None:
    client = Mock()
    client.models.generate_content.return_value = make_response(" ")
    service = GeminiService(settings=make_settings(), client=client)

    assert "did not return a text answer" in service.generate_answer(
        [{"role": "user", "content": "Question"}]
    )


def test_returns_safe_message_for_blocked_response() -> None:
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text=None,
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
        candidates=[],
    )
    service = GeminiService(settings=make_settings(), client=client)

    assert "couldn't generate an answer" in service.generate_answer(
        [{"role": "user", "content": "Question"}]
    )


def test_provider_failure_does_not_leak_sensitive_details(caplog: pytest.LogCaptureFixture) -> None:
    secret = "secret-test-key"
    client = Mock()
    client.models.generate_content.side_effect = RuntimeError(
        f"provider failure with credential {secret}"
    )
    service = GeminiService(settings=make_settings(api_key=secret), client=client)

    with pytest.raises(GeminiServiceError) as error:
        service.generate_answer([{"role": "user", "content": "Question"}])

    assert secret not in str(error.value)
    assert secret not in caplog.text
    assert "provider failure with credential" not in caplog.text


def test_rate_limit_has_friendly_error() -> None:
    client = Mock()
    error = RuntimeError("rate limit")
    error.status_code = 429
    client.models.generate_content.side_effect = error
    service = GeminiService(settings=make_settings(), client=client)

    with pytest.raises(GeminiServiceError, match="rate-limited"):
        service.generate_answer([{"role": "user", "content": "Question"}])


def test_model_not_found_has_actionable_error() -> None:
    client = Mock()
    error = RuntimeError("provider response details must not be exposed")
    error.status_code = 404
    client.models.generate_content.side_effect = error
    service = GeminiService(settings=make_settings(), client=client)

    with pytest.raises(GeminiServiceError, match="configured Gemini model") as result:
        service.generate_answer([{"role": "user", "content": "Question"}])

    assert "provider response details" not in str(result.value)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "Not a final user question"}],
        [{"role": "user", "content": "  "}],
        [{"role": "system", "content": "Unsupported role"}],
    ],
)
def test_rejects_invalid_history(messages: list[dict[str, str]]) -> None:
    service = GeminiService(settings=make_settings(), client=Mock())

    with pytest.raises(ValueError):
        service.generate_answer(messages)
