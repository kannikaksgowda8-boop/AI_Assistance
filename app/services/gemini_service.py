"""Gemini API integration using Google's official Gen AI SDK."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from google import genai
from google.genai import types

from app.config.settings import ConfigurationError, Settings, load_settings
from app.prompts.qa_prompt import SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)


class GeminiServiceError(RuntimeError):
    """Raised when Gemini cannot provide a usable answer."""


class GeminiService:
    """Generate answers from ordered conversation history."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._model = self._settings.model
        api_key = self._settings.require_api_key()

        if client is not None:
            self._client = client
            return

        try:
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=60_000),
            )
        except Exception as exc:
            logger.error(
                "Gemini client initialization failed (exception type: %s).",
                type(exc).__name__,
            )
            raise GeminiServiceError(
                "Could not initialize Gemini. Check your API configuration."
            ) from None

    def generate_answer(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> str:
        """Generate an answer from ordered user and assistant messages."""
        contents = self._format_history(messages)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=self._settings.temperature,
                ),
            )
        except Exception as exc:
            status_code = self._get_status_code(exc)
            category = self._classify_error(exc, status_code)
            logger.error(
                "Gemini request failed: category=%s, exception=%s, status=%s",
                category,
                type(exc).__name__,
                status_code if status_code is not None else "unknown",
            )
            raise GeminiServiceError(
                self._safe_error_message(category)
            ) from None

        if self._is_blocked(response):
            logger.warning("Gemini declined or blocked a response.")
            return (
                "I couldn't generate an answer for that request. "
                "Please rephrase your question and try again."
            )

        try:
            answer = response.text
        except Exception as exc:
            logger.warning(
                "Gemini response text was unavailable (exception type: %s).",
                type(exc).__name__,
            )
            answer = None

        if not isinstance(answer, str) or not answer.strip():
            logger.warning("Gemini returned no usable text.")
            return (
                "Gemini did not return a text answer. "
                "Please try rephrasing your question."
            )
        return answer.strip()

    @staticmethod
    def _format_history(
        messages: Sequence[Mapping[str, str]],
    ) -> list[types.Content]:
        """Convert application roles into Gemini's user/model roles."""
        if not messages:
            raise ValueError("Conversation history must include a user question.")

        contents: list[types.Content] = []
        for message in messages:
            role_value = message.get("role")
            text = message.get("content", "")
            if not isinstance(role_value, str):
                raise ValueError("Conversation message has an invalid role.")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Conversation messages must contain non-empty text."
                )

            role = role_value.strip().lower()
            if role == "user":
                sdk_role = "user"
            elif role in {"assistant", "model"}:
                sdk_role = "model"
            else:
                raise ValueError(
                    "Conversation roles must be user or assistant."
                )

            contents.append(
                types.Content(
                    role=sdk_role,
                    parts=[types.Part.from_text(text=text)],
                )
            )

        if contents[-1].role != "user":
            raise ValueError(
                "Conversation history must end with a user question."
            )

        for previous, current in zip(contents, contents[1:]):
            if previous.role == current.role:
                raise ValueError(
                    "Conversation history must alternate between user "
                    "and assistant messages."
                )
        return contents

    @staticmethod
    def _is_blocked(response: Any) -> bool:
        """Detect prompt-level or candidate-level safety blocking."""
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if (
            block_reason is not None
            and "UNSPECIFIED" not in str(block_reason).upper()
        ):
            return True

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return False

        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is None:
            return False
        return any(
            reason in str(finish_reason).upper()
            for reason in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT")
        )

    @staticmethod
    def _get_status_code(error: Exception) -> int | None:
        """Extract a status code without accessing provider error text."""
        for attribute in ("status_code", "code"):
            value = getattr(error, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value

            response = getattr(error, "response", None)
            response_status = getattr(response, "status_code", None)
            if isinstance(response_status, int) and not isinstance(
                response_status, bool
            ):
                return response_status
        return None

    @classmethod
    def _classify_error(
        cls,
        error: Exception,
        status_code: int | None,
    ) -> str:
        """Classify common failures without logging sensitive error details."""
        if status_code in {401, 403}:
            return "authentication"
        if status_code == 404:
            return "model_not_found"
        if status_code == 429:
            return "quota_or_rate_limit"
        if status_code is not None and status_code >= 500:
            return "provider_unavailable"
        if status_code in {400, 422}:
            return "invalid_request"

        exception_type = type(error).__name__.lower()
        if "timeout" in exception_type:
            return "timeout"
        if any(
            term in exception_type
            for term in ("connection", "network", "transport")
        ):
            return "connection"
        if "notfound" in exception_type or "not_found" in exception_type:
            return "model_not_found"
        if "permission" in exception_type or "authentication" in exception_type:
            return "authentication"
        if "resourceexhausted" in exception_type:
            return "quota_or_rate_limit"
        if "unavailable" in exception_type or "servererror" in exception_type:
            return "provider_unavailable"
        return "unknown"

    @staticmethod
    def _safe_error_message(category: str) -> str:
        """Return a user-friendly message without provider error details."""
        messages = {
            "authentication": (
                "Gemini rejected the request. Check that GEMINI_API_KEY "
                "is valid and authorized."
            ),
            "model_not_found": (
                "The configured Gemini model could not be found or is "
                "unavailable. Check GEMINI_MODEL and verify model access."
            ),
            "quota_or_rate_limit": (
                "Gemini is temporarily rate-limited or the available quota "
                "has been reached. Check your API usage and billing, then retry."
            ),
            "invalid_request": (
                "Gemini rejected the request configuration. Check the model, "
                "generation settings, and conversation history."
            ),
            "provider_unavailable": (
                "The Gemini service is temporarily unavailable. Please try again."
            ),
            "timeout": (
                "The Gemini request timed out. Check your connection and try again."
            ),
            "connection": (
                "Could not connect to Gemini. Check your internet connection "
                "and network settings."
            ),
            "unknown": (
                "Gemini could not complete the request. Check your configuration "
                "and try again."
            ),
        }
        return messages.get(category, messages["unknown"])
