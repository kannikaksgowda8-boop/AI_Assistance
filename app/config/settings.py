"""Environment-based application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from dotenv import load_dotenv

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid or incomplete."""


@dataclass(frozen=True)
class Settings:
    """Validated settings used by the application and Gemini service."""

    api_key: str | None = field(repr=False)
    model: str
    temperature: float

    def require_api_key(self) -> str:
        """Return the API key or raise a safe, actionable configuration error."""
        if not self.api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not configured. Create a .env file in the "
                "project root and add your Gemini API key."
            )
        return self.api_key


def load_settings(
    environ: Mapping[str, str] | None = None,
    *,
    load_dotenv_file: bool = True,
) -> Settings:
    """Load and validate settings without requiring an API key at startup.

    Passing ``environ`` makes configuration deterministic in tests and skips
    loading a user's local ``.env`` file.
    """
    if environ is None:
        if load_dotenv_file:
            load_dotenv()
        source = os.environ
    else:
        source = environ

    raw_temperature = source.get("TEMPERATURE", str(DEFAULT_TEMPERATURE))
    try:
        temperature = float(raw_temperature)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("TEMPERATURE must be a number between 0 and 2.") from exc

    if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
        raise ConfigurationError("TEMPERATURE must be between 0 and 2 for Gemini.")

    model = source.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    if not model:
        raise ConfigurationError("GEMINI_MODEL must not be empty.")

    api_key = source.get("GEMINI_API_KEY", "").strip() or None
    return Settings(
        api_key=api_key,
        model=model,
        temperature=temperature,
    )
