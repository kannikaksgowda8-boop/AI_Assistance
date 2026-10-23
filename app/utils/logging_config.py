"""Safe baseline logging configuration for the application."""

import logging


def configure_logging() -> None:
    """Configure concise logs without recording prompts or credentials."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
