"""Streamlit chat interface for the Gemini question-answering application."""

import logging

import streamlit as st

from app.config.settings import ConfigurationError, Settings, load_settings
from app.services.gemini_service import GeminiService, GeminiServiceError
from app.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)
CONVERSATION_KEY = "conversation"


def _render_conversation(conversation: list[dict[str, str]]) -> None:
    for message in conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _get_settings() -> Settings | None:
    try:
        return load_settings()
    except ConfigurationError:
        logger.error("Application settings are invalid.")
        st.error(
            "Configuration error: check GEMINI_MODEL and TEMPERATURE in your "
            ".env file. TEMPERATURE must be between 0 and 2."
        )
        return None


def main() -> None:
    configure_logging()
    st.set_page_config(page_title="AI Question & Answering System", page_icon="💬")
    st.title("AI Question & Answering System")
    st.caption("Ask questions and get helpful answers powered by Google Gemini.")

    settings = _get_settings()
    if settings is None:
        return

    with st.sidebar:
        st.subheader("Conversation")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state[CONVERSATION_KEY] = []
            st.rerun()
        st.caption(f"Gemini model: `{settings.model}`")
        st.caption(f"Temperature: `{settings.temperature:g}`")

    if settings.api_key is None:
        st.error(
            "Gemini is not configured. Create a `.env` file in the project root "
            "and set `GEMINI_API_KEY` to your Google Gemini API key."
        )
        return

    if CONVERSATION_KEY not in st.session_state:
        st.session_state[CONVERSATION_KEY] = []
    conversation: list[dict[str, str]] = st.session_state[CONVERSATION_KEY]
    _render_conversation(conversation)

    question = st.chat_input("Ask a question")
    if question is None:
        return
    question = question.strip()
    if not question:
        st.warning("Please enter a question before submitting.")
        return

    try:
        service = GeminiService(settings=settings)
    except ConfigurationError:
        st.error(
            "Gemini is not configured. Create a `.env` file in the project root "
            "and set `GEMINI_API_KEY` to your Google Gemini API key."
        )
        return
    except GeminiServiceError:
        logger.error("Gemini service could not be initialized.")
        st.error("Gemini is temporarily unavailable. Please try again later.")
        return

    pending_history = [*conversation, {"role": "user", "content": question}]
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = service.generate_answer(pending_history)
            except GeminiServiceError as exc:
                st.error(str(exc))
                return
            except ValueError:
                logger.warning("Conversation history failed validation.")
                st.error("Conversation history is invalid. Clear it and try again.")
                return
        st.markdown(answer)

    st.session_state[CONVERSATION_KEY] = [
        *pending_history,
        {"role": "assistant", "content": answer},
    ]


if __name__ == "__main__":
    main()
