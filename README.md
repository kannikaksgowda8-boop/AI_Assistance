# AI Question & Answering System

A modular Streamlit chat application that answers questions with Google's
Gemini API. It keeps conversation history for follow-up questions in the
current browser session and keeps API communication separate from the UI.

## Features

- ChatGPT-style interface built with Streamlit chat components.
- Multi-turn conversation history, with a one-click conversation reset.
- Google Gen AI Python SDK (`google-genai`) integration.
- Configurable model and generation temperature.
- Input and configuration validation, request timeout, and friendly API errors.
- No API keys or conversation contents in application logs.
- Unit tests that mock the Gemini SDK and do not make network requests.

## Architecture

- `app/config/settings.py` loads and validates environment-based settings.
- `app/services/gemini_service.py` validates history and encapsulates SDK calls.
- `app/prompts/qa_prompt.py` defines the model's system instruction.
- `app/main.py` renders the Streamlit UI and manages session-only conversation
  state.
- `app/utils/logging_config.py` configures safe application logging.
- `tests/` covers settings and the service using mocked SDK clients.

## Project structure

```text
question_answering_app/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── gemini_service.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── qa_prompt.py
│   └── utils/
│       ├── __init__.py
│       └── logging_config.py
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   └── test_gemini_service.py
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── run.sh
```

## Prerequisites

- Python 3.10 or newer.
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
- Internet connectivity to reach the Gemini API.

The default model, `gemini-2.5-flash`, is listed in Google's
[Gemini API model catalog](https://ai.google.dev/gemini-api/docs/models).
Model availability, quotas, and pricing can depend on your account and may
change; check the catalog and your account before deployment.

## Google Gemini API overview

The application sends the ordered conversation to the Gemini API using
Google's official `google-genai` Python SDK. The selected model receives the
question history and a separate system instruction; the configured temperature
controls response variation. The API key authenticates requests and must be
kept private. See Google's [Gemini API documentation](https://ai.google.dev/gemini-api/docs)
for supported models, API usage, rate limits, and billing details.

## Setup

Run commands from the project root.

### Create and activate a virtual environment

```shell
python -m venv .venv
```

macOS/Linux:

```shell
source .venv/bin/activate
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Configure Gemini

1. Create an API key in [Google AI Studio](https://aistudio.google.com/apikey).
2. Copy `.env.example` to `.env` in the project root.
3. Set your key and desired options in `.env`:

   ```dotenv
   GEMINI_API_KEY=your_actual_gemini_api_key
   GEMINI_MODEL=gemini-2.5-flash
   TEMPERATURE=0.2
   ```

`GEMINI_API_KEY` is required for requests. `GEMINI_MODEL` defaults to
`gemini-2.5-flash`. `TEMPERATURE` defaults to `0.2` and must be between `0` and
`2`, inclusive. Do not commit `.env`; it is ignored by Git. The example file
contains placeholders only.

### Run the application

macOS/Linux (using the helper script):

```shell
bash run.sh
```

All platforms:

```shell


python -m streamlit run app/main.py
```

Streamlit prints the local URL in the terminal. The application starts without
making a Gemini request; a request is made only after a non-empty question is
submitted.

### Run tests

```shell
python -m pytest
```

The unit tests use mocked Gemini responses and need no real API key.

## Example questions

- What is the difference between a list and a tuple in Python?
- Explain recursion with a small example.
- Can you summarize the main idea in your previous answer?
- What follow-up questions should I consider about this topic?

## Troubleshooting

- **Missing key:** Create `.env` in the project root and set `GEMINI_API_KEY`.
  Restart Streamlit after changing environment configuration.
- **Invalid key:** Confirm the key is active, copied without extra quotes or
  spaces, and authorized for the Gemini API.
- **Rate limit or quota:** Wait and retry, or review the quota and billing
  settings associated with your Google AI Studio project.
- **Model unavailable:** Check the
  [model catalog](https://ai.google.dev/gemini-api/docs/models), account access,
  and the exact `GEMINI_MODEL` value. An HTTP 404 means the selected model was
  not found or is unavailable to the API key/project; it does not by itself
  establish that the API key is invalid. The current catalog lists
  `gemini-2.5-flash`, but model access can vary. If it is unavailable to your
  project, set `GEMINI_MODEL` in `.env` to a model shown in the catalog and
  restart Streamlit. For example:

  ```dotenv
  GEMINI_MODEL=gemini-3.8-flash
  ```

  If the error continues, check the model's supported methods and API access
  for your key in Google AI Studio. Do not share your API key or raw provider
  error output when asking for help.
- **Invalid temperature:** Set `TEMPERATURE` to a number from `0` to `2`.
- **Dependency/import error:** Activate the virtual environment and rerun
  `python -m pip install -r requirements.txt`.
- **Streamlit port already in use:** Start with
  `python -m streamlit run app/main.py --server.port 8502`.

Provider error details and credentials are not displayed in the UI or written
to logs. Logs provide the exception class and, when available, HTTP status only.

## Security considerations

- Keep `.env` private; never commit or share it. `.env.example` is safe to track
  because it contains placeholders only.
- The app does not log the API key or complete prompts and answers.
- Conversation history is held in Streamlit session state and is not persisted
  by this application. Avoid entering sensitive information unless permitted by
  your organization's policy.
- For production, use the hosting platform's encrypted environment-variable or
  secrets manager rather than a local `.env` file. Restrict key permissions,
  rotate keys when needed, and monitor usage and quotas.
- Use HTTPS and appropriate authentication/access controls when deploying the
  app for users beyond your local machine.

## Stop and deactivate

- Stop Streamlit with `Ctrl+C` in the terminal running it.
- Deactivate the virtual environment with:

  ```shell
  deactivate
  ```

## Future enhancements

- Add conversation-length limits or summarization for long sessions.
- Add streaming output and richer model/API error classification.
- Add authenticated multi-user deployment and managed secret storage.
- Add optional integration tests against a separately configured test account.

## Deployment note

For Streamlit Community Cloud or another host, configure `GEMINI_API_KEY`,
`GEMINI_MODEL`, and `TEMPERATURE` as deployment environment variables and set
the app entry point to `app/main.py`. Do not publish a local `.env` file.
