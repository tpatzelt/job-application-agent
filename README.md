# job-application-agent

Job application crawler that combines Brave Search, botasaurus, and an LLM scoring loop.

## Setup

1. Install dependencies with `uv`:

```bash
uv sync
```

2. Create a `.env` file with:

```bash
BRAVE_API_KEY=your_brave_key
OPENROUTER_API_KEY=your_openrouter_key
```

3. Update `user_profile.txt` and `preferences.json`.

## Run (real mode)

```bash
uv run python -m src.main
```

## Profile selection

Select a profile from `pyproject.toml` by setting the `JOB_CRAWLER_PROFILE` environment variable. For example to use the minimal profile:

```bash
JOB_CRAWLER_PROFILE=minimal uv run python -m src.main
```

## Run (mock mode)

```bash
uv run python run_mock_test.py
```

## Outputs

- Results JSON: `data/results.json`
- Results CSV: `data/results.csv`
- Cache: `data/cache.json`

## Notes

- The default model is `openrouter/openrouter/free`, which routes to free models on OpenRouter.
- Fetches use retries and per-request timeouts; content under 500 characters is skipped as likely non-job pages.
- LLM responses are repaired when JSON parsing fails, and Brave search uses backoff on rate limits.
- Default limits reduced to 3 results and 3 search iterations to lower request volume.
