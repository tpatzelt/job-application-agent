# job-application-agent

An agentic job finder that combines Brave Search, botasaurus, and an LLM to discover and score job listings against your CV.

## How the agent works

The orchestrator runs a **plan → act → reflect** loop built on classic AI agent patterns:

- **Planning** — the LLM first produces a `SearchPlan` (target roles, key skills, locations, strategy) from your CV and preferences, which steers all query generation.
- **Tool use** — every action (web search, page fetch, job scoring) runs through a `ToolRegistry` that records call/error telemetry.
- **Reflection** — after each iteration the LLM critiques query performance (using the history and tool telemetry) and suggests adjustments that feed into the next round of queries. If the LLM fails, a deterministic heuristic reflection takes over.
- **URL triage** — search results are classified as individual postings (ATS pages, job-ID URLs), generic careers pages, or job-board index pages; postings are processed first and index pages only if capacity remains.
- **Persistent memory** — `data/memory.json` tracks query effectiveness and per-domain outcomes across runs, so the agent skips queries that never produced new URLs and leans into domains that yielded accepted jobs.
- **Effort budget** — a shared run-wide cap on LLM calls and search iterations keeps costs bounded.

Planning and reflection can be disabled via `[tool.job_crawler.agent]` in `pyproject.toml` (`enable_planning`, `enable_reflection`).

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

![CI](https://github.com/tpatzelt/job-application-agent/actions/workflows/ci.yml/badge.svg)

## Tests

```bash
uv run pytest -q               # unit tests (memory, tools, agent loop, LLM parsing)
uv run python run_mock_test.py # mock end-to-end run of the real orchestrator, no network
```

Mock mode wires `MockLLM`/`MockCrawler` fakes through the same `Orchestrator` used in production and asserts exact call counts for planning, query generation, evaluation, and reflection. CI runs both on every push/PR to `main`.

## Outputs

- Results JSON: `data/results.json`
- Results CSV: `data/results.csv`
- Results URLs: `data/results.txt`
- Cache (seen URLs): `data/cache.json`
- Agent memory (query/domain effectiveness): `data/memory.json`

## Notes

- The default model is `openrouter/openrouter/free`, which routes to free models on OpenRouter.
- Fetches use retries and per-request timeouts; content under 500 characters is skipped as likely non-job pages.
- LLM responses are repaired when JSON parsing fails, and Brave search uses backoff on rate limits.
- Default limits reduced to 3 results and 3 search iterations to lower request volume.
