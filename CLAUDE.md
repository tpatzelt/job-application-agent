# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agentic job finder: it uses the Brave Search API to find job listings, `botasaurus` to fetch and scrape page content, and an LLM (via `litellm`, default `openrouter/openrouter/free`) in a plan → act → reflect loop with persistent cross-run memory. Results above a score threshold are written to `data/results.json` / `data/results.csv`.

## Commands

```bash
uv sync                          # install dependencies
uv run python -m src.main        # run the real crawler (needs .env with BRAVE_API_KEY, OPENROUTER_API_KEY)
uv run python run_mock_test.py   # run the mock end-to-end loop (no network/API keys needed)
uv run pytest -q                 # run unit tests
uv run pytest tests/test_agent_loop.py::test_plan_generated_once_and_passed_to_query_context  # run a single test
```

Profile selection (see `pyproject.toml` `[tool.job_crawler.profiles.*]`, currently only `minimal`):

```bash
JOB_CRAWLER_PROFILE=minimal uv run python -m src.main
```

`LOG_LEVEL` env var controls log verbosity (default `INFO`).

CI (`.github/workflows/ci.yml`) runs `pytest -q` then `python run_mock_test.py` on every push/PR to `main`.

## Architecture

Everything is wired together in `src/main.py`, which loads config/secrets and constructs the three collaborating services passed into `Orchestrator`:

- **`config_manager.py`** — loads `Config` and `EffortBudget` from `pyproject.toml` (`[tool.job_crawler]`, merged with an optional `[tool.job_crawler.profiles.<name>]` override selected via `JOB_CRAWLER_PROFILE`). Also loads `.env` secrets (`BRAVE_API_KEY`, `OPENROUTER_API_KEY`), `user_profile.txt` (raw CV text), and `preferences.json`. `EffortBudget` is a shared mutable counter (LLM calls / search iterations) passed by reference into both `LLMService` and `CrawlerEngine` so a single run-wide cap is enforced across both.
- **`llm_service.py`** (`LLMService`) — builds JSON-only prompts for four tasks (plan the search, generate search queries, evaluate a job against the CV, reflect on search performance) and calls `litellm.completion`. Since free/small models often return malformed JSON, there's a multi-layer repair pipeline: `_parse_json_payload` → `_extract_json_object` (strip markdown fences, slice between first `{`/last `}`) → `_retry_json_response` (a second LLM call asking it to fix its own output). Every retry/repair call also consumes the shared `EffortBudget`. `plan_search`/`reflect` return safe defaults on validation failure.
- **`crawler_engine.py`** (`CrawlerEngine`) — `search()` calls the Brave API (via a `botasaurus`-wrapped `@request` task with retries) with rate limiting (min 1.5s between calls, exponential backoff on 429s) and filters out video results by URL/hostname heuristics. `fetch_job_text()` fetches a URL and strips it to plain text via BeautifulSoup (`soupify`), removing script/style/nav/header/footer. When called with `use_browser_fallback=True` (the orchestrator sets this for POSTING-classified URLs) and the plain fetch yields under `min_job_text_chars`, it retries with a headless-browser `@browser` task to handle JS-rendered ATS pages and 403-blocking boards, keeping whichever text is longer; browser failures degrade back to the plain result. Gated by `browser_fallback` in `[tool.job_crawler.search]`.
- **`orchestrator.py`** (`Orchestrator.run`) — the agent loop: (1) **plan** — ask the LLM for a `SearchPlan` (roles/skills/locations/strategy; failure is non-fatal); (2) **act** — while under `max_results`/budget, generate queries (context includes the plan, memory summary, and latest reflection), skip queries already searched this run or known-ineffective from memory, search each query, triage new URLs via `url_heuristics.classify_url()` and process individual postings first, generic careers pages next, and board index pages only if capacity remains (non-job URLs are dropped), skip pages under 800 chars, score remaining pages with the LLM, keep results scoring `>= min_score`; (3) **reflect** — between iterations, ask the LLM to critique query performance (falling back to `_heuristic_reflection` on any error) and feed the result into the next iteration's context. All search/fetch/evaluate calls go through a `ToolRegistry` whose telemetry is included in reflection prompts. If an iteration processes no new URLs the loop stops (stall detection). Maintains a `seen_urls` cache (`data/cache.json`) across runs. Writes `results.json`, `results.csv`, and a plain-text `results.txt` (one URL per line) at the end of every run.
- **`agent_memory.py`** (`AgentMemory`) — persistent cross-run memory (`data/memory.json`): per-query stats (uses, URLs found, new URLs, accepted/rejected), per-domain accept/reject counts, and recent reflection assessments (capped at 10). `summary_for_prompt()` produces the compact dict injected into planning/query-generation contexts; `ineffective_queries()` (used but never yielded a new URL) drives query skipping.
- **`tools.py`** (`ToolRegistry`) — named tool wrappers with call/error counts and last-error capture; exceptions propagate to the orchestrator's existing per-URL error handling.
- **`url_heuristics.py`** — pure-function URL classifier: `classify_url()` returns `POSTING` (ATS hosts like greenhouse/lever/workday/personio, or job-ID-shaped paths), `LISTING` (generic careers/jobs page), `INDEX` (aggregator search/list pages: Glassdoor `SRCH`, Stepstone `/jobs/<term>`, Indeed/LinkedIn search, or search-style query params), or `OTHER`. The orchestrator's `_triage_urls()` uses this to prioritize postings and down-rank index pages.
- **`models.py`** — Pydantic schemas (`SearchQueries`, `SearchPlan`, `Reflection`, `JobEvaluation`, `JobResult`) that constrain LLM output and the final result records. `SearchPlan`/`Reflection` fields all default, so partial LLM JSON still validates.

Agent behavior toggles live in `[tool.job_crawler.agent]` (`enable_planning`, `enable_reflection`, `ats_query_boost`, all default true; with reflection disabled the heuristic fallback is still used) and `memory_path` in `[tool.job_crawler.output]`. `ats_query_boost` makes the orchestrator prepend up to 2 deterministic `site:`-targeted queries per iteration (built from plan roles × `ATS_QUERY_SITES`, rotating via the already-searched set) so Brave surfaces individual postings instead of board indexes; the query-generation prompt also asks the LLM for such queries, but injection doesn't rely on the model complying.

### Mock mode

`src/mock_runner.py` provides `MockLLM`/`MockCrawler` fakes (fixed responses from `tests/mock_data.py`) wired through the *same* `Orchestrator` used in production, exercising the real control flow (plan, query generation, evaluation, reflection) without network calls. `run_mock_test.py` runs this loop and asserts exact call counts — useful as a fast sanity check and it's what CI runs after `pytest`. The unit tests in `tests/test_agent_loop.py` use richer `ScriptedLLM`/`ScriptedCrawler` fakes to pin down loop behavior (plan/reflection propagation into contexts, memory-driven query skipping, stall detection, budget/max-results stops).

## Conventions

- All modules use `from __future__ import annotations` and dataclass/Pydantic models for structured data.
- Config is threaded explicitly through constructors (no globals/singletons) — `Config` and `EffortBudget` are passed into `LLMService`, `CrawlerEngine`, and `Orchestrator` alike, which is what makes swapping in mocks straightforward.
- Failures inside the orchestrator's per-URL loop (fetch/score errors) are caught, logged, and the URL is added to `seen_urls` so a bad URL doesn't get retried on the next run.
