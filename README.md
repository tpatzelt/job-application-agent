# job-application-agent

An agentic job finder that combines Brave Search, botasaurus, and an LLM to discover and score job listings against your CV.

## How the agent works

The orchestrator runs a **plan → act → reflect** loop built on classic AI agent patterns:

- **Planning** — the LLM first produces a `SearchPlan` (target roles, key skills, locations, strategy) from your CV and preferences, which steers all query generation.
- **Tool use** — every action (web search, page fetch, job scoring) runs through a `ToolRegistry` that records call/error telemetry.
- **Reflection** — after each iteration the LLM critiques query performance (using the history and tool telemetry) and suggests adjustments that feed into the next round of queries. If the LLM fails, a deterministic heuristic reflection takes over.
- **ATS-targeted queries** — each iteration injects a couple of `site:`-targeted queries against ATS hosts (Greenhouse, Lever, Personio, SmartRecruiters, Ashby, Workable, Join, Recruitee), rotating through the plan's roles, so search results contain individual postings rather than only board search pages.
- **Company-targeted queries** — the search plan also names real companies likely to hire for the target roles, and each iteration injects a `"<company>" careers <role>` query rotating through them, steering results onto employer career sites instead of aggregators.
- **URL triage** — search results are classified as individual postings (ATS pages, job-ID URLs), generic careers pages, or job-board index pages; employer-hosted postings are processed first, careers pages next, and aggregator-hosted postings only if capacity remains. Aggregator index pages (LinkedIn/Indeed/StepStone/Glassdoor search results) are dropped outright by default (`exclude_aggregator_sites`).
- **Careers-page harvesting** — a careers or board page that links to individual postings isn't scored itself: up to `max_harvest_links` posting links are extracted from it (employer/ATS hosts first) and each posting is fetched and scored instead, so results point at the actual job on the employer's site.
- **Browser fallback** — postings and careers pages that return little or no text over plain HTTP (JS-rendered ATS pages, 403-blocking boards) are refetched with a headless browser (`botasaurus` `@browser`); disable via `browser_fallback = false` in `[tool.job_crawler.search]`.
- **Persistent memory** — `data/memory.json` tracks query effectiveness and per-domain outcomes across runs, so the agent skips queries that never produced new URLs and leans into domains that yielded accepted jobs.
- **Effort budget** — a shared run-wide cap on LLM calls and search iterations keeps costs bounded.

Planning and reflection can be disabled via `[tool.job_crawler.agent]` in `pyproject.toml` (`enable_planning`, `enable_reflection`).

## Telegram bot service (multi-user, always on)

The recommended way to run the agent is as a persistent Telegram bot. Any
user who messages the bot gets their own onboarding flow and their own
isolated search profile, cache, and results:

1. **`/start`** — the bot welcomes the user and asks for their **CV**
   (PDF, DOCX, or text upload — or pasted as a message).
2. **Motivation letter** — uploaded next, or skipped with `/skip`.
3. **Job description** — a free-text description of the jobs they want
   (roles, industries, remote/on-site, locations).
4. The bot **extracts search parameters** (job titles, keywords,
   locations) from the documents with the LLM, and **asks follow-up
   questions** for anything essential that's missing — e.g. which country
   or cities to search in.
5. Once set up, the bot scans automatically every
   `scan_interval_hours` (default 6, see `[tool.job_crawler.bot]`) and
   messages the user when new matching jobs are found. `/run` triggers a
   scan immediately, `/status` shows the current parameters, `/reset`
   restarts onboarding.

Per-user state lives under `data/users/<chat_id>/` (documents, record,
seen-URL cache, agent memory, results), so users never share state.

### Run with Docker (recommended)

```bash
# .env needs BRAVE_API_KEY, OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN
docker compose up -d --build
```

The service restarts automatically (`restart: unless-stopped`) and
persists all user data in the mounted `./data` volume. Create the bot
token by messaging [@BotFather](https://t.me/BotFather) with `/newbot`.
The image includes Chrome for the headless-browser fallback (amd64; on
arm64 set `browser_fallback = false` or swap in chromium).

A prebuilt image is published to GHCR on every push
(`ghcr.io/tpatzelt/job-application-agent:latest`) — see
[docs/HOSTING.md](docs/HOSTING.md) for running it on a server/homelab
without building anything locally.

### Run without Docker

```bash
uv run python -m src.bot_service
```

## Setup

1. Install dependencies with `uv`:

```bash
uv sync
```

2. Create a `.env` file with:

```bash
BRAVE_API_KEY=your_brave_key
OPENROUTER_API_KEY=your_openrouter_key
# Optional, for Telegram notifications about new jobs:
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

3. (Single-user CLI only) copy the sample presets and edit them with your own
   details:

   ```bash
   cp user_profile.example.txt user_profile.txt
   cp preferences.example.json preferences.json
   ```

   These files are git-ignored and used only by the `src.main` CLI for local
   dev/testing. In the multi-user Telegram bot each user's profile and search
   preferences are collected through the intake conversation, not from these
   files — see [Telegram bot service](#telegram-bot-service-multi-user-always-on).
   If you
   skip this step the CLI falls back to the committed `*.example.*` samples.

## Telegram notifications

When `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set, every run ends by
sending the newly accepted jobs (title, score, URL) to your Telegram chat.
Runs that find nothing send nothing. Disable via `telegram = false` in
`[tool.job_crawler.notify]`.

One-time setup:

1. Message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`,
   and copy the token it gives you.
2. Send any message to your new bot (bots can't message you first).
3. Run the setup script, which verifies the token, discovers your chat ID
   from that message, and writes both into `.env`:

```bash
uv run python scripts/telegram_setup.py <bot_token>
```

It sends a test message to confirm delivery works. If it reports multiple
chat IDs (e.g. the bot is in a group too), set `TELEGRAM_CHAT_ID` manually.

To *keep* getting updates, schedule the crawler, e.g. with cron (twice daily):

```cron
0 9,18 * * * cd /path/to/job-application-agent && uv run python -m src.main
```

The seen-URL cache (`data/cache.json`) persists across runs, so each
notification only contains jobs you haven't been shown before.

## Run (single-user CLI mode)

The original one-shot CLI mode still works and uses the repo-level
`user_profile.txt` / `preferences.json` (git-ignored; falls back to the
committed `*.example.*` samples if you haven't created your own). This mode is
for a single user's local dev/testing — production, multi-user runs go through
the Telegram bot, which sets each user's preferences via intake:

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

## Offline result-quality harness (G1)

The harness is fully offline: it needs no `BRAVE_API_KEY` or `OPENROUTER_API_KEY`, replaying a
hand-labelled, committed corpus (`evals/fixtures/`, since `evals/runs/` is gitignored) through
the real triage/dedup logic instead of calling Brave or an LLM.

```bash
uv run python -m evals.offline_eval                 # print the metric table, write a report
uv run python -m evals.offline_eval --check-baseline # compare against evals/baseline.json, exit 1 on regression
```

The report lands in `evals/runs/offline/report.json` (gitignored). `--corpus` can point at a
different corpus directory, e.g. one built from a real recorded run, instead of the default
fixtures.

"Kept" means *would be reported as a result*, not merely "would be fetched". The replay applies
every drop rule in production order (dedup by canonical URL, non-job URL, aggregator index page,
redirected/dead/empty/stale page, no preferred location) and then keeps only URLs that classify
as `POSTING`: `Orchestrator._process_url` harvests posting links out of a careers or board page
and returns without scoring the hub itself, so a LISTING or INDEX page never becomes a
`JobResult`. One case is deliberately approximated — a LISTING page from which no posting links
can be harvested *is* scored directly in production, but corpus records carry no links field, so
the harness cannot replay that branch and counts such hubs as dropped.

Each metric is a rate in `[0, 1]` (`n/a` when its denominator is empty):

- **posting_shape_rate** — of the records kept by triage, the fraction that are actually job postings.
- **aggregator_drop_rate** — of the labelled aggregator index pages, the fraction triage correctly dropped.
- **location_match_rate** — of the records kept by triage, the fraction that mention one of the profile's preferred locations.
- **staleness_detection_rate** — of the labelled-stale records, the fraction triage correctly dropped.
- **dedup_rate** — of the labelled duplicate records, the fraction that collapse onto their original URL under canonicalization.

### The corpus

One JSONL file per search profile under `evals/fixtures/`; the file stem must name an entry in
`evals/profiles.py`, because that is where the profile's preferred locations come from (an
unknown stem yields no locations and `location_match_rate` stops meaning anything). Each line is
one recorded page — `url`, `title`, `final_url`, `http_status`, `text` — plus a `label` holding
the ground truth: `kind` (`posting`/`listing`/`index`/`other`), `aggregator`, `stale`,
`location_ok`, and `duplicate_of` (the URL this record is a duplicate of, or `null`).

Labels are read off the page by a human. They are never computed from `classify_url`,
`is_aggregator_url` or `find_stale_marker` — deriving them from the code under test would make
every metric tautological.

The corpus deliberately carries the URL shapes the deterministic logic gets *wrong*, since a
corpus it already handles cannot show a change in quality:

- postings and board roots on ATS vendors `url_heuristics._ats_kind` has no rule for
  (BambooHR, Teamtailor, iCIMS, Jobvite, softgarden, Pinpoint);
- non-postings whose path looks like a job id — aggregator category, employer and skill pages
  such as `jobs.meinestadt.de/<city>/skills/<id>`, and `/careers/<year>/<slug>` blog posts;
- index pages on each of the twelve aggregator hosts added to `AGGREGATOR_HOSTS` after arming;
- near-duplicate pairs that differ only by scheme, a leading `www.`, a tracking parameter,
  an `/apply` suffix, or query-parameter order;
- postings closed with the English and German wordings `page_signals.STALE_PHRASES` grew later.

URL shapes and page wording were sourced from live Brave searches against the real boards, so
the corpus reflects what the agent actually meets rather than what the heuristics expect.

### Baseline

`evals/baseline.json` is committed against the *arming* revision's `src/`, not the working
tree, so G2's "improves over baseline" comparison stays honest as the codebase and corpus both
move; `uv run python -m evals.rebaseline --corpus evals/fixtures` recomputes it by exporting
`src/` at a chosen revision (`--rev`, default the arming revision) via `git archive` into an
isolated tmpdir and replaying there — pass `--out <file>` to write a new baseline JSON, or
omit it to print only (it never writes `evals/baseline.json` itself).

Re-freeze the baseline whenever the corpus changes — the numbers only compare if both sides ran
over the same records:

```bash
uv run python -m evals.rebaseline --corpus evals/fixtures --out evals/baseline.json
```

The tool writes the resolved absolute corpus path into `corpus_dir`; nothing reads that field
(only `totals.metrics` is compared), so set it back to `evals/fixtures` before committing.

## Outputs

- Results JSON: `data/results.json` (accumulates across runs; earlier jobs are kept and marked `seen`)
- Results CSV: `data/results.csv`
- Results URLs: `data/results.txt`
- Run history (per-run counters, errors, queries): `data/runs.json`
- Cache (seen URLs, canonicalized): `data/cache.json`
- Agent memory (query/domain effectiveness): `data/memory.json`

## Notes

- The default model is `openrouter/openrouter/free`, which routes to free models on OpenRouter.
- Fetches use retries and per-request timeouts; content under `min_job_text_chars` (default 800) is skipped as likely non-job pages.
- LLM responses are repaired when JSON parsing fails, and Brave search uses backoff on rate limits.
- Per-run effort is capped by `[tool.job_crawler.budget]` (default 40 LLM calls, 8 search iterations) and `max_results` (default 5).
