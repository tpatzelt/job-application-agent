# Hosting the Telegram bot service

The bot is a single always-on container that long-polls the Telegram API.
It exposes **no ports** — nothing to reverse-proxy or firewall — so hosting
it means: pull the image, give it an env file and a data volume, keep it
running.

## The image

Every push to `main` (and version tags) builds and publishes a public
image via GitHub Actions ([.github/workflows/docker.yml](../.github/workflows/docker.yml)):

```
ghcr.io/tpatzelt/job-application-agent:latest    # most recent build
ghcr.io/tpatzelt/job-application-agent:main      # last build of main
ghcr.io/tpatzelt/job-application-agent:sha-<sha> # immutable, per commit
```

No registry login is needed to pull. The image is **linux/amd64 only**
because it bundles Google Chrome for the headless-browser fallback; to run
on arm64, build locally with `chromium` swapped in (see the note in the
[Dockerfile](../Dockerfile)) or set `browser_fallback = false` in
`[tool.job_crawler.search]`.

## Requirements

- Docker with Compose, on amd64
- ~1.5 GB disk for the image; user data itself is tiny (text files + JSON)
- Outbound HTTPS only (Telegram, Brave Search, OpenRouter, job sites)

## Compose stack

```yaml
services:
  job-agent:
    image: ghcr.io/tpatzelt/job-application-agent:latest
    container_name: job-agent
    restart: unless-stopped
    env_file: ./.env
    volumes:
      - /opt/dockerdata/job-agent:/app/data
    # Headless Chrome needs more shared memory than Docker's 64MB default.
    shm_size: "1gb"
```

The `.env` file next to the compose file:

```bash
BRAVE_API_KEY=...        # https://api-dashboard.search.brave.com
OPENROUTER_API_KEY=...   # https://openrouter.ai/keys
TELEGRAM_BOT_TOKEN=...   # from @BotFather (/newbot)
LOG_LEVEL=INFO           # optional
JOB_CRAWLER_MAX_LLM_CALLS=40         # optional: overrides the per-run LLM-call cap
JOB_CRAWLER_MAX_SEARCH_ITERATIONS=8  # optional: overrides the per-run search-iteration cap
```

The two `JOB_CRAWLER_MAX_*` variables are the search budget (cost cap per
crawl). When unset they fall back to `[tool.job_crawler.budget]` in
`pyproject.toml`; setting them in `.env` lets you tune cost per deployment
without editing the checked-in config.

(`TELEGRAM_CHAT_ID` is only used by the one-shot CLI notifier; the bot
service discovers each user's chat itself.)

Start it:

```bash
docker compose up -d
docker compose logs -f   # expect: "Bot @<YourBot> online"
```

## State and migration

Everything the service knows lives in the mounted `/app/data` volume:

- `data/users/<chat_id>/` — per-user documents, preferences, seen-URL
  cache, agent memory, results
- `data/bot_offset.json` — Telegram update offset (prevents replaying
  messages after a restart)
- `data/logs/` — daily-rotated log files

To move an existing installation (e.g. from a dev machine to the server),
stop the old container and copy the whole `data/` directory into the new
host's volume path before starting. Without it the bot still works — users
just onboard again with `/start`.

Back up the volume path (`/opt/dockerdata/job-agent` above) like any other
container state; there is no database.

## Only run one instance

Telegram allows **one** `getUpdates` long-poller per bot token. Two
running instances (e.g. dev machine + server) will fight over updates and
one gets `409 Conflict` errors. Stop the old instance before starting the
new one, or create a second bot via @BotFather for development.

## Updating

```bash
docker compose pull && docker compose up -d
```

`:latest` moves on every push, so pin `:sha-<sha>` instead if you want
explicit control over upgrades.

## Configuration

Scan cadence, effort budgets, score threshold, etc. are baked into the
image from `pyproject.toml` (`[tool.job_crawler.*]` — see
[CLAUDE.md](../CLAUDE.md) for the full map). To change them, edit
`pyproject.toml` and push (CI builds a new image), or build locally:

```bash
docker build -t job-agent . && docker run -d --env-file .env \
  -v /opt/dockerdata/job-agent:/app/data --shm-size 1g job-agent
```
