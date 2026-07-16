from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .user_store import UserStore

logger = logging.getLogger(__name__)

# Top-level results/memory files that are dev artifacts, not run output.
_IGNORED_NAME_PARTS = ("mock", "backup")

StatusProvider = Callable[[], dict[str, Any]]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _is_run_artifact(path: Path) -> bool:
    return not any(part in path.name for part in _IGNORED_NAME_PARTS)


def _result_sources(data_dir: Path) -> list[tuple[str, str, Path]]:
    """(source id, display name, results.json path) for the CLI and each user."""
    sources: list[tuple[str, str, Path]] = []
    for path in sorted(data_dir.glob("results*.json")):
        if _is_run_artifact(path):
            label = "cli" if path.name == "results.json" else f"cli ({path.stem})"
            sources.append((path.stem, label, path))
    store = UserStore(data_dir)
    for chat_id in store.list_chat_ids():
        record = store.load(chat_id)
        name = record.name or chat_id
        sources.append(
            (f"user:{chat_id}", name, store.user_dir(chat_id) / "results.json")
        )
    return sources


def collect_results(data_dir: Path) -> list[dict[str, Any]]:
    """All accepted jobs across the CLI output and every bot user."""
    jobs: list[dict[str, Any]] = []
    for source_id, source_name, path in _result_sources(data_dir):
        entries = _read_json(path)
        if not isinstance(entries, list):
            continue
        updated_at = path.stat().st_mtime if path.exists() else None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            jobs.append(
                {
                    "source": source_id,
                    "source_name": source_name,
                    "title": str(entry.get("title", "")),
                    "company": str(entry.get("company", "")),
                    "url": str(entry.get("url", "")),
                    "score": entry.get("score"),
                    "reason": str(entry.get("reason", "")),
                    "status": str(entry.get("status", "")),
                    "updated_at": updated_at,
                }
            )
    jobs.sort(key=lambda job: (job["updated_at"] or 0, job["score"] or 0), reverse=True)
    return jobs


def collect_users(data_dir: Path) -> list[dict[str, Any]]:
    """Per-user intake state, scan recency, and result counts for the bot."""
    store = UserStore(data_dir)
    users: list[dict[str, Any]] = []
    for chat_id in store.list_chat_ids():
        record = store.load(chat_id)
        results = _read_json(store.user_dir(chat_id) / "results.json")
        preferences = record.preferences or {}
        users.append(
            {
                "chat_id": chat_id,
                "name": record.name,
                "state": record.state,
                "last_scan_at": record.last_scan_at or None,
                "results_count": len(results) if isinstance(results, list) else 0,
                "locations": preferences.get("locations") or [],
                "job_titles": preferences.get("job_titles") or [],
                "language": preferences.get("language") or "",
            }
        )
    return users


def collect_memory(data_dir: Path) -> list[dict[str, Any]]:
    """Query/domain stats and reflections from every memory.json."""
    sources: list[tuple[str, str, Path]] = []
    for path in sorted(data_dir.glob("memory*.json")):
        if _is_run_artifact(path):
            sources.append((path.stem, "cli", path))
    store = UserStore(data_dir)
    for chat_id in store.list_chat_ids():
        record = store.load(chat_id)
        sources.append(
            (
                f"user:{chat_id}",
                record.name or chat_id,
                store.user_dir(chat_id) / "memory.json",
            )
        )
    memories: list[dict[str, Any]] = []
    for source_id, source_name, path in sources:
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        queries = [
            {"query": query, **stats}
            for query, stats in data.get("queries", {}).items()
            if isinstance(stats, dict)
        ]
        queries.sort(key=lambda item: item.get("new_urls", 0), reverse=True)
        domains = [
            {"domain": domain, **stats}
            for domain, stats in data.get("domains", {}).items()
            if isinstance(stats, dict)
        ]
        domains.sort(
            key=lambda item: item.get("accepted", 0) + item.get("rejected", 0),
            reverse=True,
        )
        memories.append(
            {
                "source": source_id,
                "source_name": source_name,
                "queries": queries,
                "domains": domains,
                "reflections": data.get("reflections", []),
            }
        )
    return memories


def list_log_files(data_dir: Path) -> list[dict[str, Any]]:
    log_dir = Path(os.getenv("LOG_DIR") or data_dir / "logs")
    if not log_dir.is_dir():
        return []
    files = []
    for path in sorted(log_dir.iterdir()):
        if path.is_file():
            stat = path.stat()
            files.append(
                {"name": path.name, "size": stat.st_size, "mtime": stat.st_mtime}
            )
    return files


def tail_log(data_dir: Path, name: str, lines: int = 200) -> list[str]:
    """Last ``lines`` lines of a file inside the log directory.

    ``name`` must match a file listed by :func:`list_log_files`, which rules
    out path traversal without needing to sanitize.
    """
    if name not in {entry["name"] for entry in list_log_files(data_dir)}:
        return []
    path = Path(os.getenv("LOG_DIR") or data_dir / "logs") / name
    lines = max(1, min(lines, 2000))
    # Read backwards in blocks so multi-day logs don't get slurped whole.
    chunks: list[bytes] = []
    newlines = 0
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position > 0 and newlines <= lines:
            step = min(65536, position)
            position -= step
            handle.seek(position)
            chunk = handle.read(step)
            chunks.append(chunk)
            newlines += chunk.count(b"\n")
    text = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    return text.splitlines()[-lines:]


def collect_overview(
    data_dir: Path, status_provider: StatusProvider | None = None
) -> dict[str, Any]:
    users = collect_users(data_dir)
    jobs = collect_results(data_dir)
    last_scans = [user["last_scan_at"] for user in users if user["last_scan_at"]]
    overview: dict[str, Any] = {
        "users_total": len(users),
        "users_active": sum(1 for user in users if user["state"] == "active"),
        "jobs_total": len(jobs),
        "last_scan_at": max(last_scans) if last_scans else None,
        "log_files": list_log_files(data_dir),
        "users": users,
        "status": None,
    }
    if status_provider is not None:
        try:
            overview["status"] = status_provider()
        except Exception as exc:
            logger.warning("Dashboard status provider failed: %s", exc)
    return overview


class _Handler(BaseHTTPRequestHandler):
    server_version = "JobAgentDashboard/1.0"
    # Set by DashboardServer.
    data_dir: Path
    status_provider: StatusProvider | None = None

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path in ("/", "/index.html"):
                self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/overview":
                self._send_json(collect_overview(self.data_dir, self.status_provider))
            elif parsed.path == "/api/results":
                self._send_json(collect_results(self.data_dir))
            elif parsed.path == "/api/memory":
                self._send_json(collect_memory(self.data_dir))
            elif parsed.path == "/api/logs":
                name = query.get("file", [""])[0]
                try:
                    lines = int(query.get("lines", ["200"])[0])
                except ValueError:
                    lines = 200
                self._send_json(
                    {
                        "files": list_log_files(self.data_dir),
                        "file": name,
                        "lines": tail_log(self.data_dir, name, lines) if name else [],
                    }
                )
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:
            logger.exception("Dashboard request %s failed: %s", self.path, exc)
            self._send(500, b"internal error", "text/plain")

    def _send_json(self, payload: Any) -> None:
        self._send(
            200,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s %s", self.address_string(), format % args)


class DashboardServer:
    """Read-only monitoring UI over the ``data/`` directory.

    Serves a single-page dashboard plus JSON APIs. It only reads files the
    crawler/bot already write, so it can run inside the bot service (as a
    daemon thread, with live queue status injected via ``status_provider``)
    or standalone via ``python -m src.dashboard`` next to CLI runs.
    """

    def __init__(
        self,
        data_dir: Path,
        host: str = "127.0.0.1",
        port: int = 8765,
        status_provider: StatusProvider | None = None,
    ) -> None:
        handler = type(
            "BoundHandler",
            (_Handler,),
            {"data_dir": data_dir, "status_provider": staticmethod(status_provider)
             if status_provider else None},
        )
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="dashboard", daemon=True
        )
        self._thread.start()
        host, port = self._server.server_address[:2]
        logger.info("Dashboard listening on http://%s:%s", host, port)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job Agent Dashboard</title>
<style>
:root {
  color-scheme: light dark;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --accent: #2a78d6; --good: #0ca30c; --warning: #fab219;
  --serious: #ec835a; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --accent: #3987e5;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
header {
  display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  padding: 16px 24px 0;
}
header h1 { font-size: 18px; margin: 0; }
#service-status { color: var(--ink-2); font-size: 13px; }
main { padding: 16px 24px 40px; max-width: 1200px; margin: 0 auto; }
.tiles { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0 20px; }
.tile {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 16px; min-width: 150px; flex: 1;
}
.tile .label { color: var(--muted); font-size: 12px; }
.tile .value { font-size: 26px; font-weight: 600; }
.tile .sub { color: var(--ink-2); font-size: 12px; }
nav { display: flex; gap: 4px; border-bottom: 1px solid var(--grid); }
nav button {
  background: none; border: none; border-bottom: 2px solid transparent;
  color: var(--ink-2); font: inherit; padding: 8px 14px; cursor: pointer;
}
nav button.active { color: var(--ink); border-bottom-color: var(--accent); font-weight: 600; }
section { display: none; padding-top: 16px; }
section.active { display: block; }
table {
  width: 100%; border-collapse: collapse; background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
}
th, td {
  text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--grid);
  vertical-align: top;
}
th { color: var(--muted); font-size: 12px; font-weight: 600; }
tr:last-child td { border-bottom: none; }
td.num { font-variant-numeric: tabular-nums; text-align: right; }
th.num { text-align: right; }
a { color: var(--accent); }
.badge {
  display: inline-block; padding: 1px 8px; border-radius: 999px;
  font-size: 12px; font-weight: 600; border: 1px solid var(--border);
}
.badge.good { color: var(--good); }
.badge.warning { color: var(--warning); }
.badge.muted { color: var(--muted); }
.reason { color: var(--ink-2); font-size: 12px; max-width: 480px; }
.controls { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
select, .controls label { font: inherit; color: var(--ink-2); }
select {
  background: var(--surface); color: var(--ink); border: 1px solid var(--border);
  border-radius: 6px; padding: 4px 8px;
}
pre#log-view {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px; overflow-x: auto; font-size: 12px; line-height: 1.6;
  max-height: 70vh; overflow-y: auto; white-space: pre-wrap; word-break: break-word;
}
.log-ERROR, .log-CRITICAL { color: var(--critical); font-weight: 600; }
.log-WARNING { color: var(--serious); }
.log-DEBUG { color: var(--muted); }
.empty { color: var(--muted); padding: 24px; text-align: center; }
h3 { margin: 20px 0 8px; font-size: 14px; }
.wrap { overflow-x: auto; }
</style>
</head>
<body>
<header>
  <h1>Job Agent Dashboard</h1>
  <span id="service-status">loading…</span>
</header>
<main>
  <div class="tiles">
    <div class="tile"><div class="label">Jobs found</div><div class="value" id="tile-jobs">–</div></div>
    <div class="tile"><div class="label">Active users</div><div class="value" id="tile-users">–</div><div class="sub" id="tile-users-sub"></div></div>
    <div class="tile"><div class="label">Last scan</div><div class="value" id="tile-scan">–</div><div class="sub" id="tile-scan-sub"></div></div>
    <div class="tile"><div class="label">Scans queued / running</div><div class="value" id="tile-queue">–</div><div class="sub" id="tile-queue-sub"></div></div>
  </div>
  <nav>
    <button data-tab="jobs" class="active">Jobs</button>
    <button data-tab="users">Users</button>
    <button data-tab="logs">Logs</button>
    <button data-tab="memory">Memory</button>
  </nav>
  <section id="tab-jobs" class="active">
    <div class="controls">
      <label>User <select id="jobs-source"><option value="">all</option></select></label>
    </div>
    <div class="wrap"><table id="jobs-table">
      <thead><tr><th>Score</th><th>Job</th><th>User</th><th>Status</th><th>Why</th></tr></thead>
      <tbody></tbody>
    </table></div>
  </section>
  <section id="tab-users">
    <div class="wrap"><table id="users-table">
      <thead><tr><th>User</th><th>State</th><th>Locations</th><th>Roles</th><th class="num">Jobs</th><th>Last scan</th></tr></thead>
      <tbody></tbody>
    </table></div>
  </section>
  <section id="tab-logs">
    <div class="controls">
      <label>File <select id="log-file"></select></label>
      <label>Lines <select id="log-lines">
        <option>100</option><option selected>200</option><option>500</option><option>1000</option>
      </select></label>
      <label><input type="checkbox" id="log-follow" checked> follow</label>
    </div>
    <pre id="log-view">loading…</pre>
  </section>
  <section id="tab-memory"><div id="memory-view"></div></section>
</main>
<script>
"use strict";
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmtAgo = (ts) => {
  if (!ts) return "never";
  const mins = Math.round((Date.now() / 1000 - ts) / 60);
  if (mins < 1) return "just now";
  if (mins < 60) return mins + " min ago";
  if (mins < 48 * 60) return Math.round(mins / 60) + " h ago";
  return Math.round(mins / 1440) + " d ago";
};
const getJSON = (url) => fetch(url).then((r) => {
  if (!r.ok) throw new Error(r.status);
  return r.json();
});
const scoreClass = (s) => (s >= 80 ? "good" : s >= 70 ? "warning" : "muted");

let jobsCache = [];

function renderOverview(data) {
  $("#tile-jobs").textContent = data.jobs_total;
  $("#tile-users").textContent = data.users_active;
  $("#tile-users-sub").textContent = data.users_total + " total";
  $("#tile-scan").textContent = data.last_scan_at ? fmtAgo(data.last_scan_at) : "never";
  const st = data.status;
  if (st) {
    const queued = st.queued_or_running || [];
    $("#tile-queue").textContent = queued.length;
    $("#tile-queue-sub").textContent = st.running ? "running: " + st.running : "";
    $("#service-status").textContent =
      "bot service up since " + fmtAgo(st.started_at).replace(" ago", "");
  } else {
    $("#tile-queue").textContent = "–";
    $("#tile-queue-sub").textContent = "standalone mode";
    $("#service-status").textContent = "standalone (reading data/ only)";
  }
  const users = data.users || [];
  $("#users-table tbody").innerHTML = users.map((u) => `<tr>
    <td>${esc(u.name || u.chat_id)}</td>
    <td><span class="badge ${u.state === "active" ? "good" : "muted"}">${esc(u.state)}</span></td>
    <td>${esc((u.locations || []).join(", "))}</td>
    <td>${esc((u.job_titles || []).join(", "))}</td>
    <td class="num">${u.results_count}</td>
    <td>${fmtAgo(u.last_scan_at)}</td>
  </tr>`).join("") || `<tr><td colspan="6" class="empty">no bot users yet</td></tr>`;
}

function renderJobs() {
  const filter = $("#jobs-source").value;
  const jobs = jobsCache.filter((j) => !filter || j.source === filter);
  $("#jobs-table tbody").innerHTML = jobs.map((j) => `<tr>
    <td class="num"><span class="badge ${scoreClass(j.score)}">${esc(j.score)}</span></td>
    <td><a href="${esc(j.url)}" target="_blank" rel="noopener">${esc(j.title || j.url)}</a>
      ${j.company && j.company !== "Unknown" ? "<br><span class='reason'>" + esc(j.company) + "</span>" : ""}</td>
    <td>${esc(j.source_name)}</td>
    <td>${esc(j.status)}</td>
    <td class="reason">${esc(j.reason)}</td>
  </tr>`).join("") || `<tr><td colspan="5" class="empty">no jobs found yet</td></tr>`;
}

function loadJobs() {
  getJSON("/api/results").then((jobs) => {
    jobsCache = jobs;
    const select = $("#jobs-source");
    const current = select.value;
    const sources = [...new Map(jobs.map((j) => [j.source, j.source_name]))];
    select.innerHTML = '<option value="">all</option>' + sources.map(
      ([id, name]) => `<option value="${esc(id)}">${esc(name)}</option>`).join("");
    select.value = current;
    renderJobs();
  }).catch(() => {});
}

function loadLogs() {
  const file = $("#log-file").value;
  const lines = $("#log-lines").value;
  getJSON(`/api/logs?file=${encodeURIComponent(file)}&lines=${lines}`).then((data) => {
    const select = $("#log-file");
    if (select.options.length !== data.files.length + 0 ||
        [...select.options].some((o, i) => o.value !== (data.files[i] || {}).name)) {
      const current = select.value;
      select.innerHTML = data.files.map(
        (f) => `<option>${esc(f.name)}</option>`).join("");
      select.value = data.files.some((f) => f.name === current)
        ? current : (data.files[0] || {}).name || "";
      if (!file && select.value) { loadLogs(); return; }
    }
    const view = $("#log-view");
    const follow = $("#log-follow").checked;
    view.innerHTML = data.lines.map((line) => {
      const m = line.match(/ (DEBUG|INFO|WARNING|ERROR|CRITICAL) /);
      return `<span class="log-${m ? m[1] : "INFO"}">${esc(line)}</span>`;
    }).join("\\n") || '<span class="empty">log is empty</span>';
    if (follow) view.scrollTop = view.scrollHeight;
  }).catch(() => {});
}

function loadMemory() {
  getJSON("/api/memory").then((memories) => {
    $("#memory-view").innerHTML = memories.map((m) => `
      <h3>${esc(m.source_name)}</h3>
      <div class="wrap"><table>
        <thead><tr><th>Query</th><th class="num">Used</th><th class="num">Found</th>
        <th class="num">New</th><th class="num">Accepted</th><th class="num">Rejected</th></tr></thead>
        <tbody>${m.queries.map((q) => `<tr>
          <td>${esc(q.query)}</td><td class="num">${q.times_used ?? 0}</td>
          <td class="num">${q.urls_found ?? 0}</td><td class="num">${q.new_urls ?? 0}</td>
          <td class="num">${q.accepted ?? 0}</td><td class="num">${q.rejected ?? 0}</td>
        </tr>`).join("") || '<tr><td colspan="6" class="empty">no queries yet</td></tr>'}</tbody>
      </table></div>
      ${m.reflections.length ? `<div class="reason" style="margin-top:8px">
        Latest reflection: ${esc(m.reflections[m.reflections.length - 1])}</div>` : ""}
    `).join("") || '<div class="empty">no agent memory yet</div>';
  }).catch(() => {});
}

function selectTab(name) {
  const button = document.querySelector(`nav button[data-tab="${name}"]`);
  if (!button) return;
  document.querySelectorAll("nav button").forEach((b) => b.classList.remove("active"));
  document.querySelectorAll("section").forEach((s) => s.classList.remove("active"));
  button.classList.add("active");
  $("#tab-" + name).classList.add("active");
}
document.querySelectorAll("nav button").forEach((button) => {
  button.addEventListener("click", () => {
    location.hash = button.dataset.tab;
    selectTab(button.dataset.tab);
    refresh();
  });
});
if (location.hash) selectTab(location.hash.slice(1));
$("#jobs-source").addEventListener("change", renderJobs);
$("#log-file").addEventListener("change", loadLogs);
$("#log-lines").addEventListener("change", loadLogs);

function refresh() {
  getJSON("/api/overview").then(renderOverview).catch(() => {
    $("#service-status").textContent = "dashboard unreachable";
  });
  const tab = document.querySelector("nav button.active").dataset.tab;
  if (tab === "jobs") loadJobs();
  if (tab === "logs") loadLogs();
  if (tab === "memory") loadMemory();
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


def main() -> None:
    from dotenv import load_dotenv

    from .config_manager import load_config
    from .logging_setup import configure_logging

    root = Path(__file__).resolve().parent.parent
    configure_logging(root, "dashboard")
    load_dotenv()
    config = load_config(root, profile=os.getenv("JOB_CRAWLER_PROFILE"))
    server = DashboardServer(
        root / "data", config.dashboard_host, config.dashboard_port
    )
    server.start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
