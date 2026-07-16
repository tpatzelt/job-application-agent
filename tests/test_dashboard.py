from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from src.dashboard import (
    DashboardServer,
    collect_memory,
    collect_overview,
    collect_results,
    collect_users,
    list_log_files,
    tail_log,
)


def _make_data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    user = data / "users" / "111"
    user.mkdir(parents=True)
    (user / "record.json").write_text(
        json.dumps(
            {
                "chat_id": "111",
                "name": "Ada",
                "state": "active",
                "preferences": {
                    "locations": ["Berlin"],
                    "job_titles": ["ML Engineer"],
                    "language": "english",
                },
                "last_scan_at": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    (user / "results.json").write_text(
        json.dumps(
            [
                {
                    "title": "ML Engineer",
                    "company": "Acme",
                    "url": "https://acme.example/jobs/1",
                    "score": 85,
                    "reason": "good fit",
                    "status": "new",
                }
            ]
        ),
        encoding="utf-8",
    )
    (user / "memory.json").write_text(
        json.dumps(
            {
                "queries": {
                    "ml engineer berlin": {
                        "times_used": 2,
                        "urls_found": 10,
                        "new_urls": 4,
                        "accepted": 1,
                        "rejected": 0,
                    }
                },
                "domains": {"acme.example": {"accepted": 1, "rejected": 0}},
                "reflections": ["queries are working"],
            }
        ),
        encoding="utf-8",
    )
    (data / "results.json").write_text(
        json.dumps(
            [
                {
                    "title": "CLI Job",
                    "company": "Unknown",
                    "url": "https://cli.example/jobs/2",
                    "score": 72,
                    "reason": "ok",
                    "status": "new",
                }
            ]
        ),
        encoding="utf-8",
    )
    # Dev artifacts that must not show up as sources.
    (data / "mock_results.json").write_text("[]", encoding="utf-8")
    (data / "results_run1_backup.json").write_text("[]", encoding="utf-8")
    logs = data / "logs"
    logs.mkdir()
    (logs / "bot.log").write_text(
        "2026-07-16 10:00:00 INFO BotService: Bot online\n"
        "2026-07-16 10:01:00 ERROR BotService: boom\n",
        encoding="utf-8",
    )
    return data


def test_collect_results_aggregates_cli_and_users(tmp_path, monkeypatch):
    monkeypatch.delenv("LOG_DIR", raising=False)
    data = _make_data_dir(tmp_path)
    jobs = collect_results(data)
    assert {job["source"] for job in jobs} == {"results", "user:111"}
    user_job = next(job for job in jobs if job["source"] == "user:111")
    assert user_job["source_name"] == "Ada"
    assert user_job["score"] == 85
    assert all("mock" not in job["source"] for job in jobs)


def test_collect_users_and_overview(tmp_path, monkeypatch):
    monkeypatch.delenv("LOG_DIR", raising=False)
    data = _make_data_dir(tmp_path)
    users = collect_users(data)
    assert users == [
        {
            "chat_id": "111",
            "name": "Ada",
            "state": "active",
            "last_scan_at": 1000.0,
            "results_count": 1,
            "locations": ["Berlin"],
            "job_titles": ["ML Engineer"],
            "language": "english",
        }
    ]
    overview = collect_overview(data, status_provider=lambda: {"running": "111"})
    assert overview["users_active"] == 1
    assert overview["jobs_total"] == 2
    assert overview["last_scan_at"] == 1000.0
    assert overview["status"] == {"running": "111"}
    assert [f["name"] for f in overview["log_files"]] == ["bot.log"]


def test_overview_survives_failing_status_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("LOG_DIR", raising=False)
    data = _make_data_dir(tmp_path)

    def broken() -> dict:
        raise RuntimeError("nope")

    assert collect_overview(data, status_provider=broken)["status"] is None


def test_collect_memory(tmp_path, monkeypatch):
    monkeypatch.delenv("LOG_DIR", raising=False)
    data = _make_data_dir(tmp_path)
    memories = collect_memory(data)
    assert len(memories) == 1
    memory = memories[0]
    assert memory["source"] == "user:111"
    assert memory["queries"][0]["query"] == "ml engineer berlin"
    assert memory["domains"][0]["domain"] == "acme.example"
    assert memory["reflections"] == ["queries are working"]


def test_tail_log_and_traversal_guard(tmp_path, monkeypatch):
    monkeypatch.delenv("LOG_DIR", raising=False)
    data = _make_data_dir(tmp_path)
    lines = tail_log(data, "bot.log", lines=1)
    assert lines == ["2026-07-16 10:01:00 ERROR BotService: boom"]
    assert tail_log(data, "bot.log", lines=10) == [
        "2026-07-16 10:00:00 INFO BotService: Bot online",
        "2026-07-16 10:01:00 ERROR BotService: boom",
    ]
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    assert tail_log(data, "../../secret.txt") == []
    assert tail_log(data, "missing.log") == []


def test_list_log_files_empty_without_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("LOG_DIR", raising=False)
    assert list_log_files(tmp_path / "nothing") == []


def test_http_endpoints(tmp_path, monkeypatch):
    monkeypatch.delenv("LOG_DIR", raising=False)
    data = _make_data_dir(tmp_path)
    server = DashboardServer(
        data, host="127.0.0.1", port=0, status_provider=lambda: {"service": "bot"}
    )
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"

        def get(path: str) -> tuple[int, bytes]:
            with urllib.request.urlopen(base + path) as response:
                return response.status, response.read()

        status, body = get("/")
        assert status == 200
        assert b"Job Agent Dashboard" in body

        status, body = get("/api/overview")
        overview = json.loads(body)
        assert overview["status"] == {"service": "bot"}
        assert overview["jobs_total"] == 2

        status, body = get("/api/results")
        assert len(json.loads(body)) == 2

        status, body = get("/api/logs?file=bot.log&lines=1")
        payload = json.loads(body)
        assert payload["lines"] == ["2026-07-16 10:01:00 ERROR BotService: boom"]

        status, body = get("/api/memory")
        assert json.loads(body)[0]["source"] == "user:111"

        try:
            get("/nope")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        server.stop()
