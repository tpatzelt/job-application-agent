from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Intake conversation states.
STATE_NEW = "new"
STATE_AWAITING_CV = "awaiting_cv"
STATE_AWAITING_MOTIVATION = "awaiting_motivation"
STATE_AWAITING_JOB_PREFS = "awaiting_job_prefs"
STATE_AWAITING_ANSWER = "awaiting_answer"
STATE_ACTIVE = "active"

_RECORD_FILE = "record.json"
_DOC_FILES = {
    "cv": "cv.txt",
    "motivation": "motivation.txt",
    "job_prefs": "job_description.txt",
}


@dataclass
class UserRecord:
    chat_id: str
    name: str = ""
    state: str = STATE_NEW
    preferences: dict[str, Any] = field(default_factory=dict)
    pending_questions: list[str] = field(default_factory=list)
    answers: list[dict[str, str]] = field(default_factory=list)
    last_scan_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "name": self.name,
            "state": self.state,
            "preferences": self.preferences,
            "pending_questions": self.pending_questions,
            "answers": self.answers,
            "last_scan_at": self.last_scan_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserRecord:
        return cls(
            chat_id=str(data.get("chat_id", "")),
            name=str(data.get("name", "")),
            state=str(data.get("state", STATE_NEW)),
            preferences=dict(data.get("preferences", {})),
            pending_questions=list(data.get("pending_questions", [])),
            answers=list(data.get("answers", [])),
            last_scan_at=float(data.get("last_scan_at", 0.0)),
        )


class UserStore:
    """Per-user persistence under <root>/users/<chat_id>/.

    Each user directory holds the intake record, the extracted document
    texts, and that user's crawl state (cache, memory, results) so runs
    for different users never share URL caches or query memory.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def user_dir(self, chat_id: str) -> Path:
        safe = re.sub(r"[^0-9A-Za-z_-]", "_", str(chat_id))
        return self._root / "users" / safe

    def exists(self, chat_id: str) -> bool:
        return (self.user_dir(chat_id) / _RECORD_FILE).exists()

    def load(self, chat_id: str) -> UserRecord:
        record_path = self.user_dir(chat_id) / _RECORD_FILE
        if not record_path.exists():
            return UserRecord(chat_id=str(chat_id))
        with record_path.open("r", encoding="utf-8") as handle:
            return UserRecord.from_dict(json.load(handle))

    def save(self, record: UserRecord) -> None:
        directory = self.user_dir(record.chat_id)
        directory.mkdir(parents=True, exist_ok=True)
        record_path = directory / _RECORD_FILE
        with record_path.open("w", encoding="utf-8") as handle:
            json.dump(record.to_dict(), handle, indent=2, ensure_ascii=False)

    def list_chat_ids(self) -> list[str]:
        users_dir = self._root / "users"
        if not users_dir.exists():
            return []
        return sorted(
            path.name
            for path in users_dir.iterdir()
            if (path / _RECORD_FILE).exists()
        )

    def save_document(self, chat_id: str, kind: str, text: str) -> None:
        directory = self.user_dir(chat_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _DOC_FILES[kind]).write_text(text, encoding="utf-8")

    def load_document(self, chat_id: str, kind: str) -> str:
        path = self.user_dir(chat_id) / _DOC_FILES[kind]
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def crawl_paths(self, chat_id: str) -> dict[str, Path]:
        directory = self.user_dir(chat_id)
        return {
            "cache_path": directory / "cache.json",
            "memory_path": directory / "memory.json",
            "results_json": directory / "results.json",
            "results_csv": directory / "results.csv",
        }

    def reset(self, chat_id: str) -> UserRecord:
        """Restart intake for a user, keeping crawl cache/memory intact."""
        record = UserRecord(chat_id=str(chat_id))
        directory = self.user_dir(chat_id)
        for name in _DOC_FILES.values():
            path = directory / name
            if path.exists():
                path.unlink()
        self.save(record)
        return record
