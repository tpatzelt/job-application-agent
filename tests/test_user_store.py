from __future__ import annotations

from pathlib import Path

from src.user_store import STATE_ACTIVE, UserRecord, UserStore


def test_record_round_trip(tmp_path: Path) -> None:
    store = UserStore(tmp_path)
    record = UserRecord(
        chat_id="123",
        name="Tim",
        state=STATE_ACTIVE,
        preferences={"location": "Berlin, Germany", "job_titles": ["PM"]},
        answers=[{"question": "q", "answer": "a"}],
        last_scan_at=1234.5,
    )
    store.save(record)
    loaded = store.load("123")
    assert loaded.to_dict() == record.to_dict()


def test_missing_user_returns_fresh_record(tmp_path: Path) -> None:
    store = UserStore(tmp_path)
    record = store.load("999")
    assert record.chat_id == "999"
    assert record.state == "new"
    assert not store.exists("999")


def test_list_chat_ids_and_isolation(tmp_path: Path) -> None:
    store = UserStore(tmp_path)
    store.save(UserRecord(chat_id="1"))
    store.save(UserRecord(chat_id="2"))
    assert store.list_chat_ids() == ["1", "2"]
    paths_1 = store.crawl_paths("1")
    paths_2 = store.crawl_paths("2")
    assert paths_1["cache_path"] != paths_2["cache_path"]
    assert "users/1" in str(paths_1["memory_path"])


def test_documents_and_reset(tmp_path: Path) -> None:
    store = UserStore(tmp_path)
    store.save(UserRecord(chat_id="7", state=STATE_ACTIVE))
    store.save_document("7", "cv", "my cv")
    assert store.load_document("7", "cv") == "my cv"

    record = store.reset("7")
    assert record.state == "new"
    assert store.load_document("7", "cv") == ""
    assert store.load("7").state == "new"


def test_chat_id_is_sanitized_for_paths(tmp_path: Path) -> None:
    store = UserStore(tmp_path)
    directory = store.user_dir("../evil")
    assert directory.parent == tmp_path / "users"
    assert ".." not in directory.name
