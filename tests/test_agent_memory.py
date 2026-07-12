from pathlib import Path

from src.agent_memory import AgentMemory


def test_record_query_and_effectiveness():
    memory = AgentMemory()
    memory.record_query("python jobs berlin", urls_found=5, new_urls=3)
    memory.record_query("cobol jobs mars", urls_found=0, new_urls=0)

    assert memory.known_queries() == {"python jobs berlin", "cobol jobs mars"}
    assert memory.ineffective_queries() == ["cobol jobs mars"]


def test_record_evaluation_updates_query_and_domain_stats():
    memory = AgentMemory()
    memory.record_query("python jobs", urls_found=2, new_urls=2)
    memory.record_evaluation(
        "https://boards.greenhouse.io/acme/jobs/1", accepted=True, query="python jobs"
    )
    memory.record_evaluation(
        "https://example.com/jobs/2", accepted=False, query="python jobs"
    )

    assert memory.effective_queries() == ["python jobs"]
    assert memory.productive_domains() == ["boards.greenhouse.io"]


def test_effective_queries_sorted_by_accepted_count():
    memory = AgentMemory()
    for query, accepted_count in [("a", 1), ("b", 3), ("c", 2)]:
        memory.record_query(query, urls_found=1, new_urls=1)
        for i in range(accepted_count):
            memory.record_evaluation(
                f"https://{query}.example.com/jobs/{i}", accepted=True, query=query
            )
    assert memory.effective_queries() == ["b", "c", "a"]


def test_save_and_load_roundtrip(tmp_path: Path):
    memory = AgentMemory()
    memory.record_query("python jobs", urls_found=4, new_urls=2)
    memory.record_evaluation(
        "https://example.com/jobs/1", accepted=True, query="python jobs"
    )
    memory.add_reflection("Broaden locations next time.")
    path = tmp_path / "memory.json"
    memory.save(path)

    loaded = AgentMemory.load(path)
    assert loaded.known_queries() == {"python jobs"}
    assert loaded.effective_queries() == ["python jobs"]
    assert loaded.productive_domains() == ["example.com"]
    assert "Broaden locations next time." in loaded.summary_for_prompt()[
        "recent_reflections"
    ]


def test_load_missing_or_corrupt_file_returns_empty(tmp_path: Path):
    assert AgentMemory.load(tmp_path / "nope.json").known_queries() == set()

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert AgentMemory.load(corrupt).known_queries() == set()


def test_reflections_capped(tmp_path: Path):
    memory = AgentMemory()
    for i in range(25):
        memory.add_reflection(f"reflection {i}")
    path = tmp_path / "memory.json"
    memory.save(path)
    loaded = AgentMemory.load(path)
    reflections = loaded.summary_for_prompt()["recent_reflections"]
    assert reflections == ["reflection 22", "reflection 23", "reflection 24"]


def test_summary_for_prompt_shape():
    memory = AgentMemory()
    summary = memory.summary_for_prompt()
    assert set(summary) == {
        "effective_queries",
        "ineffective_queries",
        "productive_domains",
        "recent_reflections",
    }
