"""Tests for `evals/rebaseline.py`. Each test pins one T-0018 guarantee:
(a) replay reproduces the committed baseline totals exactly; (b) the
export uses the given revision's src, not the working tree (pinned via
T-0006's stellenanzeigen.de addition); (c) a missing exported src/ fails
the replay loudly instead of falling back to the repo's src; (d) --out
matches evals/baseline.json's schema plus the resolved revision, and a
default run writes nothing; (e) the revision pinned in
evals/baseline.json wins, an unresolvable pin refuses rather than
falling back, candidates are only tried when no pin is recorded, and an
explicit --rev is used verbatim; (f) an unavailable git/revision skips
rather than fails.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evals import rebaseline

ROOT = Path(__file__).resolve().parent.parent


def _resolve_or_skip(rev: str | None = None) -> str:
    try:
        return rebaseline.resolve_revision(rev)
    except rebaseline.RebaselineError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def arming_rev() -> str:
    return _resolve_or_skip(None)


def test_replay_reproduces_committed_baseline_totals(arming_rev: str) -> None:
    # Compared against the committed baseline rather than against
    # hard-coded numbers: the arming replay is *how* that file is
    # produced, so this stays true across every legitimate re-freeze
    # instead of having to be edited alongside one.
    committed = json.loads((ROOT / "evals" / "baseline.json").read_text(encoding="utf-8"))
    report = rebaseline.replay(rebaseline.DEFAULT_CORPUS_DIR, arming_rev)
    totals = report["totals"]
    assert totals["records"] == committed["totals"]["records"]
    assert totals["kept"] == committed["totals"]["kept"]
    assert totals["metrics"] == committed["totals"]["metrics"]


def test_export_src_uses_old_src_not_working_tree(arming_rev: str, tmp_path: Path) -> None:
    rebaseline.export_src(arming_rev, tmp_path)
    exported = (tmp_path / "src" / "url_heuristics.py").read_text(encoding="utf-8")
    working_tree = (ROOT / "src" / "url_heuristics.py").read_text(encoding="utf-8")
    # kununu.com / devjobs.de were already in AGGREGATOR_HOSTS at arming;
    # T-0006 added stellenanzeigen.de (and eleven other hosts) afterwards.
    assert "devjobs.de" in exported
    assert "stellenanzeigen.de" not in exported
    assert "devjobs.de" in working_tree
    assert "stellenanzeigen.de" in working_tree


def test_run_offline_eval_fails_loudly_without_exported_src(arming_rev: str, tmp_path: Path) -> None:
    rebaseline.export_src(arming_rev, tmp_path)
    rebaseline._copy_evals_package(tmp_path)
    shutil.rmtree(tmp_path / "src")
    with pytest.raises(rebaseline.RebaselineError):
        rebaseline._run_offline_eval(tmp_path, rebaseline.DEFAULT_CORPUS_DIR, tmp_path / "out")


def test_out_writes_baseline_schema_and_default_leaves_baseline_untouched(arming_rev: str, tmp_path: Path) -> None:
    baseline_path = ROOT / "evals" / "baseline.json"
    before = baseline_path.read_bytes()
    out_path = tmp_path / "rebaselined.json"
    assert rebaseline.main(["--rev", arming_rev, "--out", str(out_path)]) == 0

    written = json.loads(out_path.read_text(encoding="utf-8"))
    committed = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert set(written) == set(committed) | {"revision"}
    assert set(written["totals"]) == set(committed["totals"])
    assert set(written["totals"]["metrics"]) == set(committed["totals"]["metrics"])
    assert written["revision"] == arming_rev

    assert rebaseline.main(["--rev", arming_rev]) == 0
    assert baseline_path.read_bytes() == before


def test_resolve_revision_prefers_the_baseline_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_verify(rev: str) -> str | None:
        calls.append(rev)
        return "deadbeef" if rev == "pinnedsha" else None

    monkeypatch.setattr(rebaseline, "_pinned_revision", lambda: "pinnedsha")
    monkeypatch.setattr(rebaseline, "_verify", fake_verify)
    assert rebaseline.resolve_revision(None) == "deadbeef"
    # The pin wins outright; the moving refs are never consulted.
    assert calls == ["pinnedsha"]


def test_resolve_revision_refuses_an_unresolvable_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    # Falling back to origin/main here would replay today's src against its own
    # baseline and report a pass, which is the one outcome that must not happen.
    monkeypatch.setattr(rebaseline, "_pinned_revision", lambda: "missingsha")
    monkeypatch.setattr(rebaseline, "_verify", lambda rev: None)
    with pytest.raises(rebaseline.RebaselineError, match="missingsha"):
        rebaseline.resolve_revision(None)


def test_resolve_revision_tries_candidates_in_order_and_verbatim_rev(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_verify(rev: str) -> str | None:
        calls.append(rev)
        return "deadbeef" if rev == "main" else None

    # No pin recorded: fall back to the moving refs, in order.
    monkeypatch.setattr(rebaseline, "_pinned_revision", lambda: None)
    monkeypatch.setattr(rebaseline, "_verify", fake_verify)
    assert rebaseline.resolve_revision(None) == "deadbeef"
    assert calls == list(rebaseline.CANDIDATE_REVISIONS)

    calls.clear()
    monkeypatch.setattr(rebaseline, "_verify", lambda rev: "cafef00d" if rev == "feature" else None)
    assert rebaseline.resolve_revision("feature") == "cafef00d"

    calls.clear()
    monkeypatch.setattr(rebaseline, "_verify", fake_verify)
    with pytest.raises(rebaseline.RebaselineError):
        rebaseline.resolve_revision("nonexistent")
    assert calls == ["nonexistent"]  # explicit --rev does not fall back to the candidates


def test_pinned_revision_reads_the_committed_baseline() -> None:
    assert rebaseline._pinned_revision() == json.loads(
        (ROOT / "evals" / "baseline.json").read_text(encoding="utf-8")
    )["revision"]


def test_skips_rather_than_fails_when_revision_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_unavailable(rev: str | None = None) -> str:
        raise rebaseline.RebaselineError("no candidate revision resolves")

    monkeypatch.setattr(rebaseline, "resolve_revision", raise_unavailable)
    with pytest.raises(pytest.skip.Exception):
        _resolve_or_skip()
