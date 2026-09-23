"""Recompute the G1 baseline against a past revision's `src/`, not the
working tree, so G2's "improves over baseline" comparison keeps meaning
"the quality the arming revision produced" as the codebase moves on.
Exports `src/` at a revision (default: the arming revision) into an
isolated tmpdir via `git archive`, copies the current `evals/` package
next to it, and runs `evals.offline_eval` there as a subprocess so it
imports the exported (old) `src`, never the working copy.

Usage: uv run python -m evals.rebaseline --corpus evals/fixtures [--rev <sha>] [--out <file>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .offline_eval import format_table

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = ROOT / "evals" / "fixtures"
CANDIDATE_REVISIONS = ("origin/main", "refs/remotes/origin/main", "main")

class RebaselineError(RuntimeError):
    """Raised when the revision can't be resolved or the replay fails."""

def _git(*args: str) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RebaselineError(f"git is not available: {exc}") from exc
    if result.returncode != 0:
        raise RebaselineError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()

def _verify(rev: str) -> str | None:
    try:
        return _git("rev-parse", "--verify", f"{rev}^{{commit}}")
    except RebaselineError:
        return None

def resolve_revision(rev: str | None = None) -> str:
    """Full commit hash to replay against: `rev` verbatim if given (no
    fallback), else the first of `CANDIDATE_REVISIONS` that verifies."""
    if rev is not None:
        resolved = _verify(rev)
        if resolved is None:
            raise RebaselineError(f"revision {rev!r} does not resolve to a commit")
        return resolved
    for candidate in CANDIDATE_REVISIONS:
        resolved = _verify(candidate)
        if resolved is not None:
            return resolved
    raise RebaselineError(
        f"none of {', '.join(CANDIDATE_REVISIONS)} resolve to a commit here; pass --rev explicitly"
    )

def export_src(rev: str, dest: Path) -> None:
    """Export `src/` at `rev` into `dest` via `git archive | tar -x`."""
    try:
        archive = subprocess.run(["git", "archive", rev, "src"], cwd=ROOT, capture_output=True)
    except FileNotFoundError as exc:
        raise RebaselineError(f"git is not available: {exc}") from exc
    if archive.returncode != 0:
        raise RebaselineError(archive.stderr.decode(errors="replace").strip() or f"git archive {rev} failed")
    extract = subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, capture_output=True)
    if extract.returncode != 0:
        raise RebaselineError(extract.stderr.decode(errors="replace").strip() or "tar extraction failed")

def _copy_evals_package(dest: Path) -> None:
    shutil.copytree(ROOT / "evals", dest / "evals")

def _run_offline_eval(tmpdir: Path, corpus_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Run `evals.offline_eval` with cwd=`tmpdir` so its `src` import
    resolves inside the tmpdir, not the working tree."""
    result = subprocess.run(
        [sys.executable, "-m", "evals.offline_eval", "--corpus", str(corpus_dir.resolve()), "--out", str(out_dir)],
        cwd=tmpdir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RebaselineError(f"replay subprocess exited {result.returncode}:\n{result.stderr}")
    return json.loads((out_dir / "report.json").read_text(encoding="utf-8"))

def replay(corpus_dir: Path, rev: str) -> dict[str, Any]:
    """Replay `corpus_dir` in an isolated tmpdir holding `src/` at `rev`."""
    with tempfile.TemporaryDirectory(prefix="rebaseline-") as tmp:
        tmpdir = Path(tmp)
        export_src(rev, tmpdir)
        _copy_evals_package(tmpdir)
        return _run_offline_eval(tmpdir, corpus_dir, tmpdir / "out")

def build_baseline(report: dict[str, Any], rev: str) -> dict[str, Any]:
    """Shape a replay report into `evals/baseline.json`'s schema plus `rev`."""
    return {
        "generated_at": report["generated_at"], "corpus_dir": report["corpus_dir"],
        "revision": rev, "totals": report["totals"],
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute the G1 baseline against a past revision's src/")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--rev", default=None, help="default: origin/main, refs/remotes/origin/main, main")
    parser.add_argument("--out", type=Path, default=None, help="write a baseline JSON (default: print only)")
    args = parser.parse_args(argv)

    try:
        rev = resolve_revision(args.rev)
        report = replay(args.corpus, rev)
    except RebaselineError as exc:
        print(f"rebaseline: {exc}", file=sys.stderr)
        return 1

    print(f"Replayed {args.corpus} against {rev}\n")
    print(format_table(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(build_baseline(report, rev), indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
