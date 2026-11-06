"""S4 path-class-aware submission scan: exclusion, prune, and budget pins.

The pins here cover the v0.7.3 S4 slice — the scanner's additive
``exclude_roots`` / ``prune_names`` / ``skipped`` / ``count_qualifying_only``
surface, the ``classify_scan_path`` path-class partition, the submission gate
that consumes both, and the untouched pre-commit hook. Seeded secrets are
assembled by concatenation (the pattern the other scanner pins use) so this
file itself stays clean under the secret scan.
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel
import pytest

from chartreux.core.events import AssistantEvent, ToolResultEvent
from chartreux.core.subagent_submission import (
    PRUNED_NAMES,
    UNKNOWN_WRITE_FOOTPRINT,
    classify_scan_path,
)
from chartreux.core.subagents import SubagentRunAccumulator
from chartreux.core.tools import secret_scan, secret_scan_hook
from chartreux.core.tools.builtins.bash_start import BashStart
from chartreux.core.tools.builtins.write_file import WriteFile, WriteFileResult
from chartreux.core.tools.secret_scan import (
    DEFAULT_MAX_SCAN_BYTES,
    DEFAULT_MAX_WALK_ENTRIES,
    scan_paths,
)
from chartreux.observability.logging import logger

AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
SECRET_LINE = f"GITHUB_TOKEN = {GITHUB_TOKEN}"
GITHUB_RULE_ID = "github-token"
AWS_RULE_ID = "aws-access-key-id"


def _seed_github(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{SECRET_LINE}\n", encoding="utf-8")


def _seed_aws(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# config\naws_access_key_id = {AWS_KEY}\n", encoding="utf-8")


def _set_mtime_ns(path: Path, mtime_ns: int) -> None:
    current = path.stat()
    os.utime(path, ns=(current.st_atime_ns, mtime_ns))


def _write_event(path: Path) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="write_file",
        tool_class=WriteFile,
        result=WriteFileResult(file_path=str(path), bytes_written=1, content="x"),
        tool_call_id="write-1",
    )


# --- Scanner: exclude_roots -------------------------------------------------


def test_named_file_under_an_excluded_root_is_skipped_and_recorded(
    tmp_path: Path,
) -> None:
    # The deliberate deviation from "an explicitly named file is always
    # inspected": a path under a caller-declared transient root is skipped
    # before any metadata is read and recorded on the out-list.
    scratch = tmp_path / "scratch"
    log = scratch / "suite.log"
    _seed_github(log)
    skipped: list[str] = []

    findings = scan_paths([log], exclude_roots=[scratch], skipped=skipped)

    assert findings == []
    assert skipped == [str(log)]
    # The exclusion is the caller's declaration, not a property of the
    # content: without it the same file still blocks. (The seeded line also
    # trips the entropy-gated generic rule; the provider rule is the pin.)
    assert GITHUB_RULE_ID in {finding.rule_id for finding in scan_paths([log])}


def test_excluded_root_directory_walk_skips_pre_and_post_cutoff_content(
    tmp_path: Path,
) -> None:
    # An excluded root is skipped as a walk root too: no enumeration, no read,
    # and no scan-incomplete sentinel for content the caller owns.
    scratch = tmp_path / "scratch"
    pre_session = scratch / "old.log"
    _seed_github(pre_session)
    _set_mtime_ns(pre_session, 1_000_000)
    in_window = scratch / "fresh.log"
    _seed_aws(in_window)
    cutoff = in_window.stat().st_mtime_ns - 1
    skipped: list[str] = []

    findings = scan_paths(
        [scratch], modified_since_ns=cutoff, exclude_roots=[scratch], skipped=skipped
    )

    assert findings == []
    assert skipped == [str(scratch)]
    # Control: the same walk without the exclusion inspects the in-window
    # file and finds its secret.
    assert [
        finding.rule_id for finding in scan_paths([scratch], modified_since_ns=cutoff)
    ] == [AWS_RULE_ID]


def test_exclusion_does_not_follow_a_link_out_of_the_excluded_tree(
    tmp_path: Path,
) -> None:
    # Both sides resolve: a link whose target lives outside the excluded root
    # is not excluded, so the skip cannot launder outside content.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    outside = tmp_path / "repo" / "app.py"
    _seed_github(outside)
    link = scratch / "link.py"
    link.symlink_to(outside)
    skipped: list[str] = []

    findings = scan_paths([link], exclude_roots=[scratch], skipped=skipped)

    assert skipped == []
    assert {finding.path for finding in findings} == {str(link)}
    assert GITHUB_RULE_ID in {finding.rule_id for finding in findings}


# --- Scanner: prune_names and the qualifying-only budget --------------------


def test_repo_walk_prunes_vendored_dirs_and_still_finds_repo_secrets(
    tmp_path: Path,
) -> None:
    # The pruned directories hold fresh (in-window) and oversized content: if
    # the walk entered them it would find their secrets or report the
    # over-cap binaries. The budget is one entry, so a counted pruned entry
    # would also exhaust the walk — none of that happens.
    repo = tmp_path / "repo"
    pack_dir = repo / ".git" / "objects" / "pack"
    pack_dir.mkdir(parents=True)
    oversized = DEFAULT_MAX_SCAN_BYTES + 16
    (pack_dir / "huge.pack").write_bytes(b"\x00" + b"x" * oversized)
    _seed_github(repo / ".git" / "staged-secret.txt")
    venv_lib = repo / ".venv" / "lib"
    venv_lib.mkdir(parents=True)
    (venv_lib / "blob.so").write_bytes(b"\x00" * oversized)
    _seed_aws(repo / ".venv" / "seeded.cfg")
    _seed_github(repo / ".pytest_cache" / "nodeids.txt")
    repo_file = repo / "src" / "app.py"
    _seed_github(repo_file)

    findings = scan_paths(
        [repo], prune_names=PRUNED_NAMES, count_qualifying_only=True, max_walk_entries=1
    )

    assert [finding.path for finding in findings] == [str(repo_file)] * len(findings)
    assert {finding.path for finding in findings} == {str(repo_file)}
    assert GITHUB_RULE_ID in {finding.rule_id for finding in findings}


def test_over_budget_sentinel_still_fires_for_qualifying_entries(
    tmp_path: Path,
) -> None:
    # The qualifying-only budget is still a budget: entries that DO qualify
    # exhaust it exactly like the default counting does.
    tree = tmp_path / "tree"
    tree.mkdir()
    for index in range(4):
        (tree / f"fresh-{index}.txt").write_text("clean\n", encoding="utf-8")

    findings = scan_paths([tree], max_walk_entries=2, count_qualifying_only=True)

    assert [finding.rule_id for finding in findings] == ["scan-incomplete"]
    assert findings[0].path == str(tree)
    assert "budget" in findings[0].description


def test_absolute_enumeration_cap_bounds_a_structural_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Structural entries are free under the qualifying-only budget, but the
    # raw enumeration cap still bounds the walk: past it the tree reports
    # scan-incomplete, never a silent clean scan.
    tree = tmp_path / "tree"
    tree.mkdir()
    for index in range(4):
        (tree / f"dir-{index}").mkdir()
    monkeypatch.setattr(secret_scan, "DEFAULT_MAX_WALK_ENUMERATIONS", 2)

    findings = scan_paths([tree], count_qualifying_only=True)

    assert [finding.rule_id for finding in findings] == ["scan-incomplete"]
    assert findings[0].path == str(tree)
    assert "2-entry" in findings[0].description


def test_a_large_repo_root_completes_under_the_qualifying_budget(
    tmp_path: Path,
) -> None:
    # The release-relevant shape: a repository root whose tree exceeds the
    # walk budget while only one entry is in the run's write window.
    repo = tmp_path / "repo"
    pre_session = repo / "old"
    pre_session.mkdir(parents=True)
    for index in range(DEFAULT_MAX_WALK_ENTRIES + 128):
        entry = pre_session / f"module-{index}.py"
        entry.write_text("value = 1\n", encoding="utf-8")
        _set_mtime_ns(entry, 1_000_000)
    repo_file = repo / "fresh.py"
    _seed_github(repo_file)
    cutoff = repo_file.stat().st_mtime_ns - 1

    # The default budget counts every enumerated entry, so this tree exhausts
    # it: the pre-S4 block, even though nothing relevant was left to inspect.
    default_findings = scan_paths([repo], modified_since_ns=cutoff)
    default_rules = [finding.rule_id for finding in default_findings]
    assert "scan-incomplete" in default_rules
    assert GITHUB_RULE_ID in default_rules

    # Counting only the entries the write window qualifies lets the walk
    # finish, and the repo artifact it does inspect is still found.
    scoped = scan_paths(
        [repo],
        modified_since_ns=cutoff,
        prune_names=PRUNED_NAMES,
        count_qualifying_only=True,
    )
    assert {finding.path for finding in scoped} == {str(repo_file)}
    assert GITHUB_RULE_ID in {finding.rule_id for finding in scoped}
    assert "scan-incomplete" not in {finding.rule_id for finding in scoped}

    # With no write window every entry qualifies, so the qualifying-only
    # budget exhausts the same 4096-entry budget: the sentinel still fires.
    unbounded = scan_paths([repo], prune_names=PRUNED_NAMES, count_qualifying_only=True)
    assert "scan-incomplete" in {finding.rule_id for finding in unbounded}


def test_a_top_level_path_is_never_pruned_by_name(tmp_path: Path) -> None:
    # Pruning applies during walks only: a witness root named directly keeps
    # its own content in scope.
    git_dir = tmp_path / ".git"
    _seed_github(git_dir / "hooks" / "pre-commit")

    findings = scan_paths([git_dir], prune_names=PRUNED_NAMES)

    assert {finding.path for finding in findings} == {
        str(git_dir / "hooks" / "pre-commit")
    }
    assert GITHUB_RULE_ID in {finding.rule_id for finding in findings}


# --- Classifier ------------------------------------------------------------


def test_paths_under_a_scratchpad_root_classify_transient(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    (scratch / "logs").mkdir(parents=True)
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)

    assert (
        classify_scan_path(scratch / "logs" / "suite.log", [scratch], [repo])
        == "transient"
    )
    assert classify_scan_path(repo / "src" / "app.py", [scratch], [repo]) == "repo"


def test_a_workspace_rooted_under_the_tempdir_keeps_its_subtree_in_the_scan(
    tmp_path: Path,
) -> None:
    # The precedence guard: a session workspace that lives under the OS
    # tempdir is repository content for that session, while its tempdir
    # siblings are transient.
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    if not work.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
        pytest.skip("this platform's pytest tmp dir is outside the OS tempdir")

    assert classify_scan_path(work / "src" / "app.py", (), [work]) == "repo"
    assert classify_scan_path(tmp_path / "unowned" / "x.log", (), [work]) == "transient"
    # With no authorized root established the tempdir branch cannot prove
    # anything, so the path stays in the scan (fail-closed).
    assert classify_scan_path(work / "src" / "app.py", (), ()) == "repo"


def test_a_scratchpad_symlink_into_the_repository_classifies_repo(
    tmp_path: Path,
) -> None:
    # The path resolves first, so only content that is really transient is
    # skipped.
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    link = scratch / "link.py"
    link.symlink_to(target)

    assert classify_scan_path(link, [scratch], [repo]) == "repo"


def test_classification_failures_classify_repo(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    not_a_path: Any = 123
    nothing: Any = None

    assert classify_scan_path(not_a_path, [scratch], ()) == "repo"
    assert classify_scan_path(nothing, [scratch], ()) == "repo"
    # A bare root string is not a root collection (it is never read as a
    # per-character root list): nothing is established, so nothing is skipped.
    assert classify_scan_path(scratch / "suite.log", str(scratch), ()) == "repo"


# --- Submission gate -------------------------------------------------------


class _NullFootprintResult(BaseModel):
    # A result model declaring write_paths with a None value: the shell's
    # write surface cannot be proven established, so it reads as unknown.
    write_paths: list[str] | None = None
    write_paths_unknown: bool = False


def test_gate_skips_scratchpad_only_paths_with_a_notice(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A scratchpad artifact carrying a real provider pattern no longer blocks
    # the submission: the path is transient, the skip is one harness log line,
    # and nothing about it reaches the agent.
    scratch = tmp_path / "scratch"
    log = scratch / "suite.log"
    _seed_github(log)
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="wrote the log"), tool_call_id="task")
    accumulator.observe(_write_event(log), tool_call_id="task")

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = accumulator.build_result(
            turns_used=1, contract=SimpleNamespace(shared_roots=(str(scratch),))
        )

    assert result.completed is True
    assert result.response == "wrote the log"
    assert "skipped by the secret scan as transient" in caplog.text
    assert str(log) in caplog.text
    assert "[transient]" in caplog.text
    assert "1 of 1 touched path(s)" in caplog.text
    # The skip is harness diagnostics only: no finding, no block, and no
    # scan detail in the agent-facing response.
    assert GITHUB_TOKEN not in result.response
    assert GITHUB_RULE_ID not in result.response


def test_gate_still_blocks_a_repo_artifact(tmp_path: Path) -> None:
    # Repo-class content is scanned exactly as before: the same real pattern
    # in a tracked workspace still blocks the submission.
    repo = tmp_path / "repo"
    artifact = repo / "deploy.py"
    _seed_github(artifact)
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="wrote the config"), tool_call_id="task")
    accumulator.observe(_write_event(artifact), tool_call_id="task")

    result = accumulator.build_result(
        turns_used=1,
        contract=SimpleNamespace(
            shared_roots=(str(tmp_path / "scratch"),), roots=(str(repo),), cwd=str(repo)
        ),
    )

    assert result.completed is False
    assert "failed the secret scan" in result.response
    assert str(artifact) in result.response
    assert GITHUB_TOKEN not in result.response
    assert GITHUB_RULE_ID not in result.response


def test_gate_still_blocks_the_unknown_write_footprint(tmp_path: Path) -> None:
    # The unknown-footprint sentinel stays blocking, contract or no contract.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="ran a command"), tool_call_id="task")
    accumulator.observe(
        ToolResultEvent(
            tool_name="bash_start",
            tool_class=BashStart,
            result=_NullFootprintResult(),
            tool_call_id="bash-start-1",
        ),
        tool_call_id="task",
    )

    result = accumulator.build_result(
        turns_used=1, contract=SimpleNamespace(shared_roots=(str(scratch),))
    )

    assert result.completed is False
    assert "could not be fully inspected" in result.response
    assert UNKNOWN_WRITE_FOOTPRINT in result.response


# --- Pre-commit hook: untouched --------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    return tmp_path


def test_the_hook_has_no_exclusion_surface() -> None:
    # S4 is a submission-gate scope change: the hook keeps its single
    # argument surface, so nothing it scans can be excluded.
    assert list(inspect.signature(secret_scan_hook.scan_staged).parameters) == [
        "filenames"
    ]


def test_the_hook_still_scans_staged_content_under_a_pruned_name(
    git_repo: Path,
) -> None:
    # A staged path under a prune-named directory is still scanned: pruning
    # narrows the submission walk only, and nothing reaches a commit
    # unexamined.
    (git_repo / "dist").mkdir()
    (git_repo / "dist" / "bundle.js").write_text(f"{SECRET_LINE}\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", "dist/bundle.js"], cwd=str(git_repo), check=True
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "chartreux.core.tools.secret_scan_hook",
            "dist/bundle.js",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(git_repo),
    )

    assert result.returncode == 1
    assert GITHUB_RULE_ID in result.stdout
    assert "Commit blocked" in result.stdout
    assert GITHUB_TOKEN not in result.stdout
