from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from chartreux.core.config import ProjectContextConfig
from chartreux.core.system_prompt import ProjectContextProvider


@pytest.mark.skipif(os.name == "nt", reason="fake git shell script is POSIX-only")
def test_run_git_survives_non_utf8_output(tmp_path: Path, monkeypatch) -> None:
    # Fake git that prints bytes 0x80 0x81 (invalid UTF-8, and invalid gbk here)
    fake_git = tmp_path / "trusted-bin" / "git"
    fake_git.parent.mkdir()
    fake_git.write_text('#!/bin/sh\nprintf "commit \\200\\201 msg\\n"\n')
    fake_git.chmod(0o755)
    # Put the trusted fake first on PATH so _run_git executes it instead of real git
    monkeypatch.delenv("GIT_PYTHON_GIT_EXECUTABLE", raising=False)
    monkeypatch.setenv("PATH", f"{fake_git.parent}{os.pathsep}{os.environ['PATH']}")

    project = tmp_path / "project"
    project.mkdir()
    provider = ProjectContextProvider(ProjectContextConfig(), root_path=project)

    # Without encoding="utf-8", errors="replace" this raises UnicodeDecodeError
    result = provider._run_git(["log"], timeout=5.0)

    # The bad bytes are replaced with U+FFFD instead of crashing
    assert "\ufffd" in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="fake git shell script is POSIX-only")
def test_run_git_disables_fsmonitor_hook(tmp_path: Path, monkeypatch) -> None:
    # Fake git that records the argv it was invoked with, one arg per line.
    fake_git = tmp_path / "trusted-bin" / "git"
    fake_git.parent.mkdir()
    fake_git.write_text('#!/bin/sh\nfor a in "$@"; do echo "$a"; done\n')
    fake_git.chmod(0o755)
    monkeypatch.delenv("GIT_PYTHON_GIT_EXECUTABLE", raising=False)
    monkeypatch.setenv("PATH", f"{fake_git.parent}{os.pathsep}{os.environ['PATH']}")

    project = tmp_path / "project"
    project.mkdir()
    provider = ProjectContextProvider(ProjectContextConfig(), root_path=project)
    result = provider._run_git(["status", "--porcelain"], timeout=5.0)

    argv = result.stdout.splitlines()
    # -c core.fsmonitor= must come before any positional git subcommand so it
    # actually overrides the repo's own config, and must not be overridable by
    # anything the invoked repo could inject via its own .git/config.
    assert "-c" in argv
    assert argv[argv.index("-c") + 1] == "core.fsmonitor="
    assert argv.index("-c") < argv.index("status")


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell payload")
def test_fetch_git_status_does_not_execute_malicious_fsmonitor_hook(
    tmp_path: Path,
) -> None:
    # Regression test for the RCE reported in #942: a repo's own .git/config
    # can declare core.fsmonitor as an arbitrary command, which git runs on
    # `status` (and other worktree-refreshing commands) with the invoking
    # user's full privileges -- and this runs on every session start, before
    # any trust dialog is shown to the user.
    repo = tmp_path / "malicious_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("# README\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    payload_marker = tmp_path / "PWNED"
    subprocess.run(
        ["git", "config", "core.fsmonitor", f"touch {payload_marker}"],
        cwd=repo,
        check=True,
    )

    provider = ProjectContextProvider(ProjectContextConfig(), root_path=repo)
    status = provider.get_git_status()

    assert not payload_marker.exists()
    # The fix must not break normal status reporting.
    assert "Current branch:" in status
    assert "Git operations timed out" not in status
    assert "Not a git repository" not in status


def test_fetch_git_context_does_not_inspect_worktree_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = ProjectContextProvider(ProjectContextConfig(), root_path=tmp_path)
    calls: list[list[str]] = []

    def fake_run_git(
        args: list[str], _timeout: float
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        output = {
            ("branch", "--show-current"): "feature\n",
            ("branch", "-r"): "  origin/master\n",
            ("log", "--oneline", "-5", "--decorate"): "abc123 message\n",
        }[tuple(args)]
        return subprocess.CompletedProcess(args, 0, stdout=output)

    monkeypatch.setattr(provider, "_run_git", fake_run_git)

    context = provider._fetch_git_status()

    assert not any(args and args[0] == "status" for args in calls)
    assert "Current branch: feature" in context
    assert "Main branch (you will usually use this for PRs): master" in context
    assert "abc123 message" in context
    assert "Status:" not in context


@pytest.mark.parametrize("filter_kind", ["clean", "process"])
def test_automatic_context_never_executes_repository_filters(
    tmp_path: Path, filter_kind: str
) -> None:
    repo = tmp_path / "project"
    repo.mkdir()

    def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=repo, check=check, capture_output=True, text=True
        )

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (repo / ".gitattributes").write_text("fixture.txt filter=wp2\n")
    target = repo / "fixture.txt"
    target.write_text("original\n")
    git("add", ".")
    git("commit", "-q", "-m", "fixture")
    marker = tmp_path / "filter-executed"
    git(
        "config",
        f"filter.wp2.{filter_kind}",
        f"touch {marker}; cat" if filter_kind == "clean" else f"touch {marker}; exit 1",
    )
    target.write_text("modified\n")
    provider = ProjectContextProvider(ProjectContextConfig(), root_path=repo)
    context = provider.get_full_context()
    assert "Current branch:" in context
    assert "fixture" in context
    assert not marker.exists()
    # Positive control: the same dirty worktree really triggers the configured
    # filter through status; the process fixture deliberately aborts its protocol.
    git("status", "--porcelain", check=False)
    assert marker.exists()


@pytest.mark.parametrize("sources", [("user", "project"), ("project",), ("user",), ()])
@pytest.mark.parametrize("user_content", [None, "   \n", "user marker"])
@pytest.mark.parametrize("include_context", [False, True])
def test_prompt_instruction_provenance_tracks_only_loaded_documents(
    tmp_path, config_dir, monkeypatch, sources, user_content, include_context
):
    from chartreux.core.agents import AgentManager
    from chartreux.core.config.harness_files import HarnessFilesManager
    from chartreux.core.skills.manager import SkillManager
    from chartreux.core.system_prompt import get_universal_system_prompt
    from chartreux.core.trusted_folders import TrustedFoldersManager
    from tests.conftest import build_test_vibe_config
    from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

    project = tmp_path / "trusted" / "workspace"
    project.mkdir(parents=True)
    ancestor = project.parent / "AGENTS.md"
    ancestor.write_text("ancestor marker")
    project_doc = project / "AGENTS.md"
    project_doc.write_text("project marker")
    user_doc = config_dir / "AGENTS.md"
    if user_content is not None:
        user_doc.write_text(user_content)
    trust = TrustedFoldersManager()
    trust.trust_for_session(project.parent)
    harness = HarnessFilesManager(sources=sources, cwd=project, trust_store=trust)
    original = HarnessFilesManager.load_instruction_documents
    loads = []

    def load(self):
        documents = original(self)
        loads.append(documents)
        # Prove provenance and text do not rediscover/read documents independently.
        for doc in documents:
            doc.path.write_text("changed after loading")
        return documents

    monkeypatch.setattr(HarnessFilesManager, "load_instruction_documents", load)
    config = build_test_vibe_config(include_project_context=include_context)
    prompt = get_universal_system_prompt(
        config,
        SkillManager(lambda: config, harness_files=harness),
        AgentManager(FakeConfigOrchestrator(config), harness_files=harness),
        cwd=project,
        harness_files=harness,
    )
    expected = set()
    if include_context:
        assert len(loads) == 1
        if "user" in sources and user_content and user_content.strip():
            expected.add(user_doc.resolve())
            assert "user marker" in prompt
        if "project" in sources:
            expected.update([ancestor.resolve(), project_doc.resolve()])
            assert "ancestor marker" in prompt and "project marker" in prompt
        assert "changed after loading" not in prompt
    else:
        assert not loads
    assert prompt.instruction_read_files == frozenset(expected)
