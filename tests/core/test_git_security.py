from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from git import Repo
import pytest

from chartreux.core.config import ProjectContextConfig
from chartreux.core.git.worktree import WorktreeRepository
from chartreux.core.system_prompt import ProjectContextProvider
from chartreux.utils.platform import resolve_ssh_executable


def _executable(path: Path, body: str = "exit 0") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def _fresh_resolution(
    project: Path,
    path: str,
    override: str | None = None,
    *,
    git_dir: str | None = None,
    git_work_tree: str | None = None,
) -> str | None:
    code = f"""
import json
from pathlib import Path
from chartreux.utils.platform import resolve_git_executable
print(json.dumps(resolve_git_executable(cwd=Path({str(project)!r}))))
"""
    env = os.environ.copy()
    env["PATH"] = path
    env.pop("GIT_PYTHON_GIT_EXECUTABLE", None)
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    if override is not None:
        env["GIT_PYTHON_GIT_EXECUTABLE"] = override
    if git_dir is not None:
        env["GIT_DIR"] = git_dir
    if git_work_tree is not None:
        env["GIT_WORK_TREE"] = git_work_tree
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout)


def test_trusted_git_discovery_rejects_project_relative_absolute_and_symlinked_entries(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    local_git = _executable(project / "git")
    link_dir = tmp_path / "links"
    link_dir.mkdir()
    (link_dir / "git").symlink_to(local_git)
    trusted_git = _executable(tmp_path / "trusted" / "git")

    discovered = _fresh_resolution(
        project,
        os.pathsep.join((
            "",
            ".",
            str(project),
            str(link_dir),
            str(trusted_git.parent),
        )),
    )

    assert discovered == str(trusted_git.resolve())


def test_trusted_git_discovery_rejects_repository_sibling_bin_from_nested_cwd(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    nested = project / "src"
    nested.mkdir(parents=True)
    (project / ".git").mkdir()
    local_git = _executable(project / "bin" / "git")
    trusted_git = _executable(tmp_path / "trusted" / "git")

    discovered = _fresh_resolution(
        nested, os.pathsep.join((str(local_git.parent), str(trusted_git.parent)))
    )

    assert discovered == str(trusted_git.resolve())


def test_trusted_git_discovery_checks_all_repository_ancestors(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    nested = outer / "inner" / "src"
    nested.mkdir(parents=True)
    (outer / ".git").mkdir()
    (outer / "inner" / ".git").write_text("not a gitdir marker\n")
    local_git = _executable(outer / "bin" / "git")
    trusted_git = _executable(tmp_path / "trusted" / "git")

    discovered = _fresh_resolution(
        nested, os.pathsep.join((str(local_git.parent), str(trusted_git.parent)))
    )

    assert discovered == str(trusted_git.resolve())


def test_trusted_git_discovery_rejects_bare_repository_executable(
    tmp_path: Path,
) -> None:
    bare = tmp_path / "repo.git"
    nested = bare / "work"
    nested.mkdir(parents=True)
    (bare / "HEAD").write_text("ref: refs/heads/main\n")
    (bare / "objects").mkdir()
    (bare / "refs").mkdir()
    local_git = _executable(bare / "bin" / "git")
    trusted_git = _executable(tmp_path / "trusted" / "git")

    discovered = _fresh_resolution(
        nested, os.pathsep.join((str(local_git.parent), str(trusted_git.parent)))
    )

    assert discovered == str(trusted_git.resolve())


def test_trusted_git_discovery_honors_git_dir_boundary(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    git_dir = project / ".metadata"
    git_dir.mkdir()
    local_git = _executable(project / "bin" / "git")
    trusted_git = _executable(tmp_path / "trusted" / "git")

    discovered = _fresh_resolution(
        project,
        os.pathsep.join((str(local_git.parent), str(trusted_git.parent))),
        git_dir=str(git_dir),
    )

    assert discovered == str(trusted_git.resolve())


def test_trusted_git_discovery_honors_git_work_tree_boundary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "src"
    nested.mkdir(parents=True)
    git_dir = tmp_path / "meta"
    git_dir.mkdir()
    local_git = _executable(workspace / "bin" / "git")
    trusted_git = _executable(tmp_path / "trusted" / "git")

    discovered = _fresh_resolution(
        nested,
        os.pathsep.join((str(local_git.parent), str(trusted_git.parent))),
        git_dir=str(git_dir),
        git_work_tree=str(workspace),
    )

    assert discovered == str(trusted_git.resolve())


def test_auto_discovered_git_pin_is_revalidated_for_each_cwd(tmp_path: Path) -> None:
    project = tmp_path / "project"
    nested = project / "src"
    nested.mkdir(parents=True)
    (project / ".git").mkdir()
    local_git = _executable(project / "bin" / "git")
    trusted_git = _executable(tmp_path / "trusted" / "git")
    outside = tmp_path / "outside"
    outside.mkdir()
    code = f"""
import json
from pathlib import Path
from chartreux.utils.platform import configure_git_python_executable, resolve_git_executable
first = configure_git_python_executable(cwd=Path({str(outside)!r}))
second = resolve_git_executable(cwd=Path({str(nested)!r}))
print(json.dumps([first, second]))
"""
    env = os.environ.copy()
    env.pop("GIT_PYTHON_GIT_EXECUTABLE", None)
    env["PATH"] = os.pathsep.join((str(local_git.parent), str(trusted_git.parent)))

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert json.loads(result.stdout) == [
        str(local_git.resolve()),
        str(trusted_git.resolve()),
    ]


def test_absolute_process_override_is_preserved_in_fresh_process(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    local_git = _executable(project / "tools" / "git")

    assert _fresh_resolution(project, "", str(local_git)) == str(local_git.resolve())


def test_resolve_ssh_executable_uses_trusted_absolute_path_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    trusted_ssh = _executable(tmp_path / "trusted" / "ssh")
    monkeypatch.setenv("PATH", str(trusted_ssh.parent))

    assert resolve_ssh_executable(cwd=project) == str(trusted_ssh.resolve())


def test_resolve_ssh_executable_skips_relative_path_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(tmp_path)
    _executable(tmp_path / "relative-bin" / "ssh")
    trusted_ssh = _executable(tmp_path / "trusted" / "ssh")
    monkeypatch.setenv(
        "PATH", os.pathsep.join(("relative-bin", str(trusted_ssh.parent)))
    )

    assert resolve_ssh_executable(cwd=project) == str(trusted_ssh.resolve())


def test_resolve_ssh_executable_rejects_project_local_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _executable(project / "ssh")
    trusted_ssh = _executable(tmp_path / "trusted" / "ssh")
    monkeypatch.setenv("PATH", os.pathsep.join((str(project), str(trusted_ssh.parent))))

    assert resolve_ssh_executable(cwd=project) == str(trusted_ssh.resolve())


def test_resolve_ssh_executable_returns_none_when_cwd_cannot_be_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()

    assert resolve_ssh_executable() is None


def test_git_unavailable_import_and_startup_are_graceful_in_fresh_process(
    tmp_path: Path,
) -> None:
    code = """
from pathlib import Path
from chartreux.core.git.errors import GitUnavailableError
from chartreux.core.git.repo import GitRepo
from chartreux.core.system_prompt import ProjectContextProvider
from chartreux.core.config import ProjectContextConfig
try:
    GitRepo.open(Path.cwd())
except GitUnavailableError:
    pass
else:
    raise AssertionError('GitRepo.open should report unavailable git')
status = ProjectContextProvider(ProjectContextConfig(), Path.cwd()).get_git_status()
assert 'No trusted Git executable' in status
"""
    env = os.environ.copy()
    env["PATH"] = ""
    env.pop("GIT_PYTHON_GIT_EXECUTABLE", None)
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)


def test_application_session_start_rejects_poisoned_project_git(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q", str(project)], check=True)
    marker = tmp_path / "poisoned-git-ran"
    poisoned = _executable(project / "bin" / "git", f'echo ran > "{marker}"; exit 99')
    code = f"""
from pathlib import Path
from chartreux.core.config.harness_files import init_harness_files_manager
from tests.conftest import build_test_agent_loop
from tests.stubs.fake_backend import FakeBackend

init_harness_files_manager("user", "project")
build_test_agent_loop(backend=FakeBackend(), cwd=Path({str(project)!r}))
"""
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(poisoned.parent), "/usr/bin", "/usr/local/bin"))
    env.pop("GIT_PYTHON_GIT_EXECUTABLE", None)

    subprocess.run([sys.executable, "-c", code], check=True, env=env)

    assert not marker.exists()


def _init_repo(root: Path) -> Repo:
    repo = Repo.init(root, initial_branch="main")
    repo.config_writer().set_value("user", "name", "Tester").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()
    (root / "file.txt").write_text("initial\n")
    repo.index.add(["file.txt"])
    repo.index.commit("initial")
    return repo


def _hook(repo: Repo, marker: Path, name: str) -> None:
    hook = Path(repo.git_dir) / "hooks" / name
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(f'#!/bin/sh\necho ran >> "{marker}"\n')
    hook.chmod(0o755)


@pytest.mark.skipif(os.name == "nt", reason="marker hooks use POSIX shell")
def test_successive_worktree_snapshot_and_removal_commands_stay_sanitized(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    repo = _init_repo(root)
    marker = tmp_path / "hook-ran"
    _hook(repo, marker, "post-checkout")
    _hook(repo, marker, "reference-transaction")
    fsmonitor = _executable(tmp_path / "fsmonitor", f'echo ran >> "{marker}"')
    repo.config_writer().set_value("core", "fsmonitor", str(fsmonitor)).release()

    with WorktreeRepository.open(root) as worktrees:
        prepared = worktrees.prepare("secured", branch="feat/secured")
    (prepared.root / "new.txt").write_text("new\n")
    recovery_ref = prepared.snapshot()
    prepared.remove()

    assert repo.commit(recovery_ref).hexsha
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="fake git uses POSIX shell")
def test_system_prompt_invokes_trusted_absolute_git(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    invocation = tmp_path / "invocation"
    fake_git = _executable(
        tmp_path / "trusted" / "git", f'printf "%s\\n" "$0" "$@" > "{invocation}"'
    )
    monkeypatch.delenv("GIT_PYTHON_GIT_EXECUTABLE", raising=False)
    monkeypatch.setenv("PATH", str(fake_git.parent))

    provider = ProjectContextProvider(ProjectContextConfig(), project)
    provider._run_git(["status", "--porcelain"], 5)

    argv = invocation.read_text().splitlines()
    assert argv[0] == str(fake_git.resolve())
    assert "core.fsmonitor=" in argv
    assert "core.hooksPath=/nonexistent-chartreux-disabled-git-hooks" in argv
