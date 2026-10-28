"""A2 acceptance pins: the fail-closed clean-tree checkout exemption.

``git checkout <existing-local-branch>`` and ``git checkout -b <new>
[<start>]`` are exempt from dangerous-command approval only when the entire
composed shell command is exactly one eligible git invocation and every
candidate repository scope passes the clean-tree guardrails: a fully clean
work tree, no in-progress operation marker, no repository configuration
that selects a program, and — for the plain branch form — an existing LOCAL
target branch (an exact-ref sanitized probe, so commit-ishes never ride the
exemption). Every other shape stays gated:
pathspecs, forcing/conflict options, other options, chained or wrapped
calls, inline ``-c`` configuration, repository relocation, and non-branch
operands (tags, remote-tracking names, commit-ishes). ``git switch`` is
untouched.

The executed command string carries ``--no-overwrite-ignore`` — inserted
immediately after the ``checkout`` subcommand token — so an ignored-file
collision fails closed (checkout refuses) instead of silently clobbering
local content, while the original command string stays original on every
display and result surface, including the stored summary of a managed
(bash_start) job.

Coverage: branch-vs-file ambiguity; -b with/without start; every excluded
shape; clean/dirty/unknown state; every operation marker; hooks config; the
per-call guardrail cache. The deep-review repairs land their own regressions
here: the guardrail ORDERING (a program-selecting config declines before any
state probe spawns), the EFFECTIVE-WORKTREE probe (a core.worktree-relocated
work tree is the tree assessed), and the injection FAIL-CLOSED contract (a
rewrite failure on an exempt shape fails the call before execution on both
the foreground and the managed-start paths). A4's extension (below) lands
the dimensions A2's checkpoint deferred: the cross-call cache lifetime (no
cache survives a resolution, in neither direction), the reachable slice of
the scope-permutation dimension (scope resolution from a subdirectory —
true multi-scope permutations stay unreachable under the whole-call gate
and are documented there, not forced), the end-to-end conflicted workflows
(a conflicted squash-merge and an unpublished-branch conflicted rebase
complete through the real permission boundary with zero denials), the
adjudicated race cases (a conflicting concurrent change refuses and is
preserved; a non-conflicting one survives the successful non-forcing
checkout), the ignored-collision race (an ignored file created after the
permission probe still refuses through the executed flag), the
reviewer-context composition, and the denied destructive forms refused
before execution.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
import shutil
import subprocess
import time

import pytest

from chartreux.core.background_jobs import (
    BackgroundJobRegistry,
    BashListArgs,
    BashReadArgs,
    BashStartArgs,
    JobSummary,
)
from chartreux.core.tools.base import BaseToolState, InvokeContext, ToolPermission
from chartreux.core.tools.builtins import (
    _shell_permission_resolver as permission_resolver,
)
from chartreux.core.tools.builtins._shell_git_state import RepositoryState
from chartreux.core.tools.builtins._shell_permission_resolver import (
    CheckoutInjectionError,
    shell_command_checkout_no_overwrite_ignore,
)
import chartreux.core.tools.builtins.bash as bash_module
from chartreux.core.tools.builtins.bash import (
    Bash,
    BashArgs,
    BashToolConfig,
    FailedShellCommand,
)
from chartreux.core.tools.builtins.bash_start import BashStart
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.utils import DEFAULT_SENSITIVE_PATTERNS, PathAuthority
from chartreux.core.workspace import Workspace
from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run one git command in a fixture repository."""
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=check, capture_output=True, text=True
    )


def make_repo(path: Path) -> Path:
    """A clean repository: ``main`` with one commit, ``topic`` one commit
    ahead, and a tracked file named ``topic`` (the branch-vs-file ambiguity
    fixture), plus a tracked subdirectory for subdirectory resolution.
    """
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "pins@example.com")
    git(path, "config", "user.name", "Pins")
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (path / "topic").write_text("main version\n", encoding="utf-8")
    (path / "sub").mkdir()
    (path / "sub" / "inner.txt").write_text("inner\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")
    git(path, "checkout", "-q", "-b", "topic")
    (path / "topic").write_text("topic version\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "topic work")
    git(path, "checkout", "-q", "main")
    return path


def ignored_collision_repo(path: Path, *, collision: bool = True) -> Path:
    """A clean repo whose target branch tracks a path the work tree holds as
    an IGNORED file: a plain checkout silently clobbers it, the
    --no-overwrite-ignore form refuses. With ``collision=False`` the ignored
    file is left ABSENT so a test can create it after the permission probe
    (the A4 race-window case).
    """
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "pins@example.com")
    git(path, "config", "user.name", "Pins")
    (path / ".gitignore").write_text("build.txt\n", encoding="utf-8")
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")
    git(path, "checkout", "-q", "-b", "topic")
    (path / "build.txt").write_text("topic content\n", encoding="utf-8")
    git(path, "add", "-f", "build.txt")
    git(path, "commit", "-q", "-m", "topic adds build.txt")
    git(path, "checkout", "-q", "main")
    if collision:
        # On main the file is untracked and ignored, so the tree stays clean.
        (path / "build.txt").write_text("local precious content\n", encoding="utf-8")
    return path


def bash_tool(cwd: Path) -> Bash:
    return Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)


def resolve(cwd: Path, command: str) -> PermissionContext | None:
    return bash_tool(cwd).resolve_permission(BashArgs(command=command))


def assert_allowed(context: PermissionContext | None, command: str) -> None:
    assert isinstance(context, PermissionContext), command
    assert context.permission is ToolPermission.ALWAYS, command


def assert_gated(context: PermissionContext | None, command: str) -> None:
    assert isinstance(context, PermissionContext), command
    assert context.permission is ToolPermission.NEVER, command
    assert context.denial_kind == "dangerous_command", command
    proposal = context.dangerous_command_proposal
    assert proposal is not None, command
    assert proposal.operations == ("git.checkout",), command


def workspace_and_authority(cwd: Path) -> tuple[Workspace, PathAuthority]:
    workspace = Workspace.for_session(cwd)
    authority = PathAuthority(
        "bash",
        BashToolConfig().permission,
        (),
        (),
        tuple(DEFAULT_SENSITIVE_PATTERNS),
        workspace,
    )
    return workspace, authority


# --- the exempt shapes ---------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git checkout main",
        "git checkout topic",
        "git checkout -b feature",
        "git checkout -b feature main",
        "git checkout -b feature topic",
        "git --no-pager checkout topic",
        "git checkout 'topic'",
    ],
)
def test_exempt_checkout_shapes_are_allowed_on_a_clean_tree(
    tmp_path: Path, command: str
) -> None:
    repo = make_repo(tmp_path / "repo")
    assert_allowed(resolve(repo, command), command)


@pytest.mark.asyncio
async def test_exempt_checkout_executes_and_keeps_the_original_string(
    tmp_path: Path,
) -> None:
    # Branch-vs-file ambiguity: `git checkout topic` names both a local
    # branch and a tracked file; git switches to the branch, and the
    # exemption rides the branch form.
    repo = make_repo(tmp_path / "repo")
    assert_allowed(resolve(repo, "git checkout topic"), "git checkout topic")

    result = await collect_result(
        bash_tool(repo).run(BashArgs(command="git checkout topic"))
    )

    # The result surface keeps the ORIGINAL command string...
    assert result.command == "git checkout topic"
    # ...the branch switch happened (the branch won over the same-named
    # file)...
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "topic"
    assert (repo / "topic").read_text(encoding="utf-8") == "topic version\n"
    # ...and the A1 mutation-roots footprint is collected on the executed
    # string exactly as before.
    assert result.mutation_roots == [str(repo)]
    assert result.mutation_roots_unknown is False


@pytest.mark.asyncio
async def test_exempt_create_checkout_executes_with_the_flag(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    result = await collect_result(
        bash_tool(repo).run(BashArgs(command="git checkout -b feature"))
    )
    assert result.command == "git checkout -b feature"
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "feature"


@pytest.mark.asyncio
async def test_create_form_for_an_existing_branch_fails_at_execution(
    tmp_path: Path,
) -> None:
    # The create form does not probe the new branch's non-existence: git
    # itself refuses, so the call fails closed at execution.
    repo = make_repo(tmp_path / "repo")
    assert_allowed(resolve(repo, "git checkout -b topic"), "git checkout -b topic")
    with pytest.raises(FailedShellCommand):
        await collect_result(
            bash_tool(repo).run(BashArgs(command="git checkout -b topic"))
        )
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"


# --- the excluded shapes -------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # Pathspec forms without an end-of-options marker.
        "git checkout topic file.txt",
        # Forcing and conflict options.
        "git checkout -f topic",
        "git checkout --force topic",
        "git checkout --discard-changes topic",
        "git checkout -m topic",
        "git checkout --merge topic",
        "git checkout --orphan new",
        "git checkout --conflict=merge topic",
        "git checkout --conflict merge topic",
        "git checkout --pathspec-from-file=paths.txt topic",
        "git checkout --recurse-submodules topic",
        # Force-create and any other option.
        "git checkout -B new",
        "git checkout -q topic",
        "git checkout --detach topic",
        "git checkout -p topic",
        # Operand arity.
        "git checkout",
        "git checkout topic extra",
        "git checkout -b new main extra",
        # Non-branch operands: a tag, a remote-tracking name, commit-ishes,
        # and a branch that does not exist.
        "git checkout v1.0",
        "git checkout origin/main",
        "git checkout HEAD",
        "git checkout topic~1",
        "git checkout topic^",
        "git checkout no-such-branch",
    ],
)
def test_excluded_checkout_shapes_stay_gated(tmp_path: Path, command: str) -> None:
    repo = make_repo(tmp_path / "repo")
    git(repo, "tag", "v1.0")
    # A remote-tracking ref without any network: origin/main points at topic.
    git(repo, "update-ref", "refs/remotes/origin/main", "topic")
    assert_gated(resolve(repo, command), command)


@pytest.mark.parametrize(
    "command",
    [
        # An end-of-options marker is never a modeled grantable form.
        "git checkout -- file.txt",
        "git checkout main -- file.txt",
        "git checkout -- topic",
        # A dangling required option value is unmodeled too.
        "git checkout -b",
    ],
)
def test_unmodeled_checkout_forms_are_hard_guards(tmp_path: Path, command: str) -> None:
    # These forms are hard-denied with no grantable proposal, exactly as
    # before the exemption.
    repo = make_repo(tmp_path / "repo")
    context = resolve(repo, command)
    assert isinstance(context, PermissionContext), command
    assert context.permission is ToolPermission.NEVER, command
    assert context.denial_kind is None, command
    assert context.dangerous_command_proposal is None, command


# --- the whole-call eligibility gate -------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # Chained, piped, and subshelled statements.
        "git checkout topic && git status",
        "git checkout topic; git status",
        "git checkout topic | cat",
        "(git checkout topic)",
        "sh -c 'git checkout topic'",
        # Environment assignments and wrappers.
        "FOO=1 git checkout topic",
        "export FOO=1; git checkout topic",
        "env FOO=1 git checkout topic",
        "sudo -u root git checkout topic",
        # Inline configuration can alter the invocation's behavior.
        "git -c core.hooksPath=/tmp/hooks checkout topic",
        "git --config-env core.hooksPath=HOOKS checkout topic",
        # Repository relocation is not the bare form.
        "git -C . checkout topic",
        "git --git-dir .git checkout topic",
        "git --work-tree . checkout topic",
        # A redirect adds a second statement.
        "git checkout topic > out.txt",
        # Unmodeled brace-expansion syntax fails closed as a policy denial.
        "git checkout main^{}",
    ],
)
def test_whole_call_ineligible_shapes_stay_gated(tmp_path: Path, command: str) -> None:
    # The exemption applies only when the entire composed shell command is
    # exactly one eligible git invocation; any wrapper, assignment, chain,
    # pipe, subshell, redirect, inline configuration, or relocation fails
    # closed to the existing approval class.
    repo = make_repo(tmp_path / "repo")
    context = resolve(repo, command)
    assert isinstance(context, PermissionContext), command
    assert context.permission is ToolPermission.NEVER, command
    assert context.reason, command


def test_git_switch_classification_is_untouched(tmp_path: Path) -> None:
    # A2 exempts checkout forms only: the plain switch form stays routine,
    # the discarding forms stay gated with their own operation.
    repo = make_repo(tmp_path / "repo")
    assert_allowed(resolve(repo, "git switch topic"), "git switch topic")
    for command in ("git switch -f topic", "git switch --discard-changes topic"):
        context = resolve(repo, command)
        assert isinstance(context, PermissionContext), command
        assert context.permission is ToolPermission.NEVER, command
        assert context.denial_kind == "dangerous_command", command
        proposal = context.dangerous_command_proposal
        assert proposal is not None, command
        assert proposal.operations == ("git.switch.discard",), command


# --- the dirty / marker / hooks fail-closed cases ------------------------------


@pytest.mark.parametrize("dirty", ["modified", "staged", "untracked"])
def test_dirty_tree_fails_closed(tmp_path: Path, dirty: str) -> None:
    repo = make_repo(tmp_path / "repo")
    if dirty == "modified":
        (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    elif dirty == "staged":
        (repo / "tracked.txt").write_text("staged\n", encoding="utf-8")
        git(repo, "add", "-A")
    else:
        (repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    for command in ("git checkout topic", "git checkout -b feature"):
        assert_gated(resolve(repo, command), command)


@pytest.mark.parametrize(
    "marker",
    [
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "BISECT_LOG",
        "rebase-merge",
        "rebase-apply",
        "sequencer",
    ],
)
def test_in_progress_operation_marker_fails_closed(tmp_path: Path, marker: str) -> None:
    repo = make_repo(tmp_path / "repo")
    marker_path = repo / ".git" / marker
    if marker in {"rebase-merge", "rebase-apply", "sequencer"}:
        marker_path.mkdir()
    else:
        marker_path.write_text("in progress\n", encoding="utf-8")
    for command in ("git checkout topic", "git checkout -b feature"):
        assert_gated(resolve(repo, command), command)


@pytest.mark.parametrize(
    ("key", "value"),
    [("core.hooksPath", "/tmp/chartreux-fixture-hooks"), ("core.editor", "vim")],
)
def test_program_selecting_config_fails_closed(
    tmp_path: Path, key: str, value: str
) -> None:
    repo = make_repo(tmp_path / "repo")
    git(repo, "config", key, value)
    for command in ("git checkout topic", "git checkout -b feature"):
        assert_gated(resolve(repo, command), command)


def test_program_selecting_config_declines_before_any_state_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deep-review F1: the guardrail checks run the configuration-safety
    # probe FIRST — a pure config read that spawns nothing — because
    # `git status` can pass work-tree contents through repository-configured
    # clean/smudge process filters. A repository whose config selects a
    # program must be declined WITHOUT the subprocess-touching state probes
    # ever running.
    repo = make_repo(tmp_path / "repo")
    git(repo, "config", "core.hooksPath", "hooks")
    probes: list[str] = []
    config_calls: list[str] = []
    original_config_check = permission_resolver.config_selects_program

    def config_spy(state: RepositoryState) -> bool | None:
        config_calls.append("config_selects_program")
        return original_config_check(state)

    def clean_spy(state: RepositoryState) -> bool | None:
        probes.append("work_tree_is_clean")
        return True

    def no_marker_spy(state: RepositoryState) -> bool | None:
        probes.append("in_progress_operation")
        return False

    def branch_spy(state: RepositoryState, branch: str) -> bool:
        probes.append("local_branch_exists")
        return True

    monkeypatch.setattr(permission_resolver, "config_selects_program", config_spy)
    monkeypatch.setattr(permission_resolver, "work_tree_is_clean", clean_spy)
    monkeypatch.setattr(permission_resolver, "in_progress_operation", no_marker_spy)
    monkeypatch.setattr(permission_resolver, "local_branch_exists", branch_spy)
    for command in ("git checkout topic", "git checkout -b feature"):
        assert_gated(resolve(repo, command), command)
    # The configuration check ran and declined; no state probe ever ran.
    assert config_calls == ["config_selects_program", "config_selects_program"]
    assert probes == []


def test_unknown_state_fails_closed(tmp_path: Path) -> None:
    # A directory governed by no repository: the nominal scope carries no
    # git dir, so the guardrails cannot be established.
    plain = tmp_path / "plain"
    plain.mkdir()
    context = resolve(plain, "git checkout main")
    assert isinstance(context, PermissionContext)
    assert context.permission is ToolPermission.NEVER

    # A .git marker that is not a usable repository pointer.
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / ".git").write_text("not a repository pointer\n", encoding="utf-8")
    context = resolve(broken, "git checkout main")
    assert isinstance(context, PermissionContext)
    assert context.permission is ToolPermission.NEVER


def test_the_effective_work_tree_is_probed_not_the_discovery_directory(
    tmp_path: Path,
) -> None:
    # Deep-review F2: the exemption probes the EFFECTIVE work tree — the
    # same core.worktree-aware resolution an actual checkout honors — never
    # the discovery directory alone. A clean discovery directory whose
    # configured work tree is DIRTY must be denied. The workspace authorizes
    # the whole fixture root, so the relocated work tree passes the path
    # layer and the guardrails decide.
    def resolve_wide(cwd: Path, command: str) -> PermissionContext | None:
        tool = bash_tool(cwd)
        tool.workspace = Workspace.for_session(tmp_path)
        return tool.resolve_permission(BashArgs(command=command))

    repo = make_repo(tmp_path / "repo")
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    git(repo, "config", "core.worktree", str(worktree))
    for command in ("git checkout topic", "git checkout -b feature"):
        assert_gated(resolve_wide(repo, command), command)

    # Control: the divergence reads the other way — a DIRTY discovery
    # directory whose configured work tree is CLEAN keeps the exemption,
    # proving the probes assess the effective work tree, not the discovery
    # one.
    mirror = make_repo(tmp_path / "mirror")
    clean_worktree = tmp_path / "clean-worktree"
    clean_worktree.mkdir()
    for name in ("tracked.txt", "topic"):
        shutil.copyfile(mirror / name, clean_worktree / name)
    (clean_worktree / "sub").mkdir()
    shutil.copyfile(mirror / "sub" / "inner.txt", clean_worktree / "sub" / "inner.txt")
    (mirror / "tracked.txt").write_text("unsaved\n", encoding="utf-8")
    git(mirror, "config", "core.worktree", str(clean_worktree))
    for command in ("git checkout topic", "git checkout -b feature"):
        assert_allowed(resolve_wide(mirror, command), command)


def test_the_guardrail_cache_is_per_call(tmp_path: Path) -> None:
    # The exemption's state probes are cached per resolution, never across
    # calls: a tree that passes once and is dirtied afterwards gates the
    # very next resolution of the same command.
    repo = make_repo(tmp_path / "repo")
    tool = bash_tool(repo)
    first = tool.resolve_permission(BashArgs(command="git checkout topic"))
    assert_allowed(first, "git checkout topic")
    (repo / "tracked.txt").write_text("dirtied\n", encoding="utf-8")
    second = tool.resolve_permission(BashArgs(command="git checkout topic"))
    assert_gated(second, "git checkout topic")


# --- the executed-string flag injection ----------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git checkout topic", "git checkout --no-overwrite-ignore topic"),
        ("git checkout -b feature", "git checkout --no-overwrite-ignore -b feature"),
        (
            "git checkout -b feature main",
            "git checkout --no-overwrite-ignore -b feature main",
        ),
        (
            "git --no-pager checkout topic",
            "git --no-pager checkout --no-overwrite-ignore topic",
        ),
        ("git  checkout   topic", "git  checkout --no-overwrite-ignore   topic"),
        ("git checkout 'topic'", "git checkout --no-overwrite-ignore 'topic'"),
    ],
)
def test_exempt_spellings_rewrite_the_executed_string(
    tmp_path: Path, command: str, expected: str
) -> None:
    # The flag is injected immediately after the checkout subcommand token;
    # every other byte of the command string is preserved exactly. The flag
    # is not required as user input.
    repo = make_repo(tmp_path / "repo")
    workspace, authority = workspace_and_authority(repo)
    assert (
        shell_command_checkout_no_overwrite_ignore(
            command, workspace=workspace, authority=authority
        )
        == expected
    )


@pytest.mark.parametrize(
    "command",
    [
        "git checkout -f topic",
        "git checkout topic -- file.txt",
        "git checkout topic && git status",
        "FOO=1 git checkout topic",
        "git -c core.hooksPath=/tmp/hooks checkout topic",
        "git -C . checkout topic",
        "git checkout topic > out.txt",
        "git switch topic",
        "git switch -c feature",
        "git status",
    ],
)
def test_ineligible_spellings_do_not_rewrite(tmp_path: Path, command: str) -> None:
    repo = make_repo(tmp_path / "repo")
    workspace, authority = workspace_and_authority(repo)
    assert (
        shell_command_checkout_no_overwrite_ignore(
            command, workspace=workspace, authority=authority
        )
        is None
    )


def _injected_rewrite_failure(command: str, exemption: object) -> str:
    """The fault injection: the rewrite of an eligible exempt checkout fails."""
    raise RuntimeError("injected rewrite failure")


def test_an_eligible_rewrite_failure_raises_instead_of_returning_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deep-review F3, the unit pin of the fail-closed contract: the
    # executed-string re-derivation distinguishes "not an eligible checkout"
    # (None — the caller executes the command unmodified) from "eligible but
    # could not safely rewrite" (a hard error). A rewrite failure on an
    # exempt shape raises, never returns None, which would execute the
    # unprotected original string.
    repo = make_repo(tmp_path / "repo")
    workspace, authority = workspace_and_authority(repo)
    # Not an eligible checkout: None — the caller executes it unmodified.
    assert (
        shell_command_checkout_no_overwrite_ignore(
            "git checkout -f topic", workspace=workspace, authority=authority
        )
        is None
    )
    # Eligible, but the rewrite fails: a hard error, never None.
    monkeypatch.setattr(
        permission_resolver,
        "_exempt_checkout_executed_string",
        _injected_rewrite_failure,
    )
    with pytest.raises(CheckoutInjectionError):
        shell_command_checkout_no_overwrite_ignore(
            "git checkout topic", workspace=workspace, authority=authority
        )


@pytest.mark.asyncio
async def test_a_failed_injection_rewrite_fails_the_call_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deep-review F3, the foreground path: a rewrite failure on an exempt
    # shape fails the tool call BEFORE any subprocess spawns — the
    # unprotected original string never executes.
    repo = make_repo(tmp_path / "repo")
    spawned: list[str] = []

    async def refusing_spawn(
        command: str,
        *,
        cwd: Path | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> asyncio.subprocess.Process:
        spawned.append(command)
        raise AssertionError("the unprotected string must never execute")

    monkeypatch.setattr(bash_module, "spawn_shell_command", refusing_spawn)
    monkeypatch.setattr(
        permission_resolver,
        "_exempt_checkout_executed_string",
        _injected_rewrite_failure,
    )
    with pytest.raises(CheckoutInjectionError):
        await collect_result(
            bash_tool(repo).run(BashArgs(command="git checkout topic"))
        )
    assert spawned == []
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"


@pytest.mark.asyncio
async def test_a_failed_injection_rewrite_fails_the_managed_start_before_a_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deep-review F3, the managed-start (bash_start) path: the same
    # fail-closed contract — a rewrite failure on an exempt shape fails the
    # start BEFORE any job is admitted, so no job exists afterwards and the
    # unprotected original string never runs.
    repo = make_repo(tmp_path / "repo")
    registry = BackgroundJobRegistry()
    try:
        port = registry.root_port()
        ctx = InvokeContext(tool_call_id="start", background_jobs=port)
        tool = BashStart(
            config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo
        )
        monkeypatch.setattr(
            permission_resolver,
            "_exempt_checkout_executed_string",
            _injected_rewrite_failure,
        )
        with pytest.raises(CheckoutInjectionError):
            await _run_start(tool, BashStartArgs(command="git checkout topic"), ctx)
        listed = port.list(BashListArgs(include_finished=True))
        assert listed.jobs == ()
        assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"
    finally:
        await registry.aclose()


# --- the ignored-collision regression ------------------------------------------


@pytest.mark.asyncio
async def test_ignored_file_collision_refuses_instead_of_clobbering(
    tmp_path: Path,
) -> None:
    repo = ignored_collision_repo(tmp_path / "repo")
    # The exemption applies here: the tree is clean (the ignored file is
    # invisible to git status) and topic exists.
    assert_allowed(resolve(repo, "git checkout topic"), "git checkout topic")

    with pytest.raises(FailedShellCommand) as err:
        await collect_result(
            bash_tool(repo).run(BashArgs(command="git checkout topic"))
        )

    # The ignored local artifact survives: the executed string carried
    # --no-overwrite-ignore, so checkout refused the collision.
    assert (repo / "build.txt").read_text(encoding="utf-8") == (
        "local precious content\n"
    )
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"
    # The error surface keeps the ORIGINAL command string.
    assert str(err.value).splitlines()[0] == "Command failed: 'git checkout topic'"


async def _run_start(tool: BashStart, args: BashStartArgs, ctx: InvokeContext):
    return [result async for result in tool.run(args, ctx)][0]


async def _wait_for_terminal(
    registry: BackgroundJobRegistry, job_id: str, timeout: float = 10.0
) -> JobSummary:
    port = registry.root_port()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        listed = port.list(BashListArgs(include_finished=True))
        job = next((job for job in listed.jobs if job.job_id == job_id), None)
        if job is not None and job.state in {"exited", "failed", "stopped"}:
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach a terminal state")


@pytest.mark.asyncio
async def test_managed_start_executes_the_flag_and_stores_the_original(
    tmp_path: Path,
) -> None:
    # The same ignored-collision regression through the managed-start
    # (bash_start) path: the job runs the rewritten string and git refuses,
    # and every later snapshot shows the ORIGINAL command string.
    repo = ignored_collision_repo(tmp_path / "repo")
    registry = BackgroundJobRegistry()
    try:
        port = registry.root_port()
        ctx = InvokeContext(tool_call_id="start", background_jobs=port)
        tool = BashStart(
            config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo
        )
        result = await _run_start(
            tool, BashStartArgs(command="git checkout topic"), ctx
        )
        # The stored summary keeps the ORIGINAL command string.
        assert result.job.command == "git checkout topic"
        job_id = result.job.job_id

        summary = await _wait_for_terminal(registry, job_id)
        # The executed string carried the flag: git refused the collision.
        assert summary.exit_code != 0
        assert (repo / "build.txt").read_text(encoding="utf-8") == (
            "local precious content\n"
        )
        assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"

        # A later bash_list snapshot still shows the original command.
        listed = port.list(BashListArgs(include_finished=True))
        stored = next(job for job in listed.jobs if job.job_id == job_id)
        assert stored.command == "git checkout topic"
        # ...and so does a bash_read snapshot.
        read = await port.read(BashReadArgs(job_id=job_id))
        assert read.job.command == "git checkout topic"
    finally:
        await registry.aclose()


@pytest.mark.asyncio
async def test_managed_start_of_an_exempt_checkout_succeeds(tmp_path: Path) -> None:
    # The managed-start path executes the rewritten string and keeps the
    # stored summary original on success too.
    repo = make_repo(tmp_path / "repo")
    registry = BackgroundJobRegistry()
    try:
        port = registry.root_port()
        ctx = InvokeContext(tool_call_id="start", background_jobs=port)
        tool = BashStart(
            config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo
        )
        result = await _run_start(
            tool, BashStartArgs(command="git checkout topic"), ctx
        )
        assert result.job.command == "git checkout topic"
        summary = await _wait_for_terminal(registry, result.job.job_id)
        assert summary.exit_code == 0
        assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "topic"
        listed = port.list(BashListArgs(include_finished=True))
        stored = next(job for job in listed.jobs if job.job_id == result.job.job_id)
        assert stored.command == "git checkout topic"
    finally:
        await registry.aclose()


# --- A4: end-to-end conflicted workflows through the real boundary ------------


def diverged_repo(path: Path) -> Path:
    """A clean repository where ``main`` and ``topic`` diverged on
    ``shared.txt``: a squash-merge of topic into main conflicts, and so does
    a rebase of topic onto main.
    """
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "pins@example.com")
    git(path, "config", "user.name", "Pins")
    (path / "shared.txt").write_text("base\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")
    git(path, "checkout", "-q", "-b", "topic")
    (path / "shared.txt").write_text("topic\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "topic work")
    git(path, "checkout", "-q", "main")
    (path / "shared.txt").write_text("main\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "main work")
    return path


def assert_not_denied(context: PermissionContext | None, command: str) -> None:
    """Zero denials: the step's permission resolution at the real boundary
    must not be a dangerous-command refusal (or any refusal at all).
    """
    assert context is None or context.permission is ToolPermission.ALWAYS, command
    assert context is None or context.denial_kind != "dangerous_command", command


async def _run_step(tool: Bash, command: str) -> None:
    """One workflow step through the real permission boundary: resolved
    without a denial, then executed successfully.
    """
    assert_not_denied(tool.resolve_permission(BashArgs(command=command)), command)
    await collect_result(tool.run(BashArgs(command=command)))


@pytest.mark.asyncio
async def test_conflicted_squash_merge_completes_with_zero_denials(
    tmp_path: Path,
) -> None:
    # A conflicted squash-merge completes through the real permission
    # boundary with zero denials: the branch switches ride the exemption,
    # the merge itself is routine, git fails the conflicted merge on its
    # own (never a permission denial), and the resolution plus the
    # completing commit pass the boundary too.
    repo = diverged_repo(tmp_path / "repo")
    tool = bash_tool(repo)
    for command in ("git checkout topic", "git checkout main"):
        assert_allowed(tool.resolve_permission(BashArgs(command=command)), command)
        await _run_step(tool, command)
    with pytest.raises(FailedShellCommand):
        # git refuses the conflicted merge itself; the suite never needs a
        # permission denial to stop it.
        await _run_step(tool, "git merge --squash topic")
    # A conflicted squash-merge stops mid-way with unmerged paths and no
    # MERGE_HEAD: the squash is completed by a plain commit.
    assert git(repo, "status", "--porcelain").stdout.startswith("UU")
    assert not (repo / ".git" / "MERGE_HEAD").exists()
    (repo / "shared.txt").write_text("merged\n", encoding="utf-8")
    for command in ("git add shared.txt", "git commit -m 'squash topic'"):
        await _run_step(tool, command)
    assert git(repo, "status", "--porcelain").stdout == ""
    assert (repo / "shared.txt").read_text(encoding="utf-8") == "merged\n"
    # A squash merge is never a merge commit: exactly one parent.
    parents = git(repo, "rev-list", "--parents", "-n", "1", "HEAD").stdout.split()
    assert len(parents) == 2


@pytest.mark.asyncio
async def test_unpublished_branch_conflicted_rebase_completes_with_zero_denials(
    tmp_path: Path,
) -> None:
    # An unpublished-branch conflicted rebase completes through the real
    # permission boundary with zero denials: the sanctioned -b creation of
    # the unpublished branch rides the exemption, the rebase is routine, and
    # the conflicted rebase stops on git's own refusal before the resolution
    # and the continuation complete it. The continuation cannot block on an
    # editor: the conflict is committed with --no-edit first (rebase
    # --continue then only advances), so no program-selecting editor config
    # is ever needed — which also keeps the -b creation exempt.
    repo = diverged_repo(tmp_path / "repo")
    tool = bash_tool(repo)
    command = "git checkout -b feature topic"
    assert_allowed(tool.resolve_permission(BashArgs(command=command)), command)
    await _run_step(tool, command)
    with pytest.raises(FailedShellCommand):
        await _run_step(tool, "git rebase main")
    assert git(repo, "status", "--porcelain").stdout.startswith("UU")
    (repo / "shared.txt").write_text("merged\n", encoding="utf-8")
    for command in (
        "git add shared.txt",
        "git commit --no-edit",
        "git rebase --continue",
    ):
        await _run_step(tool, command)
    # The rebase completed onto main: main's commit is an ancestor, the
    # feature commit is replayed on top with its message, and the resolved
    # content won. The branch never had a remote: unpublished, and the whole
    # workflow stayed agent-executable.
    assert git(repo, "merge-base", "--is-ancestor", "main", "feature").returncode == 0
    assert git(repo, "log", "--format=%s", "-n", "1", "feature").stdout.strip() == (
        "topic work"
    )
    assert (repo / "shared.txt").read_text(encoding="utf-8") == "merged\n"
    assert git(repo, "remote").stdout.strip() == ""


def _denial_case_repo(tmp_path: Path, case: str) -> Path:
    """The fixture state for one denied destructive form."""
    repo = make_repo(tmp_path / "repo")
    if case == "dirty checkout":
        (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    elif case == "stash drop":
        (repo / "tracked.txt").write_text("stashed\n", encoding="utf-8")
        git(repo, "stash")
    elif case == "push --force":
        # A local bare clone as origin: no network, a configured remote.
        remote = tmp_path / "remote.git"
        git(tmp_path, "clone", "-q", "--bare", str(repo), str(remote))
        git(repo, "remote", "add", "origin", str(remote))
    return repo


def _untouched_state(repo: Path, case: str) -> str | None:
    """The state one denial case must leave untouched, captured before the
    denial is asserted; None for the cases with no observable state.
    """
    if case == "update-ref":
        return git(repo, "rev-parse", "refs/heads/topic").stdout.strip()
    if case == "reset --hard":
        return git(repo, "rev-parse", "HEAD").stdout.strip()
    if case == "stash drop":
        return git(repo, "stash", "list").stdout
    return None


@pytest.mark.parametrize(
    ("case", "command", "operations"),
    [
        ("dirty checkout", "git checkout topic", ("git.checkout",)),
        ("switch -f", "git switch -f topic", ("git.switch.discard",)),
        ("stash drop", "git stash drop", ("git.stash.drop",)),
        # Hard guards: no grantable proposal exists at all.
        ("reset --hard", "git reset --hard", None),
        ("update-ref", "git update-ref refs/heads/topic HEAD~1", None),
        # The --ours form is only meaningful mid-merge; out of one it is
        # still a gated checkout, never an exempt shape.
        (
            "out-of-merge checkout --ours",
            "git checkout --ours topic",
            ("git.checkout",),
        ),
        ("push --force", "git push --force origin main", ("git.push.force",)),
    ],
)
def test_denied_destructive_forms_are_refused_before_execution(
    tmp_path: Path, case: str, command: str, operations: tuple[str, ...] | None
) -> None:
    # The denial is asserted at the permission boundary — the same context the
    # executor consults BEFORE spawning — and the suite never executes a denied
    # destructive command; the untouched-state assertions below double as
    # evidence that the refusal preceded any execution.
    repo = _denial_case_repo(tmp_path, case)
    untouched = _untouched_state(repo, case)
    context = resolve(repo, command)
    assert isinstance(context, PermissionContext), command
    assert context.permission is ToolPermission.NEVER, command
    if operations is None:
        assert context.denial_kind is None, command
        assert context.dangerous_command_proposal is None, command
    else:
        assert context.denial_kind == "dangerous_command", command
        proposal = context.dangerous_command_proposal
        assert proposal is not None, command
        assert proposal.operations == operations, command
    if untouched is not None:
        assert _untouched_state(repo, case) == untouched


@pytest.mark.asyncio
async def test_conflicting_concurrent_change_after_the_probe_refuses_and_is_preserved(
    tmp_path: Path,
) -> None:
    # The permission-time state probe is not a transaction: a concurrent actor
    # can change the tree after the probe and before execution. The adjudicated
    # invariant is PRESERVATION, not unconditional failure — a CONFLICTING
    # concurrent change must cause a refusal (git refuses to overwrite
    # uncommitted content), and the change must survive it.
    repo = make_repo(tmp_path / "repo")
    command = "git checkout topic"
    tool = bash_tool(repo)
    assert_allowed(tool.resolve_permission(BashArgs(command=command)), command)
    # The conflicting change lands AFTER the permission probe: the file named
    # `topic` differs between main and topic, so checking out topic would
    # overwrite its uncommitted modification.
    (repo / "topic").write_text("concurrent edit\n", encoding="utf-8")
    with pytest.raises(FailedShellCommand):
        await collect_result(tool.run(BashArgs(command=command)))
    # The concurrent change survived and the checkout never happened.
    assert (repo / "topic").read_text(encoding="utf-8") == "concurrent edit\n"
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"


@pytest.mark.asyncio
async def test_non_conflicting_concurrent_change_survives_the_checkout(
    tmp_path: Path,
) -> None:
    # The other adjudicated outcome: a NON-conflicting concurrent change may
    # survive a successful non-forcing checkout — git never clobbers
    # untracked content it does not have to.
    repo = make_repo(tmp_path / "repo")
    command = "git checkout topic"
    tool = bash_tool(repo)
    assert_allowed(tool.resolve_permission(BashArgs(command=command)), command)
    (repo / "concurrent.txt").write_text("user work\n", encoding="utf-8")
    await collect_result(tool.run(BashArgs(command=command)))
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "topic"
    assert (repo / "concurrent.txt").read_text(encoding="utf-8") == "user work\n"


@pytest.mark.asyncio
async def test_ignored_collision_created_after_the_probe_refuses(
    tmp_path: Path,
) -> None:
    # The ignored-collision RACE case: the ignored file appears after the
    # permission probe, in the window the probe cannot see — the tree reads
    # clean at both probe times, so only the executed
    # --no-overwrite-ignore string can prevent the silent overwrite. The
    # collision must refuse, never clobber.
    repo = ignored_collision_repo(tmp_path / "repo", collision=False)
    command = "git checkout topic"
    tool = bash_tool(repo)
    assert_allowed(tool.resolve_permission(BashArgs(command=command)), command)
    (repo / "build.txt").write_text("local precious content\n", encoding="utf-8")
    with pytest.raises(FailedShellCommand):
        await collect_result(tool.run(BashArgs(command=command)))
    assert (repo / "build.txt").read_text(
        encoding="utf-8"
    ) == "local precious content\n"
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"


def reviewer_manager(cwd: Path) -> ToolManager:
    """A reviewer seat's tool manager: bash-only (B2 unchanged), cwd at the
    repository.
    """
    resolved = build_test_vibe_config(agent={"identity": "reviewer"})
    return ToolManager(
        lambda: resolved,
        cwd=cwd,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
    )


@pytest.mark.asyncio
async def test_reviewer_seat_composes_the_exemption_with_the_injected_flag(
    tmp_path: Path,
) -> None:
    # The reviewer-context composition: a reviewer seat (bash-only, unchanged
    # by B2) performing an exempt checkout is allowed at the permission
    # boundary AND the executed string carries the injected flag — on the
    # ignored-collision repo the boundary allows the checkout while the run
    # still refuses the collision, so the ignored file is never silently
    # overwritten.
    repo = ignored_collision_repo(tmp_path / "repo", collision=False)
    manager = reviewer_manager(repo)
    assert "bash" in manager.available_tools
    assert "edit" not in manager.available_tools
    tool = manager.get("bash")
    command = "git checkout topic"
    assert_allowed(tool.resolve_permission(BashArgs(command=command)), command)
    # The ignored file appears after the permission probe, in the window the
    # probe cannot see.
    (repo / "build.txt").write_text("local precious content\n", encoding="utf-8")
    with pytest.raises(FailedShellCommand):
        await collect_result(tool.run(BashArgs(command=command)))
    assert (repo / "build.txt").read_text(
        encoding="utf-8"
    ) == "local precious content\n"
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"


def test_the_guardrail_cache_has_no_cross_call_lifetime(tmp_path: Path) -> None:
    # Pickup clause (A4): A2 pinned the per-call lifetime; the cross-call
    # semantics land here. The cache is created per resolution and never
    # outlives it — in NEITHER direction: a state change between calls is
    # re-read every time (no positive cross-call caching), and a restored
    # state re-earns the exemption (no negative cross-call caching either).
    repo = make_repo(tmp_path / "repo")
    tool = bash_tool(repo)
    command = "git checkout topic"
    assert_allowed(tool.resolve_permission(BashArgs(command=command)), command)
    (repo / "tracked.txt").write_text("dirtied\n", encoding="utf-8")
    assert_gated(tool.resolve_permission(BashArgs(command=command)), command)
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    assert_allowed(tool.resolve_permission(BashArgs(command=command)), command)


def test_exempt_scope_resolution_from_a_subdirectory(tmp_path: Path) -> None:
    # Pickup clause (A4): the scope-permutation dimension. The reachable
    # slice: the single candidate scope is resolved from wherever the tool
    # runs — a subdirectory of the repo resolves to the same repository
    # scope, and the exemption holds there too. What is NOT reachable under
    # the whole-call gate is a true multi-scope permutation: a single-part
    # command has exactly one candidate scope (only a `cd` can add candidate
    # directories, and any second statement breaks whole-call eligibility),
    # so the per-scope loop and its per-scope cache stay defensive for this
    # shape — documented here and in the resolver, not forced through a
    # synthetic call.
    repo = make_repo(tmp_path / "repo")
    sub = repo / "sub"
    for command in ("git checkout topic", "git checkout -b feature"):
        tool = bash_tool(sub)
        # The real session composition: the session started at the repo root,
        # so the authorized workspace root is the repository while the shell
        # cwd sits in a subdirectory.
        tool.workspace = Workspace.for_session(repo)
        assert_allowed(tool.resolve_permission(BashArgs(command=command)), command)
