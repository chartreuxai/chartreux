"""A1 acceptance pins: checkout/switch work-tree mutation-roots footprints.

A checkout/switch invocation's mutation scope is the work tree it runs in, not
the individual paths it names. The resolver collection therefore produces a
distinct mutation-roots footprint — the resolved work-tree root — alongside the
write-classified paths, and every transport of the write footprint carries the
mutation-roots footprint too: the bash result (and its failed, timed-out, and
cancelled error carriers), the tool executor's error-event copy, and the
bash_start managed-start result (with its failed and cancelled starts).

The footprint is a mutation-scope/PERMISSION footprint only: the submission
secret scan must never walk it (a work-tree root would scan the whole
repository and burn the bounded walk's budget, blocking publication), while
the permission write-footprint collection and the completion audit's
touched-path comparisons still consume it. The regression here exercises the
ACTUAL shell-result-to-accumulator path — a real shell result produced by the
real Bash tool, carried through the accumulator's observe collection — for the
success, failure, cancellation, and managed-job (background bash) paths, and
the publication regression in tests/core/test_subagent_submission.py proves a
completed run whose chronology includes an eligible checkout produces no
blocking incomplete-scan finding.

Coverage: subdirectory execution, chained -C, linked worktrees, explicit
relocation, multiple candidate scopes, and unresolvable state — which reads
as an unknown footprint, never as a partial success. The deep-review repair
lands its regression here too: a checkout/switch mutation root feeds the
permission-side recovery/destruction analysis (an earlier checkout/switch
fails a later recursive-rm segment closed), while the submission scan's
path list stays free of it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
from typing import Any

from pydantic import BaseModel
import pytest

from chartreux.core.background_jobs import BackgroundJobRegistry, BashStartArgs
from chartreux.core.events import AssistantEvent, BaseEvent, ToolResultEvent
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.subagents import (
    UNKNOWN_MUTATION_ROOT_FOOTPRINT,
    SubagentRunAccumulator,
    mutation_roots_from_result_event,
)
from chartreux.core.tools.base import BaseToolState, InvokeContext, ToolPermission
from chartreux.core.tools.builtins._shell_permission_resolver import (
    shell_command_mutation_roots,
    shell_command_write_paths,
)
from chartreux.core.tools.builtins.bash import (
    Bash,
    BashArgs,
    BashToolConfig,
    CancelledShellCommand,
    CapturedShellResult,
    FailedShellCommand,
)
from chartreux.core.tools.builtins.bash_start import BashStart, FailedJobStart
from chartreux.core.tools.builtins.read_file import ReadFile
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.utils import DEFAULT_SENSITIVE_PATTERNS, PathAuthority
from chartreux.core.workspace import Workspace
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import collect_result, mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_tool import FakeTool


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run one git command in a fixture repository."""
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=check, capture_output=True, text=True
    )


def make_repo(path: Path) -> Path:
    """A git repository with one commit on ``main``."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "pins@example.com")
    git(path, "config", "user.name", "Pins")
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")
    return path


def mutation_roots(command: str, cwd: Path) -> tuple[list[str], bool]:
    """The mutation-roots footprint one command collects in a scratch workspace."""
    workspace = Workspace.for_session(cwd)
    authority = PathAuthority(
        "bash",
        BashToolConfig().permission,
        (),
        (),
        tuple(DEFAULT_SENSITIVE_PATTERNS),
        workspace,
    )
    return shell_command_mutation_roots(
        command, cwd=cwd, workspace=workspace, authority=authority
    )


# --- the resolver collection ---------------------------------------------------


def test_checkout_from_the_repo_root_resolves_the_work_tree_root(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path / "repo")
    for command in ("git checkout main", "git switch main"):
        roots, unknown = mutation_roots(command, repo)
        assert roots == [str(repo)]
        assert unknown is False


def test_checkout_from_a_subdirectory_resolves_the_repository_root(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path / "repo")
    sub = repo / "sub"
    sub.mkdir()
    roots, unknown = mutation_roots("git checkout main", sub)
    # The mutation scope is the work tree the checkout mutates — the whole
    # repository — not the subdirectory the command runs from.
    assert roots == [str(repo)]
    assert unknown is False


def test_file_checkout_form_resolves_the_work_tree_root(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    roots, unknown = mutation_roots("git checkout -- tracked.txt", repo)
    # A pathspec checkout writes work-tree files in bulk either way: the work
    # tree is the mutation scope.
    assert roots == [str(repo)]
    assert unknown is False


def test_chained_C_relocations_resolve_the_final_work_tree(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    roots, unknown = mutation_roots(
        f"git -C {tmp_path} -C repo checkout main", tmp_path
    )
    assert roots == [str(repo)]
    assert unknown is False


def test_linked_worktree_resolves_its_own_work_tree(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    git(repo, "worktree", "add", "-q", "-b", "wt-branch", str(tmp_path / "wt"))
    roots, unknown = mutation_roots("git checkout main", tmp_path / "wt")
    # A sibling worktree is its own mutation scope: the main checkout's work
    # tree is not.
    assert roots == [str(tmp_path / "wt")]
    assert unknown is False


def test_explicit_relocation_resolves_the_relocated_work_tree(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    roots, unknown = mutation_roots(f"git --work-tree {relocated} checkout main", repo)
    assert roots == [str(relocated)]
    assert unknown is False


def test_multiple_candidate_scopes_collect_every_root(tmp_path: Path) -> None:
    # The command may run in either statically possible directory, so both
    # candidate scopes' work-tree roots are the footprint.
    first = make_repo(tmp_path / "first")
    second = make_repo(tmp_path / "second")
    roots, unknown = mutation_roots(f"cd {second} && git checkout main", first)
    # The candidate scopes come from a set: the footprint is the root set.
    assert sorted(roots) == sorted([str(first), str(second)])
    assert unknown is False


def test_unresolvable_repository_state_is_an_unknown_footprint(tmp_path: Path) -> None:
    # A .git file that is not a usable repository pointer: git itself would
    # refuse the call, and the work-tree root cannot be established — an
    # unknown footprint, never a partial one.
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / ".git").write_text("not a repository pointer\n", encoding="utf-8")
    assert mutation_roots("git checkout main", broken) == ([], True)


def test_git_prefix_environment_assignment_is_unknown(tmp_path: Path) -> None:
    # A GIT_-prefixed assignment relocates the invocation outside the modeled
    # resolution (GIT_DIR/GIT_WORK_TREE): the root it would mutate cannot be
    # established.
    repo = make_repo(tmp_path / "repo")
    assert mutation_roots(
        f"GIT_DIR={tmp_path / 'elsewhere'} git checkout main", repo
    ) == ([], True)


def test_unknown_cwd_is_an_unknown_footprint(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    # A cd the analysis cannot pin (an option it does not model) leaves every
    # later segment's cwd unknown; the checkout's mutation scope is unknown.
    roots, unknown = mutation_roots("cd --unmodeled sub && git checkout main", repo)
    assert roots == []
    assert unknown is True


def test_non_checkout_commands_collect_nothing(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    for command in (
        "git status",
        "git commit -am message",
        "git restore -- tracked.txt",
        "echo hi > other.txt",
    ):
        assert mutation_roots(command, repo) == ([], False)


def test_analysis_failure_and_eval_are_unknown(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    # An analysis-level failure is a hard denial the resolver never clears, so
    # the command never runs: the extraction refuses to certify a partial set.
    assert mutation_roots('git checkout "unterminated', repo) == ([], True)
    assert mutation_roots('eval "git checkout main"', repo) == ([], True)


# --- the permission-side footprint consumption (deep-review F4) -----------------


def make_checkout_rm_repo(path: Path) -> Path:
    """A clean repository with a tracked ``build/`` directory and a ``topic``
    branch: the baseline recursive-rm target plus an earlier checkout/switch
    mutation.
    """
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "pins@example.com")
    git(path, "config", "user.name", "Pins")
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (path / "build").mkdir()
    (path / "build" / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")
    git(path, "checkout", "-q", "-b", "topic")
    git(path, "checkout", "-q", "main")
    return path


def resolve_permission(cwd: Path, command: str) -> PermissionContext | None:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=cwd)
    return tool.resolve_permission(BashArgs(command=command))


def write_paths(command: str, cwd: Path) -> tuple[list[str], bool]:
    """The write-classified-path footprint one command collects in a scratch
    workspace (the submission scan's path list).
    """
    workspace = Workspace.for_session(cwd)
    authority = PathAuthority(
        "bash",
        BashToolConfig().permission,
        (),
        (),
        tuple(DEFAULT_SENSITIVE_PATTERNS),
        workspace,
    )
    return shell_command_write_paths(
        command, cwd=cwd, workspace=workspace, authority=authority
    )


@pytest.mark.parametrize("mutation", ["git checkout topic", "git switch topic"])
def test_a_checkout_mutation_root_feeds_the_permission_side_footprint(
    tmp_path: Path, mutation: str
) -> None:
    # Deep-review F4: a checkout/switch invocation's mutation scope — its
    # resolved work-tree root — is a segment write the permission-side
    # recovery/destruction analysis accounts, exactly like any other earlier
    # statement's write. A composed `<checkout>; rm -rf build` mutates the
    # whole work tree before the deletion, dirtying the clean-tree state
    # the rm segment's run-class eligibility relies on, so the composed
    # call resolves handoff while the rm alone stays run-class.
    repo = make_checkout_rm_repo(tmp_path / "repo")
    command = f"{mutation}; rm -rf build"
    # The baseline: the recursive rm alone resolves run-class...
    context = resolve_permission(repo, "rm -rf build")
    assert context is not None
    assert context.permission is ToolPermission.ALWAYS
    # ...and a read-only earlier statement keeps it run-class.
    context = resolve_permission(repo, "git status; rm -rf build")
    assert context is not None
    assert context.permission is ToolPermission.ALWAYS
    # The composed call fails closed: the earlier checkout/switch's
    # work-tree root is a prior write inside the rm segment's eligibility
    # scope.
    context = resolve_permission(repo, command)
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "recursive rm" in (context.reason or "")

    # The mutation root is collected as the mutation-roots footprint...
    roots, unknown = mutation_roots(command, repo)
    assert roots == [str(repo)]
    assert unknown is False
    # ...but the submission scan's path list never sees it: the write-path
    # extraction stays free of the work-tree root.
    paths, paths_unknown = write_paths(command, repo)
    assert paths_unknown is False
    assert str(repo) not in paths


def test_an_unresolvable_checkout_root_fails_the_rm_segment_closed(
    tmp_path: Path,
) -> None:
    # Deep-review F4 (unknown propagation): a checkout whose work-tree root
    # cannot be established (a GIT_-prefixed assignment relocates the
    # invocation outside the modeled resolution) leaves the segment's write
    # surface unknown, and the later recursive rm fails closed on it — an
    # unknown footprint, never a partial one.
    repo = make_checkout_rm_repo(tmp_path / "repo")
    command = f"GIT_DIR={tmp_path / 'elsewhere'} git checkout main; rm -rf build"
    context = resolve_permission(repo, command)
    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "recursive rm" in (context.reason or "")


def test_mutation_roots_fail_closed_on_malformed_results() -> None:
    # The same fail-closed shape contract as the write footprint: a result
    # carrying the fields without well-formed values reads as unknown, never
    # as an established empty one; a non-shell result without the attributes
    # carries no footprint.

    class _NonListResult(BaseModel):
        mutation_roots: str = "not-a-list"

    class _PlainResult(BaseModel):
        message: str = "x"

    def event(result: BaseModel | None, tool_class: type | None) -> ToolResultEvent:
        return ToolResultEvent(
            tool_name="bash", tool_class=tool_class, result=result, tool_call_id="c"
        )

    assert mutation_roots_from_result_event(event(_NonListResult(), Bash)) == ([], True)
    # A result-less event carries the executor-copied error fields, which
    # default to an established empty footprint — an error raised before the
    # command ran carries none.
    assert mutation_roots_from_result_event(event(None, Bash)) == ([], False)
    assert mutation_roots_from_result_event(event(_PlainResult(), Bash)) == ([], True)
    assert mutation_roots_from_result_event(event(_PlainResult(), ReadFile)) == (
        [],
        False,
    )
    assert mutation_roots_from_result_event(
        event(_PlainResult(), Bash).model_copy(update={"skipped": True})
    ) == ([], False)


# --- the bash tool's result and error carriers ---------------------------------


@pytest.mark.asyncio
async def test_successful_checkout_result_carries_the_mutation_roots(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path / "repo")
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo)

    result = await collect_result(tool.run(BashArgs(command="git checkout -b side")))

    assert result.mutation_roots == [str(repo)]
    assert result.mutation_roots_unknown is False
    # The projected payload keeps its shape (the footprint is in-process
    # accounting, not wire output).
    assert "mutation_roots" not in result.model_dump(mode="json")


@pytest.mark.asyncio
async def test_failed_checkout_carries_the_mutation_roots(tmp_path: Path) -> None:
    # The footprint is computed before execution: a checkout that git refuses
    # (unknown branch) still carries the resolved work-tree root it would have
    # mutated, exactly as a successful run's would be.
    repo = make_repo(tmp_path / "repo")
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo)

    with pytest.raises(FailedShellCommand) as err:
        await collect_result(tool.run(BashArgs(command="git checkout no-such-branch")))

    assert err.value.mutation_roots == [str(repo)]
    assert err.value.mutation_roots_unknown is False


@pytest.mark.asyncio
async def test_timed_out_checkout_carries_the_mutation_roots(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo)

    with pytest.raises(FailedShellCommand) as err:
        await collect_result(
            tool.run(BashArgs(command="git checkout -b side; sleep 30", timeout=1))
        )

    assert err.value.mutation_roots == [str(repo)]
    assert err.value.mutation_roots_unknown is False


@pytest.mark.asyncio
async def test_cancelled_checkout_carries_the_mutation_roots(tmp_path: Path) -> None:
    # Observed inside the task, as the executor sees it: a task-cancelled
    # run's exception is replaced by a fresh instance at the task boundary.
    repo = make_repo(tmp_path / "repo")
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo)
    captured: dict[str, BaseException] = {}

    async def run_and_capture():
        try:
            await collect_result(
                tool.run(BashArgs(command="git checkout -b cancelled; sleep 30"))
            )
        except asyncio.CancelledError as exc:
            captured["cancellation"] = exc
            raise

    task = asyncio.ensure_future(run_and_capture())
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    cancellation = captured["cancellation"]
    assert isinstance(cancellation, CancelledShellCommand)
    assert cancellation.mutation_roots == [str(repo)]
    assert cancellation.mutation_roots_unknown is False


# --- the tool executor's error-event copy ---------------------------------------


def _tool_call(name: str, arguments: str) -> ToolCall:
    return ToolCall(
        id="call-1", index=0, function=FunctionCall(name=name, arguments=arguments)
    )


async def _error_event_for_raised_exception(
    exception: BaseException, *, cancelled: bool
) -> ToolResultEvent:
    """The real ToolResultEvent the tool executor produces for one raised error.

    A real agent loop runs the tool call; the stub raises the REAL exception a
    real bash run produced (captured above), so the executor's error-event
    footprint copy runs exactly as it does in production.
    """
    config = build_test_vibe_config(enabled_tools=["stub_tool"])
    agent_loop = build_test_agent_loop(
        config=config,
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="Let me use the tool.",
                    tool_calls=[_tool_call("stub_tool", "{}")],
                )
            ],
            [mock_llm_chunk(content="Done.")],
        ]),
    )
    agent_loop.tool_manager._all_tools["stub_tool"] = FakeTool
    stub = agent_loop.tool_manager.get("stub_tool")
    assert isinstance(stub, FakeTool)
    stub._exception_to_raise = exception

    events: list[BaseEvent] = []
    async for event in agent_loop.act("Execute tool"):
        events.append(event)
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert result.cancelled is cancelled
    return result


@pytest.mark.asyncio
async def test_executor_copies_the_mutation_roots_onto_error_events(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path / "repo")
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo)
    with pytest.raises(FailedShellCommand) as err:
        await collect_result(tool.run(BashArgs(command="git checkout no-such-branch")))
    failure = err.value

    event = await _error_event_for_raised_exception(failure, cancelled=False)

    assert event.error_mutation_roots == [str(repo)]
    assert event.error_mutation_roots_unknown is False


@pytest.mark.asyncio
async def test_executor_copies_the_mutation_roots_onto_cancelled_events(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path / "repo")
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo)
    captured: dict[str, BaseException] = {}

    async def run_and_capture():
        try:
            await collect_result(
                tool.run(BashArgs(command="git checkout -b cancelled; sleep 30"))
            )
        except asyncio.CancelledError as exc:
            captured["cancellation"] = exc
            raise

    task = asyncio.ensure_future(run_and_capture())
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    cancellation = captured["cancellation"]
    assert isinstance(cancellation, CancelledShellCommand)

    event = await _error_event_for_raised_exception(cancellation, cancelled=True)

    assert event.error_mutation_roots == [str(repo)]
    assert event.error_mutation_roots_unknown is False


# --- the managed-job (background bash) path -------------------------------------


def _manager(tmp_path: Path) -> ToolManager:
    return ToolManager(lambda: build_test_vibe_config(), cwd=tmp_path, defer_mcp=True)


async def _run(tool: Any, args: Any, ctx: InvokeContext | None = None) -> Any:
    return [result async for result in tool.run(args, ctx)][0]


@pytest.mark.asyncio
async def test_managed_start_carries_the_mutation_roots(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    registry = BackgroundJobRegistry()
    try:
        ctx = InvokeContext(tool_call_id="start", background_jobs=registry.root_port())
        result = await _run(
            _manager(repo).get("bash_start"),
            BashStartArgs(command="git checkout -b side"),
            ctx,
        )
        assert result.mutation_roots == [str(repo)]
        assert result.mutation_roots_unknown is False
        assert "mutation_roots" not in result.model_dump(mode="json")
        # The write footprint rides alongside, unchanged.
        assert result.write_paths == []

        roots, unknown = mutation_roots_from_result_event(
            ToolResultEvent(
                tool_name="bash_start",
                tool_class=BashStart,
                result=result,
                tool_call_id="start",
            )
        )
        assert roots == [str(repo)]
        assert unknown is False
    finally:
        await registry.aclose()


@pytest.mark.asyncio
async def test_failed_managed_start_carries_the_mutation_roots(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    registry = BackgroundJobRegistry()
    port = registry.root_port()
    await registry.aclose()
    tool = _manager(repo).get("bash_start")
    ctx = InvokeContext(tool_call_id="start", background_jobs=port)
    with pytest.raises(FailedJobStart) as excinfo:
        await _run(tool, BashStartArgs(command="git checkout -b side"), ctx)
    assert excinfo.value.mutation_roots == [str(repo)]
    assert excinfo.value.mutation_roots_unknown is False


# --- the shell-result-to-accumulator path (the regression core) -----------------


async def _accumulator_with_real_checkout(
    tmp_path: Path, command: str
) -> tuple[SubagentRunAccumulator, Path, ToolResultEvent]:
    """A real bash checkout's result, observed by a real accumulator.

    The repo carries an artifact larger than the per-file scan cap, so IF the
    work-tree root entered the submission scan's path list, the recursive walk
    would blow the budget and block the submission — the assertion that the
    scan excludes the mutation-roots footprint is behavioral, not structural.
    """
    from chartreux.core.tools.secret_scan import DEFAULT_MAX_SCAN_BYTES

    repo = make_repo(tmp_path / "repo")
    (repo / "big-artifact.bin").write_bytes(b"x" * (DEFAULT_MAX_SCAN_BYTES + 1))
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo)
    result = await collect_result(tool.run(BashArgs(command=command)))

    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="checked out"), tool_call_id="task")
    event = ToolResultEvent(
        tool_name="bash", tool_class=Bash, result=result, tool_call_id="bash-1"
    )
    accumulator.observe(event, tool_call_id="task")
    return accumulator, repo, event


@pytest.mark.asyncio
async def test_real_checkout_result_flows_into_the_accumulator_mutation_roots(
    tmp_path: Path,
) -> None:
    accumulator, repo, _event = await _accumulator_with_real_checkout(
        tmp_path, "git checkout -b side"
    )

    # The mutation-roots list is present in the collected metadata...
    assert accumulator.mutation_roots == [str(repo)]
    # ...and the work-tree root NEVER entered the submission scan's path list.
    assert str(repo) not in accumulator._touched_paths

    result = accumulator.build_result(turns_used=1)
    # The submission passes: the recursive scan did NOT walk the work-tree
    # root (the oversized artifact would have produced a blocking
    # incomplete-scan finding had it done so).
    assert result.completed is True
    assert result.response == "checked out"


@pytest.mark.asyncio
async def test_real_failed_checkout_flows_into_the_accumulator_mutation_roots(
    tmp_path: Path,
) -> None:
    from chartreux.core.tools.secret_scan import DEFAULT_MAX_SCAN_BYTES

    repo = make_repo(tmp_path / "repo")
    (repo / "big-artifact.bin").write_bytes(b"x" * (DEFAULT_MAX_SCAN_BYTES + 1))
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo)
    with pytest.raises(FailedShellCommand) as err:
        await collect_result(tool.run(BashArgs(command="git checkout no-such-branch")))
    failure = err.value

    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="attempted"), tool_call_id="task")
    accumulator.observe(
        ToolResultEvent(
            tool_name="bash",
            tool_class=Bash,
            # The exact copy the tool executor's failure handler performs.
            error_mutation_roots=getattr(failure, "mutation_roots", None) or [],
            error_mutation_roots_unknown=bool(
                getattr(failure, "mutation_roots_unknown", False)
            ),
            tool_call_id="bash-1",
        ),
        tool_call_id="task",
    )

    # The failed run's footprint arrived on the error event (no result model)
    # and was collected all the same.
    assert accumulator.mutation_roots == [str(repo)]
    assert str(repo) not in accumulator._touched_paths
    assert accumulator.build_result(turns_used=1).completed is True


@pytest.mark.asyncio
async def test_real_cancelled_checkout_flows_into_the_accumulator_mutation_roots(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path / "repo")
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo)
    captured: dict[str, BaseException] = {}

    async def run_and_capture():
        try:
            await collect_result(
                tool.run(BashArgs(command="git checkout -b cancelled; sleep 30"))
            )
        except asyncio.CancelledError as exc:
            captured["cancellation"] = exc
            raise

    task = asyncio.ensure_future(run_and_capture())
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    cancellation = captured["cancellation"]
    assert isinstance(cancellation, CancelledShellCommand)

    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="attempted"), tool_call_id="task")
    accumulator.observe(
        ToolResultEvent(
            tool_name="bash",
            tool_class=Bash,
            error="interrupted",
            cancelled=True,
            # The exact copy the tool executor's cancellation handler performs.
            error_write_paths=getattr(cancellation, "write_paths", None) or [],
            error_write_paths_unknown=bool(
                getattr(cancellation, "write_paths_unknown", False)
            ),
            error_mutation_roots=getattr(cancellation, "mutation_roots", None) or [],
            error_mutation_roots_unknown=bool(
                getattr(cancellation, "mutation_roots_unknown", False)
            ),
            tool_call_id="bash-1",
        ),
        tool_call_id="task",
    )

    assert accumulator.mutation_roots == [str(repo)]
    assert str(repo) not in accumulator._touched_paths
    assert accumulator.build_result(turns_used=1).completed is True


def test_unresolvable_checkout_records_the_unknown_mutation_root_sentinel() -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="done"), tool_call_id="task")
    accumulator.observe(
        ToolResultEvent(
            tool_name="bash",
            tool_class=Bash,
            result=CapturedShellResult(
                command="git checkout main",
                mutation_roots=[],
                mutation_roots_unknown=True,
            ),
            tool_call_id="bash-1",
        ),
        tool_call_id="task",
    )
    assert accumulator.mutation_roots == [UNKNOWN_MUTATION_ROOT_FOOTPRINT]
    # The sentinel is mutation-scope accounting only: the scan's path list
    # never sees it.
    assert accumulator._touched_paths == {}
    assert accumulator.build_result(turns_used=1).completed is True
