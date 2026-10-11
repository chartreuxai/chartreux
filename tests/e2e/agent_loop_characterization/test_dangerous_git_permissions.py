"""End-to-end characterization of dangerous git command permissions.

These tests drive a real Chartreux TUI process against the streaming mock
server: a real interactive session registers, the model's dangerous git call
opens the real "Dangerous command" question in the TUI, the user's choice is
typed at the real question widget, and the original call resumes and executes
(or is refused) in an isolated local repository whose only remote is a local
bare repository. No real remote is contacted and no destructive git operation
is ever executed.

The scenarios pin the composed user-visible system: every approval choice
resolving through the real question (deny with refusal memory, allow once
with re-prompt, allow this session, always for this project with the
config-file write), the explicit denylist veto, hard guards, other denials
winning over the question, invocation-specific once approvals for concurrent
identical calls, saved project authority not following a sibling worktree or
another repository, the parent ceiling for subagent calls, session-only
grants not surviving a resumed session, and unchanged routine policy.

The question widget swallows keystrokes during its input grace period and
repaints its title several times per mount, so answers are sent on a retry
cadence until the turn's final text renders: keys typed before the question
mounts or inside its grace period are swallowed or land harmlessly in the
chat input, and the first key past the grace period answers the question.
"""

from __future__ import annotations

from collections.abc import Callable
import io
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib

import pexpect
import pytest
import tomli_w

from tests.e2e.agent_loop_characterization.support import (
    assert_tool_result_contains,
    assistant_text_chunks,
    multi_tool_call_chunks,
    set_tool_denylist,
    single_tool_call_chunks,
    wait_for_request_count_while_draining_child_output,
)
from tests.e2e.common import (
    SpawnedChartreuxProcessFixture,
    drain_child_output,
    send_ctrl_c_until_quit_confirmation,
    strip_ansi,
    wait_for_main_screen,
    wait_for_rendered_text,
)
from tests.e2e.mock_server import ChatCompletionsRequestPayload, StreamingMockServer

QUESTION_NEEDLE = "Allow this dangerous git command?"
DECLINED_FEEDBACK = "user declined; do not retry this command"
DENY_KEY = "1"
ALLOW_ONCE_KEY = "2"
ALLOW_SESSION_KEY = "3"
ALLOW_PROJECT_KEY = "4"
GRANTS_FIELD = "dangerous_command_grants_by_project"
ANSWER_RETRY_INTERVAL_S = 0.8


def _git(project: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(project), *args], check=True)


def _make_project(workdir: Path, remote: Path) -> None:
    """An isolated repository whose only remote is a local bare repository."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(workdir)], check=True)
    _git(workdir, "config", "user.email", "t@t")
    _git(workdir, "config", "user.name", "t")
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True
    )
    _git(workdir, "remote", "add", "origin", str(remote))
    _git(workdir, "commit", "--allow-empty", "-q", "-m", "init")


def _remote_has_main(remote: Path) -> bool:
    return (
        subprocess.run(
            ["git", "--git-dir", str(remote), "rev-parse", "--verify", "-q", "main"],
            check=False,
        ).returncode
        == 0
    )


def _commit_count(project: Path) -> int:
    output = subprocess.run(
        ["git", "-C", str(project), "log", "--oneline"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return len(output.strip().splitlines())


def _vibe_home() -> Path:
    return Path(os.environ["CHARTREUX_HOME"])


def _saved_grants() -> dict[str, list[str]]:
    config_path = _vibe_home() / "config.toml"
    if not config_path.exists():
        return {}
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    grants = data.get(GRANTS_FIELD, {})
    return {str(key): [str(item) for item in value] for key, value in grants.items()}


def _set_saved_grant(project: Path, operations: list[str]) -> None:
    config_path = _vibe_home() / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config[GRANTS_FIELD] = {str(project.resolve()): operations}
    config_path.write_bytes(tomli_w.dumps(config).encode())


def _question_count(captured: io.StringIO) -> int:
    return strip_ansi(captured.getvalue()).count(QUESTION_NEEDLE)


def _wait_for_question(
    child: pexpect.spawn, captured: io.StringIO, baseline: int, timeout: float = 15
) -> None:
    """Wait until one more dangerous-command question has rendered."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if _question_count(captured) > baseline:
            return
        drain_child_output(child, captured)
        time.sleep(0.02)
    raise AssertionError(
        f"Timed out waiting for a dangerous-command question after "
        f"{baseline} rendered title(s); count is {_question_count(captured)}"
    )


def _answer_until(
    child: pexpect.spawn,
    captured: io.StringIO,
    key: str,
    done: Callable[[], bool],
    *,
    timeout: float = 20,
) -> None:
    """Send the choice key until *done* reports the turn has moved on.

    Keys sent before the question mounts or inside its grace period are
    swallowed; keys sent after the question is answered land harmlessly in
    the chat input, so the retry cadence is safe.
    """
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        drain_child_output(child, captured)
        if done():
            return
        child.send(key)
        time.sleep(ANSWER_RETRY_INTERVAL_S)
    raise AssertionError("Timed out answering the dangerous-command question")


def _settle(child: pexpect.spawn, captured: io.StringIO) -> int:
    """Let the render settle and return the question-title baseline."""
    drain_child_output(child, captured)
    time.sleep(0.3)
    drain_child_output(child, captured)
    return _question_count(captured)


def _latest_session_id() -> str:
    session_root = _vibe_home() / "logs" / "session"
    metas = sorted(
        session_root.glob("session_*/meta.json"), key=lambda path: path.stat().st_mtime
    )
    assert metas, "no persisted session found"
    metadata = json.loads(metas[-1].read_text(encoding="utf-8"))
    session_id = metadata.get("session_id")
    assert isinstance(session_id, str)
    return session_id


def _bash_turn(command: str, call_id: str, created: int = 90):
    return single_tool_call_chunks(
        call_id=call_id,
        tool_name="bash",
        arguments={"command": command},
        created=created,
    )


# --- Scenario 1: every approval choice resolves through the real question ---


def _deny_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return _bash_turn("git push origin main; touch deny-marker", "deny-push-1")
    if request_index == 1:
        return assistant_text_chunks("Push denied.", created=100)
    if request_index == 2:
        return _bash_turn(
            "git push origin main; touch deny-marker", "deny-push-2", created=110
        )
    return assistant_text_chunks("Push refused again.", created=120)


@pytest.mark.timeout(75)
@pytest.mark.parametrize(
    "streaming_mock_server",
    [pytest.param(_deny_factory, id="deny-and-refusal-memory")],
    indirect=True,
)
def test_deny_skips_execution_and_suppresses_the_reprompt(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    tmp_path: Path,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedChartreuxProcessFixture,
) -> None:
    remote = tmp_path / "remote.git"
    _make_project(e2e_workdir, remote)
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Push the work")
        child.send("\r")
        _wait_for_question(child, captured, 0)
        _answer_until(
            child,
            captured,
            DENY_KEY,
            lambda: "Push denied." in strip_ansi(captured.getvalue()),
        )
        baseline = _settle(child, captured)
        # The same call again: refusal memory suppresses the re-prompt.
        child.send("Push the work again")
        child.send("\r")
        wait_for_rendered_text(
            child, captured, needle="Push refused again.", timeout=15
        )
        assert _settle(child, captured) == baseline
        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    assert not (e2e_workdir / "deny-marker").exists()
    assert not _remote_has_main(remote)
    assert_tool_result_contains(
        streaming_mock_server.requests[1],
        call_id="deny-push-1",
        expected=DECLINED_FEEDBACK,
    )
    assert_tool_result_contains(
        streaming_mock_server.requests[3],
        call_id="deny-push-2",
        expected=DECLINED_FEEDBACK,
    )


def _allow_once_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return _bash_turn("git push origin main; touch once-marker", "once-push-1")
    if request_index == 1:
        return assistant_text_chunks("Pushed once.", created=100)
    if request_index == 2:
        return _bash_turn(
            "git push origin main; touch once-marker", "once-push-2", created=110
        )
    return assistant_text_chunks("Second push denied.", created=120)


@pytest.mark.timeout(75)
@pytest.mark.parametrize(
    "streaming_mock_server",
    [pytest.param(_allow_once_factory, id="allow-once-then-reprompt")],
    indirect=True,
)
def test_allow_once_executes_exactly_one_call_and_reprompts(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    tmp_path: Path,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedChartreuxProcessFixture,
) -> None:
    remote = tmp_path / "remote.git"
    _make_project(e2e_workdir, remote)
    marker = e2e_workdir / "once-marker"
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Push the work")
        child.send("\r")
        _wait_for_question(child, captured, 0)
        _answer_until(
            child,
            captured,
            ALLOW_ONCE_KEY,
            lambda: "Pushed once." in strip_ansi(captured.getvalue()),
        )
        assert marker.exists()
        assert _remote_has_main(remote)
        baseline = _settle(child, captured)
        # The once permit authorized exactly this call: the same command
        # prompts again instead of resuming on the consumed approval.
        marker.unlink()
        child.send("Push the work again")
        child.send("\r")
        _wait_for_question(child, captured, baseline)
        _answer_until(
            child,
            captured,
            DENY_KEY,
            lambda: "Second push denied." in strip_ansi(captured.getvalue()),
        )
        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    assert not marker.exists()
    assert_tool_result_contains(
        streaming_mock_server.requests[3],
        call_id="once-push-2",
        expected=DECLINED_FEEDBACK,
    )


def _allow_session_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return _bash_turn(
            "git push origin main; touch session-marker", "session-push-1"
        )
    if request_index == 1:
        return assistant_text_chunks("Session push done.", created=100)
    if request_index == 2:
        return _bash_turn(
            "git push origin main; touch session-marker", "session-push-2", created=110
        )
    return assistant_text_chunks("Second push automatic.", created=120)


@pytest.mark.timeout(75)
@pytest.mark.parametrize(
    "streaming_mock_server",
    [pytest.param(_allow_session_factory, id="allow-this-session")],
    indirect=True,
)
def test_allow_this_session_skips_later_prompts(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    tmp_path: Path,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedChartreuxProcessFixture,
) -> None:
    remote = tmp_path / "remote.git"
    _make_project(e2e_workdir, remote)
    marker = e2e_workdir / "session-marker"
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Push the work")
        child.send("\r")
        _wait_for_question(child, captured, 0)
        _answer_until(
            child,
            captured,
            ALLOW_SESSION_KEY,
            lambda: "Session push done." in strip_ansi(captured.getvalue()),
        )
        assert marker.exists()
        baseline = _settle(child, captured)
        # The second identical call runs without prompting.
        marker.unlink()
        child.send("Push the work again")
        child.send("\r")
        wait_for_rendered_text(
            child, captured, needle="Second push automatic.", timeout=15
        )
        assert _settle(child, captured) == baseline
        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    assert marker.exists()
    assert _remote_has_main(remote)
    # A session grant is never persisted to the user config.
    assert _saved_grants() == {}


def _allow_project_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return _bash_turn("git push origin main; touch always-marker", "always-push-1")
    if request_index == 1:
        return assistant_text_chunks("Always push done.", created=100)
    if request_index == 2:
        return _bash_turn(
            "git push origin main; touch always-marker", "always-push-2", created=110
        )
    return assistant_text_chunks("Second push automatic.", created=120)


@pytest.mark.timeout(75)
@pytest.mark.parametrize(
    "streaming_mock_server",
    [pytest.param(_allow_project_factory, id="always-for-this-project")],
    indirect=True,
)
def test_always_for_project_persists_the_grant_in_the_user_config(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    tmp_path: Path,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedChartreuxProcessFixture,
) -> None:
    remote = tmp_path / "remote.git"
    _make_project(e2e_workdir, remote)
    marker = e2e_workdir / "always-marker"
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Push the work")
        child.send("\r")
        _wait_for_question(child, captured, 0)
        _answer_until(
            child,
            captured,
            ALLOW_PROJECT_KEY,
            lambda: "Always push done." in strip_ansi(captured.getvalue()),
        )
        assert marker.exists()
        baseline = _settle(child, captured)
        # The saved grant covers the second call without prompting.
        marker.unlink()
        child.send("Push the work again")
        child.send("\r")
        wait_for_rendered_text(
            child, captured, needle="Second push automatic.", timeout=15
        )
        assert _settle(child, captured) == baseline
        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    assert marker.exists()
    assert _remote_has_main(remote)
    # The user-layer config file gains the project entry.
    assert _saved_grants() == {str(e2e_workdir.resolve()): ["git.push"]}


# --- Scenario 10: project authority never follows another repository ---


def _scope_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return _bash_turn("git push origin main; touch main-marker", "main-push")
    if request_index == 1:
        return assistant_text_chunks("Main push done.", created=100)
    if request_index == 2:
        return _bash_turn(
            "git -C wt push origin main; touch wt-marker", "wt-push", created=110
        )
    if request_index == 3:
        return assistant_text_chunks("Worktree push denied.", created=120)
    if request_index == 4:
        return _bash_turn(
            "git -C nested push origin main; touch nested-marker",
            "nested-push",
            created=130,
        )
    return assistant_text_chunks("Nested push denied.", created=140)


@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    "streaming_mock_server",
    [pytest.param(_scope_factory, id="worktree-and-other-repo-scopes")],
    indirect=True,
)
def test_saved_project_grant_does_not_cover_a_worktree_or_other_repo(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    tmp_path: Path,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedChartreuxProcessFixture,
) -> None:
    remote = tmp_path / "remote.git"
    _make_project(e2e_workdir, remote)
    subprocess.run(
        ["git", "-C", str(e2e_workdir), "worktree", "add", "-q", "wt", "-b", "wt"],
        check=True,
    )
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(e2e_workdir / "nested")], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(e2e_workdir / "nested"),
            "remote",
            "add",
            "origin",
            str(remote),
        ],
        check=True,
    )
    _set_saved_grant(e2e_workdir, ["git.push"])
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        # The saved project grant covers the session's own repository.
        child.send("Push the work")
        child.send("\r")
        wait_for_rendered_text(child, captured, needle="Main push done.", timeout=15)
        assert (e2e_workdir / "main-marker").exists()
        assert _remote_has_main(remote)
        assert _settle(child, captured) == 0
        # A sibling worktree of the same repository prompts again.
        child.send("Push from the worktree")
        child.send("\r")
        _wait_for_question(child, captured, 0)
        _answer_until(
            child,
            captured,
            DENY_KEY,
            lambda: "Worktree push denied." in strip_ansi(captured.getvalue()),
        )
        baseline = _settle(child, captured)
        # Another repository inside the workspace prompts again.
        child.send("Push from the nested repo")
        child.send("\r")
        _wait_for_question(child, captured, baseline)
        _answer_until(
            child,
            captured,
            DENY_KEY,
            lambda: "Nested push denied." in strip_ansi(captured.getvalue()),
        )
        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    assert not (e2e_workdir / "wt-marker").exists()
    assert not (e2e_workdir / "nested-marker").exists()
    # The denials persisted nothing beyond the original project entry.
    assert _saved_grants() == {str(e2e_workdir.resolve()): ["git.push"]}


# --- Scenario 2: the explicit denylist vetoes even with a saved grant ---


def _veto_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return _bash_turn("git push origin main; touch veto-marker", "veto-push")
    return assistant_text_chunks("Push vetoed.", created=100)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "streaming_mock_server",
    [pytest.param(_veto_factory, id="denylist-veto-with-saved-grant")],
    indirect=True,
)
def test_explicit_denylist_vetoes_the_question_even_with_a_saved_grant(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    tmp_path: Path,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedChartreuxProcessFixture,
) -> None:
    remote = tmp_path / "remote.git"
    _make_project(e2e_workdir, remote)
    _set_saved_grant(e2e_workdir, ["git.push"])
    set_tool_denylist(_vibe_home(), "bash", ["git push"])
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Push the work")
        child.send("\r")
        wait_for_rendered_text(child, captured, needle="Push vetoed.", timeout=15)
        assert _settle(child, captured) == 0
        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    assert not (e2e_workdir / "veto-marker").exists()
    assert not _remote_has_main(remote)
    assert QUESTION_NEEDLE not in strip_ansi(captured.getvalue())
    assert_tool_result_contains(
        streaming_mock_server.requests[1],
        call_id="veto-push",
        expected="matches denylist pattern 'git push'",
    )


# --- Scenario 3: hard guards never prompt ---


def _reset_guard_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return _bash_turn("git reset --hard; touch guard-marker", "guard-call")
    return assistant_text_chunks("Guard held.", created=100)


def _clean_guard_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return _bash_turn("git clean -fd; touch guard-marker", "guard-call")
    return assistant_text_chunks("Guard held.", created=100)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "streaming_mock_server",
    [
        pytest.param(_reset_guard_factory, id="git-reset-hard"),
        pytest.param(_clean_guard_factory, id="git-clean-fd"),
    ],
    indirect=True,
)
def test_hard_guards_stay_denied_without_a_question(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    tmp_path: Path,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedChartreuxProcessFixture,
) -> None:
    remote = tmp_path / "remote.git"
    _make_project(e2e_workdir, remote)
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Run the guard")
        child.send("\r")
        wait_for_rendered_text(child, captured, needle="Guard held.", timeout=15)
        assert _settle(child, captured) == 0
        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    assert not (e2e_workdir / "guard-marker").exists()
    assert QUESTION_NEEDLE not in strip_ansi(captured.getvalue())
    assert _commit_count(e2e_workdir) == 1
    assert_tool_result_contains(
        streaming_mock_server.requests[1],
        call_id="guard-call",
        expected="Command denied",
    )


# --- Scenario 4: another denial wins over the dangerous question ---


def _sensitive_compound_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return _bash_turn("git push origin main; sudo true", "compound-push")
    return assistant_text_chunks("Compound denied.", created=100)


def _path_compound_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return _bash_turn("git push origin main; cat /etc/hostname", "compound-push")
    return assistant_text_chunks("Compound denied.", created=100)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("streaming_mock_server", "expected_denial"),
    [
        pytest.param(
            _sensitive_compound_factory,
            "Command denied by a sensitive command rule",
            id="sensitive-segment",
        ),
        pytest.param(
            _path_compound_factory,
            "Shell path is outside the authorized workspace",
            id="path-segment",
        ),
    ],
    indirect=["streaming_mock_server"],
)
def test_other_denial_wins_over_the_dangerous_question(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    tmp_path: Path,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedChartreuxProcessFixture,
    expected_denial: str,
) -> None:
    remote = tmp_path / "remote.git"
    _make_project(e2e_workdir, remote)
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Run the compound command")
        child.send("\r")
        wait_for_rendered_text(child, captured, needle="Compound denied.", timeout=15)
        assert _settle(child, captured) == 0
        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    assert not _remote_has_main(remote)
    assert QUESTION_NEEDLE not in strip_ansi(captured.getvalue())
    assert_tool_result_contains(
        streaming_mock_server.requests[1],
        call_id="compound-push",
        expected=expected_denial,
    )


# --- Scenario 5: concurrent identical once calls need two approvals ---


def _twin_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return multi_tool_call_chunks(
            [
                (
                    "twin-a",
                    "bash",
                    {"command": "git push origin main; touch twin-marker"},
                ),
                (
                    "twin-b",
                    "bash",
                    {"command": "git push origin main; touch twin-marker"},
                ),
            ],
            created=90,
        )
    return assistant_text_chunks("Both pushes settled.", created=100)


@pytest.mark.timeout(75)
@pytest.mark.parametrize(
    "streaming_mock_server",
    [pytest.param(_twin_factory, id="two-concurrent-once-calls")],
    indirect=True,
)
def test_two_concurrent_identical_once_calls_need_two_approvals(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    tmp_path: Path,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedChartreuxProcessFixture,
) -> None:
    remote = tmp_path / "remote.git"
    _make_project(e2e_workdir, remote)
    marker = e2e_workdir / "twin-marker"
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Push the work twice")
        child.send("\r")
        # Two identical concurrent calls: the first approval executes exactly
        # one invocation (the marker proves it), then the second call's own
        # question must be answered separately.
        _wait_for_question(child, captured, 0)
        _answer_until(child, captured, ALLOW_ONCE_KEY, lambda: marker.exists())
        assert marker.exists()
        assert _remote_has_main(remote)
        _answer_until(
            child,
            captured,
            DENY_KEY,
            lambda: "Both pushes settled." in strip_ansi(captured.getvalue()),
        )
        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    # Exactly one invocation executed: the other was refused, so a single
    # approval satisfied exactly one call.
    payload = streaming_mock_server.requests[1]
    messages = payload.get("messages")
    assert isinstance(messages, list)
    tool_messages = [
        message
        for message in messages
        if message.get("role") == "tool"
        and message.get("tool_call_id") in {"twin-a", "twin-b"}
    ]
    assert len(tool_messages) == 2
    declined = [
        message
        for message in tool_messages
        if DECLINED_FEEDBACK in str(message.get("content", ""))
    ]
    assert len(declined) == 1


# --- Scenario 12: routine git commands never prompt ---


def _routine_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return multi_tool_call_chunks(
            [
                ("routine-status", "bash", {"command": "git status"}),
                ("routine-log", "bash", {"command": "git log --oneline"}),
                ("routine-add", "bash", {"command": "git add seed.txt"}),
                ("routine-commit", "bash", {"command": "git commit -m routine"}),
            ],
            created=90,
        )
    return assistant_text_chunks("Routine commands done.", created=100)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "streaming_mock_server",
    [pytest.param(_routine_factory, id="routine-git-commands")],
    indirect=True,
)
def test_routine_git_commands_execute_without_prompts(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    tmp_path: Path,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedChartreuxProcessFixture,
) -> None:
    remote = tmp_path / "remote.git"
    _make_project(e2e_workdir, remote)
    seed = e2e_workdir / "seed.txt"
    seed.write_text("seed\n", encoding="utf-8")
    _git(e2e_workdir, "add", "seed.txt")
    _git(e2e_workdir, "commit", "-q", "-m", "seed")
    seed.write_text("seed v2\n", encoding="utf-8")
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Run the routine git commands")
        child.send("\r")
        wait_for_rendered_text(
            child, captured, needle="Routine commands done.", timeout=15
        )
        assert _settle(child, captured) == 0
        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    assert QUESTION_NEEDLE not in strip_ansi(captured.getvalue())
    assert _commit_count(e2e_workdir) == 3
    assert_tool_result_contains(
        streaming_mock_server.requests[1],
        call_id="routine-status",
        expected="On branch main",
    )


# --- Scenario 11: a subagent call hits the parent ceiling ---


def _ceiling_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return single_tool_call_chunks(
            call_id="ceiling-task",
            tool_name="task",
            arguments={
                "agent_type": "worker",
                "background": False,
                "task": "Publish the worktree.",
            },
            created=90,
        )
    if request_index == 1:
        return _bash_turn(
            "git push origin main; touch ceiling-marker", "ceiling-push", created=100
        )
    if request_index == 2:
        return assistant_text_chunks("Worker done.", created=110)
    return assistant_text_chunks("Parent concluded.", created=120)


@pytest.mark.timeout(75)
@pytest.mark.parametrize(
    "streaming_mock_server",
    [pytest.param(_ceiling_factory, id="subagent-parent-ceiling")],
    indirect=True,
)
def test_subagent_dangerous_call_hits_the_parent_ceiling(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    tmp_path: Path,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedChartreuxProcessFixture,
) -> None:
    remote = tmp_path / "remote.git"
    _make_project(e2e_workdir, remote)
    with (_vibe_home() / "config.toml").open("a", encoding="utf-8") as config:
        config.write('\n[tools.task]\nallowlist = ["worker"]\n')
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Delegate the publication")
        child.send("\r")
        wait_for_request_count_while_draining_child_output(
            child,
            captured,
            lambda: len(streaming_mock_server.requests),
            expected_count=4,
            timeout=20,
        )
        wait_for_rendered_text(child, captured, needle="Parent concluded.", timeout=15)
        assert _settle(child, captured) == 0
        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    assert not (e2e_workdir / "ceiling-marker").exists()
    assert not _remote_has_main(remote)
    assert QUESTION_NEEDLE not in strip_ansi(captured.getvalue())
    # The subagent's call was denied by the parent ceiling: the parent's
    # dangerous-authority gap is never a grantable local denial.
    assert_tool_result_contains(
        streaming_mock_server.requests[2],
        call_id="ceiling-push",
        expected="requires approval for a dangerous git operation",
    )


# --- Scenario 9: session-only grants do not survive a resumed session ---


def _resume_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return _bash_turn("git push origin main; touch resume-marker", "resume-push-1")
    if request_index == 1:
        return assistant_text_chunks("Session push done.", created=100)
    if request_index == 2:
        return _bash_turn(
            "git push origin main; touch resume-marker", "resume-push-2", created=110
        )
    if request_index == 3:
        return assistant_text_chunks("Second push automatic.", created=120)
    if request_index == 4:
        return _bash_turn(
            "git push origin main; touch resume-marker", "resume-push-3", created=130
        )
    return assistant_text_chunks("Resumed push denied.", created=140)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "streaming_mock_server",
    [pytest.param(_resume_factory, id="session-grant-not-restored-on-resume")],
    indirect=True,
)
def test_resumed_session_does_not_restore_session_only_grants(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    tmp_path: Path,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedChartreuxProcessFixture,
) -> None:
    remote = tmp_path / "remote.git"
    _make_project(e2e_workdir, remote)
    marker = e2e_workdir / "resume-marker"
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Push the work")
        child.send("\r")
        _wait_for_question(child, captured, 0)
        _answer_until(
            child,
            captured,
            ALLOW_SESSION_KEY,
            lambda: "Session push done." in strip_ansi(captured.getvalue()),
        )
        assert marker.exists()
        baseline = _settle(child, captured)
        # The session grant covers the second call without prompting.
        marker.unlink()
        child.send("Push the work again")
        child.send("\r")
        wait_for_rendered_text(
            child, captured, needle="Second push automatic.", timeout=15
        )
        assert _settle(child, captured) == baseline
        send_ctrl_c_until_quit_confirmation(child, captured, timeout=5)
        child.expect(pexpect.EOF, timeout=10)
    assert marker.exists()
    assert _remote_has_main(remote)
    # A session-only grant never reaches the user config.
    assert _saved_grants() == {}
    session_id = _latest_session_id()

    # A resumed session starts with no runtime authority: the same call
    # prompts again.
    marker.unlink()
    with spawned_vibe_process(e2e_workdir, extra_args=["--resume", session_id]) as (
        resumed_child,
        resumed_captured,
    ):
        wait_for_main_screen(resumed_child, timeout=15)
        resumed_child.send("Push the work once more")
        resumed_child.send("\r")
        _wait_for_question(resumed_child, resumed_captured, 0)
        _answer_until(
            resumed_child,
            resumed_captured,
            DENY_KEY,
            lambda: "Resumed push denied." in strip_ansi(resumed_captured.getvalue()),
        )
        send_ctrl_c_until_quit_confirmation(resumed_child, resumed_captured, timeout=5)
        resumed_child.expect(pexpect.EOF, timeout=10)
    assert not marker.exists()
    assert_tool_result_contains(
        streaming_mock_server.requests[5],
        call_id="resume-push-3",
        expected=DECLINED_FEEDBACK,
    )
