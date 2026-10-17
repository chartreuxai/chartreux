"""End-to-end characterization of dangerous git command reprompt behavior.

These tests drive a real Chartreux TUI process against the streaming mock
server: a real interactive session registers, the model's dangerous git call
opens the real "Dangerous command" question in the TUI, the user's choice is
typed at the real question widget, and the original call resumes and executes
(or is refused) in an isolated local repository whose only remote is a local
bare repository. No real remote is contacted and no destructive git operation
is ever executed.

The file is leaned (stage 12d): every other scenario of the original
characterization file is name-mirrored by acceptance-layer twins (see
tests/e2e/SURVIVAL_MAP.md) and was retired; the two retained scenarios pin the
PTY-only observation of the design §6.11 — reprompt suppression: the question
widget renders exactly once for a denied call (refusal memory suppresses the
re-prompt) and renders again for the identical call after a once-approval was
consumed by exactly one invocation.

The question widget swallows keystrokes during its input grace period and
repaints its title several times per mount, so answers are sent on a retry
cadence until the turn's final text renders: keys typed before the question
mounts or inside its grace period are swallowed or land harmlessly in the
chat input, and the first key past the grace period answers the question.
"""

from __future__ import annotations

from collections.abc import Callable
import io
from pathlib import Path
import subprocess
import time

import pexpect
import pytest

from tests.e2e.agent_loop_characterization.support import (
    assert_tool_result_contains,
    assistant_text_chunks,
    single_tool_call_chunks,
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


def _bash_turn(command: str, call_id: str, created: int = 90):
    return single_tool_call_chunks(
        call_id=call_id,
        tool_name="bash",
        arguments={"command": command},
        created=created,
    )


# --- Retained scenario 1: deny suppresses the re-prompt ---


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


# --- Retained scenario 2: a consumed once-approval re-prompts ---


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
