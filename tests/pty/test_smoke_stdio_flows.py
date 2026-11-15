"""The §6.5 flows that need a second process or a scripted provider.

This sibling of ``test_smoke_stdio.py`` owns the flows whose scenario needs
more than one spawned process or a scripted provider. ``test_smoke_stdio.py``
explains the state of the remaining §6.5 flows.

The flows that land here:

* **root-death recovery** (design §5, after-confirmed-root-death): killing the
  root must never fake success -- a submitted turn after the death refuses
  without reaching the provider and restores its draft -- and the durable
  recovery exists: a fresh ``--continue`` process restores the saved session,
  spawns a replacement root, and its next submission runs a real provider turn
  on the stdio topology.
* **transcript Load More** (§6.5): a session restored with more history than
  the resume tail (HISTORY_RESUME_TAIL_MESSAGES) reports the remaining older
  messages, and the transcript's head affordance renders with the exact
  remaining count after the shift+up scroll that mounts it. The affordance's
  click activation needs a mouse-driven terminal; from the PTY harness it is
  exercised at the in-process windowing layer (windowing/ tests), so this node
  pins the observable spawned rendering.
* **composed OSC deferral** (session-status design §5 criterion 1, the P2/R2
  leg): a real process-backed child runs through the spawned stdio topology
  while the main turn settles, and the terminal holds the exact OSC 7501
  ``working`` report while the child is active (allowing only the design §4
  disclosed shed-frame transient, self-corrected within two poll intervals)
  and reports ``done`` only after the child settles, where the deferred
  outcome stays final. The tail-driven root and child both run real turns
  through the in-gate provider; the OSC bytes are asserted verbatim from the
  PTY byte stream.
* **agent browser open** (§6.5): the browser mounts over the spawned stdio
  topology with a real launched child -- and, since the F10 emission arm
  landed, with the child's row present (the node was landed at F8 while
  root-owned children still never reached the ``agents/update`` events
  layer; the rows were probed empty through the whole child lifecycle at
  the F7/F9 tip b1c04d2b, and the F10 legs below now own the row
  lifecycle). The mount itself stays pinned here: the kitty CSI-u
  encoding opens the browser behind the launch ack with no wire refusals.
* **agent browser rows, transcript view, release, cancel** (§6.5, the F10
  legs): the F10 owner-side emission arm publishes every child-inventory
  mutation as an ``agent/agents/update`` id-less notification, the
  coordinator's pump relays it onto the client-facing ``agents/update``,
  and the browser's event-fed inventory now carries root-owned children on
  the process topology. These nodes drive the full row lifecycle through a
  real spawned child: the row appears at spawn (an ``Active`` entry the
  ``C Stop run`` action answers), settles into ``Retained Idle`` once the
  child's turn completes (where ``Enter`` opens the transcript viewer the
  ``agent/transcript/get`` carrier serves from the child's own process,
  and the guarded ``R Release`` action reaps the row), and a live run
  stops through the confirmed ``Stop run`` action only for the row's
  state to fall back to the retained-idle release target inside the
  poll budget.
* **browser row telemetry** (G3): the settled row's details pane carries the
  telemetry the host can honestly observe -- the run's task summary and the
  child's own committed model (read from the child's session snapshot) --
  where the pre-G3 projection left those fields unset.
* **parent wake on a settled run** (G1b): a non-retained background child
  settles while the parent is idle, the owner's ``agent/run/completed``
  carrier reaches the coordinator's wake owner, and the parent runs an
  injected turn carrying the registry's notification wording -- which then
  fetches exactly that run through ``get_agent_result``. Everything after
  the launch is harness-driven: the second turn exists only because the
  wake started it.
* **launch-ack display identity** (R2): the rendered acknowledgment carries
  the short display handle (``attempt-<first 8>``) in its prose and the exact
  full handle in the ``agent_id="..."`` continuation instruction a re-task has
  to pass back -- both read from the PTY stream with whitespace stripped, so a
  soft wrap cannot satisfy or defeat the assertion.

The module pins its own reply texts (``reply-<index>``): every provider
request gets a unique reply, so each turn's settlement is separately
waitable and a stale render can never satisfy the next wait. The child-launch
scenario routes by request CONTENT, never by request index: an index-routed
script would misroute the interleaved child turns (order not guaranteed).

Every hold in the scripted provider -- the wake child, the deferral child, the
root's follow-up and the two browser-leg runs kept open -- is a bounded poll
loop over a multiple of the liveness poller's own interval, with a hard
timeout that fails an overrun loudly: no hold here is a bare fixed sleep.
"""

from __future__ import annotations

import json
import os
import re
import signal
import time

import pytest

from chartreux.cli.textual_ui.agent_liveness import DEFAULT_POLL_INTERVAL
from tests.pty.harness.drive_tui import (
    StreamingMockServer,
    assistant_text_chunks,
    root_agent_pids,
)

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group(name="pty_gate")]

#: The prompt footer: reliable "we are back at the prompt" marker.
MAIN_SCREEN = "F1 Help · /help"

#: The turns of filler history. Twenty-two messages exceed the resumed tail
#: of twenty, so exactly two entries stay behind the Load-more backfill.
HISTORY_FILLS = 11

#: The beyond-the-tail entry the restore must NOT admit yet, and the exact
#: rendered affordance label once the transcript head is scrolled into view.
OLDEST_FILL = "history-fill-0"
AFFORDANCE = "Load more messages (2)"

#: The spawned-child scenario (composed OSC deferral + agent browser). The
#: launch marker routes the script: any provider request whose LAST user turn
#: carries it and no tool result yet answers with one ``task`` tool call, so a
#: real process-backed child spawns mid-turn (F4/F6 payload path). The root's
#: follow-up holds long enough for the child to register with the liveness
#: poller before the main turn settles; the child's own turn holds longer still
#: so it is still active when the main turn settles.
LAUNCH_MARKER = "probe-launch-a-child-agent"
LAUNCH_PROMPT = f"probe: {LAUNCH_MARKER} now"
CHILD_TASK = "child-probe report the answer to everything"
CHILD_REPLY = "child-probe-settled-marker"
ROOT_DONE = "root turn finished after launch"

#: One ``task`` call: a coder child on the in-gate mock model, background run.
_CHILD_TASK_ARGUMENTS = json.dumps({
    "task": f"{CHILD_TASK}. Say {CHILD_REPLY} and nothing else.",
    "identity": "coder",
    "model": "mock-model",
})

#: The F10 browser legs use RETAINED instances: a retained child keeps its
#: resident ``Retained Idle`` row after the run settles, which is the state the
#: transcript-view and release actions serve; the stop leg launches its own
#: retained child whose provider request is held open so the browser is driven
#: against a live ``Active`` row.
BROWSER_LAUNCH_MARKER = "probe-browser-a-child-agent"
RETAINED_PROMPT = f"probe: {BROWSER_LAUNCH_MARKER} now"
_CHILD_RETAINED_ARGUMENTS = json.dumps({
    "task": f"{CHILD_TASK}. Say {CHILD_REPLY} and nothing else.",
    "identity": "coder",
    "model": "mock-model",
    "retain": True,
})

STOP_LAUNCH_MARKER = "probe-stop-a-child-agent"
STOP_PROMPT = f"probe: {STOP_LAUNCH_MARKER} now"
STOP_CHILD_TASK = "child-probe-stop hold this run open"
STOP_CHILD_REPLY = "child-probe-stop-settled-marker"
_CHILD_STOP_ARGUMENTS = json.dumps({
    "task": f"{STOP_CHILD_TASK}. Say {STOP_CHILD_REPLY} and nothing else.",
    "identity": "coder",
    "model": "mock-model",
    "retain": True,
})

#: The live-telemetry leg (H1): the child takes two tool-only steps -- each one
#: commits an assistant message and relays its own frame -- and then holds its
#: third provider request open, so its row is still ``Running`` with a
#: non-zero committed-turn count while the browser inspects it. Without the
#: H1 observation tick and the relayed interim count, such a row renders
#: model and turns as unknown for the whole run.
RUNNING_LAUNCH_MARKER = "probe-running-a-child-agent"
RUNNING_PROMPT = f"probe: {RUNNING_LAUNCH_MARKER} now"
RUNNING_CHILD_TASK = "child-probe-running hold this row live"
RUNNING_CHILD_REPLY = "child-probe-running-settled-marker"
_RUNNING_CHILD_ARGUMENTS = json.dumps({
    "task": f"{RUNNING_CHILD_TASK}.",
    "identity": "coder",
    "model": "mock-model",
    "retain": True,
})
#: The child's two tool-only steps read the todo list (a side-effect-free
#: builtin every delegated child may call).
_RUNNING_CALL_ID = "call-probe-running-{count}"

#: The parent-wake flow (G1b): a non-retained background child settles while
#: the parent is idle, so the owner's completion carrier wakes the parent with
#: one injected turn that names the run -- and the model fetches that same run
#: through ``get_agent_result``. The child's answer string appears ONLY in the
#: fetched result (never in the echoed task brief), so rendering it is proof
#: the fetch answered with the stored run.
WAKE_LAUNCH_MARKER = "probe-wake-the-parent-agent"
WAKE_PROMPT = f"probe: {WAKE_LAUNCH_MARKER} now"
WAKE_CHILD_TASK = "child-probe-wake answer the parent quietly"
WAKE_CHILD_REPLY = "wake-child-answer-rendered-by-the-fetch"
WAKE_FETCH_DONE = "parent fetched the settled run"
_WAKE_CHILD_ARGUMENTS = json.dumps({
    "task": f"{WAKE_CHILD_TASK}.",
    "identity": "coder",
    "model": "mock-model",
})

#: The completion notification the owner publishes (the in-process wording).
_WAKE_NOTIFICATION = re.compile(
    r"Background agent .+?\((?P<agent>[^,]+), run (?P<run>[0-9a-f]+)\) completed\."
)

#: The fetch the wake turn issues, identified by its own call id: the launch
#: turn's ``task`` result is in the same conversation, so "a tool result
#: exists" never distinguishes the fetch from the launch.
_WAKE_FETCH_CALL_ID = "call-probe-wake-1"

#: The liveness poller's own interval (agent_liveness.DEFAULT_POLL_INTERVAL):
#: the scripted holds and the row budgets below are multiples of it, so a
#: change to the harness cadence moves them together.
_POLL = DEFAULT_POLL_INTERVAL

#: The scripted provider's holds, in poll intervals from the request's arrival:
#: the wake child outlasts the root's own turn, the deferral child stays active
#: across the post-settle observation window, the root's follow-up keeps the
#: main turn live long enough for the poller to sample the registered child,
#: and the two browser-leg runs stay open far past every node's wait budget.
_WAKE_CHILD_HOLD = 12 * _POLL
_ROOT_FOLLOW_UP_HOLD = 6 * _POLL
_DEFERRAL_CHILD_HOLD = 20 * _POLL
_HELD_RUN_HOLD = 240 * _POLL

#: A hold advances in slices of a tenth of a poll interval, re-reading the
#: clock each slice, and fails when it overruns its deadline by more than one
#: poll interval of scheduling slack -- a larger overrun means the wait itself
#: stalled and the reply would land outside the window it encodes.
_HOLD_SLICE = _POLL / 10
_HOLD_TIMEOUT_SLACK = _POLL

#: The browser-row waits: the mutation emission and its relay are event-driven,
#: but the generous budget below keeps a loaded machine inside the same
#: contract -- several times the liveness poll interval, never a fixed sleep.
ROW_BUDGET = 45 * _POLL

#: The browser chrome anchors (the bar's own render strings; wrap-safe).
#: The confirmation prompts wrap at the terminal width and the prompt scroll
#: is one line high, so only each prompt's FIRST line renders -- the anchors
#: are the always-rendered pieces: the row groups, the question, the action
#: buttons, and the confirmation-mode footer (``Enter Choose`` renders only
#: while a confirmation is open).
IDLE_GROUP = "Retained Idle"
EMPTY_ROWS = "No background agents yet"
RELEASE_CONFIRM = "[Release]"
STOP_CONFIRM = "[Stop run]"
CONFIRM_FOOTER = "Enter Choose  Esc Cancel"

#: The OSC 7501 report bytes (session-status design §5, line 100). Pairs join
#: with ":" inside the ";"-terminated report.
OSC_WORKING = "\x1b]7501;state=working:app=chartreux\x07"
OSC_DONE = "\x1b]7501;state=done:app=chartreux\x07"
OSC_IDLE = "\x1b]7501;state=idle:app=chartreux\x07"
OSC_STATE_REPORT = re.compile(r"\x1b\]7501;state=([a-z]+):app=chartreux\x07")


def _messages(payload: dict) -> list[dict[str, object]]:
    return list(payload.get("messages") or [])


def _has_tool_result(payload: dict) -> bool:
    return any(message.get("role") == "tool" for message in _messages(payload))


def _user_texts(payload: dict) -> list[str]:
    """The conversation's user-turn texts, in order."""
    texts: list[str] = []
    for message in _messages(payload):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str):
            texts.append(content)
    return texts


def _is_child_request(payload: dict) -> bool:
    """The child's own conversation opens with its task brief (no tool rows)."""
    users = _user_texts(payload)
    return bool(users) and CHILD_TASK in users[0]


def _is_stop_child_request(payload: dict) -> bool:
    """The stop leg's child conversation opens with its own held task brief."""
    users = _user_texts(payload)
    return bool(users) and STOP_CHILD_TASK in users[0]


def _is_running_child_request(payload: dict) -> bool:
    """The live-telemetry leg's child conversation is its own task brief."""
    users = _user_texts(payload)
    return bool(users) and RUNNING_CHILD_TASK in users[0]


def _is_launch_request(payload: dict) -> bool:
    """A launch turn: any flow's marker in a user turn, before any tool result."""
    texts = _user_texts(payload)
    return any(
        marker in text
        for text in texts
        for marker in (
            LAUNCH_MARKER,
            BROWSER_LAUNCH_MARKER,
            STOP_LAUNCH_MARKER,
            RUNNING_LAUNCH_MARKER,
            WAKE_LAUNCH_MARKER,
        )
    ) and not _has_tool_result(payload)


def _launch_arguments(payload: dict) -> str:
    """The task arguments a marker-launched turn calls, routed by its marker."""
    texts = _user_texts(payload)
    if any(RUNNING_LAUNCH_MARKER in text for text in texts):
        return _RUNNING_CHILD_ARGUMENTS
    if any(STOP_LAUNCH_MARKER in text for text in texts):
        return _CHILD_STOP_ARGUMENTS
    if any(BROWSER_LAUNCH_MARKER in text for text in texts):
        return _CHILD_RETAINED_ARGUMENTS
    if any(WAKE_LAUNCH_MARKER in text for text in texts):
        return _WAKE_CHILD_ARGUMENTS
    return _CHILD_TASK_ARGUMENTS


def _is_wake_child_request(payload: dict) -> bool:
    """The wake flow's child conversation opens with its own quiet task brief."""
    users = _user_texts(payload)
    return bool(users) and WAKE_CHILD_TASK in users[0]


def _wake_notification(payload: dict) -> re.Match[str] | None:
    """The completion notification a parent turn carries, if it carries one."""
    for text in reversed(_user_texts(payload)):
        match = _WAKE_NOTIFICATION.search(text)
        if match is not None:
            return match
    return None


def _has_wake_fetch_result(payload: dict) -> bool:
    """Whether the conversation already carries the wake turn's fetch result."""
    return any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == _WAKE_FETCH_CALL_ID
        for message in _messages(payload)
    )


def _held_reply(hold: float, reply: str, why: str) -> list[dict[str, object]]:
    """Hold a scripted reply open for ``hold`` seconds, then answer it.

    The wait is a bounded poll loop, not one blind sleep: it advances toward
    its deadline in slices of a fraction of the liveness poll interval
    (``hold`` itself is a poll-interval multiple), re-reading the clock each
    slice, and the hard timeout -- the deadline plus one poll interval of
    scheduling slack -- raises rather than letting the reply land outside the
    window it encodes.
    """
    start = time.monotonic()
    deadline = start + hold
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(_HOLD_SLICE, remaining))
    elapsed = time.monotonic() - start
    if elapsed > hold + _HOLD_TIMEOUT_SLACK:
        raise AssertionError(
            f"the scripted provider's {why} hold overran its {hold:g}s "
            f"deadline by {elapsed - hold:.2f}s (hard timeout "
            f"{hold + _HOLD_TIMEOUT_SLACK:g}s)"
        )
    return assistant_text_chunks(f"## Summary\n\n{reply}")


def PROVIDER_CHUNK_SCRIPT(index: int, payload: dict) -> list[dict[str, object]]:
    """Unique replies per request, with the child-launch scenarios routed by content."""
    if _is_wake_child_request(payload):
        # Hold the wake child past the root's own turn: the flow under test is
        # the IDLE-arm wake, so the completion must arrive after the parent's
        # turn settled -- a live turn would be steered instead.
        return _held_reply(_WAKE_CHILD_HOLD, WAKE_CHILD_REPLY, "wake child")
    notification = _wake_notification(payload)
    if notification is not None:
        if _has_wake_fetch_result(payload):
            return assistant_text_chunks(f"## Summary\n\n{WAKE_FETCH_DONE}")
        # The wake named the run: fetch exactly that run, as the notification
        # instructs, so the parent reads the outcome the owner stored.
        return [
            StreamingMockServer.build_chunk(
                created=320,
                delta=StreamingMockServer.build_tool_call_delta(
                    call_id=_WAKE_FETCH_CALL_ID,
                    tool_name="get_agent_result",
                    arguments=json.dumps({
                        "agent_id": notification.group("agent"),
                        "run_id": notification.group("run"),
                        "mode": "full",
                    }),
                ),
                finish_reason=None,
            ),
            StreamingMockServer.build_chunk(
                created=321,
                delta={},
                finish_reason="tool_calls",
                usage={"prompt_tokens": 3, "completion_tokens": 4},
            ),
        ]
    if _is_launch_request(payload):
        return [
            StreamingMockServer.build_chunk(
                created=300,
                delta=StreamingMockServer.build_tool_call_delta(
                    call_id="call-probe-1",
                    tool_name="task",
                    arguments=_launch_arguments(payload),
                ),
                finish_reason=None,
            ),
            StreamingMockServer.build_chunk(
                created=301,
                delta={},
                finish_reason="tool_calls",
                usage={"prompt_tokens": 3, "completion_tokens": 4},
            ),
        ]
    if _is_running_child_request(payload):
        results = [
            message for message in _messages(payload) if message.get("role") == "tool"
        ]
        if len(results) >= 2:
            # Hold the run open at a NON-ZERO committed-turn count: the browser
            # node inspects this row while the run is still in flight.
            return _held_reply(_HELD_RUN_HOLD, RUNNING_CHILD_REPLY, "running child")
        call_id = _RUNNING_CALL_ID.format(count=len(results) + 1)
        return [
            StreamingMockServer.build_chunk(
                created=310,
                delta=StreamingMockServer.build_tool_call_delta(
                    call_id=call_id,
                    tool_name="todo",
                    arguments=json.dumps({"action": "read"}),
                ),
                finish_reason=None,
            ),
            StreamingMockServer.build_chunk(
                created=311,
                delta={},
                finish_reason="tool_calls",
                usage={"prompt_tokens": 3, "completion_tokens": 4},
            ),
        ]
    if _is_stop_child_request(payload):
        # Hold the stop leg's child open: the cancel flow drives the browser
        # while this run is live, and the run must never settle first.
        return _held_reply(_HELD_RUN_HOLD, STOP_CHILD_REPLY, "stop child")
    if _is_child_request(payload):
        # Hold the child's turn open: the deferral observation window is the
        # gap between the root turn settling and this hold ending.
        return _held_reply(_DEFERRAL_CHILD_HOLD, CHILD_REPLY, "deferral child")
    if _has_tool_result(payload):
        # Hold the root's follow-up so the poller samples the registered child
        # while the main turn is still active (the P2/R2 retention window).
        return _held_reply(_ROOT_FOLLOW_UP_HOLD, ROOT_DONE, "root follow-up")
    return assistant_text_chunks(f"## Summary\n\nreply-{index} done")


def _settle_to_prompt(driver) -> None:
    driver.wait_for(MAIN_SCREEN, 40)


def _wait_prompt_idle(driver, timeout: float = 90) -> None:
    driver.wait_until(
        lambda: "Esc/Ctrl+C to interrupt" not in driver.text()[-800:],
        timeout,
        "the running turn to settle",
    )


def _submit(driver, text: str) -> int:
    """Type a line, submit it, and return the transcript mark to inspect from."""
    mark = driver.mark()
    driver.send(text)
    time.sleep(0.4)
    driver.send("\r")
    return mark


def _fill_history(driver) -> None:
    """Run the filler turns, each settled before the next starts."""
    for i in range(HISTORY_FILLS):
        _submit(driver, f"history-fill-{i}")
        driver.wait_for(f"reply-{i} done", 60)
        _wait_prompt_idle(driver)


def test_root_death_recovery_restores_the_session(
    pty_driver_factory, local_mock_provider
) -> None:
    """A killed root never fakes success, and a fresh process recovers it.

    The death leg pins the design §5 semantics: the post-death submission
    reaches no provider and comes back as the input draft. The recovery leg
    pins the durable resume: the replacement root agent is a NEW process on
    the same stdio topology, and the recovered session takes one more real
    provider turn.
    """
    driver = pty_driver_factory()
    _settle_to_prompt(driver)
    root_pid = driver.wait_for_root_agent(30)
    _submit(driver, "root-death-probe marker-1")
    driver.wait_for("reply-0 done", 60)
    _wait_prompt_idle(driver)
    requests_before = len(local_mock_provider.requests)

    os.kill(root_pid, signal.SIGKILL)
    time.sleep(2)
    draft = "after-death-probe marker-2"
    mark = _submit(driver, draft)
    driver.wait_until(
        lambda: draft.replace(" ", "") in "".join(driver.text()[mark:].split()),
        30,
        "the refused post-death submission to restore its draft",
    )
    assert len(local_mock_provider.requests) == requests_before

    driver.close()
    driver2 = pty_driver_factory(["--continue"])
    driver2.wait_for("root-death-probe marker-1", 110)
    replacement = None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        alive = [pid for pid in root_agent_pids(driver2.child.pid) if pid != root_pid]
        if alive:
            replacement = alive[-1]
            break
        time.sleep(0.2)
    assert replacement is not None, "the resumed process spawned no replacement root"

    requests_before_recovery = len(local_mock_provider.requests)
    _submit(driver2, "post-recovery-probe marker-3")
    driver2.wait_until(
        lambda: len(local_mock_provider.requests) > requests_before_recovery,
        60,
        "the recovered session's turn to reach the provider",
    )
    _wait_prompt_idle(driver2)
    errors = driver2.errors_since(0)
    assert not errors, "\n".join(errors)


def test_resumed_history_loads_more_after_scroll(pty_driver_factory) -> None:
    """A restored session reports its older history in the transcript head.

    The restore admits only the resume tail; the oldest entry stays behind the
    backfill until shifted into view, where the head affordance renders the
    exact remaining count.
    """
    driver = pty_driver_factory()
    _settle_to_prompt(driver)
    _fill_history(driver)
    driver.send("\x03")
    driver.wait_for("Press Ctrl+C again to quit", 20)
    driver.send("\x03")
    driver.wait_for_exit(30)

    driver2 = pty_driver_factory(["--continue"])
    driver2.wait_for(f"history-fill-{HISTORY_FILLS - 1}", 110)
    driver2.wait_until(
        lambda: "Resumed session" in driver2.text(), 40, "the resume marker"
    )
    time.sleep(1.5)
    driver2.drain(idle=0.2)
    assert OLDEST_FILL not in driver2.text()[-4000:], (
        "the restore admitted the whole history, so there is no backfill to offer"
    )

    def affordance_visible() -> bool:
        return AFFORDANCE in driver2.text()

    for _ in range(60):
        driver2.send("\x1b[1;2A")
        time.sleep(0.12)
        driver2.drain(idle=0.0)
        if affordance_visible():
            break
    assert affordance_visible(), (
        "the transcript head's Load-more affordance never rendered after the "
        "shift+up scroll: " + driver2.text()[-1200:]
    )
    errors = driver2.errors_since(0)
    assert not errors, "\n".join(errors)


def _osc_report_states(driver, raw_offset: int) -> list[str]:
    """The OSC states the terminal reported since ``raw_offset``, deduplicated.

    Deduplication is adjacency-only, exactly what the design's observable
    sequences assume: the emitter writes one report per transition, so any
    repeat in the byte stream is a re-emission the design excludes, while a
    genuine bounce (working, done, working) must stay visible.
    """
    states = OSC_STATE_REPORT.findall(driver.captured.getvalue()[raw_offset:])
    return [state for i, state in enumerate(states) if not i or states[i - 1] != state]


def test_spawned_child_defers_done_until_the_child_settles(
    pty_driver, local_mock_provider
) -> None:
    """A process-backed child holds the OSC report at ``working`` (P2/R2).

    Session-status design §5 criterion 1 on the real spawned stdio topology:
    one turn launches a real child (the ``task`` call crosses the F4/F6
    spawn payload path and the child's own turn reaches the in-gate provider);
    the main turn settles while the child is still active and the composed
    report must be ``working``. The sample the liveness poller accepted while
    the main turn ran cannot yet carry the child on this topology, so the
    settle may first emit the design §4 disclosed transient -- a shed-frame
    false ``done``, "bounded ... to approximately two poll intervals and
    self-corrected by the periodic read". What the criterion pins on this
    leg is that observable contract: by two poll intervals after the settle
    the report holds the exact ``working`` bytes while the child is still
    active (never a persistent false terminal, never ``idle``), and once the
    child settles the deferred ``done`` lands and stays the owned final
    outcome. The deliberate frame-drop injections stay covered in-process
    (test_program_status_composition.py), where the events layer can be
    doubled and the transient suppressed outright.
    """
    driver = pty_driver
    _settle_to_prompt(driver)
    time.sleep(1.5)
    raw_offset = len(driver.captured.getvalue())
    _submit(driver, LAUNCH_PROMPT)
    driver.wait_for(ROOT_DONE, 90)
    _wait_prompt_idle(driver)

    # The retention checkpoint is two poll intervals after the settle: any
    # settle-time false done must be self-corrected by then, and the report
    # must hold the exact working bytes (the child is still active -- its
    # provider request is held open here).
    time.sleep(3.0)
    assert _osc_report_states(driver, raw_offset)[-1:] == ["working"], (
        "the composed report did not return to working while the child was "
        f"still active: {_osc_report_states(driver, raw_offset)}"
    )
    raw_since = driver.captured.getvalue()[raw_offset:]
    assert OSC_WORKING in raw_since
    # From the retention checkpoint on, while the child is the only live work,
    # nothing may report idle (this flow never interrupts): the settle-time
    # shed-frame transient may emit a false done, but never a false idle.
    checkpoint_offset = len(driver.captured.getvalue())

    # The child settles; the deferred done lands within the poll budget.
    driver.wait_for(CHILD_REPLY, 120)
    child_requests = [
        request
        for request in local_mock_provider.requests
        if any(
            CHILD_TASK in str(message) for message in (request.get("messages") or [])
        )
    ]
    assert child_requests, "the spawned child turns never reached the provider"
    assert OSC_IDLE not in driver.captured.getvalue()[checkpoint_offset:], (
        "the composed report fell back to idle while the child was active"
    )
    settled_offset = len(driver.captured.getvalue())
    driver.wait_until(
        lambda: _osc_report_states(driver, settled_offset) == ["done"],
        45,
        "the deferred done report after the child settled",
    )
    # The deferred outcome is owned and final: it holds after further poll
    # cycles instead of flipping back with a stale frame.
    time.sleep(3.0)
    raw_since = driver.captured.getvalue()[raw_offset:]
    assert OSC_DONE in raw_since
    assert _osc_report_states(driver, settled_offset) == ["done"], (
        "the deferred done did not stay the final composed outcome: "
        f"{_osc_report_states(driver, settled_offset)}"
    )
    errors = driver.errors_since(0)
    assert not errors, "\n".join(errors)


def test_agent_browser_opens_over_the_spawned_child(pty_driver) -> None:
    """The agent browser mounts on the spawned stdio topology (§6.5).

    A real child is launched first, then the browser opens with the kitty
    keyboard encoding of ctrl+shift+a -- the binding the app registers for the
    background-agents toggle; through a PTY the modifier combination only
    travels as the CSI-u ``\\x1b[97;6u`` encoding (legacy terminals send a
    plain \\x01 that cannot express the shift chord).

    This node was landed at F8 against the events-layer boundary: on the
    process topology a root-owned child answered the control-plane inventory
    reads while never arriving in the notification-fed inventory the browser
    lists rows from -- the coordinator's registry did not own this topology's
    children (``AgentOwnedSubagentRunner``), so no ``agents/update``
    publication existed for their mutations and the rows were probed empty
    through the whole child lifecycle at the F7/F9 tip b1c04d2b. The F10
    emission arm closed that gap (the owner now publishes its inventory
    mutations over ``agent/agents/update``), so the node also pins the row's
    presence -- and the row-lifecycle interactions are covered by the three
    dedicated F10 nodes below. The mount itself stays pinned here: the
    browser mounts with no wire refusals while the launch ack renders.
    """
    driver = pty_driver
    _settle_to_prompt(driver)
    _submit(driver, LAUNCH_PROMPT)
    driver.wait_for(ROOT_DONE, 90)
    _wait_prompt_idle(driver)
    driver.wait_for("Launched agent coder", 30)

    driver.send("\x1b[97;6u")
    driver.drain(idle=0.5)
    driver.wait_for("Background agents", 30)
    driver.wait_for("↑↓ Move  ←→ Filter  Enter Open  D Details", 30)
    errors = driver.errors_since(0)
    assert not errors, "\n".join(errors)


def _launch_retained_child_and_settle_it(driver) -> None:
    """Launch the retained browser child and settle the root turn on top."""
    _submit(driver, RETAINED_PROMPT)
    driver.wait_for(ROOT_DONE, 90)
    _wait_prompt_idle(driver)
    driver.wait_for("Launched agent coder", 30)


def _open_browser_and_select_the_row(driver) -> None:
    """Mount the browser, wait the row, and move the selection onto it.

    The two arrow presses are the keyboard path to the first agent row
    (``Main agent`` occupies the movement slot ahead of it); the footer
    gaining the row's action name is the anchor that the selection really
    sits on the child.
    """
    driver.send("\x1b[97;6u")
    driver.drain(idle=0.5)
    driver.wait_for("Background agents", 30)
    driver.wait_for("↑↓ Move  ←→ Filter  Enter Open", 30)
    driver.send("\x1b[B")
    driver.send("\x1b[B")


def test_agent_browser_opens_the_child_transcript_view(pty_driver) -> None:
    """The transcript viewer serves the root-owned child's transcript (§6.5).

    After the child settles, its retained row opens through Enter into the
    agent transcript viewer: the viewer mounts on the selection the events
    layer fed (the F10 emissions -- on the pre-F10 base the row never
    arrives and this flow stops at the first wait), its header names the
    child, and the page itself renders the child's real answer because the
    ``agent/transcript/get`` carrier reads the transcript where the child
    runs -- the process-backend forwarding, not a coordinator copy. Esc
    returns to the browser and closes it with no wire refusals.
    """
    driver = pty_driver
    _settle_to_prompt(driver)
    _launch_retained_child_and_settle_it(driver)
    driver.wait_for(CHILD_REPLY, 120)

    _open_browser_and_select_the_row(driver)
    driver.wait_for(IDLE_GROUP, ROW_BUDGET)
    driver.wait_until(
        lambda: "R Release" in driver.text(), ROW_BUDGET, "the idle row's selection"
    )
    # The transcript text accumulates everything the terminal ever rendered,
    # so the viewer's own page is asserted against the render DELTA since
    # the Enter press -- the child's answer already rendered in the main
    # transcript, and only the ``agent/transcript/get`` carrier can render
    # the child's task brief again inside the viewer.
    mark = driver.mark()
    driver.send("\r")
    driver.wait_until(
        lambda: "Subagent:" in driver.text()[mark:],
        30,
        "the agent transcript viewer to mount",
    )
    driver.wait_until(
        lambda: "and nothing else" in driver.text()[mark:],
        ROW_BUDGET * 2,
        "the child's transcript page to render inside the viewer",
    )
    esc_mark = driver.mark()
    driver.send("\x1b")
    driver.wait_until(
        lambda: "Subagent:" not in driver.text()[esc_mark:],
        30,
        "the transcript viewer to close back to the browser",
    )
    driver.send("\x1b")
    errors = driver.errors_since(0)
    assert not errors, "\n".join(errors)


def test_agent_browser_row_carries_the_host_observed_telemetry(pty_driver) -> None:
    """The browser row shows the telemetry the host can honestly observe (G3).

    The child's details pane is fed by the same projection the inventory
    emits: the run timing and the task summaries are host-side facts the
    owner stamps at admission and settle, and the model is the child's own
    committed identity read from its session snapshot. The pane is asserted
    against the render DELTA since the Enter press -- the row's own launch
    text already rendered in the main transcript -- and ``Model`` is the
    field the pre-G3 projection left unset, so this flow stops there on the
    base.
    """
    driver = pty_driver
    _settle_to_prompt(driver)
    _launch_retained_child_and_settle_it(driver)
    driver.wait_for(CHILD_REPLY, 120)

    _open_browser_and_select_the_row(driver)
    driver.wait_for(IDLE_GROUP, ROW_BUDGET)
    driver.wait_until(
        lambda: "R Release" in driver.text(), ROW_BUDGET, "the idle row's selection"
    )
    mark = driver.mark()
    driver.send("\r")
    driver.wait_until(
        lambda: "Initial task: child-probe" in driver.text()[mark:],
        30,
        "the details pane to carry the run's task summary",
    )
    driver.wait_until(
        lambda: "Model: mock-provider/mock-model" in driver.text()[mark:],
        ROW_BUDGET,
        "the details pane to carry the child's committed model",
    )
    esc_mark = driver.mark()
    driver.send("\x1b")
    driver.wait_until(
        lambda: "Subagent:" not in driver.text()[esc_mark:],
        30,
        "the transcript viewer to close back to the browser",
    )
    driver.send("\x1b")
    errors = driver.errors_since(0)
    assert not errors, "\n".join(errors)


def test_agent_browser_shows_a_running_child_model_and_turns(pty_driver) -> None:
    """A RUNNING child's row carries its model and its live turn count (H1).

    The row is the R1 defect's own surface: the child never settles inside
    this flow (its third provider request holds for 240 poll intervals), so a row that
    renders its model and a non-zero ``turns`` line can only have been fed by
    the live observation tick and by the run-progress count the child stamps
    on its relayed frames. On the pre-H1 base the same held run renders the
    model and ``turns`` as unknown for the whole window, because the child's
    snapshot is read once, at settle, and no frame carries a count.

    The footer's ``C Stop run`` is the anchor that the selection really sits
    on the live row; the details line is asserted on the render DELTA since
    that anchor, so the ``turns`` value is the running row's own.
    """
    driver = pty_driver
    _settle_to_prompt(driver)
    _submit(driver, RUNNING_PROMPT)
    driver.wait_for(ROOT_DONE, 90)
    _wait_prompt_idle(driver)
    driver.wait_for("Launched agent coder", 30)

    _open_browser_and_select_the_row(driver)
    driver.wait_until(
        lambda: "C Stop run" in driver.text(), ROW_BUDGET, "the running row's selection"
    )
    mark = driver.mark()
    driver.wait_until(
        lambda: re.search(r"turns [1-9]", driver.text()[mark:]) is not None,
        ROW_BUDGET,
        "the running row's details to carry its live committed-turn count",
    )
    driver.wait_until(
        lambda: "mock-provider/mock-model" in driver.text()[mark:],
        ROW_BUDGET,
        "the running row's details to carry the child's model",
    )
    driver.send("\x1b")
    errors = driver.errors_since(0)
    assert not errors, "\n".join(errors)


def test_agent_browser_releases_the_retained_child(pty_driver) -> None:
    """The confirmed release reaps the root-owned child's row (§6.5).

    The release action is guarded by its confirmation screen (the Cancel
    button is the default), and confirming posts the run-pinned
    ``agents/release`` through the process carrier; the owner reaps the
    handle, the next emitted inventory drops the row, and the browser's
    authoritative list answers with its empty state -- the release is what
    removed the row, never a client-side filter.
    """
    driver = pty_driver
    _settle_to_prompt(driver)
    _launch_retained_child_and_settle_it(driver)
    driver.wait_for(CHILD_REPLY, 120)

    _open_browser_and_select_the_row(driver)
    driver.wait_for(IDLE_GROUP, ROW_BUDGET)
    driver.wait_until(
        lambda: "R Release" in driver.text(), ROW_BUDGET, "the idle row's selection"
    )
    driver.send("r")
    driver.drain(idle=0.5)
    driver.wait_for(RELEASE_CONFIRM, 30)
    driver.wait_for(CONFIRM_FOOTER, 30)
    driver.send("\x1b[D")
    driver.send("\r")
    driver.wait_for(EMPTY_ROWS, ROW_BUDGET)
    driver.send("\x1b")
    errors = driver.errors_since(0)
    assert not errors, "\n".join(errors)


def test_agent_browser_cancels_the_running_child(pty_driver) -> None:
    """The confirmed stop cancels the live child's run from the row (§6.5).

    A retained child whose provider request is held open keeps its row in
    the Active group, so the selection's footer offers ``C Stop run``; the
    stop confirmation posts the ``agents/cancel`` carrier the F7 runner arm
    answers, the owner interrupts and joins the live run, and the row's
    state falls back to the retained-idle release target inside the poll
    budget -- the child's own answer never arrives, because its process was
    torn down with the run.
    """
    driver = pty_driver
    _settle_to_prompt(driver)
    _submit(driver, STOP_PROMPT)
    driver.wait_for(ROOT_DONE, 90)
    _wait_prompt_idle(driver)
    driver.wait_for("Launched agent coder", 30)

    _open_browser_and_select_the_row(driver)
    driver.wait_until(
        lambda: "C Stop run" in driver.text(), ROW_BUDGET, "the running row's selection"
    )
    driver.send("c")
    driver.drain(idle=0.5)
    driver.wait_for(STOP_CONFIRM, 30)
    driver.wait_for(CONFIRM_FOOTER, 30)
    # The terminal's text accumulates every rendered frame, so the stopped
    # row is asserted against the delta since the confirm press: the running
    # footer with ``C Stop run`` was already rendered and cannot un-render.
    mark = driver.mark()
    driver.send("\x1b[D")
    driver.send("\r")
    driver.wait_until(
        lambda: (
            "Cancelled" in driver.text()[mark:] and "R Release" in driver.text()[mark:]
        ),
        ROW_BUDGET,
        "the stopped row to report Cancelled and fall back to the release target",
    )
    # The child's provider request holds for 240 poll intervals, far past this
    # window, so the stop is the only way its row leaves the Active group --
    # and the "Cancelled" row state is the documented stop disposition, not a
    # settled run. The marker text itself cannot serve as the discriminator:
    # the launch ack echoes the task brief (marker included) into the
    # transcript long before the stop.
    driver.send("\x1b")
    errors = driver.errors_since(0)
    assert not errors, "\n".join(errors)


def test_the_settled_child_wakes_the_parent_and_the_model_fetches_the_run(
    pty_driver, local_mock_provider
) -> None:
    """A settled background child wakes the idle parent (G1b, spawned stdio).

    On this topology the root process owns the child, so only it can report the
    run's outcome: the settled run publishes an id-less ``agent/run/completed``
    notification, the coordinator's wake owner (bound behind the ready frame's
    completion capability) starts the parent's own injected turn carrying the
    registry's notification wording, and that turn fetches exactly the run the
    notification named. Everything after the launch is driven by the harness --
    no further user input is sent -- so the turn that follows is the wake.

    The fetched answer renders in the main transcript, which is the visible
    proof that ``get_agent_result`` answered with the stored run: the child's
    reply string never appears in the echoed task brief, so no earlier frame
    can satisfy the wait. On the pre-G1b base the notification never exists,
    no wake turn starts, and the flow stops at the first fetch wait.
    """
    driver = pty_driver
    _settle_to_prompt(driver)
    _submit(driver, WAKE_PROMPT)
    driver.wait_for(ROOT_DONE, 90)
    _wait_prompt_idle(driver)
    driver.wait_for("Launched agent coder", 30)

    # The child settles while the parent is idle; the wake turn runs itself.
    driver.wait_for(WAKE_FETCH_DONE, 180)

    notifications = [
        text
        for request in local_mock_provider.requests
        for text in _user_texts(request)
        if _WAKE_NOTIFICATION.search(text) is not None
    ]
    assert notifications, "no provider request carried the completion notification"
    for text in notifications:
        assert "Use get_agent_result to retrieve the result." in text
        assert "Retrieve this run by its run_id" in text
    fetched = [
        message
        for request in local_mock_provider.requests
        for message in _messages(request)
        if message.get("role") == "tool"
        and message.get("tool_call_id") == _WAKE_FETCH_CALL_ID
    ]
    assert fetched, "the wake turn never fetched the run it was told about"
    assert any(
        WAKE_CHILD_REPLY in str(message.get("content")) for message in fetched
    ), "the fetch did not answer with the settled run's stored result"
    # The render side: the wake turn's fetch is a visible tool call, and the
    # turn it started settles into the parent's own reply.
    driver.wait_for("get_agent_result", 60)
    driver.wait_for(WAKE_FETCH_DONE, 60)
    errors = driver.errors_since(0)
    assert not errors, "\n".join(errors)


#: A rendered attempt handle: the short DISPLAY form and the full wire one.
_SHORT_ATTEMPT_HANDLE = re.compile(r"attempt-[0-9a-f]{8}(?![0-9a-f])")
_FULL_ATTEMPT_HANDLE = re.compile(r"attempt-[0-9a-f]{32}")


def test_agent_browser_row_shows_the_short_handle(pty_driver) -> None:
    """The browser row renders the short display handle, not the wire one (R2).

    The process-path child's handle is ``coder-attempt-<32 hex>``: the row used
    to print it whole in its dim secondary segment, where it filled the row and
    read as noise. It now renders ``attempt-<first 8 of the uuid>`` -- the row
    is display-only, and the actions it offers (release, stop, open) still pin
    the full handle.

    The assertion is scoped to the row's OWN rendered lines (the ones carrying
    the ``Retained Idle`` state), because the launch ack's collapsed transcript
    line still names the full handle above the browser. On the pre-H2 base the
    same flow renders the 32-hex handle inside the row and fails here.
    """
    driver = pty_driver
    _settle_to_prompt(driver)
    _launch_retained_child_and_settle_it(driver)
    driver.wait_for(CHILD_REPLY, 120)

    mark = driver.mark()
    _open_browser_and_select_the_row(driver)
    driver.wait_for(IDLE_GROUP, ROW_BUDGET)
    driver.wait_until(
        lambda: "R Release" in driver.text()[mark:],
        ROW_BUDGET,
        "the idle row's selection",
    )
    rows = [
        line
        for line in driver.text()[mark:].splitlines()
        if "attempt-" in line and "(run " not in line
    ]
    assert rows, "the browser rendered no attempt handle row"
    assert any(_SHORT_ATTEMPT_HANDLE.search(line) for line in rows), (
        "the browser row did not render a short attempt handle: " + repr(rows[-4:])
    )
    assert not any(_FULL_ATTEMPT_HANDLE.search(line) for line in rows), (
        "the browser row still rendered the full wire handle: " + repr(rows[-4:])
    )
    driver.send("\x1b")
    errors = driver.errors_since(0)
    assert not errors, "\n".join(errors)
