"""Session-status design §5 criterion-4 drop legs on the spawned stdio backend.

The session-status design's criterion 4 requires the dropped-frame scenarios
observable against BOTH actual backends (criterion 7). In-process the events
layer can be doubled (test_program_status_composition.py pins that leg); this
module injects the drops against a REAL spawned stdio backend, where the
mutation frames cross real process boundaries, and asserts the exact OSC
7501 bytes straight from the PTY stream. Every wait here derives from the
liveness poller's own interval -- never a fixed sleep -- and every node
asserts the COMPLETE deduplicated state sequence, not one checkpoint.

Since F10 the root owner emits its child-inventory mutations onto
``agent/agents/update`` and the process-session backend relays them onto the
client-facing ``agents/update``, so a root-owned child's start and completion
DO reach the TUI as presentation frames: a drop is no longer free on this
topology, it has to be injected and proven. The injection is a test-only seam
inside the SPAWNED TUI: the driver puts a ``sitecustomize`` on the spawned
process's ``PYTHONPATH`` (the same technique ``test_installed_contracts``
uses for its network guard) and that seam wraps two production seams --

* ``ProcessSessionBackend._relay_agents_update`` (the F10 relay): a frame is
  either SUPPRESSED (the drop window is open) or relayed, and both outcomes
  are written to a JSONL receipt in the control directory. Every node's drop
  asserts its own receipt: the intended frame was suppressed, no presentation
  frame was relayed inside the asserted window, and the seam relayed again
  once the window closed (a seam that broke every frame would fail that).
* ``ChartreuxApp._read_agent_liveness`` (the poller's control read): every
  accepted read is receipted, so a drop window also proves fresh control
  reads remained available inside it -- the recovery is the poll's, not a
  frame's.

Each OSC observation window starts at the launch turn's own first ``working``
report: the design's expected sequences exclude initial setup, and under gate
load the prompt can still emit boot-adjacent reports after it renders.

The injection mechanisms, one per leg, are deterministic under the gate:

* **dropped final** (criterion 4): the drop window opens immediately before
  the child is SIGKILLed mid-turn (its own provider request is provably still
  held open), so the host's terminal publication is suppressed -- after F10
  that is the frame that would carry the completion, and SIGKILL alone no
  longer establishes its absence. Since G1a the settled run is not reaped out
  of the inventory, it leaves a persistent CLOSED row, so the suppressed frame
  projects that closed row instead of the vanished handle the pre-G1a reap
  left behind. The receipt pins suppressed frames with a revision newer than
  anything the client had already been told, none of them projecting a running
  child, and the periodic fresh read must self-correct: the owned deferred done
  lands through the poll and stays final. While the child was provably alive
  the report holds working (the design §4 disclosed settle-time shed-frame
  transient, self-corrected within approximately two poll intervals, is the
  only allowed excursion; never a persistent false terminal, never idle).
* **dropped start** (criterion 4): the drop window opens BEFORE the launch is
  submitted, so the spawn row and the running flip are both suppressed and
  the working report can only come from the periodic fresh read. The child is
  held active for many poll intervals so it spans far more than the
  criterion's "before one accepted sample until after the NEXT accepted
  sample" window, and two liveness checkpoints pin the working bytes and the
  child's control reads. The window then closes and the child's normal settle
  must relay a frame again, proving the seam delivers when it is not
  dropping, and the poll yields the deferred done. Since G1b that settle also
  WAKES the parent for the settled run, so the composed report continues with
  the wake turn's own working leg and its terminal done -- the deferred done
  the poll owns stays the first post-settle leg.
* **retain-on-unknown** (criterion 4's stale/timeout/root-death injections):
  SIGSTOP freezes the real root process -- the wire goes silent while the
  poll's read is in flight, the response that eventually arrives is
  genuinely stale, and a killed root drives every read to a transport
  failure. Silence and stale data are UNKNOWN, never evidence of no
  agents: the report must not flip to any terminal during either window,
  and the dead session's owned done must never resurrect or turn idle. The
  blackout offset is taken BEFORE the SIGSTOP, so an emission written at any
  point inside the blackout window is inside the inspected range.

Negative controls (the proof each assertion above can fail):

* **poll disabled** (both recovery legs): the same launch flow, with the
  seam failing every control read from a marked point on. The positive
  node's recovery assertion is then REQUIRED to fail -- and the report must
  never reach ``done`` -- proving the recovery really rides the poll.
* **injected blackout-window emission**: a fabricated accepted sample is
  driven through the app's own reconcile path while the root is stopped, and
  the pre-SIGSTOP offset's no-emission assertion is REQUIRED to catch it.

Criterion 4's rotation leg (an old incarnation resurrecting its deferred
outcome across a resume) stays in-process: the poller's per-incarnation
floor is pinned there, and the process-topology resume spawns a fresh TUI
whose emitter has no shared state with the dead process, so no observable
on one PTY can discriminate it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import time

import pytest

from chartreux.app_server.child_agent import CHILD_AGENT_MODULE
from chartreux.cli.textual_ui.agent_liveness import DEFAULT_POLL_INTERVAL
from tests.pty.harness.drive_tui import (
    StreamingMockServer,
    TuiDriver,
    assistant_text_chunks,
    process_tree,
)
from tests.pty.test_smoke_stdio_flows import (
    OSC_IDLE,
    OSC_STATE_REPORT,
    OSC_WORKING,
    _has_tool_result,
    _osc_report_states,
    _settle_to_prompt,
    _user_texts,
    _wait_prompt_idle,
)

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group(name="pty_gate")]

#: The liveness poller's own interval (agent_liveness.DEFAULT_POLL_INTERVAL):
#: every wait and hold below is a multiple of it, never a fixed sleep.
_POLL = DEFAULT_POLL_INTERVAL

#: Per-node scenario tags, carried inside the launch marker so the scripted
#: provider routes by request CONTENT (F8's contract: index routing would
#: misroute the interleaved child turns).
_LAUNCH_TAG = "drops-launch"

#: How long each node's child holds its own provider request open, in poll
#: intervals from the request's arrival. The child answers only after the
#: hold, so the child is provably mid-turn for the whole window.
_CHILD_HOLDS: dict[str, float] = {
    "final": 28 * _POLL,
    "start": 16 * _POLL,
    "blackout": 28 * _POLL,
    # The negative-control legs only need to reach their fault point and the
    # recovery window, so their children hold for fewer intervals.
    "fault-start": 12 * _POLL,
    "fault-final": 12 * _POLL,
}

#: Wait budgets, all poll-interval multiples: the reported-completion wait,
#: the settle-time transient self-correction checkpoint (the disclosed bound
#: is approximately two poll intervals; three guarantees the correction has
#: landed), the post-drop self-correction budget, the cross-cycles stability
#: hold, and the two injection windows.
_TURN_BUDGET = 90 * _POLL
_RETENTION_CHECKPOINT = 3 * _POLL
_SELF_CORRECT_BUDGET = 8 * _POLL
_STABILITY_HOLD = 2 * _POLL
_BLACKOUT_WINDOW = 3 * _POLL
_DEATH_WINDOW = 3 * _POLL
_FAULT_SETTLE = 3 * _POLL

#: The composed report bytes (working/idle above) are the flows module's
#: exact design-§5 constants, imported verbatim; the sequence each node
#: allows end-to-end is working -> done, or that with the one disclosed
#: settle-time transient working -> done -> working -> done.
LAUNCH_ACK = "Launched agent coder"
_CHILD_REPLY = "drops-child-settled-marker"
_ROOT_DONE = "root turn finished after the drops launch"

#: The dropped-start flow's post-settle shape: the deferred done the periodic
#: read owns, then the working leg the G1b completion wake starts, then that
#: wake turn's own terminal done.
_SETTLE_SHAPE = ("done", "working", "done")

#: The control-directory channel between this test process and the spawned
#: TUI's injection seam. Deliberately not a ``CHARTREUX_*`` name: that prefix
#: is the runtime's configuration namespace.
_CONTROL_ENV = "PYTEST_CHARTREUX_DROP_CONTROL"

#: The test-only seam installed on the spawned TUI's ``PYTHONPATH`` as
#: ``sitecustomize``. Intentionally self-contained: it patches the two
#: production presentation seams once their modules load (a meta-path finder,
#: so nothing is imported before the app would import it), is inert without
#: the control environment, and never raises into the app.
_HOOK_SOURCE = '''"""G4 presentation-drop injection seam (written by the drops test module).

Inert unless ``PYTEST_CHARTREUX_DROP_CONTROL`` names the control directory.
See tests/pty/test_smoke_stdio_drops.py for the contract of each seam.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

_CONTROL_ENV = "PYTEST_CHARTREUX_DROP_CONTROL"
_WATCHER = None
_PATCHES = {}


class _Control:
    def __init__(self, path):
        self.path = path

    def mode(self):
        try:
            with open(os.path.join(self.path, "mode"), encoding="utf-8") as stream:
                return stream.read().strip() or "pass"
        except OSError:
            return "pass"

    def has(self, marker):
        return os.path.exists(os.path.join(self.path, marker))

    def take(self, marker):
        try:
            os.replace(
                os.path.join(self.path, marker),
                os.path.join(self.path, marker + ".used"),
            )
        except OSError:
            return False
        return True

    def record(self, record):
        record["at"] = time.monotonic()
        try:
            with open(
                os.path.join(self.path, "receipts.jsonl"), "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(record) + "\\n")
        except OSError:
            return


def _control():
    path = os.environ.get(_CONTROL_ENV)
    return _Control(path) if path else None


def _field(mapping, *names):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _frame(payload):
    return {
        "session": _field(payload, "sessionId", "session_id"),
        "revision": payload.get("revision"),
        "agents": [
            {
                "id": _field(agent, "agentId", "agent_id"),
                "availability": agent.get("availability"),
                "last": _field(agent, "lastRunStatus", "last_run_status"),
            }
            for agent in (payload.get("agents") or [])
        ],
    }


def _response_frame(response):
    return {
        "session": response.session_id,
        "revision": response.revision,
        "agents": [
            {
                "id": agent.agent_id,
                "availability": agent.availability,
                "last": agent.last_run_status,
            }
            for agent in response.agents
        ],
    }


def _patch_psb(module):
    original = module.ProcessSessionBackend._relay_agents_update

    async def relay(self, method, payload):
        control = _control()
        if control is None:
            return await original(self, method, payload)
        frame = _frame(dict(payload))
        if control.mode() == "drop":
            control.record({"kind": "suppressed", "method": method, "frame": frame})
            return None
        result = await original(self, method, payload)
        control.record({"kind": "delivered", "method": method, "frame": frame})
        return result

    module.ProcessSessionBackend._relay_agents_update = relay


def _patch_app(module):
    for name in ("ChartreuxApp", "App"):
        cls = getattr(module, name, None)
        if cls is None or not hasattr(cls, "_read_agent_liveness"):
            continue
        original = cls._read_agent_liveness

        async def read(self, _original=original):
            control = _control()
            if control is None:
                return await _original(self)
            _ensure_watcher(self, control)
            if control.has("poll-fail"):
                control.record({"kind": "read-failed"})
                raise ConnectionError(
                    "injected liveness read failure (negative control)"
                )
            response = await _original(self)
            control.record({"kind": "read", "frame": _response_frame(response)})
            return response

        cls._read_agent_liveness = read
        return


def _ensure_watcher(app, control):
    global _WATCHER
    if _WATCHER is not None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _WATCHER = loop.create_task(_watch(app, control))


async def _watch(app, control):
    while True:
        try:
            if control.take("inject-answer"):
                _inject_accepted_sample(app, control)
        except Exception:
            pass
        await asyncio.sleep(0.25)


def _inject_accepted_sample(app, control):
    """Drive one fabricated accepted sample through the app's own reconcile path.

    A negative control: the sample claims no active child (a fabricated
    no-children answer), so the real composition decides a terminal and the
    real emitter writes the OSC bytes -- the emission the blackout window's
    pre-SIGSTOP offset must catch.
    """
    from chartreux.app_server.protocol import AgentsInventoryReadResponse

    poller = app._agent_liveness
    previous = poller.accepted
    revision = 0 if previous is None else previous.revision + 1
    now = time.monotonic()
    response = AgentsInventoryReadResponse(
        session_id=None, revision=revision, agents=[]
    )
    snapshot = poller._accept(response, poller.fence, now, now)
    if snapshot is None:
        control.record({"kind": "inject-rejected"})
        return
    app._reconcile_liveness_snapshot(snapshot)
    control.record({"kind": "inject-accepted"})


class _PatchLoader:
    def __init__(self, inner, patch):
        self._inner = inner
        self._patch = patch

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        self._patch(module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _PatchFinder:
    def find_spec(self, fullname, path=None, target=None):
        patch = _PATCHES.get(fullname)
        if patch is None or fullname in sys.modules:
            return None
        for finder in list(sys.meta_path):
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is None or spec.loader is None:
                continue
            spec.loader = _PatchLoader(spec.loader, patch)
            return spec
        return None


_PATCHES.update({
    "chartreux.app_server.process_session_backend": _patch_psb,
    "chartreux.cli.textual_ui.app": _patch_app,
})

if os.environ.get(_CONTROL_ENV):
    sys.meta_path.insert(0, _PatchFinder())
'''


@dataclass(frozen=True)
class _Frame:
    """One presentation frame's observable projection, as the seam saw it."""

    session: str | None
    revision: int | None
    availabilities: tuple[str | None, ...] = field(default_factory=tuple)

    @property
    def carries_child(self) -> bool:
        """Whether the frame projected at least one inventory row."""
        return bool(self.availabilities)

    @property
    def running(self) -> bool:
        """Whether the frame still projected a running child."""
        return any(state == "running" for state in self.availabilities)


@dataclass(frozen=True)
class _Receipt:
    """One line of the seam's JSONL receipt."""

    kind: str
    frame: _Frame


_EMPTY_FRAME = _Frame(session=None, revision=None)


def _child_task_text(tag: str) -> str:
    return f"drops-{tag} child report the answer to everything"


_CHILD_TASK_ARGUMENTS: dict[str, str] = {
    tag: json.dumps({
        "task": f"{_child_task_text(tag)}. Say {_CHILD_REPLY} and nothing else.",
        "identity": "coder",
        "model": "mock-model",
    })
    for tag in _CHILD_HOLDS
}


def _launch_tag(payload: dict) -> str | None:
    """The node tag a launch marker carries in any of the user texts."""
    for text in _user_texts(payload):
        for tag in _CHILD_HOLDS:
            if f"{_LAUNCH_TAG} {tag}" in text:
                return tag
    return None


def _is_child_request(payload: dict) -> str | None:
    """The child's own task brief opens with this node's tagged brief."""
    users = _user_texts(payload)
    if not users:
        return None
    for tag in _CHILD_HOLDS:
        if _child_task_text(tag) in users[0]:
            return tag
    return None


def _launch_prompt(tag: str) -> str:
    return f"probe: {_LAUNCH_TAG} {tag} now"


def PROVIDER_CHUNK_SCRIPT(index: int, payload: dict) -> list[dict[str, object]]:
    """Content-routed replies: launch, held child, held follow-up, plain."""
    tag = _launch_tag(payload)
    if tag is not None and not _has_tool_result(payload):
        return [
            StreamingMockServer.build_chunk(
                created=300,
                delta=StreamingMockServer.build_tool_call_delta(
                    call_id=f"call-drops-{tag}",
                    tool_name="task",
                    arguments=_CHILD_TASK_ARGUMENTS[tag],
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
    child_tag = _is_child_request(payload)
    if child_tag is not None:
        # The child stays mid-turn for the whole hold: its liveness at the
        # drop point is the open provider request itself.
        time.sleep(_CHILD_HOLDS[child_tag])
        return assistant_text_chunks(f"## Summary\n\n{_CHILD_REPLY}")
    if _has_tool_result(payload):
        # Hold the root's follow-up so the freshly spawned child registers
        # with the poller before the main turn settles.
        time.sleep(3 * _POLL)
        return assistant_text_chunks(f"## Summary\n\n{_ROOT_DONE}")
    return assistant_text_chunks(f"## Summary\n\ndrops-reply-{index} done")


def _hold(driver: TuiDriver, seconds: float) -> None:
    """Hold the wall clock in small slices, draining the PTY as we go.

    An undrained PTY blocks the app's writer thread, so every timed window
    here reads continuously instead of sleeping blindly.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if driver.drain(idle=0.2):
            return


def _raw_offset(driver: TuiDriver) -> int:
    return len(driver.captured.getvalue())


def _osc_reports(driver: TuiDriver) -> list[str]:
    """Every OSC state report the terminal wrote so far, in order."""
    return OSC_STATE_REPORT.findall(driver.captured.getvalue())


def _turn_window_start(driver: TuiDriver, submit_offset: int) -> int:
    """The raw offset of the launch turn's OWN first ``working`` report.

    The design's expected sequences exclude initial setup, and the boot-
    adjacent prompt can still emit an idle (or a transient report pair) after
    the prompt text renders -- under load the quiet heuristic anchored before
    one of those, which put setup chatter inside the asserted window. The
    turn's first ``working`` is the observation window's start: the whole turn
    and its deferred terminal stay inside it, and nothing from setup does.
    """
    deadline = time.monotonic() + 30 * _POLL
    while time.monotonic() < deadline:
        index = driver.captured.getvalue().find(OSC_WORKING, submit_offset)
        if index != -1:
            return index
        if driver.drain(idle=0.1):
            break
    raise AssertionError(
        f"the launch turn never reported working: {_osc_reports(driver)[-8:]}"
    )


def _states(driver: TuiDriver, offset: int) -> list[str]:
    """The OSC state reports since ``offset``, deduplicated by adjacency.

    A transition the emitter wrote more than once shows repeats; the shared
    helper dedups them so only genuine bounces count as sequence legs.
    """
    return _osc_report_states(driver, offset)


def _child_agent_pids(driver: TuiDriver) -> list[int]:
    """The spawned child agent processes in the TUI's tree."""
    child = driver.child
    if child is None or child.pid is None:
        return []
    return [
        pid
        for pid, argv in process_tree(child.pid)
        if any(CHILD_AGENT_MODULE in argument for argument in argv)
    ]


def _wait_for_child_pid(driver: TuiDriver) -> int:
    """The one spawned child, waited for deterministically."""
    deadline = time.monotonic() + 30 * _POLL
    while time.monotonic() < deadline:
        pids = _child_agent_pids(driver)
        if pids:
            return pids[0]
        if driver.drain(idle=0.2):
            break
    raise AssertionError(
        "no spawned child agent process appeared in the tree: "
        f"{_child_agent_pids(driver)}"
    )


def _assert_child_request_live(
    local_mock_provider: StreamingMockServer, tag: str
) -> int:
    """The child's own turn reached the provider; returns its request count."""
    requests = [
        request
        for request in local_mock_provider.requests
        if any(
            _child_task_text(tag) in str(message)
            for message in (request.get("messages") or [])
        )
    ]
    assert requests, f"the spawned child's own turn never reached the provider ({tag})"
    return len(requests)


def _assert_working_retention(driver: TuiDriver, raw_offset: int) -> list[str]:
    """After the settle, past the transient bound: working, never idle."""
    states = _states(driver, raw_offset)
    assert states[-1:] == ["working"], (
        "the composed report did not hold working past the disclosed "
        f"transient bound while the child is active: {states}"
    )
    assert OSC_WORKING in driver.captured.getvalue()[raw_offset:]
    assert OSC_IDLE not in driver.captured.getvalue()[raw_offset:], (
        "the composed report fell back to idle while the child was active"
    )
    return states


def _assert_full_sequence(driver: TuiDriver, raw_offset: int) -> None:
    """The COMPLETE end-to-end sequence, not one checkpoint.

    Exactly working -> done, or that with the ONE further working leg in
    the middle (working -> done -> working -> done) -- the design-§4
    settle-time transient, or since G1b the completion wake's own turn. A
    persistent false done while the child was active would have to hold
    done before the settle, and neither shape admits it. Idle is
    forbidden outright in a completed-turn flow.
    """
    states = _states(driver, raw_offset)
    assert states in ("working done".split(), "working done working done".split()), (
        f"the composed sequence left the allowed shapes: {states}"
    )


def _wait_transition(
    driver: TuiDriver, offset: int, state: str, budget: float, what: str
) -> None:
    """Wait (a poll-multiple budget) for the report to reach ``state``."""
    driver.wait_until(lambda: _states(driver, offset)[-1:] == [state], budget, what)


def _assert_no_reports(driver: TuiDriver, offset: int, message: str) -> None:
    """No OSC state report may have been written since ``offset``."""
    states = _states(driver, offset)
    assert states == [], f"{message}: {states}"


# -- the control directory and the seam's receipts --------------------------


def _set_mode(control: Path, mode: str) -> None:
    """Open or close the drop window atomically (never a torn control file)."""
    staged = control / "mode.tmp"
    staged.write_text(mode, encoding="utf-8")
    os.replace(staged, control / "mode")


def _touch(control: Path, marker: str) -> None:
    (control / marker).write_text("", encoding="utf-8")


def _frame_from_raw(raw: object) -> _Frame:
    assert isinstance(raw, dict), f"malformed receipt frame: {raw!r}"
    session = raw.get("session")
    revision = raw.get("revision")
    agents = raw.get("agents")
    assert session is None or isinstance(session, str), raw
    assert revision is None or isinstance(revision, int), raw
    assert isinstance(agents, list), raw
    availabilities: list[str | None] = []
    for agent in agents:
        assert isinstance(agent, dict), agent
        availability = agent.get("availability")
        assert availability is None or isinstance(availability, str), agent
        availabilities.append(availability)
    return _Frame(
        session=session, revision=revision, availabilities=tuple(availabilities)
    )


def _receipts(control: Path) -> list[_Receipt]:
    """Every well-formed receipt line the seam has written so far.

    A trailing line that is not yet newline-terminated is a write still in
    flight (the app appends one line per seam call); it is skipped, never
    parsed as a torn record.
    """
    try:
        raw = (control / "receipts.jsonl").read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    lines = raw.split("\n")
    pending = lines.pop() if lines else ""
    receipts: list[_Receipt] = []
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        assert isinstance(entry, dict), f"malformed receipt: {line!r}"
        kind = entry.get("kind")
        assert isinstance(kind, str), f"malformed receipt: {line!r}"
        frame = entry.get("frame")
        receipts.append(
            _Receipt(
                kind=kind,
                frame=_EMPTY_FRAME if frame is None else _frame_from_raw(frame),
            )
        )
    assert not pending.strip(), f"unterminated receipt line: {pending!r}"
    return receipts


def _wait_receipt(
    driver: TuiDriver,
    control: Path,
    predicate: Callable[[_Receipt], bool],
    budget: float,
    what: str,
) -> _Receipt:
    """Wait -- draining the PTY -- for the seam to receipt a matching frame."""
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        for receipt in _receipts(control):
            if predicate(receipt):
                return receipt
        driver.drain(idle=0)
        time.sleep(0.05)
    raise AssertionError(
        f"the presentation seam never receipted {what}; recent receipts: "
        f"{_receipts(control)[-6:]}"
    )


def _assert_drop_window(
    control: Path,
    since: int,
    *,
    phase: str,
    require_child_row: bool,
    expect_terminal: bool,
    require_child_read: bool,
    min_reads: int,
    newer_than_revision: int | None = None,
) -> None:
    """The receipt proof for one injected drop.

    The intended frame was actually suppressed (a suppressed frame carrying
    the child's row, and/or one projecting the child out of the running state
    with a revision newer than anything the client already had), no
    presentation frame was relayed while the window was open, and fresh
    control reads kept answering inside it.
    """
    entries = _receipts(control)[since:]
    delivered = [entry for entry in entries if entry.kind == "delivered"]
    assert not delivered, (
        f"presentation traffic reached the client inside the {phase} drop "
        f"window: {[entry.frame.availabilities for entry in delivered[:2]]}"
    )
    suppressed = [entry for entry in entries if entry.kind == "suppressed"]
    assert suppressed, (
        f"the presentation relay recorded no suppressed frame for the {phase} "
        "drop: the intended update was never exercised"
    )
    assert all(entry.frame.session is not None for entry in suppressed), (
        f"a suppressed frame carried no session ({phase}): {suppressed[:2]}"
    )
    if require_child_row:
        assert any(entry.frame.carries_child for entry in suppressed), (
            f"no suppressed frame projected the child's row ({phase})"
        )
    if expect_terminal:
        assert any(not entry.frame.running for entry in suppressed), (
            "no suppressed frame projected the child out of the running "
            f"state ({phase}): the completion update was never dropped"
        )
    if newer_than_revision is not None:
        assert any(
            entry.frame.revision is not None
            and entry.frame.revision > newer_than_revision
            for entry in suppressed
        ), (
            "no suppressed frame carried a revision newer than the last "
            f"relayed one ({phase}): the completion mutation was not dropped"
        )
    reads = [entry for entry in entries if entry.kind == "read"]
    assert len(reads) >= min_reads, (
        f"only {len(reads)} fresh control read(s) answered inside the {phase} "
        f"drop window (expected at least {min_reads})"
    )
    if require_child_read:
        assert any(entry.frame.carries_child for entry in reads), (
            f"no control read inside the {phase} window returned the child's row"
        )


@contextmanager
def _instrumented_session(
    tmp_path: Path,
    provider: StreamingMockServer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
) -> Iterator[tuple[TuiDriver, Path]]:
    """A spawned TUI whose presentation seams report to a control directory."""
    control = tmp_path / "drop-control"
    control.mkdir(parents=True, exist_ok=True)
    (control / "receipts.jsonl").write_text("", encoding="utf-8")
    _set_mode(control, mode)
    hook_dir = tmp_path / "drop-hook"
    hook_dir.mkdir(parents=True, exist_ok=True)
    (hook_dir / "sitecustomize.py").write_text(_HOOK_SOURCE, encoding="utf-8")
    inherited = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(part for part in (str(hook_dir), inherited) if part),
    )
    monkeypatch.setenv(_CONTROL_ENV, str(control))
    driver = TuiDriver(tmp_path / "tui", provider).spawn()
    try:
        yield driver, control
    finally:
        driver.close()
        driver.snapshot()


def _launch(driver: TuiDriver, tag: str) -> int:
    """Drive one tagged launch turn and return the turn's observation offset."""
    _settle_to_prompt(driver)
    driver.send(_launch_prompt(tag))
    time.sleep(0.4)
    submit_offset = _raw_offset(driver)
    driver.send("\r")
    driver.wait_for(_ROOT_DONE, _TURN_BUDGET)
    _wait_prompt_idle(driver)
    driver.wait_for(LAUNCH_ACK, _TURN_BUDGET)
    return _turn_window_start(driver, submit_offset)


def _assert_recovery_impossible(driver: TuiDriver, offset: int, what: str) -> None:
    """The positive node's recovery assertion, REQUIRED to fail (negative control)."""
    with pytest.raises(AssertionError):
        _wait_transition(driver, offset, "done", _SELF_CORRECT_BUDGET, what)
    assert "done" not in _states(driver, offset), (
        f"the report reached done without a control read: {_states(driver, offset)}"
    )


def test_dropped_final_update_self_corrects_through_the_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdio_after_autouse: str,
    local_mock_provider: StreamingMockServer,
) -> None:
    """Criterion 4, sole-final drop on the spawned stdio topology.

    The host relays its completion as an ``agents/update`` frame since F10
    and a SIGKILLed child would still trigger it, so the drop window opens
    first and the frame is provably suppressed. Since G1a that frame is the
    settled run's persistent closed row -- the handle is no longer reaped out
    of the inventory, so the suppressed mutation is the row going closed.
    While the child is alive the report must hold the exact working bytes past
    the disclosed §4 transient bound; once the (dropped) completion becomes
    observable only through the control read, the poll must self-correct to
    the owned deferred done within its budget, and it must stay the final
    outcome across further poll cycles -- never idle, never resurrected.
    """
    with _instrumented_session(
        tmp_path, local_mock_provider, monkeypatch, mode="pass"
    ) as (driver, control):
        raw_offset = _launch(driver, "final")
        # The seam's live control: the spawn row was relayed before the drop.
        spawn_frame = _wait_receipt(
            driver,
            control,
            lambda receipt: receipt.kind == "delivered" and receipt.frame.carries_child,
            _TURN_BUDGET,
            "the child's spawn publication (the drop seam's live control)",
        )
        assert spawn_frame.frame.session is not None

        # Retention past the settle: any false terminal self-corrected already.
        _hold(driver, _RETENTION_CHECKPOINT)
        _assert_working_retention(driver, raw_offset)
        # The child is provably alive at the drop: open request, live pid.
        _assert_child_request_live(local_mock_provider, "final")
        child_pid = _wait_for_child_pid(driver)
        os.kill(child_pid, 0)

        # Drop every presentation update from here: the terminal projection --
        # since G1a the settled run's persistent closed row, not the vanished
        # handle a reap used to leave -- is the frame that would otherwise
        # carry the completion.
        _set_mode(control, "drop")
        armed = len(_receipts(control))
        # Everything the client had been told so far: the suppressed frames
        # must be NEWER mutations, not re-reads of a revision it already had.
        relayed_revisions = [
            entry.frame.revision
            for entry in _receipts(control)
            if entry.kind == "delivered" and entry.frame.revision is not None
        ]
        assert relayed_revisions, "no delivered frame carried a revision"
        newest_relayed = max(relayed_revisions)
        kill_offset = _raw_offset(driver)
        os.kill(child_pid, signal.SIGKILL)
        _wait_transition(
            driver,
            kill_offset,
            "done",
            _SELF_CORRECT_BUDGET,
            "the periodic poll to self-correct to the deferred done "
            "after the child's completion was dropped",
        )
        # Full sequence since the launch: working, at most the one disclosed
        # settle-time transient, then done as the owned final outcome.
        _hold(driver, _STABILITY_HOLD)
        assert _states(driver, kill_offset) == ["done"], _states(driver, kill_offset)
        _assert_full_sequence(driver, raw_offset)
        assert OSC_IDLE not in driver.captured.getvalue()[raw_offset:]

        _assert_drop_window(
            control,
            armed,
            phase="final",
            require_child_row=False,
            expect_terminal=True,
            require_child_read=False,
            min_reads=2,
            newer_than_revision=newest_relayed,
        )
        errors = driver.errors_since(0)
        assert not errors, "\n".join(errors)


def test_dropped_start_update_yields_working_through_the_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdio_after_autouse: str,
    local_mock_provider: StreamingMockServer,
) -> None:
    """Criterion 4, sole-start drop on the spawned stdio topology.

    Since F10 the spawn row and the running flip ARE relayed presentation
    frames, so the drop window opens before the launch is submitted and the
    receipt proves those frames were suppressed: the child's registration can
    only come from the periodic fresh reads. The child is active from well
    before one accepted sample until after the next accepted sample: two
    liveness checkpoints, three poll intervals apart, both pin the working
    bytes and a fresh control read that returned the child while its provider
    turn is provably open. The window then closes, the child's own settle
    relays a frame again (the seam delivers when it is not dropping), and the
    poll yields the deferred done -- proving the held working was the live
    child's, not a stuck report. Since G1b that settle also wakes the parent,
    so the composed report continues with the wake turn's working leg and its
    terminal done; the poll-owned deferred done stays the first post-settle
    leg.
    """
    with _instrumented_session(
        tmp_path, local_mock_provider, monkeypatch, mode="drop"
    ) as (driver, control):
        armed = len(_receipts(control))
        raw_offset = _launch(driver, "start")

        # The child is live: its provider turn is held open by the script.
        _assert_child_request_live(local_mock_provider, "start")
        child_pid = _wait_for_child_pid(driver)
        os.kill(child_pid, 0)

        # First liveness checkpoint: samples already accepted with the child
        # active; the report is the poll's discovery alone.
        _hold(driver, _RETENTION_CHECKPOINT)
        first = _assert_working_retention(driver, raw_offset)
        # Second liveness checkpoint, another window of accepted samples later.
        _hold(driver, _RETENTION_CHECKPOINT)
        second = _assert_working_retention(driver, raw_offset)
        assert first[-1] == "working" and second[-1] == "working"

        # The receipt proof for the whole asserted window: the start frames
        # were suppressed, nothing was relayed, and the poll kept answering.
        _assert_drop_window(
            control,
            armed,
            phase="start",
            require_child_row=True,
            expect_terminal=False,
            require_child_read=True,
            min_reads=3,
        )

        # Close the window: the child's settle must relay again, proving the
        # seam's deliveries were suppressed rather than broken.
        _set_mode(control, "pass")
        settle_offset = _raw_offset(driver)
        driver.wait_for(_CHILD_REPLY, _TURN_BUDGET)
        settled = _wait_receipt(
            driver,
            control,
            lambda receipt: receipt.kind == "delivered" and not receipt.frame.running,
            _TURN_BUDGET,
            "a relayed frame after the drop window closed",
        )
        assert settled.frame.session is not None

        # The poll yields the deferred done once the child settled.
        _wait_transition(
            driver,
            settle_offset,
            "done",
            _SELF_CORRECT_BUDGET,
            "the periodic poll to emit the deferred done once the child settled",
        )
        _hold(driver, _STABILITY_HOLD)
        # Post-G1b the settled run also wakes the parent, so past the deferred
        # done the composed report takes the wake's own working leg and its
        # terminal done. What the poll alone owns is still the FIRST leg: the
        # deferred done that arrives with no relayed frame behind it.
        driver.wait_until(
            lambda: tuple(_states(driver, settle_offset)) == _SETTLE_SHAPE,
            _TURN_BUDGET,
            "the deferred done, the completion wake's working leg and its "
            "terminal done",
        )
        _hold(driver, _STABILITY_HOLD)
        assert tuple(_states(driver, settle_offset)) == _SETTLE_SHAPE, _states(
            driver, settle_offset
        )
        _assert_full_sequence(driver, raw_offset)
        assert OSC_IDLE not in driver.captured.getvalue()[raw_offset:]
        errors = driver.errors_since(0)
        assert not errors, "\n".join(errors)


def test_transport_blackout_and_root_death_retain_on_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdio_after_autouse: str,
    local_mock_provider: StreamingMockServer,
) -> None:
    """Criterion 4's stale and root-death injections on the spawned topology.

    A SIGKILLed root would be the confirmed-death receipt case (receipt-
    scoped, in-process pinned); this node injects the two retain-on-unknown
    races around real process signals. A SIGSTOPped root is a genuine
    transport blackout: the in-flight read hangs, the eventual response is
    stale, and silence must never be read as no-children -- the report may
    not flip to any terminal during the window. The window's offset is taken
    BEFORE the SIGSTOP, so an emission at any point inside the blackout is
    inside the inspected range. After the resume the owned deferred done
    lands through the poll when the child is dropped mid-turn, and the root's
    death afterwards must emit nothing new: no resurrection of the dead
    session's outcome, no false idle, through a last window of failing reads.
    """
    with _instrumented_session(
        tmp_path, local_mock_provider, monkeypatch, mode="pass"
    ) as (driver, control):
        raw_offset = _launch(driver, "blackout")

        _hold(driver, _RETENTION_CHECKPOINT)
        _assert_working_retention(driver, raw_offset)
        _assert_child_request_live(local_mock_provider, "blackout")
        child_pid = _wait_for_child_pid(driver)
        root_pid = driver.wait_for_root_agent(30 * _POLL)

        # The blackout: the wire goes silent while a read is in flight. The
        # observation offset is recorded BEFORE the stop so the whole window
        # is inside the inspected range.
        blackout_offset = _raw_offset(driver)
        os.kill(root_pid, signal.SIGSTOP)
        _hold(driver, _BLACKOUT_WINDOW)
        _assert_no_reports(
            driver, blackout_offset, "the silent transport drove an emission"
        )
        os.kill(root_pid, signal.SIGCONT)

        # The resumed backend still answers with the live child: the late,
        # stale response may not flip the report off working.
        _hold(driver, _STABILITY_HOLD)
        resumed = _assert_working_retention(driver, raw_offset)

        # The child's completion is dropped mid-turn (the criterion 4 drop);
        # the periodic read self-corrects to the owned deferred done.
        _set_mode(control, "drop")
        kill_offset = _raw_offset(driver)
        os.kill(child_pid, signal.SIGKILL)
        _wait_transition(
            driver,
            kill_offset,
            "done",
            _SELF_CORRECT_BUDGET,
            "the periodic poll to emit the deferred done after the resume",
        )
        _hold(driver, _STABILITY_HOLD)
        assert _states(driver, kill_offset) == ["done"], _states(driver, kill_offset)
        assert resumed[-1] == "working"

        # Root death with the owned terminal on the wire: every read fails from
        # here on, which is UNKNOWN, never no-children. Nothing new may report.
        death_offset = _raw_offset(driver)
        os.kill(root_pid, signal.SIGKILL)
        _hold(driver, _DEATH_WINDOW)
        _assert_no_reports(driver, death_offset, "the dead root drove a new emission")
        assert OSC_IDLE not in driver.captured.getvalue()[raw_offset:]
        _assert_full_sequence(driver, raw_offset)


@pytest.mark.parametrize("leg", ["start", "final"])
def test_disabled_poll_never_self_corrects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdio_after_autouse: str,
    local_mock_provider: StreamingMockServer,
    leg: str,
) -> None:
    """Negative control: the recovery in each drop leg rides the control read.

    The same launch flow as the matching positive node, with the seam failing
    every control read from a marked point on. The positive node's recovery
    assertion is REQUIRED to fail and the report must never reach ``done``:
    a green positive node cannot be green because of some other path.
    """
    tag = f"fault-{leg}"
    mode = "pass" if leg == "final" else "drop"
    with _instrumented_session(
        tmp_path, local_mock_provider, monkeypatch, mode=mode
    ) as (driver, control):
        raw_offset = _launch(driver, tag)
        _assert_child_request_live(local_mock_provider, tag)
        child_pid = _wait_for_child_pid(driver)

        _hold(driver, _RETENTION_CHECKPOINT)
        _assert_working_retention(driver, raw_offset)
        # Disable the control read; the app must actually take the fault.
        _touch(control, "poll-fail")
        _wait_receipt(
            driver,
            control,
            lambda receipt: receipt.kind == "read-failed",
            _RETENTION_CHECKPOINT,
            "the app to take the injected poll failure",
        )
        check_offset = _raw_offset(driver)

        if leg == "final":
            os.kill(child_pid, signal.SIGKILL)
            _assert_recovery_impossible(
                driver,
                check_offset,
                "the periodic poll to self-correct to the deferred done "
                "after the child's completion was dropped",
            )
        else:
            driver.wait_for(_CHILD_REPLY, _TURN_BUDGET)
            _hold(driver, _FAULT_SETTLE)
            _assert_recovery_impossible(
                driver,
                check_offset,
                "the periodic poll to emit the deferred done once the child settled",
            )
        assert _states(driver, raw_offset)[-1:] == ["working"], _states(
            driver, raw_offset
        )


def test_blackout_window_catches_an_injected_emission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdio_after_autouse: str,
    local_mock_provider: StreamingMockServer,
) -> None:
    """Negative control: the pre-SIGSTOP offset inspects the whole window.

    A fabricated accepted sample (a no-children answer) is driven through the
    app's own reconcile path while the root is stopped, so the terminal
    report is real OSC output written inside the blackout. The positive
    node's no-emission assertion -- taken from the offset recorded BEFORE the
    SIGSTOP -- is REQUIRED to catch it: an offset recorded after the window
    would not.
    """
    with _instrumented_session(
        tmp_path, local_mock_provider, monkeypatch, mode="pass"
    ) as (driver, control):
        raw_offset = _launch(driver, "blackout")
        _hold(driver, _RETENTION_CHECKPOINT)
        _assert_working_retention(driver, raw_offset)
        _assert_child_request_live(local_mock_provider, "blackout")
        _wait_for_child_pid(driver)
        root_pid = driver.wait_for_root_agent(30 * _POLL)

        blackout_offset = _raw_offset(driver)
        os.kill(root_pid, signal.SIGSTOP)
        try:
            _touch(control, "inject-answer")
            driver.wait_until(
                lambda: _states(driver, blackout_offset) != [],
                _SELF_CORRECT_BUDGET,
                "the injected blackout-window emission to reach the terminal",
            )
            with pytest.raises(
                AssertionError, match="silent transport drove an emission"
            ):
                _assert_no_reports(
                    driver, blackout_offset, "the silent transport drove an emission"
                )
        finally:
            os.kill(root_pid, signal.SIGCONT)
