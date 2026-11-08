"""Pilot (Textual) adapter scenario harness (v0.7.5 stage 2, WP4).

The stage-2 cells run on two adapter paths (design v3 :88-91): the headless
app-server path WP3b builds, and the Pilot (Textual) path this module builds.
Both paths drive the SAME committed workload definitions — the seed, prompt
sizes, deterministic response shape and fixture hash that travel with the WP3a
fixture — so a measurement cell can be repeated on either adapter and the two
runs can be compared.

What the harness pins:

* **Workload parity.** :func:`workload_fingerprint` digests what an adapter
  actually delivered per turn (the committed user prompt and the rendered
  assistant text). :func:`assert_workload_parity` requires the headless and
  Pilot transcripts to produce the SAME fingerprint, which is the cross-adapter
  equality the plan's WP4 row gates. The fingerprint is built from delivered
  content, never from the definition alone, so an adapter that drops, reorders
  or truncates a response cannot pass.
* **Real Pilot path.** The turns are driven through a real Textual application
  under the production stylesheet: the committed prompt is mounted as the real
  :class:`UserMessage` and the committed response streams chunk by chunk
  through the real :class:`AssistantMessage` flush boundary, with the Pilot
  advancing the message pump between frames. The rendered markdown is asserted
  against the delivered text, so the D1 seam is measured on a live render path.
* **D1 producer.** Flush timings come from the WP2 observer
  (:mod:`tests.perf._runtime_render`), installed here so a bare-script runner
  or a per-cell subprocess can use the harness without pytest fixtures; the run
  record is the WP2 record, unchanged, and its chunks-coverage fields are the
  ones WP6 transports.
* **Idle cells (r12).** The idle observation cells are DEFINED by the r12
  measurement plan: a 60 s requested window on both adapters, gated on the
  achieved window with ``budget_s`` = 75 s and a kill deadline of at least
  2 x that budget. This module references those numbers from one place and
  verifies an achieved window against them; it defines no second idle cell and
  no second budget rule (plan §3 WP4: verify, do not duplicate).
* **Fail-closed egress (§1).** :class:`~tests.perf._runtime_scenario.EgressGuard`
  is reused rather than reimplemented, so the socket block, the credential
  strip, the config/scratch redirection and the preserved
  ``PYTHONDONTWRITEBYTECODE`` flag hold on this path too.

Nothing here changes production behaviour: the flush observer is installed
from the test side on the widget class and removed on exit, and every widget
the harness drives is a production widget.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.pilot import Pilot

from chartreux.cli.textual_ui.widgets.messages import AssistantMessage, UserMessage
from tests.perf._runtime_fixture import FIXTURE_TURN_WIDTH
from tests.perf._runtime_render import (
    D1_WARMUP_FLUSHES,
    RenderTimingObserver,
    build_run_record,
)
from tests.perf._runtime_scenario import (
    CELL_FRESH_HEADLESS,
    FRESH_TOTAL_AGENTS,
    SCENARIO_METRIC_VERSION,
    SHORTENED_SCALE,
    ScenarioScale,
    WorkloadSpec,
    response_chunks,
    root_user_content,
    topology_fingerprint,
    workload_spec,
)

PILOT_ADAPTER = "pilot"
HEADLESS_ADAPTER = "headless"
PILOT_SCENARIO_SCHEMA = "chartreux.runtime_pilot_scenario.v1"
WORKLOAD_FINGERPRINT_SCHEMA = "chartreux.runtime_workload_fingerprint.v1"

#: The structural labels of the Pilot adapter path. The Pilot path has no
#: app-server projector: the live render surface is the transcript widget
#: itself, which is what its D1 flush samples come from.
PILOT_TRANSPORT_KIND = "textual_app"
PILOT_EVENT_LOOP_ROLE = "textual_app_message_loop"
PILOT_REGISTRY_PATH = "none_textual_adapter"
PILOT_PROJECTOR_LIVE = False

#: The production stylesheet the Pilot path renders under (the same file the
#: Chartreux application loads), resolved from this module's own location.
PILOT_CSS_PATH = (
    Path(__file__).resolve().parents[2]
    / "chartreux"
    / "cli"
    / "textual_ui"
    / "app.tcss"
)
PILOT_HISTORY_CONTAINER_ID = "history"

#: Frame pacing of the contract run. The production default
#: (``STREAM_WRITE_FRAME_SECONDS`` = 50 ms) makes a shortened node wait on the
#: scheduler for every frame; the contract run steers the widget's own frame
#: delay down to the production floor so each committed chunk still crosses the
#: real flush boundary, one frame at a time, without spending 50 ms per frame.
PILOT_FRAME_DELAY_S = 0.002
PILOT_SETTLE_POLL_S = 0.002
PILOT_FLUSH_DEADLINE_S = 2.0

#: The in-suite parity geometry: the shortened contract scale with the child
#: cohort removed, because the Pilot streaming cell is a fresh single-root
#: transcript (the loaded child matrix stays on the headless path).
CONTRACT_TURNS = 2
PILOT_PARITY_SCALE = replace(
    SHORTENED_SCALE,
    name="pilot_parity",
    root_turns=CONTRACT_TURNS,
    history_messages=FIXTURE_TURN_WIDTH,
    child_agents=0,
    burst_agents=0,
)

# --------------------------------------------------------------------------
# The existing idle cells (r12 definition, referenced not redefined)
# --------------------------------------------------------------------------

#: r12: idle cells gate the measured WINDOW — 60 s requested plus a 15 s
#: timer/scheduler tolerance, with setup/teardown inside the kill deadline.
IDLE_PHASE = "idle"
IDLE_REQUESTED_WINDOW_S = 60.0
IDLE_WINDOW_TOLERANCE_S = 15.0
IDLE_WINDOW_BUDGET_S = IDLE_REQUESTED_WINDOW_S + IDLE_WINDOW_TOLERANCE_S
IDLE_KILL_DEADLINE_FACTOR = 2
IDLE_KILL_DEADLINE_S = IDLE_KILL_DEADLINE_FACTOR * IDLE_WINDOW_BUDGET_S
#: r12: the phase split every cell reports, idle included.
IDLE_PHASES: tuple[str, ...] = ("startup", "steady", "drain_teardown", IDLE_PHASE)
#: r12 defines exactly two idle cells — one per adapter. The Pilot cell is that
#: definition applied to this adapter, never a second one.
IDLE_CELL_ADAPTERS: tuple[str, ...] = (HEADLESS_ADAPTER, PILOT_ADAPTER)
IDLE_GATED_QUANTITY = "measured_window"


@dataclass(frozen=True, slots=True)
class IdleWindowVerdict:
    """One idle cell's window outcome, in the r12 status taxonomy."""

    status: str
    measured_window_s: float
    budget_s: float
    reasons: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "measured_window_s": round(self.measured_window_s, 3),
            "budget_s": self.budget_s,
            "gated_quantity": IDLE_GATED_QUANTITY,
            "reasons": list(self.reasons),
        }


def evaluate_idle_window(
    measured_window_s: float, *, completed: bool = True
) -> IdleWindowVerdict:
    """Gate one idle cell's achieved window against the r12 idle budget.

    The gated quantity is the measured window itself: a nominal 60 s observation
    cannot be mislabelled by scheduler latency, and a run that never reached the
    requested window is ``incomplete`` with its reason rather than ``ok``.
    """
    if measured_window_s < 0:
        raise ValueError("measured_window_s must be non-negative")
    if not completed:
        return IdleWindowVerdict(
            status="incomplete",
            measured_window_s=measured_window_s,
            budget_s=IDLE_WINDOW_BUDGET_S,
            reasons=("window_not_completed",),
        )
    if measured_window_s > IDLE_WINDOW_BUDGET_S:
        return IdleWindowVerdict(
            status="over_budget",
            measured_window_s=measured_window_s,
            budget_s=IDLE_WINDOW_BUDGET_S,
            reasons=("measured_window_over_budget",),
        )
    return IdleWindowVerdict(
        status="ok", measured_window_s=measured_window_s, budget_s=IDLE_WINDOW_BUDGET_S
    )


@dataclass(frozen=True, slots=True)
class IdleCell:
    """One existing idle cell, expressed from the single r12 constant set."""

    adapter: str
    requested_window_s: float = IDLE_REQUESTED_WINDOW_S
    budget_s: float = IDLE_WINDOW_BUDGET_S
    kill_deadline_s: float = IDLE_KILL_DEADLINE_S
    gated_quantity: str = IDLE_GATED_QUANTITY
    phase: str = IDLE_PHASE

    def record(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "phase": self.phase,
            "requested_window_s": self.requested_window_s,
            "budget_s": self.budget_s,
            "kill_deadline_s": self.kill_deadline_s,
            "gated_quantity": self.gated_quantity,
        }


def idle_cell(adapter: str) -> IdleCell:
    """Return the existing idle cell of ``adapter`` (r12; not a new cell)."""
    if adapter not in IDLE_CELL_ADAPTERS:
        raise ValueError(f"no idle cell for adapter: {adapter}")
    return IdleCell(adapter=adapter)


# --------------------------------------------------------------------------
# Workload parity
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkloadTurn:
    """What one adapter actually delivered for one committed turn."""

    turn: int
    user_chars: int
    user_digest: str
    assistant_chars: int
    assistant_digest: str

    def record(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "user_chars": self.user_chars,
            "user_digest": self.user_digest,
            "assistant_chars": self.assistant_chars,
            "assistant_digest": self.assistant_digest,
        }


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def make_workload_turn(
    turn: int, *, user_text: str, assistant_text: str
) -> WorkloadTurn:
    """Digest one delivered turn into its adapter-independent shape."""
    return WorkloadTurn(
        turn=turn,
        user_chars=len(user_text),
        user_digest=_digest(user_text),
        assistant_chars=len(assistant_text),
        assistant_digest=_digest(assistant_text),
    )


def workload_fingerprint(
    cell: str, workload_hash: str, turns: Sequence[WorkloadTurn]
) -> str:
    """Digest the delivered transcript of one run, adapter-independently.

    Only delivered content enters the digest: turn order, the user prompt and
    the assistant text each adapter produced. Two adapters driving the same
    committed workload therefore produce the same value, and any drift in what
    an adapter delivered changes it.
    """
    if not turns:
        raise ValueError("workload_fingerprint requires at least one turn")
    digest = sha256()
    digest.update(f"{WORKLOAD_FINGERPRINT_SCHEMA}|{cell}|{workload_hash}|".encode())
    for turn in turns:
        digest.update(
            (
                f"{turn.turn}|{turn.user_chars}|{turn.user_digest}"
                f"|{turn.assistant_chars}|{turn.assistant_digest}|"
            ).encode()
        )
    return digest.hexdigest()


def assert_workload_parity(
    *,
    cell: str,
    workload_hash: str,
    headless_turns: Sequence[WorkloadTurn],
    pilot_turns: Sequence[WorkloadTurn],
) -> str:
    """Assert both adapters delivered the same transcript; return its digest.

    Per-turn content is compared first (so a failure names the diverging turn),
    then the whole-run fingerprints. Equality of the fingerprints is the
    cross-adapter parity the plan's WP4 row gates.
    """
    if len(headless_turns) != len(pilot_turns):
        raise AssertionError(
            f"adapters delivered {len(headless_turns)} and {len(pilot_turns)} turns"
        )
    for headless_turn, pilot_turn in zip(headless_turns, pilot_turns, strict=True):
        if headless_turn != pilot_turn:
            raise AssertionError(
                f"adapters diverged on turn {headless_turn.turn}:"
                f" headless={headless_turn.record()} pilot={pilot_turn.record()}"
            )
    headless = workload_fingerprint(cell, workload_hash, headless_turns)
    pilot = workload_fingerprint(cell, workload_hash, pilot_turns)
    if headless != pilot:
        raise AssertionError(
            f"cross-adapter workload fingerprint mismatch: {headless} != {pilot}"
        )
    return pilot


#: The structural labels both adapters must agree on, and the per-adapter
#: labels each side must pin. Object identity never crosses a process boundary
#: (WP3b's stable-label rule), so only these labels are compared.
CROSS_ADAPTER_SHARED_LABELS: tuple[str, ...] = ("cell", "total_agents")
ADAPTER_LABELS: dict[str, dict[str, Any]] = {
    HEADLESS_ADAPTER: {
        "adapter": HEADLESS_ADAPTER,
        "projector_live": True,
        "transport": "memory_json_rpc",
        "event_loop_role": "single_test_loop",
        "registry_construction_path": "SessionRuntimeRegistry._build_child_runtime",
    },
    PILOT_ADAPTER: {
        "adapter": PILOT_ADAPTER,
        "projector_live": PILOT_PROJECTOR_LIVE,
        "transport": PILOT_TRANSPORT_KIND,
        "event_loop_role": PILOT_EVENT_LOOP_ROLE,
        "registry_construction_path": PILOT_REGISTRY_PATH,
    },
}


def pilot_topology(
    *, cell: str, total_agents: int = FRESH_TOTAL_AGENTS
) -> dict[str, Any]:
    """Return the Pilot path's stable structural labels."""
    labels = ADAPTER_LABELS[PILOT_ADAPTER]
    return topology_fingerprint(
        cell=cell,
        projector_live=bool(labels["projector_live"]),
        total_agents=total_agents,
        transport_kind=str(labels["transport"]),
        event_loop_role=str(labels["event_loop_role"]),
        registry_construction_path=str(labels["registry_construction_path"]),
        adapter=PILOT_ADAPTER,
    )


def assert_cross_adapter_topology(
    headless: dict[str, Any], pilot: dict[str, Any]
) -> None:
    """Assert both paths describe the same structural scenario.

    The shared labels must match; every adapter-specific label must equal its
    pinned value, so a silently changed transport, event-loop role or
    registry path fails instead of degrading the comparison.
    """
    if set(headless) != set(pilot):
        raise AssertionError(
            f"topology label sets differ: {sorted(set(headless) ^ set(pilot))}"
        )
    for label in CROSS_ADAPTER_SHARED_LABELS:
        if headless[label] != pilot[label]:
            raise AssertionError(
                f"topology label {label!r} differs: {headless[label]!r} != {pilot[label]!r}"
            )
    for comparison, adapter in ((headless, HEADLESS_ADAPTER), (pilot, PILOT_ADAPTER)):
        for label, expected in ADAPTER_LABELS[adapter].items():
            if comparison[label] != expected:
                raise AssertionError(
                    f"{adapter} label {label!r} is {comparison[label]!r},"
                    f" expected {expected!r}"
                )
    if headless["adapter"] == pilot["adapter"]:
        raise AssertionError("both topologies claim the same adapter")


# --------------------------------------------------------------------------
# The Pilot scenario
# --------------------------------------------------------------------------


class PilotScenarioApp(App[None]):
    """The real Textual application the Pilot path drives.

    It is deliberately the transcript surface the streaming render path needs:
    the production stylesheet and the history container the messages mount
    into. The Chartreux application composes the same container with the same
    stylesheet, so the widget tree under measurement is the production one.
    """

    CSS_PATH = PILOT_CSS_PATH

    def compose(self) -> ComposeResult:
        yield Vertical(id=PILOT_HISTORY_CONTAINER_ID)


@contextmanager
def pilot_render_observer(
    *, warmup_flushes: int = D1_WARMUP_FLUSHES
) -> Iterator[RenderTimingObserver]:
    """Install the WP2 D1 observer on the real flush boundary.

    Self-contained on purpose (§1): a bare-script runner or a per-cell
    subprocess has no pytest fixtures, and the observer needs a patch manager
    to install and undo itself.
    """
    observer = RenderTimingObserver(warmup_flushes=warmup_flushes)
    patch = pytest.MonkeyPatch()
    observer.install(patch)
    try:
        yield observer
    finally:
        patch.undo()


def _rendered_markdown(message: AssistantMessage) -> str:
    markdown = message._markdown
    if markdown is None:
        raise AssertionError("the assistant message rendered no markdown widget")
    return markdown._markdown


async def _settle_flush(
    pilot: Pilot[None], message: AssistantMessage, expected: str
) -> None:
    """Advance the Pilot until the frame's flush is rendered."""
    deadline = asyncio.get_running_loop().time() + PILOT_FLUSH_DEADLINE_S
    while _rendered_markdown(message) != expected:
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(
                f"the flush never landed: rendered {_rendered_markdown(message)!r}"
                f" of {expected!r}"
            )
        await pilot.pause(PILOT_SETTLE_POLL_S)


@dataclass(frozen=True, slots=True)
class PilotTurnResult:
    """One turn driven through the Pilot adapter path."""

    turn: int
    user: WorkloadTurn
    chunks: int
    flushes: int
    wall_ms: float

    def record(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "chunks": self.chunks,
            "flushes": self.flushes,
            "wall_ms": round(self.wall_ms, 3),
            **self.user.record(),
        }


class PilotScenario:
    """The Pilot adapter path over one cell's committed workload.

    :meth:`run_workload` drives the turns and releases the application; the
    records it exposes afterwards are the run's — the D1 render record comes
    from the WP2 producer, and the workload turns feed the cross-adapter
    fingerprint comparison.
    """

    def __init__(
        self,
        *,
        cell: str = CELL_FRESH_HEADLESS,
        scale: ScenarioScale = PILOT_PARITY_SCALE,
        warmup_flushes: int = D1_WARMUP_FLUSHES,
        frame_delay_s: float = PILOT_FRAME_DELAY_S,
    ) -> None:
        if scale.child_agents or scale.burst_agents:
            raise ValueError("the Pilot streaming cell is a single-root transcript")
        self.cell = cell
        self.spec: WorkloadSpec = workload_spec(cell)
        self.scale = scale
        self.warmup_flushes = warmup_flushes
        self.frame_delay_s = frame_delay_s
        self.turns: list[PilotTurnResult] = []
        self._observer: RenderTimingObserver | None = None

    @property
    def adapter(self) -> str:
        return PILOT_ADAPTER

    def topology(self) -> dict[str, Any]:
        return pilot_topology(cell=self.cell, total_agents=FRESH_TOTAL_AGENTS)

    def _require_observer(self) -> RenderTimingObserver:
        if self._observer is None:
            raise RuntimeError("the scenario has not run a workload yet")
        return self._observer

    async def run_workload(self, *, turns: int | None = None) -> list[PilotTurnResult]:
        """Drive the committed workload through a live Pilot application."""
        if self.turns:
            raise RuntimeError("a scenario drives one workload")
        count = self.scale.root_turns if turns is None else turns
        if count <= 0:
            raise ValueError("turns must be positive")
        app = PilotScenarioApp()
        with pilot_render_observer(warmup_flushes=self.warmup_flushes) as observer:
            async with app.run_test() as pilot:
                history = app.query_one(f"#{PILOT_HISTORY_CONTAINER_ID}", Vertical)
                for turn in range(count):
                    self.turns.append(
                        await self._run_turn(pilot, history, turn, observer)
                    )
        self._observer = observer
        return list(self.turns)

    async def _run_turn(
        self,
        pilot: Pilot[None],
        history: Vertical,
        turn: int,
        observer: RenderTimingObserver,
    ) -> PilotTurnResult:
        prompt = root_user_content(turn)
        user = UserMessage(prompt)
        await history.mount(user)
        message = AssistantMessage("")
        await history.mount(message)
        # Steer this widget's own frame pacing; the flush boundary, its
        # scheduling and the render path stay production's.
        message._frame_delay = self.frame_delay_s
        await pilot.pause()
        flushes_before = observer.log.non_empty_flushes
        started = perf_counter()
        expected = ""
        for chunk in response_chunks(turn):
            expected += chunk
            await message.append_content(chunk)
            await _settle_flush(pilot, message, expected)
        wall_ms = (perf_counter() - started) * 1000
        delivered = message.get_content()
        if delivered != expected:
            raise AssertionError(
                f"turn {turn} delivered {len(delivered)} chars,"
                f" the committed response is {len(expected)}"
            )
        if user.get_content() != prompt:
            raise AssertionError(f"turn {turn} rendered a different user prompt")
        return PilotTurnResult(
            turn=turn,
            user=make_workload_turn(
                turn, user_text=user.get_content(), assistant_text=delivered
            ),
            chunks=len(response_chunks(turn)),
            flushes=observer.log.non_empty_flushes - flushes_before,
            wall_ms=wall_ms,
        )

    def workload_turns(self) -> list[WorkloadTurn]:
        """Return the delivered turns in their adapter-independent shape."""
        return [turn.user for turn in self.turns]

    def workload_fingerprint(self) -> str:
        return workload_fingerprint(
            self.cell, self.spec.digest(), self.workload_turns()
        )

    def d1_record(self) -> dict[str, Any]:
        """Return the D1 run record the WP2 producer builds from these flushes."""
        return build_run_record(
            self._require_observer().log,
            completed_turns=len(self.turns),
            fixture=self.spec.fixture_hash,
        )

    def chunks_coverage(self) -> dict[str, Any]:
        """Return the run's chunks-coverage fields (§3) for the pilot cell."""
        record = self.d1_record()
        coverage = dict(record["chunks_coverage"])
        chunks = sum(turn.chunks for turn in self.turns)
        missing: list[str] = []
        if not self.turns:
            missing.append("turns")
        if not chunks:
            missing.append("chunks")
        if not coverage["measured_flushes"] and not coverage["warmup_flushes"]:
            missing.append("render_flushes")
        coverage["turns"] = len(self.turns)
        coverage["chunks"] = chunks
        coverage["complete"] = not missing
        coverage["missing"] = missing
        return coverage

    def run_record(self) -> dict[str, Any]:
        """Return the Pilot run record (workload, coverage, D1, turn samples)."""
        return {
            "schema": PILOT_SCENARIO_SCHEMA,
            "metric_version": SCENARIO_METRIC_VERSION,
            "cell": self.cell,
            "adapter": PILOT_ADAPTER,
            "scale": self.scale.dimensions(),
            "workload": self.spec.record(),
            "workload_hash": self.spec.digest(),
            "workload_fingerprint": self.workload_fingerprint(),
            "fixture_hash": self.spec.fixture_hash,
            "topology": self.topology(),
            "chunks_coverage": self.chunks_coverage(),
            "d1": self.d1_record(),
            "turns": [turn.record() for turn in self.turns],
        }


# --------------------------------------------------------------------------
# The headless side of the comparison
# --------------------------------------------------------------------------


def _role_is(message: Any, role: str) -> bool:
    return str(getattr(message, "role", "")).rsplit(".", 1)[-1] == role


def headless_workload_turns(
    messages: Sequence[Any], *, turns: int
) -> list[WorkloadTurn]:
    """Extract the delivered turns of a headless transcript.

    The committed prompts are unique per turn, so each turn is located by its
    own prompt rather than by an offset: a seeded history that the loop trims
    or that a child completion extends cannot shift the extraction.
    """
    entries: list[WorkloadTurn] = []
    for turn in range(turns):
        prompt = root_user_content(turn)
        prompt_index = next(
            (
                index
                for index, message in enumerate(messages)
                if _role_is(message, "user")
                and (getattr(message, "content", None) or "") == prompt
            ),
            None,
        )
        if prompt_index is None:
            raise AssertionError(f"the headless transcript never carried turn {turn}")
        if prompt_index + 1 >= len(messages):
            raise AssertionError(f"turn {turn} has no assistant reply")
        reply = messages[prompt_index + 1]
        if not _role_is(reply, "assistant"):
            raise AssertionError(f"turn {turn}'s reply is not an assistant message")
        entries.append(
            make_workload_turn(
                turn,
                user_text=prompt,
                assistant_text=getattr(reply, "content", "") or "",
            )
        )
    return entries


__all__ = [
    "ADAPTER_LABELS",
    "CONTRACT_TURNS",
    "CROSS_ADAPTER_SHARED_LABELS",
    "HEADLESS_ADAPTER",
    "IDLE_CELL_ADAPTERS",
    "IDLE_GATED_QUANTITY",
    "IDLE_KILL_DEADLINE_S",
    "IDLE_PHASE",
    "IDLE_PHASES",
    "IDLE_REQUESTED_WINDOW_S",
    "IDLE_WINDOW_BUDGET_S",
    "IDLE_WINDOW_TOLERANCE_S",
    "PILOT_ADAPTER",
    "PILOT_PARITY_SCALE",
    "PILOT_SCENARIO_SCHEMA",
    "IdleCell",
    "IdleWindowVerdict",
    "PilotScenario",
    "PilotScenarioApp",
    "PilotTurnResult",
    "WorkloadTurn",
    "assert_cross_adapter_topology",
    "assert_workload_parity",
    "evaluate_idle_window",
    "headless_workload_turns",
    "idle_cell",
    "make_workload_turn",
    "pilot_render_observer",
    "pilot_topology",
    "workload_fingerprint",
]
