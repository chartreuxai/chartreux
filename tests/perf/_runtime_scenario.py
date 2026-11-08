"""Real headless transport/projector scenario harness (v0.7.5 stage 2, WP3b).

Every headless runtime cell of the stage-2 campaign measures the real
app-server machinery, not a stand-in: the root session's turn controller keeps
its live :class:`~chartreux.app_server._projector.EventProjector`, the
server-to-client notifications travel through a real
:func:`~chartreux.app_server.transport.memory_transport_pair`, and the client
side of that pair feeds a real
:class:`~chartreux.app_server.events.ClientProjection`. The
``tests/perf/_harness.py`` projector-disabled precedent is deliberately not
followed — a cell that observes a disabled projector measures nothing the plan
gates.

What the harness pins (plan §2, §3, §5):

* **Topology.** :func:`topology_fingerprint` reports the STABLE STRUCTURAL
  labels a cross-process comparison may use (adapter, event-loop role,
  registry construction path, projector liveness, transport kind); object
  identity is asserted inside the process instead.
* **P1 provenance.** The projector's per-delta serialization happens once per
  side and its id-to-position index is the maintained one; the harness carries
  :func:`instrument_delta_dumps` so a cell can prove both claims against the
  real ``_patch`` / ``_update_entry`` path instead of assuming them.
* **P2 memo.** The gateway's ``_prepare_backend_messages`` snapshot memo is the
  request context a cell measures; :meth:`HeadlessScenario.backend_request_context`
  reports the memo object identity the backend actually received.
* **Workload.** The deterministic growing-history and synchronized-child
  definitions (seed, message sizes, response shape) are committed with the
  WP3a fixture hash in :data:`WORKLOAD_HASHES`; the loaded cell is 8 agents
  (root + 7 children) holding the 200k-equivalent history each.
* **Chunks coverage (§3).** The harness produces root-turn counts and the
  serialized notification count/byte fields per root turn (X2's gated
  quantities) and reports them as :meth:`HeadlessScenario.chunks_coverage`.
* **Fail-closed egress (§1).** :class:`EgressGuard` installs and asserts the
  socket block, the credential strip, the config/scratch redirection and the
  preserved ``PYTHONDONTWRITEBYTECODE`` flag. It is self-contained because the
  pytest fixtures do not cover bare-script runners or per-cell subprocesses.

Nothing here changes production behaviour: instrumentation is installed on
instances from the test side and removed on teardown.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
from time import perf_counter, perf_counter_ns
from typing import Any

from chartreux.app_server._projector import EventProjector
from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server.events import ClientProjection
from chartreux.app_server.models import (
    IdleSessionStatus,
    PublicHistoryEntry,
    PublicSession,
    PublicSessionState,
    TextContentBlock,
)
from chartreux.app_server.protocol import (
    AgentEvictionModel,
    AgentSummaryModel,
    AgentsUpdateParams,
    Notification,
    TurnStartParams,
)
from chartreux.app_server.transport import MemoryJsonRpcTransport, memory_transport_pair
from chartreux.core.agent_loop.llm_gateway import CompletionInputs, LLMGateway
from chartreux.core.llm_models import LLMChunk
from chartreux.core.message_list import (
    MessageList,
    RequestEnvelope,
    freeze_request_provenance,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.subagents import (
    AgentEviction,
    LaunchConfig,
    TaskArgs,
    TaskResult,
    WaitStatus,
)
from chartreux.core.tools.base import InvokeContext
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.perf._runtime_fixture import (
    FIXTURE_APPROX_TOKENS,
    FIXTURE_HASH,
    FIXTURE_MESSAGE_CHARS,
    FIXTURE_MESSAGE_COUNT,
    FIXTURE_RESPONSE_CHARS,
    FIXTURE_TURN_WIDTH,
    FIXTURE_TURNS,
    build_fixture,
    fixture_spec,
    response_content,
)
from tests.stubs.fake_backend import FakeBackend

CELL_FRESH_HEADLESS = "x1_fresh_headless_fb10_c4"
CELL_LOADED_HEADLESS = "x2_loaded_headless_8x200k"
SCENARIO_CELLS: tuple[str, ...] = (CELL_FRESH_HEADLESS, CELL_LOADED_HEADLESS)

#: The run record's metric version (Appendix B ``run_eligibility.metric_version``),
#: distinct from the determinants document's own schema field.
SCENARIO_METRIC_VERSION = "chartreux.runtime_perf.v1"
SCENARIO_SCHEMA = "chartreux.runtime_scenario.v1"

#: The loaded cell's binding agent dimension: 8 total agents, root + 7 children.
LOADED_TOTAL_AGENTS = 8
LOADED_CHILD_AGENTS = 7
FRESH_TOTAL_AGENTS = 1

#: X1's canonical injected-delay cell: fb = 10 ms plus 4 x 5 ms chunk gaps.
INJECTED_FIRST_BYTE_MS = 10.0
INJECTED_CHUNK_COUNT = 4
INJECTED_CHUNK_GAP_MS = 5.0

#: Deterministic workload provenance (committed with the fixture hash).
WORKLOAD_SEED = 20261108
ROOT_USER_CHARS = 400
RESPONSE_CHUNK_COUNT = INJECTED_CHUNK_COUNT
CHILD_TASK_CHARS = 200
#: The soak shape of plan §5: 400 synthetic root turns, checkpoints every 100.
ROOT_TURNS = FIXTURE_TURNS
WARMUP_TURNS = 10
BURST_INTERVAL_TURNS = 10
CHECKPOINT_TURNS: tuple[int, ...] = (0, 100, 200, 300, 400)

_CONTENT_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_WORKLOAD_STREAM_CHARS = 8_192
_ROOT_USER_PREFIX = "perf-root-{turn:04d} "
_CHILD_TASK_PREFIX = "perf-child-{index:02d} "
_USER_STRIDE = 733
_CHILD_STRIDE = 617
_NANOSECONDS_PER_MILLISECOND = 1_000_000

#: The provider credentials a measurement process must never carry.
PROVIDER_CREDENTIAL_NAMES: tuple[str, ...] = (
    "MISTRAL_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)

CONFIG_HOME_ENV = "CHARTREUX_HOME"
BYTECODE_ENV = "PYTHONDONTWRITEBYTECODE"
SCRATCHPAD_ENV = "CHARTREUX_PERF_SCRATCHPAD"
_EGRESS_MARKER = "provider network blocked by performance scenario egress guard"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "::", "0.0.0.0"})


def _seeded_stream() -> str:
    """Return the seeded content stream, doubled so any window fits in a slice."""
    generator = _RandomStream(WORKLOAD_SEED)
    single = "".join(
        _CONTENT_ALPHABET[generator.next_index()] for _ in range(_WORKLOAD_STREAM_CHARS)
    )
    return single + single


class _RandomStream:
    """A tiny deterministic byte stream for the committed prompt sizes.

    A local mixed-linear generator keeps the workload independent of any
    library's RNG implementation, so the committed sizes and the digest below
    cannot drift with a third-party version bump.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._index = 0

    def next_index(self) -> int:
        self._index += 1
        mixed = self._seed * 6_364_136_223_846_793_005 + self._index * 1_442_695_040
        return (mixed >> 33) % len(_CONTENT_ALPHABET)


@functools.lru_cache(maxsize=1)
def _cached_stream() -> str:
    return _seeded_stream()


def _seeded_text(prefix: str, offset: int, char_count: int) -> str:
    stream = _cached_stream()
    start = offset % _WORKLOAD_STREAM_CHARS
    return (prefix + stream[start : start + char_count])[:char_count]


def root_user_content(turn: int) -> str:
    """Return the committed user prompt of one synthetic root turn."""
    if turn < 0:
        raise ValueError("turn must be non-negative")
    return _seeded_text(
        _ROOT_USER_PREFIX.format(turn=turn), turn * _USER_STRIDE, ROOT_USER_CHARS
    )


def child_task_text(index: int) -> str:
    """Return the committed task text of one synchronized child agent."""
    if index < 0:
        raise ValueError("index must be non-negative")
    return _seeded_text(
        _CHILD_TASK_PREFIX.format(index=index), index * _CHILD_STRIDE, CHILD_TASK_CHARS
    )


def response_chunks(turn: int) -> tuple[str, ...]:
    """Split the committed deterministic response of ``turn`` into fixed chunks.

    The chunk count is the canonical cell's c4 schedule, so a streaming turn
    yields exactly the committed number of transport chunks and each chunk gap
    the A3 metric measures is one delivered chunk boundary.
    """
    content = response_content(turn)
    if len(content) != FIXTURE_RESPONSE_CHARS:
        raise ValueError("committed response shape changed")
    width = FIXTURE_RESPONSE_CHARS // RESPONSE_CHUNK_COUNT
    return tuple(
        content[index * width : (index + 1) * width]
        for index in range(RESPONSE_CHUNK_COUNT)
    )


def response_chunk_messages(turn: int) -> list[Any]:
    """Return the deterministic terminal streaming chunks of one root turn."""
    chunks = response_chunks(turn)
    return [
        mock_llm_chunk(
            content=text, stop_reason="stop" if index == len(chunks) - 1 else None
        )
        for index, text in enumerate(chunks)
    ]


@dataclass(frozen=True, slots=True)
class ScenarioScale:
    """How much of the committed workload one scenario run executes.

    The committed definitions are the full scale; in-suite contract nodes run
    the shortened scale (plan §5: external cells are opt-in, in-suite contract
    nodes use shortened fixtures) and record ``mode = contract`` so the run can
    never be mistaken for eligible baseline data.
    """

    name: str
    root_turns: int
    history_messages: int
    child_agents: int
    burst_agents: int
    burst_interval_turns: int
    mode: str

    def dimensions(self) -> dict[str, Any]:
        return {
            "root_turns": self.root_turns,
            "history_messages": self.history_messages,
            "child_agents": self.child_agents,
            "burst_agents": self.burst_agents,
            "burst_interval_turns": self.burst_interval_turns,
            "mode": self.mode,
        }


#: The committed full-scale shapes of the two headless cells.
FULL_SCALES: dict[str, ScenarioScale] = {
    CELL_FRESH_HEADLESS: ScenarioScale(
        name="full",
        root_turns=ROOT_TURNS,
        history_messages=FIXTURE_MESSAGE_COUNT,
        child_agents=0,
        burst_agents=0,
        burst_interval_turns=BURST_INTERVAL_TURNS,
        mode="measure",
    ),
    CELL_LOADED_HEADLESS: ScenarioScale(
        name="full",
        root_turns=ROOT_TURNS,
        history_messages=FIXTURE_MESSAGE_COUNT,
        child_agents=LOADED_CHILD_AGENTS,
        burst_agents=LOADED_CHILD_AGENTS,
        burst_interval_turns=BURST_INTERVAL_TURNS,
        mode="measure",
    ),
}

#: The in-suite shape: short enough for a <= 0.5 s contract node.
SHORTENED_SCALE = ScenarioScale(
    name="shortened",
    root_turns=3,
    history_messages=FIXTURE_TURN_WIDTH * 4,
    child_agents=1,
    burst_agents=1,
    burst_interval_turns=2,
    mode="contract",
)


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """One cell's committed workload definition.

    The seed, the message sizes and the deterministic response shape are part
    of the definition, so the digest below changes when any of them drifts —
    the same drift rule eligibility applies to the fixture hash.
    """

    cell: str
    adapter: str
    fixture_hash: str
    fixture_spec: dict[str, Any]
    seed: int
    history_messages: int
    history_tokens: int
    root_turns: int
    warmup_turns: int
    total_agents: int
    child_agents: int
    burst_interval_turns: int
    checkpoint_turns: tuple[int, ...]
    injected_first_byte_ms: float
    injected_chunk_count: int
    injected_chunk_gap_ms: float
    root_user_chars: int
    response_chars: int
    response_chunks: int
    notification_window: str = "turn_submission->projector_completed_state"

    @property
    def injected_turn_delay_ms(self) -> float:
        """The exact total injected delay of one canonical-cell turn."""
        return (
            self.injected_first_byte_ms
            + self.injected_chunk_count * self.injected_chunk_gap_ms
        )

    def record(self) -> dict[str, Any]:
        """Return the run-record workload block eligibility compares on."""
        return {
            "cell": self.cell,
            "adapter": self.adapter,
            "fixture_hash": self.fixture_hash,
            "seed": self.seed,
            "history_messages": self.history_messages,
            "history_tokens": self.history_tokens,
            "root_turns": self.root_turns,
            "warmup_turns": self.warmup_turns,
            "total_agents": self.total_agents,
            "child_agents": self.child_agents,
            "burst_interval_turns": self.burst_interval_turns,
            "checkpoint_turns": list(self.checkpoint_turns),
            "injected_first_byte_ms": self.injected_first_byte_ms,
            "injected_chunk_count": self.injected_chunk_count,
            "injected_chunk_gap_ms": self.injected_chunk_gap_ms,
            "injected_turn_delay_ms": self.injected_turn_delay_ms,
            "root_user_chars": self.root_user_chars,
            "response_chars": self.response_chars,
            "response_chunks": self.response_chunks,
            "notification_window": self.notification_window,
        }

    def digest(self) -> str:
        """Return the sha256 digest of the committed workload definition."""
        encoded = json.dumps(
            self.record(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(f"{SCENARIO_SCHEMA}|{self.fixture_hash}|".encode())
        digest.update(encoded)
        return digest.hexdigest()


def workload_spec(cell: str) -> WorkloadSpec:
    """Return the committed workload definition of one headless cell."""
    if cell == CELL_FRESH_HEADLESS:
        total_agents, child_agents = FRESH_TOTAL_AGENTS, 0
        injected = (INJECTED_FIRST_BYTE_MS, INJECTED_CHUNK_COUNT, INJECTED_CHUNK_GAP_MS)
    elif cell == CELL_LOADED_HEADLESS:
        total_agents, child_agents = LOADED_TOTAL_AGENTS, LOADED_CHILD_AGENTS
        # The loaded cell carries no injected delay: the load is the fixture.
        injected = (0.0, 0, 0.0)
    else:
        raise ValueError(f"unknown scenario cell: {cell}")
    return WorkloadSpec(
        cell=cell,
        adapter="headless",
        fixture_hash=FIXTURE_HASH,
        fixture_spec=fixture_spec(),
        seed=WORKLOAD_SEED,
        history_messages=FIXTURE_MESSAGE_COUNT,
        history_tokens=FIXTURE_APPROX_TOKENS,
        root_turns=ROOT_TURNS,
        warmup_turns=WARMUP_TURNS,
        total_agents=total_agents,
        child_agents=child_agents,
        burst_interval_turns=BURST_INTERVAL_TURNS,
        checkpoint_turns=CHECKPOINT_TURNS,
        injected_first_byte_ms=injected[0],
        injected_chunk_count=injected[1],
        injected_chunk_gap_ms=injected[2],
        root_user_chars=ROOT_USER_CHARS,
        response_chars=FIXTURE_RESPONSE_CHARS,
        response_chunks=RESPONSE_CHUNK_COUNT,
    )


#: Committed workload digests, one per headless cell. A digest that no longer
#: matches this table is fixture drift and makes the cell ineligible.
WORKLOAD_HASHES: dict[str, str] = {
    CELL_FRESH_HEADLESS: "379a228d128b90aa8d73d14d6307f35dd36ccb60eb21a37b0f7865d29a8326b2",
    CELL_LOADED_HEADLESS: "46ac0283ff4d52d119ad70ab1c5cb3d9240beda31658cff478f5569dcc250ac3",
}


# --------------------------------------------------------------------------
# Fail-closed egress
# --------------------------------------------------------------------------


def _is_loopback(address: Any) -> bool:
    if not isinstance(address, tuple):
        return False
    if len(address) != 2:
        raise OSError(_EGRESS_MARKER)
    host = address[0]
    if not isinstance(host, str):
        return False
    if host in _LOOPBACK_HOSTS:
        return True
    try:
        return host.startswith("127.") or host == "::1"
    except AttributeError:  # pragma: no cover - host is always a str here
        return False


class EgressGuard:
    """Install and assert the measurement process's fail-closed egress policy.

    Four properties are enforced for the whole scenario, including for code that
    never runs under the pytest fixtures (a bare-script runner, a per-cell
    subprocess): non-loopback sockets are refused at the transport boundary,
    provider credentials are absent from the environment, the config/scratchpad
    homes point outside the repository, and bytecode writing stays disabled.
    :meth:`assert_closed` re-checks each property live, so a silently vanished
    guard fails the run instead of degrading it.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root is not None else None
        self._config_home: Path | None = None
        self._scratch_home: Path | None = None
        self._original_connect: Callable[[socket.socket, Any], Any] | None = None
        self._original_connect_ex: Callable[[socket.socket, Any], Any] | None = None
        self._original_create_connection: Callable[..., socket.socket] | None = None
        self._environment: dict[str, str | None] = {}
        self._installed = False

    @property
    def root(self) -> Path:
        if self._root is None:
            raise RuntimeError("the egress guard has no root directory")
        return self._root

    @property
    def config_home(self) -> Path:
        if self._config_home is None:
            raise RuntimeError("the egress guard is not installed")
        return self._config_home

    @property
    def scratch_home(self) -> Path:
        if self._scratch_home is None:
            raise RuntimeError("the egress guard is not installed")
        return self._scratch_home

    def install(self, *, scratch_root: Path | None = None) -> EgressGuard:
        """Install the policy and return ``self``."""
        if self._installed:
            raise RuntimeError("the egress guard is already installed")
        if self._root is None:
            self._root = Path(tempfile.mkdtemp(prefix="chartreux-perf-egress-"))
        self._root.mkdir(parents=True, exist_ok=True)
        self._config_home = self._root / "config"
        self._scratch_home = (
            Path(scratch_root) if scratch_root is not None else self._root / "scratch"
        )
        self._config_home.mkdir(parents=True, exist_ok=True)
        self._scratch_home.mkdir(parents=True, exist_ok=True)

        self._original_connect = socket.socket.connect
        self._original_connect_ex = socket.socket.connect_ex
        self._original_create_connection = socket.create_connection
        original_connect = self._original_connect
        original_connect_ex = self._original_connect_ex

        def blocked_connect(sock: socket.socket, address: Any) -> Any:
            if _is_loopback(address):
                return original_connect(sock, address)
            raise OSError(_EGRESS_MARKER)

        def blocked_connect_ex(sock: socket.socket, address: Any) -> Any:
            if _is_loopback(address):
                return original_connect_ex(sock, address)
            raise OSError(_EGRESS_MARKER)

        def blocked_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
            if _is_loopback(address):
                return self._original_create_connection(address, *args, **kwargs)  # type: ignore[misc]
            raise OSError(_EGRESS_MARKER)

        socket.socket.connect = blocked_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = blocked_connect_ex  # type: ignore[method-assign]
        socket.create_connection = blocked_create_connection  # type: ignore[assignment]

        for name in PROVIDER_CREDENTIAL_NAMES:
            self._environment[name] = os.environ.pop(name, None)
        for name, value in (
            (CONFIG_HOME_ENV, os.environ.get(CONFIG_HOME_ENV)),
            (SCRATCHPAD_ENV, os.environ.get(SCRATCHPAD_ENV)),
            (BYTECODE_ENV, os.environ.get(BYTECODE_ENV)),
        ):
            self._environment.setdefault(name, value)
        os.environ[CONFIG_HOME_ENV] = str(self._config_home)
        os.environ[SCRATCHPAD_ENV] = str(self._scratch_home)
        os.environ[BYTECODE_ENV] = "1"
        self._installed = True
        return self

    def assert_closed(self) -> None:
        """Re-check every enforced property; raise on the first violation."""
        if not self._installed:
            raise RuntimeError("the egress guard is not installed")
        loopback = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # A loopback connect attempt must reach the socket layer, so the
            # block below is proven to be address-selective, not a blanket kill.
            try:
                loopback.connect(("127.0.0.1", 9))
            except OSError as error:
                assert _EGRESS_MARKER not in str(error), "the guard blocked loopback"
        finally:
            loopback.close()
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            try:
                probe.connect(("192.0.2.1", 80))
            except OSError as error:
                assert _EGRESS_MARKER in str(error), str(error)
            else:
                raise AssertionError("a non-loopback connect was not blocked")
        finally:
            probe.close()
        try:
            socket.create_connection(("192.0.2.1", 80), timeout=0.01)
        except OSError as error:
            assert _EGRESS_MARKER in str(error), str(error)
        else:
            raise AssertionError("create_connection was not blocked")
        present = [name for name in PROVIDER_CREDENTIAL_NAMES if name in os.environ]
        assert not present, f"provider credentials leaked into the run: {present}"
        config_home = Path(os.environ[CONFIG_HOME_ENV])
        scratch_home = Path(os.environ[SCRATCHPAD_ENV])
        repository = Path(__file__).resolve().parents[2]
        for home in (config_home, scratch_home):
            assert home.is_dir(), f"{home} is not a directory"
            assert home != repository and repository not in home.parents, (
                f"{home} is inside the repository tree"
            )
        assert os.environ.get(BYTECODE_ENV) == "1", "bytecode writing is not disabled"

    def restore(self) -> None:
        """Undo every change :meth:`install` made."""
        if not self._installed:
            return
        if self._original_connect is not None:
            socket.socket.connect = self._original_connect  # type: ignore[method-assign]
        if self._original_connect_ex is not None:
            socket.socket.connect_ex = self._original_connect_ex  # type: ignore[method-assign]
        if self._original_create_connection is not None:
            socket.create_connection = self._original_create_connection
        for name, value in self._environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._environment.clear()
        self._installed = False

    def __enter__(self) -> EgressGuard:
        return self.install()

    def __exit__(self, *exc_info: object) -> None:
        self.restore()


# --------------------------------------------------------------------------
# Notification transport and the per-turn window
# --------------------------------------------------------------------------


@dataclass(slots=True)
class NotificationWindow:
    """One root turn's server-to-client notification window.

    The window opens at the root turn's user-message submission and closes at
    the projector's completed-state event for that turn (Appendix B B3/B4), so
    every notification counted here is one a real client would serialize.
    """

    kind: str
    turn_id: str | None
    opened_ns: int
    closed_ns: int | None = None
    methods: list[str] = field(default_factory=list)
    wire_bytes: int = 0

    @property
    def count(self) -> int:
        return len(self.methods)

    @property
    def open(self) -> bool:
        return self.closed_ns is None

    @property
    def duration_ms(self) -> float | None:
        if self.closed_ns is None:
            return None
        return (self.closed_ns - self.opened_ns) / _NANOSECONDS_PER_MILLISECOND

    def record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "turn_id": self.turn_id,
            "notifications": self.count,
            "notification_bytes": self.wire_bytes,
            "methods": list(self.methods),
            "duration_ms": (
                None if self.duration_ms is None else round(self.duration_ms, 3)
            ),
            "open": self.open,
        }


@dataclass(frozen=True, slots=True)
class SerializedNotification:
    """One notification as serialized onto the wire, measured exactly once."""

    method: str
    session_id: str | None
    wire_bytes: int


@dataclass(frozen=True, slots=True)
class ProjectorSnapshot:
    """The live projector's state at the close of one root-turn window."""

    live: bool
    history_ids: tuple[str, ...]
    history_fingerprint: str
    index_consistent: bool
    index_size: int


def projector_snapshot(projector: EventProjector | None) -> ProjectorSnapshot:
    """Snapshot a live projector's history and its maintained id-to-position index."""
    if projector is None:
        return ProjectorSnapshot(
            live=False,
            history_ids=(),
            history_fingerprint=projected_history_fingerprint(()),
            index_consistent=False,
            index_size=0,
        )
    positions = projector._entry_positions
    consistent = positions == {
        entry.id: position for position, entry in enumerate(projector.history)
    }
    return ProjectorSnapshot(
        live=True,
        history_ids=tuple(entry.id for entry in projector.history),
        history_fingerprint=projected_history_fingerprint(projector.history),
        index_consistent=consistent,
        index_size=len(positions),
    )


async def _noop_deliver_callback(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


class NotificationSink:
    """The scenario's server-to-client notification boundary.

    Production's notify surface is ``(method, params)``; the sink serializes the
    params model EXACTLY ONCE (the wire form the transport carries), hands the
    same envelope dict to the real memory transport, records the count and the
    UTF-8 byte width for the run's chunks coverage, and feeds the client
    projection so the client side of the P1 contract runs for real.
    """

    def __init__(
        self,
        transport: MemoryJsonRpcTransport,
        projection: ClientProjection,
        *,
        session_id: str,
    ) -> None:
        self._transport = transport
        self._projection = projection
        self._session_id = session_id
        self._event_ids: dict[str | None, int] = {}
        self.serialized: list[SerializedNotification] = []
        self.windows: list[NotificationWindow] = []
        self._open: NotificationWindow | None = None
        #: Invoked with the closing window before it is released, so a scenario
        #: can snapshot state that only exists while the projector is live.
        self.on_window_close: Callable[[NotificationWindow], None] | None = None

    @property
    def event_id(self) -> int:
        return self._event_ids.get(self._session_id, 0)

    @property
    def current_window(self) -> NotificationWindow | None:
        return self._open

    def open_window(self, kind: str, turn_id: str | None) -> NotificationWindow:
        if self._open is not None:
            raise RuntimeError(f"a {self._open.kind} window is already open")
        window = NotificationWindow(
            kind=kind, turn_id=turn_id, opened_ns=perf_counter_ns()
        )
        self.windows.append(window)
        self._open = window
        return window

    def close_window(self) -> NotificationWindow:
        window = self._open
        if window is None:
            raise RuntimeError("no notification window is open")
        window.closed_ns = perf_counter_ns()
        self._open = None
        if self.on_window_close is not None:
            self.on_window_close(window)
        return window

    async def __call__(self, method: str, params: Any) -> None:
        payload = (
            params.model_dump(mode="json", by_alias=True)
            if hasattr(params, "model_dump")
            else dict(params)
        )
        session = payload.get("sessionId")
        # Event ids are PER SESSION (the registry's watermark callback is
        # session-keyed), so a child notification never leaves a gap in the
        # root stream the client projection enforces.
        next_id = self._event_ids.get(session, 0) + 1
        self._event_ids[session] = next_id
        payload["eventId"] = next_id
        envelope = {"jsonrpc": "2.0", "method": method, "params": payload}
        wire_bytes = len(json.dumps(envelope, separators=(",", ":")).encode("utf-8"))
        self.serialized.append(
            SerializedNotification(
                method=method, session_id=session, wire_bytes=wire_bytes
            )
        )
        window = self._open
        if window is not None:
            window.methods.append(method)
            window.wire_bytes += wire_bytes
        if payload.get("sessionId") == self._session_id:
            self._projection.consume(Notification(method=method, params=payload))
        await self._transport.send(envelope)
        if window is not None and _is_completed_state_event(method, payload):
            self.close_window()


def _is_completed_state_event(method: str, payload: dict[str, Any]) -> bool:
    """Whether one serialized update is the projector's completed-state event."""
    if method != "history/entryUpdated":
        return False
    for operation in payload.get("patch") or ():
        if (
            operation.get("path") == "/generationStatus"
            and operation.get("value") == "completed"
        ):
            return True
    return False


class _CollectingClient:
    """The client end of the real transport pair: drain and record deliveries."""

    def __init__(self, transport: MemoryJsonRpcTransport) -> None:
        self._transport = transport
        self.delivered: list[dict[str, Any]] = []
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._drain(), name="perf-transport-drain")

    async def _drain(self) -> None:
        async for message in self._transport.messages():
            self.delivered.append(message)

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._task = None

    async def settle(self, expected: int, *, timeout: float = 5.0) -> None:
        """Wait until the client end has received ``expected`` messages."""
        deadline = perf_counter() + timeout
        while len(self.delivered) < expected:
            if perf_counter() > deadline:
                raise TimeoutError(
                    f"client received {len(self.delivered)} of {expected} notifications"
                )
            await asyncio.sleep(0)


# --------------------------------------------------------------------------
# Topology and P1/P2 provenance
# --------------------------------------------------------------------------


def topology_fingerprint(
    *,
    cell: str,
    projector_live: bool,
    total_agents: int,
    transport_kind: str = "memory_json_rpc",
    event_loop_role: str = "single_test_loop",
    registry_construction_path: str = "SessionRuntimeRegistry._build_child_runtime",
    adapter: str = "headless",
) -> dict[str, Any]:
    """Return the STABLE STRUCTURAL labels of one scenario topology.

    Only labels that survive a process boundary belong here; object identity is
    asserted inside the measuring process instead (r12 R1-08/R2-09/R3-09).
    """
    return {
        "adapter": adapter,
        "cell": cell,
        "event_loop_role": event_loop_role,
        "registry_construction_path": registry_construction_path,
        "projector_live": projector_live,
        "transport": transport_kind,
        "total_agents": total_agents,
    }


def assert_fingerprint_equality(left: dict[str, Any], right: dict[str, Any]) -> None:
    """Assert two topology fingerprints describe the same structure."""
    if left != right:
        differing = sorted(
            key for key in left.keys() | right.keys() if left.get(key) != right.get(key)
        )
        raise AssertionError(f"topology fingerprints differ on {differing}")


def projected_history_fingerprint(entries: Sequence[PublicHistoryEntry]) -> str:
    """Return a stable digest of a projected history (ids, types, statuses)."""
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry.id}|{type(entry).__name__}|{entry.generation_status}|".encode()
        )
    return digest.hexdigest()


@dataclass(slots=True)
class DeltaDumpLog:
    """Counts of the projector/client per-delta serializations (the P1 pin)."""

    per_entry: dict[str, int] = field(default_factory=dict)
    shared_raw_calls: int = 0
    unshared_raw_calls: int = 0

    def reset(self) -> None:
        self.per_entry.clear()
        self.shared_raw_calls = 0
        self.unshared_raw_calls = 0

    def dumps(self, entry_id: str) -> int:
        return self.per_entry.get(entry_id, 0)


@contextmanager
def instrument_delta_dumps(entry_type: type[Any]) -> Any:
    """Count ``entry_type.model_dump`` calls and whether predicates got ``raw``.

    The plan's P1 pin is that one delta serializes each side's entry ONCE and
    the predicates share that snapshot. This instruments the real
    ``_patch``/``_update_entry`` path from the test side; nothing here assumes a
    per-predicate dump, which is exactly the assumption the pin forbids. The
    predicates are patched at every importing module, so the pin is measured on
    the references the projector and the client actually call.
    """
    from chartreux.app_server import (
        _patch as patch_module,
        _projector as projector_module,
        events as events_module,
    )

    log = DeltaDumpLog()
    original_dump = entry_type.model_dump
    originals = {
        (module, name): getattr(module, name)
        for module in (patch_module, projector_module, events_module)
        for name in ("is_timing_only_patch", "is_completed_effect_replay")
        if hasattr(module, name)
    }

    @functools.wraps(original_dump)
    def counting_dump(self: Any, *args: Any, **kwargs: Any) -> Any:
        entry_id = getattr(self, "id", None)
        if isinstance(entry_id, str):
            log.per_entry[entry_id] = log.per_entry.get(entry_id, 0) + 1
        return original_dump(self, *args, **kwargs)

    def instrumented(name: str) -> Any:
        original = getattr(patch_module, name)

        def counted(*args: Any, **kwargs: Any) -> bool:
            if kwargs.get("raw") is None:
                log.unshared_raw_calls += 1
            else:
                log.shared_raw_calls += 1
            return original(*args, **kwargs)

        return counted

    entry_type.model_dump = counting_dump  # type: ignore[method-assign]
    for (module, name), _previous in originals.items():
        setattr(module, name, instrumented(name))
    try:
        yield log
    finally:
        entry_type.model_dump = original_dump  # type: ignore[method-assign]
        for (module, name), previous in originals.items():
            setattr(module, name, previous)


@dataclass(slots=True)
class BackendRequestContext:
    """The request context one root turn actually handed the backend."""

    request_count: int
    message_count: int
    context_chars: int
    memo_present: bool
    memo_message_count: int
    memo_matches_request: bool


# --------------------------------------------------------------------------
# Workload backends
# --------------------------------------------------------------------------


class InjectedDelayBackend(FakeBackend):
    """A deterministic backend that reproduces the canonical injected delays.

    The first byte is held for ``first_byte_ms`` and every later chunk for
    ``chunk_gap_ms``; the total injected delay of one turn is therefore known
    exactly and is recorded per call for the A1 residual.
    """

    def __init__(
        self,
        chunks: list[Any] | None = None,
        *,
        first_byte_ms: float = 0.0,
        chunk_gap_ms: float = 0.0,
    ) -> None:
        super().__init__(chunks)
        self.first_byte_ms = first_byte_ms
        self.chunk_gap_ms = chunk_gap_ms
        self.injected_ms: list[float] = []

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        total = self.first_byte_ms
        self.injected_ms.append(total)
        await asyncio.sleep(self.first_byte_ms / 1000)
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk
            if self.chunk_gap_ms:
                total += self.chunk_gap_ms
                self.injected_ms[-1] = total
                await asyncio.sleep(self.chunk_gap_ms / 1000)


class RootTurnBackend(InjectedDelayBackend):
    """The root session's deterministic backend for the synthetic turns."""

    def __init__(self, *, first_byte_ms: float, chunk_gap_ms: float) -> None:
        super().__init__(None, first_byte_ms=first_byte_ms, chunk_gap_ms=chunk_gap_ms)
        self.turn = 0

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        self._streams.append(response_chunk_messages(self.turn))
        self.turn += 1
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk

    async def complete(self, **kwargs: Any) -> Any:
        self._streams.append(response_chunk_messages(self.turn))
        self.turn += 1
        return await super().complete(**kwargs)


class GatedChildBackend(FakeBackend):
    """Children block on a shared barrier so one burst completes together."""

    def __init__(self, *, agents: int) -> None:
        super().__init__()
        self._agents = agents
        self.started: list[asyncio.Event] = []
        self.releases: list[asyncio.Event] = []
        self._next_gate_index = 0
        self.context_chars: list[int] = []
        self.completed = 0
        self._reset_gates()

    def _reset_gates(self) -> None:
        self.started = [asyncio.Event() for _ in range(self._agents)]
        self.releases = [asyncio.Event() for _ in range(self._agents)]

    def prepare(self) -> None:
        """Open one fresh barrier cohort for the next burst."""
        self._next_gate_index = 0
        self.context_chars.clear()
        self._reset_gates()

    async def complete(self, *, messages: list[Any], **kwargs: Any) -> Any:
        gate_index = self._next_gate_index
        self._next_gate_index += 1
        if gate_index >= len(self.started):
            raise AssertionError(f"no gate prepared for backend call {gate_index}")
        self.context_chars.append(
            sum(
                len(message.content)
                for message in messages
                if isinstance(getattr(message, "content", None), str)
            )
        )
        self.started[gate_index].set()
        await self.releases[gate_index].wait()
        self._streams.append([mock_llm_chunk(content="child done")])
        self.completed += 1
        return await super().complete(messages=messages, **kwargs)


@contextmanager
def _patched_create_backend(backend: FakeBackend) -> Any:
    """Swap the loop-level backend factory for one deterministic backend."""
    module = llm_gateway_module()
    original = module.create_backend

    def factory(**_kwargs: Any) -> FakeBackend:
        return backend

    module.create_backend = factory
    try:
        yield backend
    finally:
        module.create_backend = original


def llm_gateway_module() -> Any:
    """Return the module whose ``create_backend`` the loop resolves at call time."""
    from chartreux.core.agent_loop import _loop as loop_module

    return loop_module


# --------------------------------------------------------------------------
# The scenario
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RootTurnResult:
    """One measured root turn: window, latency and injected delay."""

    turn: int
    turn_id: str
    status: str
    wall_ms: float
    injected_delay_ms: float
    window: NotificationWindow
    history_messages: int

    def record(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "turn_id": self.turn_id,
            "status": self.status,
            "wall_ms": round(self.wall_ms, 3),
            "injected_delay_ms": self.injected_delay_ms,
            "application_overhead_ms": round(self.wall_ms - self.injected_delay_ms, 3),
            "window": self.window.record(),
            "history_messages": self.history_messages,
        }


@dataclass(frozen=True, slots=True)
class ChildBurstResult:
    """One synchronized child burst: cohort size, completion boundary, wall."""

    burst: int
    agent_count: int
    completed: int
    wall_ms: float
    context_chars: list[int]

    def record(self) -> dict[str, Any]:
        return {
            "burst": self.burst,
            "agent_count": self.agent_count,
            "completed": self.completed,
            "wall_ms": round(self.wall_ms, 3),
            "context_chars": list(self.context_chars),
        }


class HeadlessScenario:
    """A real headless app-server scenario over one cell's committed workload."""

    def __init__(
        self,
        *,
        cell: str,
        spec: WorkloadSpec,
        scale: ScenarioScale,
        registry: SessionRuntimeRegistry,
        root_runtime: Any,
        root_loop: Any,
        sink: NotificationSink,
        projection: ClientProjection,
        client_end: _CollectingClient,
        client_transport: MemoryJsonRpcTransport,
        server_transport: MemoryJsonRpcTransport,
        backend: RootTurnBackend,
        model_names: list[str],
        child_session_dirs: list[Path],
    ) -> None:
        self.cell = cell
        self.spec = spec
        self.scale = scale
        self.registry = registry
        self.root_runtime = root_runtime
        self.root_loop = root_loop
        self.sink = sink
        self.projection = projection
        self._client_end = client_end
        self._client_transport = client_transport
        self._server_transport = server_transport
        self._backend = backend
        self.model_names = model_names
        self.child_session_dirs = child_session_dirs
        self.turns: list[RootTurnResult] = []
        self.bursts: list[ChildBurstResult] = []
        self.projector_snapshots: list[ProjectorSnapshot] = []
        self._burst_backend = GatedChildBackend(agents=max(scale.burst_agents, 1))
        self._closed = False
        self.sink.on_window_close = self._capture_projector

    def _capture_projector(self, window: NotificationWindow) -> None:
        """Snapshot the still-live projector as the turn's window closes."""
        del window
        self.projector_snapshots.append(projector_snapshot(self.projector))

    # -- topology ---------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self.root_loop.session_id

    @property
    def projector(self) -> EventProjector | None:
        """The live projector of the running turn, or ``None`` between turns."""
        return self.root_runtime.turns._projector

    def topology(self, *, projector_live: bool | None = None) -> dict[str, Any]:
        return topology_fingerprint(
            cell=self.cell,
            projector_live=(
                self.projector is not None if projector_live is None else projector_live
            ),
            total_agents=(
                self.scale.child_agents + 1
                if self.scale.child_agents
                else FRESH_TOTAL_AGENTS
            ),
        )

    @property
    def transports(self) -> tuple[MemoryJsonRpcTransport, MemoryJsonRpcTransport]:
        return (self._client_transport, self._server_transport)

    # -- workload ---------------------------------------------------------

    async def run_root_turn(self, turn: int | None = None) -> RootTurnResult:
        """Submit one deterministic root turn and wait for its completion."""
        index = len(self.turns) if turn is None else turn
        # A child cohort's completion publishes an injected root turn of its
        # own; the workload's schedule is deterministic only once that lands.
        await self._await_root_idle()
        window = self.sink.open_window("root_turn", None)
        params = TurnStartParams(
            session_id=self.session_id,
            message=[TextContentBlock(text=root_user_content(index))],
        )
        response, action = self.root_runtime.turns.start(params)
        window.turn_id = response.turn.id
        projector_live = self.projector is not None
        if not projector_live:
            raise AssertionError("the root turn started without a live projector")
        action()
        terminal = await asyncio.wait_for(
            self.root_runtime.turns.wait_for_operation(response.turn.id), timeout=60
        )
        if self.sink.current_window is not None:
            raise AssertionError("the turn completed without its completed-state event")
        # The terminal operation future can resolve a moment before the turn
        # task itself retires; the next turn's admission checks that task.
        await self._await_root_idle()
        wall_ms = window.duration_ms
        if wall_ms is None:
            raise AssertionError("the notification window never closed")
        injected = self.spec.injected_turn_delay_ms
        result = RootTurnResult(
            turn=index,
            turn_id=response.turn.id,
            status=str(terminal.status),
            wall_ms=wall_ms,
            injected_delay_ms=injected,
            window=window,
            history_messages=len(self.root_loop.messages),
        )
        self.turns.append(result)
        return result

    async def run_child_burst(self, burst: int | None = None) -> ChildBurstResult:
        """Release one synchronized child cohort and wait for its completion."""
        if not self.scale.child_agents:
            raise RuntimeError("this cell has no child agents")
        index = len(self.bursts) if burst is None else burst
        backend = self._burst_backend
        backend.prepare()
        started_at = perf_counter()
        with _patched_create_backend(backend):
            launches = await asyncio.wait_for(
                self._launch_children(self.scale.burst_agents), timeout=60
            )
            await asyncio.wait_for(
                asyncio.gather(*(gate.wait() for gate in backend.started)), timeout=60
            )
            for release in backend.releases:
                release.set()
            completions = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        self.registry.wait_for_agent(agent_id, run_id, timeout=60)
                        for result in launches
                        if (agent_id := result.agent_id) is not None
                        and (run_id := result.run_id) is not None
                    )
                ),
                timeout=60,
            )
        wall_ms = (perf_counter() - started_at) * 1000
        completed = sum(
            1
            for completion in completions
            if completion.status is WaitStatus.COMPLETED
            and completion.result is not None
            and completion.result.completed
        )
        if completed != self.scale.burst_agents:
            raise AssertionError(
                f"only {completed} of {self.scale.burst_agents} children completed"
            )
        # The burst's completion boundary includes the cohort's teardown: a
        # child turn that is still publishing must never outlive the boundary.
        await self._settle_child_closes()
        result = ChildBurstResult(
            burst=index,
            agent_count=self.scale.burst_agents,
            completed=completed,
            wall_ms=wall_ms,
            context_chars=list(backend.context_chars),
        )
        self.bursts.append(result)
        return result

    async def _launch_children(self, agent_count: int) -> list[TaskResult]:
        context = InvokeContext(
            tool_call_id=f"perf-burst-{len(self.bursts)}", session_id=self.session_id
        )
        launches: list[TaskResult] = []
        for index in range(agent_count):
            args = TaskArgs(
                task=child_task_text(index),
                background=True,
                config=LaunchConfig(model=self.model_names[index]),
            )
            task_context = InvokeContext(
                tool_call_id=f"{context.tool_call_id}-{index}",
                session_id=self.session_id,
            )
            result: TaskResult | None = None
            async for event in self.registry.run(args, task_context):
                if isinstance(event, TaskResult):
                    result = event
            if result is None:
                raise AssertionError("a child launch produced no task result")
            launches.append(result)
        return launches

    async def run_workload(self) -> list[RootTurnResult]:
        """Run the shortened or full workload of this scenario."""
        for index in range(self.scale.root_turns):
            if (
                self.scale.child_agents
                and index
                and index % self.scale.burst_interval_turns == 0
            ):
                await self.run_child_burst(index // self.scale.burst_interval_turns - 1)
            await self.run_root_turn(index)
        return list(self.turns)

    # -- P2 request context ----------------------------------------------

    def backend_request_context(self) -> BackendRequestContext:
        """Report the request the backend received and its P2 memo provenance.

        The backend copies the list it is handed (as a real backend would build
        its payload), so identity is compared per message: the recorded request
        must carry the memo's own message objects, which is what makes the
        repeat-attempt prep cost zero (plan §2 P2).
        """
        requests = self._backend.requests_messages
        if not requests:
            raise AssertionError("the backend received no request")
        latest = requests[-1]
        gateway: LLMGateway = self.root_loop._llm_gateway
        slot = gateway._prepared_slot
        memo = None if slot is None else slot[1]
        contents = [getattr(message, "content", None) for message in latest]
        return BackendRequestContext(
            request_count=len(requests),
            message_count=len(latest),
            context_chars=sum(
                len(content) for content in contents if isinstance(content, str)
            ),
            memo_present=memo is not None,
            memo_message_count=0 if memo is None else len(memo),
            memo_matches_request=(
                memo is not None
                and [id(message) for message in latest]
                == [id(message) for message in memo]
            ),
        )

    async def settle_transport(self, *, timeout: float = 5.0) -> None:
        """Wait until the client end of the real transport pair has everything."""
        await self._client_end.settle(len(self.sink.serialized), timeout=timeout)

    @property
    def delivered_notifications(self) -> list[dict[str, Any]]:
        """The notifications the client end of the real transport pair received."""
        return self._client_end.delivered

    def prepared_messages_identity_probe(self) -> tuple[int, int, int]:
        """Exercise the memo directly: hit identity, then the live-version miss.

        Returns the ids of (first preparation, repeated hit, post-mutation
        preparation) so a cell can assert the memo hit is an identity hit and
        the live-version re-read invalidates it.
        """
        messages = MessageList(list(self.root_loop.messages))
        model = self.root_loop.config.get_active_model()
        frozen, envelope = freeze_request_provenance(
            messages, model_alias=model.alias, supports_images=model.supports_images
        )
        if envelope is None:
            raise AssertionError("a MessageList must carry request provenance")
        inputs = self._completion_inputs(frozen, messages, envelope)
        gateway: LLMGateway = self.root_loop._llm_gateway
        first = gateway._prepare_backend_messages(inputs)
        hit = gateway._prepare_backend_messages(inputs)
        messages.append(mock_llm_chunk(content="mutation").message)
        live, live_envelope = freeze_request_provenance(
            messages,
            model_alias=inputs.model.alias,
            supports_images=inputs.model.supports_images,
        )
        if live_envelope is None:
            raise AssertionError("provenance disappeared after the mutation")
        mutated = gateway._prepare_backend_messages(
            self._completion_inputs(live, messages, live_envelope)
        )
        return (id(first), id(hit), id(mutated))

    def _completion_inputs(
        self, frozen: Sequence[Any], source: MessageList, envelope: RequestEnvelope
    ) -> CompletionInputs:
        model = self.root_loop.config.get_active_model()
        return CompletionInputs(
            model=model,
            provider_name="test-perf",
            emits_finish_reason=True,
            messages=tuple(frozen),
            tools=None,
            tool_choice=None,
            extra_headers={},
            metadata={},
            max_tokens=None,
            live_source=live_request_source(source, envelope),
        )

    # -- records ----------------------------------------------------------

    def chunks_coverage(self) -> dict[str, Any]:
        """Return the run's chunks-coverage fields (plan §3)."""
        measured = [turn for turn in self.turns if turn.turn >= self.spec.warmup_turns]
        notifications = sum(len(turn.window.methods) for turn in self.turns)
        notification_bytes = sum(turn.window.wire_bytes for turn in self.turns)
        missing: list[str] = []
        if not self.turns:
            missing.append("root_turns")
        if self.scale.child_agents and not self.bursts:
            missing.append("child_bursts")
        return {
            "unit": "root turns",
            "root_turns": len(self.turns),
            "warmup_root_turns": len(self.turns) - len(measured),
            "measured_root_turns": len(measured),
            "notification_windows": len(self.turns),
            "notifications": notifications,
            "notification_bytes": notification_bytes,
            "notifications_per_root_turn": [
                len(turn.window.methods) for turn in self.turns
            ],
            "notification_bytes_per_root_turn": [
                turn.window.wire_bytes for turn in self.turns
            ],
            "child_bursts": len(self.bursts),
            "child_turns": sum(burst.completed for burst in self.bursts),
            "complete": not missing,
            "missing": missing,
        }

    def run_record(self) -> dict[str, Any]:
        """Return the scenario run record (workload, coverage, turn samples)."""
        return {
            "schema": SCENARIO_SCHEMA,
            "metric_version": SCENARIO_METRIC_VERSION,
            "cell": self.cell,
            "scale": self.scale.dimensions(),
            "workload": self.spec.record(),
            "workload_hash": self.spec.digest(),
            "fixture_hash": self.spec.fixture_hash,
            "topology": self.topology(projector_live=True),
            "chunks_coverage": self.chunks_coverage(),
            "root_turns": [turn.record() for turn in self.turns],
            "child_bursts": [burst.record() for burst in self.bursts],
        }

    # -- teardown ---------------------------------------------------------

    async def close(self) -> None:
        """Drain the retained children and release every transport and loop."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._settle_child_closes()
            await self.registry.drain_children()
            await self._settle_child_closes()
        finally:
            # Close the runtimes while the transport is still alive: a turn
            # cancellation publishes its terminal state on the way out.
            await self.root_runtime.close()
            await self._client_end.stop()
            await self._client_transport.close()
            await self._server_transport.close()

    async def _settle_child_closes(self) -> None:
        """Await the cohort teardown tasks the registry still has in flight."""
        pending = [
            task for _runtime, task in self.registry._pending_child_closes.values()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _await_root_idle(self, *, timeout: float = 60.0) -> None:
        """Wait for the root runtime to be genuinely idle.

        Two things keep it busy after a turn: the turn task retires a moment
        after its terminal future resolves, and a background child's completion
        publishes its own injected root turn. Both are production behaviour, so
        the workload waits for them instead of racing them.
        """
        deadline = perf_counter() + timeout
        while perf_counter() < deadline:
            task = self.root_runtime.turns._active_task
            if task is not None and not task.done():
                await asyncio.gather(task, return_exceptions=True)
                continue
            pending = [
                candidate
                for candidate in self.registry._teardown_tasks
                if not candidate.done()
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                continue
            turns = self.root_runtime.turns
            if turns._active_task is None and turns.active_turn is None:
                return
            await asyncio.sleep(0)
        raise TimeoutError("the root runtime did not become idle")

    def assert_cleanup(self) -> None:
        """Assert no child runtime or link outlived the drained cohort.

        Retention is not a leak: a closed child stays available to readers as a
        read-only snapshot (``_readable_children``) or an on-disk record
        (``_stored_children``). What must be gone is every LIVE child runtime,
        every parent link, and every in-flight child close.
        """
        live = self.registry._children
        if live:
            raise AssertionError(f"{len(live)} child runtimes outlived the run")
        if self.registry._child_links:
            raise AssertionError("child links outlived the run")
        if self.registry._pending_child_closes:
            raise AssertionError("a child close was still in flight after teardown")

    @property
    def retained_child_sessions(self) -> int:
        """Report the child session records the run actually kept on disk."""
        return sum(1 for directory in self.child_session_dirs if directory.is_dir())

    def assert_child_session_records(self) -> None:
        """Assert every child of the run kept a non-empty session record."""
        if len(self.child_session_dirs) != len(self.bursts) * self.scale.burst_agents:
            raise AssertionError("a burst child left no session directory record")
        for directory in self.child_session_dirs:
            transcript = directory / "messages.jsonl"
            if not transcript.is_file() or transcript.stat().st_size == 0:
                raise AssertionError(f"child session record is empty: {directory}")

    def assert_child_history_fidelity(self, minimum_chars: int | None = None) -> None:
        """Assert every child request carried the seeded history it was given."""
        expected = (
            minimum_chars
            if minimum_chars is not None
            else (self.scale.history_messages * FIXTURE_MESSAGE_CHARS)
        )
        for burst in self.bursts:
            for chars in burst.context_chars:
                if chars < expected:
                    raise AssertionError(
                        f"a child request carried {chars} chars, expected >= {expected}"
                    )


def live_request_source(source: MessageList, envelope: RequestEnvelope) -> Any:
    """Build the gateway's live-source carrier for one prepared request."""
    from chartreux.core.agent_loop.llm_gateway import LiveRequestSource

    return LiveRequestSource(source=source, envelope=envelope)


# --------------------------------------------------------------------------
# Building a scenario
# --------------------------------------------------------------------------


def _ensure_harness_files() -> None:
    """Initialise the harness-file manager a bare-script runner lacks.

    pytest supplies this through an autouse fixture; a per-cell subprocess or a
    standalone campaign script does not, and the agent loop requires it at
    construction. Initialisation is idempotent and reads only the redirected
    config home the egress guard owns.
    """
    from chartreux.core.config.harness_files import (
        get_harness_files_manager,
        init_harness_files_manager,
    )

    try:
        get_harness_files_manager()
    except RuntimeError:
        init_harness_files_manager("user", "project")


def _catalog_snapshot(model_names: Sequence[str]) -> CatalogSnapshot:
    providers = {"test-perf": {"api_base": "https://perf.test", "backend": "generic"}}
    models = {
        name: {"deployments": [{"provider": "test-perf", "name": f"{name}-wire"}]}
        for name in model_names
    }
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": providers,
            "models": models,
            "roles": {
                "main": {
                    "description": "Performance scenario main preset",
                    "model": model_names[0],
                    "thinking": "high",
                }
            },
        }),
        "perf-headless-scenario",
    )


def fixture_history_slice(message_count: int) -> list[Any]:
    """Return the committed fixture's first ``message_count`` messages."""
    if not 0 < message_count <= FIXTURE_MESSAGE_COUNT:
        raise ValueError("message_count is outside the committed fixture")
    return build_fixture()[:message_count]


def assert_history_fidelity(
    messages: Sequence[Any], *, expected_count: int, expected_chars: int
) -> None:
    """Assert a seeded transcript carries the committed geometry exactly."""
    if len(messages) < expected_count:
        raise AssertionError(
            f"transcript carries {len(messages)} messages, expected >= {expected_count}"
        )
    window = messages[-expected_count:]
    chars = sum(
        len(message.content or "")
        for message in window
        if isinstance(getattr(message, "content", None), str)
    )
    if chars != expected_chars:
        raise AssertionError(
            f"transcript carries {chars} content chars, expected {expected_chars}"
        )


async def build_headless_scenario(
    cell: str = CELL_FRESH_HEADLESS,
    *,
    scale: ScenarioScale | None = None,
    session_root: Path | None = None,
    notify_agents: bool = True,
) -> HeadlessScenario:
    """Build a real headless scenario for ``cell`` at ``scale``.

    The scenario owns its registry, its root runtime, its live projector path
    and its real memory transport pair; :meth:`HeadlessScenario.close` releases
    all of them.
    """
    spec = workload_spec(cell)
    resolved_scale = FULL_SCALES[cell] if scale is None else scale
    _ensure_harness_files()
    child_session_dirs: list[Path] = []
    model_names = [
        f"perf-model-{index}" for index in range(max(resolved_scale.burst_agents, 1))
    ]
    snapshot = _catalog_snapshot(model_names)
    session_logging: dict[str, Any] = {"enabled": False}
    if session_root is not None:
        session_logging = {
            "enabled": True,
            "save_dir": str(session_root),
            "session_prefix": "perf",
        }
    config = build_test_vibe_config(
        enabled_tools=["task"],
        tools={"task": {"permission": "always"}},
        session_logging=session_logging,
    ).attach_catalog_snapshot(snapshot)

    backend = RootTurnBackend(
        first_byte_ms=spec.injected_first_byte_ms,
        chunk_gap_ms=spec.injected_chunk_gap_ms,
    )
    root_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    seed_history(root_loop, resolved_scale.history_messages)

    client_transport, server_transport = memory_transport_pair()
    projection = ClientProjection(
        PublicSessionState(
            event_id=0,
            session=PublicSession(
                id=root_loop.session_id,
                status=IdleSessionStatus(),
                created_at=1,
                updated_at=1,
            ),
            history=[],
            active_callbacks=[],
            turns=[],
        )
    )
    sink = NotificationSink(
        server_transport, projection, session_id=root_loop.session_id
    )
    client_end = _CollectingClient(client_transport)
    client_end.start()

    agents_notify = (
        _agents_notification_sink(sink, root_loop.session_id) if notify_agents else None
    )
    registry = SessionRuntimeRegistry(
        sink,
        _noop_deliver_callback,
        lambda _session_id: 0,
        runtime_factory=AgentRuntimeFactory(),
        notify_agents=agents_notify,
    )
    _seed_child_factory(registry, resolved_scale, child_session_dirs)
    root_runtime = registry._build_child_runtime(root_loop)
    registry.bind_root(root_runtime)
    return HeadlessScenario(
        cell=cell,
        spec=spec,
        scale=resolved_scale,
        registry=registry,
        root_runtime=root_runtime,
        root_loop=root_loop,
        sink=sink,
        projection=projection,
        client_end=client_end,
        client_transport=client_transport,
        server_transport=server_transport,
        backend=backend,
        model_names=model_names,
        child_session_dirs=child_session_dirs,
    )


def seed_history(agent_loop: Any, message_count: int) -> None:
    """Append the committed fixture slice to a live transcript."""
    agent_loop.messages.extend(fixture_history_slice(message_count))


def _seed_child_factory(
    registry: SessionRuntimeRegistry, scale: ScenarioScale, session_dirs: list[Path]
) -> None:
    """Give every child the seeded history the loaded cell commits.

    The child's session directory is recorded as it is created, so the run can
    assert that the cohort's records survived the burst (plan §5: child session
    records retained).
    """
    factory = registry._runtime_factory
    original = factory.create_child
    expected_messages = scale.history_messages
    expected_chars = expected_messages * FIXTURE_MESSAGE_CHARS

    @functools.wraps(original)
    async def create_child_with_history(*args: Any, **kwargs: Any) -> Any:
        child = await original(*args, **kwargs)
        seed_history(child, expected_messages)
        assert_history_fidelity(
            child.messages,
            expected_count=expected_messages,
            expected_chars=expected_chars,
        )
        session_dir = child.session_logger.session_dir
        if session_dir is not None:
            session_dirs.append(session_dir)
        return child

    factory.create_child = create_child_with_history  # type: ignore[method-assign]


def _agents_notification_sink(
    sink: NotificationSink, session_id: str
) -> Callable[[list[AgentSummaryModel], list[AgentEviction]], Awaitable[None]]:
    """Wire the agent inventory into the same serialized notification path."""

    async def notify(
        agents: list[AgentSummaryModel], evictions: list[AgentEviction]
    ) -> None:
        await sink(
            "agents/update",
            AgentsUpdateParams(
                event_id=sink.event_id + 1,
                session_id=session_id,
                emitted_at=0,
                agents=agents,
                evictions=[
                    AgentEvictionModel(
                        agent_id=eviction.agent_id,
                        run_id=eviction.run_id,
                        reason=eviction.reason,
                        idle_duration_seconds=eviction.idle_duration_seconds,
                        root_generation=eviction.root_generation,
                    )
                    for eviction in evictions
                ],
            ),
        )

    return notify


__all__ = [
    "CELL_FRESH_HEADLESS",
    "CELL_LOADED_HEADLESS",
    "FULL_SCALES",
    "SCENARIO_CELLS",
    "SCENARIO_METRIC_VERSION",
    "SHORTENED_SCALE",
    "WORKLOAD_HASHES",
    "EgressGuard",
    "HeadlessScenario",
    "NotificationSink",
    "NotificationWindow",
    "ScenarioScale",
    "WorkloadSpec",
    "assert_fingerprint_equality",
    "assert_history_fidelity",
    "build_headless_scenario",
    "child_task_text",
    "fixture_history_slice",
    "instrument_delta_dumps",
    "projected_history_fingerprint",
    "response_chunks",
    "root_user_content",
    "topology_fingerprint",
    "workload_spec",
]
