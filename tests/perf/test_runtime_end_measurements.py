"""WP9 [9A] offline end measurements: the plan's serial observation set.

Each node here takes ONE family of the observations the plan's WP9 [9A] chunk
names — PTY input and paint gaps, the zero-delay turn wall, per-process and
total RSS/PSS at 0 and 8 children, the coordinator's loop delay, CPU, shed
frames and control latency, and process start-to-ready — and writes the raw
samples plus their labels into the external results directory. Nothing gates a
number: the nodes assert only that the measurement itself was taken (a family
document exists, its samples are non-empty, and the two topologies it compares
were both observed), never that a figure beats a bound.

The run is serial by construction (``-n0``): every figure is contention-free on
one machine, and the labels say so. ``tests/perf`` keeps its own envelope —
these nodes are ``perf``-marked and never installed-contract material.

Labels follow design.md:45-48: every figure carries the interpreter/tree it was
taken on and its evidence class. There are no live-provider figures here (the
campaign ledger is terminal and this chunk takes no provider call), and the
production interactive path still resolves to the in-memory transport, so the
new-topology PTY comparison the chunk asks for is recorded as an explicit gap
with its reason rather than approximated.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from tests.e2e.mock_server import ChatCompletionsRequestPayload, StreamChunk

from tests.perf._runtime_end_measurements import (
    EVIDENCE_OFFLINE,
    EVIDENCE_REPLAY,
    MEASUREMENT_SEED,
    P99_MIN_SAMPLES,
    Observation,
    child_environment,
    emit_family,
    measure_loop_window,
    prepare,
    repository_root,
    scratch_root,
    seed_content,
    shed_frames_under_saturation,
    tree_memory,
)
from tests.perf._runtime_process_tree import observe_process_tree

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="the process-tree and PTY observations are Linux-only",
    ),
    pytest.mark.skipif(
        os.environ.get("PYTEST_XDIST_WORKER") is not None,
        reason="[9A] measurements are serial-only: run with -n0",
    ),
]

_REPO_ROOT = repository_root()

#: Independent runs per metric family; the medians in the results are over the
#: pooled samples of these runs, and each run's own median is kept alongside.
_RUNS = 3

#: Turns per run for the turn-wall comparison (first turn per run is warmup).
_TURNS = 5

#: Children in the loaded window of the memory and coordinator families.
_CHILDREN = 8

#: Keystrokes per PTY session; the plan requires n>=100 for a p99 figure.
_PTY_INPUT_SAMPLES = 100

#: Control-class round trips per coordinator window (n>=100 for its p99).
_CONTROL_SAMPLES = 100

#: Loop-delay window: 3 s at a 10 ms cadence is ~300 samples per window.
_LOOP_WINDOW_S = 3.0
_LOOP_INTERVAL_S = 0.01

#: Presentation chunks the loopback mock provider streams per turn, and the
#: server's own inter-chunk delay (tests/e2e/mock_server.py:277).
_PTY_STREAM_CHUNKS = 120
_PTY_CHUNK_DELAY_S = 0.03
_PTY_LAST_TOKEN = f"tok{_PTY_STREAM_CHUNKS - 1:03d}"
_PTY_TURN_CAP_S = 30.0
"""Turn paints are measured only until the last streamed token is rendered.

The app keeps repainting after the turn (a blinking cursor is a 500 ms paint
tick), so an idle-gap heuristic would record the post-turn blink as a paint
stall. Completion on the last token is the deterministic cut.
"""

_ROOT_AGENT_SCRIPT = """\
import asyncio

from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.app_server.root_agent import serve_root_agent
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.models import SessionLoggingConfig
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class _Factory:
    def __init__(self, loop):
        self._loop = loop

    async def open(self, params):
        return self._loop

    async def close(self):
        pass


init_harness_files_manager("user", "project")
config = build_test_vibe_config(
    session_logging=SessionLoggingConfig(enabled=True, save_dir={save_dir!r})
)
loop = build_test_agent_loop(
    config=config,
    backend=FakeBackend([[mock_llm_chunk(content="done")] for _ in range({streams})]),
)
asyncio.run(
    serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""

_CHILD_AGENT_SCRIPT = """\
import asyncio

from chartreux.app_server.child_agent import serve_child_agent
from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.models import SessionLoggingConfig
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class _Factory:
    def __init__(self, loop):
        self._loop = loop

    async def open(self, params):
        return self._loop

    async def close(self):
        pass


init_harness_files_manager("user", "project")
config = build_test_vibe_config(
    session_logging=SessionLoggingConfig(enabled=True, save_dir={save_dir!r})
)
loop = build_test_agent_loop(
    config=config, backend=FakeBackend([[mock_llm_chunk(content="done")]])
)
asyncio.run(
    serve_child_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""


def _root_agent_command(save_dir: Path, *, streams: int) -> list[str]:
    script = _ROOT_AGENT_SCRIPT.format(save_dir=str(save_dir), streams=streams)
    return [sys.executable, "-c", script]


def _child_agent_command(save_dir: Path) -> list[str]:
    return [sys.executable, "-c", _CHILD_AGENT_SCRIPT.format(save_dir=str(save_dir))]


def _family_dir(name: str) -> Path:
    path = scratch_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {"median": ordered[len(ordered) // 2], "min": ordered[0], "max": ordered[-1]}


def _spawner(env: dict[str, str], save_dir: Path) -> Any:
    from chartreux.app_server.child_agent_process import ProcessSubagentSpawn

    return ProcessSubagentSpawn(
        env=env,
        cwd=str(_REPO_ROOT),
        command=_child_agent_command(save_dir),
        ready_timeout=60,
        request_timeout=60,
    )


def _spec(index: int, *, parent_session: str) -> Any:
    from chartreux.app_server.subagent_ports import SubagentSpawnSpec

    agent_id = f"measurement-agent-{index}"
    return SubagentSpawnSpec(
        agent_id=agent_id,
        parent_session_id=parent_session,
        session_dir=f"sessions/{parent_session}/agents/{agent_id}",
        payload={"task_summary": "measurement child"},
    )


async def _open_tree(run_dir: Path, children: int) -> tuple[Any, list[Any], Any]:
    """One coordinator tree: a root agent process plus ``children`` agents."""
    from chartreux.app_server._connection_protocol import ClientInfo
    from chartreux.app_server.protocol import AgentConfig
    from chartreux.app_server.root_agent_process import spawn_root_agent_process
    from tests.differential import egress

    guard = egress.bootstrap_directory()
    guard_dir = guard.__enter__()
    env = child_environment(guard_dir)
    spawned: list[Any] = []
    root = await spawn_root_agent_process(
        session_options=AgentConfig(),
        client_info=ClientInfo(name="wp9-measurement", version="1.0"),
        env=env,
        cwd=str(_REPO_ROOT),
        command=_root_agent_command(run_dir / "root-sessions", streams=8),
        ready_timeout=60,
        request_timeout=60,
    )
    try:
        spawner = _spawner(env, run_dir / "child-sessions")
        parent_session = root.session_id or "measurement-root"
        for index in range(children):
            spawned.append(
                await spawner.spawn(_spec(index, parent_session=parent_session))
            )
    except BaseException:
        for child in spawned:
            await child.close()
        await root.close()
        guard.__exit__(None, None, None)
        raise
    return root, spawned, guard


async def _close_tree(root: Any, spawned: list[Any], guard: Any) -> None:
    for child in spawned:
        await child.close()
    await root.close()
    guard.__exit__(None, None, None)


# --------------------------------------------------------------------------
# Family 1: the zero-delay turn wall, in-process vs process-root
# --------------------------------------------------------------------------


def _in_process_turn_walls(prompts: list[str], save_dir: Path) -> list[float]:
    """Turns driven through the in-process agent loop (today's hot path)."""
    from chartreux.core.agent_loop._loop_types import AgentTurnOptions
    from chartreux.core.config.models import SessionLoggingConfig
    from tests.conftest import build_test_agent_loop, build_test_vibe_config
    from tests.mock.utils import mock_llm_chunk
    from tests.stubs.fake_backend import FakeBackend

    save_dir.mkdir(parents=True, exist_ok=True)
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(save_dir))
    )
    loop = build_test_agent_loop(
        config=config,
        backend=FakeBackend([[mock_llm_chunk(content="done")] for _ in prompts]),
    )
    walls: list[float] = []

    async def drive() -> None:
        for index, prompt in enumerate(prompts):
            started = perf_counter()
            async for _event in loop.act(
                prompt, turn_options=AgentTurnOptions(turn_id=f"in-process-{index}")
            ):
                pass
            walls.append((perf_counter() - started) * 1000.0)

    asyncio.run(drive())
    return walls


async def _process_turn_walls(prompts: list[str], save_dir: Path) -> list[float]:
    """The same turns driven through a spawned root agent process."""
    from chartreux.app_server._connection_protocol import ClientInfo
    from chartreux.app_server.protocol import AgentConfig
    from chartreux.app_server.root_agent_process import spawn_root_agent_process
    from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
    from chartreux.core.agent_loop._loop_types import AgentTurnOptions
    from tests.differential import egress

    save_dir.mkdir(parents=True, exist_ok=True)
    walls: list[float] = []
    with egress.bootstrap_directory() as guard_dir:
        process = await spawn_root_agent_process(
            session_options=AgentConfig(),
            client_info=ClientInfo(name="wp9-measurement", version="1.0"),
            env=child_environment(guard_dir),
            cwd=str(_REPO_ROOT),
            command=_root_agent_command(save_dir, streams=len(prompts)),
            ready_timeout=60,
            request_timeout=60,
        )
        try:
            for index, prompt in enumerate(prompts):
                started = perf_counter()
                request = ExecutionTurnRequest(
                    message=prompt,
                    turn_options=AgentTurnOptions(turn_id=f"process-{index}"),
                )
                async for _event in process.run_turn(request):
                    pass
                walls.append((perf_counter() - started) * 1000.0)
        finally:
            await process.close()
    return walls


@pytest.mark.timeout(300)
def test_measure_turn_wall() -> None:
    """Zero-delay turn wall: in-process agent loop vs spawned root agent."""
    prepare()
    run_root = _family_dir("turn-wall")
    prompts = seed_content(MEASUREMENT_SEED, count=_RUNS * _TURNS)

    in_process: list[float] = []
    process: list[float] = []
    per_run: list[dict[str, Any]] = []
    for run in range(_RUNS):
        window = prompts[run * _TURNS : (run + 1) * _TURNS]
        old_walls = _in_process_turn_walls(window, run_root / f"old-{run}")
        new_walls = asyncio.run(_process_turn_walls(window, run_root / f"new-{run}"))
        in_process.extend(old_walls)
        process.extend(new_walls)
        per_run.append({
            "run": run,
            "in_process_ms": old_walls,
            "process_root_ms": new_walls,
        })

    observations = [
        Observation(
            metric="turn_wall",
            topology="in_process",
            unit="ms",
            samples=tuple(in_process),
            evidence_class=EVIDENCE_REPLAY,
            note=(
                "seeded scripted provider, in-process agent loop, logging enabled; "
                f"{_RUNS} runs x {_TURNS} turns, serial"
            ),
            detail={
                "runs": _RUNS,
                "turns_per_run": _TURNS,
                "per_run": per_run,
                "warm_turn_median_ms": _percentiles(
                    [run["in_process_ms"][1:] for run in per_run][0]
                )["median"],
            },
        ),
        Observation(
            metric="turn_wall",
            topology="process_root",
            unit="ms",
            samples=tuple(process),
            evidence_class=EVIDENCE_REPLAY,
            note=(
                "seeded scripted provider, spawned root agent process over the stdio "
                f"transport; {_RUNS} runs x {_TURNS} turns, serial"
            ),
            detail={
                "runs": _RUNS,
                "turns_per_run": _TURNS,
                "includes_ipc_round_trip": True,
                "per_run_medians_ms": [
                    _percentiles(run["process_root_ms"])["median"] for run in per_run
                ],
            },
        ),
    ]
    path = emit_family(
        "wp9a-turn-wall",
        observations,
        extra={
            "comparison": "old topology = in-process agent loop; new = spawned root process",
            "first_turn_is_warmup": True,
            "p99_note": (
                f"n={len(in_process)}/{len(process)} turn samples: below the "
                f"n>={P99_MIN_SAMPLES} p99 threshold, so no p99 is claimed"
            ),
        },
    )
    assert path.is_file()
    assert len(in_process) == len(process) == _RUNS * _TURNS


# --------------------------------------------------------------------------
# Family 2: PTY input and paint gaps
# --------------------------------------------------------------------------


def _pty_chunks(
    index: int, payload: ChatCompletionsRequestPayload
) -> list[StreamChunk]:
    """A loopback provider stream of fixed size, one token per chunk."""
    from tests.e2e.mock_server import StreamingMockServer

    made: list[dict[str, Any]] = []
    for step in range(_PTY_STREAM_CHUNKS):
        delta: dict[str, object] = {"content": f"tok{step:03d} "}
        if step == 0:
            delta["role"] = "assistant"
        made.append(
            StreamingMockServer.build_chunk(
                created=900 + step, delta=delta, finish_reason=None
            )
        )
    made.append(
        StreamingMockServer.build_chunk(
            created=999,
            delta={},
            finish_reason="stop",
            usage={"prompt_tokens": 3, "completion_tokens": 4},
        )
    )
    return made


def _pty_session(session_dir: Path) -> dict[str, Any]:
    """One real TUI session in a PTY: input gaps, then one streamed turn."""
    import pexpect

    from tests.e2e.common import ansi_tolerant_pattern, strip_ansi, write_e2e_config
    from tests.e2e.mock_server import StreamingMockServer

    home = session_dir / "home"
    workdir = session_dir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    server = StreamingMockServer(chunk_factory=_pty_chunks)
    server.start()
    child: Any = None
    try:
        write_e2e_config(home, server.api_base)
        environment = cast("os._Environ[str]", os.environ.copy())
        environment["CHARTREUX_HOME"] = str(home)
        environment["MISTRAL_API_KEY"] = "fake-key"
        environment["CHARTREUX_TEST_DISABLE_KEYRING"] = "1"
        environment["CHARTREUX_TEST_DISABLE_AUTO_TITLE"] = "1"
        environment["TERM"] = "xterm-256color"
        environment["TMPDIR"] = str(scratch_root())
        child = pexpect.spawn(
            str(Path(sys.executable).with_name("chartreux")),
            ["--workdir", str(workdir)],
            cwd=str(_REPO_ROOT),
            env=environment,
            encoding="utf-8",
            timeout=60,
            dimensions=(36, 120),
        )
        boot_started = perf_counter()
        child.expect(ansi_tolerant_pattern("F1 Help · /help"), timeout=60)
        boot_ms = (perf_counter() - boot_started) * 1000.0

        input_gaps: list[float] = []
        for _ in range(_PTY_INPUT_SAMPLES):
            started = perf_counter()
            child.send("x")
            child.expect(ansi_tolerant_pattern("x"), timeout=10)
            input_gaps.append((perf_counter() - started) * 1000.0)

        turn_started = perf_counter()
        child.send("\r")
        stamps: list[float] = []
        rendered = ""
        completed = False
        while perf_counter() - turn_started < _PTY_TURN_CAP_S:
            try:
                data = child.read_nonblocking(size=200000, timeout=0.02)
            except pexpect.TIMEOUT:
                continue
            except pexpect.EOF:
                break
            stamps.append(perf_counter() - turn_started)
            rendered += strip_ansi(data)
            if _PTY_LAST_TOKEN in rendered:
                completed = True
                break
        paint_gaps = [
            (stamps[index + 1] - stamps[index]) * 1000.0
            for index in range(len(stamps) - 1)
        ]
        return {
            "session": str(session_dir),
            "boot_ms": boot_ms,
            "input_gaps_ms": input_gaps,
            "paint_gaps_ms": paint_gaps,
            "paint_reads": len(stamps),
            "requests": len(server.requests),
            "turn_completed": completed,
            "turn_span_ms": (stamps[-1] * 1000.0) if stamps else None,
        }
    finally:
        if child is not None:
            if child.isalive():
                child.terminate(force=True)
            child.close()
        server.stop()


@pytest.mark.timeout(600)
def test_measure_pty_gaps() -> None:
    """PTY input->paint and inter-paint gaps for the in-process TUI topology."""
    prepare()
    run_root = _family_dir("pty-gaps")

    sessions: list[dict[str, Any]] = []
    for index in range(_RUNS):
        sessions.append(_pty_session(run_root / f"session-{index}"))

    input_samples = [
        sample for session in sessions for sample in session["input_gaps_ms"]
    ]
    paint_samples = [
        sample for session in sessions for sample in session["paint_gaps_ms"]
    ]
    assert input_samples, "the PTY nodes produced no input->paint samples"

    observations = [
        Observation(
            metric="pty_input_to_paint",
            topology="in_process_tui",
            unit="ms",
            samples=tuple(input_samples),
            evidence_class=EVIDENCE_REPLAY,
            note=(
                "one keystroke to its rendered echo in the real TUI, loopback "
                f"scripted provider; {_RUNS} sessions x {_PTY_INPUT_SAMPLES} "
                "keystrokes, serial"
            ),
            detail={
                "sessions": _RUNS,
                "keystrokes_per_session": _PTY_INPUT_SAMPLES,
                "per_session_medians_ms": [
                    _percentiles(session["input_gaps_ms"])["median"]
                    for session in sessions
                ],
                "boot_ms": [session["boot_ms"] for session in sessions],
            },
        ),
        Observation(
            metric="pty_paint_gap",
            topology="in_process_tui",
            unit="ms",
            samples=tuple(paint_samples),
            evidence_class=EVIDENCE_REPLAY,
            note=(
                "inter-paint gap while a "
                f"{_PTY_STREAM_CHUNKS}-chunk reply streams at the provider's "
                f"{int(_PTY_CHUNK_DELAY_S * 1000)}ms cadence; coalesced renders "
                "dominate the gap"
            ),
            detail={
                "sessions": _RUNS,
                "paint_reads": [session["paint_reads"] for session in sessions],
                "turn_span_ms": [session["turn_span_ms"] for session in sessions],
                "provider_requests": [session["requests"] for session in sessions],
                "turn_completed": [session["turn_completed"] for session in sessions],
                "cut": f"last streamed token ({_PTY_LAST_TOKEN}) rendered",
            },
        ),
    ]
    path = emit_family(
        "wp9a-pty-gaps",
        observations,
        extra={
            "new_topology_gap": {
                "measured": False,
                "reason": (
                    "the interactive path resolves to the in-memory transport "
                    "(chartreux/app_server/local.py uses memory_transport_pair), so "
                    "no TUI runs on the process-root topology and there is nothing "
                    "to compare; the process-root turn cost is measured at the RPC "
                    "boundary in wp9a-turn-wall.json instead"
                ),
            },
            "evidence_note": (
                "the provider is a loopback scripted server, so this is a replay "
                "figure, not a live turn"
            ),
        },
    )
    assert path.is_file()
    assert len(input_samples) >= P99_MIN_SAMPLES
    assert all(session["turn_completed"] for session in sessions), (
        "a PTY session did not render the last streamed token inside "
        f"{_PTY_TURN_CAP_S}s: {[session['turn_span_ms'] for session in sessions]}"
    )


# --------------------------------------------------------------------------
# Family 3: per-process and total RSS/PSS at 0 and 8 children
# --------------------------------------------------------------------------


async def _memory_window(run_dir: Path, children: int) -> dict[str, Any]:
    """One settled coordinator tree, sampled for RSS/PSS."""
    root, spawned, guard = await _open_tree(run_dir, children)
    try:
        await asyncio.sleep(0.5)
        return tree_memory(observe_process_tree())
    finally:
        await _close_tree(root, spawned, guard)


@pytest.mark.timeout(600)
def test_measure_tree_memory() -> None:
    """Coordinator + root agent + 0/8 child processes: RSS/PSS per process."""
    prepare()
    run_root = _family_dir("memory")

    windows: list[dict[str, Any]] = []
    for run in range(_RUNS):
        for children in (0, _CHILDREN):
            memory = asyncio.run(
                _memory_window(run_root / f"run-{run}-{children}", children)
            )
            windows.append({"run": run, "children": children, "memory": memory})

    def totals(children: int, key: str) -> list[float]:
        return [
            float(window["memory"][key])
            for window in windows
            if window["children"] == children and window["memory"][key] is not None
        ]

    empty_rss = totals(0, "total_rss_bytes")
    empty_pss = totals(0, "total_pss_bytes")
    loaded_rss = totals(_CHILDREN, "total_rss_bytes")
    loaded_pss = totals(_CHILDREN, "total_pss_bytes")

    per_child_rss = [
        float(row["rss_bytes"])
        for window in windows
        if window["children"] == _CHILDREN
        for row in window["memory"]["rows"][2:]
        if row["rss_bytes"] is not None
    ]
    per_child_pss = [
        float(row["pss_bytes"])
        for window in windows
        if window["children"] == _CHILDREN
        for row in window["memory"]["rows"][2:]
        if row["pss_bytes"] is not None
    ]
    coordinator_rss = [
        float(window["memory"]["rows"][0]["rss_bytes"])
        for window in windows
        if window["children"] == 0 and window["memory"]["rows"][0]["rss_bytes"]
    ]

    observations = (
        [
            Observation(
                metric="tree_total_rss",
                topology=f"coordinator+root+{width}_children",
                unit="bytes",
                samples=tuple(samples),
                evidence_class=EVIDENCE_OFFLINE,
                note=f"{_RUNS} runs at {width} children, coordinator tree sampled idle",
                detail={
                    "children": width,
                    "process_counts": [
                        window["memory"]["process_count"]
                        for window in windows
                        if window["children"] == width
                    ],
                },
            )
            for width, samples in ((0, empty_rss), (_CHILDREN, loaded_rss))
        ]
        + [
            Observation(
                metric="tree_total_pss",
                topology=f"coordinator+root+{width}_children",
                unit="bytes",
                samples=tuple(samples),
                evidence_class=EVIDENCE_OFFLINE,
                note=(
                    f"{_RUNS} runs at {width} children; PSS total is the honest tree "
                    "figure because a shared page is counted once"
                ),
                detail={"children": width},
            )
            for width, samples in ((0, empty_pss), (_CHILDREN, loaded_pss))
        ]
        + [
            Observation(
                metric="child_process_rss",
                topology="child_agent_process",
                unit="bytes",
                samples=tuple(per_child_rss),
                evidence_class=EVIDENCE_OFFLINE,
                note="one row per spawned child process across the loaded windows",
                detail={"samples_per_run": _CHILDREN},
            ),
            Observation(
                metric="child_process_pss",
                topology="child_agent_process",
                unit="bytes",
                samples=tuple(per_child_pss),
                evidence_class=EVIDENCE_OFFLINE,
                note="one row per spawned child process across the loaded windows",
                detail={"samples_per_run": _CHILDREN},
            ),
            Observation(
                metric="coordinator_process_rss",
                topology="coordinator",
                unit="bytes",
                samples=tuple(coordinator_rss),
                evidence_class=EVIDENCE_OFFLINE,
                note="the measuring process, which holds the coordinator's registries",
            ),
        ]
    )
    path = emit_family(
        "wp9a-memory",
        observations,
        extra={
            "runs": _RUNS,
            "children_per_loaded_window": _CHILDREN,
            "per_run": windows,
            "mcp_inclusive": (
                "the coordinator process carries the MCP registry surface and each "
                "spawned agent builds its own; no MCP server subprocess is started "
                "offline (no live MCP server is available to this chunk), so MCP "
                "client/registry memory is included and MCP server memory is absent "
                "by construction"
            ),
            "delta_pss_0_to_8_children_bytes": (
                (sum(loaded_pss) / len(loaded_pss)) - (sum(empty_pss) / len(empty_pss))
                if loaded_pss and empty_pss
                else None
            ),
        },
    )
    assert path.is_file()
    assert empty_pss and loaded_pss, "memory windows produced no PSS totals"
    assert all(
        window["memory"]["process_count"] == window["children"] + 2
        for window in windows
    )


# --------------------------------------------------------------------------
# Family 4: coordinator loop delay, CPU, shed frames, control latency
# --------------------------------------------------------------------------


async def _coordinator_window(run_dir: Path, children: int) -> dict[str, Any]:
    """One coordinator window at a fixed child count."""
    root, spawned, guard = await _open_tree(run_dir, children)
    try:
        await asyncio.sleep(0.3)
        loop_window = await measure_loop_window(
            seconds=_LOOP_WINDOW_S, interval_s=_LOOP_INTERVAL_S
        )
        control_ms: list[float] = []
        for _ in range(_CONTROL_SAMPLES):
            started = perf_counter()
            await root.read_page(root.session_id, limit=1)
            control_ms.append((perf_counter() - started) * 1000.0)
        return {"children": children, "loop": loop_window, "control_ms": control_ms}
    finally:
        await _close_tree(root, spawned, guard)


@pytest.mark.timeout(900)
def test_measure_coordinator() -> None:
    """Loop delay, CPU share, control latency, and shed frames."""
    prepare()
    run_root = _family_dir("coordinator")

    windows = [
        asyncio.run(_coordinator_window(run_root / f"run-{run}-{children}", children))
        for run in range(_RUNS)
        for children in (0, _CHILDREN)
    ]
    shed = [
        shed_frames_under_saturation(
            offered=200, payload_bytes=1024, per_stream_budget=8192
        )
        for _ in range(_RUNS)
    ]

    def collect(children: int, key: str) -> list[float]:
        return [
            value
            for window in windows
            if window["children"] == children
            for value in (window["loop"][key] or [])
        ]

    empty_lags = collect(0, "loop_delay_ms")
    loaded_lags = collect(_CHILDREN, "loop_delay_ms")
    empty_control = [
        value
        for window in windows
        if window["children"] == 0
        for value in window["control_ms"]
    ]
    loaded_control = [
        value
        for window in windows
        if window["children"] == _CHILDREN
        for value in window["control_ms"]
    ]
    empty_cpu = [
        window["loop"]["cpu_share"]
        for window in windows
        if window["children"] == 0 and window["loop"]["cpu_share"] is not None
    ]
    loaded_cpu = [
        window["loop"]["cpu_share"]
        for window in windows
        if window["children"] == _CHILDREN and window["loop"]["cpu_share"] is not None
    ]
    empty_cpu_s = [
        window["loop"]["cpu_s"]
        for window in windows
        if window["children"] == 0 and window["loop"]["cpu_s"] is not None
    ]
    loaded_cpu_s = [
        window["loop"]["cpu_s"]
        for window in windows
        if window["children"] == _CHILDREN and window["loop"]["cpu_s"] is not None
    ]
    shed_counts = [float(record["shed"]) for record in shed]

    observations = (
        [
            Observation(
                metric="coordinator_loop_delay",
                topology=f"coordinator_at_{width}_children",
                unit="ms",
                samples=tuple(samples),
                evidence_class=EVIDENCE_OFFLINE,
                note=(
                    f"heartbeat lateness at a {int(_LOOP_INTERVAL_S * 1000)}ms cadence "
                    f"over {_LOOP_WINDOW_S}s windows, {_RUNS} runs"
                ),
                detail={"children": width},
            )
            for width, samples in ((0, empty_lags), (_CHILDREN, loaded_lags))
        ]
        + [
            Observation(
                metric="coordinator_cpu_share",
                topology=f"coordinator_at_{width}_children",
                unit="fraction_of_one_cpu",
                samples=tuple(samples),
                evidence_class=EVIDENCE_OFFLINE,
                note="coordinator CPU time over the loop window, from /proc/self/stat",
                detail={"children": width, "windows": len(samples)},
            )
            for width, samples in ((0, empty_cpu), (_CHILDREN, loaded_cpu))
        ]
        + [
            Observation(
                metric="coordinator_cpu_seconds",
                topology=f"coordinator_at_{width}_children",
                unit="s",
                samples=tuple(samples),
                evidence_class=EVIDENCE_OFFLINE,
                note=(
                    f"CPU seconds the coordinator spent in a {_LOOP_WINDOW_S}s window "
                    "while every child sat idle, so the share reads as near zero"
                ),
                detail={"children": width},
            )
            for width, samples in ((0, empty_cpu_s), (_CHILDREN, loaded_cpu_s))
        ]
        + [
            Observation(
                metric="control_request_round_trip",
                topology=f"coordinator_at_{width}_children",
                unit="ms",
                samples=tuple(samples),
                evidence_class=EVIDENCE_OFFLINE,
                note=(
                    "control-class transcript page request to the root agent process "
                    "over the stdio transport"
                ),
                detail={"children": width},
            )
            for width, samples in ((0, empty_control), (_CHILDREN, loaded_control))
        ]
        + [
            Observation(
                metric="presentation_frames_shed",
                topology="relay_hop_saturated",
                unit="frames",
                samples=tuple(shed_counts),
                evidence_class=EVIDENCE_OFFLINE,
                note=(
                    "structural: an unstarted relay hop offered 200 presentation "
                    "frames against an 8192-byte per-stream budget; the count is the "
                    "shedding rule's outcome, not a live drop count"
                ),
                detail={"runs": shed, "offered": 200},
            )
        ]
    )
    path = emit_family(
        "wp9a-coordinator",
        observations,
        extra={
            "runs": _RUNS,
            "children_per_loaded_window": _CHILDREN,
            "windows": windows,
            "drops_note": (
                "presentation_frames_shed measures the shedding rule under "
                "saturation; no live TUI drop count exists because no presentation "
                "consumer is wired to the process-root topology yet"
            ),
        },
    )
    assert path.is_file()
    assert len(empty_lags) >= P99_MIN_SAMPLES
    assert len(loaded_lags) >= P99_MIN_SAMPLES
    assert len(empty_control) >= P99_MIN_SAMPLES
    assert len(loaded_control) >= P99_MIN_SAMPLES


# --------------------------------------------------------------------------
# Family 5: process start-to-ready
# --------------------------------------------------------------------------


async def _root_ready_ms(save_dir: Path) -> float:
    from chartreux.app_server._connection_protocol import ClientInfo
    from chartreux.app_server.protocol import AgentConfig
    from chartreux.app_server.root_agent_process import spawn_root_agent_process
    from tests.differential import egress

    save_dir.mkdir(parents=True, exist_ok=True)
    with egress.bootstrap_directory() as guard_dir:
        started = perf_counter()
        process = await spawn_root_agent_process(
            session_options=AgentConfig(),
            client_info=ClientInfo(name="wp9-measurement", version="1.0"),
            env=child_environment(guard_dir),
            cwd=str(_REPO_ROOT),
            command=_root_agent_command(save_dir, streams=1),
            ready_timeout=60,
            request_timeout=60,
        )
        elapsed = (perf_counter() - started) * 1000.0
        await process.close()
    return elapsed


async def _child_ready_ms(save_dir: Path, index: int) -> float:
    from tests.differential import egress

    with egress.bootstrap_directory() as guard_dir:
        spawner = _spawner(child_environment(guard_dir), save_dir)
        started = perf_counter()
        spawned = await spawner.spawn(_spec(index, parent_session="measurement-root"))
        elapsed = (perf_counter() - started) * 1000.0
        await spawned.close()
    return elapsed


@pytest.mark.timeout(300)
def test_measure_start_to_ready() -> None:
    """Process spawn to ready-frame handshake, for both agent kinds."""
    prepare()
    run_root = _family_dir("start-to-ready")

    root_samples = [
        asyncio.run(_root_ready_ms(run_root / f"root-{index}"))
        for index in range(_TURNS)
    ]
    child_samples = [
        asyncio.run(_child_ready_ms(run_root / f"child-{index}", index))
        for index in range(_TURNS)
    ]

    observations = [
        Observation(
            metric="spawn_to_ready",
            topology="root_agent_process",
            unit="ms",
            samples=tuple(root_samples),
            evidence_class=EVIDENCE_OFFLINE,
            note=(
                "interpreter start plus module import plus the ready-frame "
                f"handshake; {len(root_samples)} spawns"
            ),
            detail={"ready_frame": True, "cwd": str(_REPO_ROOT)},
        ),
        Observation(
            metric="spawn_to_ready",
            topology="child_agent_process",
            unit="ms",
            samples=tuple(child_samples),
            evidence_class=EVIDENCE_OFFLINE,
            note=(
                "interpreter start plus module import plus the ready-frame "
                f"handshake for a subagent child; {len(child_samples)} spawns"
            ),
            detail={"ready_frame": True, "cwd": str(_REPO_ROOT)},
        ),
    ]
    path = emit_family(
        "wp9a-start-to-ready",
        observations,
        extra={
            "spawns_per_kind": _TURNS,
            "measured_vs_bound": (
                "the plan's 15 s spawn-ready deadline is validation material, not "
                "a figure measured here: these samples are observed values on this "
                "machine, not a bound check"
            ),
        },
    )
    assert path.is_file()
    assert root_samples and child_samples


def test_measurement_results_stay_outside_the_repository() -> None:
    """The results directory is external: no artifact lands in the tree."""
    prepare()
    root = scratch_root().resolve()
    repo = _REPO_ROOT.resolve()
    assert root != repo
    assert repo not in root.parents
    json.dumps({"root": str(root)})
