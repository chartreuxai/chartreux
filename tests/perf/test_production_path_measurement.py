"""Criterion 4: the production path measured where it actually runs.

The isolated benchmark set stays untouched: those nodes measure an in-process
loop under a controlled harness. This file measures the path a user's
interactive session takes -- the selection materialized into a spawn
environment, a root agent in its own process, and a turn that crosses the pipe
-- so the production figure is taken from the production topology itself.

Nothing here gates a number. Each node asserts only that the measurement was
taken: the family document is written outside the repository, its samples are
non-empty, and its labels name the topology, the tree and the evidence class.
The run is serial by construction (``-n0``); the interpreter, the tree and the
machine travel with every sample as labels, never as a comparison against a
bound.

No provider is reached: the root agent's own loop is built over a fake backend
inside the spawned child, and the egress guard blocks any attempt to leave the
process.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from time import perf_counter

import pytest

from chartreux.app_server.local import LocalHarnessOptions
from tests.app_server.test_production_entrypoint import (
    _client,
    _collect_turn,
    _install_stdio_environment,
    _record_backends,
)
from tests.differential import egress
from tests.perf._runtime_end_measurements import (
    EVIDENCE_REPLAY,
    Observation,
    emit_family,
    prepare,
    repository_root,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux", reason="the spawn and pipe observations are Linux-only"
    ),
    pytest.mark.skipif(
        os.environ.get("PYTEST_XDIST_WORKER") is not None,
        reason="the production path is measured serial-only: run with -n0",
    ),
]

#: The topology label every observation in this file carries.
TOPOLOGY = "stdio-materialized"

#: Independent runs: each one is a fresh spawn, a fresh handshake and a fresh
#: root agent process, so the spawn figure is per-run by definition.
_RUNS = 3

#: Turns per run; the first turn of each run pays the loop's own warmup and is
#: kept out of the turn samples.
_TURNS = 5

_REPO_ROOT = repository_root()

#: The spawned root agent: the shipped serve loop over a fake backend with one
#: answer per turn, so a multi-turn run never reaches an exhausted backend.
_CHILD_SCRIPT = """\
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
    backend=FakeBackend([
        [mock_llm_chunk(content=f"done {{index}}")]
        for index in range({turns})
    ]),
)
asyncio.run(
    serve_root_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())
)
"""


def _child_command(tmp_path: Path, *, turns: int = _TURNS) -> list[str]:
    return [
        sys.executable,
        "-c",
        _CHILD_SCRIPT.format(save_dir=str(tmp_path / "sessions"), turns=max(turns, 1)),
    ]


def _options(tmp_path: Path) -> LocalHarnessOptions:
    """The production interactive options: the run names no route at all."""
    from chartreux.app_server.protocol import SessionOptions

    return LocalHarnessOptions(
        client=_client(),
        session_options=SessionOptions(cwd=str(tmp_path), headless=True),
        interactive=True,
    )


def _family_path(name: str, observations: list[Observation]) -> Path:
    """Write one node's observations and prove the document landed."""
    prepare()
    path = emit_family(name, observations)
    assert path.exists()
    return path


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_the_materialized_spawn_reaches_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Start-to-ready over the production selection, one sample per spawn."""
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.local import LocalHarness

    samples: list[float] = []
    pids: list[int] = []
    backends = _record_backends(monkeypatch)
    for index in range(_RUNS):
        run_dir = tmp_path / f"run-{index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with egress.bootstrap_directory() as guard_dir:
            _install_stdio_environment(monkeypatch, guard_dir)
            monkeypatch.setattr(
                process_session_backend,
                "root_agent_command",
                lambda run_dir=run_dir: _child_command(run_dir),
            )
            harness = LocalHarness(_options(run_dir))
            started = perf_counter()
            try:
                session = await harness.start()
                samples.append(perf_counter() - started)
                pids.append(backends[-1].process.process.pid)
                await session.close()
            finally:
                await harness.close()

    assert len(samples) == _RUNS
    assert len(backends) == _RUNS
    # Every spawn is a real process: a figure taken in this process would be
    # the in-process benchmark, which is exactly what this file is not.
    assert all(pid != os.getpid() for pid in pids)
    _family_path(
        "production_path_spawn",
        [
            Observation(
                metric="spawn_to_ready",
                topology=TOPOLOGY,
                unit="s",
                samples=tuple(samples),
                evidence_class=EVIDENCE_REPLAY,
                note="selection materialized into the spawn env; fake backend",
                detail={"runs": _RUNS, "root_pids": pids, "tree": str(_REPO_ROOT)},
            )
        ],
    )


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_a_turn_crosses_the_materialized_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turn wall time over the pipe, first turn per run excluded as warmup."""
    from chartreux.app_server import process_session_backend
    from chartreux.app_server.local import LocalHarness

    samples: list[float] = []
    warmup: list[float] = []
    with egress.bootstrap_directory() as guard_dir:
        _install_stdio_environment(monkeypatch, guard_dir)
        monkeypatch.setattr(
            process_session_backend,
            "root_agent_command",
            lambda: _child_command(tmp_path),
        )
        backends = _record_backends(monkeypatch)
        harness = LocalHarness(_options(tmp_path))
        try:
            session = await harness.start()
            for index in range(_TURNS):
                started = perf_counter()
                events = await _collect_turn(session, f"turn {index}")
                elapsed = perf_counter() - started
                assert events
                if index == 0:
                    warmup.append(elapsed)
                else:
                    samples.append(elapsed)
            await session.close()
        finally:
            await harness.close()

    # The turn really ran in the spawned root: its pid is not this one, and by
    # now it is gone.
    assert len(backends) == 1
    assert backends[0].process.process.pid != os.getpid()
    assert backends[0].process.process.poll() is not None
    assert len(samples) == _TURNS - 1
    _family_path(
        "production_path_turn",
        [
            Observation(
                metric="turn_wall",
                topology=TOPOLOGY,
                unit="s",
                samples=tuple(samples),
                evidence_class=EVIDENCE_REPLAY,
                note="one turn per pipe round trip; warmup turn excluded",
                detail={
                    "turns": _TURNS,
                    "warmup_samples": warmup,
                    "tree": str(_REPO_ROOT),
                },
            )
        ],
    )
