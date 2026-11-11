"""WP6 [6D]: the both-mode gate.

The differential suite must hold under both transport selections of design
invariant 1: the multi-process (stdio) route and the in-process escape. This
module gates that in three ways.

*Full-suite digest gate.* ``test_full_differential_suite_digests_match_across_modes``
runs the whole trace suite twice — once with ``CHARTREUX_TRANSPORT=stdio`` and
once with ``CHARTREUX_TRANSPORT=in_process`` — and compares the per-trace
normalized digests both runs recorded in the [6D] ledger. Every trace that
takes part (all nineteen presentation traces, the root oracle, the child crash
oracle, the replay skeleton, and the shutdown nodes) must produce the same
digest in both modes.

*Per-family cross-mode traces.* The ``..._digests_match_across_modes`` tests
run one trace of each family in both modes and both topologies inside this
process — twice per host, so a nondeterministic trace is reported as such — and
then compare all four streams, so a mode-dependent divergence fails with the
diverging field in hand instead of only as a digest mismatch between two nested
runs.

*The mode is real, and egress stays blocked.* The probe traces prove the
selector actually reaches both hosted components (rather than only the
driver's environment), and that a spawned component reaches neither the fake
provider's endpoint nor an independent non-loopback destination.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit

import pytest

from chartreux.app_server.transport import TRANSPORT_ENV_VAR
from tests.differential import egress, ledger
from tests.differential.crash_oracle import SCENARIOS, ChildCrashDriver
from tests.differential.lifecycle import LEASE_SESSION, LifecycleDriver
from tests.differential.presentation import (
    PresentationDriver,
    build_history,
    history_payload,
)
from tests.differential.replay import (
    COMPONENT_CHILD_CRASH,
    COMPONENT_ECHO,
    COMPONENT_LIFECYCLE,
    COMPONENT_PRESENTATION,
    COMPONENT_PROBES,
    COMPONENT_ROOT_ORACLE,
    REPLAY_ROOT,
    TRANSPORT_MODES,
    ReplayDriver,
    build_replay_script,
    digest,
    in_process_component,
    normalize_frames,
    subprocess_component,
    transport_mode,
)
from tests.differential.root_oracle import (
    GRANTED_DIR,
    INNER_FILE,
    OTHER_DIR,
    PROJECT_DIR,
    RootOracleDriver,
    build_root_oracle_script,
)
from tests.e2e.mock_server import StreamingMockServer

IN_PROCESS = "in_process"
SPAWNED = "spawned"
TOPOLOGIES = (IN_PROCESS, SPAWNED)
STDIO = "stdio"
MODES = TRANSPORT_MODES
NON_LOOPBACK_DESTINATION = "192.0.2.1"

_TRACE_TIMEOUT = 120.0
_SUITE_TIMEOUT = 600
_SUITE_WORKERS = "4"
DEFAULT_SEED = 20_261_110
_CALM_DELTA = {"jsonrpc": "2.0", "method": "usage/updated", "params": {}}

#: The trace modules the nested both-mode runs execute. This module is not in
#: the list: it drives them, and a gate that ran itself would recurse.
TRACE_MODULES = (
    "tests/differential/test_presentation_traces.py",
    "tests/differential/test_parity_root.py",
    "tests/differential/test_child_crash.py",
    "tests/differential/test_replay_skeleton.py",
    "tests/differential/test_shutdown_nodes.py",
)


def _host(component: str, topology: str, *, timeout: float = _TRACE_TIMEOUT):
    if topology == IN_PROCESS:
        return in_process_component(component=component, timeout=timeout)
    return subprocess_component(component=component, timeout=timeout)


# -- per-family trace runners ------------------------------------------------
def _replay_frames(topology: str, base: Path, seed: int) -> tuple[dict[str, Any], ...]:
    with _host(COMPONENT_ECHO, topology) as channel:
        outcome = ReplayDriver(channel).run(build_replay_script(seed))
    return outcome.frames


def _presentation_frames(
    topology: str, base: Path, seed: int
) -> tuple[dict[str, Any], ...]:
    history = build_history(seed, 4)
    init = {"clock": 100.0, "streams": ["session"]}
    ops: list[dict[str, Any]] = [
        {
            "kind": "snapshot",
            "stream": "session",
            "payloads": [history_payload(history)],
            "boundary": len(history),
        },
        {"kind": "delta", "stream": "session", "frame": _CALM_DELTA},
        {
            "kind": "delta",
            "stream": "session",
            "frame": {"jsonrpc": "2.0", "method": "session/updated", "params": {}},
        },
        {"kind": "tick", "stream": "session", "seconds": 0.25},
        {"kind": "retire", "stream": "session"},
        {
            "kind": "snapshot",
            "stream": "session",
            "payloads": [history_payload(history)],
            "boundary": len(history),
        },
        {"kind": "eof", "stream": "session"},
    ]
    with _host(COMPONENT_PRESENTATION, topology) as channel:
        driver = PresentationDriver(channel)
        driver.initialize(init)
        frames = driver.run(ops)
    install = frames[0]["streams"]["session"]["install"]
    assert install is not None
    assert install["boundary"] == len(history)
    assert frames[-1]["streams"]["session"]["phase"] == "terminal"
    assert frames[-1]["streams"]["session"]["terminal"] is not None
    return frames


def _crash_frames(topology: str, base: Path, seed: int) -> tuple[dict[str, Any], ...]:
    base.mkdir(parents=True, exist_ok=True)
    with _host(COMPONENT_CHILD_CRASH, topology) as channel:
        driver = ChildCrashDriver(channel)
        driver.initialize({"base": str(base)})
        frames = driver.run([{"kind": "crash", "scenario": SCENARIOS[0]}])
    return frames


def _root_frames(topology: str, base: Path, seed: int) -> tuple[dict[str, Any], ...]:
    project = base / PROJECT_DIR
    for directory in (project, base / GRANTED_DIR, base / OTHER_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / INNER_FILE).write_text("scenario\n", encoding="utf-8")
    config = base / "config.toml"
    script = build_root_oracle_script(seed)
    with _host(COMPONENT_ROOT_ORACLE, topology) as channel:
        driver = RootOracleDriver(channel)
        ready = driver.initialize(base=base, project=project, config=config)
        outcome = driver.run(script)
    assert ready["type"] == "ready"
    assert outcome.texts == tuple(turn.prompt for turn in script.turns)
    return (ready, *outcome.frames)


def _shutdown_frames(
    topology: str, base: Path, seed: int
) -> tuple[dict[str, Any], ...]:
    base.mkdir(parents=True, exist_ok=True)
    ops: list[dict[str, Any]] = [
        {"kind": "lease_acquire"},
        {"kind": "lease_conflict"},
        {"kind": "lease_release"},
        {"kind": "shutdown_sequence"},
    ]
    with _host(COMPONENT_LIFECYCLE, topology) as channel:
        driver = LifecycleDriver(channel)
        driver.initialize({"base": str(base)})
        frames = driver.run(ops)
    assert frames[0]["diagnostic"]["session_id"] == LEASE_SESSION
    return frames


TRACE_RUNNERS = {
    "replay": _replay_frames,
    "presentation": _presentation_frames,
    "child_crash": _crash_frames,
    "root": _root_frames,
    "shutdown": _shutdown_frames,
}


def _digests_across_modes(
    runner: Any, tmp_path: Path, seed: int
) -> dict[tuple[str, str], tuple[dict[str, Any], ...]]:
    """Run one trace in both modes and both topologies; return every frame stream.

    Each host runs the trace twice into its own directory first: a trace whose
    own repeat differs is nondeterministic, and that is reported as such instead
    of as a cross-topology divergence.
    """
    runs: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
    for mode in MODES:
        with transport_mode(mode):
            for topology in TOPOLOGIES:
                first = normalize_frames(
                    runner(topology, tmp_path / f"{mode}-{topology}-1", seed)
                )
                second = normalize_frames(
                    runner(topology, tmp_path / f"{mode}-{topology}-2", seed)
                )
                difference = _first_difference(first, second)
                assert difference is None, (
                    f"{mode}/{topology} is not deterministic across its own "
                    f"repeat: {difference}"
                )
                runs[(mode, topology)] = first
    return runs


def _first_difference(
    reference: object, actual: object, path: str = "frames"
) -> str | None:
    """The first differing path and both values, for a readable gate failure."""
    if type(reference) is not type(actual):
        return f"{path}: {type(reference).__name__} != {type(actual).__name__}"
    if isinstance(reference, dict) and isinstance(actual, dict):
        for key in sorted(set(reference) | set(actual)):
            if key not in reference or key not in actual:
                return f"{path}.{key}: present on one side only"
            difference = _first_difference(reference[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(reference, list) and isinstance(actual, list):
        if len(reference) != len(actual):
            return f"{path}: {len(reference)} != {len(actual)} frames"
        for index, (left, right) in enumerate(zip(reference, actual, strict=True)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if reference != actual:
        return f"{path}: {reference!r} != {actual!r}"
    return None


def _assert_one_stream(
    runs: Mapping[tuple[str, str], tuple[dict[str, Any], ...]], name: str
) -> None:
    reference_key = (MODES[0], TOPOLOGIES[0])
    reference = runs[reference_key]
    for key, frames in runs.items():
        difference = _first_difference(reference, frames)
        if difference is not None:
            raise AssertionError(
                f"{name} diverged between {reference_key} and {key}: {difference}"
            )
    assert len({digest(frames) for frames in runs.values()}) == 1


@pytest.mark.timeout(300)
@pytest.mark.parametrize("family", sorted(TRACE_RUNNERS))
def test_family_trace_digest_matches_across_modes(family: str, tmp_path: Path) -> None:
    runs = _digests_across_modes(TRACE_RUNNERS[family], tmp_path, DEFAULT_SEED)
    _assert_one_stream(runs, family)


# -- the selector is real ----------------------------------------------------
def _probe_report(channel: Any) -> dict[str, Any]:
    channel.send({"type": "report"})
    frame = channel.receive()
    assert frame is not None and frame.get("type") == "report", frame
    return frame


def _probe_connect(channel: Any, host: str, port: int) -> dict[str, Any]:
    channel.send({"type": "connect", "address": [host, port]})
    frame = channel.receive()
    assert frame is not None and frame.get("type") == "connect", frame
    return frame


@pytest.mark.timeout(120)
@pytest.mark.parametrize("mode", MODES)
def test_transport_selector_reaches_both_hosted_components(mode: str) -> None:
    with transport_mode(mode):
        with _host(COMPONENT_PROBES, SPAWNED, timeout=60.0) as channel:
            spawned = _probe_report(channel)
        with _host(COMPONENT_PROBES, IN_PROCESS, timeout=60.0) as channel:
            hosted = _probe_report(channel)

    expected = {"kind": mode, "selector": mode, "source": "environment"}
    # The mode is not a label on the driver: the component resolves it.
    assert spawned["selection"] == expected
    assert hosted["selection"] == expected
    # And the [6A] spawn-env marker reached the transport in both hosts.
    assert spawned["egress_marker"] is True
    assert spawned["egress_mode"] == egress.MODE_ALL
    assert hosted["egress_marker"] is True
    assert hosted["egress_mode"] == egress.MODE_NON_LOOPBACK


@pytest.mark.timeout(120)
def test_spawned_component_reaches_no_provider_or_non_loopback_destination() -> None:
    server = StreamingMockServer()
    server.start()
    try:
        parts = urlsplit(server.api_base)
        assert parts.hostname is not None and parts.port is not None
        host, port = parts.hostname, parts.port

        with transport_mode(STDIO):
            with _host(COMPONENT_PROBES, SPAWNED, timeout=60.0) as channel:
                loopback = _probe_connect(channel, host, port)
                elsewhere = _probe_connect(channel, NON_LOOPBACK_DESTINATION, 443)
            # Deny-all mode: the spawned component reaches neither endpoint.
            assert loopback["outcome"] == "denied"
            assert "provider network blocked by test isolation" in str(
                loopback["error"]
            )
            assert elsewhere["outcome"] == "denied"
            assert "provider network blocked by test isolation" in str(
                elsewhere["error"]
            )

            # Control: the endpoint is live, so the denial above is the guard,
            # not an unreachable provider; and the in-process host keeps its
            # loopback allowance while still refusing a non-loopback address.
            with _host(COMPONENT_PROBES, IN_PROCESS, timeout=60.0) as channel:
                reachable = _probe_connect(channel, host, port)
                blocked = _probe_connect(channel, NON_LOOPBACK_DESTINATION, 443)
            assert reachable["outcome"] == "connected"
            assert blocked["outcome"] == "denied"
    finally:
        server.stop()


# -- the full-suite gate -----------------------------------------------------
def _run_trace_suite(mode: str, ledger_dir: Path, basetemp: Path) -> Any:
    environment = os.environ.copy()
    environment[TRANSPORT_ENV_VAR] = mode
    environment[ledger.LEDGER_ENV] = str(ledger_dir)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        *TRACE_MODULES,
        "-q",
        "-n",
        _SUITE_WORKERS,
        "-p",
        "no:cacheprovider",
        f"--basetemp={basetemp}",
    ]
    return subprocess.run(
        command,
        cwd=str(REPLAY_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=_SUITE_TIMEOUT,
    )


@pytest.mark.timeout(900)
def test_full_differential_suite_digests_match_across_modes(tmp_path: Path) -> None:
    entries: dict[str, dict[str, dict[str, object]]] = {}
    for mode in MODES:
        ledger_dir = tmp_path / f"ledger-{mode}"
        completed = _run_trace_suite(mode, ledger_dir, tmp_path / f"basetemp-{mode}")
        assert completed.returncode == 0, (
            f"the differential suite failed with {TRANSPORT_ENV_VAR}={mode}:\n"
            f"{completed.stdout[-6000:]}\n{completed.stderr[-3000:]}"
        )
        entries[mode] = ledger.read(ledger_dir)

    # Every family took part, and all nineteen presentation traces ran.
    for mode in MODES:
        counts = ledger.families(entries[mode])
        assert counts.get(ledger.FAMILY_PRESENTATION, 0) >= 19, counts
        assert counts.get(ledger.FAMILY_ROOT, 0) >= 1, counts
        assert counts.get(ledger.FAMILY_CHILD_CRASH, 0) >= 1, counts
        assert counts.get(ledger.FAMILY_REPLAY, 0) >= 1, counts
        assert counts.get(ledger.FAMILY_SHUTDOWN, 0) >= 1, counts

    stdio, in_process = entries[STDIO], entries[IN_PROCESS]
    assert set(stdio) == set(in_process)
    mismatched = {
        key: (stdio[key]["digest"], in_process[key]["digest"])
        for key in stdio
        if stdio[key]["digest"] != in_process[key]["digest"]
    }
    assert mismatched == {}
