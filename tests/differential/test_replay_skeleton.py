"""WP6 [6A]: seeded replay skeleton, component adapter, and egress-block trace.

The parity test runs one seeded script through the same component function with
and without a process boundary; the egress traces assert that a spawned test
process reaches neither a fake provider endpoint nor a non-loopback destination
and that the marker, not luck, is what blocks it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

from tests.differential import egress, ledger
from tests.differential.replay import (
    REPLAY_ROOT,
    ReplayDriver,
    ReplayScript,
    build_replay_script,
    digest,
    in_process_component,
    normalize_frames,
    subprocess_component,
)
from tests.e2e.mock_server import StreamingMockServer

NON_LOOPBACK_DESTINATION = "192.0.2.1"


def _python_path(*directories: Path) -> str:
    inherited = os.environ.get("PYTHONPATH", "")
    entries = [str(directory) for directory in directories]
    if inherited:
        entries.append(inherited)
    return os.pathsep.join(entries)


def _spawn_probe(host: str, port: int, *, mode: str | None) -> dict[str, object]:
    environment = os.environ.copy()
    if mode is None:
        environment.pop(egress.EGRESS_BLOCK_ENV, None)
        environment.pop(egress.EGRESS_GUARD_ENV, None)
    else:
        environment.update(egress.child_environment(mode))
    with egress.bootstrap_directory() as guard_dir:
        environment["PYTHONPATH"] = _python_path(guard_dir, REPLAY_ROOT)
        completed = subprocess.run(
            [sys.executable, "-m", "tests.differential.egress", host, str(port)],
            cwd=str(REPLAY_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=8,
        )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_seeded_script_is_deterministic() -> None:
    assert build_replay_script(7) == build_replay_script(7)
    assert build_replay_script(7) != build_replay_script(8)
    assert len(build_replay_script(7).turns) == 3


def test_normalization_drops_only_process_volatile_fields() -> None:
    frames = (
        {"type": "delta", "run": "run-101", "index": 0, "text": "a", "clock": 1.5},
        {"type": "delta", "run": "run-101", "index": 1, "text": "b", "clock": 2.5},
        {
            "type": "turn_end",
            "run": "run-101",
            "sequence": 0,
            "text": "a b",
            "clock": 3,
        },
    )
    normalized = normalize_frames(frames)
    assert [frame["run"] for frame in normalized] == ["run-0", "run-0", "run-0"]
    assert all("clock" not in frame for frame in normalized)
    assert normalized[0]["text"] == "a"
    assert normalized[2]["sequence"] == 0


def test_in_process_and_subprocess_topologies_agree(
    replay_script: ReplayScript,
) -> None:
    script = replay_script

    with in_process_component() as channel:
        in_process = ReplayDriver(channel).run(script)
    with subprocess_component() as channel:
        in_subprocess = ReplayDriver(channel).run(script)

    assert in_process.texts == script.turns
    assert in_subprocess.texts == script.turns
    assert in_process.frames and in_subprocess.frames
    # The adapter really removed a process boundary: run identities differ.
    assert in_process.frames[0]["run"] != in_subprocess.frames[0]["run"]
    assert normalize_frames(in_process.frames) == normalize_frames(in_subprocess.frames)
    assert digest(normalize_frames(in_process.frames)) == digest(
        normalize_frames(in_subprocess.frames)
    )
    ledger.record(
        ledger.FAMILY_REPLAY,
        {"seed": script.seed, "turns": list(script.turns)},
        in_process.frames,
    )


def test_spawned_process_egress_block_is_marker_controlled() -> None:
    server = StreamingMockServer()
    server.start()
    try:
        parts = urlsplit(server.api_base)
        assert parts.hostname is not None and parts.port is not None
        host, port = parts.hostname, parts.port

        # Control: without the marker the fake provider endpoint is reachable,
        # so a denial below cannot be an unreachable endpoint.
        assert _spawn_probe(host, port, mode=None) == {"outcome": "connected"}
        # Deny-all mode leaves no INET path at all, loopback included.
        denied = _spawn_probe(host, port, mode=egress.MODE_ALL)
        assert denied["outcome"] == "denied"
        assert denied["error_type"] == "OSError"
        assert "provider network blocked by test isolation" in str(denied["error"])
        # Deny-all-except-loopback keeps the local endpoint available.
        assert _spawn_probe(host, port, mode=egress.MODE_NON_LOOPBACK) == {
            "outcome": "connected"
        }
    finally:
        server.stop()


def test_spawned_process_cannot_reach_a_non_loopback_destination() -> None:
    for mode in (egress.MODE_NON_LOOPBACK, egress.MODE_ALL):
        result = _spawn_probe(NON_LOOPBACK_DESTINATION, 443, mode=mode)
        assert result["outcome"] == "denied", result
        assert result["error_type"] == "OSError", result
        # An unguarded connect would time out with TimeoutError; the guard's own
        # refusal proves the spawn-env marker reached the transport.
        assert "provider network blocked by test isolation" in str(result["error"])


def test_replay_child_refuses_to_run_without_the_marker() -> None:
    environment = os.environ.copy()
    environment.pop(egress.EGRESS_BLOCK_ENV, None)
    environment.pop(egress.EGRESS_GUARD_ENV, None)
    environment["PYTHONPATH"] = _python_path(REPLAY_ROOT)
    completed = subprocess.run(
        [sys.executable, "-m", "tests.differential.replay_child"],
        cwd=str(REPLAY_ROOT),
        env=environment,
        input="",
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert completed.returncode != 0
    assert "requires the egress marker" in completed.stderr


def test_bootstrap_fails_closed_when_the_guard_path_is_missing() -> None:
    environment = os.environ.copy()
    environment[egress.EGRESS_BLOCK_ENV] = egress.MODE_ALL
    environment.pop(egress.EGRESS_GUARD_ENV, None)
    with egress.bootstrap_directory() as guard_dir:
        environment["PYTHONPATH"] = _python_path(guard_dir, REPLAY_ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", "print('unguarded')"],
            cwd=str(REPLAY_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=8,
        )
    # CPython continues after a failed sitecustomize, so the bootstrap itself
    # must terminate the child: no unguarded run may reach user code.
    assert completed.returncode == egress.EGRESS_BOOTSTRAP_FAILURE_EXIT_CODE
    assert completed.stdout == ""
    assert "egress marker set without" in completed.stderr
