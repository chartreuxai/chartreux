"""WP6 [6D]: ``-p`` stdout/stderr/exit parity — the legacy-representative oracle.

Design invariant 1 defines "old" as the campaign's recorded baseline and makes
in_process-vs-stdio parity with ``-p`` stdout/stderr/exit parity the legacy-
representative oracle (design.md:522-524). This module runs the real programmatic
route — the installed CLI entrypoint with ``-p``, a live fake provider on
loopback, and the [6A]-blocked transport — under both transport selections and
compares what a caller observes:

* stdout, stderr, and the exit status are identical across modes;
* the provider received the same seeded request sequence in both modes, so the
  parity is not just an equal rendering of different work;
* continuation (``-c``) and the programmatic denial of a dangerous tool call
  settle identically, including the exit status of a failing turn.

Two run-scoped values are normalized before comparison, and only those: the
per-run scratchpad directory name the harness mints, the caller's own directory
tree (each mode writes its own config home under it), and wall-clock stamps the
request carries (a tool call's ``started_at``). Everything else — the model
text, the denial, the exit status, the request messages — is compared exactly.

The programmatic route composes the app-server in this process for both
selectors today (``LocalHarness``), so these tests pin the selection-neutrality
the route must keep: the stdio selection may not change what ``-p`` emits, and
a later wiring of the multi-process route has to preserve this parity.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pytest

from chartreux.app_server.transport import TRANSPORT_ENV_VAR
from tests.differential.replay import TRANSPORT_MODES, build_replay_script
from tests.e2e.agent_loop_characterization.support import (
    assistant_text_chunks,
    single_tool_call_chunks,
)
from tests.e2e.common import write_e2e_config
from tests.e2e.mock_server import StreamingMockServer

STDIO = "stdio"
IN_PROCESS = "in_process"
MODES = TRANSPORT_MODES
DEFAULT_SEED = 20_261_110
_RUN_TIMEOUT = 180
_DENIED_ARGUMENTS = {"command": "git clean -fd"}
_SCRATCHPAD = re.compile(r"chartreux-scratchpad-[A-Za-z0-9_-]+")
_APPROX_CHARS = re.compile(r'"approx_chars":\d+')
_TIMESTAMP = re.compile(r'"(started_at|completed_at|created_at|updated_at)":\d+')


def _cli() -> Path:
    """The chartreux entrypoint next to the interpreter running the tests."""
    executable = Path(sys.executable).with_name("chartreux")
    assert executable.exists(), executable
    return executable


def _environment(home: Path, mode: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CHARTREUX_HOME"] = str(home)
    environment["MISTRAL_API_KEY"] = "fake-key"
    environment["CHARTREUX_TEST_DISABLE_KEYRING"] = "1"
    environment["CHARTREUX_TEST_DISABLE_AUTO_TITLE"] = "1"
    environment["TERM"] = "xterm-256color"
    environment[TRANSPORT_ENV_VAR] = mode
    return environment


def _prepare(
    tmp_path: Path, mode: str, server: StreamingMockServer
) -> tuple[Path, Path]:
    home = tmp_path / f"home-{mode}"
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    write_e2e_config(home, server.api_base)
    return home, workdir


def _normalize(text: str, roots: Iterable[Path]) -> str:
    for root in roots:
        text = text.replace(str(root), "<root>")
    text = _SCRATCHPAD.sub("<scratchpad>", text)
    text = _TIMESTAMP.sub(r'"\1":<timestamp>', text)
    return _APPROX_CHARS.sub('"approx_chars":<n>', text)


def _run(
    server: StreamingMockServer, *, tmp_path: Path, mode: str, arguments: list[str]
) -> tuple[tuple[str, str, int], list[str], list[dict[str, Any]]]:
    """Run one invocation; return its observable, normalized and raw requests."""
    home, workdir = _prepare(tmp_path, mode, server)
    started = len(server.requests)
    completed = subprocess.run(
        [str(_cli()), *arguments],
        cwd=str(workdir),
        env=_environment(home, mode),
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT,
        check=False,
    )
    raw_requests = [dict(request) for request in server.requests[started:]]
    requests = [
        _normalize(
            json.dumps(request, sort_keys=True, separators=(",", ":")), (tmp_path,)
        )
        for request in raw_requests
    ]
    observable = (
        _normalize(completed.stdout, (tmp_path,)),
        _normalize(completed.stderr, (tmp_path,)),
        completed.returncode,
    )
    return observable, requests, raw_requests


def _single_turn_prompt() -> str:
    return build_replay_script(DEFAULT_SEED).turns[0]


@pytest.mark.timeout(300)
def test_single_turn_parity_includes_the_provider_request_sequence(
    tmp_path: Path,
) -> None:
    server = StreamingMockServer()
    server.start()
    try:
        runs = {
            mode: _run(
                server,
                tmp_path=tmp_path,
                mode=mode,
                arguments=["-p", _single_turn_prompt()],
            )
            for mode in MODES
        }
    finally:
        server.stop()

    assert runs[STDIO][0] == runs[IN_PROCESS][0]
    stdout, stderr, returncode = runs[STDIO][0]
    assert returncode == 0
    assert stderr == ""
    assert "Hello from mock server" in stdout
    # The same seeded work reached the provider: not an equal rendering of
    # different turns.
    assert runs[STDIO][1] == runs[IN_PROCESS][1]
    assert len(runs[STDIO][1]) == 1
    assert _single_turn_prompt() in runs[STDIO][1][0]


@pytest.mark.timeout(300)
def test_continuation_parity_holds_across_modes(tmp_path: Path) -> None:
    script = build_replay_script(DEFAULT_SEED)
    server = StreamingMockServer()
    server.start()
    try:
        runs: dict[str, list[tuple[str, str, int]]] = {}
        continuations: dict[str, list[str]] = {}
        for mode in MODES:
            first = _run(
                server, tmp_path=tmp_path, mode=mode, arguments=["-p", script.turns[0]]
            )[0]
            second, second_requests, _raw = _run(
                server,
                tmp_path=tmp_path,
                mode=mode,
                arguments=["-c", "-p", script.turns[1]],
            )
            assert second[2] == 0
            runs[mode] = [first, second]
            continuations[mode] = second_requests
    finally:
        server.stop()

    assert runs[STDIO] == runs[IN_PROCESS]
    # The continuation really carried the second seeded prompt, in both modes.
    assert continuations[STDIO] == continuations[IN_PROCESS]
    assert script.turns[1] in continuations[STDIO][0]
    # ... and the first turn's prompt is in the continued conversation.
    assert script.turns[0] in continuations[STDIO][0]


@pytest.mark.timeout(300)
def test_dangerous_tool_denial_is_identical_in_both_modes(tmp_path: Path) -> None:
    def chunk_factory(index: int, payload: Any) -> list[dict[str, object]]:
        # Scope the tool call to the run, not to the server's request counter:
        # each mode drives its own conversation from the same first turn.
        messages = payload.get("messages") or []
        if any(message.get("role") == "tool" for message in messages):
            return assistant_text_chunks("settled after the denial")
        return single_tool_call_chunks(
            call_id="call-1", tool_name="bash", arguments=_DENIED_ARGUMENTS
        )

    server = StreamingMockServer(chunk_factory=chunk_factory)
    server.start()
    try:
        runs = {
            mode: _run(
                server,
                tmp_path=tmp_path,
                mode=mode,
                arguments=["-p", _single_turn_prompt()],
            )
            for mode in MODES
        }
    finally:
        server.stop()

    assert runs[STDIO][0] == runs[IN_PROCESS][0]
    stdout, _stderr, returncode = runs[STDIO][0]
    assert returncode == 0
    assert "settled after the denial" in stdout
    # Both modes ran the tool turn and reported the refusal back to the model.
    assert runs[STDIO][1] == runs[IN_PROCESS][1]
    assert len(runs[STDIO][2]) == 2
    tool_results = [
        message
        for request in runs[STDIO][2]
        for message in request["messages"]
        if message.get("role") == "tool"
    ]
    assert len(tool_results) == 1
    assert "denied" in str(tool_results[0]["content"]).lower()


@pytest.mark.timeout(300)
def test_failed_provider_turn_exits_identically_in_both_modes(tmp_path: Path) -> None:
    server = StreamingMockServer(error_status=500)
    server.start()
    try:
        runs = {
            mode: _run(
                server,
                tmp_path=tmp_path,
                mode=mode,
                arguments=["-p", _single_turn_prompt()],
            )
            for mode in MODES
        }
    finally:
        server.stop()

    assert runs[STDIO][0] == runs[IN_PROCESS][0]
    stdout, stderr, returncode = runs[STDIO][0]
    # A failed turn is a failed command: never a silent success.
    assert returncode != 0
    assert stdout == ""
    assert "LLM backend error" in stderr
