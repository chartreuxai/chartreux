"""Standing secret scan (WP1 [1C]): provider keys never reach the wire or disk.

The coordinator may inherit credential variables — it passes its environment
through unread (design §Credentials) — but no JSON-RPC frame, log line,
transcript, or crash report may carry the value.
"""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator
import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server import (
    _runtime as runtime,
    local as local_mod,
    stdio as stdio_mod,
)
from chartreux.app_server.local import ClientDescriptor, LocalHarnessOptions
from chartreux.app_server.process_bootstrap import (
    BOOTSTRAP_ENV_VAR,
    encode_ready_frame,
    spawn_process,
)
from chartreux.app_server.protocol import ClientInfo, SessionOptions
from chartreux.app_server.transport import MemoryJsonRpcTransport, memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from tests.conftest import build_test_vibe_config
from tests.differential import egress
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator
from tests.stubs.fake_mcp_registry import FakeMCPRegistry

# Longer than the redactor's direct-match floor, so a leak would be realistic.
# Split so the source carries no token the repository secret scan flags.
_CANARY = "sk-" + "canary-0123456789abcdef"


class _RecordingTransport:
    """Delegating transport that keeps every frame it puts on the wire."""

    def __init__(self, inner: MemoryJsonRpcTransport) -> None:
        self._inner = inner
        self.frames: list[str] = []

    async def send(self, message: dict[str, Any]) -> None:
        self.frames.append(json.dumps(message, separators=(",", ":")))
        await self._inner.send(message)

    def messages(self) -> AsyncIterator[dict[str, Any]]:
        return self._inner.messages()

    async def close(self) -> None:
        await self._inner.close()


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class _Surfaces:
    """Every place the run could have exposed a credential."""

    def __init__(self) -> None:
        self.frames: list[str] = []
        self.logs: list[str] = []
        self.transcripts: list[str] = []
        self.stdout = ""
        self.stderr = ""

    def payloads(self) -> list[str]:
        return [*self.frames, *self.logs, *self.transcripts, self.stdout, self.stderr]


def _options() -> LocalHarnessOptions:
    return LocalHarnessOptions(
        client=ClientDescriptor(
            info=ClientInfo(
                name="vibe_programmatic", version="test", entrypoint="programmatic"
            )
        ),
        session_options=SessionOptions(
            disabled_tools=["ask_user_question"], headless=True
        ),
    )


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch, *, transcripts: Path, backend: FakeBackend
) -> None:
    config = build_test_vibe_config(
        include_model_info=False,
        include_commit_signature=False,
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(transcripts), generate_titles=False
        ),
    )
    monkeypatch.setattr(
        runtime,
        "build_default_orchestrator",
        AsyncMock(return_value=FakeConfigOrchestrator(config)),
    )
    monkeypatch.setattr(
        runtime.HarnessProcess,
        "_build_mcp_registry_impl",
        AsyncMock(return_value=FakeMCPRegistry()),
    )
    real_loop = runtime.AgentLoop

    def build_loop(**kwargs: Any) -> AgentLoop:
        return real_loop(backend=backend, **kwargs)

    monkeypatch.setattr(runtime, "AgentLoop", build_loop)


def _transcripts(directory: Path) -> list[str]:
    return [
        path.read_text(encoding="utf-8")
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _assert_secret_absent(surfaces: _Surfaces) -> None:
    for payload in surfaces.payloads():
        assert _CANARY not in payload


def test_no_secret_in_frames_logs_transcripts_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from chartreux.cli.programmatic import OutputFormat, run_programmatic

    sessions = tmp_path / "sessions"
    monkeypatch.setenv("MISTRAL_API_KEY", _CANARY)
    _install_fake_runtime(
        monkeypatch,
        transcripts=sessions,
        backend=FakeBackend([mock_llm_chunk(content="Answer without the key.")]),
    )

    surfaces = _Surfaces()
    recorders: list[_RecordingTransport] = []

    def recording_pair() -> tuple[_RecordingTransport, _RecordingTransport]:
        client_side, server_side = memory_transport_pair()
        recorded = (_RecordingTransport(client_side), _RecordingTransport(server_side))
        recorders.extend(recorded)
        return recorded

    monkeypatch.setattr(local_mod, "memory_transport_pair", recording_pair)

    handler = _RecordingHandler()
    logger = logging.getLogger("vibe")
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        result = run_programmatic(
            harness_options=_options(),
            prompt="Ask about the weather",
            output_format=OutputFormat.TEXT,
        )
    finally:
        logger.setLevel(previous_level)
        logger.removeHandler(handler)

    surfaces.frames = [frame for recorder in recorders for frame in recorder.frames]
    surfaces.logs = handler.messages
    surfaces.transcripts = _transcripts(sessions)
    captured = capsys.readouterr()
    surfaces.stdout = captured.out
    surfaces.stderr = captured.err

    # The scan is only meaningful if it swept real traffic and real journals.
    assert result == "Answer without the key."
    assert surfaces.frames
    assert any("jsonrpc" in frame for frame in surfaces.frames)
    assert any("Ask about the weather" in frame for frame in surfaces.frames)
    assert any("USER: Ask about the weather" in line for line in surfaces.logs)
    assert surfaces.transcripts
    assert any("Answer without the key." in text for text in surfaces.transcripts)

    _assert_secret_absent(surfaces)


def test_no_secret_in_a_crash_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from chartreux.cli.programmatic import OutputFormat, run_programmatic

    sessions = tmp_path / "sessions"
    monkeypatch.setenv("MISTRAL_API_KEY", _CANARY)
    _install_fake_runtime(
        monkeypatch,
        transcripts=sessions,
        backend=FakeBackend(exception_to_raise=RuntimeError("provider exploded")),
    )

    crash = ""
    try:
        run_programmatic(
            harness_options=_options(),
            prompt="Ask about the weather",
            output_format=OutputFormat.TEXT,
        )
    except Exception as exc:  # the rendered crash text is the scan target
        crash = "".join(traceback.format_exception(exc))

    assert crash
    captured = capsys.readouterr()
    surfaces = _Surfaces()
    surfaces.transcripts = [crash, *_transcripts(sessions)]
    surfaces.stdout = captured.out
    surfaces.stderr = captured.err

    _assert_secret_absent(surfaces)


@pytest.mark.asyncio
async def test_spawn_env_passes_credentials_and_egress_markers_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker_env = egress.child_environment(egress.MODE_NON_LOOPBACK)
    monkeypatch.setenv("MISTRAL_API_KEY", _CANARY)
    for name, value in marker_env.items():
        monkeypatch.setenv(name, value)

    sink = tmp_path / "received.json"
    observed = (
        "MISTRAL_API_KEY",
        egress.EGRESS_BLOCK_ENV,
        egress.EGRESS_GUARD_ENV,
        BOOTSTRAP_ENV_VAR,
    )
    ready = encode_ready_frame().decode().strip()
    script = (
        "import json, os, sys\n"
        f"sys.stdout.write({ready!r} + '\\n')\n"
        "sys.stdout.flush()\n"
        f"with open({str(sink)!r}, 'w', encoding='utf-8') as handle:\n"
        f"    json.dump({{name: os.environ.get(name) for name in {observed!r}}}, handle)\n"
    )
    spawned = await spawn_process([sys.executable, "-c", script], ready_timeout=5)
    await spawned.transport.close()
    assert spawned.process.wait(timeout=5) == 0

    received = json.loads(sink.read_text(encoding="utf-8"))
    # Pass-through is unread: the coordinator never resolves the credential, but
    # the spawned agent inherits it at spawn and loads its own local sources.
    assert received["MISTRAL_API_KEY"] == _CANARY
    assert received[egress.EGRESS_BLOCK_ENV] == egress.MODE_NON_LOOPBACK
    assert received[egress.EGRESS_GUARD_ENV] == marker_env[egress.EGRESS_GUARD_ENV]
    assert received[BOOTSTRAP_ENV_VAR] == "1"
    # The only frame the child writes by itself carries protocol state only.
    assert _CANARY not in encode_ready_frame().decode()


def test_spawned_agent_loads_local_sources_before_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_serve(**_kwargs: Any) -> None:
        calls.append("serve")

    monkeypatch.setattr(stdio_mod, "parse_arguments", lambda: argparse.Namespace())
    monkeypatch.setattr(
        stdio_mod, "init_harness_files_manager", lambda *_args: calls.append("files")
    )
    monkeypatch.setattr(
        stdio_mod,
        "init_file_logging",
        lambda *_args, **_kwargs: calls.append("logging"),
    )
    monkeypatch.setattr(
        "chartreux.core.config.load_dotenv_values", lambda: calls.append("dotenv")
    )
    monkeypatch.setattr(stdio_mod, "serve_stdio", fake_serve)
    monkeypatch.setattr(stdio_mod, "bootstrap_requested", lambda: False)

    stdio_mod.main()

    # The agent process reads its own dotenv at spawn; the coordinator does not.
    assert calls == ["files", "logging", "dotenv", "serve"]
