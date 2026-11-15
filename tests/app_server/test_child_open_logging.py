"""F9/W1: the child's open-failure logging covers its whole pre-open body.

F6 wrapped only ``open_root`` in the traceback log, so everything before
it -- the spawn-spec decode, the root-dir fail-closed check, AgentConfig
and ClientInfo validation, the inherited-workspace decode -- answered its
typed wire error with NO traceback in the logs (the RCA session behind
this had zero). The logging now wraps the whole open, and a cancellation
no longer lands as an exception traceback (glm info #3, cosmetic).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError
import pytest

from chartreux.app_server._subagent_transcript import (
    AGENT_CONFIG_PAYLOAD_KEY,
    INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY,
    INHERITED_RESTRICTIONS_PAYLOAD_KEY,
    INHERITED_ROOT_GRANTS_PAYLOAD_KEY,
    INHERITED_WORKSPACE_PAYLOAD_KEY,
    ChildTranscriptPayloadError,
    encode_inherited_workspace,
)
from chartreux.app_server.child_agent import HarnessChildLoopFactory
from chartreux.app_server.subagent_ports import child_spawn_spec, spawn_spec_to_wire
from chartreux.core.workspace import Workspace

_ROOT_SESSION_ID = "root-session-open-logging"


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "sessions" / _ROOT_SESSION_ID
    root.mkdir(parents=True)
    return root


class _CollectingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _open_failure_records(handler: _CollectingHandler) -> list[logging.LogRecord]:
    return [
        record
        for record in handler.records
        if "Child agent loop open failed" in record.getMessage()
    ]


class _Process:
    """The HarnessProcess seam: the only failure point under test."""

    async def open_root(self, request: Any) -> Any:
        raise AssertionError("open_root must not be reached in these pins")

    async def close(self) -> None:
        return None


class _FailingProcess(_Process):
    def __init__(self, failure: BaseException) -> None:
        self._failure = failure

    async def open_root(self, request: Any) -> Any:
        raise self._failure


def _factory(process: _Process) -> HarnessChildLoopFactory:
    factory = object.__new__(HarnessChildLoopFactory)
    factory._process = cast(Any, process)
    return factory


def _spec(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return spawn_spec_to_wire(
        child_spawn_spec(
            agent_id="coder:py",
            parent_session_id=_ROOT_SESSION_ID,
            root_session_dir=root,
            payload=payload,
            attempt_id="attempt-open-logging-1",
        )
    )


_GOOD_PAYLOAD: dict[str, Any] = {
    "client_info": {"name": "tests", "version": "1.0"},
    INHERITED_WORKSPACE_PAYLOAD_KEY: encode_inherited_workspace(
        Workspace(cwd=Path("/project"), authorized_roots=(Path("/project"),))
    ),
    # F11: the parent-policy keys a pre-open body consumes. An empty policy
    # is a legal shape here -- these pins exercise the open's failure
    # logging, not the policy values.
    AGENT_CONFIG_PAYLOAD_KEY: {},
    INHERITED_RESTRICTIONS_PAYLOAD_KEY: [],
    INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY: [],
    INHERITED_ROOT_GRANTS_PAYLOAD_KEY: [],
}


@pytest.mark.asyncio
async def test_a_malformed_workspace_payload_logs_a_traceback(tmp_path: Path) -> None:
    # The pre-try decode failure: a strict-codec ChildTranscriptPayloadError
    # answers the wire, and the shared "vibe" logger carries the traceback.
    handler = _CollectingHandler()
    shared = logging.getLogger("vibe")
    shared.addHandler(handler)
    try:
        payload = dict(_GOOD_PAYLOAD)
        payload[INHERITED_WORKSPACE_PAYLOAD_KEY] = {"cwd": 1}
        with pytest.raises(ChildTranscriptPayloadError):
            await _factory(_Process()).open(_spec(_root(tmp_path), payload))
    finally:
        shared.removeHandler(handler)

    failures = _open_failure_records(handler)
    assert failures, handler.records
    assert failures[-1].levelno == logging.ERROR
    assert failures[-1].exc_info is not None


@pytest.mark.asyncio
async def test_a_malformed_client_info_logs_a_traceback(tmp_path: Path) -> None:
    # The pre-try payload validation: a ClientInfo missing its version
    # fails the same logged way as the decode above.
    handler = _CollectingHandler()
    shared = logging.getLogger("vibe")
    shared.addHandler(handler)
    try:
        with pytest.raises(ValidationError):
            await _factory(_Process()).open(
                _spec(_root(tmp_path), {"client_info": {"name": "tests"}})
            )
    finally:
        shared.removeHandler(handler)

    failures = _open_failure_records(handler)
    assert failures, handler.records
    assert failures[-1].levelno == logging.ERROR
    assert failures[-1].exc_info is not None


@pytest.mark.asyncio
async def test_a_loop_open_failure_logs_a_traceback(tmp_path: Path) -> None:
    # The F6 contract stays: a failure inside the process's loop open is
    # logged with its traceback before the wire error is answered.
    handler = _CollectingHandler()
    shared = logging.getLogger("vibe")
    shared.addHandler(handler)
    try:
        with pytest.raises(RuntimeError, match="loop open boom"):
            await _factory(_FailingProcess(RuntimeError("loop open boom"))).open(
                _spec(_root(tmp_path), dict(_GOOD_PAYLOAD))
            )
    finally:
        shared.removeHandler(handler)

    failures = _open_failure_records(handler)
    assert failures, handler.records
    assert failures[-1].levelno == logging.ERROR
    assert failures[-1].exc_info is not None


@pytest.mark.asyncio
async def test_cancellation_is_not_logged_as_a_traceback(tmp_path: Path) -> None:
    # Cancellation is control flow, not an open failure: it propagates
    # without the traceback record the rotating file does not need.
    handler = _CollectingHandler()
    shared = logging.getLogger("vibe")
    shared.addHandler(handler)
    try:
        with pytest.raises(asyncio.CancelledError):
            await _factory(_FailingProcess(asyncio.CancelledError())).open(
                _spec(_root(tmp_path), dict(_GOOD_PAYLOAD))
            )
    finally:
        shared.removeHandler(handler)

    assert _open_failure_records(handler) == []
