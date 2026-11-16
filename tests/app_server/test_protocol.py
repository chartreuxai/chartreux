from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import httpx
from pydantic import ValidationError
import pytest

from chartreux.app_server._dispatch import DispatchResult, RequestFailure
from chartreux.app_server._runtime import create_harness_server
from chartreux.app_server._session_backend_port import SessionBackendError
from chartreux.app_server.client import AppServerClient, AppServerConnectionClosed
from chartreux.app_server.protocol import (
    AgentTranscriptGetResponse,
    AgentTranscriptState,
    AppServerResponseError,
    CallbackCallResponse,
    ClientCapabilities,
    ClientInfo,
    JsonPatchOperation,
    JsonRpcErrorResponse,
    JsonRpcProtocolError,
    ProtocolError,
    ProtocolErrorCode,
    RootRecoveryAdmitParams,
    RootRecoveryAdmitResponse,
    RootRecoveryCancelParams,
    RootRecoveryCancelResponse,
    RootRecoveryScheduleParams,
    RootRecoveryScheduleResponse,
    ServerRequest,
    SessionOptions,
    SessionStopParams,
    SessionStopResponse,
    SessionUpdatedParams,
    validate_json_rpc_envelope,
)
from chartreux.app_server.server import CallbackDelivery, InitializationState
from chartreux.app_server.session import AppServerTurnError, SessionStopUnknownOutcome
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from chartreux.core.model_catalog.availability import AvailabilityRegistry
from chartreux.core.paths import LOG_FILE
from chartreux.core.subagents import InvalidLaunchConfigError, LaunchConfigError
from tests.app_server.backend_contract.conftest import install_memory_root_agent
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.recovery.conftest import FakeClock
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    legacy_backend,
)
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.parametrize(
    "message",
    [
        {"jsonrpc": "2.0", "id": "client-1"},
        {
            "jsonrpc": "2.0",
            "id": "client-1",
            "result": {},
            "error": {"code": "internal_error", "message": "failed"},
        },
        {"jsonrpc": "2.0", "method": "initialized", "params": {}, "extra": True},
        {"jsonrpc": "2.0", "id": True, "result": {}},
    ],
)
def test_json_rpc_envelopes_reject_malformed_shapes(message: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        validate_json_rpc_envelope(message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "field_path"),
    [
        (LaunchConfigError("config.model"), "config.model"),
        (
            InvalidLaunchConfigError(
                "launch_config.persona",
                "Stored launch identity does not match its link",
            ),
            "launch_config.persona",
        ),
    ],
)
async def test_launch_config_errors_have_stable_protocol_diagnostics(
    error: LaunchConfigError, field_path: str
) -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    server._initialization = InitializationState.INITIALIZED
    server._dispatch_request = AsyncMock(side_effect=error)

    try:
        outcome = await server._dispatch_or_error(
            ServerRequest(id="launch-config", method="session/read", params={})
        )

        assert not isinstance(outcome, DispatchResult)
        assert outcome.error.code is ProtocolErrorCode.INVALID_PARAMS
        assert outcome.error.message == "Invalid launch configuration"
        assert outcome.error.data == {
            "code": "launch_config_error",
            "fieldPath": field_path,
        }
    finally:
        await server.close()
        await client_transport.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_client_rejects_snake_case_response_fields() -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    initialize = asyncio.create_task(
        client.initialize(ClientInfo(name="test", version="1"))
    )
    request = await anext(peer_transport.messages())

    await peer_transport.send({
        "jsonrpc": "2.0",
        "id": request["id"],
        "result": {"server_info": {"name": "test-server", "version": "1"}},
    })

    with pytest.raises(ValidationError):
        await initialize

    await client.close()
    await peer_transport.close()


@pytest.mark.asyncio
async def test_client_rejects_unknown_response_id() -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    pending = asyncio.create_task(client.request("test/wait"))
    await anext(peer_transport.messages())

    await peer_transport.send({"jsonrpc": "2.0", "id": "client-unknown", "result": {}})

    with pytest.raises(AppServerConnectionClosed) as exc_info:
        await pending

    assert isinstance(exc_info.value.__cause__, JsonRpcProtocolError)
    await client.close()
    await peer_transport.close()


@pytest.mark.asyncio
async def test_client_accepts_late_response_to_cancelled_request() -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    cancelled = asyncio.create_task(client.request("test/cancel"))
    cancelled_request = await anext(peer_transport.messages())
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    await peer_transport.send({
        "jsonrpc": "2.0",
        "id": cancelled_request["id"],
        "result": {},
    })
    active = asyncio.create_task(client.request("test/active"))
    active_request = await anext(peer_transport.messages())
    await peer_transport.send({
        "jsonrpc": "2.0",
        "id": active_request["id"],
        "result": {"accepted": True},
    })

    assert await active == {"accepted": True}
    assert client._abandoned_request_ids == set()
    await client.close()
    await peer_transport.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("late_error", [False, True])
async def test_client_accepts_late_response_after_cancellation_during_send(
    monkeypatch: pytest.MonkeyPatch, late_error: bool
) -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    send_enqueued = asyncio.Event()
    release_send = asyncio.Event()
    original_send = client_transport.send

    async def paused_send(message: dict[str, Any]) -> None:
        await original_send(message)
        if message.get("method") == "usage/read":
            # The frame reached the peer, but send() has not returned to request().
            send_enqueued.set()
            await release_send.wait()

    monkeypatch.setattr(client_transport, "send", paused_send)
    cancelled = asyncio.create_task(client.request("usage/read"))
    active: asyncio.Task[dict[str, Any]] | None = None
    try:
        async with asyncio.timeout(2):
            await send_enqueued.wait()
            cancelled_request = await anext(peer_transport.messages())
            future = client._pending[cancelled_request["id"]]
            assert not future.done()
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            # Cancellation during send does not propagate into the response future.
            assert not future.cancelled()
            assert client._abandoned_request_ids == {cancelled_request["id"]}

            late_response: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": cancelled_request["id"],
            }
            if late_error:
                late_response["error"] = {
                    "code": ProtocolErrorCode.INTERNAL_ERROR,
                    "message": "late failure",
                }
            else:
                late_response["result"] = {}
            active = asyncio.create_task(client.request("test/active"))
            active_request = await anext(peer_transport.messages())
            await peer_transport.send(late_response)
            await peer_transport.send({
                "jsonrpc": "2.0",
                "id": active_request["id"],
                "result": {"accepted": True},
            })

            assert await active == {"accepted": True}
            assert client._incoming_error is None
            assert client._reader_task is not None
            assert not client._reader_task.done()
            assert client._abandoned_request_ids == set()
    finally:
        cancelled.cancel()
        if active is not None:
            active.cancel()
        await asyncio.gather(
            cancelled, *([active] if active is not None else []), return_exceptions=True
        )
        await client.close()
        await peer_transport.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("during_send", [False, True])
async def test_client_cancellation_after_response_does_not_abandon_request(
    monkeypatch: pytest.MonkeyPatch, during_send: bool
) -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    send_enqueued = asyncio.Event()
    release_send = asyncio.Event()
    waiting_for_incoming = asyncio.Event()
    original_send = client_transport.send
    original_wait = client._wait_until_processed

    async def paused_send(message: dict[str, Any]) -> None:
        await original_send(message)
        send_enqueued.set()
        if during_send:
            await release_send.wait()

    async def observed_wait(sequence: int) -> None:
        waiting_for_incoming.set()
        await original_wait(sequence)

    monkeypatch.setattr(client_transport, "send", paused_send)
    monkeypatch.setattr(client, "_wait_until_processed", observed_wait)
    cancelled = asyncio.create_task(
        client.request("usage/read", wait_for_incoming=True)
    )
    try:
        async with asyncio.timeout(2):
            await send_enqueued.wait()
            request = await anext(peer_transport.messages())
            future = client._pending[request["id"]]
            # Leave a notification unprocessed so wait_for_incoming stays suspended.
            await peer_transport.send({
                "jsonrpc": "2.0",
                "method": "test/before",
                "params": {},
            })
            await peer_transport.send({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"accepted": True},
            })
            response = await asyncio.shield(future)
            assert response.result == {"accepted": True}
            if not during_send:
                await waiting_for_incoming.wait()
            assert not cancelled.done()
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled

            assert future.done()
            assert not future.cancelled()
            assert client._abandoned_request_ids == set()
            assert client._pending == {}
            assert client._incoming_error is None
    finally:
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        await client.close()
        await peer_transport.close()


@pytest.mark.asyncio
async def test_response_boundary_is_ordered_without_blocking_nested_requests() -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    observed: list[str] = []

    async def consume() -> None:
        incoming = client.incoming()
        before = await anext(incoming)
        observed.append(before.method)
        assert await client.request("test/resync") == {"resynced": True}
        observed.append("resynced")
        after = await anext(incoming)
        observed.append(after.method)
        await incoming.aclose()

    consumer = asyncio.create_task(consume())
    request = asyncio.create_task(
        client.request(
            "test/resume", response_boundary=lambda _result: observed.append("adopted")
        )
    )
    resume_request = await anext(peer_transport.messages())
    await peer_transport.send({"jsonrpc": "2.0", "method": "test/before", "params": {}})
    await peer_transport.send({
        "jsonrpc": "2.0",
        "id": resume_request["id"],
        "result": {"resumed": True},
    })

    nested_request = await asyncio.wait_for(anext(peer_transport.messages()), timeout=1)
    assert nested_request["method"] == "test/resync"
    await peer_transport.send({
        "jsonrpc": "2.0",
        "id": nested_request["id"],
        "result": {"resynced": True},
    })
    await peer_transport.send({"jsonrpc": "2.0", "method": "test/after", "params": {}})

    assert await asyncio.wait_for(request, timeout=1) == {"resumed": True}
    await asyncio.wait_for(consumer, timeout=1)
    assert observed == ["test/before", "resynced", "adopted", "test/after"]
    await client.close()
    await peer_transport.close()


@pytest.mark.asyncio
async def test_client_burst_does_not_block_in_flight_resync() -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    observed: list[int | str] = []
    resynced = asyncio.Event()

    async def consume() -> None:
        incoming = client.incoming()
        try:
            trigger = await anext(incoming)
            assert trigger.method == "test/gap"
            assert await client.request("session/read") == {"snapshot": True}
            observed.append("resynced")
            resynced.set()
            for _ in range(400):
                message = await anext(incoming)
                observed.append(cast(int, message.params["index"]))
        finally:
            await incoming.aclose()

    await client.start()
    consumer = asyncio.create_task(consume())
    try:
        await peer_transport.send({
            "jsonrpc": "2.0",
            "method": "test/gap",
            "params": {},
        })
        request = await anext(peer_transport.messages())
        assert request["method"] == "session/read"
        request_future = client._pending[request["id"]]
        for index in range(400):
            await peer_transport.send({
                "jsonrpc": "2.0",
                "method": "test/notification",
                "params": {"index": index},
            })
        await peer_transport.send({
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"snapshot": True},
        })

        await resynced.wait()
        assert request_future.done()
        assert request_future.result().result == {"snapshot": True}
        await consumer
        assert observed == ["resynced", *range(400)]
    finally:
        await client.close()
        await peer_transport.close()
        await asyncio.gather(consumer, return_exceptions=True)


@pytest.mark.asyncio
async def test_client_slow_consumer_allows_sustained_rpc_and_ordered_drain() -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    paused = asyncio.Event()
    resume = asyncio.Event()
    observed: list[int] = []

    async def consume() -> None:
        async for message in client.incoming():
            observed.append(cast(int, message.params["index"]))
            if len(observed) == 1:
                paused.set()
                await resume.wait()

    await client.start()
    consumer = asyncio.create_task(consume())
    try:
        await peer_transport.send({
            "jsonrpc": "2.0",
            "method": "test/notification",
            "params": {"index": 0},
        })
        await paused.wait()
        for batch in range(3):
            rpc = asyncio.create_task(client.request("test/ping"))
            request = await anext(peer_transport.messages())
            for index in range(1 + batch * 200, 1 + (batch + 1) * 200):
                await peer_transport.send({
                    "jsonrpc": "2.0",
                    "method": "test/notification",
                    "params": {"index": index},
                })
            await peer_transport.send({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"batch": batch},
            })
            assert await rpc == {"batch": batch}
            assert observed == [0]
            assert not resume.is_set()
            assert client._incoming.qsize() == (batch + 1) * 200

        # Production has stopped. Closing the transport must not discard the backlog.
        await client.close()
        resume.set()
        await consumer
        assert observed == list(range(601))
        assert client._incoming.empty()
    finally:
        resume.set()
        await client.close()
        await peer_transport.close()
        await asyncio.gather(consumer, return_exceptions=True)


@pytest.mark.asyncio
async def test_client_backlog_preserves_session_snapshot_response_boundary() -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    observed: list[tuple[str, int | str]] = []
    session_id = "old"

    def adopt(result: dict[str, Any]) -> None:
        nonlocal session_id
        observed.append(("adopted", result["sessionId"]))
        session_id = result["sessionId"]

    attachment = asyncio.create_task(
        client.request("session/resume", response_boundary=adopt)
    )
    try:
        request = await anext(peer_transport.messages())
        for index in range(400):
            await peer_transport.send({
                "jsonrpc": "2.0",
                "method": "test/old",
                "params": {"index": index},
            })
        await peer_transport.send({
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"sessionId": "new"},
        })
        await peer_transport.send({
            "jsonrpc": "2.0",
            "method": "test/new",
            "params": {"index": 400},
        })
        # A plain RPC fences reader progress without draining the ordered stream.
        fence = asyncio.create_task(client.request("test/fence"))
        fence_request = await anext(peer_transport.messages())
        await peer_transport.send({
            "jsonrpc": "2.0",
            "id": fence_request["id"],
            "result": {},
        })
        assert await fence == {}
        assert not attachment.done()
        assert observed == []
        assert session_id == "old"

        incoming = client.incoming()
        try:
            for _ in range(401):
                message = await anext(incoming)
                assert message.method == f"test/{session_id}"
                observed.append((session_id, cast(int, message.params["index"])))
        finally:
            await incoming.aclose()
        assert await attachment == {"sessionId": "new"}
        assert observed == [
            *(("old", index) for index in range(400)),
            ("adopted", "new"),
            ("new", 400),
        ]
    finally:
        await client.close()
        await peer_transport.close()
        await asyncio.gather(attachment, return_exceptions=True)


@pytest.mark.asyncio
async def test_client_close_with_backlog_resolves_all_pending_requests() -> None:
    initial_tasks = asyncio.all_tasks()
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    adopted = Mock()
    attachment = asyncio.create_task(
        client.request("session/resume", response_boundary=adopted)
    )
    pending = asyncio.create_task(client.request("test/pending"))
    try:
        requests = [await anext(peer_transport.messages()) for _ in range(2)]
        request_futures = list(client._pending.values())
        attachment_request = next(
            request for request in requests if request["method"] == "session/resume"
        )
        for index in range(400):
            await peer_transport.send({
                "jsonrpc": "2.0",
                "method": "test/notification",
                "params": {"index": index},
            })
        await peer_transport.send({
            "jsonrpc": "2.0",
            "id": attachment_request["id"],
            "result": {},
        })
        fence = asyncio.create_task(client.request("test/fence"))
        fence_request = await anext(peer_transport.messages())
        await peer_transport.send({
            "jsonrpc": "2.0",
            "id": fence_request["id"],
            "result": {},
        })
        assert await fence == {}
        assert client._incoming.qsize() == 401
        assert all(not future.done() for future in request_futures)

        await client.close()
        outcomes = await asyncio.gather(attachment, pending, return_exceptions=True)
        assert all(
            isinstance(outcome, AppServerConnectionClosed) for outcome in outcomes
        )
        assert all(future.done() for future in request_futures)
        assert client._pending == {}
        assert client._response_boundaries == {}
        assert client._reader_task is not None
        assert client._reader_task.done()
        adopted.assert_not_called()
        assert asyncio.all_tasks() - initial_tasks == set()
    finally:
        await client.close()
        await peer_transport.close()
        await asyncio.gather(attachment, pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_server_rejects_unknown_response_id() -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    serving = asyncio.create_task(server.serve())

    await client_transport.send({"jsonrpc": "2.0", "id": 99, "result": {}})

    with pytest.raises(
        JsonRpcProtocolError,
        match="Response does not match a pending server request: 99",
    ):
        await serving

    await client_transport.close()
    await agent_loop.aclose()


@pytest.mark.asyncio
async def test_late_callback_delivery_error_is_ignored_after_semantic_answer() -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    server._callback_requests[7] = CallbackDelivery(
        session_id=agent_loop.session_id, callback_id="callback-1"
    )

    await server._after_response(
        ServerRequest(
            id="callback-result",
            method="callback/result",
            params={"result": {"callbackId": "callback-1"}},
        ),
        DispatchResult(response=CallbackCallResponse(callback_id="callback-1")),
    )
    assert server._callback_requests[7].answered

    await server._handle_response(
        JsonRpcErrorResponse(
            id=7,
            error=ProtocolError(
                code=ProtocolErrorCode.INTERNAL_ERROR, message="delivery failed"
            ),
        )
    )

    assert server._callback_requests == {}
    await server.close()
    await client_transport.close()
    await agent_loop.aclose()


@pytest.mark.asyncio
async def test_failed_compact_releases_deferred_events_after_its_error_response() -> (
    None
):
    """*Prepare*: A compact backend that has a failed checkpoint buffered behind its error response.
    *Do*: Dispatch the compact request.
    *Assert*: The backend releases the checkpoint only once the error is written.
    """

    class FailingCompactBackend:
        session_id = "session-1"

        def guard_request(self) -> None:
            return None

        async def compact(self, _params: object) -> object:
            raise SessionBackendError(
                ProtocolErrorCode.COMPACTION_FAILED,
                "Context compaction failed",
                after_response=release_events,
            )

    # Prepare
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    release_events = Mock()
    server._root = cast(Any, FailingCompactBackend())
    server._initialization = InitializationState.INITIALIZED

    # Do
    await server._handle_request_once(
        ServerRequest(
            id="compact", method="session/compact", params={"sessionId": "session-1"}
        )
    )

    # Assert
    response = await anext(client_transport.messages())
    assert response["error"]["code"] == ProtocolErrorCode.COMPACTION_FAILED
    release_events.assert_called_once_with()
    server._root = None
    await server.close()
    await client_transport.close()
    await agent_loop.aclose()


@pytest.mark.asyncio
async def test_response_write_failure_abandons_deferred_backend_work() -> None:
    """*Prepare*: A successful backend result whose response write will fail.
    *Do*: Dispatch that result through the server.
    *Assert*: The abandonment callback releases the backend reservation.
    """
    # Prepare
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    abandon = Mock()

    async def dispatch(_request: ServerRequest) -> DispatchResult:
        return DispatchResult(
            response=CallbackCallResponse(callback_id="callback-1"),
            on_response_abandoned=abandon,
        )

    server._dispatch_or_error = dispatch  # type: ignore[method-assign]
    server._send = AsyncMock(side_effect=ConnectionError("connection closed"))

    # Do / Assert
    with pytest.raises(ConnectionError, match="connection closed"):
        await server._handle_request_once(
            ServerRequest(id="result", method="test/result", params={})
        )
    abandon.assert_called_once_with()
    await server.close()
    await client_transport.close()
    await agent_loop.aclose()


@pytest.mark.asyncio
async def test_failed_compact_response_write_failure_abandons_deferred_events() -> None:
    """*Prepare*: A failed compact whose error response cannot be written.
    *Do*: Dispatch the compact request.
    *Assert*: The backend discards the undeliverable checkpoint and releases its gate.
    """

    class FailingCompactBackend:
        session_id = "session-1"

        def guard_request(self) -> None:
            return None

        async def compact(self, _params: object) -> object:
            raise SessionBackendError(
                ProtocolErrorCode.COMPACTION_FAILED,
                "Context compaction failed",
                on_response_abandoned=abandon_events,
            )

    # Prepare
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    abandon_events = Mock()
    server._root = cast(Any, FailingCompactBackend())
    server._initialization = InitializationState.INITIALIZED
    server._send = AsyncMock(side_effect=ConnectionError("connection closed"))

    # Do / Assert
    with pytest.raises(ConnectionError, match="connection closed"):
        await server._handle_request_once(
            ServerRequest(
                id="compact",
                method="session/compact",
                params={"sessionId": "session-1"},
            )
        )
    abandon_events.assert_called_once_with()
    server._root = None
    await server.close()
    await client_transport.close()
    await agent_loop.aclose()


@pytest.mark.asyncio
async def test_projection_notifications_cannot_be_disabled() -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    server._client_capabilities = ClientCapabilities(
        disabled_notifications=["session/updated"]
    )
    server._connection_attached = True

    await server._notify(
        "session/updated",
        SessionUpdatedParams(
            event_id=0,
            session_id=agent_loop.session_id,
            emitted_at=1,
            patch=[JsonPatchOperation(op="replace", path="/updatedAt", value=2)],
        ),
    )
    notification = await anext(client_transport.messages())

    assert notification["method"] == "session/updated"
    assert notification["params"]["eventId"] == 1
    await server.close()
    await client_transport.close()
    await agent_loop.aclose()


@pytest.mark.asyncio
async def test_unknown_method_before_session_start_is_method_not_found() -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    try:
        await client.initialize(ClientInfo(name="test", version="1"))
        await client.notify("initialized")

        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request("unknown/method")

        assert exc_info.value.error.code is ProtocolErrorCode.METHOD_NOT_FOUND
        assert server._root is None
    finally:
        await client.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_removed_feedback_method_is_method_not_found() -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    try:
        await client.initialize(ClientInfo(name="test", version="1"))
        await client.notify("initialized")

        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request("feedback/shouldShow")

        assert exc_info.value.error.code is ProtocolErrorCode.METHOD_NOT_FOUND
    finally:
        await client.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_title_update_alias_before_session_start_is_typed_unsupported() -> None:
    """C2f: the title alias is refused from the catalogue, not a session route.

    A client probing the method before opening a session gets the same typed
    answer an attached one gets -- the refusal names session/rename and no
    session is created for it.
    """
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    try:
        await client.initialize(ClientInfo(name="test", version="1"))
        await client.notify("initialized")

        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request("session/title/update", {"sessionId": "any-session"})

        assert exc_info.value.error.code is ProtocolErrorCode.NOT_IMPLEMENTED
        assert "session/rename" in exc_info.value.error.message
        assert server._root is None
    finally:
        await client.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_shutdown_closes_root_and_transport_after_child_cleanup_failure() -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    await client.initialize(ClientInfo(name="test", version="1"))
    await client.notify("initialized")
    await client.request("session/start", {"agentConfig": {"cwd": str(agent_loop.cwd)}})
    backend = legacy_backend(server)
    handler = backend.handler
    handler.close = AsyncMock()
    backend.children.close = AsyncMock(side_effect=RuntimeError("child close failed"))
    agent_loop.aclose = AsyncMock()

    with pytest.raises(RuntimeError, match="child close failed"):
        await server.close()

    handler.close.assert_awaited_once()
    agent_loop.aclose.assert_awaited_once()
    with pytest.raises(RuntimeError, match="closed"):
        await server_transport.send({"jsonrpc": "2.0", "method": "test"})
    await client.close()


@pytest.mark.asyncio
async def test_session_close_records_pointer_before_responding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
        )
    )
    await agent_loop.persist_empty_session()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(
        client_transport,
        run_peer=lambda: server.serve_connection(
            server_transport, close_on_disconnect=False
        ),
    )
    record = Mock()
    monkeypatch.setattr(
        "chartreux.app_server._session_runtime_impl.last_session_pointer.record", record
    )

    await client.initialize(ClientInfo(name="test", version="1"))
    await client.notify("initialized")
    await client.request("session/start", {"agentConfig": {"cwd": str(agent_loop.cwd)}})
    await client.request("session/stop", {"sessionId": agent_loop.session_id})

    record.assert_called_once_with(
        agent_loop.config.session_logging, agent_loop.session_id
    )
    assert server._root is None
    await client.close()


@pytest.mark.asyncio
async def test_session_close_does_not_record_unpersisted_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(
        client_transport,
        run_peer=lambda: server.serve_connection(
            server_transport, close_on_disconnect=False
        ),
    )
    record = Mock()
    monkeypatch.setattr(
        "chartreux.app_server._session_runtime_impl.last_session_pointer.record", record
    )

    await client.initialize(ClientInfo(name="test", version="1"))
    await client.notify("initialized")
    await client.request("session/start", {"agentConfig": {"cwd": str(agent_loop.cwd)}})
    await client.request("session/stop", {"sessionId": agent_loop.session_id})

    record.assert_not_called()
    await client.close()


@pytest.mark.asyncio
async def test_concurrent_stops_coalesce_into_one_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2c: a second stop during the first stop's window coalesces into the
    first stop's pending receipt; both waiters get the ack, delivered before
    the connection closes, and no second persistence runs.
    """
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(
        client_transport,
        run_peer=lambda: server.serve_connection(
            server_transport, close_on_disconnect=False
        ),
    )
    gate = asyncio.Event()
    original_run_stop = server._run_stop

    async def gated_run_stop(raw_params: dict[str, Any]) -> Any:
        await gate.wait()
        return await original_run_stop(raw_params)

    monkeypatch.setattr(server, "_run_stop", gated_run_stop)
    close_runtime_calls = 0
    original_close_runtime = server._close_attached_runtime

    async def counting_close_runtime() -> None:
        nonlocal close_runtime_calls
        close_runtime_calls += 1
        await original_close_runtime()

    monkeypatch.setattr(server, "_close_attached_runtime", counting_close_runtime)

    await client.initialize(ClientInfo(name="test", version="1"))
    await client.notify("initialized")
    await client.request("session/start", {"agentConfig": {"cwd": str(agent_loop.cwd)}})
    stop_params = SessionStopParams(session_id=agent_loop.session_id).model_dump(
        mode="json", by_alias=True
    )
    first = asyncio.create_task(client.request("session/stop", stop_params))
    for _ in range(200):
        if server._stop_receipt is not None:
            break
        await asyncio.sleep(0.01)
    assert server._stop_receipt is not None
    second = asyncio.create_task(client.request("session/stop", stop_params))
    await asyncio.sleep(0.05)
    gate.set()
    results = await asyncio.gather(first, second)
    for result in results:
        assert SessionStopResponse.model_validate(result).closed is True
    # One persistence ran; both receipts were delivered before the close.
    assert close_runtime_calls == 1
    assert server._root is None
    assert server._shutdown_complete is True
    await client.close()
    await agent_loop.aclose()


@pytest.mark.asyncio
async def test_session_close_surfaces_an_unknown_outcome_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2c: the client sends session/stop with the extended long-op timeout,
    and a stop that still times out is surfaced as the actionable
    unknown-outcome reconcile state, never a suppressed exception.
    """
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    gate = asyncio.Event()
    original_run_stop = server._run_stop

    async def gated_run_stop(raw_params: dict[str, Any]) -> Any:
        await gate.wait()
        return await original_run_stop(raw_params)

    monkeypatch.setattr(server, "_run_stop", gated_run_stop)
    monkeypatch.setattr(
        "chartreux.app_server.session.DEFAULT_STOP_REQUEST_TIMEOUT", 0.05
    )
    close_task = asyncio.create_task(session.close())
    try:
        # The stop times out at 0.05s; close() then finishes its cleanup
        # (which waits on the gated server) before raising the outcome.
        await asyncio.sleep(0.3)
        gate.set()
        with pytest.raises(SessionStopUnknownOutcome, match="session/resume"):
            await close_task
    finally:
        gate.set()
        await server.close()
        await client.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_session_close_surfaces_an_unknown_outcome_stop_on_a_pre_ack_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W4: a stop whose peer goes away before the ack is an unknown outcome.

    ``AppServerConnectionClosed`` is a ``RuntimeError``, so the stop arm used
    to swallow the pre-ack disconnect and ``close()`` returned normally even
    though the server may have applied the stop.
    """
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    stop_reached = asyncio.Event()
    release = asyncio.Event()
    original_run_stop = server._run_stop

    async def gated_run_stop(raw_params: dict[str, Any]) -> Any:
        stop_reached.set()
        await release.wait()
        return await original_run_stop(raw_params)

    monkeypatch.setattr(server, "_run_stop", gated_run_stop)
    close_task = asyncio.create_task(session.close())
    try:
        await asyncio.wait_for(stop_reached.wait(), timeout=5)
        # The stop is in flight and unacked: dropping the peer closes the
        # client's channel, which fails the pending stop future with
        # AppServerConnectionClosed.
        await server_transport.close()
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(SessionStopUnknownOutcome, match="session/resume"):
            await asyncio.wait_for(close_task, timeout=5)
    finally:
        release.set()
        await server.close()
        await client.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_session_close_treats_a_closed_client_as_teardown() -> None:
    """W4: a client closed before the stop was attempted stays teardown.

    The stop never leaves the process, so nothing was mutated and the close
    continues; only a stop that was sent and never answered is unknown.
    """
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    # An intentional shutdown stops the pump, so the closed client stays the
    # connection's current one (the reconnect path would clear it instead).
    session.begin_close()
    await client.close()
    try:
        await session.close()
    finally:
        await server.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_session_close_treats_a_channel_down_before_send_as_teardown() -> None:
    """W8: a channel that went down before the stop was attempted is not-sent
    teardown, not an unknown outcome.

    The client raises before the stop frame is queued, so the frame never
    leaves the process and nothing was mutated -- the p_parity
    mutation-timeout teardown shape, which W4 misreported as unknown.
    """
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    session.begin_close()
    await server_transport.close()
    await asyncio.wait_for(client._incoming_closed.wait(), timeout=5)
    try:
        await session.close()
    finally:
        await server.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_request_on_a_down_channel_raises_not_sent() -> None:
    """W8: a request attempted on an already-down channel is classified as
    not-sent (``AppServerRequestNotSent``), which stays an
    ``AppServerConnectionClosed`` for existing connection-loss handlers.
    """
    from chartreux.app_server.client import AppServerRequestNotSent

    client_transport, server_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    await client.start()
    await server_transport.close()
    await asyncio.wait_for(client._incoming_closed.wait(), timeout=5)
    with pytest.raises(AppServerRequestNotSent):
        await client.request("session/read", {})
    with pytest.raises(AppServerConnectionClosed):
        await client.request("session/read", {})
    await client.close()


@pytest.mark.asyncio
async def test_session_close_surfaces_an_unknown_outcome_on_a_send_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W8: a stop whose send fails AFTER its frame was queued (the writer died
    mid-send) may have been delivered -- non-delivery is not established, so
    the outcome stays unknown instead of being treated as not-sent.
    """
    from chartreux.app_server.transport import JsonRpcSendUncertain

    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    original_send = client_transport.send

    async def racing_send(message: dict[str, Any]) -> None:
        if message.get("method") == "session/stop":
            raise JsonRpcSendUncertain("writer cancelled after the frame was queued")
        await original_send(message)

    monkeypatch.setattr(client_transport, "send", racing_send)
    try:
        with pytest.raises(SessionStopUnknownOutcome, match="session/resume"):
            await session.close()
    finally:
        await server.close()
        await client.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_transport_writer_failure_after_queueing_is_uncertain() -> None:
    """W8: a writer cancelled or stopped before the frame is queued keeps the
    plain RuntimeError (non-delivery established); once the frame was queued
    the same failure is ``JsonRpcSendUncertain`` (delivery unknown). A writer
    that failed with its own error propagates that error in both modes.
    """
    from chartreux.app_server.transport import (
        JsonRpcSendUncertain,
        StdioJsonRpcTransport,
    )

    async def boom() -> None:
        raise RuntimeError("writer died")

    failed = asyncio.create_task(boom())
    with pytest.raises(RuntimeError, match="writer died"):
        await failed
    # A writer that failed with its own error propagates that error in both
    # modes; only cancellation / clean exit are reclassified post-queue.
    with pytest.raises(RuntimeError, match="writer died"):
        StdioJsonRpcTransport._check_writer(failed)
    with pytest.raises(RuntimeError, match="writer died"):
        StdioJsonRpcTransport._check_writer(failed, uncertain=True)

    cancelled = asyncio.create_task(asyncio.sleep(0))
    cancelled.cancel()
    with suppress(asyncio.CancelledError):
        await cancelled
    with pytest.raises(RuntimeError, match="was cancelled$"):
        StdioJsonRpcTransport._check_writer(cancelled)
    with pytest.raises(
        JsonRpcSendUncertain, match="cancelled after the frame was queued"
    ):
        StdioJsonRpcTransport._check_writer(cancelled, uncertain=True)

    exited = asyncio.create_task(asyncio.sleep(0))
    await exited
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        StdioJsonRpcTransport._check_writer(exited)
    with pytest.raises(
        JsonRpcSendUncertain, match="stopped after the frame was queued"
    ):
        StdioJsonRpcTransport._check_writer(exited, uncertain=True)


@pytest.mark.asyncio
async def test_client_request_times_out_against_an_unresponsive_peer() -> None:
    client_transport, peer_transport = memory_transport_pair()
    client = AppServerClient(client_transport, request_timeout=0.01)
    request = asyncio.create_task(client.request("test/unresponsive"))
    sent = await anext(peer_transport.messages())

    with pytest.raises(TimeoutError, match="test/unresponsive"):
        await request

    await peer_transport.send({"jsonrpc": "2.0", "id": sent["id"], "result": {}})
    await asyncio.sleep(0)
    assert client._abandoned_request_ids == set()
    await client.close()
    await peer_transport.close()


@pytest.mark.asyncio
async def test_agent_transcript_uses_attached_backend_extension_route() -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    try:
        await client.initialize(ClientInfo(name="test", version="1"))
        await client.notify("initialized")
        await client.request(
            "session/start", {"agentConfig": {"cwd": str(agent_loop.cwd)}}
        )
        backend = legacy_backend(server)
        backend.handler.dispatch = AsyncMock(
            return_value=DispatchResult(
                AgentTranscriptGetResponse(
                    state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
                )
            )
        )

        response = await client.request("agent/transcript/get", {"agentId": "agent-1"})

        assert response["state"] == "no_saved_transcript"
        backend.handler.dispatch.assert_awaited_once_with(
            "agent/transcript/get", {"agentId": "agent-1"}
        )
    finally:
        await client.close()
        await server.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_agent_transcript_is_rejected_without_an_attached_session() -> None:
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    try:
        await client.initialize(ClientInfo(name="test", version="1"))
        await client.notify("initialized")

        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request("agent/transcript/get", {"agentId": "agent-1"})

        assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT
        assert server._root is None
    finally:
        await client.close()
        await server.close()
        await agent_loop.aclose()


# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_superseded_session_fence_translates_only_not_found_for_superseded_ids() -> (
    None
):
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    try:
        server._session_rotations["old"] = "new"
        not_found = SessionBackendError(
            ProtocolErrorCode.NOT_FOUND, "Session not found: old"
        )

        fenced = await server._superseded_session_fence(not_found, {"sessionId": "old"})
        assert fenced is not None
        assert fenced.code is ProtocolErrorCode.CONFLICT
        data = cast(dict[str, Any], fenced.data)
        assert data["code"] == "session/fenced"
        assert data["sessionId"] == "new"

        # An id with no rotation record keeps the original error.
        assert (
            await server._superseded_session_fence(not_found, {"sessionId": "other"})
            is None
        )
        # A non-NOT_FOUND error is never rewritten.
        conflict = RequestFailure(ProtocolErrorCode.CONFLICT, "conflict")
        assert (
            await server._superseded_session_fence(conflict, {"sessionId": "old"})
            is None
        )
        # Params without a session id are untouched.
        assert await server._superseded_session_fence(not_found, {}) is None
    finally:
        await server.close()
        await client_transport.close()
        await agent_loop.aclose()


# ---------------------------------------------------------------------------
# Package C1g: recovery/root/{schedule,admit,cancel} as typed wire carriers.
# ---------------------------------------------------------------------------


def _c1g_timeout_fault() -> BackendError:
    """A streaming timeout fault with its httpx cause (classifies TIMEOUT)."""
    fault = BackendError(
        provider="test-first",
        endpoint="https://first.invalid/chat",
        status=None,
        reason="read timed out",
        headers=None,
        body_text=None,
        parsed_error="Network error",
        model="first",
        payload_summary=PayloadSummary(
            model="first",
            message_count=1,
            approx_chars=2,
            temperature=0.0,
            has_tools=False,
            tool_choice=None,
        ),
    )
    fault.__cause__ = httpx.ReadTimeout("read timed out")
    return fault


def _c1g_recovery_loop(project: Path) -> tuple[AgentLoop, FakeClock]:
    """A test loop whose first turn terminally fails, on a manual clock."""
    loop = build_test_agent_loop(
        cwd=project,
        # No gateway retry budget: the scripted timeout is terminal on the
        # first attempt, so the turn fails without real backoff.
        config=build_test_vibe_config(api_retry_max_elapsed_time=0.0),
        backend=FakeBackend(exception_to_raise=_c1g_timeout_fault()),
        enable_streaming=True,
    )
    clock = FakeClock()
    loop.config_orchestrator._availability_registry = AvailabilityRegistry(clock=clock)
    return loop, clock


async def _c1g_failed_turn_id(session: Any) -> str:
    with pytest.raises(AppServerTurnError) as turn_error:
        async for _event in session.act("hi"):
            pass
    turn_id = turn_error.value.turn_id
    assert turn_id is not None
    return turn_id


async def _c1g_exercise_carriers(client: AppServerClient, session_id: str) -> None:
    """The carrier contract both transports must answer identically."""
    # An unknown terminal turn is carried to the gate and declined typed --
    # never a NOT_IMPLEMENTED fall-through.
    unknown = RootRecoveryScheduleResponse.model_validate(
        await client.request(
            "recovery/root/schedule",
            RootRecoveryScheduleParams(
                session_id=session_id, terminal_turn_id="no-such-turn", solicited=True
            ),
        )
    )
    assert unknown.decision == "declined"
    assert unknown.reason == "no_typed_projection"

    # A wrong session id is a plain typed not-found.
    with pytest.raises(AppServerResponseError) as exc_info:
        await client.request(
            "recovery/root/schedule",
            RootRecoveryScheduleParams(
                session_id="other-session", terminal_turn_id="no-such-turn"
            ),
        )
    assert exc_info.value.error.code is ProtocolErrorCode.NOT_FOUND


@pytest.mark.asyncio
async def test_recovery_root_carriers_reach_the_gate_on_the_stdio_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """stdio: the three public methods are carried to the coordinator-local gate.

    The process-session backend owns the gate; the loop-owned evidence bundle
    (live config, transcript-integrity verdict, availability snapshot) crosses
    the agent wire once per decision, and the typed terminal projection the
    agent attached to its turn/ended frame feeds the schedule -- so a real
    failed turn schedules, and concurrent admits single-flight.
    """
    project = tmp_path / "c1g-stdio-project"
    project.mkdir()
    loop, clock = _c1g_recovery_loop(project)
    install_memory_root_agent(monkeypatch, loop)
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(server_transport, transport_kind="stdio")
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(
        client, session_options=SessionOptions(cwd=str(project))
    )
    try:
        process = cast(Any, harness._server._root)._process
        crossings: list[str] = []
        owner = process.recovery_evidence

        async def recorded(session_id: str) -> Any:
            crossings.append(session_id)
            return await owner(session_id)

        monkeypatch.setattr(process, "recovery_evidence", recorded)
        session_id = session.session_id

        await _c1g_exercise_carriers(client, session_id)
        # One evidence bundle per schedule decision, fetched owner-side.
        assert crossings == [session_id]

        turn_id = await _c1g_failed_turn_id(session)
        schedule = RootRecoveryScheduleResponse.model_validate(
            await client.request(
                "recovery/root/schedule",
                RootRecoveryScheduleParams(
                    session_id=session_id,
                    terminal_turn_id=turn_id,
                    solicited=True,
                    input_generation=0,
                ),
            )
        )
        # The typed terminal projection crossed the agent wire.
        assert schedule.decision == "scheduled", schedule.model_dump(mode="json")
        assert schedule.episode_id is not None
        assert schedule.idempotency_key is not None
        assert crossings == [session_id, session_id]

        clock.advance(40.0)
        admit_params = RootRecoveryAdmitParams(
            session_id=session_id,
            episode_id=schedule.episode_id,
            idempotency_key=schedule.idempotency_key,
            terminal_turn_id=turn_id,
            input_generation=0,
        )
        raw_admits = await asyncio.gather(*[
            client.request("recovery/root/admit", admit_params) for _ in range(2)
        ])
        decisions = sorted(
            RootRecoveryAdmitResponse.model_validate(raw).decision for raw in raw_admits
        )
        assert decisions == ["admitted", "already_admitted"]

        cancel = RootRecoveryCancelResponse.model_validate(
            await client.request(
                "recovery/root/cancel",
                RootRecoveryCancelParams(
                    session_id=session_id, episode_id="no-such-episode", reason="test"
                ),
            )
        )
        assert cancel.cancelled is False
    finally:
        await session.close()
        await client.close()
        await harness.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_recovery_root_carriers_answer_identically_in_process(
    tmp_path: Path,
) -> None:
    """in_process: the same carrier contract the stdio leg must match."""
    project = tmp_path / "c1g-in-process-project"
    project.mkdir()
    loop, clock = _c1g_recovery_loop(project)
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(
        client, session_options=SessionOptions(cwd=str(project))
    )
    try:
        session_id = session.session_id
        await _c1g_exercise_carriers(client, session_id)

        turn_id = await _c1g_failed_turn_id(session)
        schedule = RootRecoveryScheduleResponse.model_validate(
            await client.request(
                "recovery/root/schedule",
                RootRecoveryScheduleParams(
                    session_id=session_id,
                    terminal_turn_id=turn_id,
                    solicited=True,
                    input_generation=0,
                ),
            )
        )
        assert schedule.decision == "scheduled", schedule.model_dump(mode="json")
        assert schedule.episode_id is not None
        assert schedule.idempotency_key is not None

        clock.advance(40.0)
        admit_params = RootRecoveryAdmitParams(
            session_id=session_id,
            episode_id=schedule.episode_id,
            idempotency_key=schedule.idempotency_key,
            terminal_turn_id=turn_id,
            input_generation=0,
        )
        raw_admits = await asyncio.gather(*[
            client.request("recovery/root/admit", admit_params) for _ in range(2)
        ])
        decisions = sorted(
            RootRecoveryAdmitResponse.model_validate(raw).decision for raw in raw_admits
        )
        assert decisions == ["admitted", "already_admitted"]

        cancel = RootRecoveryCancelResponse.model_validate(
            await client.request(
                "recovery/root/cancel",
                RootRecoveryCancelParams(
                    session_id=session_id, episode_id="no-such-episode", reason="test"
                ),
            )
        )
        assert cancel.cancelled is False
    finally:
        await session.close()
        await client.close()
        await server.close()
        await loop.aclose()


# ---------------------------------------------------------------------------
# Package C1i: the session-scoped diagnostics reads answer on both transports.
# ---------------------------------------------------------------------------

_C1I_LOG_LINES = (
    "2026-11-13T10:00:00.000000+00:00 11 12 INFO first entry\n"
    "2026-11-13T10:00:01.000000+00:00 11 13 WARNING second entry\n"
)


def _c1i_seed_log_file() -> None:
    """Write two parseable log lines to this test's own log file."""
    log_file = LOG_FILE.path
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(_C1I_LOG_LINES, encoding="utf-8")


async def _c1i_assert_diagnostics_reads(
    client: AppServerClient, session_id: str
) -> None:
    """Both reads return the shipped owner projection with unchanged bounds."""
    listed = await client.request("diagnostics/list", {"sessionId": session_id})
    assert isinstance(listed["issues"], list)
    assert isinstance(listed["hooksCount"], int)

    page = await client.request(
        "diagnostics/logs/read", {"sessionId": session_id, "limit": 1, "offset": 0}
    )
    logs = page["logs"]
    assert [entry["message"] for entry in logs["entries"]] == ["second entry"]
    assert logs["entries"][0]["pid"] == 13
    assert logs["entries"][0]["ppid"] == 11
    assert logs["entries"][0]["rawLine"] == _C1I_LOG_LINES.splitlines()[1]
    assert logs["hasMore"] is True


@pytest.mark.asyncio
async def test_diagnostics_reads_answer_in_process() -> None:
    """in_process: the reads keep the shipped handler's bounds and entries."""
    client_transport, server_transport = memory_transport_pair()
    agent_loop = build_test_agent_loop()
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    try:
        _c1i_seed_log_file()
        await _c1i_assert_diagnostics_reads(client, session.session_id)
    finally:
        await session.close()
        await client.close()
        await server.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_diagnostics_reads_cross_to_the_agent_that_owns_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stdio: both reads cross to the agent's own loop and log reader.

    The process-session backend owns neither the loop's issues projection nor
    a log reader, so a coordinator answer would have to invent both. Each
    public call is observed crossing to the owner, the page keeps the request's
    own bounds and the owner's exact entries (pid provenance included), and a
    wrong session id stays the typed not-found the in-process route answers.
    """
    loop = build_test_agent_loop()
    install_memory_root_agent(monkeypatch, loop)
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(server_transport, transport_kind="stdio")
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(client)
    try:
        process = cast(Any, harness._server._root)._process
        crossings: list[str] = []
        list_owner = process.read_diagnostics
        logs_owner = process.read_diagnostics_logs

        async def recorded_list(session_id: str, params: Any) -> Any:
            crossings.append("read_diagnostics")
            return await list_owner(session_id, params)

        async def recorded_logs(session_id: str, params: Any) -> Any:
            crossings.append("read_diagnostics_logs")
            return await logs_owner(session_id, params)

        monkeypatch.setattr(process, "read_diagnostics", recorded_list)
        monkeypatch.setattr(process, "read_diagnostics_logs", recorded_logs)
        _c1i_seed_log_file()

        await _c1i_assert_diagnostics_reads(client, session.session_id)
        assert crossings == ["read_diagnostics", "read_diagnostics_logs"]

        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "diagnostics/logs/read", {"sessionId": "other-session"}
            )
        assert exc_info.value.error.code is ProtocolErrorCode.NOT_FOUND
    finally:
        await session.close()
        await client.close()
        await harness.close()
        await loop.aclose()
