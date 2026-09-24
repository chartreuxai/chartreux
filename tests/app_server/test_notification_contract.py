from __future__ import annotations

import pytest

from chartreux.app_server.protocol import (
    SessionReadParams,
    SessionUpdatedParams,
    server_notification_registry,
)
from chartreux.app_server.server import AppServer

# These methods are handled by the matching client dispatch path: the event
# projection, parse_server_event, or AppServerResources.consume_notification.
_PROJECTION_HANDLED = {
    "agents/update",
    "history/entryAdded",
    "history/entryUpdated",
    "session/compacted",
    "session/contextCleared",
    "session/snapshot",
    "session/statsUpdated",
    "session/updated",
    "turn/completed",
    "turn/started",
    "turn_queue_updated",
}
_SERVER_EVENT_HANDLED = {
    "error",
    "mcp_catalog/authRequired",
    "turn/retrying",
    "warning",
}
_RESOURCE_HANDLED = {"mcp_catalog/authUrl", "runtime/updated"}


def test_every_registered_server_notification_has_client_handling() -> None:
    registry = server_notification_registry()
    handled = _PROJECTION_HANDLED | _SERVER_EVENT_HANDLED | _RESOURCE_HANDLED

    assert registry
    assert set(registry) == handled


@pytest.mark.asyncio
async def test_server_notify_rejects_undeclared_and_mismatched_methods() -> None:
    server = object.__new__(AppServer)
    params = SessionUpdatedParams(
        event_id=0, session_id="session-1", patch=[], emitted_at=1
    )

    with pytest.raises(ValueError, match="does not match"):
        await server.notify("history/entryUpdated", params)

    with pytest.raises(ValueError, match="does not match"):
        await server.notify(
            "unregistered/method", SessionReadParams(session_id="session-1")
        )
