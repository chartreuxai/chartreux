from __future__ import annotations

import pytest

from chartreux.app_server.events import (
    projection_notification_methods,
    server_event_notification_methods,
)
from chartreux.app_server.protocol import (
    SessionReadParams,
    SessionUpdatedParams,
    server_notification_registry,
)
from chartreux.app_server.resources import AppServerResources
from chartreux.app_server.server import AppServer


def test_every_registered_server_notification_has_client_handling() -> None:
    registry = server_notification_registry()
    assert registry is server_notification_registry()
    handled = (
        projection_notification_methods()
        | server_event_notification_methods()
        | AppServerResources.notification_methods()
    )

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
