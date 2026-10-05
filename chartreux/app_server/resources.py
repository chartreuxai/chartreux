from __future__ import annotations

from collections.abc import Callable
import logging

from chartreux.app_server._integration_resources import MCPResource, SkillsResource
from chartreux.app_server._model import validate_wire
from chartreux.app_server._runtime_resources import (
    ConfigResource,
    IdentityResource,
    RuntimeResource,
)
from chartreux.app_server._service_resources import NarrationResource
from chartreux.app_server._session_resources import (
    LoopsResource,
    ReviewResource,
    SessionResource,
    ShellResource,
    WorkspaceResource,
)
from chartreux.app_server.client_state import ClientSessionState
from chartreux.app_server.connection import AppServerResourceConnection
from chartreux.app_server.events import AppServerEvent
from chartreux.app_server.models import UsageWindow
from chartreux.app_server.protocol import (
    Notification,
    UsageReadParams,
    UsageReadResponse,
    UsageUpdatedParams,
)

__all__ = ["AppServerResources"]


class UsageResource:
    """Host-level reads and revisioned global summaries (never session stats)."""

    def __init__(self, connection: AppServerResourceConnection) -> None:
        self._connection = connection
        self.current: UsageUpdatedParams | None = None
        self.project_key: str | None = None
        self._generation = 0
        self._subscribers: list[Callable[[UsageUpdatedParams], None]] = []

    @property
    def revision(self) -> int | None:
        return self.current.revision if self.current is not None else None

    @classmethod
    def notification_methods(cls) -> frozenset[str]:
        return frozenset({UsageUpdatedParams.NOTIFICATION_METHOD})

    def subscribe(
        self, callback: Callable[[UsageUpdatedParams], None]
    ) -> Callable[[], None]:
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    def _publish(self, summary: UsageUpdatedParams) -> None:
        if self.current is not None and summary.revision <= self.current.revision:
            return
        self.current = summary
        for callback in list(self._subscribers):
            try:
                callback(summary)
            except Exception:
                logging.getLogger(__name__).warning("Usage subscriber failed")

    async def read(
        self, window: UsageWindow | None = None, project_key: str | None = None
    ) -> UsageReadResponse:
        client = await self._connection.connect_host()
        generation = self._generation
        response = validate_wire(
            UsageReadResponse,
            await client.request(
                "usage/read",
                UsageReadParams(window=window or "day", project_key=project_key),
            ),
        )
        if generation != self._generation:
            return response
        self.project_key = response.project_key
        # Project-filtered summaries must never replace global spend.
        if project_key is None:
            self._publish(
                UsageUpdatedParams(
                    as_of=response.as_of,
                    revision=response.revision,
                    summaries=response.summaries,
                    degraded=bool(response.warnings),
                )
            )
        return response

    async def refresh_after_reconnect(self) -> None:
        # Revisions belong to the connected host, which may have restarted.
        self._generation += 1
        self.current = None
        self.project_key = None
        await self.read()

    async def consume_notification(self, notification: Notification) -> bool:
        if notification.method not in self.notification_methods():
            return False
        self._publish(validate_wire(UsageUpdatedParams, notification.params))
        return True


class AppServerResources:
    def __init__(
        self, connection: AppServerResourceConnection, state: ClientSessionState
    ) -> None:
        self.usage = UsageResource(connection)
        self.identity = IdentityResource(connection, state)
        self.config = ConfigResource(connection, state)
        self.runtime = RuntimeResource(connection, state)
        self.mcp = MCPResource(connection, state)
        self.skills = SkillsResource(connection, state)
        self.shell = ShellResource(connection, state)
        self.sessions = SessionResource(connection, state)
        self.review = ReviewResource(connection, state)
        self.workspace = WorkspaceResource(connection, state)
        self.loops = LoopsResource(connection, state)
        self.narration = NarrationResource(connection, state)

    @classmethod
    def notification_methods(cls) -> frozenset[str]:
        return (
            RuntimeResource.notification_methods()
            | MCPResource.notification_methods()
            | UsageResource.notification_methods()
        )

    async def refresh(self) -> None:
        await self.runtime.refresh()

    async def consume_notification(self, notification: Notification) -> bool:
        if await self.usage.consume_notification(notification):
            return True
        previous_config = self.config.current
        if await self.runtime.consume_notification(notification):
            self.config.publish_change(previous_config)
            return True
        if await self.mcp.consume_notification(notification):
            return True
        return False

    async def consume_event(self, event: AppServerEvent) -> bool:
        return await self.shell.consume_event(event)
