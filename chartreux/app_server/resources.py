from __future__ import annotations

from chartreux.app_server._integration_resources import MCPResource, SkillsResource
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
from chartreux.app_server.protocol import Notification

__all__ = ["AppServerResources"]


class AppServerResources:
    def __init__(
        self, connection: AppServerResourceConnection, state: ClientSessionState
    ) -> None:
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
            RuntimeResource.notification_methods() | MCPResource.notification_methods()
        )

    async def refresh(self) -> None:
        await self.runtime.refresh()

    async def consume_notification(self, notification: Notification) -> bool:
        previous_config = self.config.current
        if await self.runtime.consume_notification(notification):
            self.config.publish_change(previous_config)
            return True
        if await self.mcp.consume_notification(notification):
            return True
        return False

    async def consume_event(self, event: AppServerEvent) -> bool:
        return await self.shell.consume_event(event)
