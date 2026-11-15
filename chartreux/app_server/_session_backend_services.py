from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.models import PublicCallbackEntry
from chartreux.app_server.protocol import ClientCapabilities, ClientInfo


class SpawnClientIdentity(Protocol):
    """The client-identity subset a coordinator-side process spawn needs.

    A fresh process-backed child validates its own ``ClientInfo`` at open, so
    the payload a coordinator-built spawn carries must include the same
    validated identity the root route sends. The session backend services
    satisfy this structurally (they expose both reads), so the app-server
    services object can be handed over without an adapter.
    """

    def client_info(self) -> ClientInfo: ...

    def client_capabilities(self) -> ClientCapabilities: ...


class SessionBackendServices(Protocol):
    def client_info(self) -> ClientInfo: ...

    def client_capabilities(self) -> ClientCapabilities: ...

    def current_session_id(self) -> str: ...

    def event_watermark(self, session_id: str) -> int: ...

    def lifecycle_transition(self) -> AbstractAsyncContextManager[None]: ...

    def task_finished(self, task: asyncio.Task[None]) -> None: ...

    async def notify(self, method: str, params: ProtocolModel) -> None: ...

    async def publish_callback(self, callback: PublicCallbackEntry) -> None: ...

    async def record_child_notification(
        self, method: str, params: ProtocolModel
    ) -> None: ...

    async def request_client_result[ResultT: ProtocolModel](
        self, method: str, params: ProtocolModel, response_type: type[ResultT]
    ) -> ResultT: ...
