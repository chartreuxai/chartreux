from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chartreux.app_server.host import AppServerHost
    from chartreux.app_server.session import AppServerSession

type AppServerStarter = Callable[[], Awaitable[AppServerSession]]
type AppServerSource = AppServerSession | AppServerStarter
type AppServerBootstrap = Callable[[], Awaitable[AppServerHost | AppServerSession]]
