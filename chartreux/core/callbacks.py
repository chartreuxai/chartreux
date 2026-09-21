from __future__ import annotations

from collections.abc import Awaitable, Callable

type SwitchAgentCallback = Callable[[str], Awaitable[None]]

type ClearContextCallback = Callable[[], Awaitable[None]]
