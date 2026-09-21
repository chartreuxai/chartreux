from __future__ import annotations

from chartreux.app_server.models import MCPState


class FakeMCPResource:
    """Stands in for the MCPResource on an ACP session's app-server resources."""

    def __init__(
        self,
        state: MCPState | None = None,
        *,
        auth_url: str | None = None,
        tool_count: int = 0,
        mutated_state: MCPState | None = None,
    ) -> None:
        self.state = state if state is not None else MCPState()
        self.auth_url = auth_url
        self.tool_count = tool_count
        self._mutated_state = mutated_state
        self.read_calls = 0

    async def read(self) -> MCPState:
        self.read_calls += 1
        return self.state

    async def toggle(
        self, name: str, *, disabled: bool, tool_name: str | None = None
    ) -> MCPState:
        return self._apply_mutation()

    def _apply_mutation(self) -> MCPState:
        if self._mutated_state is not None:
            self.state = self._mutated_state
        return self.state
