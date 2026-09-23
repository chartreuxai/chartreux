from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.client.session import ClientSession
    from mcp.shared.context import RequestContext
    from mcp.types import CreateMessageRequestParams, ErrorData


class MCPSamplingHandler:
    async def __call__(
        self,
        context: RequestContext[ClientSession, Any],
        params: CreateMessageRequestParams,
    ) -> ErrorData:
        """Fail closed: MCP sampling is not supported."""
        from mcp.types import ErrorData

        return ErrorData(code=-1, message="MCP sampling is not supported")
