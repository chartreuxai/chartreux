from __future__ import annotations

from unittest.mock import MagicMock

from mcp.types import CreateMessageRequestParams, ErrorData
import pytest

from chartreux.core.tools.mcp_sampling import MCPSamplingHandler


@pytest.mark.asyncio
async def test_sampling_handler_rejects_requests() -> None:
    result = await MCPSamplingHandler()(
        MagicMock(), MagicMock(spec=CreateMessageRequestParams)
    )

    assert isinstance(result, ErrorData)
    assert result.code == -1
    assert result.message == "MCP sampling is not supported"
