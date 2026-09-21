from __future__ import annotations

import pytest

from chartreux.acp.agent import ChartreuxAcpAgent as ChartreuxAcpAgentLoop
from chartreux.acp.auth import AcpAuthController
from chartreux.acp.exceptions import InvalidRequestError


class TestACPAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_rejects_unsupported_method(
        self, acp_agent_loop: ChartreuxAcpAgentLoop
    ) -> None:
        with pytest.raises(
            InvalidRequestError, match="Unsupported auth method: vibe-setup"
        ):
            await acp_agent_loop.authenticate("vibe-setup")

    @pytest.mark.parametrize("delegated", [False, True])
    def test_browser_methods_are_not_advertised(self, delegated: bool) -> None:
        assert AcpAuthController().browser_methods(delegated=delegated) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["browser-auth", "browser-auth-delegated"])
    async def test_browser_authentication_is_rejected(self, method: str) -> None:
        with pytest.raises(InvalidRequestError, match="Unsupported auth method"):
            await AcpAuthController().authenticate(method, {})
