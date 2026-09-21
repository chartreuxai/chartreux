from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from acp.schema import TextContentBlock, UsageUpdate
import pytest

from chartreux.acp.agent import ChartreuxAcpAgent


@pytest.mark.asyncio
async def test_usage_update_serializes_runtime_statistics_in_acp_metadata(
    acp_agent_loop: ChartreuxAcpAgent,
) -> None:
    session_id = (
        await acp_agent_loop.new_session(cwd=str(Path.cwd()), mcp_servers=[])
    ).session_id
    client: Any = acp_agent_loop.client
    client._session_updates.clear()

    await acp_agent_loop.prompt(
        session_id=session_id, prompt=[TextContentBlock(type="text", text="Hello")]
    )

    def usage_updates() -> list[UsageUpdate]:
        return [
            notification.update
            for notification in client._session_updates
            if isinstance(notification.update, UsageUpdate)
        ]

    for _ in range(50):
        await asyncio.sleep(0)
        if usage_updates():
            break

    update = usage_updates()[-1]
    stats = acp_agent_loop.sessions[session_id].app_server.resources.runtime.stats
    expected_meta = {
        "steps": stats.steps,
        "promptTokens": stats.session_prompt_tokens,
        "completionTokens": stats.session_completion_tokens,
        "cachedTokens": stats.session_cached_tokens,
        "totalTokens": stats.session_total_llm_tokens,
        "tokensPerSecond": stats.tokens_per_second,
        "lastTurnDuration": stats.last_turn_duration,
        "lastTurnTotalTokens": stats.last_turn_total_tokens,
    }

    assert update.field_meta == expected_meta
    wire = update.model_dump(mode="json", by_alias=True)
    assert wire["_meta"] == expected_meta
    assert wire["used"] == stats.context_tokens
    assert wire["size"] > 0
