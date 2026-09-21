"""Persist MCP server enable/disable settings."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._mcp_save import (
    MCPApply,
    MCPPreflight,
    accepted_mcp_servers,
    save_mcp_servers,
)
from chartreux.core.config._restrictions import ConfigCandidate
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.types import ConfigSaveResult


def updated_tool_list(tools: list[str], name: str, disabled: bool) -> list[str]:
    if disabled:
        return list(dict.fromkeys([*tools, name]))
    return [tool for tool in tools if tool != name]


async def persist_mcp_toggle(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    *,
    name: str,
    disabled: bool,
    tool_name: str | None = None,
    preflight: Callable[[ChartreuxConfigSchema], Awaitable[None]] | None = None,
    candidate_preflight: MCPPreflight | None = None,
    apply: MCPApply | None = None,
) -> ConfigSaveResult | None:
    """Save an accepted user entry; absent entries remain a no-op.

    ``preflight`` retains the legacy schema callback. New runtime owners can
    prepare the complete candidate with ``candidate_preflight`` and publish
    synchronously with ``apply``. Non-success saves raise a value-free error
    carrying their exact persistence/application result, never a rollback claim.
    """
    servers, revision = accepted_mcp_servers(orchestrator)
    for server in servers:
        if server.get("name") != name:
            continue
        if tool_name is not None:
            server["disabled_tools"] = updated_tool_list(
                server.get("disabled_tools", []), tool_name, disabled
            )
        else:
            server["disabled"] = disabled

        async def prepare(candidate: ConfigCandidate[ChartreuxConfigSchema]) -> None:
            if preflight is not None:
                await preflight(candidate.config)
            if candidate_preflight is not None:
                await candidate_preflight(candidate)

        return await save_mcp_servers(
            orchestrator,
            servers,
            revision=revision,
            reason="Toggle MCP setting",
            preflight=prepare if preflight or candidate_preflight else None,
            apply=apply,
        )
    return None


__all__ = ["persist_mcp_toggle", "updated_tool_list"]
