"""Checked user-source saves shared by the MCP helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from chartreux.core.config._restrictions import ConfigCandidate
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.patch import AddOperationPatch
from chartreux.core.config.types import ConcurrencyConflictError, ConfigSaveResult

type MCPPreflight = Callable[[ConfigCandidate[ChartreuxConfigSchema]], Awaitable[None]]
type MCPApply = Callable[[ConfigCandidate[ChartreuxConfigSchema]], None]


class MCPSaveError(ValueError):
    """Value-free save diagnostics, including replacement without application."""

    def __init__(
        self, message: str, *, save_result: ConfigSaveResult | None = None
    ) -> None:
        super().__init__(message)
        self.save_result = save_result


class MCPSaveConflictError(MCPSaveError, ConcurrencyConflictError):
    def __init__(self, result: ConfigSaveResult) -> None:
        ValueError.__init__(
            self, "MCP user configuration changed; reload before retrying."
        )
        self.save_result = result
        self.expected_fp = "accepted"
        self.actual_fp = "changed"


def accepted_mcp_servers(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    error_type: type[MCPSaveError] = MCPSaveError,
) -> tuple[list[dict[str, Any]], str]:
    """Capture data and revision together, without accepting a fresh disk read."""
    layers = [layer for layer in orchestrator.layers if type(layer) is UserConfigLayer]
    if len(layers) == 1:
        layer = layers[0]
        data, revision = layer.cached_data, layer.fingerprint
        if data is not None and revision:
            raw = data.model_dump().get("mcp_servers", [])
            if isinstance(raw, list) and all(isinstance(item, dict) for item in raw):
                return raw, revision
    raise error_type(
        "MCP save requires an accepted actual user source and revision.",
        save_result=ConfigSaveResult(
            "user", "not_saved", "unchanged", error="validation"
        ),
    ) from None


async def save_mcp_servers(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    servers: list[dict[str, Any]],
    *,
    revision: str,
    reason: str,
    error_type: type[MCPSaveError] = MCPSaveError,
    preflight: MCPPreflight | None = None,
    apply: MCPApply | None = None,
) -> ConfigSaveResult:
    result = await orchestrator.save(
        [AddOperationPatch(path="/mcp_servers", value=servers)],
        target="user",
        expected_revision=revision,
        reason=reason,
        preflight=preflight,
        apply=apply,
    )
    if result.error == "conflict":
        raise MCPSaveConflictError(result) from None
    if result.persistence != "saved" or result.application != "applied" or result.error:
        raise error_type(
            f"MCP save: persistence={result.persistence}; application={result.application}; "
            f"error={result.error or 'none'}.",
            save_result=result,
        ) from None
    return result
