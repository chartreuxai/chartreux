"""Runtime snapshot, server read/update, skills/tools/stats/usage/diagnostics, and shell run wire models."""

from __future__ import annotations

from typing import ClassVar

from pydantic import AwareDatetime, Field

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.config import ConfigView
from chartreux.app_server.models import (
    AgentStatsSnapshot,
    ConfigIssue,
    DebugLogPage,
    MCPState,
    PublicError,
    PublicRetryCategory,
    SessionLogSummary,
    SkillSummary,
    ToolSummary,
    UsageComponentBreakdown,
    UsageCoverageWarning,
    UsageModelSummary,
    UsageWindow,
    UsageWindowSummaries,
)


class RuntimeSnapshot(ProtocolModel):
    config: ConfigView
    skills: list[SkillSummary]
    tools: list[ToolSummary]
    stats: AgentStatsSnapshot
    context_window: int | None
    issues: list[ConfigIssue]
    hooks_count: int
    mcp: MCPState


class RuntimeReadParams(ProtocolModel):
    session_id: str


class RuntimeReadResponse(ProtocolModel):
    runtime: RuntimeSnapshot
    session_log: SessionLogSummary
    ready: bool


class RuntimeMutationResponse(ProtocolModel):
    runtime: RuntimeSnapshot


class RuntimeUpdatedParams(ProtocolModel):
    NOTIFICATION_METHOD: ClassVar[str] = "runtime/updated"

    session_id: str
    runtime: RuntimeSnapshot


class TurnRetryingParams(ProtocolModel):
    NOTIFICATION_METHOD: ClassVar[str] = "turn/retrying"

    session_id: str
    category: PublicRetryCategory
    detail: str


class ServerWarningParams(ProtocolModel):
    NOTIFICATION_METHOD: ClassVar[str] = "warning"

    warning: PublicError


class ServerErrorParams(ProtocolModel):
    NOTIFICATION_METHOD: ClassVar[str] = "error"

    error: PublicError


class SkillsListParams(ProtocolModel):
    session_id: str


class SkillsListResponse(ProtocolModel):
    skills: list[SkillSummary]


class SkillsInstalledParams(ProtocolModel):
    session_id: str


class SkillsInstalledResponse(ProtocolModel):
    skills: list[SkillSummary]


class ToolsListParams(ProtocolModel):
    session_id: str


class ToolsListResponse(ProtocolModel):
    tools: list[ToolSummary]


class StatsReadParams(ProtocolModel):
    session_id: str


class StatsReadResponse(ProtocolModel):
    stats: AgentStatsSnapshot
    context_window: int | None


class UsageReadParams(ProtocolModel):
    """Host-level ledger read; no session attachment is required."""

    window: UsageWindow = "day"
    project_key: str | None = None


class UsageReadResponse(ProtocolModel):
    """All windows and selected-window detail share one as_of and revision."""

    as_of: AwareDatetime
    revision: int = Field(ge=0, strict=True)
    summaries: UsageWindowSummaries
    window: UsageWindow = "day"
    models: list[UsageModelSummary] = Field(default_factory=list)
    components: UsageComponentBreakdown = Field(default_factory=UsageComponentBreakdown)
    # Attached root identity, not the request filter; absent before attachment.
    project_key: str | None = None
    warnings: list[UsageCoverageWarning] = Field(default_factory=list)


class DiagnosticsListParams(ProtocolModel):
    session_id: str


class DiagnosticsListResponse(ProtocolModel):
    issues: list[ConfigIssue]
    hooks_count: int


class DiagnosticsLogsReadParams(ProtocolModel):
    session_id: str
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class DiagnosticsLogsReadResponse(ProtocolModel):
    logs: DebugLogPage


class ShellRunParams(ProtocolModel):
    """Internal DTO driving ``ShellController.run`` (no longer a wire model)."""

    session_id: str
    operation_id: str
    command: str
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    cwd: str | None = None


class ShellRunResponse(ProtocolModel):
    """Internal DTO carrying a shell result to the effect/context builders."""

    operation_id: str
    command: str
    cwd: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int
    timed_out: bool = False
    interrupted: bool = False
