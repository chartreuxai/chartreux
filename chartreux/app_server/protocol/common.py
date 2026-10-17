"""Wire primitives shared across every protocol area."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictInt

from chartreux.app_server._model import ProtocolModel


class EmptyResponse(ProtocolModel):
    pass


class EventWatermarkResponse(ProtocolModel):
    last_event_id: int = 0


class SessionMCPHttpServer(ProtocolModel):
    transport: Literal["streamable-http"]
    name: str
    url: str
    headers: dict[str, str] = Field(default_factory=dict)


class SessionMCPStdioServer(ProtocolModel):
    transport: Literal["stdio"] = "stdio"
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None


type SessionMCPServer = Annotated[
    SessionMCPHttpServer | SessionMCPStdioServer, Field(discriminator="transport")
]


class PageRequest(ProtocolModel):
    cursor: str | None = None
    limit: int = Field(default=200, ge=1, le=500)
    direction: Literal["forward", "backward"] = "backward"


class EventsFilter(ProtocolModel):
    session_ids: list[str] = Field(default_factory=list)
    root_session_ids: list[str] = Field(default_factory=list)
    parent_session_ids: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)

    def unsupported_v01_fields(self) -> list[str]:
        """Return requested filters reserved for a future protocol version."""
        return [
            name
            for name in (
                "session_ids",
                "root_session_ids",
                "parent_session_ids",
                "event_types",
            )
            if getattr(self, name)
        ]


class EventsReadParams(ProtocolModel):
    after_event_id: int | None = None
    filters: EventsFilter = Field(default_factory=EventsFilter)
    batch_size: int = Field(default=100, ge=1)


class EventBatch(ProtocolModel):
    type: Literal["events"] = "events"
    events: list[JsonValue] = Field(default_factory=list)


class CompletionConfig(ProtocolModel):
    type: str = "mistral"
    # Wire-protocol default compared in unsupported_v01_fields, not a model-selection default.
    model: str = "glm-5-2"


class ToolDefinition(ProtocolModel):
    type: Literal["client_tool"] = "client_tool"
    name: str
    description: str = ""
    input_schema: dict[str, JsonValue] = Field(default_factory=dict)
    output_schema: dict[str, JsonValue] = Field(default_factory=dict)


class HookDefinition(ProtocolModel):
    type: str
    name: str
    matcher: dict[str, JsonValue] = Field(default_factory=dict)


class ExistingWorktreeInput(ProtocolModel):
    kind: Literal["existing"] = "existing"
    cwd: str = Field(min_length=1)


class NewWorktreeInput(ProtocolModel):
    kind: Literal["create"] = "create"
    branch: str = Field(min_length=1)
    name: str = Field(min_length=1)


class AutoWorktreeInput(ProtocolModel):
    kind: Literal["auto"] = "auto"
    prompt: str | None = None


type WorktreeInput = Annotated[
    ExistingWorktreeInput | NewWorktreeInput | AutoWorktreeInput,
    Field(discriminator="kind"),
]


class AgentConfig(ProtocolModel):
    """App-server configuration plus Chartreux's runtime launch options."""

    completion: CompletionConfig | None = None
    sandbox: dict[str, JsonValue] | None = None
    instructions: str = ""
    workdir: str | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)
    hooks: list[HookDefinition] = Field(default_factory=list)
    cwd: str | None = None
    workspace_roots: list[str] = Field(default_factory=list)
    worktree: WorktreeInput | None = None
    auto_approve: bool = Field(default=False, exclude=True)
    enabled_tools: list[str] | None = None
    disabled_tools: list[str] = Field(default_factory=list)
    max_turns: int | None = None
    max_price: float | None = None
    max_session_tokens: int | None = None
    headless: bool = False
    trust_workspace: bool = False
    mcp_servers: list[SessionMCPServer] = Field(default_factory=list)

    def unsupported_v01_fields(self) -> list[str]:
        """Return configured fields that v0.1 accepts syntactically but cannot honor."""
        unsupported: list[str] = []
        if self.completion is not None and self.completion != CompletionConfig():
            unsupported.append("completion")
        if self.sandbox is not None:
            unsupported.append("sandbox")
        if self.tools:
            unsupported.append("tools")
        if self.hooks:
            unsupported.append("hooks")
        return unsupported


SessionOptions = AgentConfig


type NonNegativeStrictInt = Annotated[StrictInt, Field(ge=0)]
