from __future__ import annotations

from enum import StrEnum, auto
from functools import cache
from typing import Annotated, Any, Literal, Protocol, Self, get_origin

from pydantic import (
    Field,
    JsonValue,
    StrictInt,
    StrictStr,
    TypeAdapter,
    model_validator,
)

from chartreux.app_server._connection_protocol import (
    CallbackKind as CallbackKind,
    ClientCapabilities as ClientCapabilities,
    ClientInfo as ClientInfo,
    ClientToolCapability as ClientToolCapability,
    ClientToolMethod as ClientToolMethod,
    ClientToolReadTextFileParams as ClientToolReadTextFileParams,
    ClientToolReadTextFileResponse as ClientToolReadTextFileResponse,
    ClientToolTerminalCreateParams as ClientToolTerminalCreateParams,
    ClientToolTerminalCreateResponse as ClientToolTerminalCreateResponse,
    ClientToolTerminalOutputResponse as ClientToolTerminalOutputResponse,
    ClientToolTerminalParams as ClientToolTerminalParams,
    ClientToolTerminalWaitResponse as ClientToolTerminalWaitResponse,
    ClientToolWriteTextFileParams as ClientToolWriteTextFileParams,
    InitializeParams as InitializeParams,
    InitializeResponse as InitializeResponse,
    ServerInfo as ServerInfo,
    TransportKind as TransportKind,
)
from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.config import ConfigView, ProxySettingsView
from chartreux.app_server.models import (
    AgentStatsSnapshot,
    CallbackOutput,
    ConfigIssue,
    ContentBlock,
    DebugLogPage,
    IdentityView,
    JsonPatchOperation,
    MCPState,
    MentionStats,
    MessageAnnotations as MessageAnnotations,
    PreparedPrompt,
    PublicCallbackEntry,
    PublicError,
    PublicHistoryEntry,
    PublicRetryCategory,
    PublicSession,
    PublicSessionState,
    PublicTurn,
    PublicTurnQueue,
    ScheduledLoop,
    SessionContentBlock,
    SessionEmbeddedResourceContentBlock as SessionEmbeddedResourceContentBlock,
    SessionImageContentBlock as SessionImageContentBlock,
    SessionLogSummary,
    SessionResourceLinkContentBlock as SessionResourceLinkContentBlock,
    SessionTextContentBlock as SessionTextContentBlock,
    SkillSummary,
    ToolSummary,
    TurnContextInputEntry as TurnContextInputEntry,
    TurnInputEntry,
    TurnUserInputEntry,
    UserDisplayContent,
    WorkspaceTrustDecision,
    WorkspaceTrustDetails,
    WorkspaceTrustStatus,
    validate_turn_input_entries,
)
from chartreux.app_server.review import (
    ReviewFile,
    ReviewFileStatus,
    ReviewHunk,
    ReviewOwner,
    ReviewScope,
    ReviewTarget,
)
from chartreux.utils.mcp import MCPAddTransport

SERVER_METHODS: tuple[str, ...] = (
    "agent/transcript/get",
    "callback/result",
    "config/fields/read",
    "config/proxy/read",
    "config/proxy/write",
    "config/read",
    "config/reload",
    "config/policy/read",
    "config/policy/replace",
    "policy/roots/read",
    "policy/roots/replace",
    "config/schema",
    "config/write",
    "diagnostics/list",
    "diagnostics/logs/read",
    "events/read",
    "session/history/get",
    "identity/read",
    "loops/clear",
    "loops/create",
    "loops/delete",
    "loops/list",
    "mcp_catalog/add",
    "mcp_catalog/login",
    "mcp_catalog/logout",
    "mcp_catalog/read",
    "mcp_catalog/refresh",
    "mcp_catalog/remove",
    "mcp_catalog/toggle",
    "narration/summarize",
    "review/approve",
    "review/baseline",
    "review/hunks",
    "review/revert",
    "review/state",
    "review/turnDiff",
    "runtime/read",
    "session/compact",
    "session/continue",
    "session/context/inject",
    "session/delete",
    "session/fork",
    "session/history/clear",
    "session/history/list",
    "session/list",
    "session/log/read",
    "session/read",
    "session/ready/read",
    "session/ready/wait",
    "session/relocate",
    "session/rename",
    "session/resume",
    "session/rewind",
    "session/rewind/read",
    "session/settings/update",
    "session/shellCommand",
    "session/start",
    "session/stop",
    "session/title/update",
    "session/turns/list",
    "shell/interrupt",
    "shell/run",
    "skills/installed",
    "skills/list",
    "stats/read",
    "tools/list",
    "app_server/session/turn/enqueue",
    "app_server/session/turn/queue/read",
    "app_server/session/turn/queue/remove",
    "app_server/session/turn/queue/replace",
    "app_server/session/turn/queue/resume",
    "turn/interrupt",
    "turn/start",
    "turn/steer",
    "workspace/git/checkouts",
    "workspace/git/worktrees/list",
    "workspace/git/worktrees/remove",
    "workspace/prompt/prepare",
    "workspace/trust/decision",
    "workspace/trust/untrustedConfig",
    "workspace/trust/status",
)


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


class SessionOpenParams(ProtocolModel):
    agent_config: AgentConfig = Field(default_factory=AgentConfig)
    history_limit: int = Field(default=200, ge=1, le=500)

    @property
    def cwd(self) -> str | None:
        return self.agent_config.cwd or self.agent_config.workdir


class SessionKind(StrEnum):
    """Lifecycle role of a session as seen by the server.

    ``NORMAL`` — a genuine user-initiated session; emits new-session telemetry
    and is persisted to disk as soon as a turn runs.

    ``EPHEMERAL`` — a throwaway session used to warm up the runtime while the
    in-app picker is shown; it is discarded on resume and must not emit
    new-session telemetry or be counted as a new session.
    """

    NORMAL = auto()
    EPHEMERAL = auto()


class SessionStartParams(SessionOpenParams):
    idempotency_key: str | None = None
    kind: SessionKind = SessionKind.NORMAL


class SessionStartResponse(EventWatermarkResponse):
    state: PublicSessionState


class SessionReadParams(ProtocolModel):
    session_id: str
    history: PageRequest | None = Field(default_factory=PageRequest)
    turns: PageRequest | None = Field(default_factory=PageRequest)

    @property
    def include_history(self) -> bool:
        return self.history is not None

    @property
    def include_turns(self) -> bool:
        return self.turns is not None

    @property
    def history_limit(self) -> int:
        return self.history.limit if self.history is not None else 1

    @property
    def turns_limit(self) -> int:
        return self.turns.limit if self.turns is not None else 1


class SessionReadResponse(EventWatermarkResponse):
    state: PublicSessionState


class SessionResumeParams(SessionOpenParams):
    session_id: str


class SessionResumeResponse(EventWatermarkResponse):
    state: PublicSessionState


class SessionContinueParams(SessionOpenParams):
    """Chartreux extension that resumes the latest eligible session."""


class SessionContinueResponse(EventWatermarkResponse):
    state: PublicSessionState


class SessionForkParams(ProtocolModel):
    idempotency_key: str | None = None
    source_session_id: str
    entry_id: str | None = None
    agent_config: AgentConfig | None = None
    after_turn_id: str | None = None
    history_limit: int = Field(default=200, ge=1, le=500)
    attach: bool = True


class SessionForkResponse(EventWatermarkResponse):
    source_session_id: str
    state: PublicSessionState


class SessionStopParams(ProtocolModel):
    session_id: str
    reason: str | None = None


class SessionStopResponse(ProtocolModel):
    closed: bool = True


class SessionCloseParams(ProtocolModel):
    session_id: str


class SessionCloseResponse(ProtocolModel):
    closed: bool = True


class SessionListParams(ProtocolModel):
    cursor: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    root_session_id: str | None = None
    parent_session_id: str | None = None
    cwd: str | None = None


class SessionListResponse(ProtocolModel):
    items: list[PublicSession] = Field(default_factory=list)
    next_cursor: str | None = None
    previous_cursor: str | None = None
    # The session `--continue` would resume: the tty-scoped last-session
    # pointer when it still exists, else the most recently updated session.
    # Resolved server-side so the pointer stays behind the app-server boundary.
    continue_session_id: str | None = None

    @property
    def data(self) -> list[PublicSession]:
        return self.items


class SessionDeleteParams(ProtocolModel):
    session_id: str


class SessionTitleUpdateParams(ProtocolModel):
    session_id: str
    title: str


class SessionTitleUpdateResponse(ProtocolModel):
    title: str
    updated_at: str | None = None
    last_event_id: int | None = None


class SessionHistoryListParams(ProtocolModel):
    session_id: str
    turn_id: str | None = None
    page: PageRequest = Field(default_factory=PageRequest)

    @property
    def cursor(self) -> str | None:
        return self.page.cursor

    @property
    def limit(self) -> int:
        return self.page.limit

    @property
    def sort_direction(self) -> Literal["forward", "backward"]:
        return self.page.direction


class SessionHistoryListResponse(ProtocolModel):
    items: list[PublicHistoryEntry] = Field(default_factory=list)
    next_cursor: str | None = None
    previous_cursor: str | None = None

    @property
    def data(self) -> list[PublicHistoryEntry]:
        return self.items

    @property
    def backwards_cursor(self) -> str | None:
        return self.previous_cursor


class SessionHistoryGetParams(ProtocolModel):
    session_id: str
    history_limit: int = Field(default=200, ge=1, le=500)


class SessionHistoryGetResponse(ProtocolModel):
    history: list[PublicHistoryEntry]


MAX_AGENT_TRANSCRIPT_ID_LENGTH = 512
MAX_AGENT_TRANSCRIPT_CURSOR_LENGTH = 4096
MAX_AGENT_TRANSCRIPT_DISPLAY_TEXT_LENGTH = 8 * 1024


class AgentTranscriptEntryKind(StrEnum):
    USER_TEXT = "user_text"
    ASSISTANT_TEXT = "assistant_text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class AgentTranscriptTruncation(StrEnum):
    DISPLAY_TEXT_LIMIT = "display_text_limit"


class AgentTranscriptEntry(ProtocolModel):
    """One chronological, text-only entry in an agent transcript viewer page."""

    entry_id: str = Field(min_length=1, max_length=MAX_AGENT_TRANSCRIPT_ID_LENGTH)
    kind: AgentTranscriptEntryKind
    display_text: str = Field(max_length=MAX_AGENT_TRANSCRIPT_DISPLAY_TEXT_LENGTH)
    tool_name: str | None = Field(
        default=None, max_length=MAX_AGENT_TRANSCRIPT_ID_LENGTH
    )
    tool_call_id: str | None = Field(
        default=None, max_length=MAX_AGENT_TRANSCRIPT_ID_LENGTH
    )
    truncated: bool = False
    truncation: AgentTranscriptTruncation | None = None

    @model_validator(mode="after")
    def validate_tool_identity_and_truncation(self) -> Self:
        if self.kind in {
            AgentTranscriptEntryKind.USER_TEXT,
            AgentTranscriptEntryKind.ASSISTANT_TEXT,
        } and (self.tool_name is not None or self.tool_call_id is not None):
            raise ValueError("text entries must not carry tool identity")
        if self.kind is AgentTranscriptEntryKind.TOOL_CALL and self.tool_name is None:
            raise ValueError("tool call entries require tool_name")
        if (
            self.kind is AgentTranscriptEntryKind.TOOL_RESULT
            and self.tool_call_id is None
        ):
            raise ValueError("tool result entries require tool_call_id")
        if self.truncated != (self.truncation is not None):
            raise ValueError("truncated and truncation must agree")
        return self


class AgentTranscriptGetParams(ProtocolModel):
    """Read a bounded, backward page from an attached parent's agent transcript."""

    agent_id: str = Field(min_length=1, max_length=MAX_AGENT_TRANSCRIPT_ID_LENGTH)
    before: str | None = Field(
        default=None, max_length=MAX_AGENT_TRANSCRIPT_CURSOR_LENGTH
    )
    limit: int = Field(default=50, ge=1, le=200)


class AgentTranscriptState(StrEnum):
    AVAILABLE = "available"
    NO_SAVED_TRANSCRIPT = "no_saved_transcript"
    CHANGED = "changed"
    EXCEEDS_VIEWER_LIMIT = "exceeds_viewer_limit"


class AgentTranscriptGetResponse(ProtocolModel):
    """Typed result of ``agent/transcript/get``; never contains paths or session IDs.

    ``available`` is the only state that carries a page.  It always carries
    ``entries`` (which may be empty for a valid saved-empty transcript),
    ``oldestCursor``, and ``hasMore``.  The other states carry neither page
    fields nor a cursor: ``no_saved_transcript`` means no saved transcript is
    available, ``changed`` means ``Transcript changed; refresh``, and
    ``exceeds_viewer_limit`` means ``Transcript exceeds viewer limit``.

    A ``before`` cursor encodes the stable digest of its boundary entry: its
    persisted message ID when present plus a content-digest hash of that
    entry's canonical serialization.  On page-up, the reader must locate that
    boundary by ID and digest; if it is absent or the digest differs, it must
    return ``changed`` rather than an empty page.  The cursor for a response is
    derived from the oldest entry actually returned.
    """

    state: AgentTranscriptState
    entries: list[AgentTranscriptEntry] | None = None
    oldest_cursor: str | None = Field(
        default=None, max_length=MAX_AGENT_TRANSCRIPT_CURSOR_LENGTH
    )
    has_more: bool | None = None

    @model_validator(mode="after")
    def validate_state_payload(self) -> Self:
        if self.state is AgentTranscriptState.AVAILABLE:
            if self.entries is None or self.has_more is None:
                raise ValueError("available transcript responses require page fields")
        elif (
            self.entries is not None
            or self.oldest_cursor is not None
            or self.has_more is not None
        ):
            raise ValueError("non-available transcript responses must not carry a page")
        return self


class AgentTranscriptSource(Protocol):
    """WP6 viewer source contract, implemented by WP4's session-resources facade.

    The facade method is ``read_agent_transcript``.  WP5 emits
    ``AgentSidebar.TranscriptOpen`` and WP6 emits
    ``AgentTranscriptViewer.Closed``; WP7 owns their lifecycle wiring.
    """

    async def read_agent_transcript(
        self, agent_id: str, *, before: str | None = None, limit: int = 50
    ) -> AgentTranscriptGetResponse: ...


class SessionTurnsListParams(ProtocolModel):
    session_id: str
    page: PageRequest = Field(default_factory=PageRequest)

    @property
    def cursor(self) -> str | None:
        return self.page.cursor

    @property
    def limit(self) -> int:
        return self.page.limit

    @property
    def sort_direction(self) -> Literal["forward", "backward"]:
        return self.page.direction


class SessionTurnsListResponse(ProtocolModel):
    items: list[PublicTurn] = Field(default_factory=list)
    next_cursor: str | None = None
    previous_cursor: str | None = None

    @property
    def data(self) -> list[PublicTurn]:
        return self.items

    @property
    def backwards_cursor(self) -> str | None:
        return self.previous_cursor


class SessionShellCommandParams(ProtocolModel):
    session_id: str
    command: str | None = None
    cwd: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0, le=600)
    operation_id: str | None = None
    action: Literal["run", "interrupt"] = "run"

    @model_validator(mode="after")
    def validate_action(self) -> SessionShellCommandParams:
        if self.action == "run" and (self.command is None or not self.command.strip()):
            raise ValueError("command is required for action='run'")
        if self.action == "interrupt" and self.operation_id is None:
            raise ValueError("operation_id is required for action='interrupt'")
        return self


class SessionShellCommandResponse(ProtocolModel):
    accepted: Literal[True] = True
    last_event_id: int


class SessionReadyWaitParams(ProtocolModel):
    session_id: str


class SessionReadyReadParams(ProtocolModel):
    session_id: str


class SessionReadyReadResponse(ProtocolModel):
    ready: bool


class SessionReadyWaitResponse(ProtocolModel):
    ready: bool = True
    init_duration_ms: int | None = None


class IdentityReadParams(ProtocolModel):
    session_id: str


class IdentityReadResponse(ProtocolModel):
    identity: IdentityView | None = None


class SessionRewindReadParams(ProtocolModel):
    """Inspect a session-local, memory-only checkpoint lost on restart or resume."""

    session_id: str
    entry_id: str


class SessionRewindReadResponse(ProtocolModel):
    has_file_changes: bool
    paths: list[str] = Field(default_factory=list)


class SessionRewindParams(ProtocolModel):
    """Rewind via a session-local, memory-only checkpoint lost on restart or resume."""

    session_id: str
    entry_id: str
    restore_files: bool = False
    inplace: bool = False


class SessionRewindResponse(ProtocolModel):
    message: str
    restore_errors: list[str]
    restored_paths: list[str]
    state: PublicSessionState
    session_log: SessionLogSummary


class SessionRelocateParams(ProtocolModel):
    session_id: str
    cwd: str


class SessionRelocateResponse(ProtocolModel):
    state: PublicSessionState


class ReviewStateParams(ProtocolModel):
    """Read review state backed by session-local, memory-only checkpoints.

    Checkpoints are lost when the process restarts or the session is resumed.
    """

    session_id: str


class ReviewStateResponse(ProtocolModel):
    files: list[ReviewFile]
    scopes: list[ReviewScope]


class ReviewBaselineParams(ProtocolModel):
    session_id: str
    path: str


class ReviewBaselineResponse(ProtocolModel):
    content: str


class ReviewTurnDiffParams(ProtocolModel):
    session_id: str
    path: str
    owner: ReviewOwner


class ReviewTurnDiffResponse(ProtocolModel):
    status: ReviewFileStatus
    baseline: str
    current: str


class ReviewHunksParams(ProtocolModel):
    session_id: str
    path: str
    owner: ReviewOwner | None = None


class ReviewHunksResponse(ProtocolModel):
    hunks: list[ReviewHunk]


class ReviewMutationParams(ProtocolModel):
    session_id: str
    target: ReviewTarget


class ConfigSchemaReadParams(ProtocolModel):
    pass


class ConfigSchemaReadResponse(ProtocolModel):
    config_schema_version: str
    config_schema: dict[str, JsonValue] = Field(alias="schema")


class RootsReadParams(ProtocolModel):
    session_id: str


class RootsReadResponse(ProtocolModel):
    revision: str
    project: str
    roots: list[str]


class RootsReplaceParams(ProtocolModel):
    session_id: str
    expected_revision: str
    scope: Literal["session"] = "session"
    user_initiated: Literal[True]
    roots: list[str]


class RootsReplaceResponse(ProtocolModel):
    revision: str
    runtime: RuntimeSnapshot


class PolicyToolReplacement(ProtocolModel):
    permission: Literal["ask", "always", "never"] = "ask"
    denylist: list[str] = Field(default_factory=list)
    sensitive_patterns: list[str] = Field(default_factory=list)


class PolicyReadParams(ProtocolModel):
    session_id: str


class PolicyReadResponse(ProtocolModel):
    revision: str
    sources: list[str]


class PolicyReplaceParams(ProtocolModel):
    session_id: str
    source: str
    expected_revision: str
    scope: Literal["session"]
    user_initiated: Literal[True]
    tools: dict[str, PolicyToolReplacement]


class PolicyReplaceResponse(ProtocolModel):
    revision: str
    runtime: RuntimeSnapshot


class ConfigReloadParams(ProtocolModel):
    session_id: str
    reload_runtime: bool = True


class ConfigProxyReadParams(ProtocolModel):
    session_id: str


class ConfigProxyReadResponse(ProtocolModel):
    settings: ProxySettingsView


class ConfigProxyWriteParams(ProtocolModel):
    session_id: str
    changes: dict[str, str | None]


type NonNegativeStrictInt = Annotated[StrictInt, Field(ge=0)]


class SessionSettingsUpdateParams(ProtocolModel):
    session_id: str
    max_turns: NonNegativeStrictInt | None = None
    max_tokens: NonNegativeStrictInt | None = None

    @model_validator(mode="after")
    def require_update(self) -> SessionSettingsUpdateParams:
        if self.max_turns is None and self.max_tokens is None:
            raise ValueError("At least one session setting must be provided")
        return self


class RuntimeSnapshot(ProtocolModel):
    config: ConfigView
    skills: list[SkillSummary]
    tools: list[ToolSummary]
    stats: AgentStatsSnapshot
    context_window: int
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
    session_id: str
    runtime: RuntimeSnapshot


class TurnRetryingParams(ProtocolModel):
    session_id: str
    category: PublicRetryCategory
    detail: str


class ServerWarningParams(ProtocolModel):
    warning: PublicError


class ServerErrorParams(ProtocolModel):
    error: PublicError


class ConfigMutationResponse(RuntimeMutationResponse):
    stripped_history_images: int = 0
    launch_metadata_persisted: bool = True


class ConfigFieldKind(StrEnum):
    BOOL = auto()
    ENUM = auto()
    INT = auto()
    FLOAT = auto()
    STR = auto()
    LIST = auto()
    COMPLEX = auto()


class ConfigLayerValueWire(ProtocolModel):
    layer: str
    value: JsonValue = None


class ConfigFieldWire(ProtocolModel):
    name: str
    kind: ConfigFieldKind
    description: str
    value: JsonValue = None
    path: str
    popular: bool = False
    enum_choices: list[str] = Field(default_factory=list)
    value_labels: dict[str, str] = Field(default_factory=dict)
    layer_values: list[ConfigLayerValueWire] = Field(default_factory=list)
    writable_targets: list[str] = Field(default_factory=list)

    @property
    def origin(self) -> str:
        return self.layer_values[0].layer if self.layer_values else "default"


class ConfigFieldsReadParams(ProtocolModel):
    session_id: str


class ConfigFieldsReadResponse(ProtocolModel):
    fields: list[ConfigFieldWire]
    targets: list[str]
    revisions: dict[str, str] = Field(default_factory=dict)


class ConfigWriteOpWire(ProtocolModel):
    op: Literal["set", "remove"]
    path: str
    value: JsonValue = None
    target_layer: str | None = None


class ConfigWriteParams(ProtocolModel):
    session_id: str
    ops: list[ConfigWriteOpWire]
    reason: str = "config write"
    reload_runtime: bool = False
    target: Literal["session", "user", "project"] = "session"
    expected_revision: str | None = None


class ConfigWriteResponse(ConfigMutationResponse):
    rejected: bool = False
    failures: list[str] = Field(default_factory=list)
    target: Literal["session", "user", "project"] = "session"
    persistence: Literal["not_saved", "saved", "durability_uncertain"] = "not_saved"
    application: Literal["unchanged", "applied", "failed"] = "unchanged"
    revision: str | None = None
    fields: list[ConfigFieldWire] = Field(default_factory=list)
    saved_values: dict[str, JsonValue] = Field(default_factory=dict)


class ConfigReadParams(ProtocolModel):
    session_id: str | None = None
    cwd: str | None = None


class ConfigReadResponse(ProtocolModel):
    config: ConfigView
    startup_issue: ConfigIssue | None = None
    stripped_history_images: int = 0
    skills_count: int = 0
    hooks_count: int = 0
    mcp_servers_total: int = 0
    mcp_servers_enabled: int = 0


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
    context_window: int


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


class MCPReadParams(ProtocolModel):
    session_id: str


class MCPReadResponse(ProtocolModel):
    mcp: MCPState


class MCPRefreshParams(ProtocolModel):
    session_id: str


class MCPToggleParams(ProtocolModel):
    session_id: str | None = None
    name: str
    disabled: bool
    tool_name: str | None = None


class MCPAddParams(ProtocolModel):
    session_id: str | None = None
    url: str
    name: str | None = None
    scopes: list[str] = Field(default_factory=list)
    transport: MCPAddTransport = "streamable-http"


class MCPAddResponse(ProtocolModel):
    name: str
    url: str
    created: bool
    runtime: RuntimeSnapshot | None = None


class MCPCatalogMutationResponse(ProtocolModel):
    runtime: RuntimeSnapshot | None = None


class MCPRemoveParams(ProtocolModel):
    session_id: str | None = None
    name: str


class MCPRemoveResponse(ProtocolModel):
    name: str
    removed: bool
    runtime: RuntimeSnapshot | None = None


class MCPLogoutParams(ProtocolModel):
    session_id: str | None = None
    name: str


class MCPLoginParams(ProtocolModel):
    session_id: str | None = None
    name: str


class MCPAuthUrlParams(ProtocolModel):
    name: str
    url: str


class MCPAuthRequiredParams(ProtocolModel):
    session_id: str
    name: str
    descriptor_revision: str
    observed_connection_revision: str | None = None


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


class SessionLogReadParams(ProtocolModel):
    session_id: str


class SessionLogReadResponse(ProtocolModel):
    log: SessionLogSummary


class WorkspacePromptPrepareParams(ProtocolModel):
    session_id: str
    message: str
    title_content: list[ContentBlock] | None = None


class WorkspacePromptPrepareResponse(ProtocolModel):
    prompt: PreparedPrompt


class WorkspaceTrustStatusParams(ProtocolModel):
    cwd: str | None = None


class WorkspaceTrustStatusResponse(ProtocolModel):
    status: WorkspaceTrustStatus
    details: WorkspaceTrustDetails | None = None


class WorkspaceWorktreeListParams(ProtocolModel):
    cwd: str = Field(min_length=1)
    # Off by default because this listing sits on the read path: it resolves
    # the checkout behind every session read and enumerates a project's
    # directories for every session list. The details cost a merge base and a
    # diff per branch plus a second repository open, which only a caller that
    # renders them should pay.
    include_details: bool = False


class WorkspaceGitBranchChanges(ProtocolModel):
    additions: int
    deletions: int


class WorkspaceLinkedWorktree(ProtocolModel):
    name: str
    branch: str
    cwd: str
    root: str
    repo_root: str
    # Absent unless asked for, and null when there is no base to measure
    # against, which is not the same as a branch that has changed nothing.
    branch_changes: WorkspaceGitBranchChanges | None = None


class WorkspaceWorktreeListResponse(ProtocolModel):
    worktrees: list[WorkspaceLinkedWorktree]
    # The branch the main checkout is on. Absent unless details were asked for,
    # and null for a detached one. The worktree entries never name it: this
    # listing reports the linked worktrees, and the main checkout is not one.
    repository_branch: str | None = None
    # Where the position this listing was taken from sits in the main checkout,
    # under the same checks the worktree entries pass. Null when it does not
    # sit there at all -- a subdirectory that exists only on a feature branch
    # has no counterpart. A caller offering the main checkout as a destination
    # must take this rather than joining the root itself, because a path this
    # omits is one a move would refuse.
    repository_cwd: str | None = None


# No session_id on the wire: the caller is deleting a session that has already
# closed and dropped its holder, and accepting one would let a client name an
# arbitrary holder file to unlink.
class WorkspaceWorktreeRemoveParams(ProtocolModel):
    cwd: str = Field(min_length=1)


# Spelled out here rather than imported from chartreux.core.git.worktree: the protocol
# is the wire contract and must not pull core into the app-server clients.
type WorktreeRemoveOutcome = Literal[
    "removed",
    "kept_dirty",
    "kept_in_use",
    "kept_unmanaged",
    # Distinct from kept_unmanaged: the worktree is Chartreux's and the removal
    # itself failed. Collapsing the two would report a failure as "not ours".
    "kept_error",
    "not_found",
]


class WorkspaceWorktreeRemoveResponse(ProtocolModel):
    # A kept worktree is a normal outcome the caller has to render, not a fault,
    # so every case answers with a result rather than a JSON-RPC error.
    outcome: WorktreeRemoveOutcome
    root: str | None = None
    branch: str | None = None
    branch_deleted: bool = False
    reasons: list[str] = Field(default_factory=list)


class WorkspaceGitCheckoutsParams(ProtocolModel):
    # Every repository the project links, asked for together, because which one
    # holds the session cannot be decided from any single one. A managed
    # worktree lives outside the repository it belongs to, and a repository
    # linked inside another would otherwise let both claim the session.
    repo_local_paths: list[str]
    # Absent for a session with no working directory, which on a cloud host is
    # every session.
    session_cwd: str | None = None


class WorkspaceGitCheckout(ProtocolModel):
    repo_local_path: str
    # False when the repository could not be read; `message` says why and the
    # rest is absent. Carried rather than raised so one unreadable repository
    # does not cost the answer for the others.
    ok: bool
    # The repository the session is standing in. At most one is.
    is_primary: bool = False
    repo_url: str | None = None
    root: str | None = None
    # Absent when the session sits in the repository's own checkout rather than
    # in one of its worktrees.
    worktree: str | None = None
    branch: str | None = None
    base_branch: str | None = None
    message: str | None = None


class WorkspaceGitCheckoutsResponse(ProtocolModel):
    checkouts: list[WorkspaceGitCheckout] = Field(default_factory=list)


class WorkspaceTrustDecisionParams(ProtocolModel):
    decision: WorkspaceTrustDecision
    cwd: str | None = None
    session_id: str | None = None


class WorkspaceUntrustedConfigParams(ProtocolModel):
    cwd: str | None = None


class WorkspaceUntrustedConfigResponse(ProtocolModel):
    dirs: list[str] = Field(default_factory=list)
    settings_path: str = ""


class LoopsListParams(ProtocolModel):
    session_id: str


class LoopsListResponse(ProtocolModel):
    loops: list[ScheduledLoop]


class LoopsCreateParams(ProtocolModel):
    session_id: str
    interval: str
    prompt: str


class LoopsCreateResponse(ProtocolModel):
    loop: ScheduledLoop


class LoopsDeleteParams(ProtocolModel):
    session_id: str
    loop_id: str


class LoopsDeleteResponse(ProtocolModel):
    loop: ScheduledLoop


class LoopsClearParams(ProtocolModel):
    session_id: str


class LoopsClearResponse(ProtocolModel):
    count: int


class NarrationSummarizeParams(ProtocolModel):
    session_id: str
    user_message: str
    assistant_text: str
    error: str | None = None
    message_id: str | None = None


class NarrationSummarizeResponse(ProtocolModel):
    summary: str | None = None


class _TurnQueueInputParams(ProtocolModel):
    idempotency_key: str | None = None
    session_id: str
    entries: list[TurnInputEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        validate_turn_input_entries(self.entries)
        return self

    @property
    def input(self) -> list[SessionContentBlock]:
        return [block for entry in self.entries for block in entry.content]

    @property
    def user_entry(self) -> TurnUserInputEntry | None:
        return next(
            (entry for entry in reversed(self.entries) if entry.role == "user"), None
        )

    @property
    def message_entry_id(self) -> str | None:
        user_entry = self.user_entry
        return user_entry.entry_id if user_entry is not None else None


class TurnEnqueueParams(_TurnQueueInputParams):
    pass


class TurnEnqueueResponse(ProtocolModel):
    queue_item_id: str


class TurnQueueReplaceParams(_TurnQueueInputParams):
    queue_item_id: str

    def as_enqueue_params(self) -> TurnEnqueueParams:
        return TurnEnqueueParams(
            idempotency_key=self.idempotency_key,
            session_id=self.session_id,
            entries=self.entries,
        )


class TurnQueueReplaceResponse(ProtocolModel):
    queue_item_id: str


class TurnQueueReadParams(ProtocolModel):
    session_id: str


class TurnQueueReadResponse(ProtocolModel):
    queue: PublicTurnQueue


class TurnQueueRemoveParams(ProtocolModel):
    session_id: str
    queue_item_id: str


class TurnQueueRemoveResponse(ProtocolModel):
    pass


class TurnQueueResumeParams(ProtocolModel):
    session_id: str


class TurnQueueResumeResponse(ProtocolModel):
    pass


class TurnStartParams(ProtocolModel):
    idempotency_key: str | None = None
    session_id: str
    message: list[ContentBlock]
    injected: bool = False
    user_initiated_retry: bool = False
    client_user_message_id: str | None = None
    auto_title: str | None = None
    user_display_content: UserDisplayContent | None = None
    mention_stats: MentionStats | None = None

    @property
    def input(self) -> list[ContentBlock]:
        return self.message


class TurnStartResponse(EventWatermarkResponse):
    turn: PublicTurn


class TurnSteerParams(ProtocolModel):
    idempotency_key: str | None = None
    session_id: str
    expected_turn_id: str
    message: list[ContentBlock]
    client_user_message_id: str | None = None
    inject_invoked_skill: bool = True
    mention_stats: MentionStats | None = None

    @property
    def input(self) -> list[ContentBlock]:
        return self.message


class TurnSteerResponse(EventWatermarkResponse):
    accepted: Literal[True] = True


class TurnInterruptParams(ProtocolModel):
    session_id: str
    expected_turn_id: str


class TurnInterruptResponse(EventWatermarkResponse):
    accepted: Literal[True] = True


class ContextInjectParams(ProtocolModel):
    session_id: str
    input: list[ContentBlock]
    as_message: bool = False
    inject_invoked_skill: bool = False
    client_user_message_id: str | None = None
    mention_stats: MentionStats | None = None


class ContextInjectResponse(ProtocolModel):
    entries: list[PublicHistoryEntry]


class CallbackCallParams(ProtocolModel):
    callback: PublicCallbackEntry


class CallbackCallResponse(ProtocolModel):
    callback_id: str
    accepted: bool = True


class CallbackRespondParams(ProtocolModel):
    session_id: str
    callback_id: str
    output: CallbackOutput


class CallbackRespondResponse(ProtocolModel):
    status: Literal["accepted", "duplicate"]


class CallbackResultError(ProtocolModel):
    message: str
    code: str | None = None
    details: JsonValue = None


class CallbackResult(ProtocolModel):
    callback_id: str
    output: JsonValue = None
    error: CallbackResultError | None = None


class CallbackResultParams(ProtocolModel):
    session_id: str
    result: CallbackResult

    @property
    def callback_id(self) -> str:
        return self.result.callback_id


class CallbackResultResponse(EventWatermarkResponse):
    accepted: Literal[True] = True


class SessionHistoryClearParams(ProtocolModel):
    session_id: str


class SessionHistoryClearResponse(ProtocolModel):
    state: PublicSessionState
    session_log: SessionLogSummary


class SessionCompactParams(ProtocolModel):
    session_id: str
    extra_instructions: str = ""


class SessionCompactResponse(ProtocolModel):
    summary: str
    state: PublicSessionState
    session_log: SessionLogSummary


class EventNotificationParams(ProtocolModel):
    event_id: int = Field(ge=0, strict=True)
    session_id: str
    emitted_at: int


class HistoryEntryAddedParams(EventNotificationParams):
    turn_id: str | None = None
    entry: PublicHistoryEntry


class HistoryEntryUpdatedParams(EventNotificationParams):
    turn_id: str | None = None
    entry_id: str
    patch: list[JsonPatchOperation]


class SessionSnapshotParams(EventNotificationParams):
    state: PublicSessionState


class SessionHandoffParams(EventNotificationParams):
    old_session_id: str
    state: PublicSessionState
    session_log: SessionLogSummary


class SessionCompactedParams(SessionHandoffParams):
    summary_length: int = Field(ge=0)


class SessionContextClearedParams(SessionHandoffParams):
    plan_file_path: str | None = None


class SessionUpdatedParams(EventNotificationParams):
    patch: list[JsonPatchOperation]


class TurnQueueUpdatedParams(EventNotificationParams):
    queue: PublicTurnQueue


class TurnStartedParams(EventNotificationParams):
    turn: PublicTurn


class TurnCompletedParams(EventNotificationParams):
    turn: PublicTurn


class AgentSummaryModel(ProtocolModel):
    agent_id: str
    profile: str
    availability: str
    current_run_id: str | None = None
    current_run_status: str | None = None
    last_run_status: str | None = None
    initial_task_summary: str | None = None
    current_task_summary: str | None = None
    idle_seconds: float | None = None
    ttl_remaining_seconds: float | None = None
    effective_model: str | None = None
    base_model: str | None = None
    active_provider: str | None = None
    effective_thinking: str | None = None
    result_expired: bool = False


class AgentEvictionModel(ProtocolModel):
    agent_id: str
    run_id: str
    reason: Literal["ttl", "idle_cap"]
    idle_duration_seconds: float
    root_generation: int


class AgentsUpdateParams(EventNotificationParams):
    agents: list[AgentSummaryModel]
    evictions: list[AgentEvictionModel] = Field(default_factory=list)


class StatsUpdatedParams(EventNotificationParams):
    stats: AgentStatsSnapshot
    context_window: int


class Notification(ProtocolModel):
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, JsonValue]


class ServerRequest(ProtocolModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: StrictInt | StrictStr
    method: str
    params: dict[str, JsonValue]


class ProtocolErrorCode(StrEnum):
    INVALID_REQUEST = auto()
    INVALID_PARAMS = auto()
    NOT_INITIALIZED = auto()
    NOT_FOUND = auto()
    CONFLICT = auto()
    STALE_TURN = auto()
    NOT_STEERABLE = auto()
    CALLBACK_CLOSED = auto()
    COMPACTION_FAILED = auto()
    UNAUTHORIZED = auto()
    FORBIDDEN = auto()
    METHOD_NOT_FOUND = auto()
    NOT_IMPLEMENTED = auto()
    STALE_CURSOR = auto()
    INTERNAL_ERROR = auto()


class InvalidParamsIssue(ProtocolModel):
    path: list[str | int]
    message: str


class InvalidParamsData(ProtocolModel):
    error_count: int
    issues: list[InvalidParamsIssue]


class ProtocolError(ProtocolModel):
    code: ProtocolErrorCode
    message: str
    data: JsonValue = None


_REDACTED_VALIDATION_VALUE = "[redacted]"
_VALIDATION_MESSAGE_TEMPLATES = {
    "missing": "Field required",
    "type_error": "Invalid type",
    "value_error": "Invalid value",
}


@cache
def _schema_field_names() -> frozenset[str]:
    return frozenset(
        name
        for model in ProtocolModel.__subclasses__()
        for field_name, field in model.model_fields.items()
        for name in (field_name, field.alias)
        if name is not None
    )


@cache
def _mapping_schema_field_names() -> frozenset[str]:
    return frozenset(
        name
        for model in ProtocolModel.__subclasses__()
        for field_name, field in model.model_fields.items()
        if get_origin(field.annotation) is dict
        for name in (field_name, field.alias)
        if name is not None
    )


def redact_validation_message(error_type: str) -> str:
    """Render a diagnostic from the Pydantic error type, never its input message."""
    if error_type in _VALIDATION_MESSAGE_TEMPLATES:
        return _VALIDATION_MESSAGE_TEMPLATES[error_type]
    if error_type.endswith("_type") or error_type.endswith("_parsing"):
        return "Invalid type"
    if error_type.startswith("value_error") or error_type.endswith("_value"):
        return "Invalid value"
    return "Invalid value"


def redact_validation_path(path: tuple[str | int, ...]) -> list[str | int]:
    """Keep schema fields and indices while hiding submitted mapping keys."""
    schema_fields = _schema_field_names()
    mapping_fields = _mapping_schema_field_names()
    submitted_mapping = False
    redacted: list[str | int] = []
    for segment in path:
        if isinstance(segment, int):
            redacted.append(segment)
            continue
        if submitted_mapping or segment not in schema_fields:
            redacted.append(_REDACTED_VALIDATION_VALUE)
        else:
            redacted.append(segment)
        if segment in mapping_fields:
            submitted_mapping = True
    return redacted


def format_invalid_params_issues(data: JsonValue) -> str | None:
    """Render field-level detail carried by an INVALID_PARAMS error."""
    if not isinstance(data, dict):
        return None
    issues = data.get("issues")
    if not isinstance(issues, list) or not issues:
        return None
    parts: list[str] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        path = issue.get("path")
        location = (
            ".".join(str(segment) for segment in path)
            if isinstance(path, list) and path
            else "<root>"
        )
        message = issue.get("message")
        parts.append(f"{location}: {message}" if message else location)
    if not parts:
        return None
    return "; ".join(parts)


def _render_protocol_error(error: ProtocolError) -> str:
    detail = format_invalid_params_issues(error.data)
    if detail is None:
        return error.message
    return f"{error.message} ({detail})"


class AppServerResponseError(RuntimeError):
    def __init__(self, error: ProtocolError) -> None:
        self.error = error
        super().__init__(_render_protocol_error(error))


class JsonRpcProtocolError(RuntimeError):
    pass


class JsonRpcSuccessResponse(ProtocolModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: StrictInt | StrictStr
    result: dict[str, JsonValue]


class JsonRpcErrorResponse(ProtocolModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: StrictInt | StrictStr
    error: ProtocolError


type JsonRpcEnvelope = (
    Notification | ServerRequest | JsonRpcSuccessResponse | JsonRpcErrorResponse
)


@cache
def _json_rpc_envelope_adapter() -> TypeAdapter[JsonRpcEnvelope]:
    return TypeAdapter(JsonRpcEnvelope)


def validate_json_rpc_envelope(value: object) -> JsonRpcEnvelope:
    return _json_rpc_envelope_adapter().validate_python(
        value, by_alias=True, by_name=False
    )


def protocol_value(value: ProtocolModel | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, ProtocolModel):
        return value.model_dump(mode="json", by_alias=True)
    return value


def validate_callback_acknowledgement(
    callback_id: str, response: CallbackCallResponse
) -> CallbackCallResponse:
    if callback_id != response.callback_id:
        raise ValueError("Callback acknowledgement does not match the request")
    return response
