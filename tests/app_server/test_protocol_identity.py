"""Identity contracts for the public app-server protocol surface.

The protocol split (v0.5.5 stage 2) moves classes into ``chartreux.app_server``
``.protocol`` submodules while the package initializer re-exports them, so
importers must keep resolving the same objects at the same import path. These
contracts are the move guards for every peel: they pin the notification
registry's population, require every app-server model to rebuild, and require
each public name to stay identical to its defining module's object.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from pydantic import Field, ValidationError
import pytest

from chartreux.app_server._model import ProtocolModel, validate_wire
from chartreux.app_server.models import (
    PublicCompletionAudit,
    PublicFailureEnvelope,
    PublicRecoveryEligibility,
    RunStopReason,
)
import chartreux.app_server.protocol
from chartreux.app_server.protocol import AgentSummaryModel

PRE_SPLIT_NOTIFICATION_METHODS = frozenset({
    "agents/update",
    "callback/childClosed",
    "error",
    "history/entryAdded",
    "history/entryUpdated",
    "mcp_catalog/authRequired",
    "mcp_catalog/authUrl",
    "runtime/updated",
    "session/compacted",
    "session/contextCleared",
    "session/snapshot",
    "session/statsUpdated",
    "session/updated",
    "turn/completed",
    "turn/retrying",
    "turn/started",
    "turn/updated",
    "turn_queue_updated",
    "usage/updated",
    "warning",
})


def _public_protocol_names() -> list[str]:
    return sorted(
        name for name in dir(chartreux.app_server.protocol) if not name.startswith("_")
    )


def _app_server_models() -> list[type[ProtocolModel]]:
    """Every ProtocolModel under the app_server prefix, as the registry walks."""
    models: list[type[ProtocolModel]] = []
    pending = list(ProtocolModel.__subclasses__())
    while pending:
        model = pending.pop()
        pending.extend(model.__subclasses__())
        if model.__module__.startswith("chartreux.app_server"):
            models.append(model)
    return models


def test_public_protocol_names_are_importable() -> None:
    module = chartreux.app_server.protocol
    missing = [name for name in _public_protocol_names() if not hasattr(module, name)]

    assert not missing, missing


def test_public_names_are_identical_in_their_defining_modules() -> None:
    for name in _public_protocol_names():
        value = getattr(chartreux.app_server.protocol, name)
        module = getattr(value, "__module__", None)
        if not isinstance(module, str) or not module.startswith("chartreux.app_server"):
            continue
        owner = importlib.import_module(module)
        assert getattr(owner, name) is value, name


def test_app_server_models_rebuild() -> None:
    models = _app_server_models()
    assert models
    for model in models:
        model.model_rebuild(force=True)


def test_notification_registry_stays_under_app_server() -> None:
    registry = chartreux.app_server.protocol.server_notification_registry()
    assert frozenset(registry) == PRE_SPLIT_NOTIFICATION_METHODS
    assert all(
        model.__module__.startswith("chartreux.app_server")
        for model in registry.values()
    )


PRE_SPLIT_PUBLIC_NAMES: tuple[str, ...] = (
    "AgentConfig",
    "AgentEvictionModel",
    "AgentStatsSnapshot",
    "AgentSummaryModel",
    "AgentTranscriptEntry",
    "AgentTranscriptEntryKind",
    "AgentTranscriptGetParams",
    "AgentTranscriptGetResponse",
    "AgentTranscriptSource",
    "AgentTranscriptState",
    "AgentTranscriptToolStatus",
    "AgentTranscriptTruncation",
    "AgentsCancelParams",
    "AgentsCancelResponse",
    "AgentsReleaseParams",
    "AgentsReleaseResponse",
    "AgentsUpdateParams",
    "Annotated",
    "Any",
    "AppServerResponseError",
    "AutoWorktreeInput",
    "AwareDatetime",
    "CallbackCallParams",
    "CallbackCallResponse",
    "CallbackKind",
    "CallbackOutput",
    "CallbackRespondParams",
    "CallbackRespondResponse",
    "CallbackResult",
    "CallbackResultError",
    "CallbackResultParams",
    "CallbackResultResponse",
    "CancelOutcome",
    "ClassVar",
    "ClientCapabilities",
    "ClientInfo",
    "ClientToolCapability",
    "ClientToolMethod",
    "ClientToolReadTextFileParams",
    "ClientToolReadTextFileResponse",
    "ClientToolTerminalCreateParams",
    "ClientToolTerminalCreateResponse",
    "ClientToolTerminalOutputResponse",
    "ClientToolTerminalParams",
    "ClientToolTerminalWaitResponse",
    "ClientToolWriteTextFileParams",
    "CompletionConfig",
    "ConfigFieldKind",
    "ConfigFieldWire",
    "ConfigFieldsReadParams",
    "ConfigFieldsReadResponse",
    "ConfigIssue",
    "ConfigLayerValueWire",
    "ConfigMutationResponse",
    "ConfigProxyReadParams",
    "ConfigProxyReadResponse",
    "ConfigProxyWriteParams",
    "ConfigReadParams",
    "ConfigReadResponse",
    "ConfigReloadParams",
    "ConfigSchemaReadParams",
    "ConfigSchemaReadResponse",
    "ConfigView",
    "ConfigWriteOpWire",
    "ConfigWriteParams",
    "ConfigWriteResponse",
    "ContentBlock",
    "ContextInjectParams",
    "ContextInjectResponse",
    "DangerousCommandAuthorityReadParams",
    "DangerousCommandAuthorityReadResponse",
    "DangerousCommandDenylistBlockWire",
    "DangerousCommandGrantScopeWire",
    "DangerousCommandGrantWire",
    "DangerousCommandRevocationOutcome",
    "DangerousCommandRevokeParams",
    "DangerousCommandRevokeResponse",
    "DebugLogPage",
    "DiagnosticsListParams",
    "DiagnosticsListResponse",
    "DiagnosticsLogsReadParams",
    "DiagnosticsLogsReadResponse",
    "EffectDetail",
    "EffectState",
    "EmptyResponse",
    "EventBatch",
    "EventNotificationParams",
    "EventWatermarkResponse",
    "EventsFilter",
    "EventsReadParams",
    "ExistingWorktreeInput",
    "Field",
    "HistoryEntryAddedParams",
    "HistoryEntryUpdatedParams",
    "HookDefinition",
    "IdentityReadParams",
    "IdentityReadResponse",
    "IdentityView",
    "InitializeParams",
    "InitializeResponse",
    "InvalidParamsData",
    "InvalidParamsIssue",
    "InventoryItemStateWire",
    "JsonPatchOperation",
    "JsonRpcEnvelope",
    "JsonRpcErrorResponse",
    "JsonRpcProtocolError",
    "JsonRpcSuccessResponse",
    "JsonValue",
    "Literal",
    "LoopsClearParams",
    "LoopsClearResponse",
    "LoopsCreateParams",
    "LoopsCreateResponse",
    "LoopsDeleteParams",
    "LoopsDeleteResponse",
    "LoopsListParams",
    "LoopsListResponse",
    "MAX_AGENT_TRANSCRIPT_CURSOR_LENGTH",
    "MAX_AGENT_TRANSCRIPT_DISPLAY_TEXT_LENGTH",
    "MAX_AGENT_TRANSCRIPT_ID_LENGTH",
    "MCPAddParams",
    "MCPAddResponse",
    "MCPAddTransport",
    "MCPAuthRequiredParams",
    "MCPAuthUrlParams",
    "MCPCatalogMutationResponse",
    "MCPLoginParams",
    "MCPLogoutParams",
    "MCPReadParams",
    "MCPReadResponse",
    "MCPRefreshParams",
    "MCPRemoveParams",
    "MCPRemoveResponse",
    "MCPState",
    "MCPToggleParams",
    "Mapping",
    "MappingProxyType",
    "MentionStats",
    "MessageAnnotations",
    "NarrationSummarizeParams",
    "NarrationSummarizeResponse",
    "NewWorktreeInput",
    "NonNegativeStrictInt",
    "Notification",
    "PageRequest",
    "PolicyReadParams",
    "PolicyReadResponse",
    "PolicyReplaceParams",
    "PolicyReplaceResponse",
    "PolicyToolReplacement",
    "PreparedPrompt",
    "Protocol",
    "ProtocolError",
    "ProtocolErrorCode",
    "ProtocolModel",
    "ProxySettingsView",
    "PublicCallbackEntry",
    "PublicEntryGenerationStatus",
    "PublicError",
    "PublicHistoryEntry",
    "PublicRetryCategory",
    "PublicSession",
    "PublicSessionState",
    "PublicTurn",
    "PublicTurnQueue",
    "ReleaseOutcome",
    "ReviewBaselineParams",
    "ReviewBaselineResponse",
    "ReviewFile",
    "ReviewFileStatus",
    "ReviewHunk",
    "ReviewHunksParams",
    "ReviewHunksResponse",
    "ReviewMutationParams",
    "ReviewOwner",
    "ReviewScope",
    "ReviewStateParams",
    "ReviewStateResponse",
    "ReviewTarget",
    "ReviewTurnDiffParams",
    "ReviewTurnDiffResponse",
    "RootsReadParams",
    "RootsReadResponse",
    "RootsReplaceParams",
    "RootsReplaceResponse",
    "RootsSaveParams",
    "RootsSaveResponse",
    "RunStopReason",
    "RuntimeMutationResponse",
    "RuntimeReadParams",
    "RuntimeReadResponse",
    "RuntimeSnapshot",
    "RuntimeUpdatedParams",
    "SERVER_METHODS",
    "STATUS_LINE_PATHS",
    "ScheduledLoop",
    "Self",
    "ServerErrorParams",
    "ServerInfo",
    "ServerRequest",
    "ServerWarningParams",
    "SessionCloseParams",
    "SessionCloseResponse",
    "SessionCompactParams",
    "SessionCompactResponse",
    "SessionCompactedParams",
    "SessionContentBlock",
    "SessionContextClearedParams",
    "SessionContinueParams",
    "SessionContinueResponse",
    "SessionDeleteParams",
    "SessionEmbeddedResourceContentBlock",
    "SessionForkParams",
    "SessionForkResponse",
    "SessionHandoffParams",
    "SessionHistoryClearParams",
    "SessionHistoryClearResponse",
    "SessionHistoryGetParams",
    "SessionHistoryGetResponse",
    "SessionHistoryListParams",
    "SessionHistoryListResponse",
    "SessionImageContentBlock",
    "SessionKind",
    "SessionListParams",
    "SessionListResponse",
    "SessionLogReadParams",
    "SessionLogReadResponse",
    "SessionLogSummary",
    "SessionMCPHttpServer",
    "SessionMCPServer",
    "SessionMCPStdioServer",
    "SessionOpenParams",
    "SessionOptions",
    "SessionReadParams",
    "SessionReadResponse",
    "SessionReadyReadParams",
    "SessionReadyReadResponse",
    "SessionReadyWaitParams",
    "SessionReadyWaitResponse",
    "SessionRelocateParams",
    "SessionRelocateResponse",
    "SessionResourceLinkContentBlock",
    "SessionResumeParams",
    "SessionResumeResponse",
    "SessionRewindParams",
    "SessionRewindReadParams",
    "SessionRewindReadResponse",
    "SessionRewindResponse",
    "SessionSettingsUpdateParams",
    "SessionShellCommandParams",
    "SessionShellCommandResponse",
    "SessionSnapshotParams",
    "SessionStartParams",
    "SessionStartResponse",
    "SessionStopParams",
    "SessionStopResponse",
    "SessionTextContentBlock",
    "SessionTitleUpdateParams",
    "SessionTitleUpdateResponse",
    "SessionTurnsListParams",
    "SessionTurnsListResponse",
    "SessionUpdatedParams",
    "SettingDescriptorWire",
    "SettingLeafWire",
    "SettingsReadParams",
    "SettingsReadResponse",
    "ShellRunParams",
    "ShellRunResponse",
    "SkillSummary",
    "SkillsInstalledParams",
    "SkillsInstalledResponse",
    "SkillsListParams",
    "SkillsListResponse",
    "StatsReadParams",
    "StatsReadResponse",
    "StatsUpdatedParams",
    "StrEnum",
    "StrictInt",
    "StrictStr",
    "ToolCallPresentation",
    "ToolDefinition",
    "ToolResultPresentation",
    "ToolSummary",
    "ToolsListParams",
    "ToolsListResponse",
    "TransportKind",
    "TurnCompletedParams",
    "TurnContextInputEntry",
    "TurnEnqueueParams",
    "TurnEnqueueResponse",
    "TurnInputEntry",
    "TurnInterruptParams",
    "TurnInterruptResponse",
    "TurnQueueReadParams",
    "TurnQueueReadResponse",
    "TurnQueueRemoveParams",
    "TurnQueueRemoveResponse",
    "TurnQueueReplaceParams",
    "TurnQueueReplaceResponse",
    "TurnQueueResumeParams",
    "TurnQueueResumeResponse",
    "TurnQueueUpdatedParams",
    "TurnRetryingParams",
    "TurnStartParams",
    "TurnStartResponse",
    "TurnStartedParams",
    "TurnSteerParams",
    "TurnSteerResponse",
    "TurnUpdatedParams",
    "TurnUserInputEntry",
    "TypeAdapter",
    "UsageComponentBreakdown",
    "UsageCoverageWarning",
    "UsageModelSummary",
    "UsageReadParams",
    "UsageReadResponse",
    "UsageUpdatedParams",
    "UsageWindow",
    "UsageWindowSummaries",
    "UserDisplayContent",
    "WebSearchSettingsWire",
    "WorkspaceBranchReadParams",
    "WorkspaceBranchReadResponse",
    "WorkspaceGitBranchChanges",
    "WorkspaceGitCheckout",
    "WorkspaceGitCheckoutsParams",
    "WorkspaceGitCheckoutsResponse",
    "WorkspaceLinkedWorktree",
    "WorkspacePromptPrepareParams",
    "WorkspacePromptPrepareResponse",
    "WorkspaceTrustDecision",
    "WorkspaceTrustDecisionParams",
    "WorkspaceTrustDetails",
    "WorkspaceTrustStatus",
    "WorkspaceTrustStatusParams",
    "WorkspaceTrustStatusResponse",
    "WorkspaceUntrustedConfigParams",
    "WorkspaceUntrustedConfigResponse",
    "WorkspaceWorktreeListParams",
    "WorkspaceWorktreeListResponse",
    "WorkspaceWorktreeRemoveParams",
    "WorkspaceWorktreeRemoveResponse",
    "WorktreeInput",
    "WorktreeRemoveOutcome",
    "annotations",
    "auto",
    "cache",
    "datetime",
    "field_validator",
    "format_invalid_params_issues",
    "get_origin",
    "model_validator",
    "protocol_value",
    "redact_validation_message",
    "redact_validation_path",
    "server_notification_registry",
    "validate_callback_acknowledgement",
    "validate_json_rpc_envelope",
    "validate_notification_method",
    "validate_setting_value",
    "validate_turn_input_entries",
)


@pytest.mark.parametrize("name", PRE_SPLIT_PUBLIC_NAMES)
def test_pre_split_public_name_stays_importable_and_identical(name: str) -> None:
    """Every pre-split public protocol name stays at the same import path."""
    value = getattr(chartreux.app_server.protocol, name)
    module = getattr(value, "__module__", None)
    if not isinstance(module, str) or not module.startswith("chartreux.app_server"):
        return
    owner = importlib.import_module(module)
    assert getattr(owner, name) is value, name


def test_no_stale_protocol_module_corpse() -> None:
    """The package conversion was a rename: no dead protocol.py may remain."""
    stale = Path(__file__).parents[2] / "chartreux" / "app_server" / "protocol.py"
    assert not stale.exists()


class _PreChangeAgentSummaryModel(ProtocolModel):
    """The full pre-U2 ``AgentSummaryModel``, frozen field for field.

    pydantic cannot drop an inherited field, so the freeze is a standalone
    mirror rather than a live subclass: every pre-change field is declared
    and ``extra="forbid"`` is inherited, which makes the old-client rejection
    below non-vacuous.
    """

    agent_id: str
    identity: str | None = None
    identity_revision: str | None = None
    name: str
    launch_purposes: list[str] = Field(default_factory=list)
    availability: str
    current_run_id: str | None = None
    current_run_status: str | None = None
    turns_used: int | None = None
    last_run_status: str | None = None
    stop_reason: RunStopReason | None = None
    context_tokens: int | None = None
    context_window: int | None = None
    compacting: bool = False
    initial_task_summary: str | None = None
    current_task_summary: str | None = None
    idle_seconds: float | None = None
    run_elapsed_seconds: float | None = None
    latest_run_id: str | None = None
    ttl_remaining_seconds: float | None = None
    effective_model: str | None = None
    base_model: str | None = None
    active_provider: str | None = None
    effective_thinking: str | None = None
    result_expired: bool = False
    last_failure: PublicFailureEnvelope | None = None
    recovery_chain_id: str | None = None
    same_agent_resumes_used: int | None = None
    resumed_from_run_id: str | None = None
    recovery_eligibility: PublicRecoveryEligibility | None = None
    work_package_id: str | None = None
    predecessor_agent_id: str | None = None
    completion_audit: PublicCompletionAudit | None = None


def test_frozen_pre_change_model_mirrors_the_current_field_set() -> None:
    """U2/A3: the frozen model is the live model minus exactly the new field.

    This guards the freeze against drift: a later field added to the live
    model without updating the mirror fails here instead of silently
    weakening the old-client rejection test.
    """
    current = set(AgentSummaryModel.model_fields) - {"uncollected_run_ids"}

    assert set(_PreChangeAgentSummaryModel.model_fields) == current


def test_pre_change_client_rejects_uncollected_run_ids() -> None:
    """U2/A3: a strict pre-U2 client rejects the always-emitted new key."""
    payload = {"agentId": "agent-1", "name": "worker-base-1", "availability": "idle"}

    # Positive control: the same payload minus the new key validates against
    # the frozen pre-change model, so the rejection below is about the key.
    assert validate_wire(_PreChangeAgentSummaryModel, payload)

    with pytest.raises(ValidationError) as excinfo:
        validate_wire(_PreChangeAgentSummaryModel, payload | {"uncollectedRunIds": []})

    assert excinfo.value.errors()[0]["loc"] == ("uncollectedRunIds",)
