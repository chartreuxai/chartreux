from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from itertools import pairwise
import logging
import math
import os
from pathlib import Path
import re
import shlex
from string import Formatter
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from chartreux.config_values import THINKING_LEVELS as THINKING_LEVELS, ThinkingLevel
from chartreux.core.config._defaults import DEFAULT_AUTO_COMPACT_THRESHOLD
from chartreux.core.config.agent_selection import (
    AgentSelectionConfig as AgentSelectionConfig,
)
from chartreux.core.config.schema import WithDeepMerge
from chartreux.core.dangerous_commands import OPERATION_REGISTRY
from chartreux.core.dispatch.route_models import PurposeRoute
from chartreux.core.llm_models import Backend
from chartreux.core.paths import SESSION_LOG_DIR
from chartreux.utils.paths import get_chartreux_home_literal

logger = logging.getLogger(__name__)


class MissingAPIKeyError(RuntimeError):
    def __init__(self, env_key: str, provider_name: str) -> None:
        super().__init__(
            f"Missing {env_key} environment variable for {provider_name} provider. "
            "Run `chartreux --setup` or configure this provider in Settings > Providers (/providers)."
        )
        self.env_key = env_key
        self.provider_name = provider_name


def normalize_authorized_roots(value: dict[str, list[str]]) -> dict[str, list[str]]:
    """Canonicalize explicit absolute roots without granting effective config authority."""
    result: dict[str, list[str]] = {}
    try:
        for project, roots in value.items():
            paths = [Path(item).expanduser() for item in (project, *roots)]
            if any(not path.is_absolute() for path in paths):
                raise ValueError
            canonical = [str(path.resolve()) for path in paths]
            key, *resolved_roots = canonical
            if key in result:
                raise ValueError
            result[key] = list(dict.fromkeys(resolved_roots))
    except (ValueError, OSError, RuntimeError):
        raise ValueError(
            "Invalid authorized_roots_by_project: absolute paths and unique canonical project keys required"
        ) from None
    return result


class AuthorizedRootsInput(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    authorized_roots_by_project: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("authorized_roots_by_project")
    @classmethod
    def canonical_roots(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        return normalize_authorized_roots(value)


def normalize_dangerous_command_grants(
    value: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Canonicalize saved grants without granting effective config authority.

    Only persistable operation IDs survive: the frozen registry marks every
    consequential operation non-persistable, so a saved grant can never carry
    one even if a source spells it out.
    """
    result: dict[str, list[str]] = {}
    try:
        for project, operations in value.items():
            if not isinstance(project, str) or not isinstance(operations, list):
                raise ValueError
            project_path = Path(project).expanduser()
            if not project_path.is_absolute():
                raise ValueError
            key = str(project_path.resolve())
            if key in result:
                raise ValueError
            persistable: list[str] = []
            for operation_id in operations:
                if not isinstance(operation_id, str):
                    raise ValueError
                operation = OPERATION_REGISTRY.get(operation_id)
                if operation is None or not operation.persistable:
                    raise ValueError
                if operation_id not in persistable:
                    persistable.append(operation_id)
            result[key] = persistable
    except (ValueError, TypeError, OSError, RuntimeError):
        raise ValueError(
            "Invalid dangerous_command_grants_by_project: absolute project paths, "
            "unique canonical project keys, and persistable operation IDs required"
        ) from None
    return result


class DangerousCommandGrantsInput(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    dangerous_command_grants_by_project: dict[str, list[str]] = Field(
        default_factory=dict
    )

    @field_validator("dangerous_command_grants_by_project")
    @classmethod
    def canonical_grants(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        return normalize_dangerous_command_grants(value)


class ProjectContextConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    default_commit_count: int = 5
    timeout_seconds: float = 2.0


class SubagentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idle_ttl_seconds: int = Field(default=300, ge=0, strict=True)
    max_idle_agents: int = Field(default=4, ge=0, strict=True)
    retain_agents: bool = Field(default=False, strict=True)
    tombstone_ttl_seconds: int = Field(default=900, ge=0, strict=True)
    max_tombstones: int = Field(default=32, ge=1, strict=True)
    max_running_subagents: int = Field(default=16, ge=1, strict=True)
    # Per-run budget defaults for subagent runs (B2c): spend and token counts
    # are measured relative to each run's admission baseline. None means
    # unconfigured: the run-budget middleware is omitted entirely.
    run_max_price: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    run_max_session_tokens: int | None = Field(default=None, ge=1, strict=True)


class RecoveryEligibleClass(StrEnum):
    """Failure classes the recovery layer may act on (plan WP1b).

    This is the ONE authoritative recovery-layer allowlist (state.md R0
    eligibility note): the transient FailureInfo classes connection, timeout,
    server_error, overload, and rate_limit, plus the WP1 typed transient
    dispositions incomplete_stream and empty_response, plus the transient
    form of launch_failure. The low-level classes conflict and too_early are
    deliberately ABSENT — they stay transport-retry-eligible only, and their
    recovery-layer eligibility is decided explicitly here, by exclusion.
    """

    CONNECTION = "connection"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    OVERLOAD = "overload"
    RATE_LIMIT = "rate_limit"
    LAUNCH_FAILURE = "launch_failure"
    INCOMPLETE_STREAM = "incomplete_stream"
    EMPTY_RESPONSE = "empty_response"


class RecoveryConfig(BaseModel):
    """Validated child-subagent recovery policy (plan WP1b/WP4; handoff §2.3).

    This is the ``[recovery.subagents]`` section: the bounded AUTOMATIC
    recovery controller for child subagents (WP4). It declares counts,
    delays, jitter, correlation thresholds, deadlines, eligible classes,
    suspension limits, recovery concurrency, and eligible fallback
    constraints. AUTOMATIC RECOVERY IS ENABLED by default since v0.5.6
    (promoted at WP6 per OD5 once WP4's acceptance matrix passed;
    ``enabled`` defaults to True). Every deadline here is an ADVISORY
    scheduling/admission bound (Q2/F3: no hard in-flight kill exists —
    the retry budget is consulted only between attempts), and safety
    exclusions are code invariants, never config escape hatches.

    The section lives inside the ``[recovery]`` container
    (:class:`RecoverySectionConfig`); the root-turn controller's
    ``[recovery.root]`` section is added by R2.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, strict=True)
    auto_retries_max: int = Field(default=2, ge=0, le=10, strict=True)
    same_agent_resumes_max: int = Field(default=2, ge=0, strict=True)
    cooldowns_seconds: list[float] = Field(default_factory=lambda: [30.0, 120.0])
    jitter_fraction: float = Field(default=0.2, ge=0.0, le=1.0, allow_inf_nan=False)
    # Advisory aggregate episode bound (handoff §2.3: ~40 minutes including
    # the initial failed call); never an in-flight kill.
    episode_deadline_seconds: float = Field(default=2400.0, gt=0.0, allow_inf_nan=False)
    endpoint_correlation_window_seconds: float = Field(
        default=300.0, gt=0.0, allow_inf_nan=False
    )
    endpoint_correlation_min_events: int = Field(default=2, ge=2, strict=True)
    endpoint_correlation_min_models: int = Field(default=2, ge=2, strict=True)
    endpoint_cooldown_seconds: float = Field(default=30.0, ge=0.0, allow_inf_nan=False)
    eligible_classes: list[RecoveryEligibleClass] = Field(
        default_factory=lambda: list(RecoveryEligibleClass)
    )
    # Warm-suspension limits (handoff §1.5: SEPARATE suspension TTL/count/byte
    # limits from idle retention and tombstones). ``suspension_ttl_seconds`` is
    # the per-record warm window wired by WP3: 0 disables warm suspension
    # entirely (a recoverable failure closes/idles through the normal path).
    # ``max_suspended_agents`` bounds how many records may hold a suspended
    # warm conversation (0 disables suspension). ``max_suspended_bytes`` bounds
    # the TOTAL deterministic warm-conversation byte accounting across
    # suspended records (WP3; 0 disables byte accounting); when admitting a new
    # suspension would exceed the cap, the OLDEST suspended records expire
    # first (deterministic selection), and a record whose conversation alone
    # exceeds the cap is never suspended. All three caps are enforced in code,
    # never prose.
    suspension_ttl_seconds: float = Field(default=1800.0, ge=0.0, allow_inf_nan=False)
    max_suspended_agents: int = Field(default=8, ge=0, strict=True)
    max_suspended_bytes: int = Field(default=67108864, ge=0, strict=True)
    recovery_concurrency: int = Field(default=2, ge=1, strict=True)
    # Eligible fallback constraints (handoff §2.2/Q8; WP5 failure-time
    # replacement policy): a fallback must not share the ATTEMPTED failed
    # endpoint; an empty allowlist adds no base-model restriction beyond
    # these constraints and the price cap. Endpoint independence is enforced
    # as a code invariant by the policy layer; the flag only records the
    # operator's declared constraint. ``fallback_replacements_max_per_package``
    # bounds cross-model replacements per work package — the counter spans the
    # package lineage (a replacement cannot reset it), and exhaustion parks the
    # package (park-and-continue).
    fallback_require_endpoint_independence: bool = Field(default=True, strict=True)
    fallback_allowed_base_models: list[str] = Field(default_factory=list)
    fallback_max_output_price_per_million: float | None = Field(
        default=None, gt=0.0, allow_inf_nan=False
    )
    fallback_replacements_max_per_package: int = Field(default=2, ge=0, strict=True)

    @field_validator("cooldowns_seconds")
    @classmethod
    def validate_cooldowns(cls, value: list[float]) -> list[float]:
        for entry in value:
            if not math.isfinite(entry) or entry < 0:
                raise ValueError(
                    "cooldowns_seconds entries must be finite, nonnegative durations"
                )
        return value

    @field_validator("fallback_allowed_base_models")
    @classmethod
    def validate_fallback_allowed_base_models(cls, value: list[str]) -> list[str]:
        for entry in value:
            if not entry.strip():
                raise ValueError(
                    "fallback_allowed_base_models entries must be non-empty "
                    "canonical model names"
                )
        return value

    @model_validator(mode="after")
    def validate_recovery_policy(self) -> RecoveryConfig:
        if self.enabled and self.auto_retries_max == 0:
            raise ValueError(
                "recovery.enabled requires auto_retries_max >= 1: an enabled "
                "controller with no automatic retries is a contradiction"
            )
        cooldowns = self.cooldowns_seconds
        if any(later < earlier for earlier, later in pairwise(cooldowns)):
            raise ValueError("cooldowns_seconds must be nondecreasing")
        if self.auto_retries_max > 0 and len(cooldowns) < self.auto_retries_max:
            raise ValueError(
                "cooldowns_seconds needs at least auto_retries_max entries "
                "(one delay per automatic retry)"
            )
        return self


def _reject_bool_duration(value: Any) -> Any:
    """Reject boolean-as-number for duration fields (plan U/R2).

    Pydantic's lax float coercion accepts ``True`` as ``1.0``; a boolean is
    never a valid duration, so it is rejected before the float validators
    run.
    """
    if isinstance(value, bool):
        raise ValueError("durations must be numbers, not booleans")
    return value


class RootRecoveryConfig(BaseModel):
    """Validated root-turn automatic recovery policy (plan R2/OD3; handoff §2.3).

    This is the ``[recovery.root]`` section: the bounded AUTOMATIC recovery
    controller for terminal root model-call failures. When enabled, the
    harness schedules the continuation turn itself — an injected new turn
    carrying the retry prompt, ``user_initiated_retry=False`` (an automatic
    continuation NEVER clears shared availability cooldowns; only a genuine
    user-initiated /retry does) — bounded by one shared episode budget that
    MERGES the legacy incomplete-stream continuation, cooldown-aware through
    the shared availability registry, with the manual ``/retry`` affordance
    (R1) as the fallback on exhaustion.

    REMOVAL-BY-DEFAULT (OD4a, user-confirmed): when ``enabled`` is False, NO
    root automatic continuation runs AT ALL — including the migrated
    incomplete-stream path. One controller, no hidden second scheduler; the
    breaking change is documented in the release notes. AUTOMATIC RECOVERY
    IS ENABLED by default since v0.5.6 (promoted at WP6 per OD4b once R2's
    acceptance matrix passed; ``enabled`` defaults to True) — set it to False
    to return to manual-retry-only recovery. The aggregate
    ``deadline_seconds`` is an ADVISORY scheduling/admission bound (Q2
    negative: no hard in-flight kill exists), and the safety exclusions
    (never auto-retry preflight, permission/authorization, budget stop,
    cancellation, task blocker, permanent config/auth/billing/quota/
    context-size, unknown-internal; never clear cooldowns) are CODE
    invariants, never config escape hatches. Episode policy is snapshotted
    at episode creation — a config reload never extends an active episode
    or replenishes its count.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, strict=True)
    # ONE shared automatic-turn budget per user-turn episode, merging the
    # legacy incomplete-stream continuation (limit 2) into this controller.
    auto_retries_max: int = Field(default=2, ge=0, le=10, strict=True)
    # Advisory aggregate episode bound: no new admission after it, never an
    # in-flight kill (Q2/F3 negative honored).
    deadline_seconds: Annotated[float, BeforeValidator(_reject_bool_duration)] = Field(
        default=2400.0, ge=1.0, le=86400.0, allow_inf_nan=False
    )
    # Per-retry backoff before the cooldown-aware admission; entry i serves
    # the i-th admitted continuation.
    cooldowns_seconds: list[float] = Field(default_factory=lambda: [30.0, 120.0])
    jitter_fraction: Annotated[float, BeforeValidator(_reject_bool_duration)] = Field(
        default=0.2, ge=0.0, le=1.0, allow_inf_nan=False
    )
    # Floor for read-only availability re-checks while waiting out a
    # temporary exclusion (a probe in flight, a cooldown gap): waiting never
    # consumes a retry.
    availability_recheck_seconds: Annotated[
        float, BeforeValidator(_reject_bool_duration)
    ] = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    # The recovery-layer eligible classes (the ONE authoritative WP1b
    # allowlist of transient forms). Restricted by type to that allowlist.
    eligible_classes: list[RecoveryEligibleClass] = Field(
        default_factory=lambda: list(RecoveryEligibleClass)
    )

    @field_validator("cooldowns_seconds")
    @classmethod
    def validate_cooldowns(cls, value: list[float]) -> list[float]:
        for entry in value:
            if not math.isfinite(entry) or entry < 0:
                raise ValueError(
                    "cooldowns_seconds entries must be finite, nonnegative durations"
                )
        return value

    @model_validator(mode="after")
    def validate_root_recovery_policy(self) -> RootRecoveryConfig:
        if self.enabled and self.auto_retries_max == 0:
            raise ValueError(
                "recovery.root.enabled requires auto_retries_max >= 1: an "
                "enabled controller with no automatic retries is a contradiction"
            )
        if self.enabled and not self.eligible_classes:
            raise ValueError(
                "recovery.root.enabled requires at least one eligible class: an "
                "enabled controller that may never act is a contradiction"
            )
        cooldowns = self.cooldowns_seconds
        if any(later < earlier for earlier, later in pairwise(cooldowns)):
            raise ValueError("cooldowns_seconds must be nondecreasing")
        if self.auto_retries_max > 0 and len(cooldowns) < self.auto_retries_max:
            raise ValueError(
                "cooldowns_seconds needs at least auto_retries_max entries "
                "(one delay per automatic retry)"
            )
        if self.availability_recheck_seconds > self.deadline_seconds:
            raise ValueError(
                "availability_recheck_seconds must not exceed deadline_seconds"
            )
        return self


class RecoverySectionConfig(BaseModel):
    """The ``[recovery]`` config section: one subsection per controller.

    WP4 (plan round-3 symmetry amendment): the child subagent automatic
    recovery controller's policy is ``[recovery.subagents]``
    (:class:`RecoveryConfig`); the root-turn controller's ``[recovery.root]``
    section (:class:`RootRecoveryConfig`, R2) sits beside it, symmetric with
    ``[user_questions]``. Both subsections keep deep-merge semantics across
    sparse layers, so a layer setting only ``[recovery.root] enabled``
    preserves every other typed default. AUTOMATIC RECOVERY IS ENABLED by
    default in every subsection since v0.5.6 (promoted at WP6 per OD4b/OD5).
    """

    model_config = ConfigDict(extra="forbid")

    subagents: Annotated[RecoveryConfig, WithDeepMerge()] = Field(
        default_factory=RecoveryConfig
    )
    root: Annotated[RootRecoveryConfig, WithDeepMerge()] = Field(
        default_factory=RootRecoveryConfig
    )


class UserQuestionsConfig(BaseModel):
    """The ``[user_questions]`` section: unattended question expiry (plan U/OD1).

    Controls the server-owned unattended-expiry timer for eligible
    ``ask_user_question`` calls: explicit per-question opt-in only (``kind`` +
    ``unattended_default``), never a blanket timeout. EXPIRY IS ENABLED by
    default since v0.5.6 (promoted at WP6 per OD4b once U's acceptance
    matrix passed; the per-question opt-in marker remains the primary
    safety gate regardless). Config changes apply to newly opened
    questions only — the effective policy is snapshotted at callback open.
    Configuration can never force protected questions (``requires_user``,
    missing default, runtime approval provenance) to expire.
    """

    model_config = ConfigDict(extra="forbid")

    unattended_timeout_enabled: bool = Field(default=True, strict=True)
    default_timeout_seconds: Annotated[
        float, BeforeValidator(_reject_bool_duration)
    ] = Field(default=300.0, ge=60.0, le=3600.0, allow_inf_nan=False)
    max_timeout_seconds: Annotated[float, BeforeValidator(_reject_bool_duration)] = (
        Field(default=3600.0, ge=60.0, le=3600.0, allow_inf_nan=False)
    )

    @model_validator(mode="after")
    def validate_timeout_range(self) -> UserQuestionsConfig:
        if self.default_timeout_seconds > self.max_timeout_seconds:
            raise ValueError(
                "user_questions.default_timeout_seconds must not exceed "
                "max_timeout_seconds"
            )
        return self


type StatusLineSegment = Literal[
    "directory",
    "pid",
    "model",
    "context",
    "git-branch",
    "spend-today",
    "spend-week",
    "spend-month",
    "background-jobs",
]


def _default_purpose_routes() -> dict[str, PurposeRoute]:
    from chartreux.core.dispatch.routes import shipped_routes

    return dict(shipped_routes())


class PurposeRouteDefaultsConfig(BaseModel):
    """Typed purpose-route defaults, overridable by explicit route entries."""

    model_config = ConfigDict(extra="forbid")

    routes: dict[str, PurposeRoute] = Field(default_factory=_default_purpose_routes)


class StatusLineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[StatusLineSegment] = Field(
        default_factory=lambda: ["directory", "pid", "context"]
    )
    directory_style: Literal["name", "path"] = "name"
    context_style: Literal["tokens", "tokens-percent"] = "tokens-percent"
    separator: Literal["space", "pipe"] = "pipe"

    @field_validator("segments")
    @classmethod
    def validate_segments(
        cls, value: list[StatusLineSegment]
    ) -> list[StatusLineSegment]:
        if len(value) != len(set(value)):
            raise ValueError("Status line segments must be unique")
        if not {"directory", "context"}.issubset(value):
            raise ValueError("Status line segments require directory and context")
        return value


class SessionLoggingConfig(BaseSettings):
    save_dir: str = ""
    permission_repair_dir: str = Field(default="", exclude=True, repr=False)
    session_prefix: str = "session"
    enabled: bool = True
    # Background LLM-generated session titles (shown in --resume and the
    # terminal tab). Off falls back to the first-message preview.
    generate_titles: bool = False

    @field_validator("save_dir", mode="before")
    @classmethod
    def set_default_save_dir(cls, v: str) -> str:
        if not v:
            return str(SESSION_LOG_DIR.path)
        return v

    @model_validator(mode="before")
    @classmethod
    def preserve_permission_repair_path(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            raw = dict(value)
            save_dir = raw.get("save_dir")
            if isinstance(save_dir, str) and save_dir:
                repair_dir = Path(save_dir).expanduser()
            else:
                repair_dir = get_chartreux_home_literal() / "logs" / "session"
            raw["permission_repair_dir"] = str(repair_dir)
            return raw
        return value

    @field_validator("save_dir", mode="after")
    @classmethod
    def expand_save_dir(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())


class ProviderConfig(BaseModel):
    name: str
    api_base: str
    api_key_env_var: str = ""
    api_style: Literal["openai", "openai-responses", "anthropic"] = "openai"
    backend: Backend = Backend.GENERIC
    reasoning_field_name: str = "reasoning_content"
    # Whether this provider reliably terminates streams with a finish reason.
    # When True, a stream that ends without one is treated as an incomplete
    # stream (and retried). Set to False for OpenAI-compatible endpoints that
    # do not emit a finish reason, to avoid spurious incomplete-stream errors.
    emits_finish_reason: bool = True
    # Whether Responses-compatible providers accept images in function-call output.
    # Set False to project tool-result images into a synthetic user turn instead.
    supports_tool_result_images: bool = True
    extra_headers: dict[str, str] = Field(default_factory=dict)


def normalize_mcp_server_name(value: str | None) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", value)
    return normalized.strip("_-")[:256]


class _MCPBase(BaseModel):
    name: str = Field(description="Short alias used to prefix tool names")
    prompt: str | None = Field(
        default=None, description="Optional usage hint appended to tool descriptions"
    )
    startup_timeout_sec: float = Field(
        default=10.0,
        gt=0,
        description="Timeout in seconds for the server to start and initialize.",
    )
    tool_timeout_sec: float = Field(
        default=60.0, gt=0, description="Timeout in seconds for tool execution."
    )
    disabled: bool = Field(
        default=False,
        description="Disable all tools from this MCP server. Tools are still discovered but hidden.",
    )
    disabled_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names (without the server prefix) to disable from this server. "
            "E.g. ['search', 'read'] to hide '{alias}_search' and '{alias}_read'."
        ),
    )

    @field_validator("name", mode="after")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        normalized = normalize_mcp_server_name(v)
        if not normalized:
            raise ValueError("MCP server name must contain letters or numbers")
        return normalized


_ENV_VAR_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")


class MCPStaticAuth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["static"] = "static"
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers (e.g., Authorization or X-API-Key).",
    )
    api_key_env: str = Field(
        default="",
        description=(
            "Environment variable name containing an API token to send for HTTP transport."
        ),
    )
    api_key_header: str = Field(
        default="Authorization",
        description=(
            "HTTP header name to carry the token when 'api_key_env' is set (e.g., 'Authorization' or 'X-API-Key')."
        ),
    )
    api_key_format: str = Field(
        default="Bearer {token}",
        description=(
            "Format string for the header value when 'api_key_env' is set. Use '{token}' placeholder."
        ),
    )

    @field_validator("headers")
    @classmethod
    def _validate_headers(cls, headers: dict[str, str]) -> dict[str, str]:
        normalized_names: set[str] = set()
        for name in headers:
            if not _HEADER_NAME_PATTERN.fullmatch(name):
                raise ValueError(f"Invalid HTTP header name {name!r}")
            normalized_name = name.lower()
            if normalized_name in normalized_names:
                raise ValueError(f"Duplicate HTTP header {name!r}")
            normalized_names.add(normalized_name)
        return headers

    @field_validator("api_key_env")
    @classmethod
    def _validate_api_key_env(cls, value: str) -> str:
        if value and not _ENV_VAR_PATTERN.fullmatch(value):
            raise ValueError("api_key_env must be a valid environment variable name")
        return value

    @field_validator("api_key_header")
    @classmethod
    def _validate_api_key_header(cls, value: str) -> str:
        if not _HEADER_NAME_PATTERN.fullmatch(value):
            raise ValueError("api_key_header must be a valid HTTP header name")
        return value

    @field_validator("api_key_format")
    @classmethod
    def _validate_api_key_format(cls, value: str) -> str:
        # Enumerate the real replacement fields rather than substring-matching
        # `{token}`: escaped braces (`{{token}}`) contain that substring but
        # yield no field, and a format spec (`{token:>5}`) is valid but omits it.
        try:
            fields = [
                name for _, name, _, _ in Formatter().parse(value) if name is not None
            ]
        except ValueError as exc:
            raise ValueError("api_key_format must be a valid format string") from exc
        if any(name != "token" for name in fields):
            raise ValueError("api_key_format may only reference the token placeholder")
        if "token" not in fields:
            raise ValueError("api_key_format must contain the `{token}` placeholder")
        return value

    @model_validator(mode="after")
    def _warn_inert_api_key_fields(self) -> MCPStaticAuth:
        if not self.api_key_env and (
            self.api_key_header != "Authorization"
            or self.api_key_format != "Bearer {token}"
        ):
            logger.warning(
                "MCP static auth sets api_key_header/api_key_format without "
                "api_key_env; these fields are ignored.",
                extra={
                    "api_key_header": self.api_key_header,
                    "api_key_format": self.api_key_format,
                },
            )
        return self

    def http_headers(self) -> dict[str, str]:
        hdrs = dict(self.headers)
        has_explicit_api_key_header = any(
            name.lower() == self.api_key_header.lower() for name in hdrs
        )
        if (
            not has_explicit_api_key_header
            and self.api_key_env
            and (token := os.getenv(self.api_key_env))
        ):
            hdrs[self.api_key_header] = self.api_key_format.format(token=token)
        return hdrs


class MCPOAuth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["oauth"]
    scopes: list[str] = Field(
        description="OAuth scopes to request. Pass an empty list to accept the AS default."
    )
    client_id: str | None = Field(
        default=None,
        min_length=1,
        description="Pre-registered OAuth public client_id (PKCE). Mutually exclusive with client_metadata_url.",
    )
    client_metadata_url: HttpUrl | None = Field(
        default=None,
        description="RFC 9728 client-metadata-document URL. Mutually exclusive with client_id.",
    )
    redirect_port: int = Field(
        default=47823,
        ge=1024,
        le=65535,
        description="Loopback port for the OAuth callback handler.",
    )

    @model_validator(mode="after")
    def _check_client_identity(self) -> MCPOAuth:
        if self.client_id and self.client_metadata_url:
            raise ValueError("client_id and client_metadata_url are mutually exclusive")
        return self


MCPAuth = Annotated[MCPStaticAuth | MCPOAuth, Field(discriminator="type")]


class _MCPHttpFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="Base URL of the MCP HTTP server")
    auth: MCPAuth = Field(default_factory=MCPStaticAuth)

    @field_validator("url")
    @classmethod
    def _reject_url_userinfo(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
        except ValueError:
            return value
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("MCP HTTP URL must not include userinfo credentials")
        return value

    def http_headers(self) -> dict[str, str]:
        if isinstance(self.auth, MCPStaticAuth):
            return self.auth.http_headers()
        return {}


class MCPHttp(_MCPBase, _MCPHttpFields):
    transport: Literal["streamable-http"]


class MCPStdio(_MCPBase):
    transport: Literal["stdio"]
    command: str | list[str]
    args: list[str] = Field(default_factory=list)
    # Explicit MCP stdio env entries are user-configured per-server overrides;
    # they are not inherited credential_env_passthrough policy.
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to set for the MCP server process.",
    )
    cwd: str | None = Field(
        default=None, description="Working directory for the MCP server process."
    )

    def argv(self) -> list[str]:
        base = (
            shlex.split(self.command)
            if isinstance(self.command, str)
            else list(self.command or [])
        )
        return [*base, *self.args] if self.args else base


MCPServer = Annotated[MCPHttp | MCPStdio, Field(discriminator="transport")]


def _default_alias_to_name(data: Any) -> Any:
    if isinstance(data, dict):
        if "alias" not in data or data["alias"] is None:
            data["alias"] = data.get("name")
    return data


class ModelConfig(BaseModel):
    name: str
    provider: str
    alias: str
    display_name: str | None = None
    temperature: float = 1.0
    input_price: float = 0.0  # Price per million input tokens
    output_price: float = 0.0  # Price per million output tokens
    cached_input_price: float | None = (
        None  # Price per million cached input tokens; None bills them at input_price
    )
    # Catalog materialization retains whether a zero is a real free price or an
    # unknown price represented by the legacy numeric fields.
    input_price_known: bool = True
    output_price_known: bool = True
    cached_input_price_known: bool = True
    thinking: ThinkingLevel = "off"
    supported_thinking_levels: list[ThinkingLevel] | None = None
    supports_images: bool = False
    auto_compact_threshold: int = DEFAULT_AUTO_COMPACT_THRESHOLD
    _default_alias_to_name = model_validator(mode="before")(_default_alias_to_name)


def normalize_model_configs(value: Any) -> Any:
    """Read [[models]] lists or alias maps into the deep-mergeable internal map."""
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for alias, payload in value.items():
            normalized[alias] = _model_payload_with_alias(alias, payload)
        return normalized

    if isinstance(value, list):
        normalized: dict[str, Any] = {}
        for payload in value:
            alias = _model_alias_from_payload(payload)
            normalized[alias] = _model_payload_with_alias(alias, payload)
        return normalized

    return value


def serialize_model_configs(value: Any) -> Any:
    """Write the internal model map back as legacy [[models]] TOML entries.

    None-valued fields are dropped: TOML has no null and tomli_w rejects it.
    """
    normalized = normalize_model_configs(value)
    if not isinstance(normalized, Mapping):
        return normalized
    return [_serialize_model_entry(model) for model in normalized.values()]


def _serialize_model_entry(model: Any) -> Any:
    if isinstance(model, ModelConfig):
        return model.model_dump(exclude_none=True)
    if isinstance(model, Mapping):
        return {key: value for key, value in model.items() if value is not None}
    return model


def _model_alias_from_payload(payload: Any) -> str:
    if isinstance(payload, ModelConfig):
        return payload.alias

    if not isinstance(payload, Mapping):
        raise ValueError("Model entries must be tables with an alias or name.")

    alias = payload.get("alias", payload.get("name"))
    if not isinstance(alias, str) or not alias:
        raise ValueError("Model entries must define a non-empty alias or name.")
    return alias


def _model_payload_with_alias(alias: Any, payload: Any) -> Any:
    if not isinstance(alias, str) or not alias:
        raise ValueError("Model aliases must be non-empty strings.")

    if isinstance(payload, ModelConfig):
        if payload.alias != alias:
            raise ValueError(
                f"Model key '{alias}' does not match model alias '{payload.alias}'."
            )
        return payload

    if not isinstance(payload, Mapping):
        return payload

    model_payload = dict(payload)
    existing_alias = model_payload.get("alias")
    if existing_alias is None:
        model_payload["alias"] = alias
        return model_payload

    if existing_alias != alias:
        raise ValueError(
            f"Model key '{alias}' does not match model alias '{existing_alias}'."
        )
    return model_payload
