"""Canonical task grammar and resource-free launch resolution (design §7, §9).

This module owns the canonical task-argument surface and its resolution:

- the canonical schema :class:`TaskIdentityArgs` — ``model``, ``identity``,
  ``purpose``, and ``seat`` are TOP-LEVEL fields (``model`` is promoted from
  the nested ``config.model``; a dual supply of both is a validation error, no
  precedence guess);
- the shorthand grammar ``@<preset-or-canonical-model>[:<identity>]``
  (``@fast:researcher``, ``@fast:user:foo``) — the identity component is a bare
  built-in or a ``user:``-qualified id, missing components are never guessed,
  and shorthand/structured conflicts are rejected;
- :func:`resolve_identity_launch` — the resource-free resolution: model and
  deployment, thinking, identity revision, authority, and the effective
  package tier are all resolved BEFORE any backend, MCP client, child log,
  handle, or reservation is constructed. A validation failure leaves zero
  side effects.

``TaskArgs`` rewiring onto this schema belongs to package 11b; this module is
written to be consumable by it (the schema, the parser, and the resolver are
all importable without the task tool).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chartreux.core.agents.identities import (
    AgentIdentity,
    Delegation,
    UnknownIdentityError,
    builtin_identities,
    builtin_identity_names,
    get_builtin_identity,
    validate_identity_selection,
)
from chartreux.core.config.harness_files import get_harness_files_manager
from chartreux.core.dispatch.route_models import (
    PackageTier as RoutePackageTier,
    ReplacementCapability,
)
from chartreux.core.dispatch.routes import RouteResolutionError, resolve_route
from chartreux.core.launch_types import (
    LaunchConfig,
    ResolvedIdentityLaunch,
    ResolvedLaunchAuthority,
    validate_difficulty_for_purpose,
)
from chartreux.core.model_catalog.resolver import (
    ModelResolutionError,
    ModelResolver,
    resolver_for,
)
from chartreux.core.policy.composition import (
    IDENTITIES_DIR,
    IDENTITY_IDS,
    ExecutionContext,
    resolve_identity_facts,
)
from chartreux.core.policy.manifest import resource_revision
from chartreux.core.policy.models import ResourceKind
from chartreux.core.policy.resources import ResourceLoader, ResourceResolutionError
from chartreux.core.prompts import PROMPTS_DIR
from chartreux.core.recovery_models import PackageTier
from chartreux.core.session_types import CommittedModelIdentity

if TYPE_CHECKING:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
    from chartreux.core.dispatch.route_models import PurposeRoute

__all__ = [
    "DEFAULT_CHILD_IDENTITY",
    "DEFAULT_LAUNCH_TIER",
    "IdentityLaunchError",
    "InvalidChildIdentityError",
    "InvalidLaunchModelSelectionError",
    "InvalidLaunchPurposeError",
    "ModelShorthand",
    "ResolvedIdentityLaunch",
    "TaskGrammarError",
    "TaskIdentityArgs",
    "parse_model_shorthand",
    "resolve_identity_launch",
    "validate_identity_reference",
    "validate_launch_purpose_binding",
]

#: The default child identity (design §7): a child launch defaults to ``coder``
#: with the bounded-child execution context.
DEFAULT_CHILD_IDENTITY = "coder"

#: The identity/purpose default tier of a direct unrouted launch: the coder
#: default child performs implementation-class work, and an unknown tier
#: defaults conservatively to substantive (the replacement policy's rule).
DEFAULT_LAUNCH_TIER = PackageTier.SUBSTANTIVE

# The same custom-id grammar as chartreux.core.agents.identities (the strict
# ResourceLoader contract): a plain prompt file stem.
_CUSTOM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

# A user-resource directory that never exists: user-qualified custom
# identities fail to resolve instead of falling back to a real directory.
_MISSING_USER_DIR = Path("/nonexistent-chartreux-identity-launch")

#: The shipped overlay bodies (design §9): route-target instruction overlays
#: resolved through the same strict-loader contract as identity bodies.
OVERLAYS_DIR = PROMPTS_DIR / "overlays"
OVERLAY_IDS: tuple[str, ...] = ("review-execution",)


class IdentityLaunchError(ValueError):
    """A canonical launch could not be resolved; carries a safe field path."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.field_path = field
        super().__init__(f"{message} at {field}")


class TaskGrammarError(IdentityLaunchError):
    """A canonical task-argument grammar violation (design §7)."""


class InvalidChildIdentityError(IdentityLaunchError):
    """The resolved child identity is unknown or not child-compatible."""


class InvalidLaunchPurposeError(IdentityLaunchError):
    """The purpose/seat binding cannot be resolved without guessing."""


class InvalidLaunchModelSelectionError(IdentityLaunchError):
    """The model/deployment/thinking selection cannot be resolved."""


@dataclass(frozen=True, slots=True)
class ModelShorthand:
    """The parsed ``@<preset-or-canonical-model>[:<identity>]`` shorthand."""

    model: str
    """The model expression (``@fast``); colon-free by grammar."""
    identity: str | None
    """The identity component (``researcher`` / ``user:foo``); None = absent."""


def parse_model_shorthand(value: str) -> ModelShorthand | None:
    """Parse the model-field shorthand ``@<model>[:<identity>]`` (design §7).

    Returns ``None`` for a plain canonical model name. An ``@role`` expression
    parses as the shorthand with the identity component ABSENT
    (``identity=None`` — missing components are never guessed), and genuinely
    ambiguous colon-bearing expressions are rejected with a pointer to the
    structured fields instead of guessing.

    The identity component is a bare built-in or a ``user:``-qualified id;
    ``builtin:`` qualification and bare custom ids are structured-field-only
    forms (the error says so).
    """
    if not isinstance(value, str) or not value:
        raise TaskGrammarError("model", "the model field must be non-empty text")
    if not value.startswith("@"):
        if ":" in value:
            raise TaskGrammarError(
                "model",
                f"ambiguous model expression {value!r}: a colon-bearing model "
                "value must use the '@<model>:<identity>' shorthand; supply the "
                "structured model and identity fields instead of guessing",
            )
        return None
    model_body, separator, identity = value.partition(":")
    model_body = model_body[1:]
    if not model_body:
        raise TaskGrammarError(
            "model",
            f"malformed model shorthand {value!r}: the model component after "
            "'@' is empty; supply a preset or canonical model name",
        )
    if model_body.startswith("@"):
        raise TaskGrammarError(
            "model",
            f"malformed model shorthand {value!r}: the model component must be "
            "a preset or canonical model name, not a nested '@' expression",
        )
    if not separator:
        return ModelShorthand(model=f"@{model_body}", identity=None)
    if not identity:
        raise TaskGrammarError(
            "model",
            f"malformed model shorthand {value!r}: the identity component "
            "after ':' is empty; drop the ':' or supply a bare built-in / "
            "'user:'-qualified identity",
        )
    if identity.startswith("builtin:"):
        raise TaskGrammarError(
            "model",
            f"malformed model shorthand {value!r}: the identity component "
            "accepts a bare built-in or a 'user:'-qualified id; use the "
            "structured identity field for a 'builtin:'-qualified selection",
        )
    if identity.startswith("user:"):
        name = identity[len("user:") :]
        if not _CUSTOM_ID.fullmatch(name) or name in {".", ".."}:
            raise TaskGrammarError(
                "model",
                f"malformed model shorthand {value!r}: the custom identity "
                "name after 'user:' must be a plain prompt file name (letters, "
                "digits, '_' and '-', starting with a letter or digit) — no "
                "empty, traversal, or extra ':' forms",
            )
    elif ":" in identity:
        raise TaskGrammarError(
            "model",
            f"ambiguous model shorthand {value!r}: the identity component "
            f"{identity!r} carries an extra ':'; supply the structured "
            "identity field instead of guessing",
        )
    elif get_builtin_identity(identity) is None:
        raise TaskGrammarError(
            "model",
            f"malformed model shorthand {value!r}: the identity component "
            f"must be a bare built-in identity ({', '.join(builtin_identity_names())}) "
            f"or a 'user:'-qualified custom id; for a bare custom prompt id "
            f"use 'user:{identity}' or the structured identity field",
        )
    return ModelShorthand(model=f"@{model_body}", identity=identity)


def validate_identity_reference(reference: str) -> str:
    """Grammar-level validation of a structured identity reference (design §4).

    Validates the FORM only: qualification shape, custom-id grammar, reserved
    shipped names, and harness-known built-ins. Resource existence is
    validated at resolution (``validate_identity_selection``) against the user
    prompt directories, and the child-compatibility policy (the
    required-delegation rejection) is enforced there too — before any resource
    construction.
    """
    if not isinstance(reference, str) or not reference:
        raise TaskGrammarError("identity", "the identity field must be non-empty text")
    if reference.startswith("builtin:"):
        name = reference.split(":", 1)[1]
        if not name or ":" in name:
            raise TaskGrammarError(
                "identity",
                f"malformed identity qualification {reference!r}: the name "
                "after 'builtin:' must be a single non-empty segment",
            )
        if get_builtin_identity(name) is None:
            raise TaskGrammarError(
                "identity",
                f"unknown built-in identity {reference!r}; available built-in "
                f"identities: {', '.join(builtin_identity_names())}",
            )
        return reference
    if reference.startswith("user:"):
        name = reference.split(":", 1)[1]
        if not _CUSTOM_ID.fullmatch(name) or name in {".", ".."}:
            raise TaskGrammarError(
                "identity",
                f"malformed custom identity {reference!r}: the name after "
                "'user:' must be a plain prompt file name (letters, digits, "
                "'_' and '-', starting with a letter or digit) — no empty, "
                "traversal, or extra ':' forms",
            )
        if get_builtin_identity(name) is not None:
            raise TaskGrammarError(
                "identity",
                f"reserved identity name {reference!r}: {name!r} is a shipped "
                f"built-in identity; a custom prompt cannot shadow built-in "
                f"identities ({', '.join(builtin_identity_names())}); rename "
                "the prompt file",
            )
        return reference
    if ":" in reference:
        raise TaskGrammarError(
            "identity",
            f"malformed identity qualification {reference!r}: use "
            "'builtin:<name>' or 'user:<name>' qualification, or a bare "
            "built-in / custom prompt id",
        )
    if get_builtin_identity(reference) is not None:
        return reference
    if not _CUSTOM_ID.fullmatch(reference) or reference in {".", ".."}:
        raise TaskGrammarError(
            "identity",
            f"malformed custom identity {reference!r}: a bare custom identity "
            "must be a plain prompt file name (letters, digits, '_' and '-', "
            "starting with a letter or digit)",
        )
    return reference


class TaskIdentityArgs(BaseModel):
    """The canonical task-argument surface (design §7).

    ``model``, ``identity``, ``purpose``, and ``seat`` are top-level fields;
    ``model`` is promoted from the nested ``config.model`` (a dual supply of
    both is a validation error — no precedence guess). The model field accepts
    a plain model expression or the ``@<model>[:<identity>]`` shorthand;
    shorthand/structured conflicts are rejected. The lifecycle fields
    (``background``, ``retain``, ``agent_id``, ``replace_run``,
    ``task_summary``) and the nested ``config`` (additive instructions, tool
    overrides, thinking, lineage) are unchanged from the task tool's surface.
    Package 11b rewires ``TaskArgs`` onto this schema.
    """

    model_config = ConfigDict(extra="forbid")

    task: str = Field(description="The task for the agent to perform")
    model: str | None = Field(
        default=None,
        description=(
            "Canonical model name or '@role'; also accepts the "
            "'@<model>[:<identity>]' shorthand (e.g. '@fast:researcher')."
        ),
    )
    identity: str | None = Field(
        default=None,
        description=(
            "Child identity: a bare built-in ('coder', 'researcher', ...), a "
            "'builtin:'- or 'user:'-qualified id, or a bare custom prompt id."
        ),
    )
    purpose: str | None = Field(
        default=None,
        description="Purpose label selecting the launch route (e.g. 'research').",
    )
    seat: str | None = Field(
        default=None,
        description="Seat selector for multi-seat purposes (e.g. 'review.deep').",
    )
    task_summary: str | None = Field(
        default=None,
        description="Optional short summary of the task for retained-agent status.",
    )
    background: bool = Field(
        default=True,
        description=(
            "If true (the default), launch the subagent in the background and "
            "return a launch acknowledgment with handles instead of blocking "
            "until completion."
        ),
    )
    agent_id: str | None = Field(
        default=None,
        description=(
            "Retained agent instance handle to continue with background: true; "
            "if omitted, create a new instance."
        ),
    )
    replace_run: bool = Field(
        default=False,
        description=(
            "Stop busy background work before reusing agent_id in the same "
            "conversation."
        ),
    )
    retain: bool | None = Field(
        default=None,
        strict=True,
        description=(
            "Retain the agent instance after this run completes so it stays "
            "resident for reuse with agent_id; only valid for background "
            "launches."
        ),
    )
    config: LaunchConfig | None = Field(
        default=None,
        description="Optional semantic launch configuration overrides for this child.",
    )

    @model_validator(mode="before")
    @classmethod
    def _prepare_config(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        config = value.get("config")
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except json.JSONDecodeError as error:
                raise ValueError("config must be a valid JSON object") from error
            if config is not None and not isinstance(config, dict):
                raise ValueError("config JSON must decode to an object")
        if config is None:
            return {key: item for key, item in value.items() if key != "config"}
        return {**value, "config": config}

    @field_validator("task_summary")
    @classmethod
    def _normalize_task_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return " ".join(value.split())[:240]

    @field_validator("identity")
    @classmethod
    def _validate_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_identity_reference(value)

    @field_validator("purpose", "seat")
    @classmethod
    def _non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("purpose and seat must be non-empty text")
        return value

    @model_validator(mode="after")
    def _validate_canonical_grammar(self) -> TaskIdentityArgs:
        # Dual top-level/nested model supply: no precedence guess (design §7).
        if (
            self.model is not None
            and self.config is not None
            and "model" in self.config.model_fields_set
        ):
            raise ValueError(
                "model is supplied both at the top level and as config.model; "
                "supply exactly one (the top-level model field is the "
                "canonical location)"
            )
        # Shorthand/structured conflicts: both supplied for the same component.
        if self.model is not None:
            shorthand = parse_model_shorthand(self.model)
            if (
                shorthand is not None
                and shorthand.identity is not None
                and self.identity is not None
            ):
                raise ValueError(
                    f"the model shorthand {self.model!r} carries an identity "
                    f"component ({shorthand.identity!r}) and the structured "
                    "identity field is also supplied; supply exactly one"
                )
        # A seat selector is meaningless without a purpose route.
        if self.seat is not None and self.purpose is None:
            raise ValueError(
                "seat is supplied without purpose: a seat selects a target of "
                "a multi-seat purpose route; supply the purpose (or drop the "
                "seat)"
            )
        # Purpose-keyed difficulty admission (P1-c2): the demanding route
        # requires a qualifying difficulty category; every other purpose
        # rejects the field.
        validate_difficulty_for_purpose(
            self.purpose, self.config.difficulty if self.config is not None else None
        )
        return self


def _user_prompts_dirs(user_prompts_dirs: Sequence[Path] | None) -> tuple[Path, ...]:
    if user_prompts_dirs is not None:
        return tuple(user_prompts_dirs)
    try:
        return tuple(get_harness_files_manager().user_prompts_dirs)
    except RuntimeError:
        # The harness-files singleton is not initialized (a bare resolution
        # context): built-in identities still resolve; custom identities fail
        # validation with the actionable "no user prompts directories" error.
        return ()


def _identity_resource_loader(directories: Sequence[Path]) -> ResourceLoader:
    """The strict identity loader (the package-1 contract, identity kind only).

    Built-in identity bodies resolve from the shipped identities directory and
    cannot be shadowed; custom identities resolve from the first user prompts
    directory. Resolving a resource is a pure read — no backend, MCP client,
    child log, handle, or reservation is constructed.
    """
    return ResourceLoader(
        {ResourceKind.IDENTITY: directories[0] if directories else _MISSING_USER_DIR},
        {
            ResourceKind.IDENTITY: {
                name: IDENTITIES_DIR / f"{name}.md" for name in IDENTITY_IDS
            }
        },
    )


def _overlay_resource_loader() -> ResourceLoader:
    """The strict overlay loader (the package-1 contract, overlay kind only).

    Overlay bodies are harness-owned route-target instructions (design §9):
    the shipped bodies resolve with builtin semantics and cannot be shadowed,
    and there is no user overlay directory — an overlay reference a shipped
    route does not carry fails loudly here instead of guessing. Resolving a
    resource is a pure read.
    """
    return ResourceLoader(
        {ResourceKind.OVERLAY: _MISSING_USER_DIR},
        {
            ResourceKind.OVERLAY: {
                name: OVERLAYS_DIR / f"{name}.md" for name in OVERLAY_IDS
            }
        },
    )


def _route_inputs(
    config: ChartreuxConfigSchema,
) -> tuple[Mapping[str, object], Mapping[str, str], ModelResolver, Mapping[str, float]]:
    """The selectable model aliases, canonical bases, resolver, and the
    per-base price totals (route inputs; the totals drive the cost-tier
    default).
    """
    resolver = resolver_for(config)
    allowed = config.allowed_models
    catalog = resolver.snapshot.catalog
    aliases: dict[str, object] = {
        base: None for base in resolver.eligible_models(allowed)
    }
    aliases.update({f"@{name}": None for name in resolver.eligible_roles(allowed)})
    canonical: dict[str, str] = {base: base for base in catalog.models}
    canonical.update({f"@{name}": role.model for name, role in catalog.roles.items()})
    return aliases, canonical, resolver, resolver.catalog_prices(allowed)


def _required_delegation_message(reference: str, facts: AgentIdentity) -> str:
    child_ok = sorted(
        identity.id
        for identity in builtin_identities()
        if identity.delegation is not Delegation.REQUIRED
    )
    return (
        f"identity {reference!r} ({facts.display_name}) requires delegation: it "
        "must orchestrate from the root session and cannot be launched as a "
        "child (child launches are depth-limited and never delegate, and "
        "children never inherit the parent's workflow or strategy). Select a "
        f"child-compatible identity ({', '.join(child_ok)}, or a custom "
        "'user:'-qualified identity), or run the orchestrator as the main "
        "session"
    )


def validate_launch_purpose_binding(
    purpose: str,
    *,
    config: ChartreuxConfigSchema,
    seat: str | None = None,
    session_model: str | None = None,
    routes: Mapping[str, PurposeRoute] | None = None,
) -> None:
    """Validate one purpose/seat binding through the canonical route grammar.

    The retained-launch path (a re-task of a retained instance) resolves
    through the SAME canonical purpose grammar as a fresh launch: the purpose
    must name a configured route, the seat must select exactly one of its
    targets (a multi-seat purpose requires the seat), and the route's
    distinct-model requirement is enforced over canonical models exactly as
    a fresh launch enforces it (design §9). The retained instance's
    committed identity and model are pinned by its record, so the route's own
    model/identity defaults are validated for availability only — never
    adopted.
    """
    try:
        aliases, canonical, _resolver, prices = _route_inputs(config)
    except ModelResolutionError as exc:
        raise InvalidLaunchModelSelectionError("model", str(exc)) from exc
    try:
        resolve_route(
            purpose,
            aliases,
            session_model=session_model or config.active_model or "@main",
            seat=seat,
            routes=(routes if routes is not None else config.purpose_routes.routes),
            canonical_models=canonical,
            catalog_prices=prices,
        )
    except RouteResolutionError as exc:
        raise InvalidLaunchPurposeError("purpose", str(exc)) from exc


def resolve_identity_launch(  # noqa: PLR0914, PLR0915
    args: TaskIdentityArgs | Mapping[str, Any],
    *,
    config: ChartreuxConfigSchema,
    authorized_tool_names: frozenset[str],
    session_model: str | None = None,
    user_prompts_dirs: Sequence[Path] | None = None,
    routes: Mapping[str, PurposeRoute] | None = None,
) -> ResolvedIdentityLaunch:
    """Resolve one canonical task launch WITHOUT constructing any resource.

    Pipeline order (every step is pure computation over the parent config, the
    attached catalog, the route table, and the identity registry — no backend,
    MCP client, child log, handle, or reservation is constructed, and a
    validation failure leaves zero side effects):

    1. grammar validation (the canonical schema: dual model supply,
       shorthand/structured conflicts, and malformed shorthand/identity forms
       rejected);
    2. route/purpose/seat resolution (explicit fields > shorthand > route
       defaults; conflicts rejected; multi-seat purposes require a seat);
    3. child-identity validation and the required-delegation rejection
       (orchestrator-class identities cannot be children) — BEFORE any
       resource construction;
    4. model/deployment resolution (the concrete deployment committed at
       launch) and effective-thinking resolution;
    5. the identity revision (the one pure resource READ, through the strict
       loader — the manifest-compatible content revision), plus the matched
       route's overlay body (the same strict-loader contract);
    6. authority resolution (personal/delegation posture selectors plus the
       parent's delegation ceiling) and the effective package tier (the route
       target's explicit tier, or the identity/purpose default for a direct
       unrouted launch) — with the replacement-capability marking kept a
       SEPARATE axis from the package tier.

    ``authorized_tool_names`` is REQUIRED (the parent's delegation ceiling):
    defaulting to "all known tools" here would be a latent fail-open.
    """
    parsed = (
        args
        if isinstance(args, TaskIdentityArgs)
        else TaskIdentityArgs.model_validate(args)
    )
    # Step 1/2: split the model field; resolve the route binding.
    shorthand = parse_model_shorthand(parsed.model) if parsed.model else None
    explicit_model = shorthand.model if shorthand is not None else parsed.model
    explicit_identity = (
        shorthand.identity
        if shorthand is not None and shorthand.identity is not None
        else parsed.identity
    )
    session_expression = session_model or config.active_model or "@main"
    try:
        aliases, canonical, resolver, prices = _route_inputs(config)
    except ModelResolutionError as exc:
        raise InvalidLaunchModelSelectionError("model", str(exc)) from exc
    if parsed.purpose is not None:
        try:
            target = resolve_route(
                parsed.purpose,
                aliases,
                session_model=session_expression,
                seat=parsed.seat,
                model=explicit_model,
                identity=explicit_identity,
                routes=(routes if routes is not None else config.purpose_routes.routes),
                canonical_models=canonical,
                catalog_prices=prices,
            )
        except RouteResolutionError as exc:
            raise InvalidLaunchPurposeError("purpose", str(exc)) from exc
        identity_reference = target.identity
        model_expression = target.model
        # An explicit model override retains the selected route target's TIER.
        effective_tier = PackageTier(RoutePackageTier(target.tier).value)
        overlay = target.overlay
        seat = target.seat
        capability = ReplacementCapability(target.replacement_capability)
    else:
        # A direct unrouted launch uses the identity/purpose default.
        identity_reference = explicit_identity or DEFAULT_CHILD_IDENTITY
        model_expression = explicit_model or session_expression
        effective_tier = DEFAULT_LAUNCH_TIER
        overlay = None
        seat = None
        capability = ReplacementCapability.NONE
    # Step 3: child-identity validation (before any resource construction).
    directories = _user_prompts_dirs(user_prompts_dirs)
    try:
        validated_identity = validate_identity_selection(
            identity_reference, user_prompts_dirs=directories
        )
    except UnknownIdentityError as exc:
        raise InvalidChildIdentityError("identity", str(exc)) from exc
    facts = resolve_identity_facts(validated_identity)
    if facts is not None and facts.delegation is Delegation.REQUIRED:
        raise InvalidChildIdentityError(
            "identity", _required_delegation_message(validated_identity, facts)
        )
    # Step 4: model/deployment + effective thinking.
    explicit_thinking = parsed.config.thinking if parsed.config is not None else None
    try:
        resolved_model = resolver.resolve(
            model_expression,
            allowed_models=config.allowed_models,
            thinking_override=explicit_thinking,
        )
        model = resolved_model.materialize(
            auto_compact_threshold=config.auto_compact_threshold,
            thinking=(
                explicit_thinking
                or resolved_model.thinking
                or config.thinking_overrides.get(resolved_model.base_model)
            ),
        )
    except ModelResolutionError as exc:
        field = "config.thinking" if exc.code == "thinking_unsupported" else "model"
        raise InvalidLaunchModelSelectionError(field, str(exc)) from exc
    effective_thinking = model.thinking
    committed_model: CommittedModelIdentity = resolved_model.identity.model_copy(
        update={"thinking": effective_thinking}
    )
    # Step 5: the identity revision (a pure read, through the strict loader).
    resource = _identity_resource_loader(directories).resolve(
        ResourceKind.IDENTITY, validated_identity
    )
    if resource.content is None:
        raise InvalidChildIdentityError(
            "identity",
            f"identity resource {validated_identity!r} resolved without content",
        )
    identity_revision = resource_revision(resource.content)
    # The matched route's overlay body (the same strict-loader read): the
    # overlay rides the persona instructions at launch admission — it never
    # touches the identity binding or its revision (design §9).
    overlay_body: str | None = None
    if overlay is not None:
        try:
            overlay_resource = _overlay_resource_loader().resolve(
                ResourceKind.OVERLAY, overlay
            )
        except ResourceResolutionError as exc:
            raise InvalidLaunchPurposeError("purpose", str(exc)) from exc
        overlay_body = overlay_resource.content
        if overlay_body is None:
            raise InvalidLaunchPurposeError(
                "purpose", f"route overlay {overlay!r} resolved without content"
            )
    # Step 6: authority resolution + assembly.
    authority = ResolvedLaunchAuthority(
        personal_posture=facts.personal_tools if facts is not None else None,
        delegation_posture=facts.delegation_tools if facts is not None else None,
        authorized_tool_names=frozenset(authorized_tool_names),
    )
    return ResolvedIdentityLaunch(
        identity=validated_identity,
        identity_revision=identity_revision,
        identity_facts=facts,
        execution_context=ExecutionContext.BOUNDED_CHILD,
        workflow=None,
        strategy=None,
        model_expression=model_expression,
        committed_model=committed_model,
        effective_thinking=effective_thinking,
        purpose=parsed.purpose,
        seat=seat,
        overlay=overlay,
        overlay_body=overlay_body,
        effective_tier=effective_tier,
        replacement_capability=capability,
        authority=authority,
        semantic_overrides=(
            parsed.config.model_copy(deep=True)
            if parsed.config is not None
            else LaunchConfig()
        ),
        background=parsed.background,
        retain=parsed.retain,
        agent_id=parsed.agent_id,
        replace_run=parsed.replace_run,
        task_summary=parsed.task_summary,
    )
