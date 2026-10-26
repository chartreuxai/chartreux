"""Conversation-owned policy records, independent of saved catalog edits.

Package 10 shipped this module's kept surface: the package-7 frozen policy
bundle becomes the versioned policy storage of a conversation (design §6,
§10). The bundle carries the actual frozen content — identity/workflow/
strategy bodies and universal fragments — so a conversation resumes with
full fidelity after the source resource files are edited or removed, and
every prompt re-render path serves the captured bundle instead of rereading
live resources.

The retired surface (package 16b): the legacy session-policy bind —
``BoundDispatchPolicy``, ``bind_snapshot_policy``, ``bind_policy``, and
``resume_policy`` with its slot/profile membership checks — is gone. The
frozen-bundle surface below is its successor: the persisted bundle is the
conversation's policy record, and a record persisted by an older format
fails resume with "start a new session".
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chartreux.core.dispatch.route_models import (
    AppliedRouteChange,
    PurposeRoute,
    RouteTarget,
)
from chartreux.core.dispatch.routes import SESSION_MODEL

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema
    from chartreux.core.config.harness_files import HarnessFilesManager
    from chartreux.core.policy.bundle import ActivatedPolicy, PolicyBundle
    from chartreux.core.policy.composition import ExecutionContext
    from chartreux.core.session_types import SessionMetadata
    from chartreux.core.skills.manager import SkillManager


class SessionPolicyError(ValueError):
    """A persisted binding cannot safely be resumed."""


# Frozen policy bundle lifecycle (design §6, §10; package 10).


def validate_stored_policy_bundle(payload: Mapping[str, Any]) -> PolicyBundle:
    """Structurally validate a persisted bundle payload (the storage boundary).

    Structural checks only (version, schema, required content, storage caps);
    hash integrity and pin verification happen at ACTIVATION. Raises
    :class:`SessionPolicyError` with the actionable storage message so a
    corrupt record fails at metadata load rather than deep inside a resume.
    """
    from chartreux.core.policy.bundle import BundleError, parse_bundle
    from chartreux.core.policy.manifest import canonical_json_bytes

    try:
        return parse_bundle(canonical_json_bytes(dict(payload)))
    except BundleError as exc:
        raise SessionPolicyError(str(exc)) from exc


def resume_policy_bundle(metadata: SessionMetadata) -> ActivatedPolicy:
    """Activate the conversation's persisted policy bundle (design §10 resume).

    The persisted bundle is the conversation's policy record: resume consumes
    its carried content, never the live resource files (decision log item 6).
    A record persisted by an older format — no bundle, or a mismatched
    ``policy_storage_version`` — is a LEGACY record: the actionable failure
    says to start a new session; there is no migration path. Corrupt or
    tampered bundles reject the same way (integrity is verified at activation,
    before any client or resource is constructed).
    """
    from chartreux.core.policy.bundle import BundleError, load_and_activate
    from chartreux.core.policy.manifest import canonical_json_bytes
    from chartreux.core.session_types import POLICY_STORAGE_VERSION

    payload = metadata.policy_bundle
    if metadata.policy_storage_version != POLICY_STORAGE_VERSION or payload is None:
        raise SessionPolicyError(
            f"Session {metadata.session_id!r} was persisted by an older "
            "harness format that carries no frozen policy bundle, so its "
            "conversation policy cannot be resumed. Start a new session; "
            "there is no migration for the old persisted format."
        )
    try:
        return load_and_activate(canonical_json_bytes(payload))
    except BundleError as exc:
        raise SessionPolicyError(
            f"Session {metadata.session_id!r} has an unreadable persisted "
            f"policy bundle ({exc}); the conversation cannot be resumed. "
            "Start a new session."
        ) from exc


def bound_identity_reference(activated: ActivatedPolicy) -> str:
    """The bound identity reference a conversation's bundle carries.

    The manifest records the selection exactly as the binding validated it
    (``validate_identity_selection`` returns the same reference), so this is
    the identity binding resume re-installs — validation applies to NEW
    bindings only, never to a persisted one.
    """
    return activated.manifest.identity.selected


def bind_policy_bundle(
    config: ChartreuxConfigSchema,
    *,
    execution_context: ExecutionContext,
    skill_manager: SkillManager,
    scratchpad_dir: Path | None,
    cwd: Path,
    harness_files: HarnessFilesManager,
    role_instructions: str | None,
) -> tuple[PolicyBundle, ActivatedPolicy]:
    """Build, verify, and activate the frozen policy bundle for a NEW binding.

    This is the startup and ``/clear`` bind: the current selections AND their
    resource content are captured ONCE (design §3, §6; decision log items 3
    and 6) — the identity selection is validated first, failing closed on an
    unknown reference (validation applies to new bindings only). The returned
    bundle carries the actual content, so the conversation never rereads the
    frozen policy from disk again.
    """
    from chartreux.core.agents.identities import validate_identity_selection
    from chartreux.core.dispatch.lint import lint_composed, reject_errors
    from chartreux.core.policy.bundle import activate_bundle, build_bundle
    from chartreux.core.policy.composition import compose_policy
    from chartreux.core.policy.manifest import PolicySnapshot
    from chartreux.core.system_prompt import _build_composition_inputs

    validate_identity_selection(
        config.agent.identity, user_prompts_dirs=harness_files.user_prompts_dirs
    )
    inputs = _build_composition_inputs(
        config,
        execution_context=execution_context,
        skill_manager=skill_manager,
        scratchpad_dir=scratchpad_dir,
        cwd=cwd,
        harness_files=harness_files,
        tool_manager=None,
        role_instructions=role_instructions,
    )
    result = compose_policy(inputs)
    reject_errors(
        lint_composed(
            result.policy_text,
            identity_facts=result.identity_facts,
            workflow=result.workflow.id,
        )
    )
    snapshot = PolicySnapshot(
        identity_selection=config.agent.identity,
        workflow_selection=config.agent.workflow,
        strategy_selection=config.agent.orchestration_strategy,
        routes=config.purpose_routes.routes,
        inputs=inputs,
        result=result,
    )
    bundle = build_bundle(snapshot)
    return bundle, activate_bundle(bundle)


def pending_policy_changes(
    config: ChartreuxConfigSchema,
    activated: ActivatedPolicy,
    *,
    harness_files: HarnessFilesManager,
) -> tuple[str, ...]:
    """Report — never apply — pending policy changes against the bound bundle.

    A config reload mid-conversation does not re-posture: the frozen
    identity/workflow/strategy selections and their resource content stay
    bound (design §10), and the differences between the CURRENT saved state
    and the frozen manifest are reported as pending changes that apply to NEW
    conversations only. The resource comparison resolves the BOUND selections
    through the strict loader: a live reread for REPORTING only, never for
    prompt refresh.

    Purpose routes are NOT pending policy: routes are bindings, not frozen
    content (ADR 0022) — a validated route edit applies to subsequent launch
    admissions in the same conversation and is reported at the reload commit
    by :func:`live_route_changes` (diffed against the previous live table)
    and at resume by :func:`applied_route_changes` (diffed against the
    frozen bind-time snapshot).
    """
    from chartreux.core.policy.composition import (
        policy_resource_loader,
        user_resource_directories,
    )
    from chartreux.core.policy.manifest import resource_revision
    from chartreux.core.policy.models import ResourceKind

    manifest = activated.manifest
    selection = config.agent
    notes: list[str] = []
    records = {
        "identity": manifest.identity,
        "workflow": manifest.workflow,
        "strategy": manifest.strategy,
    }
    current = {
        "identity": selection.identity,
        "workflow": selection.workflow,
        "strategy": selection.orchestration_strategy,
    }
    for kind in ("identity", "workflow", "strategy"):
        record = records[kind]
        if current[kind] != record.selected:
            notes.append(
                f"pending [agent] {kind} selection change: the conversation "
                f"stays bound to {record.selected!r}; the saved selection "
                f"{current[kind]!r} applies to new conversations only"
            )
    loader = policy_resource_loader(user_resource_directories(harness_files))
    for kind, record in records.items():
        try:
            resolved = loader.resolve(ResourceKind(kind), record.selected)
        except ValueError as exc:
            notes.append(
                f"pending {kind} resource change: the bound resource "
                f"{record.selected!r} no longer resolves on disk ({exc}); "
                "the conversation keeps the frozen content"
            )
            continue
        if resolved.content is None or not record.revision:
            continue
        revision = resource_revision(resolved.content)
        if revision != record.revision:
            notes.append(
                f"pending {kind} resource content change: "
                f"{record.selected!r} was edited on disk (revision "
                f"{record.revision} -> {revision}); the conversation keeps "
                "the frozen content — the edit applies to new conversations "
                "only"
            )
    return tuple(notes)


def _route_target_summary(target: RouteTarget, session_model: str) -> str:
    """One target's compact summary for the applied-route report."""
    model = session_model if target.model == SESSION_MODEL else target.model
    seat = f"{target.seat}: " if target.seat else ""
    return f"{seat}{target.identity} on {model}"


def _route_summary(route: PurposeRoute | None, session_model: str) -> str:
    """A route's compact old/new summary (``—`` when the side is absent)."""
    if route is None:
        return "—"
    return "; ".join(
        _route_target_summary(target, session_model) for target in route.targets
    )


def _route_changes(
    before: Mapping[str, PurposeRoute],
    after: Mapping[str, PurposeRoute],
    session_model: str,
) -> tuple[AppliedRouteChange, ...]:
    """The old → new route report between two route tables, per purpose."""
    changes: list[AppliedRouteChange] = []
    for purpose in sorted(set(before) | set(after)):
        old = before.get(purpose)
        new = after.get(purpose)
        if old == new:
            continue
        changes.append(
            AppliedRouteChange(
                purpose=purpose,
                before=_route_summary(old, session_model),
                after=_route_summary(new, session_model),
            )
        )
    return tuple(changes)


def applied_route_changes(
    config: ChartreuxConfigSchema, activated: ActivatedPolicy
) -> tuple[AppliedRouteChange, ...]:
    """The route edits a RESUME adopts, old → new per purpose (ADR 0022).

    Routes are bindings, not frozen content: a validated route edit applies
    to subsequent launch admissions in the same conversation, while every
    already-admitted launch keeps its pinned binding. A resumed conversation
    adopts the current saved table, and this report states the diff against
    the manifest's bind-time route snapshot — the frozen integrity and diff
    anchor — so the resume notice says exactly what the conversation's
    admission source changed from. An empty tuple means the saved table
    matches the snapshot.
    """
    from chartreux.core.dispatch.renderer import session_model_alias
    from chartreux.core.policy.manifest import routes_from_snapshot

    return _route_changes(
        routes_from_snapshot(activated.manifest.route_snapshot),
        config.purpose_routes.routes,
        session_model_alias(config),
    )


def live_route_changes(
    config: ChartreuxConfigSchema,
    previous_live: Mapping[str, PurposeRoute] | None,
    activated: ActivatedPolicy,
) -> tuple[AppliedRouteChange, ...]:
    """The route edits a RELOAD applies, diffed against the previous LIVE table.

    The reload notice reports what the live admission source changed at THIS
    reload: A -> B -> C reports "A -> B" then "B -> C", a revert reports
    "B -> A", and a reload that leaves the table unchanged reports nothing —
    never the bind-time differences a frozen-snapshot diff would repeat.
    ``previous_live`` is the last-seen live table the conversation tracked;
    None means no reload has committed in this conversation yet, so the diff
    base is the frozen bind-time snapshot: the first reload reports against
    bind time, exactly as the resume notice does. The frozen snapshot stays
    the integrity anchor and the resume notice's diff base; only the reload
    notice's diff base moves to the live table. An empty tuple means the
    reload left the live route table unchanged.
    """
    from chartreux.core.dispatch.renderer import session_model_alias
    from chartreux.core.policy.manifest import routes_from_snapshot

    if previous_live is None:
        previous_live = routes_from_snapshot(activated.manifest.route_snapshot)
    return _route_changes(
        previous_live, config.purpose_routes.routes, session_model_alias(config)
    )
