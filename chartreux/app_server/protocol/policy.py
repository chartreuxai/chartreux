"""Roots, dangerous-command authority, and tool policy wire models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.protocol.runtime import RuntimeSnapshot


class RootsReadParams(ProtocolModel):
    session_id: str


class RootsReadResponse(ProtocolModel):
    revision: str
    project: str
    roots: list[str]
    saved_roots: list[str] | None = None
    user_revision: str | None = None
    effective_roots: list[str] = Field(default_factory=list)


class RootsReplaceParams(ProtocolModel):
    session_id: str
    expected_revision: str
    scope: Literal["session"] = "session"
    user_initiated: Literal[True]
    roots: list[str]


class RootsReplaceResponse(ProtocolModel):
    revision: str
    runtime: RuntimeSnapshot


class RootsSaveParams(ProtocolModel):
    session_id: str
    root: str
    expected_revision: str
    user_initiated: Literal[True]


class RootsSaveResponse(ProtocolModel):
    """Honest persistence outcome of one project root grant save.

    Mirrors the config save result: a stale expected revision is a conflict
    result, never an RPC error, so callers can keep an already-applied session
    grant and report the outcome.
    """

    target: Literal["user", "project"] = "user"
    persistence: Literal["not_saved", "saved", "durability_uncertain"]
    application: Literal["unchanged", "applied", "failed"]
    revision: str | None = None
    error: (
        Literal[
            "validation",
            "conflict",
            "write",
            "application",
            "cancelled",
            "notification",
        ]
        | None
    ) = None


class DangerousCommandGrantScopeWire(ProtocolModel):
    """Repository/worktree identity one dangerous-command grant is scoped to."""

    git_dir: str
    work_tree: str | None = None


class DangerousCommandGrantWire(ProtocolModel):
    """One dangerous-command authority entry: an operation in one scope."""

    operation_id: str
    scope: DangerousCommandGrantScopeWire


class DangerousCommandDenylistBlockWire(ProtocolModel):
    """An explicit shell denylist entry vetoing dangerous git operations.

    The denylist is an absolute veto checked before grantability: no live or
    saved grant overrides a blocking entry, so inspection reports it apart
    from grant authority instead of suggesting a grant can lift it.
    """

    pattern: str
    operations: list[str]


class DangerousCommandAuthorityReadParams(ProtocolModel):
    session_id: str


class DangerousCommandAuthorityReadResponse(ProtocolModel):
    """Typed inspection of one registered session's dangerous-command authority.

    ``live_grants`` is the runtime-owned authority of the session's own store
    (session grants and any pending once permits). ``accepted_saved_grants``
    is the saved user-config authority the session's accepted restrictions
    contribute for the registered project. ``saved_operations`` and
    ``disk_revision`` are the fresh disk read (``None`` when unavailable), and
    ``disk_revision`` is the expected revision for a saved removal.
    ``live_authority_generation`` is the expected live revision for a
    revocation. No project key is accepted: the project is always the
    registered session's own working directory.
    """

    project: str
    live_authority_generation: int
    live_grants: list[DangerousCommandGrantWire]
    accepted_saved_grants: list[DangerousCommandGrantWire]
    saved_operations: list[str] | None = None
    disk_revision: str | None = None
    blocking_denylist: list[DangerousCommandDenylistBlockWire]


class DangerousCommandRevokeParams(ProtocolModel):
    session_id: str
    operation_id: str
    expected_live_generation: int = Field(ge=0)
    expected_disk_revision: str | None = None
    user_initiated: Literal[True]

    @field_validator("user_initiated", mode="before")
    @classmethod
    def validate_user_initiated(cls, value: Any) -> Any:
        """Only the literal true passes; lax coercion of truthy values is rejected."""
        if value is not True:
            raise ValueError("user_initiated must be the literal true")
        return value


class DangerousCommandRevocationOutcome(ProtocolModel):
    """One side of a revocation, reported separately from the other side.

    ``removed``: the authority is gone (live immediately; saved on disk for
    future sessions); reported even when no such authority existed, so the
    removal is idempotent. ``conflict``: a stale expected revision rejected
    the removal; the live side carries the current ``generation``, while the
    saved side carries no current disk revision, so the client re-reads to
    learn it. ``skipped``: the side was not attempted (a stale live
    generation leaves the saved side untouched, or no disk revision was
    supplied). ``not_applicable``: the operation is never saved, so no saved
    authority can exist. ``failed``: the removal was attempted and did not
    succeed.
    """

    status: Literal["removed", "conflict", "skipped", "not_applicable", "failed"]
    revision: str | None = None
    generation: int | None = None


class DangerousCommandRevokeResponse(ProtocolModel):
    """Separate live and saved outcomes of one user-initiated revocation.

    Live session authority is removed through the session's own grant store
    and takes effect immediately: installed pending once permits are
    invalidated and cached authority is dropped, but a question already
    awaiting an answer is not cancelled, so answering it after the
    revocation still grants fresh authority for the decided scope. Saved
    authority is removed from the user config on disk with the expected disk
    revision; the running session's accepted config is unchanged until a
    reload, exactly as a save never applies to the running session.
    ``live_authority_generation`` is the session's generation after the
    attempt. ``warning`` reports that the session keeps authority the
    accepted config still grants until the next reload or start, or that a
    failed saved removal can let the authority return on the next reload
    or start.
    """

    live: DangerousCommandRevocationOutcome
    saved: DangerousCommandRevocationOutcome
    live_authority_generation: int
    warning: str | None = None


class PolicyToolReplacement(ProtocolModel):
    permission: Literal["always", "never"] = "always"
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
