"""Versioned, self-contained frozen policy bundle codec (design §6, §10).

Bundle format (version 1, canonical key-sorted JSON)::

    {
      "version": 1,
      "manifest": { ...CompositionManifest... },
      "frozen": {
        "identity_body": <str>, "workflow_body": <str>,
        "strategy_body": <str|null>,
        "universal_fragment": <str>, "context_fragment": <str>,
        "tool_semantics_fragment": <str>, "communication_fragment": <str>,
        "instruction_overlay": <str|null>,
        "regions": {<str>: <str>}, "launch_targets": <str>,
        "current_date": <str>
      }
    }

The bundle carries the ACTUAL CONTENT of the composed policy bodies and
universal fragments — content, not hashes alone — so a conversation
resumes with full fidelity after the source resource files are edited or
removed (design §6; decision log item 6). Live sections (catalogs,
AGENTS.md, environment, scratchpad, git context) are NOT carried: they
refresh at serve time (design §10), and the manifest records their hash
for change reporting.

Activation (:func:`activate_bundle`) is pure verification over the
carried content and runs BEFORE any client, resource, or renderer is
constructed: the frozen-policy hash is recomputed from the carried
bodies, the composed policy is recomposed and re-hashed against
``manifest.policy_hash``, the executable route snapshot is cross-checked
against the advertised launch-target table (both capture the same route
source at bind time), and an optional expected pin
(``expected_frozen_hash``) must agree. Corruption, tampering, and pin
mismatches reject activation with actionable errors.

:class:`BundleStore` is the bounded inline store with the eviction
guard: it never evicts the sole persisted bundle of a conversation that
remains resumable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from chartreux.core.dispatch.renderer import render_launch_target_region
from chartreux.core.policy.composition import (
    CompositionInputs,
    CompositionResult,
    ExecutionContext,
    compose_policy,
    resolve_identity_facts,
)
from chartreux.core.policy.manifest import (
    MAX_BUNDLE_BYTES,
    MAX_FROZEN_INLINE_BYTES,
    MAX_RESOURCE_BYTES,
    CompositionManifest,
    PolicySizeLimitError,
    PolicySnapshot,
    build_manifest,
    canonical_json_bytes,
    content_hash,
    decode_json,
    format_validation_error,
    frozen_hash_parts_from_content,
    frozen_policy_hash,
    routes_from_snapshot,
)
from chartreux.core.policy.models import ResolvedResource, ResourceKind

if TYPE_CHECKING:
    from chartreux.core.config.harness_files._harness_manager import InstructionDocument

__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "ActivatedPolicy",
    "BundleError",
    "BundleIntegrityError",
    "BundleStore",
    "BundleStoreError",
    "BundleTooLargeError",
    "FrozenPolicyContent",
    "LiveSections",
    "PolicyBundle",
    "PolicyPinMismatchError",
    "activate_bundle",
    "build_bundle",
    "frozen_inline_sizes",
    "load_and_activate",
    "parse_bundle",
    "serialize_bundle",
]

BUNDLE_FORMAT_VERSION = 1


class BundleError(ValueError):
    """The bundle codec failed with an actionable error."""


class BundleIntegrityError(BundleError):
    """The stored hashes disagree with the content the bundle carries."""


class PolicyPinMismatchError(BundleError):
    """An expected frozen-policy pin disagrees with the bundle."""


class BundleTooLargeError(BundleError, PolicySizeLimitError):
    """A bundle storage cap (interfaces.md) is exceeded."""


class BundleStoreError(BundleError):
    """The bundle store refused an operation (the eviction guard)."""


class FrozenPolicyContent(BaseModel):
    """The frozen inline content: the actual policy bodies and fragments.

    Content — not hashes alone — is what makes resume possible after the
    source resource files are edited or removed (design §6).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity_body: str
    workflow_body: str
    strategy_body: str | None = None
    universal_fragment: str
    context_fragment: str
    tool_semantics_fragment: str
    communication_fragment: str
    instruction_overlay: str | None = None
    regions: dict[str, str]
    launch_targets: str
    current_date: str


class PolicyBundle(BaseModel):
    """The immutable, versioned, self-contained policy bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    manifest: CompositionManifest
    frozen: FrozenPolicyContent


@dataclass(frozen=True)
class LiveSections:
    """The refreshable live sections supplied at serve time (design §10)."""

    model_info: str | None = None
    skills: str | None = None
    scratchpad: str | None = None
    environment: str | None = None
    project_context: str | None = None
    commit_signature: str | None = None
    agents_documents: tuple[InstructionDocument, ...] = ()


@dataclass(frozen=True)
class ActivatedPolicy:
    """A verified, activated policy bundle, ready to serve (design §10).

    ``policy_text`` is the harness-owned composition recomposed
    byte-identically from the bundle's carried content; :meth:`render`
    re-renders the served artifact with refreshed live sections.
    """

    manifest: CompositionManifest
    frozen: FrozenPolicyContent
    policy_text: str

    @property
    def frozen_policy_hash(self) -> str:
        """The verified (recomputed-equal) frozen-policy hash."""
        return self.manifest.frozen_policy_hash

    def render(self, live: LiveSections | None = None) -> CompositionResult:
        """Re-render the served artifact with refreshed live sections."""
        return _recompose(
            self.manifest, self.frozen, live if live is not None else LiveSections()
        )


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value is not None else None


def _recompose(
    manifest: CompositionManifest, frozen: FrozenPolicyContent, live: LiveSections
) -> CompositionResult:
    """Recompose the policy from the bundle's carried content alone."""
    identity = ResolvedResource(
        ResourceKind(manifest.identity.kind),
        manifest.identity.resolved_id,
        manifest.identity.source,
        frozen.identity_body,
        _optional_path(manifest.identity.path),
    )
    workflow = ResolvedResource(
        ResourceKind(manifest.workflow.kind),
        manifest.workflow.resolved_id,
        manifest.workflow.source,
        frozen.workflow_body,
        _optional_path(manifest.workflow.path),
    )
    strategy = ResolvedResource(
        ResourceKind(manifest.strategy.kind),
        manifest.strategy.resolved_id,
        manifest.strategy.source,
        frozen.strategy_body,
        _optional_path(manifest.strategy.path),
    )
    inputs = CompositionInputs(
        identity=identity,
        workflow=workflow,
        strategy=strategy,
        identity_facts=resolve_identity_facts(manifest.identity.selected),
        execution_context=ExecutionContext(manifest.execution_context),
        universal=frozen.universal_fragment,
        context_fragment=frozen.context_fragment,
        tool_semantics=frozen.tool_semantics_fragment,
        communication=frozen.communication_fragment,
        launch_targets=frozen.launch_targets,
        regions=dict(frozen.regions),
        model_info=live.model_info,
        skills=live.skills,
        scratchpad=live.scratchpad,
        environment=live.environment,
        project_context=live.project_context,
        commit_signature=live.commit_signature,
        instruction_overlay=frozen.instruction_overlay,
        agents_documents=live.agents_documents,
        current_date=frozen.current_date,
    )
    return compose_policy(inputs)


_REQUIRED_FROZEN_FIELDS = (
    "identity_body",
    "workflow_body",
    "universal_fragment",
    "context_fragment",
    "tool_semantics_fragment",
    "communication_fragment",
    "launch_targets",
)


def _check_required_content(bundle: PolicyBundle) -> None:
    frozen = bundle.frozen
    missing = [
        f"frozen.{name}"
        for name in _REQUIRED_FROZEN_FIELDS
        if not getattr(frozen, name).strip()
    ]
    if missing:
        raise BundleError(
            "bundle is missing required content: "
            + ", ".join(missing)
            + "; the frozen policy bodies cannot be empty"
        )


def _check_resource_caps(
    manifest: CompositionManifest, frozen: FrozenPolicyContent
) -> None:
    bodies = (
        ("identity", manifest.identity.resolved_id, frozen.identity_body),
        ("workflow", manifest.workflow.resolved_id, frozen.workflow_body),
        ("strategy", manifest.strategy.resolved_id, frozen.strategy_body),
        ("universal fragment", "universal", frozen.universal_fragment),
        ("context fragment", "context", frozen.context_fragment),
        ("tool-semantics fragment", "tool_semantics", frozen.tool_semantics_fragment),
        ("communication fragment", "communication", frozen.communication_fragment),
    )
    for what, offender, body in bodies:
        if body is None:
            continue
        size = len(body.encode("utf-8"))
        if size > MAX_RESOURCE_BYTES:
            raise BundleTooLargeError(
                f"the {what} {offender!r} carried by the bundle is {size} "
                f"bytes, exceeding the per-resource storage cap of "
                f"{MAX_RESOURCE_BYTES} bytes (64 KiB)"
            )


def frozen_inline_sizes(frozen: FrozenPolicyContent) -> dict[str, int]:
    """The encoded UTF-8 size of each carried frozen-content field."""

    def size_of(text: str | None) -> int:
        return len(text.encode("utf-8")) if text is not None else 0

    return {
        "identity_body": size_of(frozen.identity_body),
        "workflow_body": size_of(frozen.workflow_body),
        "strategy_body": size_of(frozen.strategy_body),
        "universal_fragment": size_of(frozen.universal_fragment),
        "context_fragment": size_of(frozen.context_fragment),
        "tool_semantics_fragment": size_of(frozen.tool_semantics_fragment),
        "communication_fragment": size_of(frozen.communication_fragment),
        "instruction_overlay": size_of(frozen.instruction_overlay),
        "regions": len(canonical_json_bytes(frozen.regions)),
        "launch_targets": size_of(frozen.launch_targets),
        "current_date": size_of(frozen.current_date),
    }


def _check_frozen_inline_cap(frozen: FrozenPolicyContent) -> None:
    sizes = frozen_inline_sizes(frozen)
    total = sum(sizes.values())
    if total > MAX_FROZEN_INLINE_BYTES:
        offenders = sorted(sizes.items(), key=lambda item: item[1], reverse=True)[:3]
        raise BundleTooLargeError(
            f"frozen inline bundle content is {total} bytes, exceeding the "
            f"{MAX_FROZEN_INLINE_BYTES}-byte (192 KiB) storage cap; largest "
            "fields: " + ", ".join(f"{name} ({size} bytes)" for name, size in offenders)
        )


def _check_aggregate_cap(payload: bytes) -> None:
    if len(payload) > MAX_BUNDLE_BYTES:
        raise BundleTooLargeError(
            f"serialized policy bundle is {len(payload)} bytes, exceeding "
            f"the aggregate storage cap of {MAX_BUNDLE_BYTES} bytes "
            f"(256 KiB); reduce the frozen policy content or the route "
            f"snapshot"
        )


def build_bundle(snapshot: PolicySnapshot) -> PolicyBundle:
    """Build the frozen policy bundle for one composition.

    Enforces the package-0 storage caps before activation
    (interfaces.md "Bundle storage caps") and verifies that the composed
    policy reproduces byte-identically from the captured content.
    """
    try:
        manifest = build_manifest(snapshot)
    except PolicySizeLimitError as exc:
        raise BundleTooLargeError(str(exc)) from exc
    inputs = snapshot.inputs
    frozen = FrozenPolicyContent(
        identity_body=inputs.identity.content or "",
        workflow_body=inputs.workflow.content or "",
        strategy_body=inputs.strategy.content,
        universal_fragment=inputs.universal,
        context_fragment=inputs.context_fragment,
        tool_semantics_fragment=inputs.tool_semantics,
        communication_fragment=inputs.communication,
        instruction_overlay=inputs.instruction_overlay,
        regions=dict(inputs.regions),
        launch_targets=inputs.launch_targets,
        current_date=inputs.current_date,
    )
    bundle = PolicyBundle(manifest=manifest, frozen=frozen)
    _check_required_content(bundle)
    _check_resource_caps(manifest, frozen)
    _check_frozen_inline_cap(frozen)
    _check_aggregate_cap(serialize_bundle(bundle))
    # Self-verification: the captured content must recompose the exact
    # policy the manifest hashed (content sufficiency, proven at build).
    recomposed = _recompose(manifest, frozen, LiveSections())
    if content_hash(recomposed.policy_text) != manifest.policy_hash:
        raise BundleError(
            "the captured frozen content does not reproduce the composed "
            "policy the manifest hashed; the composition inputs and result "
            "are incoherent"
        )
    return bundle


def serialize_bundle(bundle: PolicyBundle) -> bytes:
    """Serialize the bundle to canonical, byte-stable JSON."""
    return canonical_json_bytes(bundle.model_dump(mode="json"))


def parse_bundle(payload: bytes | str) -> PolicyBundle:
    """Decode and structurally validate a bundle payload.

    Structural checks only: version, schema, required content, and the
    storage caps. Hash integrity and pin verification happen at ACTIVATION
    (:func:`activate_bundle`), before any client or resource is
    constructed.
    """
    try:
        data = decode_json(payload, what="bundle payload")
    except ValueError as exc:
        raise BundleError(str(exc)) from exc
    if not isinstance(data, dict):
        raise BundleError(
            "bundle payload is corrupt: the top-level value is not a JSON object"
        )
    version = data.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != BUNDLE_FORMAT_VERSION
    ):
        raise BundleError(
            f"unsupported bundle format version {version!r}: this harness "
            f"reads version {BUNDLE_FORMAT_VERSION} policy bundles; start a "
            f"new conversation or upgrade the harness"
        )
    try:
        bundle = PolicyBundle.model_validate(data)
    except ValidationError as exc:
        raise BundleError(
            "bundle payload is corrupt: " + format_validation_error(exc)
        ) from exc
    _check_required_content(bundle)
    _check_resource_caps(bundle.manifest, bundle.frozen)
    _check_frozen_inline_cap(bundle.frozen)
    _check_aggregate_cap(serialize_bundle(bundle))
    return bundle


# One rendered launch-target row: | `purpose` | `identity` | tier | seat | `model` |.
_LAUNCH_TARGET_ROW = re.compile(
    r"^\| `(?P<purpose>[^`]+)` \| `(?P<identity>[^`]+)` \| "
    r"(?P<tier>\S+) \| (?P<seat>.+?) \| `(?P<model>[^`]+)` \|$"
)
# One rendered route-requirement line: - `purpose`: N distinct canonical models …
_ROUTE_REQUIREMENT = re.compile(
    r"^- `(?P<purpose>[^`]+)`: \d+ distinct canonical models"
)


def _row_binding(row: str) -> tuple[str, str, str]:
    """The (identity, tier, seat) binding a rendered launch-target row carries."""
    match = _LAUNCH_TARGET_ROW.match(row)
    assert match is not None
    return (match.group("identity"), match.group("tier"), match.group("seat"))


def _line_purpose(line: str) -> str | None:
    """The purpose a rendered launch-target or requirement line names."""
    match = _LAUNCH_TARGET_ROW.match(line)
    if match is not None:
        return match.group("purpose")
    requirement = _ROUTE_REQUIREMENT.match(line)
    return requirement.group("purpose") if requirement is not None else None


def _route_snapshot_mismatch_message(
    manifest: CompositionManifest,
    advertised_lines: list[str],
    rendered_lines: list[str],
) -> str:
    """The typed failure naming the purpose and both sides of the mismatch."""
    advertised_rows: dict[str, list[str]] = {}
    for line in advertised_lines:
        match = _LAUNCH_TARGET_ROW.match(line)
        if match is not None:
            advertised_rows.setdefault(match.group("purpose"), []).append(line)
    rendered_rows: dict[str, list[str]] = {}
    for line in rendered_lines:
        match = _LAUNCH_TARGET_ROW.match(line)
        if match is not None:
            rendered_rows.setdefault(match.group("purpose"), []).append(line)
    for entry in manifest.route_snapshot:
        table_rows = advertised_rows.get(entry.purpose, [])
        snapshot_rows = rendered_rows.get(entry.purpose, [])
        if len(table_rows) != len(snapshot_rows) or sorted(
            _row_binding(row) for row in table_rows
        ) != sorted(_row_binding(row) for row in snapshot_rows):
            return (
                f"the persisted route snapshot disagrees with the bundle's "
                f"advertised launch-target table for purpose "
                f"{entry.purpose!r} (design §6: both capture the same route "
                f"source at bind time) — the executable snapshot row is "
                f"{snapshot_rows[0] if snapshot_rows else '(none)'} while "
                f"the advertised table row is "
                f"{table_rows[0] if table_rows else '(none)'}; the bundle is "
                f"corrupt or was tampered with — refusing to activate"
            )
    # The data rows agree; the disagreement is in a model column or a
    # route-requirement line — quote the first differing line pair.
    only_snapshot = next(
        (line for line in rendered_lines if line not in advertised_lines), "(none)"
    )
    only_table = next(
        (line for line in advertised_lines if line not in rendered_lines), "(none)"
    )
    purpose = _line_purpose(only_snapshot) or _line_purpose(only_table)
    purpose_text = f" for purpose {purpose!r}" if purpose is not None else ""
    return (
        f"the persisted route snapshot disagrees with the bundle's advertised "
        f"launch-target table{purpose_text} (design §6: both capture the same "
        f"route source at bind time) — the executable snapshot says "
        f"{only_snapshot} while the advertised table says {only_table}; the "
        f"bundle is corrupt or was tampered with — refusing to activate"
    )


def _check_route_snapshot_consistency(
    manifest: CompositionManifest, frozen: FrozenPolicyContent
) -> None:
    """Cross-check the executable route snapshot against the advertised table.

    The manifest's route snapshot and the frozen launch-target table are
    both captured from the same route source at bind time (design §6): the
    snapshot is the table launch admission EXECUTES, the frozen string is
    the table the conversation ADVERTISES. A persisted bundle whose
    snapshot disagrees with its own advertised table re-creates the
    half-applied generation from disk — admission would launch one target
    while the model is told another — so activation refuses it loudly,
    naming the purpose and both sides, instead of silently picking either.

    A bundle with NO captured snapshot (the pre-snapshot persisted format)
    skips the cross-check: an empty snapshot already fails closed at
    launch admission with a loud RouteResolutionError.
    """
    if not manifest.route_snapshot:
        return
    routes = routes_from_snapshot(manifest.route_snapshot)
    advertised_lines = frozen.launch_targets.splitlines()
    # The rendered table substitutes the bind-time session model into every
    # ``$session`` target; that value appears in the advertised model
    # column, so re-rendering under each candidate reproduces the exact
    # table a consistent snapshot produces. The row ORDER differs (the
    # snapshot is sorted by purpose, the render followed the configured
    # table's order), so the comparison is over sorted line multisets.
    candidates: set[str] = set()
    for line in advertised_lines:
        match = _LAUNCH_TARGET_ROW.match(line)
        if match is not None:
            candidates.add(match.group("model"))
    best: tuple[int, list[str]] | None = None
    for session_model in sorted(candidates or {""}):
        rendered_lines = render_launch_target_region(routes, session_model).splitlines()
        if sorted(rendered_lines) == sorted(advertised_lines):
            return
        difference = len(set(rendered_lines) ^ set(advertised_lines))
        if best is None or difference < best[0]:
            best = (difference, rendered_lines)
    assert best is not None
    raise BundleIntegrityError(
        _route_snapshot_mismatch_message(manifest, advertised_lines, best[1])
    )


def activate_bundle(
    bundle: PolicyBundle, *, expected_frozen_hash: str | None = None
) -> ActivatedPolicy:
    """Activate a parsed bundle, verifying integrity and pins FIRST.

    The verification is pure computation over the carried content — it
    runs BEFORE any client, resource, or renderer is constructed. A stored
    hash that disagrees with the recomputed reality (corruption or
    tampering) rejects activation, and so does an expected-pin mismatch.
    """
    manifest = bundle.manifest
    frozen = bundle.frozen
    recomputed_frozen = frozen_policy_hash(
        frozen_hash_parts_from_content(
            identity_body=frozen.identity_body,
            workflow_body=frozen.workflow_body,
            strategy_body=frozen.strategy_body,
            universal_fragment=frozen.universal_fragment,
            context_fragment=frozen.context_fragment,
            tool_semantics_fragment=frozen.tool_semantics_fragment,
            communication_fragment=frozen.communication_fragment,
        )
    )
    if recomputed_frozen != manifest.frozen_policy_hash:
        raise BundleIntegrityError(
            f"the stored frozen-policy hash {manifest.frozen_policy_hash} "
            f"disagrees with the content the bundle carries (recomputed "
            f"{recomputed_frozen}); the bundle is corrupt or was tampered "
            f"with — refusing to activate"
        )
    recomposed = _recompose(manifest, frozen, LiveSections())
    policy_hash = content_hash(recomposed.policy_text)
    if policy_hash != manifest.policy_hash:
        raise BundleIntegrityError(
            f"the stored composed-policy hash {manifest.policy_hash} "
            f"disagrees with the policy recomposed from the carried content "
            f"(recomputed {policy_hash}); the bundle is corrupt or was "
            f"tampered with — refusing to activate"
        )
    _check_route_snapshot_consistency(manifest, frozen)
    if (
        expected_frozen_hash is not None
        and expected_frozen_hash != manifest.frozen_policy_hash
    ):
        raise PolicyPinMismatchError(
            f"the expected frozen-policy pin {expected_frozen_hash} "
            f"disagrees with the bundle's verified frozen-policy hash "
            f"{manifest.frozen_policy_hash} (identity "
            f"{manifest.identity.resolved_id!r}, revision "
            f"{manifest.identity_revision}); the persisted pin does not "
            f"match the bundle content — refusing to activate the "
            f"conversation"
        )
    return ActivatedPolicy(
        manifest=manifest, frozen=frozen, policy_text=recomposed.policy_text
    )


def load_and_activate(
    payload: bytes | str, *, expected_frozen_hash: str | None = None
) -> ActivatedPolicy:
    """Parse and activate a serialized bundle in one call."""
    return activate_bundle(
        parse_bundle(payload), expected_frozen_hash=expected_frozen_hash
    )


class BundleStore:
    """The bounded inline bundle store with the no-eviction guard.

    The store keeps serialized bundles keyed by conversation id; package 10
    wires it to the existing conversation persistence (interfaces.md: the
    bundle content is stored inline, bounded by the package-0 caps). The
    INVARIANT (design §6): the store never evicts the sole persisted
    bundle of a conversation that remains resumable — the bundle carries
    the actual frozen content, so evicting it would make resume
    impossible.
    """

    def __init__(self, *, max_entries: int = 64) -> None:
        self._payloads: dict[str, bytes] = {}
        self._resumable: set[str] = set()
        self._max_entries = max_entries

    def __len__(self) -> int:
        return len(self._payloads)

    def __contains__(self, conversation_id: str) -> bool:
        return conversation_id in self._payloads

    def set_resumable(self, conversation_id: str, resumable: bool) -> None:
        """Record whether a conversation remains resumable."""
        if resumable:
            self._resumable.add(conversation_id)
        else:
            self._resumable.discard(conversation_id)

    def _has_room(self, conversation_id: str) -> bool:
        """Whether ``put`` can persist ``conversation_id`` without refusing."""
        return (
            conversation_id in self._payloads
            or len(self._payloads) < self._max_entries
            or any(persisted not in self._resumable for persisted in self._payloads)
        )

    def _capacity_refusal(self) -> BundleStoreError:
        """The loud, typed exhaustion error (shared by ``put`` and the check)."""
        return BundleStoreError(
            f"the bundle store is full ({self._max_entries} entries) and "
            f"every persisted bundle belongs to a resumable conversation; "
            f"the eviction guard refuses to evict any of them"
        )

    def ensure_capacity(self, conversation_id: str) -> None:
        """Settle capacity for ``conversation_id`` before any durable change.

        Session-transition preparation calls this ahead of the loop's
        synchronous commit: a store that cannot hold the conversation's
        bundle refuses HERE — the reset or resume fails with the prior
        generation fully intact — instead of ``put`` throwing after the
        session id and generation already moved (design §6: the store
        never holds entries from mixed generations). The refusal is the
        same loud, typed error ``put`` raises, and it never deletes the
        sole durable bundle of a resumable conversation to make room.
        """
        if not self._has_room(conversation_id):
            raise self._capacity_refusal()

    def put(self, conversation_id: str, bundle: PolicyBundle) -> None:
        """Persist one conversation's bundle (caps enforced at build)."""
        payload = serialize_bundle(bundle)
        _check_aggregate_cap(payload)
        if not self._has_room(conversation_id):
            raise self._capacity_refusal()
        if (
            conversation_id not in self._payloads
            and len(self._payloads) >= self._max_entries
        ):
            self._evict_for_room()
        self._payloads[conversation_id] = payload

    def get(self, conversation_id: str) -> PolicyBundle | None:
        """Return the persisted bundle, or None when absent."""
        payload = self._payloads.get(conversation_id)
        return None if payload is None else parse_bundle(payload)

    def activate(
        self, conversation_id: str, *, expected_frozen_hash: str | None = None
    ) -> ActivatedPolicy:
        """Activate a conversation's persisted bundle (integrity + pin)."""
        bundle = self.get(conversation_id)
        if bundle is None:
            raise BundleStoreError(
                f"no persisted policy bundle for conversation "
                f"{conversation_id!r}; the conversation cannot be resumed"
            )
        return activate_bundle(bundle, expected_frozen_hash=expected_frozen_hash)

    def evict(self, conversation_id: str, *, conversation_resumable: bool) -> None:
        """Evict one conversation's bundle — refused while it is resumable.

        The eviction guard: the sole persisted bundle of a conversation
        that remains resumable is never evicted (content, not hashes
        alone, is what makes resume possible).
        """
        if conversation_resumable:
            raise BundleStoreError(
                f"refusing to evict the sole persisted bundle of "
                f"conversation {conversation_id!r}: the conversation remains "
                f"resumable and the bundle carries its frozen policy content"
            )
        self._payloads.pop(conversation_id, None)
        self._resumable.discard(conversation_id)

    def _evict_for_room(self) -> None:
        for conversation_id in list(self._payloads):
            if conversation_id not in self._resumable:
                self._payloads.pop(conversation_id)
                return
        raise self._capacity_refusal()
