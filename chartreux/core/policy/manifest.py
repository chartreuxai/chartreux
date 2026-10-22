"""Deterministic composition manifest (design §6).

The manifest is the persisted, byte-stable record of one composition: the
selected and resolved identity/workflow/strategy IDs, their sources, the
content revisions (identity revision, universal-fragment revision), the
route snapshot, the AGENTS provenance with revisions, and the policy
hashes. The same inputs always produce a byte-identical manifest.

Two hashes separate the frozen policy from the refreshable sections:

- the **frozen-policy hash** (pinned) covers EXACTLY the policy bodies
  (identity, workflow, strategy) and the universal fragments (universal,
  context, tool semantics, communication) — the content frozen per
  conversation (design §3, §10; decision log items 3 and 6);
- the **live-fragments hash** (recorded, never pinned) covers the
  refreshable sections (design §10): the rendered launch-target table,
  the model and skills catalogs, the environment block, the commit
  signature, the per-launch instruction overlay, and the AGENTS.md
  layers with their provenance.

Volatile session details — the date stamp, the scratchpad location, and
the git/project context — are in NEITHER hash. A third hash,
``policy_hash``, pins the exact served harness-owned policy text
(``policy_text``, date stamp included); bundle activation recomputes it
from the carried content to prove content sufficiency.

The storage caps are the package-0 settled decision
(``plan/interfaces.md`` "Bundle storage caps"): per-resource 64 KiB,
frozen inline bundle content 192 KiB, aggregate bundle 256 KiB — counted
in encoded UTF-8 bytes and rejected before activation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from chartreux.core.dispatch.route_models import (
    PackageTier,
    PurposeRoute,
    ReplacementCapability,
    RouteTarget,
)
from chartreux.core.policy.composition import CompositionInputs, CompositionResult
from chartreux.core.policy.models import ResolvedResource
from chartreux.utils.io import content_revision

__all__ = [
    "MANIFEST_FORMAT_VERSION",
    "MAX_BUNDLE_BYTES",
    "MAX_FROZEN_INLINE_BYTES",
    "MAX_RESOURCE_BYTES",
    "AgentsLayerRecord",
    "CompositionManifest",
    "ManifestError",
    "PolicySizeLimitError",
    "PolicySnapshot",
    "ResourceRecord",
    "RouteSnapshotEntry",
    "RouteTargetRecord",
    "build_manifest",
    "canonical_json_bytes",
    "content_hash",
    "decode_json",
    "format_validation_error",
    "frozen_hash_parts",
    "frozen_hash_parts_from_content",
    "frozen_policy_hash",
    "live_fragments_hash",
    "live_hash_parts",
    "parse_manifest",
    "resource_revision",
    "routes_from_snapshot",
    "serialize_manifest",
]

MANIFEST_FORMAT_VERSION = 1

#: Per-resource storage cap (interfaces.md "Bundle storage caps").
MAX_RESOURCE_BYTES = 64 * 1024
#: Frozen inline bundle content cap (interfaces.md "Bundle storage caps").
MAX_FROZEN_INLINE_BYTES = 192 * 1024
#: Aggregate serialized bundle cap (interfaces.md "Bundle storage caps").
MAX_BUNDLE_BYTES = 256 * 1024

_HASH_PREFIX = "sha256:"
_FROZEN_HASH_DOMAIN = "chartreux/frozen-policy/v1"
_LIVE_HASH_DOMAIN = "chartreux/live-fragments/v1"


class ManifestError(ValueError):
    """The manifest cannot be built, serialized, or parsed."""


class PolicySizeLimitError(ValueError):
    """A bundle storage cap (interfaces.md) is exceeded."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` to canonical, deterministic JSON bytes."""
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def decode_json(payload: bytes | str, *, what: str = "payload") -> Any:
    """Decode JSON bytes/text, raising ValueError on a corrupt payload."""
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{what} is not valid JSON (corrupt): {exc}") from exc


def format_validation_error(exc: ValidationError) -> str:
    """Format a pydantic ValidationError as ``field: message`` details."""
    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        details.append(f"{location}: {error['msg']}")
    return "; ".join(details)


def content_hash(text: str) -> str:
    """The sha256 content hash of a UTF-8 text, prefixed ``sha256:``."""
    return _HASH_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_payload(payload: Any) -> str:
    return _HASH_PREFIX + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def resource_revision(content: str) -> str:
    """The content revision (``blake2b-128:<hex>``) of one resource body."""
    return content_revision(content.encode("utf-8"))


def frozen_hash_parts_from_content(
    *,
    identity_body: str,
    workflow_body: str,
    strategy_body: str | None,
    universal_fragment: str,
    context_fragment: str,
    tool_semantics_fragment: str,
    communication_fragment: str,
) -> dict[str, str | None]:
    """The exact frozen-hash inputs: the policy bodies + universal fragments."""
    return {
        "identity_body": identity_body,
        "workflow_body": workflow_body,
        "strategy_body": strategy_body,
        "universal_fragment": universal_fragment,
        "context_fragment": context_fragment,
        "tool_semantics_fragment": tool_semantics_fragment,
        "communication_fragment": communication_fragment,
    }


def frozen_hash_parts(inputs: CompositionInputs) -> dict[str, str | None]:
    """The frozen-hash inputs of a composition (bodies + fragments only)."""
    return frozen_hash_parts_from_content(
        identity_body=inputs.identity.content or "",
        workflow_body=inputs.workflow.content or "",
        strategy_body=inputs.strategy.content,
        universal_fragment=inputs.universal,
        context_fragment=inputs.context_fragment,
        tool_semantics_fragment=inputs.tool_semantics,
        communication_fragment=inputs.communication,
    )


def frozen_policy_hash(parts: Mapping[str, str | None]) -> str:
    """The pinned frozen-policy hash over the policy bodies + fragments."""
    return _hash_payload({_FROZEN_HASH_DOMAIN: dict(parts)})


def live_hash_parts(
    inputs: CompositionInputs, result: CompositionResult
) -> dict[str, Any]:
    """The live-hash inputs: every refreshable section of the served text.

    The live hash records (never pins) the sections the lifecycle refreshes
    (design §10): the rendered launch-target table, the model and skills
    catalogs, the environment block, the commit signature, the per-launch
    instruction overlay, and the AGENTS.md layers with their provenance.
    Volatile session details — the date stamp, the scratchpad location, and
    the git/project context — are in NEITHER hash.
    """
    return {
        "launch_targets": inputs.launch_targets,
        "model_info": inputs.model_info,
        "skills": inputs.skills,
        "environment": inputs.environment,
        "commit_signature": inputs.commit_signature,
        "instruction_overlay": inputs.instruction_overlay,
        "agents_layers": [
            {
                "path": str(entry.path),
                "canonical_path": str(entry.canonical_path),
                "sources": list(entry.sources),
                "content": entry.content,
            }
            for entry in result.agents_provenance
        ],
    }


def live_fragments_hash(parts: Mapping[str, Any]) -> str:
    """The recorded (not pinned) live-fragments hash."""
    return _hash_payload({_LIVE_HASH_DOMAIN: dict(parts)})


class ResourceRecord(BaseModel):
    """One selected resource: selection, resolution, source, and revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    selected: str
    resolved_id: str
    source: str
    path: str | None
    revision: str
    content_bytes: int


class AgentsLayerRecord(BaseModel):
    """One deduplicated AGENTS.md layer with its provenance and revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    canonical_path: str
    sources: tuple[str, ...]
    revision: str
    content_bytes: int


class RouteTargetRecord(BaseModel):
    """One launch-target binding of the route snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    identity: str
    tier: str
    seat: str | None = None
    overlay: str | None = None
    replacement_capability: str = "none"


class RouteSnapshotEntry(BaseModel):
    """One purpose's route binding, sorted deterministically by purpose."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    purpose: str
    targets: tuple[RouteTargetRecord, ...]
    requires_distinct_models: int = 1


class CompositionManifest(BaseModel):
    """The deterministic composition manifest (design §6)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: Literal[1] = 1
    identity: ResourceRecord
    workflow: ResourceRecord
    strategy: ResourceRecord
    identity_revision: str
    universal_fragment_revision: str
    execution_context: str
    identity_delegation: str | None = None
    route_snapshot: tuple[RouteSnapshotEntry, ...] = ()
    agents: tuple[AgentsLayerRecord, ...] = ()
    frozen_policy_hash: str
    live_fragments_hash: str
    policy_hash: str
    policy_bytes: int
    text_bytes: int


@dataclass(frozen=True)
class PolicySnapshot:
    """Everything the manifest and bundle capture at composition time.

    The selections are the raw ``[agent]`` references; ``routes`` is the
    route snapshot; ``inputs``/``result`` are the package-6 composition.
    """

    identity_selection: str
    workflow_selection: str
    strategy_selection: str
    routes: Mapping[str, PurposeRoute]
    inputs: CompositionInputs
    result: CompositionResult


@dataclass(frozen=True)
class _FrozenBodies:
    identity: str
    workflow: str
    strategy: str | None
    universal: str
    context: str
    tool_semantics: str
    communication: str


def _required_body(what: str, content: str | None) -> str:
    if content is None or not content.strip():
        raise ManifestError(
            f"missing required content: the {what} has no policy body; the "
            f"manifest cannot be built"
        )
    return content


def _check_cap(what: str, content: str | None) -> None:
    if content is None:
        return
    size = len(content.encode("utf-8"))
    if size > MAX_RESOURCE_BYTES:
        raise PolicySizeLimitError(
            f"the {what} is {size} bytes, exceeding the per-resource "
            f"storage cap of {MAX_RESOURCE_BYTES} bytes (64 KiB); shorten "
            f"the resource before activation"
        )


def _frozen_bodies(snapshot: PolicySnapshot) -> _FrozenBodies:
    """Validate and extract the frozen bodies and fragments."""
    inputs = snapshot.inputs
    bodies = _FrozenBodies(
        identity=_required_body(
            f"identity resource {snapshot.identity_selection!r}",
            inputs.identity.content,
        ),
        workflow=_required_body(
            f"workflow resource {snapshot.workflow_selection!r}",
            inputs.workflow.content,
        ),
        strategy=inputs.strategy.content,
        universal=_required_body("universal fragment", inputs.universal),
        context=_required_body("context fragment", inputs.context_fragment),
        tool_semantics=_required_body("tool-semantics fragment", inputs.tool_semantics),
        communication=_required_body("communication fragment", inputs.communication),
    )
    for what, content in (
        (f"identity resource {snapshot.identity_selection!r}", bodies.identity),
        (f"workflow resource {snapshot.workflow_selection!r}", bodies.workflow),
        (f"strategy resource {snapshot.strategy_selection!r}", bodies.strategy),
        ("universal fragment", bodies.universal),
        ("context fragment", bodies.context),
        ("tool-semantics fragment", bodies.tool_semantics),
        ("communication fragment", bodies.communication),
    ):
        _check_cap(what, content)
    return bodies


def _resource_record(
    kind: str, selected: str, resource: ResolvedResource
) -> ResourceRecord:
    content = resource.content
    if content is None:
        # The concrete ``none`` strategy resolution: no body, no revision.
        return ResourceRecord(
            kind=kind,
            selected=selected,
            resolved_id=resource.id,
            source=resource.source,
            path=None,
            revision="",
            content_bytes=0,
        )
    return ResourceRecord(
        kind=kind,
        selected=selected,
        resolved_id=resource.id,
        source=resource.source,
        path=str(resource.path) if resource.path is not None else None,
        revision=resource_revision(content),
        content_bytes=len(content.encode("utf-8")),
    )


def _check_selection_match(kind: str, selection: str, record: ResourceRecord) -> None:
    expected = selection.split(":", 1)[1] if ":" in selection else selection
    if record.resolved_id != expected:
        raise ManifestError(
            f"the {kind} selection {selection!r} resolved to "
            f"{record.resolved_id!r}; the manifest cannot record an "
            f"incoherent selection"
        )


def _route_snapshot(
    routes: Mapping[str, PurposeRoute],
) -> tuple[RouteSnapshotEntry, ...]:
    entries: list[RouteSnapshotEntry] = []
    for purpose in sorted(routes):
        route = routes[purpose]
        entries.append(
            RouteSnapshotEntry(
                purpose=purpose,
                targets=tuple(
                    RouteTargetRecord(
                        model=target.model,
                        identity=target.identity,
                        tier=target.tier.value,
                        seat=target.seat,
                        overlay=target.overlay,
                        replacement_capability=(target.replacement_capability.value),
                    )
                    for target in route.targets
                ),
                requires_distinct_models=route.requires_distinct_models,
            )
        )
    return tuple(entries)


def routes_from_snapshot(
    snapshot: tuple[RouteSnapshotEntry, ...],
) -> dict[str, PurposeRoute]:
    """Rebuild the captured route table from a manifest route snapshot.

    The frozen inverse of :func:`_route_snapshot`: the persisted route
    records become the executable ``[orchestration.routes]`` table the
    conversation captured at bind time (design §6, §10). Launch admission
    resolves purpose routes against THIS table — the same source the
    rendered launch-target table carries — never the reloaded live
    config, so a mid-conversation route edit stays pending instead of
    producing a mixed-generation launch. An empty snapshot (a corrupt or
    hand-made bundle) yields an empty table, and purpose-keyed launches
    then fail loudly — fail-closed, never a silent fallback.
    """
    routes: dict[str, PurposeRoute] = {}
    for entry in snapshot:
        routes[entry.purpose] = PurposeRoute(
            targets=tuple(
                RouteTarget(
                    model=target.model,
                    identity=target.identity,
                    tier=PackageTier(target.tier),
                    seat=target.seat,
                    overlay=target.overlay,
                    replacement_capability=ReplacementCapability(
                        target.replacement_capability
                    ),
                )
                for target in entry.targets
            ),
            requires_distinct_models=entry.requires_distinct_models,
        )
    return routes


def _agents_records(result: CompositionResult) -> tuple[AgentsLayerRecord, ...]:
    records: list[AgentsLayerRecord] = []
    for entry in result.agents_provenance:
        records.append(
            AgentsLayerRecord(
                path=str(entry.path),
                canonical_path=str(entry.canonical_path),
                sources=tuple(entry.sources),
                revision=resource_revision(entry.content),
                content_bytes=len(entry.content.encode("utf-8")),
            )
        )
    return tuple(records)


def build_manifest(snapshot: PolicySnapshot) -> CompositionManifest:
    """Build the deterministic manifest for one composition (design §6)."""
    inputs = snapshot.inputs
    result = snapshot.result
    bodies = _frozen_bodies(snapshot)
    identity = _resource_record(
        "identity", snapshot.identity_selection, inputs.identity
    )
    workflow = _resource_record(
        "workflow", snapshot.workflow_selection, inputs.workflow
    )
    strategy = _resource_record(
        "strategy", snapshot.strategy_selection, inputs.strategy
    )
    for kind, selection, record in (
        ("identity", snapshot.identity_selection, identity),
        ("workflow", snapshot.workflow_selection, workflow),
        ("strategy", snapshot.strategy_selection, strategy),
    ):
        _check_selection_match(kind, selection, record)
    facts = result.identity_facts
    return CompositionManifest(
        identity=identity,
        workflow=workflow,
        strategy=strategy,
        identity_revision=identity.revision,
        universal_fragment_revision=resource_revision(bodies.universal),
        execution_context=result.execution_context.value,
        identity_delegation=(facts.delegation.value if facts is not None else None),
        route_snapshot=_route_snapshot(snapshot.routes),
        agents=_agents_records(result),
        frozen_policy_hash=frozen_policy_hash(
            frozen_hash_parts_from_content(
                identity_body=bodies.identity,
                workflow_body=bodies.workflow,
                strategy_body=bodies.strategy,
                universal_fragment=bodies.universal,
                context_fragment=bodies.context,
                tool_semantics_fragment=bodies.tool_semantics,
                communication_fragment=bodies.communication,
            )
        ),
        live_fragments_hash=live_fragments_hash(live_hash_parts(inputs, result)),
        policy_hash=content_hash(result.policy_text),
        policy_bytes=len(result.policy_text.encode("utf-8")),
        text_bytes=len(result.text.encode("utf-8")),
    )


def serialize_manifest(manifest: CompositionManifest) -> bytes:
    """Serialize the manifest to canonical, byte-stable JSON."""
    return canonical_json_bytes(manifest.model_dump(mode="json"))


def parse_manifest(payload: bytes | str) -> CompositionManifest:
    """Parse a manifest payload, failing with actionable errors."""
    try:
        data = decode_json(payload, what="manifest payload")
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ManifestError(
            "manifest payload is corrupt: the top-level value is not a JSON object"
        )
    version = data.get("manifest_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != MANIFEST_FORMAT_VERSION
    ):
        raise ManifestError(
            f"unsupported manifest format version {version!r}: this harness "
            f"reads version {MANIFEST_FORMAT_VERSION} manifests"
        )
    try:
        return CompositionManifest.model_validate(data)
    except ValidationError as exc:
        raise ManifestError(
            "manifest payload is corrupt: " + format_validation_error(exc)
        ) from exc
