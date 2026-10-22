"""The frozen policy bundle codec and store (design §6, §10)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chartreux.core.config.harness_files import FileSource
from chartreux.core.config.harness_files._harness_manager import InstructionDocument
from chartreux.core.dispatch.renderer import (
    render_composition_regions,
    render_launch_target_region,
)
from chartreux.core.dispatch.route_models import PackageTier, PurposeRoute, RouteTarget
from chartreux.core.dispatch.routes import shipped_routes
from chartreux.core.policy.bundle import (
    BundleError,
    BundleIntegrityError,
    BundleStore,
    BundleStoreError,
    BundleTooLargeError,
    LiveSections,
    PolicyPinMismatchError,
    activate_bundle,
    build_bundle,
    load_and_activate,
    parse_bundle,
    serialize_bundle,
)
from chartreux.core.policy.composition import (
    CONTEXT_FRAGMENTS,
    CompositionInputs,
    CompositionResult,
    ExecutionContext,
    compose_policy,
    fragment_loader,
    policy_resource_loader,
    resolve_identity_facts,
)
from chartreux.core.policy.manifest import (
    MAX_FROZEN_INLINE_BYTES,
    MAX_RESOURCE_BYTES,
    PolicySnapshot,
    build_manifest,
    canonical_json_bytes,
    routes_from_snapshot,
)
from chartreux.core.policy.models import ResolvedResource, ResourceKind

SESSION_MODEL = "beta-model"
PINNED_DATE = "2000-01-01 (Saturday)"


def _fragment(name: str) -> str:
    resolved = fragment_loader().resolve(ResourceKind.IDENTITY, name)
    assert resolved.content is not None
    return resolved.content


def _user_dirs(tmp_path: Path) -> dict[ResourceKind, Path]:
    return {
        ResourceKind.IDENTITY: tmp_path / "prompts",
        ResourceKind.WORKFLOW: tmp_path / "workflows",
        ResourceKind.STRATEGY: tmp_path / "strategies",
    }


def _inputs(
    tmp_path: Path,
    *,
    identity: str = "coder",
    workflow: str = "minimal",
    strategy: str = "none",
    execution_context: ExecutionContext = ExecutionContext.INTERACTIVE_MAIN,
    agents_documents: tuple[InstructionDocument, ...] = (),
    instruction_overlay: str | None = None,
    current_date: str = PINNED_DATE,
    model_info: str | None = None,
    skills: str | None = None,
    scratchpad: str | None = None,
    environment: str | None = None,
    project_context: str | None = None,
    commit_signature: str | None = None,
    user_dirs: dict[ResourceKind, Path] | None = None,
    regions: dict[str, str] | None = None,
    launch_targets: str | None = None,
) -> CompositionInputs:
    """Build composition inputs from the shipped bodies and route facts."""
    dirs = user_dirs if user_dirs is not None else _user_dirs(tmp_path)
    for directory in dirs.values():
        directory.mkdir(exist_ok=True)
    loader = policy_resource_loader(dirs)
    facts = resolve_identity_facts(identity)
    return CompositionInputs(
        identity=loader.resolve(ResourceKind.IDENTITY, identity),
        workflow=loader.resolve(ResourceKind.WORKFLOW, workflow),
        strategy=loader.resolve(ResourceKind.STRATEGY, strategy),
        identity_facts=facts,
        execution_context=execution_context,
        universal=_fragment("universal"),
        context_fragment=_fragment(CONTEXT_FRAGMENTS[execution_context]),
        tool_semantics=_fragment("tool_semantics"),
        communication=_fragment("communication"),
        launch_targets=(
            launch_targets
            if launch_targets is not None
            else render_launch_target_region(dict(shipped_routes()), SESSION_MODEL)
        ),
        regions=(
            regions
            if regions is not None
            else render_composition_regions(
                delegation=facts.delegation if facts is not None else None
            )
        ),
        model_info=model_info,
        skills=skills,
        scratchpad=scratchpad,
        environment=environment,
        project_context=project_context,
        commit_signature=commit_signature,
        instruction_overlay=instruction_overlay,
        agents_documents=agents_documents,
        current_date=current_date,
    )


def _snapshot(
    tmp_path: Path,
    *,
    identity: str = "coder",
    workflow: str = "minimal",
    strategy: str = "none",
    routes=None,
    **kwargs,
) -> tuple[PolicySnapshot, CompositionResult]:
    inputs = _inputs(
        tmp_path, identity=identity, workflow=workflow, strategy=strategy, **kwargs
    )
    result = compose_policy(inputs)
    return (
        PolicySnapshot(
            identity_selection=identity,
            workflow_selection=workflow,
            strategy_selection=strategy,
            routes=shipped_routes() if routes is None else routes,
            inputs=inputs,
            result=result,
        ),
        result,
    )


def _document(
    source: FileSource, tmp_path: Path, name: str, content: str
) -> InstructionDocument:
    path = tmp_path / name
    return InstructionDocument(source, path, path, content)


# --- Persisted record -------------------------------------------------------


def test_bundle_persists_the_full_policy_record(tmp_path):
    dirs = _user_dirs(tmp_path)
    identity_file = dirs[ResourceKind.IDENTITY] / "custom.md"
    identity_file.parent.mkdir(parents=True)
    identity_file.write_text("custom identity body", encoding="utf-8")
    documents = (_document("project", tmp_path, "AGENTS.md", "project rules"),)
    snapshot, _ = _snapshot(
        tmp_path,
        identity="user:custom",
        user_dirs=dirs,
        agents_documents=documents,
        model_info="catalog",
        skills="skills catalog",
    )
    bundle = build_bundle(snapshot)
    manifest = bundle.manifest
    assert bundle.version == 1
    assert manifest.identity.selected == "user:custom"
    assert manifest.identity.resolved_id == "custom"
    assert manifest.identity.source == "user"
    assert manifest.identity_revision == manifest.identity.revision
    assert bundle.frozen.identity_body == "custom identity body"
    assert bundle.frozen.workflow_body == snapshot.inputs.workflow.content
    assert bundle.frozen.strategy_body is None
    assert bundle.frozen.universal_fragment == snapshot.inputs.universal
    assert bundle.frozen.context_fragment == snapshot.inputs.context_fragment
    assert bundle.frozen.tool_semantics_fragment == snapshot.inputs.tool_semantics
    assert bundle.frozen.communication_fragment == snapshot.inputs.communication
    assert bundle.frozen.launch_targets == snapshot.inputs.launch_targets
    assert bundle.frozen.current_date == snapshot.inputs.current_date
    assert bundle.frozen.regions == dict(snapshot.inputs.regions)
    assert manifest.agents[0].sources == ("project",)
    assert manifest.frozen_policy_hash.startswith("sha256:")
    assert manifest.live_fragments_hash.startswith("sha256:")
    assert manifest == build_manifest(snapshot)


def test_bundle_serialization_is_byte_identical(tmp_path):
    first, _ = _snapshot(tmp_path)
    second, _ = _snapshot(tmp_path)
    assert serialize_bundle(build_bundle(first)) == serialize_bundle(
        build_bundle(second)
    )


# --- Round-trip and content sufficiency -------------------------------------


def test_round_trip_reactivates_with_full_fidelity(tmp_path):
    documents = (_document("project", tmp_path, "AGENTS.md", "project rules"),)
    snapshot, result = _snapshot(
        tmp_path,
        model_info="model catalog",
        skills="skills catalog",
        environment="environment block",
        agents_documents=documents,
    )
    bundle = build_bundle(snapshot)
    payload = serialize_bundle(bundle)
    activated = load_and_activate(payload)
    assert activated.manifest == bundle.manifest
    assert activated.frozen == bundle.frozen
    assert activated.policy_text == result.policy_text
    rendered = activated.render(
        LiveSections(
            model_info="model catalog",
            skills="skills catalog",
            environment="environment block",
            agents_documents=documents,
        )
    )
    assert rendered.text == result.text
    assert rendered.policy_text == result.policy_text


def test_bundle_resumes_after_source_files_are_edited_or_removed(tmp_path):
    dirs = _user_dirs(tmp_path)
    identity_file = dirs[ResourceKind.IDENTITY] / "custom.md"
    workflow_file = dirs[ResourceKind.WORKFLOW] / "customwf.md"
    identity_file.parent.mkdir(parents=True)
    workflow_file.parent.mkdir(parents=True)
    identity_file.write_text("custom identity body v1", encoding="utf-8")
    workflow_file.write_text("custom workflow body v1", encoding="utf-8")
    documents = (_document("project", tmp_path, "AGENTS.md", "project rules"),)
    snapshot, result = _snapshot(
        tmp_path,
        identity="user:custom",
        workflow="user:customwf",
        user_dirs=dirs,
        model_info="model catalog",
        agents_documents=documents,
    )
    payload = serialize_bundle(build_bundle(snapshot))

    # The source resource files are edited and then removed entirely: the
    # bundle carries the actual content, so resume still works.
    identity_file.write_text("TAMPERED identity body v2", encoding="utf-8")
    workflow_file.unlink()
    identity_file.unlink()

    activated = load_and_activate(payload)
    assert activated.frozen.identity_body == "custom identity body v1"
    assert activated.frozen.workflow_body == "custom workflow body v1"
    assert activated.policy_text == result.policy_text
    rendered = activated.render(
        LiveSections(model_info="model catalog", agents_documents=documents)
    )
    assert rendered.text == result.text


def test_build_rejects_incoherent_inputs_and_result(tmp_path):
    inputs = _inputs(tmp_path)
    other = replace(
        inputs,
        identity=ResolvedResource(
            ResourceKind.IDENTITY, "coder", "builtin", "a different body"
        ),
    )
    snapshot = PolicySnapshot(
        identity_selection="coder",
        workflow_selection="minimal",
        strategy_selection="none",
        routes=shipped_routes(),
        inputs=inputs,
        result=compose_policy(other),
    )
    with pytest.raises(BundleError, match="does not reproduce"):
        build_bundle(snapshot)


# --- Activation: integrity and pins ----------------------------------------


def test_expected_pin_mismatch_rejects_activation_before_construction(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    bundle = build_bundle(snapshot)
    payload = serialize_bundle(bundle)
    # The correct pin activates.
    activated = load_and_activate(
        payload, expected_frozen_hash=bundle.manifest.frozen_policy_hash
    )
    assert activated.frozen_policy_hash == bundle.manifest.frozen_policy_hash
    # A disagreeing pin rejects activation; the structural parse succeeds,
    # so the rejection is the activation gate, before any client or
    # resource is constructed.
    parsed = parse_bundle(payload)
    with pytest.raises(PolicyPinMismatchError, match="expected frozen-policy pin"):
        activate_bundle(parsed, expected_frozen_hash="sha256:" + "0" * 64)
    with pytest.raises(PolicyPinMismatchError, match="does not match"):
        load_and_activate(payload, expected_frozen_hash="sha256:" + "1" * 64)


def test_tampered_frozen_content_rejected_at_activation(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    data = build_bundle(snapshot).model_dump(mode="json")
    data["frozen"]["identity_body"] = "tampered body"
    tampered = parse_bundle(canonical_json_bytes(data))
    with pytest.raises(BundleIntegrityError, match="frozen-policy hash"):
        activate_bundle(tampered)
    with pytest.raises(BundleIntegrityError, match="recomputed"):
        load_and_activate(canonical_json_bytes(data))


def test_tampered_overlay_rejected_by_recomposition(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    data = build_bundle(snapshot).model_dump(mode="json")
    # The overlay is not a frozen-hash part, but the recomposed policy
    # must still reproduce the stored composed-policy hash.
    data["frozen"]["instruction_overlay"] = "tampered overlay"
    tampered = parse_bundle(canonical_json_bytes(data))
    with pytest.raises(BundleIntegrityError, match="composed-policy hash"):
        activate_bundle(tampered)


def test_tampered_route_snapshot_rejected_at_activation(tmp_path):
    # DF2 (delta re-review): the manifest's route snapshot is the EXECUTABLE
    # route table and the frozen launch-target string is the ADVERTISED one
    # — both captured from the same source at bind time. A persisted bundle
    # whose snapshot disagrees with its own advertised table re-creates the
    # half-applied generation from disk (admission would launch the tampered
    # target while the model is told the advertised one), so activation
    # refuses it loudly, naming the purpose and both sides.
    snapshot, _ = _snapshot(tmp_path)
    data = build_bundle(snapshot).model_dump(mode="json")
    research = next(
        entry
        for entry in data["manifest"]["route_snapshot"]
        if entry["purpose"] == "research"
    )
    research["targets"][0]["identity"] = "coder"
    tampered = parse_bundle(canonical_json_bytes(data))
    with pytest.raises(BundleIntegrityError, match="purpose 'research'"):
        activate_bundle(tampered)
    with pytest.raises(BundleIntegrityError, match="advertised launch-target table"):
        load_and_activate(canonical_json_bytes(data))
    # The typed error quotes both sides: the tampered snapshot row and the
    # advertised table row.
    with pytest.raises(BundleIntegrityError) as excinfo:
        activate_bundle(tampered)
    message = str(excinfo.value)
    assert "`coder`" in message
    assert "`researcher`" in message


def test_tampered_launch_target_table_rejected_at_activation(tmp_path):
    # DF2 other direction: the advertised launch-target table is part of the
    # recomposed policy, so tampering it fails the composed-policy hash at
    # activation — the same loud refusal, from the hash side.
    snapshot, _ = _snapshot(tmp_path)
    data = build_bundle(snapshot).model_dump(mode="json")
    data["frozen"]["launch_targets"] = data["frozen"]["launch_targets"].replace(
        "`researcher`", "`coder`"
    )
    tampered = parse_bundle(canonical_json_bytes(data))
    with pytest.raises(BundleIntegrityError, match="composed-policy hash"):
        activate_bundle(tampered)


def test_consistent_route_snapshot_activates(tmp_path):
    # DF2 control: an internally-consistent bundle — the snapshot and the
    # advertised table captured from the same source — activates. The
    # custom table is deliberately NOT sorted by purpose (the snapshot is
    # sorted, the render followed the configured order) and carries an
    # explicit (non-`$session`) model binding.
    routes = {
        "zeta-purpose": PurposeRoute(
            targets=(
                RouteTarget(
                    model="$session", identity="coder", tier=PackageTier.MECHANICAL
                ),
            )
        ),
        "alpha-purpose": PurposeRoute(
            targets=(
                RouteTarget(
                    model="explicit-model",
                    identity="reviewer",
                    tier=PackageTier.SUBSTANTIVE,
                ),
            )
        ),
    }
    snapshot, _ = _snapshot(
        tmp_path,
        routes=routes,
        launch_targets=render_launch_target_region(routes, SESSION_MODEL),
    )
    activated = activate_bundle(build_bundle(snapshot))
    assert routes_from_snapshot(activated.manifest.route_snapshot) == routes
    # The shipped default bundle is consistent too.
    shipped = activate_bundle(build_bundle(_snapshot(tmp_path)[0]))
    research = next(
        entry
        for entry in shipped.manifest.route_snapshot
        if entry.purpose == "research"
    )
    assert research.targets[0].identity == "researcher"


def test_empty_route_snapshot_bundle_still_activates(tmp_path):
    # DF2 compat control: the pre-snapshot persisted format carries
    # route_snapshot=() — activation skips the cross-check for it (an empty
    # snapshot already fails closed at launch admission with a loud
    # RouteResolutionError), so old bundles keep resuming.
    snapshot, _ = _snapshot(tmp_path)
    data = build_bundle(snapshot).model_dump(mode="json")
    data["manifest"]["route_snapshot"] = []
    legacy = parse_bundle(canonical_json_bytes(data))
    activated = activate_bundle(legacy)
    assert activated.manifest.route_snapshot == ()


def test_corrupt_payload_fails_actionably(tmp_path):
    with pytest.raises(BundleError, match="not valid JSON"):
        parse_bundle(b'{"version": 1,')
    with pytest.raises(BundleError, match="top-level value is not"):
        parse_bundle(b"[1, 2]")


def test_missing_required_content_fails_actionably(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    data = build_bundle(snapshot).model_dump(mode="json")
    data["frozen"]["identity_body"] = ""
    with pytest.raises(BundleError, match="frozen.identity_body"):
        parse_bundle(canonical_json_bytes(data))
    data = build_bundle(snapshot).model_dump(mode="json")
    del data["manifest"]
    with pytest.raises(BundleError, match="manifest"):
        parse_bundle(canonical_json_bytes(data))
    data = build_bundle(snapshot).model_dump(mode="json")
    data["frozen"]["universal_fragment"] = "   "
    with pytest.raises(BundleError, match="frozen.universal_fragment"):
        parse_bundle(canonical_json_bytes(data))


def test_unsupported_version_fails_actionably(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    data = build_bundle(snapshot).model_dump(mode="json")
    data["version"] = 2
    with pytest.raises(BundleError, match="unsupported bundle format version 2"):
        parse_bundle(canonical_json_bytes(data))
    data["version"] = "1"
    with pytest.raises(BundleError, match="unsupported bundle format version '1'"):
        parse_bundle(canonical_json_bytes(data))
    data = build_bundle(snapshot).model_dump(mode="json")
    data["manifest"]["manifest_version"] = 7
    with pytest.raises(BundleError, match="manifest.manifest_version"):
        parse_bundle(canonical_json_bytes(data))


# --- Storage caps (interfaces.md "Bundle storage caps") ----------------------


def test_per_resource_cap_names_the_cap_and_the_offender(tmp_path):
    inputs = _inputs(tmp_path)
    oversized = ResolvedResource(
        ResourceKind.IDENTITY, "huge", "user", "x" * (MAX_RESOURCE_BYTES + 1)
    )
    patched = replace(inputs, identity=oversized)
    snapshot = PolicySnapshot(
        identity_selection="user:huge",
        workflow_selection="minimal",
        strategy_selection="none",
        routes=shipped_routes(),
        inputs=patched,
        result=compose_policy(patched),
    )
    with pytest.raises(BundleTooLargeError, match="65536") as excinfo:
        build_bundle(snapshot)
    assert "user:huge" in str(excinfo.value)
    # The parse path re-checks the cap on a tampered payload.
    data = build_bundle(_snapshot(tmp_path)[0]).model_dump(mode="json")
    data["frozen"]["identity_body"] = "x" * (MAX_RESOURCE_BYTES + 1)
    with pytest.raises(BundleTooLargeError, match="65536"):
        parse_bundle(canonical_json_bytes(data))


def test_frozen_inline_cap_names_the_cap_and_the_offenders(tmp_path):
    dirs = _user_dirs(tmp_path)
    big = "x" * MAX_RESOURCE_BYTES
    for kind, name in (
        (ResourceKind.IDENTITY, "bigid"),
        (ResourceKind.WORKFLOW, "bigwf"),
        (ResourceKind.STRATEGY, "bigst"),
    ):
        directory = dirs[kind]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.md").write_text(big, encoding="utf-8")
    snapshot, _ = _snapshot(
        tmp_path,
        identity="user:bigid",
        workflow="user:bigwf",
        strategy="user:bigst",
        user_dirs=dirs,
    )
    with pytest.raises(BundleTooLargeError, match="196608") as excinfo:
        build_bundle(snapshot)
    assert "192 KiB" in str(excinfo.value)
    assert "identity_body" in str(excinfo.value)
    assert MAX_FROZEN_INLINE_BYTES == 192 * 1024


def _huge_route_snapshot(count: int) -> list[dict[str, object]]:
    return [
        {
            "purpose": f"purpose-{index:05d}",
            "targets": [
                {
                    "model": "$session",
                    "identity": "coder",
                    "tier": "substantive",
                    "seat": None,
                    "overlay": None,
                    "replacement_capability": "none",
                }
            ],
            "requires_distinct_models": 1,
        }
        for index in range(count)
    ]


def test_aggregate_cap_names_the_cap(tmp_path):
    routes = {
        f"purpose-{index:05d}": PurposeRoute(
            targets=(
                RouteTarget(
                    model="$session", identity="coder", tier=PackageTier.SUBSTANTIVE
                ),
            )
        )
        for index in range(3000)
    }
    snapshot, _ = _snapshot(tmp_path, routes=routes)
    with pytest.raises(BundleTooLargeError, match="262144") as excinfo:
        build_bundle(snapshot)
    assert "256 KiB" in str(excinfo.value)
    # The parse path re-checks the aggregate cap on an inflated manifest.
    data = build_bundle(_snapshot(tmp_path)[0]).model_dump(mode="json")
    data["manifest"]["route_snapshot"] = _huge_route_snapshot(3000)
    with pytest.raises(BundleTooLargeError, match="262144"):
        parse_bundle(canonical_json_bytes(data))


# --- Bundle store and the eviction guard ------------------------------------


def test_store_round_trip_resumes_after_source_removal(tmp_path):
    dirs = _user_dirs(tmp_path)
    identity_file = dirs[ResourceKind.IDENTITY] / "custom.md"
    identity_file.parent.mkdir(parents=True)
    identity_file.write_text("custom identity body", encoding="utf-8")
    snapshot, result = _snapshot(tmp_path, identity="user:custom", user_dirs=dirs)
    store = BundleStore()
    store.put("conv-1", build_bundle(snapshot))
    store.set_resumable("conv-1", True)
    identity_file.unlink()
    assert len(store) == 1
    assert "conv-1" in store
    activated = store.activate("conv-1")
    assert activated.policy_text == result.policy_text
    assert store.get("conv-1") is not None


def test_store_activate_unknown_conversation_fails():
    store = BundleStore()
    with pytest.raises(BundleStoreError, match="conv-missing"):
        store.activate("conv-missing")
    assert store.get("conv-missing") is None


def test_eviction_guard_refuses_while_the_conversation_is_resumable(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    store = BundleStore()
    store.put("conv-1", build_bundle(snapshot))
    store.set_resumable("conv-1", True)
    with pytest.raises(BundleStoreError, match="conv-1") as excinfo:
        store.evict("conv-1", conversation_resumable=True)
    assert "remains resumable" in str(excinfo.value)
    assert "conv-1" in store  # the sole persisted bundle survives
    store.evict("conv-1", conversation_resumable=False)
    assert "conv-1" not in store


def test_store_bound_evicts_only_non_resumable_bundles(tmp_path):
    snapshots = [_snapshot(tmp_path)[0] for _ in range(3)]
    store = BundleStore(max_entries=2)
    store.put("conv-1", build_bundle(snapshots[0]))
    store.put("conv-2", build_bundle(snapshots[1]))
    store.set_resumable("conv-1", True)
    store.set_resumable("conv-2", True)
    with pytest.raises(BundleStoreError, match="eviction guard refuses"):
        store.put("conv-3", build_bundle(snapshots[2]))
    store.set_resumable("conv-2", False)
    store.put("conv-3", build_bundle(snapshots[2]))
    assert "conv-1" in store
    assert "conv-2" not in store
    assert "conv-3" in store


def test_ensure_capacity_mirrors_the_put_bound(tmp_path):
    # The preparation-time capacity check mirrors put exactly: it settles
    # the store's bound BEFORE any durable change, refusing loudly and
    # deleting nothing — never the sole durable bundle of a resumable
    # conversation.
    snapshots = [_snapshot(tmp_path)[0] for _ in range(3)]
    # Below capacity the check passes for a new conversation.
    BundleStore().ensure_capacity("conv-new")
    store = BundleStore(max_entries=2)
    store.put("conv-1", build_bundle(snapshots[0]))
    store.put("conv-2", build_bundle(snapshots[1]))
    store.set_resumable("conv-1", True)
    store.set_resumable("conv-2", True)
    # An existing conversation always fits (its put overwrites in place).
    store.ensure_capacity("conv-1")
    # A full store of resumable conversations refuses loudly and deletes
    # nothing — both sole durable bundles survive.
    with pytest.raises(BundleStoreError, match="eviction guard refuses"):
        store.ensure_capacity("conv-3")
    assert "conv-1" in store
    assert "conv-2" in store
    assert len(store) == 2
    # A non-resumable entry makes room: the check passes and the paired
    # put evicts exactly that entry.
    store.set_resumable("conv-2", False)
    store.ensure_capacity("conv-3")
    store.put("conv-3", build_bundle(snapshots[2]))
    assert "conv-1" in store
    assert "conv-2" not in store
    assert "conv-3" in store
