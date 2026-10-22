"""The deterministic composition manifest (design §6)."""

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
from chartreux.core.dispatch.routes import shipped_routes
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
    MAX_RESOURCE_BYTES,
    ManifestError,
    PolicySizeLimitError,
    PolicySnapshot,
    build_manifest,
    canonical_json_bytes,
    content_hash,
    frozen_hash_parts,
    frozen_policy_hash,
    live_fragments_hash,
    live_hash_parts,
    parse_manifest,
    resource_revision,
    serialize_manifest,
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


# --- Persisted record (design §6) ------------------------------------------


def test_manifest_records_ids_sources_revisions_and_hashes(tmp_path):
    snapshot, result = _snapshot(tmp_path, model_info="catalog")
    manifest = build_manifest(snapshot)
    assert manifest.manifest_version == 1
    assert manifest.identity.selected == "coder"
    assert manifest.identity.resolved_id == "coder"
    assert manifest.identity.source == "builtin"
    assert manifest.identity.path is not None
    assert manifest.identity.path.endswith("coder.md")
    assert manifest.workflow.resolved_id == "minimal"
    assert manifest.workflow.source == "builtin"
    assert manifest.strategy.selected == "none"
    assert manifest.strategy.resolved_id == "none"
    assert manifest.strategy.source == "builtin"
    assert manifest.strategy.revision == ""
    assert manifest.strategy.content_bytes == 0
    assert manifest.identity_revision == manifest.identity.revision
    assert manifest.identity_revision == resource_revision(
        snapshot.inputs.identity.content or ""
    )
    assert manifest.universal_fragment_revision == resource_revision(
        snapshot.inputs.universal
    )
    assert manifest.execution_context == "interactive_main"
    assert manifest.identity_delegation == "optional"
    assert manifest.frozen_policy_hash.startswith("sha256:")
    assert manifest.live_fragments_hash.startswith("sha256:")
    assert manifest.policy_hash == content_hash(result.policy_text)
    assert manifest.policy_bytes == len(result.policy_text.encode("utf-8"))
    assert manifest.text_bytes == len(result.text.encode("utf-8"))


def test_manifest_records_the_route_snapshot(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    manifest = build_manifest(snapshot)
    purposes = [entry.purpose for entry in manifest.route_snapshot]
    assert purposes == sorted(shipped_routes())
    deep = next(
        entry for entry in manifest.route_snapshot if entry.purpose == "review.deep"
    )
    assert deep.requires_distinct_models == 3
    assert [target.seat for target in deep.targets] == [
        "analytical",
        "peer",
        "execution",
    ]
    assert [target.identity for target in deep.targets] == [
        "reviewer",
        "reviewer",
        "coder",
    ]
    assert all(target.tier == "substantive" for target in deep.targets)
    implementation = next(
        entry for entry in manifest.route_snapshot if entry.purpose == "implementation"
    )
    assert implementation.targets[0].model == "$session"
    assert implementation.targets[0].tier == "substantive"


def test_manifest_records_agents_provenance(tmp_path):
    documents = (
        _document("user", tmp_path, "home-AGENTS.md", "user rules"),
        _document("project", tmp_path, "repo-AGENTS.md", "project rules"),
    )
    snapshot, _ = _snapshot(tmp_path, agents_documents=documents)
    manifest = build_manifest(snapshot)
    assert len(manifest.agents) == 2
    user_layer, project_layer = manifest.agents
    assert user_layer.sources == ("user",)
    assert user_layer.revision == resource_revision("user rules")
    assert user_layer.content_bytes == len(b"user rules")
    assert project_layer.sources == ("project",)
    assert project_layer.revision == resource_revision("project rules")


def test_manifest_records_deduplicated_agents_provenance(tmp_path):
    canonical = tmp_path / "AGENTS.md"
    documents = (
        InstructionDocument("user", canonical, canonical, "shared rules"),
        InstructionDocument("project", canonical, canonical, "shared rules"),
    )
    snapshot, _ = _snapshot(tmp_path, agents_documents=documents)
    manifest = build_manifest(snapshot)
    assert len(manifest.agents) == 1
    assert manifest.agents[0].sources == ("user", "project")
    assert manifest.agents[0].revision == resource_revision("shared rules")


# --- Determinism ------------------------------------------------------------


def test_manifest_is_byte_identical_for_the_same_inputs(tmp_path):
    first, _ = _snapshot(tmp_path)
    second, _ = _snapshot(tmp_path)
    assert serialize_manifest(build_manifest(first)) == serialize_manifest(
        build_manifest(second)
    )
    # The route mapping's insertion order does not matter.
    routes = dict(reversed(list(shipped_routes().items())))
    third = PolicySnapshot(
        identity_selection="coder",
        workflow_selection="minimal",
        strategy_selection="none",
        routes=routes,
        inputs=second.inputs,
        result=second.result,
    )
    assert serialize_manifest(build_manifest(third)) == serialize_manifest(
        build_manifest(second)
    )


def test_manifest_serialization_round_trip(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    manifest = build_manifest(snapshot)
    payload = serialize_manifest(manifest)
    assert parse_manifest(payload) == manifest
    assert parse_manifest(payload.decode("utf-8")) == manifest


# --- Frozen vs live hash separation -----------------------------------------


def test_frozen_hash_covers_exactly_the_policy_bodies_and_fragments(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    manifest = build_manifest(snapshot)
    assert manifest.frozen_policy_hash == frozen_policy_hash(
        frozen_hash_parts(snapshot.inputs)
    )
    # The live sections are NOT part of the frozen hash.
    live_only, _ = _snapshot(
        tmp_path,
        model_info="catalog",
        skills="skills catalog",
        environment="environment block",
        commit_signature="signature",
        instruction_overlay="per-launch overlay",
        agents_documents=(
            _document("project", tmp_path, "AGENTS.md", "project rules"),
        ),
    )
    assert build_manifest(live_only).frozen_policy_hash == manifest.frozen_policy_hash


def test_resource_edits_change_the_frozen_hash(tmp_path):
    dirs = _user_dirs(tmp_path)
    identity_file = dirs[ResourceKind.IDENTITY] / "custom.md"
    identity_file.parent.mkdir(parents=True)
    identity_file.write_text("custom identity v1", encoding="utf-8")
    first, _ = _snapshot(tmp_path, identity="user:custom", user_dirs=dirs)
    first_manifest = build_manifest(first)

    identity_file.write_text("custom identity v2 — edited", encoding="utf-8")
    second, _ = _snapshot(tmp_path, identity="user:custom", user_dirs=dirs)
    second_manifest = build_manifest(second)
    assert second_manifest.frozen_policy_hash != first_manifest.frozen_policy_hash
    assert second_manifest.live_fragments_hash == first_manifest.live_fragments_hash
    assert second_manifest.identity.revision != first_manifest.identity.revision
    assert second_manifest.identity_revision == second_manifest.identity.revision


def test_workflow_and_strategy_edits_change_the_frozen_hash(tmp_path):
    dirs = _user_dirs(tmp_path)
    workflow_file = dirs[ResourceKind.WORKFLOW] / "customwf.md"
    strategy_file = dirs[ResourceKind.STRATEGY] / "customst.md"
    workflow_file.parent.mkdir(parents=True)
    strategy_file.parent.mkdir(parents=True)
    workflow_file.write_text("custom workflow v1", encoding="utf-8")
    strategy_file.write_text("custom strategy v1", encoding="utf-8")
    first, _ = _snapshot(
        tmp_path, workflow="user:customwf", strategy="user:customst", user_dirs=dirs
    )
    first_frozen = build_manifest(first).frozen_policy_hash

    workflow_file.write_text("custom workflow v2", encoding="utf-8")
    second, _ = _snapshot(
        tmp_path, workflow="user:customwf", strategy="user:customst", user_dirs=dirs
    )
    assert build_manifest(second).frozen_policy_hash != first_frozen

    strategy_file.write_text("custom strategy v2", encoding="utf-8")
    third, _ = _snapshot(
        tmp_path, workflow="user:customwf", strategy="user:customst", user_dirs=dirs
    )
    assert build_manifest(third).frozen_policy_hash != first_frozen


def test_agents_and_catalog_changes_affect_only_the_live_hash(tmp_path):
    base_snapshot, _ = _snapshot(tmp_path)
    base = build_manifest(base_snapshot)

    documents = (_document("project", tmp_path, "AGENTS.md", "project rules v1"),)
    with_agents, _ = _snapshot(tmp_path, agents_documents=documents)
    agents_manifest = build_manifest(with_agents)
    assert agents_manifest.live_fragments_hash != base.live_fragments_hash
    assert agents_manifest.frozen_policy_hash == base.frozen_policy_hash

    edited_documents = (
        _document("project", tmp_path, "AGENTS.md", "project rules v2"),
    )
    edited, _ = _snapshot(tmp_path, agents_documents=edited_documents)
    edited_manifest = build_manifest(edited)
    assert edited_manifest.live_fragments_hash != agents_manifest.live_fragments_hash
    assert edited_manifest.frozen_policy_hash == base.frozen_policy_hash

    with_catalog, _ = _snapshot(
        tmp_path,
        model_info="model catalog v1",
        skills="skills catalog v1",
        environment="environment v1",
    )
    catalog_manifest = build_manifest(with_catalog)
    assert catalog_manifest.live_fragments_hash != base.live_fragments_hash
    assert catalog_manifest.frozen_policy_hash == base.frozen_policy_hash
    assert catalog_manifest.live_fragments_hash == live_fragments_hash(
        live_hash_parts(with_catalog.inputs, with_catalog.result)
    )


def test_volatile_inputs_affect_neither_hash(tmp_path):
    base_snapshot, _ = _snapshot(tmp_path)
    base = build_manifest(base_snapshot)
    volatile_snapshot, _ = _snapshot(
        tmp_path,
        current_date="2030-12-31 (Wednesday)",
        scratchpad="/a/different/scratchpad",
        project_context="On branch main\nnothing to commit, working tree clean",
    )
    volatile = build_manifest(volatile_snapshot)
    assert volatile.frozen_policy_hash == base.frozen_policy_hash
    assert volatile.live_fragments_hash == base.live_fragments_hash


# --- Actionable failures ----------------------------------------------------


def test_parse_manifest_rejects_corrupt_and_unsupported_payloads(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    manifest = build_manifest(snapshot)
    with pytest.raises(ManifestError, match="not valid JSON"):
        parse_manifest(b"{\n")
    with pytest.raises(ManifestError, match="top-level value is not"):
        parse_manifest(b"[1, 2]")
    data = manifest.model_dump(mode="json")
    data["manifest_version"] = 2
    with pytest.raises(ManifestError, match="unsupported manifest format version 2"):
        parse_manifest(canonical_json_bytes(data))
    data = manifest.model_dump(mode="json")
    del data["identity"]
    with pytest.raises(ManifestError, match="identity"):
        parse_manifest(canonical_json_bytes(data))
    data = manifest.model_dump(mode="json")
    data["surprise"] = 1
    with pytest.raises(ManifestError, match="surprise"):
        parse_manifest(canonical_json_bytes(data))


def test_oversized_resource_fails_with_the_cap_and_the_offender(tmp_path):
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
    with pytest.raises(PolicySizeLimitError, match="65536") as excinfo:
        build_manifest(snapshot)
    assert "user:huge" in str(excinfo.value)
    assert "64 KiB" in str(excinfo.value)


def test_oversized_fragment_fails_with_the_cap_and_the_offender(tmp_path):
    inputs = _inputs(tmp_path)
    patched = replace(inputs, universal="x" * (MAX_RESOURCE_BYTES + 1))
    snapshot = PolicySnapshot(
        identity_selection="coder",
        workflow_selection="minimal",
        strategy_selection="none",
        routes=shipped_routes(),
        inputs=patched,
        result=compose_policy(patched),
    )
    with pytest.raises(PolicySizeLimitError, match="65536") as excinfo:
        build_manifest(snapshot)
    assert "universal fragment" in str(excinfo.value)


def test_missing_required_content_fails_actionably(tmp_path):
    inputs = _inputs(tmp_path)
    patched = replace(
        inputs, identity=ResolvedResource(ResourceKind.IDENTITY, "empty", "user", "")
    )
    snapshot = PolicySnapshot(
        identity_selection="user:empty",
        workflow_selection="minimal",
        strategy_selection="none",
        routes=shipped_routes(),
        inputs=patched,
        result=compose_policy(patched),
    )
    with pytest.raises(ManifestError, match="missing required content"):
        build_manifest(snapshot)


def test_incoherent_selection_and_resolution_rejected(tmp_path):
    inputs = _inputs(tmp_path)
    patched = replace(
        inputs,
        workflow=ResolvedResource(
            ResourceKind.WORKFLOW, "ceremony", "builtin", "ceremony body"
        ),
    )
    snapshot = PolicySnapshot(
        identity_selection="coder",
        workflow_selection="minimal",
        strategy_selection="none",
        routes=shipped_routes(),
        inputs=patched,
        result=compose_policy(patched),
    )
    with pytest.raises(ManifestError, match="incoherent selection"):
        build_manifest(snapshot)
