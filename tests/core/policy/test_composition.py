"""The nine-step composition, AGENTS provenance dedup, and the S8 lint."""

from __future__ import annotations

import inspect
from pathlib import Path
import re

import pytest

from chartreux.core.agents import AgentManager
from chartreux.core.agents.identities import Delegation, get_builtin_identity
from chartreux.core.config.agent_selection import AgentSelectionConfig
from chartreux.core.config.harness_files import (
    FileSource,
    HarnessFilesManager,
    get_harness_files_manager,
)
from chartreux.core.config.harness_files._harness_manager import InstructionDocument
from chartreux.core.dispatch.lint import (
    RESIDUAL_CEREMONY_ANCHORS,
    UNIVERSAL_ANCHORS,
    lint_composed,
)
from chartreux.core.dispatch.renderer import (
    DispatchRenderError,
    render_composition_regions,
    render_launch_target_region,
    render_task_description_for_composition,
    task_skeleton,
)
from chartreux.core.dispatch.routes import shipped_routes
from chartreux.core.policy.composition import (
    COMPOSITION_STEPS,
    CONTEXT_FRAGMENTS,
    CompositionInputs,
    ExecutionContext,
    compose_policy,
    deduplicate_agents_documents,
    fragment_loader,
    policy_resource_loader,
    render_agents_section,
    resolve_identity_facts,
    resolve_policy_resources,
    user_resource_directories,
)
from chartreux.core.policy.models import ResourceKind
from chartreux.core.skills.manager import SkillManager
from chartreux.core.system_prompt import (
    ProjectContextProvider,
    get_universal_system_prompt,
)
from chartreux.core.trusted_folders import TrustedFoldersManager
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

PINNED_DATE = "2000-01-01 (Saturday)"
SESSION_MODEL = "beta-model"
CODER_OPENING = "You are Chartreux, an independently maintained CLI coding agent."
ORCHESTRATED = AgentSelectionConfig(
    identity="orchestrator",
    workflow="ceremony",
    orchestration_strategy="builtin:default",
)


def _fragment(name: str) -> str:
    resolved = fragment_loader().resolve(ResourceKind.IDENTITY, name)
    assert resolved.content is not None
    return resolved.content


def _document(
    source: FileSource, path: str, canonical: str, content: str
) -> InstructionDocument:
    return InstructionDocument(source, Path(path), Path(canonical), content)


def _inputs(
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
    launch_targets: str | None = None,
    regions: dict[str, str] | None = None,
    user_dirs: dict[ResourceKind, Path] | None = None,
) -> CompositionInputs:
    """Build composition inputs from the shipped bodies and route facts."""
    harness = get_harness_files_manager()
    dirs = user_dirs if user_dirs is not None else user_resource_directories(harness)
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


def _session_prompt(config, **kwargs):
    return get_universal_system_prompt(
        config,
        SkillManager(lambda: config),
        AgentManager(FakeConfigOrchestrator(config)),
        **kwargs,
    )


# --- Nine-step ordering (design §5) ----------------------------------------


def test_nine_step_order_opening_at_character_zero_reanchor_last() -> None:
    documents = (
        _document("user", "/u/AGENTS.md", "/shared/AGENTS.md", "shared marker"),
    )
    result = compose_policy(
        _inputs(strategy="builtin:default", agents_documents=documents)
    )
    # The full selection realizes every step, in the design's fixed order.
    assert [step.name for step in result.steps] == list(COMPOSITION_STEPS)
    # The steps tile the served artifact in order.
    assert result.text == "\n\n".join(step.content for step in result.steps)
    # Step 1: the identity opening sits at CHARACTER ZERO, with the date.
    assert result.text.startswith(CODER_OPENING)
    assert result.steps[0].name == "identity_opening"
    assert f"Today's date is {PINNED_DATE}." in result.steps[0].content
    # Step 9: the closing re-anchor is last and adds no new rules.
    assert result.text.rstrip().endswith("this closing note adds no new rules.")
    # The order is positional, not just nominal.
    positions = [
        result.text.index(marker)
        for marker in (
            CODER_OPENING,  # 1 identity opening
            "## Instruction hierarchy",  # 2 universal instructions
            "## Coder identity",  # 3 identity behavior
            "## Execution context — interactive main session",  # 4 context
            "# Workflow: minimal",  # 5 workflow
            "# Orchestration strategy: default",  # 6 strategy
            "## Tool semantics",  # 7 shared references
            "Codebase and user instructions are shown below",  # 8 AGENTS
            "## Resolved contract",  # 9 closing re-anchor
        )
    ]
    assert positions == sorted(positions)


def test_strategy_none_omits_the_strategy_step() -> None:
    result = compose_policy(_inputs(strategy="none"))
    assert "orchestration_strategy" not in [s.name for s in result.steps]
    assert result.strategy.id == "none"
    assert result.strategy.content is None
    assert "# Orchestration strategy" not in result.text
    # The re-anchor names the selection honestly: no strategy.
    assert "and no orchestration strategy." in result.text


def test_strategy_selected_adds_the_strategy_step_after_the_workflow() -> None:
    result = compose_policy(_inputs(strategy="builtin:default"))
    names = [step.name for step in result.steps]
    assert names.index("orchestration_strategy") == names.index("workflow") + 1
    assert "# Orchestration strategy: default" in result.text
    assert "and the `default` orchestration strategy." in result.text


def test_bounded_child_skips_workflow_and_strategy() -> None:
    result = compose_policy(
        _inputs(
            strategy="builtin:default", execution_context=ExecutionContext.BOUNDED_CHILD
        )
    )
    names = [step.name for step in result.steps]
    assert "workflow" not in names
    assert "orchestration_strategy" not in names
    assert "## Execution context — bounded child" in result.text
    assert "The parent owns the workflow." in result.text
    assert "# Workflow:" not in result.text
    assert "# Orchestration strategy" not in result.text
    assert (
        "as a bounded child, with no workflow or orchestration strategy of its own"
        in result.text
    )


def test_headless_main_uses_the_headless_fragment() -> None:
    result = compose_policy(_inputs(execution_context=ExecutionContext.HEADLESS_MAIN))
    assert "## Execution context — headless main session" in result.text
    assert "non-interactive (headless) session" in result.text
    assert "## Execution context — interactive main session" not in result.text


def test_instruction_overlay_rides_in_step_three() -> None:
    result = compose_policy(_inputs(instruction_overlay="Launch overlay."))
    behavior = next(s for s in result.steps if s.name == "identity_behavior")
    assert "Launch overlay." in behavior.content
    assert result.text.index("Launch overlay.") < result.text.index(
        "## Execution context"
    )


def test_policy_text_excludes_session_content() -> None:
    # The S8-linted policy text is harness-owned: the AGENTS layers and the
    # optional detail sections are session content and never linted.
    documents = (
        _document("project", "/p/AGENTS.md", "/p/AGENTS.md", "project marker"),
    )
    result = compose_policy(
        _inputs(
            agents_documents=documents,
            model_info="Your model name is: `beta-model`",
            skills="# Available Skills",
            scratchpad="# Scratchpad Directory",
            environment="The operating system is Linux",
            project_context="# Project context",
            commit_signature="git commit -m",
        )
    )
    for marker in (
        "Your model name is: `beta-model`",
        "# Available Skills",
        "# Scratchpad Directory",
        "The operating system is Linux",
        "# Project context",
        "git commit -m",
        "project marker",
    ):
        assert marker in result.text
        assert marker not in result.policy_text
    # The harness-owned shared references are policy.
    for marker in ("## Tool semantics", "### Launch targets", "## Communication"):
        assert marker in result.policy_text


# --- Placeholder contract (package-5 bodies) -------------------------------


def test_orchestrator_body_splices_every_cli_placeholder() -> None:
    result = compose_policy(_inputs(identity="orchestrator", workflow="ceremony"))
    assert "$dispatch" not in result.text
    assert "$dispatch" not in result.policy_text
    # The orchestrator identity body carries the delegation-posture regions.
    assert "Before planning, read named instruction and routing files" in (result.text)
    assert "You are an orchestrator, not the implementor." in result.text
    assert "Use these tools directly for orchestration only:" in result.text
    assert "Every task brief must carry the worker disclosure requirement" in (
        result.text
    )
    assert "Route by purpose: the launch-target table" in result.text
    # The ceremony workflow carries the implement placeholder.
    assert "never edit repo files yourself" in result.text
    # The launch-target table is a shared reference (step 7), not only a
    # routing-region concern.
    assert "### Launch targets" in result.text


def test_coder_body_carries_no_placeholder_regions() -> None:
    # The coder body carries no $dispatch_* tokens: the direct-tool posture
    # line (step 3) is the composition-rendered posture, and the regions
    # splice only where a body carries the placeholder contract.
    result = compose_policy(_inputs())
    assert "$dispatch" not in result.text
    assert "**Delegation posture.** Delegation is optional:" in result.text
    assert "Direct tools: the full granted tool set." in result.text
    assert "You may implement directly" not in result.text
    assert "never edit repo files yourself" not in result.text


def test_unresolved_placeholder_in_custom_identity_fails_closed(tmp_path: Path) -> None:
    user_dir = tmp_path / "prompts"
    user_dir.mkdir()
    (user_dir / "custom.md").write_text(
        "You are a custom identity.\n\n## Custom behavior\n\n$dispatch_reuse\n",
        encoding="utf-8",
    )
    inputs = _inputs(
        identity="user:custom",
        user_dirs={
            ResourceKind.IDENTITY: user_dir,
            ResourceKind.WORKFLOW: tmp_path / "no-workflows",
            ResourceKind.STRATEGY: tmp_path / "no-strategies",
        },
    )
    with pytest.raises(DispatchRenderError, match=r"\$dispatch_reuse"):
        compose_policy(inputs)


def test_custom_identity_composes_with_universal_and_no_posture_line(
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "prompts"
    user_dir.mkdir()
    (user_dir / "custom.md").write_text(
        "You are the custom specialist.\n\n## Custom behavior\n\nDo custom work.\n",
        encoding="utf-8",
    )
    result = compose_policy(
        _inputs(
            identity="user:custom",
            user_dirs={
                ResourceKind.IDENTITY: user_dir,
                ResourceKind.WORKFLOW: tmp_path / "no-workflows",
                ResourceKind.STRATEGY: tmp_path / "no-strategies",
            },
        )
    )
    assert result.text.startswith("You are the custom specialist.")
    # The universal fragment is mandatory for every identity.
    assert "## Instruction hierarchy" in result.text
    assert "## Critical instructions — not overridable" in result.text
    assert "## Custom behavior" in result.text
    # A custom identity has no harness-known posture: no posture line, no
    # delegation-obligation anchor for the lint to require.
    assert result.identity_facts is None
    assert "**Delegation posture.**" not in result.text
    assert (
        lint_composed(result.policy_text, identity_facts=None, workflow="minimal") == ()
    )


def test_composition_regions_are_keyed_on_typed_delegation() -> None:
    required = render_composition_regions(delegation=Delegation.REQUIRED)
    optional = render_composition_regions(delegation=Delegation.OPTIONAL)
    custom = render_composition_regions(delegation=None)
    # The delegation-required identity gets the delegation-posture regions.
    assert "never edit repo files yourself" in required["implement"]
    assert "You are an orchestrator, not the implementor." in required["identity"]
    assert "Use these tools directly for orchestration only:" in required["tools"]
    # Every other identity — including custom identities — gets the
    # direct-execution regions.
    for regions in (optional, custom):
        assert "You may implement directly" in regions["implement"]
        assert "You execute directly: you are the implementor" in (regions["identity"])
        assert "Use your full tool set directly:" in regions["tools"]
    # The disclosure and routing prose are delegation-independent.
    for regions in (required, optional, custom):
        assert "worker disclosure requirement" in regions["disclosure"]
        assert "Route by purpose:" in regions["routing"]


def test_launch_target_region_renders_the_route_facts() -> None:
    region = render_launch_target_region(dict(shipped_routes()), "beta-model")
    assert "### Launch targets" in region
    assert "| Purpose | Identity | Tier | Seat | Model |" in region
    # Every shipped default target binds the session model: the $session
    # sentinel never leaks into the rendered table.
    assert "`beta-model`" in region
    assert "$session" not in region
    assert "| `implementation` | `coder` | substantive | — | `beta-model` |" in (region)
    assert (
        "| `review.deep` | `reviewer` | substantive | analytical | `beta-model` |"
        in region
    )
    assert "| `review.deep` | `coder` | substantive | execution | `beta-model` |" in (
        region
    )
    # Unmet model-diversity requirements are reported, never faked.
    assert "### Route requirements" in region
    assert "`review.deep`: 3 distinct canonical models" in region
    # The purpose meanings come from the shipped vocabulary.
    assert "### Purpose meanings" in region
    assert "- `implementation`: All substantive implementation" in region


def test_launch_target_region_substitutes_the_session_model() -> None:
    region = render_launch_target_region(dict(shipped_routes()), "solo-model")
    assert "`solo-model`" in region
    assert "`beta-model`" not in region


def test_task_description_composes_from_route_facts() -> None:
    task = render_task_description_for_composition(
        routes=dict(shipped_routes()),
        session_model="beta-model",
        skeleton=task_skeleton(),
    )
    assert "$dispatch" not in task
    assert "Every task brief must carry the worker disclosure requirement" in task
    # The task-side routing region is the launch-target table.
    assert "### Launch targets" in task
    assert "| `implementation` | `coder` | substantive | — | `beta-model` |" in (task)
    # The reuse and review regions render from the composition scaffolding.
    assert "To continue genuine work on the same feature" in task
    assert "For an independent review, launch a fresh reviewer" in task


# --- AGENTS provenance dedup (design §5 step 8; findings item 4) -----------


def test_canonical_agents_duplicate_renders_once_project_label_wins() -> None:
    user_doc = _document(
        "user", "/u/.chartreux/AGENTS.md", "/shared/AGENTS.md", "shared marker"
    )
    project_doc = _document(
        "project", "/repo/AGENTS.md", "/shared/AGENTS.md", "shared marker"
    )
    provenance = deduplicate_agents_documents((user_doc, project_doc))
    assert len(provenance) == 1
    entry = provenance[0]
    # BOTH provenance positions are retained, in discovery order.
    assert entry.sources == ("user", "project")
    # The project label wins.
    assert entry.path == Path("/repo/AGENTS.md")
    assert entry.canonical_path == Path("/shared/AGENTS.md")
    # The composition renders the document once and keeps the canonical id
    # in the meta tool's read list.
    result = compose_policy(_inputs(agents_documents=(user_doc, project_doc)))
    assert result.text.count("shared marker") == 1
    assert "## Project instructions (checked into the codebase)" in result.text
    assert "also loaded as the user-level instructions" in result.text
    assert "## User instructions" not in result.text
    assert result.instruction_read_files == frozenset({Path("/shared/AGENTS.md")})
    assert result.agents_provenance == provenance


def test_distinct_agents_documents_render_in_discovery_order() -> None:
    user_doc = _document("user", "/u/AGENTS.md", "/u/AGENTS.md", "user marker")
    outer = _document("project", "/repo/AGENTS.md", "/repo/AGENTS.md", "outer m")
    inner = _document(
        "project", "/repo/sub/AGENTS.md", "/repo/sub/AGENTS.md", "inner m"
    )
    result = compose_policy(_inputs(agents_documents=(user_doc, outer, inner)))
    assert [entry.sources for entry in result.agents_provenance] == [
        ("user",),
        ("project",),
        ("project",),
    ]
    assert result.instruction_read_files == frozenset({
        Path("/u/AGENTS.md"),
        Path("/repo/AGENTS.md"),
        Path("/repo/sub/AGENTS.md"),
    })
    # Discovery order is preserved in the rendered section.
    assert (
        result.text.index("user marker")
        < result.text.index("outer m")
        < result.text.index("inner m")
    )
    section = render_agents_section(result.agents_provenance)
    assert "## User instructions" in section
    assert section.count("## Project instructions (checked into the codebase)") == 1


def test_agents_section_uses_the_shipped_template() -> None:
    provenance = deduplicate_agents_documents((
        _document("user", "/u/AGENTS.md", "/u/AGENTS.md", "user marker"),
    ))
    section = render_agents_section(provenance)
    assert section.startswith("Codebase and user instructions are shown below.")
    assert "IMPORTANT: this context may or may not be relevant to your tasks." in (
        section
    )
    assert "user marker" in section


def test_canonical_duplicate_keeps_one_read_file_entry(
    tmp_path: Path, config_dir: Path
) -> None:
    # The user AGENTS.md and the project AGENTS.md resolve to the same
    # canonical file: the composition renders it once and the meta tool's
    # read list keeps the single canonical identity.
    project = tmp_path / "workspace"
    project.mkdir()
    user_doc = config_dir / "AGENTS.md"
    user_doc.write_text("shared marker", encoding="utf-8")
    (project / "AGENTS.md").symlink_to(user_doc)
    trust = TrustedFoldersManager()
    trust.trust_for_session(project)
    harness = HarnessFilesManager(
        sources=("user", "project"), cwd=project, trust_store=trust
    )
    config = build_test_vibe_config(include_project_context=True)
    prompt = get_universal_system_prompt(
        config,
        SkillManager(lambda: config, harness_files=harness),
        AgentManager(FakeConfigOrchestrator(config), harness_files=harness),
        cwd=project,
        harness_files=harness,
    )
    assert prompt.count("shared marker") == 1
    assert "also loaded as the user-level instructions" in prompt
    assert prompt.instruction_read_files == frozenset({user_doc.resolve()})


def test_provenance_uses_the_same_documents_as_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Provenance is collected from the very instruction documents the
    # composition renders: rewriting a document after loading changes
    # neither the rendered text nor the read list.
    project = tmp_path / "workspace"
    project.mkdir()
    (project / "AGENTS.md").write_text("project marker", encoding="utf-8")
    trust = TrustedFoldersManager()
    trust.trust_for_session(project)
    harness = HarnessFilesManager(sources=("project",), cwd=project, trust_store=trust)
    original = HarnessFilesManager.load_instruction_documents
    loads = []

    def load(self):
        documents = original(self)
        loads.append(documents)
        for doc in documents:
            doc.path.write_text("changed after loading")
        return documents

    monkeypatch.setattr(HarnessFilesManager, "load_instruction_documents", load)
    config = build_test_vibe_config(include_project_context=True)
    prompt = get_universal_system_prompt(
        config,
        SkillManager(lambda: config, harness_files=harness),
        AgentManager(FakeConfigOrchestrator(config), harness_files=harness),
        cwd=project,
        harness_files=harness,
    )
    assert len(loads) == 1
    assert "project marker" in prompt
    assert "changed after loading" not in prompt
    assert prompt.instruction_read_files == frozenset({
        (project / "AGENTS.md").resolve()
    })


# --- Optional-detail settings and the session-scoped git cache -------------


@pytest.mark.parametrize(
    "flags",
    [
        {
            "include_model_info": True,
            "include_prompt_detail": True,
            "include_project_context": True,
            "include_commit_signature": True,
        },
        {
            "include_model_info": False,
            "include_prompt_detail": False,
            "include_project_context": False,
            "include_commit_signature": False,
        },
    ],
    ids=["all-detail", "no-detail"],
)
def test_universal_sections_survive_optional_detail_settings(flags) -> None:
    # Suppressing optional detail never drops universal/critical content.
    config = build_test_vibe_config(**flags)
    prompt = _session_prompt(config)
    for anchor in UNIVERSAL_ANCHORS:
        assert anchor in prompt.lower()
    assert prompt.startswith(CODER_OPENING)
    assert prompt.rstrip().endswith("this closing note adds no new rules.")
    assert "$dispatch" not in prompt


def test_git_context_cache_is_session_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "workspace"
    project.mkdir()
    calls = []

    def fake_fetch(self) -> str:
        calls.append(self.root_path)
        return "Current branch: main"

    monkeypatch.setattr(ProjectContextProvider, "_fetch_git_status", fake_fetch)
    config = build_test_vibe_config(include_project_context=True)

    def render(harness: HarnessFilesManager) -> str:
        return str(
            get_universal_system_prompt(
                config,
                SkillManager(lambda: config, harness_files=harness),
                AgentManager(FakeConfigOrchestrator(config), harness_files=harness),
                cwd=project,
                harness_files=harness,
            )
        )

    first_session = HarnessFilesManager(sources=("user", "project"), cwd=project)
    render(first_session)
    render(first_session)
    # One fetch per session, shared across renders in that session.
    assert len(calls) == 1
    # A new session starts fresh.
    second_session = HarnessFilesManager(sources=("user", "project"), cwd=project)
    render(second_session)
    assert len(calls) == 2


# --- S8: the composed-artifact lint (design §5) -----------------------------


def test_s8_lint_is_keyed_on_identity_and_workflow_not_dispatch_mode() -> None:
    signature = inspect.signature(lint_composed)
    assert set(signature.parameters) == {"text", "identity_facts", "workflow"}


def test_s8_accepts_every_shipped_selection() -> None:
    for identity in ("coder", "orchestrator", "researcher", "reviewer", "advisor"):
        for workflow in ("minimal", "ceremony"):
            for strategy in ("none", "builtin:default"):
                result = compose_policy(
                    _inputs(identity=identity, workflow=workflow, strategy=strategy)
                )
                assert "$dispatch" not in result.text
                assert (
                    lint_composed(
                        result.policy_text,
                        identity_facts=result.identity_facts,
                        workflow=result.workflow.id,
                    )
                    == ()
                ), (identity, workflow, strategy)


def test_truthful_reporting_anchor_survives_every_workflow() -> None:
    # The self-verification prohibition is universal: it holds irrespective
    # of the workflow axis.
    for workflow in ("minimal", "ceremony"):
        result = compose_policy(_inputs(identity="orchestrator", workflow=workflow))
        normalized = result.policy_text.lower()
        assert "never claim a check you did not run" in normalized
        assert (
            'do not claim "verified", "tested", "working", or "complete"' in normalized
        )
        assert (
            lint_composed(
                result.policy_text,
                identity_facts=result.identity_facts,
                workflow=result.workflow.id,
            )
            == ()
        )


@pytest.mark.parametrize("anchor", UNIVERSAL_ANCHORS)
def test_s8_rejects_a_composition_missing_each_universal_anchor(anchor: str) -> None:
    result = compose_policy(_inputs())
    mutated = re.sub(
        re.escape(anchor), "redacted", result.policy_text, flags=re.IGNORECASE
    )
    assert anchor not in mutated.lower()
    diagnostics = lint_composed(
        mutated, identity_facts=result.identity_facts, workflow="minimal"
    )
    assert "S8" in {item.id for item in diagnostics}


def test_s8_rejects_orchestrator_missing_its_delegation_obligation() -> None:
    result = compose_policy(_inputs(identity="orchestrator", workflow="ceremony"))
    mutated = re.sub(
        r"delegation is required", "redacted", result.policy_text, flags=re.IGNORECASE
    )
    assert "delegation is required" not in mutated.lower()
    diagnostics = lint_composed(
        mutated, identity_facts=result.identity_facts, workflow="ceremony"
    )
    assert "S8" in {item.id for item in diagnostics}


def test_s8_rejects_ceremony_missing_delegated_verification() -> None:
    result = compose_policy(_inputs(identity="orchestrator", workflow="ceremony"))
    anchor = "never run tests, builds, or other verification yourself"
    mutated = re.sub(
        re.escape(anchor), "redacted", result.policy_text, flags=re.IGNORECASE
    )
    assert anchor not in mutated.lower()
    diagnostics = lint_composed(
        mutated, identity_facts=result.identity_facts, workflow="ceremony"
    )
    assert "S8" in {item.id for item in diagnostics}


@pytest.mark.parametrize("anchor", RESIDUAL_CEREMONY_ANCHORS)
def test_s8_rejects_residual_gate_language_in_a_minimal_composition(
    anchor: str,
) -> None:
    result = compose_policy(_inputs())
    assert anchor not in result.policy_text.lower()
    injected = f"{result.policy_text}\n\n{anchor}"
    diagnostics = lint_composed(
        injected, identity_facts=result.identity_facts, workflow="minimal"
    )
    assert "S8" in {item.id for item in diagnostics}


# --- Config-driven selections ----------------------------------------------


def test_resolve_policy_resources_reads_the_agent_selection(config_dir: Path) -> None:
    # A custom identity resolves from the user prompts directory, no facts.
    user_prompts = config_dir / "prompts"
    user_prompts.mkdir()
    (user_prompts / "custom.md").write_text(
        "You are the custom specialist.\n", encoding="utf-8"
    )
    harness = get_harness_files_manager()
    loader = policy_resource_loader(user_resource_directories(harness))
    # The shipped default selection is coder + minimal + none.
    config = build_test_vibe_config()
    identity, workflow, strategy, facts = resolve_policy_resources(config, loader)
    assert (identity.id, identity.source) == ("coder", "builtin")
    assert workflow.id == "minimal"
    assert strategy.id == "none" and strategy.content is None
    assert facts is not None and facts.delegation is Delegation.OPTIONAL
    # The orchestrator selection resolves its typed facts.
    identity, workflow, strategy, facts = resolve_policy_resources(
        build_test_vibe_config(agent=ORCHESTRATED), loader
    )
    assert identity.id == "orchestrator"
    assert workflow.id == "ceremony"
    assert strategy.id == "default" and strategy.content is not None
    assert facts is not None and facts.delegation is Delegation.REQUIRED
    identity, workflow, strategy, facts = resolve_policy_resources(
        build_test_vibe_config(agent=AgentSelectionConfig(identity="user:custom")),
        loader,
    )
    assert (identity.id, identity.source) == ("custom", "user")
    assert identity.content is not None
    assert facts is None


def test_resolve_identity_facts_tracks_the_builtin_identities() -> None:
    for name in ("coder", "orchestrator", "researcher", "reviewer", "advisor"):
        facts = resolve_identity_facts(name)
        assert facts is not None
        assert facts.id == name
        assert facts == get_builtin_identity(name)
    assert resolve_identity_facts("builtin:coder") == get_builtin_identity("coder")
    assert resolve_identity_facts("user:custom") is None


def test_get_universal_system_prompt_serves_the_composed_artifact() -> None:
    config = build_test_vibe_config()
    prompt = _session_prompt(config)
    # The served prompt is the nine-step composition: opening at character
    # zero, universal fragment, identity behavior, execution context,
    # workflow, shared references, and the closing re-anchor last.
    assert prompt.startswith(CODER_OPENING)
    assert "## Instruction hierarchy" in prompt
    assert "## Coder identity" in prompt
    assert "## Execution context — interactive main session" in prompt
    assert "# Workflow: minimal" in prompt
    assert "## Tool semantics" in prompt
    assert "### Launch targets" in prompt
    assert "## Communication" in prompt
    assert "Your model name is:" in prompt
    assert prompt.rstrip().endswith("this closing note adds no new rules.")
    assert "$dispatch" not in prompt
    # The launch-target table binds the session model of the config's catalog.
    alias = config.get_active_model().alias
    assert f"| `implementation` | `coder` | substantive | — | `{alias}` |" in (prompt)


def test_get_universal_system_prompt_serves_the_orchestrated_selection() -> None:
    config = build_test_vibe_config(agent=ORCHESTRATED)
    prompt = _session_prompt(config)
    assert prompt.startswith("You are the orchestrator:")
    assert "## Orchestrator identity" in prompt
    assert "# Workflow: ceremony" in prompt
    assert "# Orchestration strategy: default" in prompt
    assert "## Opening gate" in prompt
    assert "never edit repo files yourself" in prompt
    assert "never run tests, builds, or other verification yourself" in prompt
    assert "Every task brief must carry the worker disclosure requirement" in (prompt)
    assert prompt.rstrip().endswith("this closing note adds no new rules.")
    assert "$dispatch" not in prompt


def test_get_universal_system_prompt_serves_a_bounded_child() -> None:
    config = build_test_vibe_config()
    prompt = _session_prompt(config, is_subagent=True)
    assert "## Execution context — bounded child" in prompt
    assert "Do not spawn, delegate to, or coordinate nested subagents." in prompt
    # Children never inherit the parent's workflow or strategy bodies.
    assert "# Workflow:" not in prompt
    assert "# Orchestration strategy" not in prompt
    # The model catalog section is omitted for children.
    assert "# Model catalog" not in prompt
    assert "Your model name is:" in prompt
    assert prompt.rstrip().endswith("this closing note adds no new rules.")


def test_get_universal_system_prompt_serves_headless_main() -> None:
    config = build_test_vibe_config()
    prompt = _session_prompt(config, headless=True)
    assert "## Execution context — headless main session" in prompt
    assert "non-interactive (headless) session" in prompt
    assert "## Execution context — interactive main session" not in prompt


def test_role_instructions_ride_as_the_per_launch_overlay() -> None:
    config = build_test_vibe_config()
    prompt = _session_prompt(config, role_instructions="Review with care.")
    assert "Review with care." in prompt
    assert prompt.index("Review with care.") < prompt.index("## Execution context")
    # The overlay suppresses the model catalog section.
    assert "# Model catalog" not in prompt
