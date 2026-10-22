"""Deterministic nine-step prompt composition (design §5).

The composition consumes the package-5 resource bodies (identities,
fragments, workflows, orchestration strategies) resolved through the
package-1 strict loader, the typed identity facts (design §3), and the
package-4 route facts, and assembles them in the design's fixed order:

1. identity opening at character zero (short, direct self-policy)
2. universal critical instructions + instruction hierarchy (mandatory)
3. identity behavior + direct-tool posture (+ per-launch overlay)
4. execution context (interactive main / headless main / bounded child)
5. selected workflow body (absent for bounded children)
6. selected orchestration strategy body (only if selected; ``none`` = absent)
7. shared references (tool semantics, launch-target table, catalogs,
   communication, scratchpad, environment)
8. AGENTS.md layers — discovery unchanged, canonical dedup at composition
9. short closing re-anchor (no new rules)

Deduplication is composition-only: the source documents are never edited.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from chartreux.core.agents.identities import (
    AgentIdentity,
    Delegation,
    ToolPosture,
    get_builtin_identity,
)
from chartreux.core.dispatch.renderer import splice_composition_regions
from chartreux.core.paths import CHARTREUX_HOME
from chartreux.core.policy.models import ResolvedResource, ResourceKind
from chartreux.core.policy.resources import ResourceLoader
from chartreux.core.prompts import PROMPTS_DIR, UtilityPrompt

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema
    from chartreux.core.config.harness_files import HarnessFilesManager
    from chartreux.core.config.harness_files._harness_manager import InstructionDocument

IDENTITIES_DIR = PROMPTS_DIR / "identities"
FRAGMENTS_DIR = PROMPTS_DIR / "fragments"
WORKFLOWS_DIR = PROMPTS_DIR / "workflows"
STRATEGIES_DIR = PROMPTS_DIR / "orchestration_strategies"

IDENTITY_IDS: tuple[str, ...] = (
    "coder",
    "orchestrator",
    "researcher",
    "reviewer",
    "advisor",
)
WORKFLOW_IDS: tuple[str, ...] = ("minimal", "ceremony")
STRATEGY_IDS: tuple[str, ...] = ("default",)
FRAGMENT_IDS: tuple[str, ...] = (
    "universal",
    "communication",
    "interactive_main",
    "headless_main",
    "bounded_child",
    "tool_semantics",
)

# A user-resource directory that never exists: user-qualified custom
# resources fail to resolve instead of falling back to a real directory.
_MISSING_USER_DIR = Path("/nonexistent-chartreux-user-resources")


class ExecutionContext(StrEnum):
    """The design §5 step-4 execution context of the composed session."""

    INTERACTIVE_MAIN = "interactive_main"
    HEADLESS_MAIN = "headless_main"
    BOUNDED_CHILD = "bounded_child"


CONTEXT_FRAGMENTS: Mapping[ExecutionContext, str] = {
    ExecutionContext.INTERACTIVE_MAIN: "interactive_main",
    ExecutionContext.HEADLESS_MAIN: "headless_main",
    ExecutionContext.BOUNDED_CHILD: "bounded_child",
}

COMPOSITION_STEPS: tuple[str, ...] = (
    "identity_opening",
    "universal_instructions",
    "identity_behavior",
    "execution_context",
    "workflow",
    "orchestration_strategy",
    "shared_references",
    "agents_instructions",
    "closing_reanchor",
)


@dataclass(frozen=True)
class AgentsProvenance:
    """One rendered AGENTS.md layer and its provenance positions."""

    path: Path
    canonical_path: Path
    sources: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class CompositionStep:
    """One ordered step of the nine-step composition."""

    name: str
    source: str
    content: str


@dataclass(frozen=True)
class CompositionInputs:
    """Everything the composition needs, pre-rendered by its caller."""

    identity: ResolvedResource
    workflow: ResolvedResource
    strategy: ResolvedResource
    identity_facts: AgentIdentity | None
    execution_context: ExecutionContext
    universal: str
    context_fragment: str
    tool_semantics: str
    communication: str
    launch_targets: str
    regions: Mapping[str, str]
    model_info: str | None
    skills: str | None
    scratchpad: str | None
    environment: str | None
    project_context: str | None
    commit_signature: str | None
    instruction_overlay: str | None
    agents_documents: tuple[InstructionDocument, ...]
    current_date: str


@dataclass(frozen=True)
class CompositionResult:
    """The composed artifact with its step manifest and provenance."""

    text: str
    policy_text: str
    steps: tuple[CompositionStep, ...]
    instruction_read_files: frozenset[Path]
    agents_provenance: tuple[AgentsProvenance, ...]
    identity: ResolvedResource
    workflow: ResolvedResource
    strategy: ResolvedResource
    identity_facts: AgentIdentity | None
    execution_context: ExecutionContext


def split_identity_body(content: str) -> tuple[str, str]:
    """Split an identity body into its opening and behavior sections.

    The opening is the leading prose before the first top-level heading; the
    design pins it at character zero (design §5 step 1). The behavior section
    carries the body's ``$dispatch_*`` placeholders (step 3).
    """
    lines = content.strip().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("## "):
            return ("\n".join(lines[:index]).strip(), "\n".join(lines[index:]).strip())
    return content.strip(), ""


def deduplicate_agents_documents(
    documents: Iterable[InstructionDocument],
) -> tuple[AgentsProvenance, ...]:
    """Canonical-file dedup at composition (findings item 4; design §5 step 8).

    The source documents are never edited. When the same canonical file is
    discovered at more than one position (the user AGENTS.md and a project
    AGENTS.md resolving to the same file), it renders once: the surviving
    entry keeps the higher-precedence (project) label and the provenance
    record retains every position.
    """
    entries: dict[Path, AgentsProvenance] = {}
    order: list[Path] = []
    for document in documents:
        canonical = document.canonical_path
        existing = entries.get(canonical)
        if existing is None:
            entries[canonical] = AgentsProvenance(
                path=document.path,
                canonical_path=canonical,
                sources=(document.source,),
                content=document.content,
            )
            order.append(canonical)
            continue
        sources = tuple(dict.fromkeys((*existing.sources, document.source)))
        entries[canonical] = AgentsProvenance(
            path=document.path if document.source == "project" else existing.path,
            canonical_path=canonical,
            sources=sources,
            content=existing.content,
        )
    return tuple(entries[canonical] for canonical in order)


def render_agents_section(provenance: tuple[AgentsProvenance, ...]) -> str:
    """Render the deduplicated AGENTS layers (mechanism unchanged, §5 step 8)."""
    doc_sections: list[str] = []
    project_heading_added = False
    for entry in provenance:
        if "project" in entry.sources and "user" in entry.sources:
            if not project_heading_added:
                doc_sections.append(
                    "## Project instructions (checked into the codebase)"
                )
                project_heading_added = True
            doc_sections.append(
                f"Contents of {entry.path} (project-level instructions; also "
                f"loaded as the user-level instructions):\n\n{entry.content}"
            )
        elif entry.sources == ("user",):
            doc_sections.append(
                f"## User instructions\n\nContents of {entry.path} "
                f"(user-level instructions):\n\n{entry.content}"
            )
        else:
            if not project_heading_added:
                doc_sections.append(
                    "## Project instructions (checked into the codebase)"
                )
                project_heading_added = True
            doc_sections.append(f"Contents of {entry.path}:\n\n{entry.content}")
    template = UtilityPrompt.AGENTS_DOC.read()
    return Template(template).safe_substitute(sections="\n\n".join(doc_sections))


def delegation_posture_line(facts: AgentIdentity | None) -> str | None:
    """The typed direct-tool posture line (design §5 step 3; §3/§8 facts).

    Built-in identities carry harness-known posture; the composition renders
    it so the S8 delegation-obligation anchor is uniform across every
    built-in. Custom identities have no harness-known posture (design
    decision 8) and render no line.
    """
    if facts is None:
        return None
    if facts.delegation is Delegation.REQUIRED:
        return (
            "**Delegation posture.** Delegation is required: you perform the "
            "work by delegating it to child agents; you do not implement. "
            "Direct tools: non-mutating orchestration tools only."
        )
    if facts.delegation is Delegation.OPTIONAL:
        if facts.delegation_tools is ToolPosture.NON_MUTATING:
            return (
                "**Delegation posture.** Delegation is optional: you may "
                "pass non-mutating lookup work to a child; you stay "
                "read-only. Direct tools: non-mutating tools only."
            )
        return (
            "**Delegation posture.** Delegation is optional: you may delegate "
            "a bounded piece of work to a child and stay responsible for its "
            "result. Direct tools: the full granted tool set."
        )
    return (
        "**Delegation posture.** Delegation is not available: you perform "
        "all the work yourself. Direct tools: the full granted tool set."
    )


def closing_reanchor(inputs: CompositionInputs) -> str:
    """The short closing re-anchor (design §5 step 9): no new rules."""
    if inputs.execution_context is ExecutionContext.BOUNDED_CHILD:
        selection = (
            f"the `{inputs.identity.id}` identity as a bounded child, with no "
            "workflow or orchestration strategy of its own"
        )
    else:
        selection = (
            f"the `{inputs.identity.id}` identity with the "
            f"`{inputs.workflow.id}` workflow"
        )
        if inputs.strategy.content is not None:
            selection += f" and the `{inputs.strategy.id}` orchestration strategy"
        else:
            selection += " and no orchestration strategy"
    return (
        "## Resolved contract\n\n"
        f"This conversation runs {selection}. The sections above are the "
        "resolved contract for this session: follow them as written. The "
        "critical instructions remain in force, and this closing note adds no "
        "new rules."
    )


def _resource_source(resource: ResolvedResource) -> str:
    return f"{resource.kind.value}:{resource.id}"


def compose_policy(inputs: CompositionInputs) -> CompositionResult:  # noqa: PLR0915
    """Assemble the nine-step composition (design §5).

    ``text`` is the served artifact; ``policy_text`` is the harness-owned
    composition (the AGENTS layers and the optional detail sections are
    session content, never S8-linted) that the S8 lint validates.
    """
    opening, behavior = split_identity_body(inputs.identity.content or "")
    if behavior:
        behavior = splice_composition_regions(behavior, inputs.regions)

    steps: list[CompositionStep] = []
    policy_parts: list[str] = []

    # 1. Identity opening at character zero (+ the date injection).
    opening_text = opening
    if inputs.current_date:
        date_line = f"Today's date is {inputs.current_date}."
        opening_text = f"{opening}\n{date_line}" if opening else date_line
    steps.append(
        CompositionStep(
            "identity_opening", _resource_source(inputs.identity), opening_text
        )
    )
    policy_parts.append(opening_text)

    # 2. Universal critical instructions + instruction hierarchy.
    steps.append(
        CompositionStep(
            "universal_instructions", "fragment:universal", inputs.universal
        )
    )
    policy_parts.append(inputs.universal)

    # 3. Identity behavior + direct-tool posture (+ per-launch overlay).
    behavior_parts = [
        part
        for part in (
            behavior,
            delegation_posture_line(inputs.identity_facts),
            inputs.instruction_overlay,
        )
        if part and part.strip()
    ]
    if behavior_parts:
        behavior_text = "\n\n".join(behavior_parts)
        steps.append(
            CompositionStep(
                "identity_behavior", _resource_source(inputs.identity), behavior_text
            )
        )
        policy_parts.append(behavior_text)

    # 4. Execution context.
    context_name = CONTEXT_FRAGMENTS[inputs.execution_context]
    steps.append(
        CompositionStep(
            "execution_context", f"fragment:{context_name}", inputs.context_fragment
        )
    )
    policy_parts.append(inputs.context_fragment)

    # 5-6. Workflow + strategy (children never inherit the parent's bodies).
    if inputs.execution_context is not ExecutionContext.BOUNDED_CHILD:
        workflow_text = inputs.workflow.content or ""
        if workflow_text:
            workflow_text = splice_composition_regions(workflow_text, inputs.regions)
            steps.append(
                CompositionStep(
                    "workflow", _resource_source(inputs.workflow), workflow_text
                )
            )
            policy_parts.append(workflow_text)
        if inputs.strategy.content is not None:
            steps.append(
                CompositionStep(
                    "orchestration_strategy",
                    _resource_source(inputs.strategy),
                    inputs.strategy.content,
                )
            )
            policy_parts.append(inputs.strategy.content)

    # 7. Shared references: harness-owned parts are always present; the
    # optional detail sections are session content (never S8-linted).
    shared_parts = [inputs.tool_semantics, inputs.launch_targets]
    if inputs.model_info:
        shared_parts.append(inputs.model_info)
    if inputs.skills:
        shared_parts.append(inputs.skills)
    shared_parts.append(inputs.communication)
    if inputs.scratchpad:
        shared_parts.append(inputs.scratchpad)
    if inputs.environment:
        shared_parts.append(inputs.environment)
    if inputs.commit_signature:
        shared_parts.append(inputs.commit_signature)
    if inputs.project_context:
        shared_parts.append(inputs.project_context)
    steps.append(
        CompositionStep(
            "shared_references", "composition:shared", "\n\n".join(shared_parts)
        )
    )
    policy_parts.append(
        "\n\n".join((
            inputs.tool_semantics,
            inputs.launch_targets,
            inputs.communication,
        ))
    )

    # 8. AGENTS.md layers (canonical dedup; provenance from the same
    # documents used for rendering).
    agents_provenance = deduplicate_agents_documents(inputs.agents_documents)
    if agents_provenance:
        steps.append(
            CompositionStep(
                "agents_instructions",
                "agents:documents",
                render_agents_section(agents_provenance),
            )
        )

    # 9. Short closing re-anchor, last.
    reanchor = closing_reanchor(inputs)
    steps.append(CompositionStep("closing_reanchor", "composition:reanchor", reanchor))
    policy_parts.append(reanchor)

    return CompositionResult(
        text="\n\n".join(step.content for step in steps),
        policy_text="\n\n".join(policy_parts),
        steps=tuple(steps),
        instruction_read_files=frozenset(
            entry.canonical_path for entry in agents_provenance
        ),
        agents_provenance=agents_provenance,
        identity=inputs.identity,
        workflow=inputs.workflow,
        strategy=inputs.strategy,
        identity_facts=inputs.identity_facts,
        execution_context=inputs.execution_context,
    )


def policy_resource_loader(user_dirs: Mapping[ResourceKind, Path]) -> ResourceLoader:
    """The one strict loader contract across identities, workflows, strategies."""
    return ResourceLoader(
        directories=dict(user_dirs),
        builtins={
            ResourceKind.IDENTITY: {
                name: IDENTITIES_DIR / f"{name}.md" for name in IDENTITY_IDS
            },
            ResourceKind.WORKFLOW: {
                name: WORKFLOWS_DIR / f"{name}.md" for name in WORKFLOW_IDS
            },
            ResourceKind.STRATEGY: {
                name: STRATEGIES_DIR / f"{name}.md" for name in STRATEGY_IDS
            },
        },
    )


def fragment_loader() -> ResourceLoader:
    """Fragments are harness-owned composition inputs, not selectable resources.

    They register as loader builtins so they resolve with builtin semantics
    and cannot be shadowed by user resources.
    """
    return ResourceLoader(
        {},
        {
            ResourceKind.IDENTITY: {
                name: FRAGMENTS_DIR / f"{name}.md" for name in FRAGMENT_IDS
            }
        },
    )


def user_resource_directories(
    harness_files: HarnessFilesManager,
) -> dict[ResourceKind, Path]:
    """The user resource directories for each selectable kind."""
    prompts = harness_files.user_prompts_dirs
    return {
        ResourceKind.IDENTITY: (prompts[0] if prompts else _MISSING_USER_DIR),
        ResourceKind.WORKFLOW: CHARTREUX_HOME.path / "workflows",
        ResourceKind.STRATEGY: CHARTREUX_HOME.path / "orchestration_strategies",
    }


def resolve_identity_facts(reference: str) -> AgentIdentity | None:
    """The typed facts of a built-in identity selection, else None."""
    if reference.startswith("user:"):
        return None
    return get_builtin_identity(reference.removeprefix("builtin:"))


def resolve_policy_resources(
    config: ChartreuxConfigSchema, loader: ResourceLoader
) -> tuple[ResolvedResource, ResolvedResource, ResolvedResource, AgentIdentity | None]:
    """Resolve the ``[agent]`` selections through the strict loader.

    Returns the identity, workflow, and strategy resources plus the typed
    identity facts. ``none`` resolves concretely without loading a strategy
    body, and a future ``delegation = none`` identity plus an explicit
    strategy fails on harness-known facts.
    """
    selection = config.agent
    identity = loader.resolve(ResourceKind.IDENTITY, selection.identity)
    workflow = loader.resolve(ResourceKind.WORKFLOW, selection.workflow)
    strategy = loader.resolve(ResourceKind.STRATEGY, selection.orchestration_strategy)
    facts = resolve_identity_facts(selection.identity)
    if facts is not None:
        loader.validate_strategy(facts.delegation, selection.orchestration_strategy)
    return identity, workflow, strategy, facts
