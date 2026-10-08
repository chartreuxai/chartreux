"""Rendered dispatch prompts: the mode-variable prompt regions.

The CLI prompt skeleton (``prompts/cli.md``) and the task tool skeleton
(``tools/builtins/prompts/task.md``) carry ``$dispatch_*`` placeholder lines
where their mode-variable paragraphs live. This module renders those regions
from a dispatch policy: the policy supplies the names (slots, profiles,
purposes) and the rule rows (the curated compatibility blocks), while the
imperative scaffolding sentences are fixed, developer-owned templates here.

Roster-aware rendering (contracts D/G): a single canonical model renders
task-kind sentences with no tier names, while a multi-model roster renders the
tier routing. The orchestrated multi-model rendering reproduces the legacy
routing bytes captured by the WP0 baseline goldens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import TYPE_CHECKING

from chartreux import CHARTREUX_ROOT
from chartreux.core.dispatch.lint import (
    RosterShape,
    lint_rendered,
    reject_errors,
    roster_shape,
)
from chartreux.core.dispatch.schema import DispatchMode, DispatchPolicy, DispatchSlot
from chartreux.utils.io import read_safe

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema
    from chartreux.core.model_catalog.schema import ModelCatalog

TASK_PROMPT_PATH = (
    CHARTREUX_ROOT / "core" / "tools" / "builtins" / "prompts" / "task.md"
)

# region name -> placeholder token, in skeleton order.
CLI_PLACEHOLDERS: Mapping[str, str] = {
    "reading": "$dispatch_reading",
    "implement": "$dispatch_implement",
    "identity": "$dispatch_identity",
    "tools": "$dispatch_tools",
    "routing": "$dispatch_routing",
}
TASK_PLACEHOLDERS: Mapping[str, str] = {
    "reuse": "$dispatch_reuse",
    "review": "$dispatch_review",
    "routing": "$dispatch_routing",
}
_PLACEHOLDER_PREFIX = "$dispatch_"


class DispatchRenderError(ValueError):
    """A skeleton could not be rendered completely."""


# --- Fixed developer-owned scaffolding: orchestrated mode -------------------

_READING_ORCHESTRATED = (
    "Before planning, read named instruction and routing files; delegate "
    "reading relevant source, tests, and entry points to a subagent. Before "
    "relying on an API or library function, have a subagent check its "
    "established use rather than guessing versions or signatures."
)

_IMPLEMENT_ORCHESTRATED = (
    "5. **Implement** the approved plan by dispatching bounded repo edits to "
    "subagents; never edit repo files yourself."
)

_IDENTITY_ORCHESTRATED = (
    "You are an orchestrator, not the implementor. Direct tool output grows "
    "your context: every file, search result, and shell output you inspect "
    "is re-sent on every subsequent API call. Delegate bounded specialist "
    "work with `task`; subagents read their own targets. Send intent and "
    "known constraints, not copied file contents. Parallelize independent, "
    "non-conflicting work."
)

_TOOLS_ORCHESTRATED = """Use these tools directly for orchestration only:
- `task` — primary dispatch tool; give each launch a self-contained task and concise `task_summary` (action plus subsystem).
- `read_file` — instruction files, user-named files needed for routing, or specific cited lines to check a result; not source investigation.
- `write_file` / `edit` — scratchpad only, never repo edits.
- `bash` — read-only orchestration metadata only, not exploration, tests, builds, or mutation. Delegate searches and verification instead of using direct `grep` or shell commands.
- `skill` and `todo` — procedures and task tracking.
- `web_search` / `web_fetch` — quick single-question lookups; delegate multi-step research.
- `check_agents`, `get_agent_result`, `wait_for_agent`, `cancel_agent`, and `release_agent` — background-agent lifecycle, not specialist work."""

# The role-semantics and background-work paragraphs are roster-independent
# orchestrated scaffolding; the multi-model rendering preserves their bytes.
_ROLE_SENTENCES = """An `@role` selects that role's one model and thinking level. If that pair is
unavailable, report the reason and repair the preset or credential; do not
silently choose another model. An explicit launch `config.thinking` overrides
the preset for that launch. To get independent reviews, launch separate tasks
with explicit presets or models. The retired `fan_out: true` flag is rejected."""

_BACKGROUND_SENTENCES = (
    "For background work, `check_agents` before reuse; re-task an idle agent "
    "only for a genuine continuation, with the current scope and intervening "
    "changes. Model, thinking, and tools can change on re-task, but persona "
    "(`instructions` / `system_prompt_id`), role, stack, and independent "
    "judgment require a new agent. Running agents are busy unless explicitly "
    "superseded with `task(agent_id=..., replace_run=True, background=True)`. "
    "Busy replacement keeps the conversation, automatically injects "
    "supersession framing, joins old cleanup before launching, and forbids "
    "config/profile changes. Check `launch_outcome`; stopping/finishing/reserved "
    "refusals are not launches. The acknowledgment's `metadata.replaced_run_id` "
    "and `metadata.replacement_run_id` identify both runs. Use "
    "`cancel_agent(agent_id, run_id)` to request a stop without replacement, "
    "then wait for the terminal result; stop acceptance is not completion and "
    "neither stop nor retask rolls back side effects. Retrieve results before "
    "release; idle retention is best-effort, and evicted agents must be "
    "recreated for further work."
)

# --- Fixed developer-owned scaffolding: standalone mode (contracts B) -------

_READING_STANDALONE = (
    "Before planning, read named instruction and routing files yourself; "
    "delegate reading relevant source, tests, and entry points to a subagent "
    "when the reading is broad. Before relying on an API or library "
    "function, have a subagent check its established use rather than "
    "guessing versions or signatures."
)

_IMPLEMENT_STANDALONE = (
    "5. **Implement** the approved plan. You may implement directly: make "
    "the approved repo edits yourself. Stay within the approved plan; change "
    "minimally."
)

_IDENTITY_STANDALONE = (
    "Verification stays delegated. Dispatch checks to a subagent, scaled to "
    "the change, then dispatch an independent review against the approved "
    "plan — a fresh reviewer, never the author. Never run tests, builds, or "
    "other verification yourself; never review your own work; never claim a "
    "check you did not run.\n\n"
    "Direct execution grows your context: everything you read or run is "
    "re-sent on every call. Keep reads targeted, bound shell output, move "
    "large artifacts to the scratchpad, and say when your context is getting "
    "long. When context is already long (including when the user says so), "
    "or a step would read or produce more than a few hundred lines, delegate "
    "that bounded piece to a worker before searching or reading source "
    "yourself; keep only its concise result in the main context. This "
    "context-hygiene requirement overrides optional delegation; keep the "
    "session in standalone mode."
)

_TOOLS_STANDALONE = (
    "write_file/edit — repo files within the approved scope; scratchpad for "
    "temporary artifacts. read_file — files the task names or cites. bash — "
    "orchestration metadata only, always with timeouts; delegate searches, "
    "exploration, tests, and builds."
)

_STANDALONE_ROUTING_SENTENCE = (
    "Delegation is optional. When you delegate, route by purpose through "
    "the configured slots and pass the tier explicitly; the slot table is "
    "authoritative. When you implement directly, the tier rules govern only "
    "work you delegate."
)

_GATE_SENTENCE = (
    "The opening gate and acceptance lattice remain unchanged. Saved policy "
    "changes apply next session."
)

_VALVE_SENTENCE = (
    "Delegate bounded work when context grows long without changing the session mode."
)

_REVIEW_SENTENCES = (
    "Use a fresh reviewer for every independent judgment; never reuse an "
    "advisor as a reviewer or give a second-round reviewer earlier "
    "conclusions. Keep an advisor across related design refinements."
)

# --- Fixed developer-owned scaffolding: the task tool routing region --------

_REUSE_PREFIX = (
    "To continue genuine work on the same feature, first use `check_agents` "
    "to inspect each idle agent's effective model, thinking, and retained "
    "context. Reuse an idle handle with `agent_id` and `background: true`; "
    "omit `agent_type` to retain its profile. Omit `config`, use `{}`, or "
    "omit nested fields to retain effective launch choices; supplied lists "
    "replace earlier launch-layer lists and supplied per-tool fields patch "
    "that tool. Re-task the same idle agent for a genuine continuation, with "
    "a different `model`, `thinking`, or tool configuration when the work "
    "calls for it. A retained agent keeps its last model and thinking unless "
    "overridden; "
)

_REUSE_SUFFIX = (
    "Persona is immutable on reuse: `instructions` and `system_prompt_id` may "
    "only repeat their existing effective values. Launch a new agent when its "
    "persona, role, or stack must change, or when work is unrelated or needs "
    "independent judgment. Include the current scope and intervening changes "
    "rather than restating its full context. A retained conversation is not "
    "proof that the working tree is unchanged. A running agent is busy; wait "
    "when its result is a dependency, and spawn another only for independent, "
    "non-conflicting work. An evicted agent cannot be reused; retrieve its "
    "retained result with `get_agent_result` if needed, then start a new "
    "agent."
)

_REVIEW_EXAMPLE_MULTI_MODEL = (
    'For example, launch a deep review with `config: {"model": "@large", '
    '"instructions": "Review carefully.", "enabled_tools": ["read_file"]}`. '
    "For a second-round deep review, launch a fresh reviewer with a `@large` "
    'model override — `task(agent_type="reviewer", task="Review the retry '
    'changes.", config={"model": "@large"})` — rather than re-tasking an idle '
    "implementor: a fresh reviewer must not inherit an implementor's persona "
    "or history."
)

_REVIEW_EXAMPLE_SINGLE_MODEL = (
    'For example, launch a deep review with `config: {"instructions": '
    '"Review carefully.", "enabled_tools": ["read_file"]}`. For a '
    "second-round deep review, launch a fresh reviewer on a deep-review "
    'slot — `task(agent_type="reviewer", task="Review the retry changes.")` '
    "— rather than re-tasking an idle implementor: a fresh reviewer must not "
    "inherit an implementor's persona or history."
)

# The isolated orchestrated compatibility block: the legacy task routing
# paragraph, preserved byte-identically for the multi-model fixture.
_TASK_ROUTING_MULTI_MODEL = (
    "Route by the work, not the profile default: pass the tier explicitly in "
    "`config`. Mechanical delegated work — single-file edits, bounded "
    "searches, verification runs — launches with a model override, for "
    'example `task(agent_type="worker", task="Rename add to plus in utils.py '
    'and update its call sites.", config={"model": "@small"})`. Substantive '
    "implementation launches at the `@medium` worker default with no override, "
    'for example `task(agent_type="worker", task="Add a retry helper with '
    'exponential backoff to utils.py and use it in app.py.")`. `@large` '
    "launches are for architecture, design, planning, and deep review only — "
    "never implementation."
)


def _join_oxford(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    *head, last = items
    if len(head) == 1:
        return f"{head[0]} and {last}"
    return f"{', '.join(head)}, and {last}"


def _slot_for_purpose(policy: DispatchPolicy, purpose: str, fallback: str) -> str:
    for name, slot in policy.slots.items():
        if purpose in slot.purposes:
            return name
    return fallback


def _slot_rows(policy: DispatchPolicy) -> list[str]:
    """One task-kind sentence per (profile, purposes) slot group."""
    grouped: list[tuple[str, tuple[str, ...], list[str], list[DispatchSlot]]] = []
    for name, slot in policy.slots.items():
        for group in grouped:
            if group[0] == slot.profile and group[1] == slot.purposes:
                group[2].append(name)
                group[3].append(slot)
                break
        else:
            grouped.append((slot.profile, slot.purposes, [name], [slot]))
    rows: list[str] = []
    for profile, purposes, names, slots in grouped:
        names_text = _join_oxford([f"`{name}`" for name in names])
        verb = "take" if len(names) > 1 else "takes"
        clause = ""
        if any(slot.implements == "escalation" for slot in slots):
            clause = "; state the reason for this route"
        elif all(
            slot.implements == "never" and not slot.review_eligible for slot in slots
        ):
            clause = "; never implementation"
        rows.append(
            f"{names_text} (profile `{profile}`) {verb} "
            f"{_join_oxford(list(purposes))}{clause}."
        )
    return rows


def _task_kind_paragraph(policy: DispatchPolicy) -> str:
    rows = " ".join(_slot_rows(policy))
    return (
        "Route by task kind through the configured slots; one canonical model "
        "serves every slot, and each slot's binding decides its thinking "
        f"level. {rows}"
    )


def _failure_paragraph(policy: DispatchPolicy) -> str:
    implementor = _slot_for_purpose(policy, "implementation", "implementor")
    advisor = (
        _slot_for_purpose(policy, "design-analysis", "")
        or _slot_for_purpose(policy, "planning-analysis", "")
        or "advisor"
    )
    return (
        "Select a route afresh for each task. Uncertainty, file count, "
        "session length, or wanting a better answer are not escalation "
        "criteria. If an implementation attempt fails, diagnose the failure, "
        f"retry on the `{implementor}` slot with a different approach, or "
        f"dispatch the `{advisor}` slot for read-only analysis; never "
        f"re-dispatch implementation to the `{advisor}` slot. Escalate "
        "architectural blockers to an advisor and authorization or scope "
        "blockers to the user."
    )


def _slot_table(policy: DispatchPolicy, shape: RosterShape) -> str:
    """Render launch authority, including unavailable slots, for every roster."""
    lines = [
        "### Authoritative active slot bindings",
        "The active table overrides compatibility examples and profile defaults. Pass the model and thinking explicitly; unavailable slots fail closed.",
        "",
        "| Slot | Profile | Role | Model | Thinking | Purposes | Implements | Review eligible |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, slot in policy.slots.items():
        binding = shape.slot_bindings.get(name)
        model, thinking = binding or ("unavailable", None)
        if name in shape.failures:
            model = f"{model} (unavailable)" if binding else "unavailable"
        # Tier-free rosters expose concrete launch choices, not tier names.
        role = "—" if shape.single_model or not shape.bindings else f"`{slot.role}`"
        model = model.replace("|", "&#124;").replace("\n", " ")
        lines.append(
            f"| `{name}` | `{slot.profile}` | {role} | `{model}` | {thinking or 'default'} | "
            f"{', '.join(slot.purposes)} | {slot.implements} | {slot.review_eligible} |"
        )
    return "\n".join(lines)


def _task_routing_slots(policy: DispatchPolicy) -> str:
    mechanical = _slot_for_purpose(policy, "mechanical-edit", "mechanical")
    implementor = _slot_for_purpose(policy, "implementation", "implementor")
    escalation = _slot_for_purpose(
        policy, "implementation-demanding-settled", "escalation-implementor"
    )
    return (
        "Mechanical delegated work — single-file edits, bounded searches, "
        f"verification runs — launches on the `{mechanical}` slot; substantive "
        f"implementation launches on the `{implementor}` slot; demanding "
        f"execution with a settled approach launches on the `{escalation}` "
        "slot and states the reason; architecture, design, planning, and deep "
        "review launch on the advisor and reviewer slots — never "
        "implementation."
    )


# --- Roster ----------------------------------------------------------------


def roster_for(catalog: ModelCatalog, policy: DispatchPolicy) -> RosterShape:
    """Resolve the policy's slot bindings into a roster shape.

    Unresolvable bindings are skipped: the shipped presets and accepted
    overlays resolve, and a broken binding must not break prompt rendering.
    """
    from chartreux.core.model_catalog.loader import CatalogSnapshot
    from chartreux.core.model_catalog.resolver import (
        ModelResolutionError,
        ModelResolver,
    )

    resolver = ModelResolver(CatalogSnapshot(catalog, "dispatch-render"))
    bindings = []
    slot_bindings = {}
    failures = {}
    for name, slot in policy.slots.items():
        try:
            resolved = resolver.resolve(slot.role)
            bindings.append(resolved)
            slot_bindings[name] = (
                resolved.base_model,
                resolved.thinking or resolved.definition.thinking,
            )
        except ModelResolutionError as exc:
            failures[name] = str(exc)
    return RosterShape(roster_shape(bindings).bindings, slot_bindings, failures)


def _config_policy_and_roster(
    config: ChartreuxConfigSchema,
) -> tuple[DispatchPolicy, RosterShape]:
    from chartreux.core.dispatch.presets import DEFAULT_DISPATCH_MODE, SHIPPED_PRESETS
    from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG

    if (bound := config.bound_dispatch_policy) is not None:
        return bound.render_policy, bound.roster
    snapshot = config.catalog_snapshot
    if snapshot is None:
        policy = SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE]
        return policy, roster_for(SHIPPED_CATALOG, policy)
    policy = snapshot.dispatch
    return policy, roster_for(snapshot.catalog, policy)


# --- Rendering -------------------------------------------------------------


def _active_sections(
    policy: DispatchPolicy, shape: RosterShape, *, compatibility: bool = False
) -> str:
    from chartreux.core.dispatch.presets import (
        ORCHESTRATED_PRESET,
        SHIPPED_PRESETS,
        STANDALONE_PRESET,
    )

    sections = [
        _slot_table(policy, shape),
        "### Purpose meanings\n"
        + "\n".join(
            f"- `{name}`: {entry.description}"
            for name, entry in policy.vocabulary.items()
        ),
    ]
    if shape.failures:
        sections.append(
            "### Dispatch degradation\n"
            + "\n".join(
                f"- Slot `{name}` unavailable: {reason}. Launches fail closed."
                for name, reason in shape.failures.items()
            )
        )
    if policy.compositions.strip():
        sections.append("### Review compositions\n" + policy.compositions)
    if not compatibility:
        failure = policy.failure_routing
        contrasts = policy.contrasts
        if shape.single_model or not shape.bindings:
            if failure == ORCHESTRATED_PRESET.failure_routing:
                failure = _failure_paragraph(policy)
            if contrasts == ORCHESTRATED_PRESET.contrasts:
                contrasts = STANDALONE_PRESET.contrasts
        for heading, block in (("Failure routes", failure), ("Contrasts", contrasts)):
            if block.strip():
                sections.append(f"### {heading}\n{block}")
        if (
            policy.instructions != SHIPPED_PRESETS[policy.mode].instructions
            and policy.instructions.strip()
        ):
            sections.append("### Policy instructions\n" + policy.instructions)
    return "\n\n".join(sections)


def render_routing_region(policy: DispatchPolicy, shape: RosterShape) -> str:
    """Render the CLI routing prose (the ``$dispatch_routing`` region)."""
    if policy.mode is DispatchMode.STANDALONE:
        return "\n\n".join([
            _STANDALONE_ROUTING_SENTENCE,
            _active_sections(policy, shape),
            _GATE_SENTENCE,
            _VALVE_SENTENCE,
        ])
    if shape.single_model or not shape.bindings:
        task_kind = _task_kind_paragraph(policy)
        if not shape.bindings:
            task_kind = task_kind.replace(
                "one canonical model serves every slot, and each slot's binding decides its thinking level.",
                "no slot binding is available; repair the listed failures before delegation.",
            )
        return "\n\n".join([
            task_kind,
            _failure_paragraph(policy),
            _REVIEW_SENTENCES,
            _ROLE_SENTENCES,
            _BACKGROUND_SENTENCES,
            _active_sections(policy, shape),
        ])
    blocks = [
        block
        for block in (policy.instructions, policy.contrasts, policy.failure_routing)
        if block.strip()
    ]
    return "\n\n".join([
        *blocks,
        _ROLE_SENTENCES,
        _BACKGROUND_SENTENCES,
        _active_sections(policy, shape, compatibility=True),
    ])


def render_cli_regions(policy: DispatchPolicy, shape: RosterShape) -> dict[str, str]:
    """Render every mode-variable region of the CLI prompt."""
    standalone = policy.mode is DispatchMode.STANDALONE
    return {
        "reading": _READING_STANDALONE if standalone else _READING_ORCHESTRATED,
        "implement": (_IMPLEMENT_STANDALONE if standalone else _IMPLEMENT_ORCHESTRATED),
        "identity": _IDENTITY_STANDALONE if standalone else _IDENTITY_ORCHESTRATED,
        "tools": _TOOLS_STANDALONE if standalone else _TOOLS_ORCHESTRATED,
        "routing": render_routing_region(policy, shape),
    }


def render_task_regions(policy: DispatchPolicy, shape: RosterShape) -> dict[str, str]:
    """Render the task tool's routing region (reuse, review, routing)."""
    single = shape.single_model or not shape.bindings
    if policy.mode is DispatchMode.STANDALONE:
        routing = (
            "Route by the work, not the profile default: pass the tier "
            "explicitly in `config`. Delegation is optional — when you "
            "delegate, route by purpose through the configured slots; the "
            f"slot table is authoritative. {_task_routing_slots(policy)} When "
            "you implement directly, the tier rules govern only work you "
            "delegate."
        )
    elif single:
        routing = (
            "Route by the work, not the profile default: pass the slot's "
            "launch binding explicitly in `config`. One canonical model "
            f"serves every slot; route by task kind. {_task_routing_slots(policy)}"
        )
    else:
        routing = _TASK_ROUTING_MULTI_MODEL
    if not shape.bindings:
        routing = "No slot binding is available; repair the listed failures before delegation. Route by task kind."
    routing += "\n\n" + _active_sections(policy, shape)
    implementor = _slot_for_purpose(policy, "implementation", "implementor")
    if single:
        tier = (
            "re-state the slot explicitly on every reuse, so substantive "
            "work on an agent last used for mechanical work runs on the "
            f"`{implementor}` slot. "
        )
    else:
        tier = (
            "re-state the tier explicitly on every reuse, so substantive "
            "work on a previously `@small` agent runs at `@medium`. "
        )
    return {
        "reuse": _REUSE_PREFIX + tier + _REUSE_SUFFIX,
        "review": (
            _REVIEW_EXAMPLE_SINGLE_MODEL if single else _REVIEW_EXAMPLE_MULTI_MODEL
        ),
        "routing": routing,
    }


def _splice(
    skeleton: str,
    regions: Mapping[str, str],
    placeholders: Mapping[str, str],
    kind: str,
) -> tuple[str, bool]:
    text = skeleton
    replaced = False
    for region, token in placeholders.items():
        if token in text:
            text = text.replace(token, regions[region])
            replaced = True
    if _PLACEHOLDER_PREFIX in text:
        raise DispatchRenderError(
            f"unresolved {kind} dispatch placeholder in skeleton: "
            f"{sorted(set(re.findall(r'\$dispatch_[a-zA-Z0-9_]+', text)))}"
        )
    return text, replaced


def render_cli_prompt(policy: DispatchPolicy, shape: RosterShape, skeleton: str) -> str:
    """Render the CLI prompt document from its skeleton.

    A skeleton without placeholders (a custom prompt) passes through
    unchanged. A rendered document is validated by the S8 anchor lint.
    """
    text, replaced = _splice(
        skeleton, render_cli_regions(policy, shape), CLI_PLACEHOLDERS, "cli"
    )
    if replaced:
        reject_errors(lint_rendered(text, policy.mode))
    return text


def render_task_description(
    policy: DispatchPolicy, shape: RosterShape, skeleton: str
) -> str:
    """Render the task tool description from its skeleton."""
    text, _ = _splice(
        skeleton, render_task_regions(policy, shape), TASK_PLACEHOLDERS, "task"
    )
    return text


def contains_dispatch_placeholder(text: str | None) -> bool:
    """Whether a description is a dispatch skeleton awaiting rendering."""
    if not text:
        return False
    return _PLACEHOLDER_PREFIX in text


def task_skeleton() -> str:
    """The shipped, unstripped task tool skeleton (what the tool serves)."""
    return read_safe(TASK_PROMPT_PATH).text


def render_cli_prompt_for_config(config: ChartreuxConfigSchema, skeleton: str) -> str:
    """Render the CLI prompt document for a session's bound policy."""
    policy, shape = _config_policy_and_roster(config)
    return render_cli_prompt(policy, shape, skeleton)


def render_task_description_for_config(
    config: ChartreuxConfigSchema, skeleton: str
) -> str:
    """Render the task tool description for a session's bound policy."""
    policy, shape = _config_policy_and_roster(config)
    return render_task_description(policy, shape, skeleton)


def task_description_for_config(config: ChartreuxConfigSchema) -> str:
    """The policy-bound task tool description served for a configuration."""
    return render_task_description_for_config(config, task_skeleton())
