"""Rendered dispatch prompts: the composed-artifact regions and placeholders.

The composition path (design §5) renders its own regions from the typed
identity facts and the package-4 route facts — never from a dispatch mode —
and splices the package-5 placeholder contract into the identity and
workflow bodies. This module owns that composition rendering, the placeholder
contract, and the config-keyed task-tool description entry points.

The retired surface (package 16b): the legacy policy-keyed rendering —
``roster_for``, ``render_routing_region``, ``render_cli_regions``,
``render_task_regions``, ``render_cli_prompt``, ``render_task_description``,
``render_cli_prompt_for_config``, and the slot-table machinery behind them —
is gone with ``chartreux.core.dispatch.schema``/``presets``. Routing is
configured through ``[orchestration.routes.<purpose>]`` (see
``chartreux.core.dispatch.routes``); the task tool description renders from
those route facts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import TYPE_CHECKING

from chartreux import CHARTREUX_ROOT
from chartreux.core.agents.identities import Delegation
from chartreux.core.dispatch.purposes import SHIPPED_PURPOSES
from chartreux.core.dispatch.route_models import PurposeRoute
from chartreux.core.dispatch.routes import SESSION_MODEL
from chartreux.utils.io import read_safe

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema

TASK_PROMPT_PATH = (
    CHARTREUX_ROOT / "core" / "tools" / "builtins" / "prompts" / "task.md"
)

# region name -> placeholder token, in skeleton order.
CLI_PLACEHOLDERS: Mapping[str, str] = {
    "reading": "$dispatch_reading",
    "implement": "$dispatch_implement",
    "identity": "$dispatch_identity",
    "tools": "$dispatch_tools",
    "disclosure": "$dispatch_disclosure",
    "routing": "$dispatch_routing",
}
TASK_PLACEHOLDERS: Mapping[str, str] = {
    "disclosure": "$dispatch_disclosure",
    "reuse": "$dispatch_reuse",
    "review": "$dispatch_review",
    "routing": "$dispatch_routing",
}
_PLACEHOLDER_PREFIX = "$dispatch_"

# The worker disclosure requirement is dispatch scaffolding: the orchestrator
# composes every task brief, so the brief-composition rule is fixed here and
# spliced into both the CLI prompt and the task tool description.
_BRIEF_DISCLOSURE = (
    "Every task brief must carry the worker disclosure requirement: whatever "
    "report format the brief or a loaded skill specifies, the worker's final "
    "report must include the denial record (each denied command and the "
    "runtime's stated reason, or none) and the complete cumulative "
    "touched-path set (every repository path the worker created, edited, "
    "renamed, or deleted, including edits later reverted). State this in the "
    "brief whenever it names a report format."
)


class DispatchRenderError(ValueError):
    """A skeleton could not be rendered completely."""


# --- Fixed developer-owned scaffolding shared by the composition regions ----

_READING_ORCHESTRATED = (
    "Before planning, read named instruction and routing files; delegate "
    "reading relevant source, tests, and entry points to a subagent. Before "
    "relying on an API or library function, have a subagent check its "
    "established use rather than guessing versions or signatures."
)

_READING_STANDALONE = (
    "Before planning, read named instruction and routing files yourself; "
    "delegate reading relevant source, tests, and entry points to a subagent "
    "when the reading is broad. Before relying on an API or library "
    "function, have a subagent check its established use rather than "
    "guessing versions or signatures."
)

_IMPLEMENT_ORCHESTRATED = (
    "5. **Implement** the approved plan by dispatching bounded repo edits to "
    "subagents; never edit repo files yourself."
)

_IMPLEMENT_STANDALONE = (
    "5. **Implement** the approved plan. You may implement directly: make "
    "the approved repo edits yourself. Stay within the approved plan; change "
    "minimally."
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
- `write_file` / `edit` — persistence and trivial state/bookkeeping edits (scratchpad notes, design documents, plans, ledgers, state notes); never repository source edits.
- `bash` — one-line read-only probes and orchestration metadata; not exploration, tests, builds, or mutation. Delegate searches and verification instead of using direct `grep` or shell commands.
- `skill` and `todo` — procedures and task tracking.
- `web_search` / `web_fetch` — quick single-question lookups; delegate multi-step research.
- `check_agents`, `get_agent_result`, `wait_for_agent`, `cancel_agent`, and `release_agent` — background-agent lifecycle, not specialist work."""

# --- Composition rendering (design §5): typed-facts-keyed regions -----------
#
# The package-5 bodies carry the ``$dispatch_*`` placeholder contract. The
# composition regions render from the identity's typed delegation posture
# (design §3) and the package-4 route facts — never from a dispatch mode.

_IDENTITY_DIRECT = (
    "You execute directly: you are the implementor, with the full granted "
    "tool set. Direct tool output grows your context — every file, search "
    "result, and shell output you inspect is re-sent on every subsequent API "
    "call. Keep reads targeted, bound shell output, move large artifacts to "
    "the scratchpad, and say when your context is getting long. When context "
    "is already long (including when the user says so), or a step would read "
    "or produce more than a few hundred lines, delegate that bounded piece "
    "to a child before searching or reading source yourself; keep only its "
    "concise result in the main context."
)

_TOOLS_DIRECT = (
    "Use your full tool set directly: write_file/edit — repo files within "
    "the assigned scope; scratchpad for temporary artifacts. read_file — "
    "files the task names or cites. bash — the work itself, always with "
    "timeouts. skill and todo — procedures and task tracking. web_search / "
    "web_fetch — quick single-question lookups. check_agents, "
    "get_agent_result, wait_for_agent, cancel_agent, and release_agent — "
    "background-agent lifecycle when you delegate."
)

_ROUTING_PROSE = (
    "Route by purpose: the launch-target table in the shared references "
    "binds each purpose to its launch target. Explicit `model` and "
    "`identity` fields override the route's corresponding field; multi-seat "
    "purposes require an explicit seat, with no implicit fan-out. Purpose "
    "labels organize work; they never grant permission. Select a route afresh "
    "for each task — uncertainty, file count, session length, or wanting a "
    "better answer are not escalation criteria. If an implementation attempt "
    "fails, diagnose the failure, retry with a different approach, or "
    "dispatch read-only analysis; escalate architectural blockers to the "
    "design-analysis purpose and authorization or scope blockers to the user."
)

_COMPOSITION_REUSE = (
    "To continue genuine work on the same feature, first use `check_agents` "
    "to inspect each idle agent's effective model, thinking, and retained "
    "context. Reuse an idle handle with `agent_id` and `background: true`; "
    "omit `retain` to inherit the instance's committed retention policy "
    "(pass `retain: false` for a final continuation that closes afterward). "
    "Re-task the same idle agent for a genuine continuation, with the "
    "current scope and intervening changes in the brief; re-state the "
    "purpose explicitly on every reuse, so substantive work on an agent "
    "last used for a mechanical purpose runs on the implementation route. "
    "A retained agent keeps its last model and thinking unless overridden. "
    "Launch a new agent when its identity, stack, or assignment must change, "
    "or when work is unrelated or needs independent judgment. A running "
    "agent is busy; wait for its result with `wait_for_agent` (a "
    "`still_running` timeout is a normal outcome: end your turn and "
    "re-invoke the wait in a later turn), and spawn another only for "
    "independent, non-conflicting work. A `closed` or `evicted` agent cannot "
    "be reused; retrieve its retained result with `get_agent_result` if "
    "needed, then start a new agent."
)

# The independent-review guidance: the opening clause is pinned verbatim by
# tests/core/policy/test_composition.py; the composition sentence renders the
# configured review seats, so it stays neutral to the configured roster.
_COMPOSITION_REVIEW_OPENING = (
    "For an independent review, launch a fresh reviewer on the review "
    "purpose — never the author, and never a reused advisor; give a "
    "second-round reviewer no earlier conclusions. Keep an advisor across "
    "related design refinements. "
)
_COMPOSITION_REVIEW_FALLBACK = (
    "Deep review composes the configured seats; fail closed when a required "
    "seat is unavailable."
)


def composition_review_region(routes: Mapping[str, PurposeRoute]) -> str:
    """The independent-review guidance, generalized to the configured seats.

    The shipped three-seat composition renders its pinned enumeration; a
    configured supplementary seat renders with the others, in route order.
    """

    def names(seats: Sequence[str]) -> str:
        if len(seats) == 1:
            return seats[0]
        return ", ".join(seats[:-1]) + f", and {seats[-1]}"

    route = routes.get("review.deep")
    seats = [target.seat for target in route.targets if target.seat] if route else []
    if not seats:
        return _COMPOSITION_REVIEW_OPENING + _COMPOSITION_REVIEW_FALLBACK
    sentence = f"Deep review composes the {names(seats)} "
    if len(seats) == 1:
        sentence += "seat; fail closed when a required seat is unavailable."
    else:
        sentence += "seats; fail closed when a required seat is unavailable."
    return _COMPOSITION_REVIEW_OPENING + sentence


def render_composition_regions(*, delegation: Delegation | None) -> dict[str, str]:
    """The CLI-side ``$dispatch_*`` regions keyed on typed identity facts.

    A delegation-required identity (the orchestrator) gets the delegation
    posture regions; every other identity — and custom identities, which
    carry no harness-known posture — get the direct-execution regions.
    """
    direct = delegation is not Delegation.REQUIRED
    return {
        "reading": _READING_STANDALONE if direct else _READING_ORCHESTRATED,
        "implement": _IMPLEMENT_STANDALONE if direct else _IMPLEMENT_ORCHESTRATED,
        "identity": _IDENTITY_DIRECT if direct else _IDENTITY_ORCHESTRATED,
        "tools": _TOOLS_DIRECT if direct else _TOOLS_ORCHESTRATED,
        "disclosure": _BRIEF_DISCLOSURE,
        "routing": _ROUTING_PROSE,
    }


def render_launch_target_region(
    routes: Mapping[str, PurposeRoute], session_model: str
) -> str:
    """The rendered launch-target table (design §5 step 7; package-4 facts).

    Shipped targets bind the session model except the cost-tier-aware four,
    which render the ``$cheapest-capable`` marker (resolved at launch from
    the catalog's price data); unmet model-diversity requirements are
    reported, never faked.
    """
    lines = [
        "### Launch targets",
        "",
        "Delegated work routes by purpose. This is the conversation's live "
        "route table: a validated route edit applies to subsequent launches, "
        "while children already admitted keep their bindings. Shipped "
        "targets bind the session model except `implementation`, "
        "`mechanical-edit`, `search`, and `verification`, which default to "
        "the cheapest capable model in the catalog — the `$cheapest-capable` "
        "binding, resolved at launch from the catalog's price data. Explicit "
        "fields override the route's corresponding field.",
        "",
        "| Purpose | Identity | Tier | Seat | Model |",
        "| --- | --- | --- | --- | --- |",
    ]
    requirements: list[str] = []
    for purpose, route in routes.items():
        for target in route.targets:
            model = session_model if target.model == SESSION_MODEL else target.model
            lines.append(
                f"| `{purpose}` | `{target.identity}` | {target.tier.value} "
                f"| {target.seat or '—'} | `{model}` |"
            )
        if route.requires_distinct_models > 1:
            requirements.append(
                f"- `{purpose}`: {route.requires_distinct_models} distinct "
                "canonical models (counted by canonical model, not "
                "deployment)."
            )
    if requirements:
        lines.extend(["", "### Route requirements", "", *requirements])
    lines.extend(["", "### Purpose meanings"])
    for name, entry in SHIPPED_PURPOSES.items():
        lines.append(f"- `{name}`: {entry.description}")
    return "\n".join(lines)


def render_composition_task_regions(
    *, routes: Mapping[str, PurposeRoute], session_model: str
) -> dict[str, str]:
    """The task tool's regions keyed on the package-4 route facts."""
    return {
        "disclosure": _BRIEF_DISCLOSURE,
        "reuse": _COMPOSITION_REUSE,
        "review": composition_review_region(routes),
        "routing": render_launch_target_region(routes, session_model),
    }


def splice_composition_regions(skeleton: str, regions: Mapping[str, str]) -> str:
    """Splice the CLI placeholder contract into a composition body.

    The package-5 placeholder contract (``$dispatch_reading``,
    ``$dispatch_identity``, ``$dispatch_tools``, ``$dispatch_disclosure``,
    ``$dispatch_routing``, ``$dispatch_implement``) is spliced exactly as the
    bodies expect; an unresolved ``$dispatch_*`` token fails closed.
    """
    text, _ = _splice(skeleton, regions, CLI_PLACEHOLDERS, "composition")
    return text


def render_task_description_for_composition(
    *, routes: Mapping[str, PurposeRoute], session_model: str, skeleton: str
) -> str:
    """Render the task tool description from the route facts (new pipeline)."""
    text, _ = _splice(
        skeleton,
        render_composition_task_regions(routes=routes, session_model=session_model),
        TASK_PLACEHOLDERS,
        "task",
    )
    return text


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


def session_model_alias(config: ChartreuxConfigSchema) -> str:
    """The session model the shipped route defaults bind (design §9).

    The model the launch-target table renders for ``$session`` targets: the
    configured active model's canonical alias when a catalog is attached,
    else the resolved shipped selection.
    """
    if config.catalog_snapshot is not None:
        return config.get_active_model().alias
    from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
    from chartreux.core.model_catalog.loader import CatalogSnapshot
    from chartreux.core.model_catalog.resolver import ModelResolver

    resolver = ModelResolver(CatalogSnapshot(SHIPPED_CATALOG, "composition"))
    return resolver.resolve(config.active_model or "@main").base_model


def contains_dispatch_placeholder(text: str | None) -> bool:
    """Whether a description is a dispatch skeleton awaiting rendering."""
    if not text:
        return False
    return _PLACEHOLDER_PREFIX in text


def task_skeleton() -> str:
    """The shipped, unstripped task tool skeleton (what the tool serves)."""
    return read_safe(TASK_PROMPT_PATH).text


def render_task_description_for_config(
    config: ChartreuxConfigSchema,
    skeleton: str,
    *,
    routes: Mapping[str, PurposeRoute] | None = None,
) -> str:
    """Render the task tool description from a configuration's route facts.

    The description is composition-rendered: the launch-target table and the
    purpose meanings come from the configured ``[orchestration.routes.*]``
    and the session model, never from a dispatch policy. The DEFAULT reads
    the configuration's LIVE route table — the same source launch admission
    resolves against (ADR 0022) — so the model-facing description and
    executable admission can never diverge; ``routes`` remains an explicit
    override for callers that must render one specific table (a config-only
    surface with no conversation, or a pinned fixture).
    """
    return render_task_description_for_composition(
        routes=dict(routes if routes is not None else config.purpose_routes.routes),
        session_model=session_model_alias(config),
        skeleton=skeleton,
    )


def task_description_for_config(
    config: ChartreuxConfigSchema, *, routes: Mapping[str, PurposeRoute] | None = None
) -> str:
    """The route-bound task tool description served for a configuration."""
    return render_task_description_for_config(config, task_skeleton(), routes=routes)
