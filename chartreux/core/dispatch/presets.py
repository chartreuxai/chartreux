"""Shipped dispatch data; prompt rendering is owned by the renderer."""

from __future__ import annotations

from types import MappingProxyType

from chartreux.core.dispatch.schema import DispatchMode, DispatchPolicy, DispatchSlot

DEFAULT_DISPATCH_MODE = DispatchMode.STANDALONE

_SLOTS = {
    "mechanical": DispatchSlot(
        profile="worker",
        role="@scout",
        purposes=("search", "exploration", "verification", "mechanical-edit"),
        implements="routine",
        review_eligible=False,
    ),
    "implementor": DispatchSlot(
        profile="worker",
        role="@worker",
        purposes=("implementation",),
        implements="routine",
        review_eligible=False,
    ),
    "escalation-implementor": DispatchSlot(
        profile="worker",
        role="@heavy",
        purposes=("implementation-demanding-settled",),
        implements="escalation",
        review_eligible=False,
    ),
    "advisor": DispatchSlot(
        profile="advisor",
        role="@heavy",
        purposes=("design-analysis", "planning-analysis"),
        implements="never",
        review_eligible=False,
    ),
    "reviewer": DispatchSlot(
        profile="reviewer",
        role="@worker",
        purposes=("review.quick", "review.standard"),
        implements="never",
        review_eligible=True,
    ),
    "analytical-reviewer": DispatchSlot(
        profile="reviewer",
        role="@heavy",
        purposes=("review.deep",),
        implements="never",
        review_eligible=True,
    ),
    "peer-reviewer": DispatchSlot(
        profile="reviewer",
        role="@heavy",
        purposes=("review.deep",),
        implements="never",
        review_eligible=True,
    ),
    "execution-reviewer": DispatchSlot(
        profile="reviewer",
        role="@worker",
        purposes=("review.deep",),
        implements="never",
        review_eligible=True,
    ),
}

# Isolated compatibility prose: preserve the legacy multi-model routing semantics.
_ORCHESTRATED_ROUTING = """Route through the `worker`, `advisor`, or `reviewer` agent profile (including user/project TOML profiles). `config.model` accepts a canonical model name or an `@role` from the configured catalog; do not guess model names. The shipped model roles are `orchestrator` for the main assistant and `worker`, `scout`, and `heavy` for subagent work. Route by task, not by difficulty: `@scout` for search, grep, exploration, verification, and mechanical single-file edits; `@worker` for all substantive implementation, however demanding (novel algorithmic reasoning, difficult refactoring, and broad-impact work included); `@heavy` for architecture, cross-subsystem design, design and planning analysis, and deep review, plus demanding execution with a settled approach through the escalation-implementor route — never routine implementation. The worker and reviewer profiles use `@worker` by default; the advisor profile uses `@heavy` for architecture, design, planning, and destructive-operation analysis only; it never implements. Demanding execution with a settled approach runs through the worker-profile escalation-implementor route. The reviewer profile stays `@worker`; "deep review at `@heavy`" means the reviewer profile with a `@heavy` model override. Use fresh reviewers for independent judgments; never reuse an advisor as a reviewer or give a second-round reviewer earlier conclusions. Keep an advisor across related design refinements."""

_ORCHESTRATED_CONTRASTS = """Pass the tier explicitly — the profile default is not the routing decision. Mechanical work (single-file edits, bounded searches, verification runs) launches with a `config` model override: `task(agent_type="worker", task="Rename add to plus in utils.py and update its call sites.", config={"model": "@scout"})`. Substantive implementation launches at the `@worker` worker default with no override: `task(agent_type="worker", task="Add a retry helper with exponential backoff to utils.py and use it in app.py.")`. `@heavy` launches are for architecture, design, planning, and deep review, plus demanding execution with a settled approach through the escalation-implementor route — never routine implementation."""

_ORCHESTRATED_FAILURE_ROUTING = """Select a tier afresh for each task. Uncertainty, file count, session length, or wanting a better answer are not escalation criteria. If an implementation attempt fails, diagnose the failure, retry at `@worker` with a different approach, or dispatch a `@heavy` advisor for read-only analysis; never re-dispatch implementation to the `@heavy` advisor. Escalate architectural blockers to an advisor and authorization or scope blockers to the user."""

_COMPOSITIONS = """review.quick and review.standard: use slot `reviewer` in a fresh context, never the author.
review.deep: use slots `analytical-reviewer`, `peer-reviewer`, and `execution-reviewer` in fresh contexts. If `execution-reviewer` authored the work, substitute slot `reviewer`; if `reviewer` authored the work, substitute slot `execution-reviewer`. Never reuse an advisor as a reviewer or give a second-round reviewer earlier conclusions.
For one canonical model, including multiple thinking levels, use fresh-context review on the available model instead of claiming model diversity. Fail closed if a required independent slot is unavailable; never silently substitute."""

ORCHESTRATED_PRESET = DispatchPolicy(
    mode=DispatchMode.ORCHESTRATED,
    identity="orchestrated",
    version=1,
    slots=_SLOTS,
    instructions=_ORCHESTRATED_ROUTING,
    failure_routing=_ORCHESTRATED_FAILURE_ROUTING,
    compositions=_COMPOSITIONS,
    contrasts=_ORCHESTRATED_CONTRASTS,
)

STANDALONE_PRESET = DispatchPolicy(
    mode=DispatchMode.STANDALONE,
    identity="standalone",
    version=1,
    slots=_SLOTS,
    instructions="""You may implement directly: make the approved repo edits yourself. Stay within the approved plan; change minimally.

Verification stays delegated. Dispatch checks to a subagent, scaled to the change, then dispatch an independent review against the approved plan — a fresh reviewer, never the author. Never run tests, builds, or other verification yourself; never review your own work; never claim a check you did not run.

Direct execution grows your context: everything you read or run is re-sent on every call. Keep reads targeted, bound shell output, move large artifacts to the scratchpad, and say when your context is getting long. When context is already long (including when the user says so), or a step would read or produce more than a few hundred lines, delegate that bounded piece to a worker before searching or reading source yourself; keep only its concise result in the main context. This context-hygiene requirement overrides optional delegation; keep the session in standalone mode.

write_file/edit — repo files within the approved scope; scratchpad for temporary artifacts. read_file — files the task names or cites. bash — orchestration metadata only, always with timeouts; delegate searches, exploration, tests, and builds.

Delegation is optional. When you delegate, route by purpose through the configured slots and pass the tier explicitly; the slot table is authoritative. When you implement directly, the tier rules govern only work you delegate.

The opening gate and acceptance lattice remain unchanged. Saved policy changes apply next session.
Delegate bounded work when context grows long without changing the session mode.""",
    failure_routing=_ORCHESTRATED_FAILURE_ROUTING,
    compositions=_COMPOSITIONS,
    contrasts="""Use slot `mechanical` for bounded searches, verification runs, and mechanical single-file edits; use slot `implementor` for substantive implementation. Use slot `escalation-implementor` for demanding execution with a settled approach and state the reason; it is a worker-profile route. Use slot `advisor` for read-only design and planning analysis only; the advisor profile never implements.""",
)

SHIPPED_PRESETS = MappingProxyType({
    DispatchMode.ORCHESTRATED: ORCHESTRATED_PRESET,
    DispatchMode.STANDALONE: STANDALONE_PRESET,
})
