"""Workflow-skill eligibility: the harness-side compatibility table (design §5).

Shipped workflow skills join the identity/workflow alignment: ceremony-bearing
skills load only for workflow/identity combinations that carry the ceremony,
and the minimal default exposes/loads only identity-neutral skills (design §5;
decision log item 10). The classification is a HARNESS-SIDE TABLE in this
module — it is deliberately NOT encoded in SKILL.md frontmatter, and
``chartreux/core/skills/parser.py`` is untouched. Only the shipped copies
(``SkillSource.SHIPPED``) are classified: a personal shadow copy (a local skill
that reuses a shipped name) is the user's own content and is never filtered or
rewritten by the harness.

Eligibility reads the CONVERSATION'S FROZEN POLICY — the bound identity and
workflow selections (design §3, §6) — never pending saved selections: the
manager snapshots :class:`ConversationPolicyFacts` when the conversation binds
its policy, so a saved selection change that is pending for new conversations
does not alter a bound conversation's eligibility.

The same rule is wired through every exposure surface — the skills catalog
(the ``<available_skills>`` section), the ``skill`` tool, and slash
invocation — because all three resolve through
:attr:`SkillManager.available_skills` / :meth:`SkillManager.get_skill`.
Filtering is NOT automatic catalog pruning: discovery still lists every
incompatible skill (marked via :meth:`SkillManager.skill_eligibility` and
:attr:`SkillManager.workflow_incompatibilities`); only exposure and loading
are filtered.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from chartreux.core.skills.models import SkillSource, SkillWorkflowClass

#: The only shipped workflow that carries the process ceremony (the opening
#: gate, the acceptance lattice, delegated verification, review requirements).
CEREMONY_WORKFLOW_ID = "ceremony"


@dataclass(frozen=True)
class ConversationPolicyFacts:
    """The conversation's frozen policy facts (design §3, §6).

    ``identity`` and ``workflow`` are the bound ``[agent]`` selections of the
    conversation — the frozen policy — not pending saved selections.
    """

    identity: str
    workflow: str


@dataclass(frozen=True)
class SkillEligibility:
    """Whether a skill may be exposed and loaded in one conversation."""

    eligible: bool
    requirement: str | None = None
    """What the conversation lacks, phrased for diagnostics; None when eligible."""


#: The harness-side compatibility table (package 9): shipped skill name ->
#: workflow class. Identity-neutral shipped skills are NOT listed — they are
#: eligible in every workflow — and neither are non-shipped skills (builtins
#: and local/personal skills), which the harness never classifies.
SHIPPED_SKILL_WORKFLOW_CLASS: Mapping[str, SkillWorkflowClass] = MappingProxyType({
    # Main-agent workflow procedures that mandate advisor/reviewer dispatch
    # and acceptance gates independently of the selected workflow (design §5).
    "main-design": SkillWorkflowClass.CEREMONY_BEARING,
    "main-plan": SkillWorkflowClass.CEREMONY_BEARING,
    "main-review": SkillWorkflowClass.CEREMONY_BEARING,
    # Test generation's self-review step is a review gate: it depends on the
    # review ceremony (it invokes `main-review`), so it is ceremony-bearing.
    "main-test-generator": SkillWorkflowClass.CEREMONY_BEARING,
})


def normalize_workflow_id(selection: str) -> str:
    """The resolved workflow id of a selection (``builtin:`` qualification stripped)."""
    return selection.removeprefix("builtin:")


def skill_workflow_class(name: str, *, source: SkillSource) -> SkillWorkflowClass:
    """The table lookup; only the shipped copies are classified."""
    if source is not SkillSource.SHIPPED:
        return SkillWorkflowClass.IDENTITY_NEUTRAL
    return SHIPPED_SKILL_WORKFLOW_CLASS.get(name, SkillWorkflowClass.IDENTITY_NEUTRAL)


def workflow_carries_ceremony(workflow: str) -> bool:
    """Whether a workflow selection carries the ceremony (design §5)."""
    return normalize_workflow_id(workflow) == CEREMONY_WORKFLOW_ID


def check_skill_eligibility(
    name: str, *, source: SkillSource, facts: ConversationPolicyFacts
) -> SkillEligibility:
    """One rule: may this skill be exposed/loaded in this conversation?"""
    klass = skill_workflow_class(name, source=source)
    if klass is not SkillWorkflowClass.CEREMONY_BEARING:
        return SkillEligibility(eligible=True)
    if workflow_carries_ceremony(facts.workflow):
        return SkillEligibility(eligible=True)
    return SkillEligibility(
        eligible=False,
        requirement=(
            f"the `{CEREMONY_WORKFLOW_ID}` workflow (this conversation runs the "
            f"`{normalize_workflow_id(facts.workflow)}` workflow with the "
            f"`{facts.identity}` identity)"
        ),
    )


def format_incompatibility_diagnostic(
    name: str, *, source: SkillSource, facts: ConversationPolicyFacts
) -> str:
    """The actionable diagnostic for an incompatible explicit invocation.

    Names the workflow/identity requirement and the remedy (a new conversation
    that selects the ceremony workflow); the selection is frozen per
    conversation, so the remedy is a new conversation, never a mid-conversation
    switch.
    """
    eligibility = check_skill_eligibility(name, source=source, facts=facts)
    requirement = eligibility.requirement or (f"the `{CEREMONY_WORKFLOW_ID}` workflow")
    return (
        f'Skill "{name}" is not available in this conversation: it is a '
        f"ceremony-bearing workflow skill and requires {requirement}. "
        f"Skill workflow eligibility is frozen per conversation; start a new "
        f"conversation with the ceremony workflow selected (for example "
        f'`[agent] workflow = "{CEREMONY_WORKFLOW_ID}"` in config.toml) to use '
        f"this skill."
    )
