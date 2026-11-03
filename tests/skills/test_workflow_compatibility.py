"""Workflow-skill eligibility acceptance pins (design §5; plan package 9).

Ceremony-bearing shipped skills load only in conversations whose frozen
workflow carries the ceremony; the minimal default exposes/loads only
identity-neutral skills. The SAME rule is pinned across the three surfaces
— the skills catalog (the ``<available_skills>`` section of the composed
system prompt), the ``skill`` tool, and slash invocation — because all three
resolve through ``SkillManager.available_skills`` / ``get_skill``.

Eligibility reads the conversation's FROZEN policy (the bound identity and
workflow selections), never pending saved selections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chartreux.core.agents import AgentManager
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.skills.builtins import BUILTIN_SKILLS
from chartreux.core.skills.compatibility import (
    CEREMONY_WORKFLOW_ID,
    SHIPPED_SKILL_WORKFLOW_CLASS,
    ConversationPolicyFacts,
    check_skill_eligibility,
    format_incompatibility_diagnostic,
)
from chartreux.core.skills.manager import SHIPPED_SKILLS_DIR, SkillManager
from chartreux.core.skills.models import SkillSource, SkillWorkflowClass
from chartreux.core.skills.parser import parse_skill_markdown
from chartreux.core.system_prompt import get_universal_system_prompt
from chartreux.core.tools.base import BaseToolState, InvokeContext, ToolError
from chartreux.core.tools.builtins.skill import Skill, SkillArgs, SkillToolConfig
from tests.conftest import ConfigBuilder, OrchestratorLoader, build_test_vibe_config
from tests.mock.utils import collect_result
from tests.skills.conftest import create_skill

SKILLS_ROOT = SHIPPED_SKILLS_DIR

SHIPPED_SKILL_NAMES = sorted(
    path.parent.name for path in SHIPPED_SKILLS_DIR.glob("*/SKILL.md")
)
CEREMONY_BEARING_SKILLS = sorted(SHIPPED_SKILL_WORKFLOW_CLASS)
NEUTRAL_SHIPPED_SKILLS = sorted(set(SHIPPED_SKILL_NAMES) - set(CEREMONY_BEARING_SKILLS))

# Ceremony anchors: content that would reinstate the ceremony workflow
# (gates, acceptance lattice, review requirements, model-diversity doctrine)
# inside a minimal session. The minimal session's rendered skill content must
# contain none of them.
CEREMONY_ANCHORS = (
    "Obtain the user's explicit design acceptance",
    "Obtain the user's explicit plan acceptance",
    "review.quick",
    "review.standard",
    "review.deep",
    "seat=",
    "opening gate",
    "acceptance lattice",
    "delegated verification",
    "ceremony",
    "requires_distinct_models",
    "distinct canonical models",
    "model diversity",
    "Dispatch the advisor",
    "rendered slot table",
    "acceptance gate",
)

# Retired routing constructs: no rewritten shipped skill reinstates them.
RETIRED_ROUTING_ANCHORS = ("agent_type", "slot", "profile")


def minimal_config(**kwargs: Any) -> ChartreuxConfigSchema:
    """The shipped default: coder identity, minimal workflow, no strategy."""
    return build_test_vibe_config(**kwargs)


def ceremony_config(**kwargs: Any) -> ChartreuxConfigSchema:
    return build_test_vibe_config(agent={"workflow": CEREMONY_WORKFLOW_ID}, **kwargs)


def minimal_manager(**kwargs: object) -> SkillManager:
    config = minimal_config(**kwargs)
    return SkillManager(lambda: config)


def ceremony_manager(**kwargs: object) -> SkillManager:
    config = ceremony_config(**kwargs)
    return SkillManager(lambda: config)


def skill_body(name: str) -> str:
    """The SKILL.md body (frontmatter excluded) of a shipped skill."""
    _, body = parse_skill_markdown((SKILLS_ROOT / name / "SKILL.md").read_text())
    return body


def skill_frontmatter(name: str) -> dict[str, object]:
    frontmatter, _ = parse_skill_markdown((SKILLS_ROOT / name / "SKILL.md").read_text())
    return dict(frontmatter)


def render_session_prompt(
    config: ChartreuxConfigSchema,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> str:
    """Render the composed session prompt, including the skills catalog."""
    return str(
        get_universal_system_prompt(
            config,
            SkillManager(lambda: config),
            AgentManager(load_orchestrator(config)),
        )
    )


def catalog_section(prompt: str) -> str:
    start = prompt.find("<available_skills>")
    end = prompt.find("</available_skills>")
    assert start != -1 and end != -1, "the rendered prompt carries the skills catalog"
    return prompt[start : end + len("</available_skills>")]


@pytest.fixture
def skill_tool() -> Skill:
    return Skill(config_getter=lambda: SkillToolConfig(), state=BaseToolState())


class TestClassificationTable:
    """The harness-side compatibility table (design §5 mechanism pin)."""

    def test_table_classifies_exactly_the_ceremony_bearing_shipped_skills(self) -> None:
        assert CEREMONY_BEARING_SKILLS == [
            "main-design",
            "main-plan",
            "main-review",
            "main-test-generator",
        ]
        assert set(SHIPPED_SKILL_WORKFLOW_CLASS) <= set(SHIPPED_SKILL_NAMES)
        assert set(SHIPPED_SKILL_WORKFLOW_CLASS.values()) == {
            SkillWorkflowClass.CEREMONY_BEARING
        }

    def test_identity_neutral_shipped_skills_are_not_in_the_table(self) -> None:
        assert NEUTRAL_SHIPPED_SKILLS == [
            "main-debugging",
            "main-git-workflow",
            "sub-architecture-mapper",
            "sub-explorer",
            "sub-finder",
            "sub-implementor",
            "sub-researcher",
            "sub-summarizer",
            "sub-verifier",
            "web-search",
            "workspace",
        ]
        for name in NEUTRAL_SHIPPED_SKILLS:
            assert name not in SHIPPED_SKILL_WORKFLOW_CLASS

    def test_compatibility_is_not_encoded_in_skill_md_frontmatter(self) -> None:
        # The mechanism pin: the table is harness-side; no SKILL.md
        # frontmatter declares workflow compatibility.
        for name in SHIPPED_SKILL_NAMES:
            frontmatter = skill_frontmatter(name)
            assert "workflow" not in frontmatter
            assert "ceremony" not in frontmatter
            assert "workflow-class" not in frontmatter
            assert "requires-workflow" not in frontmatter

    def test_builtins_and_local_skills_are_identity_neutral(self) -> None:
        facts = ConversationPolicyFacts(identity="coder", workflow="minimal")
        for name in BUILTIN_SKILLS:
            assert name not in SHIPPED_SKILL_WORKFLOW_CLASS
            assert check_skill_eligibility(
                name, source=SkillSource.BUILTIN, facts=facts
            ).eligible
        assert check_skill_eligibility(
            "my-own-skill", source=SkillSource.LOCAL, facts=facts
        ).eligible

    def test_ceremony_bearing_skills_require_the_ceremony_workflow(self) -> None:
        facts = ConversationPolicyFacts(identity="coder", workflow="minimal")
        for name in CEREMONY_BEARING_SKILLS:
            eligibility = check_skill_eligibility(
                name, source=SkillSource.SHIPPED, facts=facts
            )
            assert eligibility.eligible is False
            assert eligibility.requirement is not None
            assert CEREMONY_WORKFLOW_ID in eligibility.requirement
        ceremony = ConversationPolicyFacts(
            identity="coder", workflow="builtin:ceremony"
        )
        for name in CEREMONY_BEARING_SKILLS:
            assert check_skill_eligibility(
                name, source=SkillSource.SHIPPED, facts=ceremony
            ).eligible

    def test_custom_workflows_do_not_carry_the_ceremony(self) -> None:
        # Fail closed: only the shipped ceremony workflow carries it.
        facts = ConversationPolicyFacts(identity="coder", workflow="user:custom")
        assert (
            check_skill_eligibility(
                "main-design", source=SkillSource.SHIPPED, facts=facts
            ).eligible
            is False
        )


class TestMinimalDefaultExposure:
    """Acceptance: the minimal default exposes/loads only identity-neutral
    shipped skills, across the catalog, the skill tool, and slash.
    """

    def test_catalog_lists_only_neutral_shipped_skills(
        self,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> None:
        config = build_config(include_prompt_detail=True, include_model_info=False)
        prompt = render_session_prompt(config, load_orchestrator)
        section = catalog_section(prompt)

        for name in CEREMONY_BEARING_SKILLS:
            assert f"<name>{name}</name>" not in section
        for name in NEUTRAL_SHIPPED_SKILLS:
            assert f"<name>{name}</name>" in section
        for name in BUILTIN_SKILLS:
            assert f"<name>{name}</name>" in section

    @pytest.mark.asyncio
    async def test_skill_tool_loads_only_neutral_shipped_skills(
        self, skill_tool: Skill
    ) -> None:
        manager = minimal_manager()
        ctx = InvokeContext(tool_call_id="t", skill_manager=manager)

        for name in NEUTRAL_SHIPPED_SKILLS:
            if not manager.available_skills[name].user_invocable:
                continue
            result = await collect_result(skill_tool.run(SkillArgs(name=name), ctx))
            assert result.name == name
        for name in CEREMONY_BEARING_SKILLS:
            with pytest.raises(ToolError):
                await collect_result(skill_tool.run(SkillArgs(name=name), ctx))

    def test_slash_invocation_parses_only_neutral_shipped_skills(self) -> None:
        manager = minimal_manager()

        for name in CEREMONY_BEARING_SKILLS:
            assert manager.parse_skill_command(f"/{name}") is None
            assert manager.parse_skill_command(f"/{name} extra instructions") is None
        for name in ("main-debugging", "main-git-workflow", "workspace", "web-search"):
            parsed = manager.parse_skill_command(f"/{name}")
            assert parsed is not None
            assert parsed.name == name

    @pytest.mark.asyncio
    async def test_ceremony_workflow_exposes_every_shipped_skill(
        self,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
        skill_tool: Skill,
    ) -> None:
        config = build_config(
            agent={"workflow": CEREMONY_WORKFLOW_ID},
            include_prompt_detail=True,
            include_model_info=False,
        )
        prompt = render_session_prompt(config, load_orchestrator)
        section = catalog_section(prompt)
        manager = SkillManager(lambda: config)
        ctx = InvokeContext(tool_call_id="t", skill_manager=manager)

        for name in SHIPPED_SKILL_NAMES:
            assert f"<name>{name}</name>" in section
            result = await collect_result(skill_tool.run(SkillArgs(name=name), ctx))
            assert result.name == name
            if manager.available_skills[name].user_invocable:
                assert manager.parse_skill_command(f"/{name}") is not None


class TestOneRuleThreeSurfaces:
    """Acceptance: the same compatibility rule on all three surfaces,
    pinned identically — per skill, exposure agrees across the surfaces.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", SHIPPED_SKILL_NAMES)
    async def test_exposure_agrees_across_catalog_tool_and_slash(
        self,
        name: str,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
        skill_tool: Skill,
    ) -> None:
        for config in (
            build_config(include_prompt_detail=True, include_model_info=False),
            build_config(
                agent={"workflow": CEREMONY_WORKFLOW_ID},
                include_prompt_detail=True,
                include_model_info=False,
            ),
        ):
            manager = SkillManager(lambda config=config: config)
            prompt = render_session_prompt(config, load_orchestrator)
            section = catalog_section(prompt)
            ctx = InvokeContext(tool_call_id="t", skill_manager=manager)

            eligible = manager.skill_eligibility(name).eligible
            exposed = name in manager.available_skills
            in_catalog = f"<name>{name}</name>" in section
            try:
                result = await collect_result(skill_tool.run(SkillArgs(name=name), ctx))
                tool_loads = result.name == name
            except ToolError:
                tool_loads = False
            info = manager.available_skills.get(name)
            slash_parses = (
                manager.parse_skill_command(f"/{name}") is not None
                if info is not None and info.user_invocable
                else None
            )

            assert exposed == eligible
            assert in_catalog == eligible
            assert tool_loads == eligible
            if slash_parses is not None:
                assert slash_parses == eligible


class TestFrozenPolicyEligibility:
    """Acceptance: eligibility reads the conversation's frozen policy, not
    pending saved selections.
    """

    def test_policy_facts_are_the_bound_selections(self) -> None:
        assert minimal_manager().policy_facts == ConversationPolicyFacts(
            identity="coder", workflow="minimal"
        )
        assert ceremony_manager().policy_facts == ConversationPolicyFacts(
            identity="coder", workflow=CEREMONY_WORKFLOW_ID
        )

    def test_pending_saved_selection_does_not_change_eligibility(self) -> None:
        served: dict[str, ChartreuxConfigSchema] = {"config": minimal_config()}
        manager = SkillManager(lambda: served["config"])

        # The saved selection changes and the config layer starts serving it
        # (a pending change: it applies to new conversations only).
        served["config"] = served["config"].model_copy(
            update={
                "agent": served["config"].agent.model_copy(
                    update={"workflow": CEREMONY_WORKFLOW_ID}
                )
            }
        )
        assert served["config"].agent.workflow == CEREMONY_WORKFLOW_ID

        # The bound conversation's eligibility is unchanged.
        assert manager.policy_facts.workflow == "minimal"
        for name in CEREMONY_BEARING_SKILLS:
            assert name not in manager.available_skills
            assert manager.skill_eligibility(name).eligible is False
            assert manager.parse_skill_command(f"/{name}") is None

    def test_pending_selection_reversed_does_not_change_eligibility(self) -> None:
        served: dict[str, ChartreuxConfigSchema] = {"config": ceremony_config()}
        manager = SkillManager(lambda: served["config"])

        served["config"] = served["config"].model_copy(
            update={
                "agent": served["config"].agent.model_copy(
                    update={"workflow": "minimal"}
                )
            }
        )

        assert manager.policy_facts.workflow == CEREMONY_WORKFLOW_ID
        for name in CEREMONY_BEARING_SKILLS:
            assert name in manager.available_skills
            assert manager.skill_eligibility(name).eligible
            assert manager.parse_skill_command(f"/{name}") is not None

    def test_explicit_frozen_facts_are_used_when_supplied(self) -> None:
        config = minimal_config()
        manager = SkillManager(
            lambda: config,
            policy_facts=ConversationPolicyFacts(
                identity="coder", workflow=CEREMONY_WORKFLOW_ID
            ),
        )
        assert "main-design" in manager.available_skills


class TestShippedSkillRewrites:
    """Acceptance: main-design/plan/review use purpose/seat canonical calls
    (package 8's grammar); execution review resolves the coder identity and
    the analysis seats stay read-only.

    The convergence-protocol checks below are text contracts over the shipped
    skill body: each one pins required wording and the absence of wording that
    would contradict it. They are not behavioral tests of model obedience, and
    prompt prose does not itself create a deterministic review state machine.
    """

    def test_design_and_plan_use_purpose_canonical_calls(self) -> None:
        design = skill_body("main-design")
        plan = skill_body("main-plan")

        assert 'purpose="design-analysis"' in design
        assert 'purpose="planning-analysis"' in plan
        assert 'task(task="Design analysis for:' in design
        assert 'task(task="Planning analysis for:' in plan
        assert 'purpose="exploration"' in design
        assert 'purpose="exploration"' in plan

    def test_review_uses_purpose_and_seat_canonical_calls(self) -> None:
        review = skill_body("main-review")

        assert 'purpose="review.quick"' in review
        assert 'purpose="review.standard"' in review
        assert 'purpose="review.deep"' in review
        assert 'seat="analytical"' in review
        assert 'seat="peer"' in review
        assert 'seat="execution"' in review
        assert 'purpose="<review purpose>"' in review
        assert 'seat="<seat>"' in review

    def test_execution_seat_binds_coder_and_analysis_seats_are_read_only(self) -> None:
        review = skill_body("main-review")

        assert "execution seat binds the `coder` identity" in review
        assert (
            "analytical and peer seats bind the read-only `reviewer` identity" in review
        )
        assert "it is not a read-only seat" in review

    def test_rewritten_skills_drop_retired_routing_language(self) -> None:
        for name in ("main-design", "main-plan", "main-review"):
            body = skill_body(name)
            for anchor in RETIRED_ROUTING_ANCHORS:
                assert anchor not in body, f"{name} still mentions {anchor!r}"
            assert "<slot" not in body
            assert "slot table" not in body
            assert "config={model=" not in body

    def test_acceptance_gates_are_preserved_in_ceremony_skills(self) -> None:
        # The ceremony-bearing skills keep their acceptance gates; the
        # minimal workflow simply does not load them.
        assert (
            "Obtain the user's explicit design acceptance before proceeding; "
            "continued discussion, questions, or silence are not acceptance."
            in skill_body("main-design")
        )
        assert (
            "Obtain the user's explicit plan acceptance before proceeding to "
            "implementation; continued discussion, questions, or silence are not "
            "acceptance." in skill_body("main-plan")
        )

    def test_no_skill_reinstates_mandatory_model_diversity(self) -> None:
        # Diversity requirements are declared by routes and validated by the
        # launch; no skill mandates them itself or pretends context
        # separation is diversity.
        for name in ("main-design", "main-plan", "main-review"):
            body = skill_body(name)
            assert "requires_distinct_models" not in body
            assert "require distinct models" not in body
            assert "must use distinct models" not in body
            assert "must use different models" not in body
            assert "enforce model diversity" not in body
        review = skill_body("main-review")
        assert (
            "a fresh reviewer context on the same model is context separation, "
            "not model diversity" in review
        )

    def test_quick_substitution_uses_model_line_families(self) -> None:
        review = skill_body("main-review")

        assert (
            "Model-line substitution applies only to quick single-seat reviews"
            in review
        )
        assert "`zai-glm*`" in review
        assert "`gpt-6*`" in review
        assert "`mistral*`" in review
        assert "it never applies to standard or deep reviews" in review

    def test_deep_review_has_no_authorship_substitution(self) -> None:
        review = skill_body("main-review")

        assert "deep panels use no authorship substitution" in review
        assert "regardless of the implementor" in review
        assert (
            "Dispatch each seat through the route with no per-seat overrides" in review
        )

    def test_deep_panel_acceptance_requires_all_seat_statuses_and_identities(
        self,
    ) -> None:
        review = skill_body("main-review")

        assert "gate acceptance requires status for every selected seat" in review
        assert "missing or incomplete reports treated explicitly" in review
        assert "resolved canonical-model identities" in review

    def test_routed_deep_admits_live_route_edits(self) -> None:
        review = skill_body("main-review")

        assert "LIVE route table" in review
        assert "applies to subsequent launches in the same conversation" in review
        assert "children already admitted keep their bindings" in review
        assert "not a validator defect" in review
        assert "rejected at reload" in review
        assert "the prior table stays effective" in review
        assert "FROZEN route table" not in review
        assert "saved config edits apply to new conversations only" not in review

    def test_routed_deep_requires_configured_multi_model_route(self) -> None:
        review = skill_body("main-review")

        assert "binds every seat to the session model" in review
        assert "routed deep dispatch" in review
        assert "configured multi-model route" in review

    def test_direct_fallback_inlines_overlay_body(self) -> None:
        review = skill_body("main-review")

        assert "explicit `model` and `identity` on the launch" in review
        assert "copied into `config.instructions`" in review

    def test_execution_brief_pins_ranked_probes_bounded_discovery_and_escalation(
        self,
    ) -> None:
        review = skill_body("main-review")

        assert "parent-ranked probe list" in review
        assert "one hypothesis, or about five turns" in review
        assert "25-30 turn schedule" in review
        assert "stop polishing established blockers" in review
        assert "INCOMPLETE" in review
        assert "never converted into a pass" in review

    def test_supplementary_seats_are_reviewer_identities(self) -> None:
        review = skill_body("main-review")

        assert "Supplementary seats, when the route configures them" in review
        assert "`luna`" not in review
        assert "`flash`" not in review
        assert (
            "bind the read-only `reviewer` identity, whose tool surface carries only the bash verification carve-out"
            in review
        )
        assert "intent, regression coverage, and evidence quality" in review
        assert "negative cases and boundary questions" in review

    def test_probe_seat_final_message_is_report(self) -> None:
        review = skill_body("main-review")

        assert "a probe seat's final message must be the report itself" in review

    def test_review_predeclares_rule_cap_and_full_read_convergence_scope(self) -> None:
        review = skill_body("main-review")

        assert "Before round 1 the main agent records the artifact set" in review
        assert "content-based baseline and target revision pins" in review
        assert (
            "one complete clean panel on the exact current revision, with all prior "
            "blockers independently verified resolved or explicitly user-accepted and "
            "no unresolved in-scope blockers" in review
        )
        assert "any artifact edit after that panel invalidates convergence" in review
        assert (
            "The convergence-granting panel scope is a full read of the exact "
            "candidate revision." in review
        )
        assert "one initial panel and at most two refinement panels" in review
        assert "any larger cap needs explicit user authorization" in review
        assert "An initial clean full-read panel may itself grant convergence" in review
        assert "Do not change these retrospectively to claim a pass." in review
        # Negative: the rule is fixed up front and the cap is not implicitly
        # extensible.
        assert "unlimited" not in review
        assert "extend the cap automatically" not in review
        assert "change the rule after a panel to claim a pass" not in review

    def test_review_counts_incomplete_panels_and_replaces_seats_in_place(self) -> None:
        review = skill_body("main-review")

        assert (
            "A panel dispatched with an incomplete seat counts toward the cap" in review
        )
        assert "does not consume another round" in review
        assert "Replace a missing seat only within the same pinned panel" in review
        assert (
            "an authorized fresh independent seat on the same route, model, and "
            "identity" in review
        )
        assert "record the failed launch and the replacement" in review
        assert "never silently substitute" in review
        # Negative: a lost seat is neither dropped nor quietly swapped.
        assert "an incomplete seat may be dropped" not in review
        assert "silently substitute a different identity" not in review

    def test_review_pins_content_and_treats_a_stale_target_as_not_converged(
        self,
    ) -> None:
        review = skill_body("main-review")

        assert "Pin content rather than a branch name" in review
        assert "canonical ordered artifact paths and content hashes" in review
        assert "the exact last-reviewed baseline and target hashes" in review
        assert "the immutable reviewer-visible ledger projection" in review
        assert "The mutable ledger stays outside the target digest" in review
        assert "Freeze the target while a panel runs" in review
        assert (
            "recompute the pins at synthesis and before any convergence claim" in review
        )
        assert "a changed artifact or criterion leaves the target stale" in review
        assert "ledger-only status updates change nothing" in review
        assert (
            "A missing or stale baseline, target, ledger, required seat, or panel "
            "makes the review INCOMPLETE, never converged" in review
        )
        # Negative: a branch name or a stale pin never stands in for the reviewed
        # content.
        assert "pinning Git HEAD is sufficient" not in review
        assert "the current branch is the baseline" not in review
        assert "a stale target may still converge" not in review

    def test_review_ledger_schema_and_dispositions_are_pinned(self) -> None:
        review = skill_body("main-review")

        assert "artifact-local adjudication ledger" in review
        assert "for an inline artifact retain an inline record" in review
        assert (
            "ID | round/seat | severity | claim+source | disposition | decision "
            "evidence | amendment/verification" in review
        )
        for disposition in (
            "open",
            "accepted-for-fix",
            "resolved",
            "rejected-needs-contrary-evidence",
            "deferred",
            "user-accepted",
        ):
            assert disposition in review
        assert (
            "Rejection requires cited contrary evidence, not a majority vote." in review
        )
        assert (
            "preserve IDs across consolidation, mapping split and merged claims"
            in review
        )
        # Negative: no invented disposition labels.
        assert "rejected-without-evidence" not in review
        assert "consensus disposition" not in review

    def test_review_refinement_briefs_carry_only_a_neutral_projection(self) -> None:
        review = skill_body("main-review")

        assert "without earlier reviewers' conclusions or votes" in review
        assert "include only the neutral ledger projection specified below" in review
        assert "A refinement brief gives the round number" in review
        assert "pinned baseline and target, exact delta, criteria" in review
        assert (
            "a neutral ledger projection containing only IDs, claims, disposition "
            "labels and references" in review
        )
        assert "never prior reports, counts, votes or reviewer conclusions" in review
        # Negative: the projection never carries conclusions or a vote count.
        assert "share earlier reviewers' conclusions" not in review
        assert "include prior votes" not in review

    def test_review_repeated_settled_claim_needs_cited_new_evidence(self) -> None:
        review = skill_body("main-review")

        assert (
            "Repetition of an adjudicated claim without cited new evidence is "
            "advisory" in review
        )
        assert "cited material new evidence can reopen it" in review
        assert "Reopen only with cited material new evidence" in review
        assert "recording the old and new state" in review
        assert (
            "a fresh independent seat may find that evidence on its own judgment"
            in review
        )
        # Negative: repetition alone neither reopens nor blocks.
        assert "a repeated claim reopens" not in review
        assert "repetition is blocking on its own" not in review

    def test_review_unverified_fix_is_not_resolved(self) -> None:
        review = skill_body("main-review")

        assert (
            "A fix remains accepted-for-fix until independently verified on the "
            "target revision" in review
        )
        assert "sufficiently discriminating observed check" in review
        assert "Deferred blockers remain blocking" in review
        assert "only explicit user acceptance records a risk as user-accepted" in review
        # Negative: an applied or planned fix is not a resolution.
        assert "is resolved once the fix is applied" not in review
        assert "a planned fix counts as resolved" not in review

    def test_review_delta_scope_covers_regressions_and_acceptance_contradictions(
        self,
    ) -> None:
        review = skill_body("main-review")

        assert (
            "The initial panel and the convergence-granting panel full-read the "
            "complete target and acceptance criteria." in review
        )
        assert "Refinement panels are delta-scoped" in review
        assert "they inspect prior blocker fixes" in review
        assert (
            "regressions and interactions introduced by the exact effective delta"
            in review
        )
        assert "evidenced acceptance-criterion contradictions anywhere" in review
        assert "with relevant surrounding context" in review
        # Negative: a delta round is not a licence to ignore acceptance criteria
        # outside the delta.
        assert "acceptance criteria apply only inside the delta" not in review
        assert "delta rounds ignore interactions" not in review

    def test_review_records_every_seat_disposition_and_never_drops_a_garbled_one(
        self,
    ) -> None:
        review = skill_body("main-review")

        assert (
            "collected, incomplete, garbled, submission-blocked, or cancelled" in review
        )
        assert "Assess the actual report for completeness, intelligibility" in review
        assert "a completion-audit stamp is not a substitute" in review
        assert "a failed launch is recorded as incomplete with its reason" in review
        assert "an honest incomplete report is not a zero-finding seat" in review
        assert "actual seat identities and models" in review
        assert "run IDs or timestamps, evidence pointers" in review
        # Negative: an audit stamp or a garbled report is not a collected seat.
        assert "a completion-audit stamp is sufficient" not in review
        assert "garbled reports may be discarded" not in review

    def test_review_exhausted_cap_reports_not_converged(self) -> None:
        review = skill_body("main-review")

        assert (
            "When the cap is reached, report NOT CONVERGED with the consolidated "
            "ledger" in review
        )
        assert "remaining blockers, missing evidence, seat states" in review
        assert "do not auto-extend and do not imply a pass" in review
        assert (
            "A stale or incomplete panel cannot be retroactively made converged."
            in review
        )
        assert (
            "At the cap, consolidate for reporting and claim no convergence." in review
        )
        # Negative: cap exhaustion never becomes a pass or an extension.
        assert "at the cap the panel passes by default" not in review
        assert "extend the cap when blockers remain" not in review

    def test_review_artifact_edit_after_a_clean_panel_invalidates_convergence(
        self,
    ) -> None:
        review = skill_body("main-review")

        assert "any artifact edit after that panel invalidates convergence" in review
        assert (
            "a consolidation that keeps the reviewed normative content byte-for-byte "
            "may rely on that panel by updating only the separately pinned ledger"
            in review
        )
        assert (
            "any normative edit makes a new candidate needing a new qualifying "
            "full-read panel within the budget" in review
        )
        # Negative: an edit cannot ride on the earlier clean panel.
        assert (
            "edits after a clean panel keep the earlier convergence claim" not in review
        )
        assert "a post-panel edit may be passed without review" not in review

    def test_review_consolidates_the_artifact_after_every_round(self) -> None:
        review = skill_body("main-review")

        assert (
            "After every panel consolidate accepted amendments into the one "
            "authoritative artifact" in review
        )
        assert "preserving ledger IDs and history by reference" in review
        assert (
            "consolidation creates a new target that needs the review the selected "
            "rule requires" in review
        )
        # Negative: consolidation is not deferred to a single final round.
        assert "consolidate only after the final round" not in review
        assert "defer consolidation" not in review

    def test_review_records_warnings_without_amending_them(self) -> None:
        review = skill_body("main-review")

        assert "Amend accepted blockers only" in review
        assert (
            "record warnings without amendment unless new evidence shows one meets "
            "the declared blocker rule or the user directs the change" in review
        )
        # Negative: warnings do not become amendments by default.
        assert "amend every warning" not in review
        assert "warnings are accepted for fix" not in review

    def test_review_briefs_require_direct_report_and_full_disclosure(self) -> None:
        review = skill_body("main-review")

        assert "return the report directly without file writes" in review
        assert "authorized scratchpad-only test outputs and probes" in review
        assert "Never copy the whole repository with cp -r" in review
        assert "use git archive for a necessary tracked-tree copy" in review
        assert "disclose any omitted untracked inputs" in review
        assert "complete denial record with runtime reasons" in review
        assert "cumulative touched-path disclosure, including reverted paths" in review
        assert "an honest empty set stays explicit" in review
        # Negative: reviewers do not write into the repository.
        assert "reviewers may write" not in review
        assert "cp -r is acceptable" not in review

    def test_review_clean_panel_is_not_user_acceptance(self) -> None:
        review = skill_body("main-review")

        assert "A finding from two or more reviewers is consensus, not proof." in review
        assert "Apply the selected convergence rule" in review
        assert "a clean review is not user acceptance" in review
        assert (
            "A blocker remains open until independently verified resolved, rejected "
            "on cited contrary evidence, or explicitly accepted by the user." in review
        )
        # Negative: consensus or a clean panel never substitutes for acceptance.
        assert "consensus is proof" not in review
        assert "a clean review is user acceptance" not in review

    def test_design_and_plan_apply_the_review_protocol_to_their_artifact(self) -> None:
        for name, artifact in (("main-design", "design"), ("main-plan", "plan")):
            body = skill_body(name)

            assert (
                f"When reviewing a {artifact}, apply main-review's predeclared "
                "convergence rule, cap, fresh-seat scopes, completeness checks, and "
                "consolidation." in body
            )
            assert (
                "After each panel the main agent updates the artifact-local "
                "adjudication ledger before any further dispatch" in body
            )
            assert f"for an inline {artifact} keep the ledger inline" in body
            assert "Link the ledger from task state rather than duplicating it." in body
            # Negative: the ledger is neither copied into task state nor written
            # once for the whole review chain.
            assert f"copy the {artifact} ledger into task state" not in body
            assert "update the ledger once after the review chain" not in body

    def test_design_and_plan_ledger_does_not_replace_explicit_acceptance(self) -> None:
        for name, artifact, gate in (
            (
                "main-design",
                "design",
                "Obtain the user's explicit design acceptance before proceeding; "
                "continued discussion, questions, or silence are not acceptance.",
            ),
            (
                "main-plan",
                "plan",
                "Obtain the user's explicit plan acceptance before proceeding to "
                "implementation; continued discussion, questions, or silence are "
                "not acceptance.",
            ),
        ):
            body = skill_body(name)

            assert gate in body
            assert f"A clean review is not {artifact} acceptance." in body
            # Negative: neither a clean panel nor a converged ledger grants the
            # acceptance only the user gives.
            assert f"a clean review is {artifact} acceptance" not in body
            assert "the ledger replaces the approval gate" not in body


class TestNoCeremonyInMinimalSessions:
    """Acceptance: no skill silently reinstates retired slots, mandatory
    model diversity, or ceremony in minimal sessions — the minimal session's
    rendered behavior contains no ceremony anchors from skill content.
    """

    def test_minimal_rendered_skill_content_has_no_ceremony_anchors(
        self,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> None:
        config = build_config(include_prompt_detail=True, include_model_info=False)
        manager = SkillManager(lambda: config)
        prompt = render_session_prompt(config, load_orchestrator)

        rendered = [catalog_section(prompt)]
        rendered.extend(info.prompt for info in manager.available_skills.values())
        rendered.extend(info.description for info in manager.available_skills.values())
        text = "\n".join(rendered)

        for anchor in CEREMONY_ANCHORS:
            assert anchor not in text, (
                f"minimal session leaks ceremony anchor {anchor!r}"
            )

    @pytest.mark.asyncio
    async def test_minimal_session_cannot_load_ceremony_bearing_content(
        self, skill_tool: Skill
    ) -> None:
        manager = minimal_manager()
        ctx = InvokeContext(tool_call_id="t", skill_manager=manager)

        for name in CEREMONY_BEARING_SKILLS:
            with pytest.raises(ToolError) as excinfo:
                await collect_result(skill_tool.run(SkillArgs(name=name), ctx))
            message = str(excinfo.value)
            # The withheld skill's content never reaches the model.
            assert "<skill_content" not in message
            assert "# Skill:" not in message


class TestIncompatibleInvocationDiagnostic:
    """Acceptance: an incompatible explicit invocation produces an actionable
    diagnostic naming the workflow/identity requirement.
    """

    @pytest.mark.asyncio
    async def test_skill_tool_error_names_the_requirement(
        self, skill_tool: Skill
    ) -> None:
        manager = minimal_manager()
        ctx = InvokeContext(tool_call_id="t", skill_manager=manager)

        with pytest.raises(ToolError) as excinfo:
            await collect_result(skill_tool.run(SkillArgs(name="main-design"), ctx))
        message = str(excinfo.value)

        assert "main-design" in message
        assert "not found" not in message
        # The requirement is named: the ceremony workflow, against the
        # conversation's actual workflow and identity.
        assert CEREMONY_WORKFLOW_ID in message
        assert "minimal" in message
        assert "coder" in message
        # The remedy is actionable: a new conversation with the ceremony
        # workflow selected.
        assert "new conversation" in message
        assert f'[agent] workflow = "{CEREMONY_WORKFLOW_ID}"' in message

    @pytest.mark.asyncio
    async def test_manager_diagnostic_matches_the_tool_error(
        self, skill_tool: Skill
    ) -> None:
        manager = minimal_manager()
        ctx = InvokeContext(tool_call_id="t", skill_manager=manager)

        diagnostic = manager.incompatibility_diagnostic("main-review")
        assert diagnostic is not None
        with pytest.raises(ToolError) as excinfo:
            await collect_result(skill_tool.run(SkillArgs(name="main-review"), ctx))
        assert str(excinfo.value) == diagnostic

    def test_slash_invocation_of_withheld_skill_is_filtered(self) -> None:
        manager = minimal_manager()
        assert manager.parse_skill_command("/main-design") is None
        # The diagnostic text is available to the invocation surface.
        diagnostic = manager.incompatibility_diagnostic("main-design")
        assert diagnostic is not None
        assert CEREMONY_WORKFLOW_ID in diagnostic

    def test_eligible_and_unknown_skills_have_no_diagnostic(self) -> None:
        manager = minimal_manager()
        assert manager.incompatibility_diagnostic("main-debugging") is None
        assert manager.incompatibility_diagnostic("nonexistent-skill") is None
        assert ceremony_manager().incompatibility_diagnostic("main-design") is None

    def test_diagnostic_is_actionable_for_every_withheld_skill(self) -> None:
        manager = minimal_manager()
        for name in CEREMONY_BEARING_SKILLS:
            diagnostic = manager.incompatibility_diagnostic(name)
            assert diagnostic is not None
            assert name in diagnostic
            assert CEREMONY_WORKFLOW_ID in diagnostic
            assert "new conversation" in diagnostic


class TestInvariants:
    """Plan invariants: discovery precedence intact; compatibility filtering
    is not automatic catalog pruning; personal shadow copies are not
    rewritten or filtered.
    """

    def test_filtering_is_not_catalog_pruning(self) -> None:
        manager = minimal_manager()

        # The discovery inventory still lists every shipped skill...
        assert set(SHIPPED_SKILL_NAMES) <= set(manager.settings_inventory)
        assert set(BUILTIN_SKILLS) <= set(manager.settings_inventory)
        # ...marked with the workflow requirement for the withheld ones.
        assert set(manager.workflow_incompatibilities) == set(CEREMONY_BEARING_SKILLS)
        for requirement in manager.workflow_incompatibilities.values():
            assert CEREMONY_WORKFLOW_ID in requirement
            assert "minimal" in requirement

    def test_discovery_precedence_is_intact(self, tmp_path: Path) -> None:
        # A local skill shadows the shipped copy of the same name (first
        # match wins), and the workflow rule never demotes it.
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        create_skill(skills_dir, "main-debugging", "My personal debugging skill")
        config = build_test_vibe_config(skill_paths=[skills_dir])
        manager = SkillManager(lambda: config)

        skill = manager.get_skill("main-debugging")
        assert skill is not None
        assert skill.source is SkillSource.LOCAL
        assert skill.description == "My personal debugging skill"

    def test_personal_shadow_copy_of_a_ceremony_bearing_name_stays_eligible(
        self, tmp_path: Path
    ) -> None:
        # Personal shadow copies (user-level skill dirs) are the user's own
        # content: the harness classifies only its shipped copies.
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        create_skill(skills_dir, "main-design", "My personal design skill")
        config = build_test_vibe_config(skill_paths=[skills_dir])
        manager = SkillManager(lambda: config)

        skill = manager.get_skill("main-design")
        assert skill is not None
        assert skill.source is SkillSource.LOCAL
        assert manager.incompatibility_diagnostic("main-design") is None

    def test_enabled_disabled_filters_apply_on_top_of_the_workflow_rule(
        self, tmp_path: Path
    ) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        create_skill(skills_dir, "local-skill", "A local skill")
        config = build_test_vibe_config(
            skill_paths=[skills_dir], disabled_skills=["main-debugging"]
        )
        manager = SkillManager(lambda: config)

        assert "local-skill" in manager.available_skills
        assert "main-debugging" not in manager.available_skills
        for name in CEREMONY_BEARING_SKILLS:
            assert name not in manager.available_skills

    def test_format_incompatibility_diagnostic_names_the_requirement(self) -> None:
        facts = ConversationPolicyFacts(identity="orchestrator", workflow="minimal")
        diagnostic = format_incompatibility_diagnostic(
            "main-plan", source=SkillSource.SHIPPED, facts=facts
        )
        assert "main-plan" in diagnostic
        assert CEREMONY_WORKFLOW_ID in diagnostic
        assert "minimal" in diagnostic
        assert "orchestrator" in diagnostic
        assert "new conversation" in diagnostic
