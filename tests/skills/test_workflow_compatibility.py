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

        assert "Supplementary seats" in review
        assert "`luna`" in review
        assert "`flash`" in review
        assert (
            "bind the read-only `reviewer` identity, whose tool surface carries only the bash verification carve-out"
            in review
        )
        assert "intent, regression coverage, and evidence quality" in review
        assert "negative cases and boundary questions" in review

    def test_flash_seat_final_message_is_report(self) -> None:
        review = skill_body("main-review")

        assert "flash's final message must be the report itself" in review


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
