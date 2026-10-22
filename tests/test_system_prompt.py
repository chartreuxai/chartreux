from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from chartreux.core.agents import AgentManager
from chartreux.core.agents.identities import (
    UnknownIdentityError,
    validate_identity_selection,
)
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.dispatch import SHIPPED_PURPOSES
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.policy.resources import ResourceResolutionError
from chartreux.core.prompts import PROMPTS_DIR, UtilityPrompt, load_prompt
from chartreux.core.recovery_policy import (
    UNATTENDED_RESILIENCE_MARKER,
    UNATTENDED_RESILIENCE_RULE,
)
from chartreux.core.scratchpad import init_scratchpad
from chartreux.core.skills.manager import SkillManager
from chartreux.core.system_prompt import get_universal_system_prompt
from tests.conftest import ConfigBuilder, OrchestratorLoader, multi_model_catalog

_PROMPTS_ROOT = Path(__file__).parents[1] / "chartreux" / "core" / "prompts"
_TASK_PROMPT = (
    Path(__file__).parents[1] / "chartreux/core/tools/builtins/prompts/task.md"
)

# The retired SystemPrompt enum ids (package 16b): the enum, its loader, and
# the seven enum-mapped bodies are gone; their content lives on in the
# composition corpus (identities/fragments/workflows).
_RETIRED_ENUM_IDS = (
    "cli",
    "explore",
    "tests",
    "minimal",
    "worker",
    "advisor",
    "reviewer",
)
_UNKNOWN_PROMPT_IDS = ("cli_2026-07_v2", "cli_2026-08_v3")

# The surviving load path takes its built-in bodies from an explicit map —
# here the utility prompts — never by scanning the prompts directory.
_UTILITY_BUILTINS = {prompt.value: prompt.path for prompt in UtilityPrompt}


def _identity_body(name: str) -> str:
    return (_PROMPTS_ROOT / "identities" / f"{name}.md").read_text(encoding="utf-8")


def _fragment(name: str) -> str:
    return (_PROMPTS_ROOT / "fragments" / f"{name}.md").read_text(encoding="utf-8")


def _workflow(name: str) -> str:
    return (_PROMPTS_ROOT / "workflows" / f"{name}.md").read_text(encoding="utf-8")


def _compose_child_prompt(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    identity: str,
) -> str:
    config = build_config(
        agent={"identity": identity},
        include_model_info=False,
        include_commit_signature=False,
    )
    return get_universal_system_prompt(
        config,
        SkillManager(lambda: config),
        AgentManager(load_orchestrator(config)),
        is_subagent=True,
    )


def _compose_orchestrator_prompt(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> str:
    config = build_config(
        agent={
            "identity": "orchestrator",
            "workflow": "ceremony",
            "orchestration_strategy": "builtin:default",
        },
        include_model_info=False,
        include_commit_signature=False,
    )
    config.attach_catalog_snapshot(
        CatalogSnapshot(multi_model_catalog(), "routing-test")
    )
    return get_universal_system_prompt(
        config, SkillManager(lambda: config), AgentManager(load_orchestrator(config))
    )


def test_system_prompt_reports_resolved_model_when_unpinned(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    # The unpinned default (active_model == "") must still report the resolved
    # model alias, not an empty name.
    config = build_config(
        include_model_info=True,
        include_prompt_detail=False,
        include_commit_signature=False,
    )
    assert config.active_model == ""
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(load_orchestrator(config))

    prompt = get_universal_system_prompt(config, skill_manager, agent_manager)

    assert f"Your model name is: `{config.get_active_model().alias}`" in prompt
    assert "Your model name is: ``" not in prompt


def test_model_catalog_section_renders_thin_descriptions_and_routes_by_vocabulary(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(
        active_model="base",
        include_prompt_detail=False,
        include_project_context=False,
        include_commit_signature=False,
    )
    catalog = ModelCatalog.model_validate({
        "providers": {"test-provider": {"api_base": "https://example.test"}},
        "models": {
            name: {
                "disabled": name == "disabled",
                "deployments": [{"provider": "test-provider", "name": f"wire-{name}"}],
            }
            for name in ("base", "backup", "disabled")
        },
        "roles": {
            "worker": {
                "model": "base",
                "thinking": "high",
                "description": "worker preset",
            },
            "orchestrator": {
                "model": "base",
                "thinking": "high",
                "description": "main assistant preset",
            },
            "quick": {
                "model": "base",
                "thinking": "low",
                "description": "quick preset",
            },
            "standard": {
                "model": "base",
                "thinking": "medium",
                "description": "standard preset",
            },
            "deep": {"model": "base", "thinking": "high", "description": "deep preset"},
            "unavailable": {
                "model": "disabled",
                "thinking": "high",
                "description": "unavailable preset",
            },
        },
    })
    config.attach_catalog_snapshot(CatalogSnapshot(catalog, "prompt-test"))
    prompt = get_universal_system_prompt(
        config, SkillManager(lambda: config), AgentManager(load_orchestrator(config))
    )

    assert "# Model catalog" in prompt
    assert "| Canonical name | Display name | Provider |" in prompt
    assert "| base | test-provider/wire-base | test-provider |" in prompt
    assert "| backup | test-provider/wire-backup | test-provider |" in prompt
    # The catalog section renders the thin preset descriptions from the catalog.
    assert "| @worker | base | high | worker preset |" in prompt
    assert "| @quick | base | low | quick preset |" in prompt
    assert "| disabled |" not in prompt
    assert "@unavailable" not in prompt
    # The routing renders from the package-4 route facts: the launch-target
    # table binds every shipped default target to the session model `base`,
    # so the rendering is model-bound and role-free rather than role-routed.
    assert "### Launch targets" in prompt
    assert "| Purpose | Identity | Tier | Seat | Model |" in prompt
    assert "| `implementation` | `coder` | substantive | — | `base` |" in prompt
    assert "### Purpose meanings" in prompt
    assert "| `mechanical` | `worker` |" not in prompt
    assert "`@scout` for search" not in prompt
    assert "$dispatch" not in prompt

    restricted = config.model_copy(update={"allowed_models": ["base"]})
    restricted_prompt = get_universal_system_prompt(
        restricted,
        SkillManager(lambda: restricted),
        AgentManager(load_orchestrator(restricted)),
    )
    assert "| backup |" not in restricted_prompt
    assert "| @worker | base | high | worker preset |" in restricted_prompt


def test_subagent_child_omits_model_catalog_without_role_instructions(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(
        include_prompt_detail=False,
        include_project_context=False,
        include_commit_signature=False,
    )
    prompt = get_universal_system_prompt(
        config,
        SkillManager(lambda: config),
        AgentManager(load_orchestrator(config)),
        role_instructions=None,
        is_subagent=True,
    )
    assert "Your model name is:" in prompt
    assert "# Model catalog" not in prompt
    assert "Do not spawn, delegate to, or coordinate nested subagents." in prompt


def test_model_catalog_omits_empty_section(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_config(
        include_prompt_detail=False,
        include_project_context=False,
        include_commit_signature=False,
    )
    monkeypatch.setattr(ChartreuxConfigSchema, "available_models", lambda self: {})
    prompt = get_universal_system_prompt(
        config, SkillManager(lambda: config), AgentManager(load_orchestrator(config))
    )
    assert "Your model name is:" in prompt
    assert "# Model catalog" not in prompt
    assert "| Canonical name |" not in prompt


def test_model_catalog_escapes_configured_table_cells(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_config(
        active_model="base",
        include_prompt_detail=False,
        include_project_context=False,
        include_commit_signature=False,
    )
    catalog = ModelCatalog.model_validate({
        "providers": {"test-provider": {"api_base": "https://example.test"}},
        "models": {
            "base": {
                "deployments": [{"provider": "test-provider", "name": "wire-base"}]
            }
        },
        "roles": {
            "worker": {
                "model": "base",
                "thinking": "high",
                "description": "Bounded | work\nwith care",
            },
            "orchestrator": {
                "model": "base",
                "thinking": "high",
                "description": "main assistant preset",
            },
            "quick": {
                "model": "base",
                "thinking": "low",
                "description": "quick preset",
            },
            "standard": {
                "model": "base",
                "thinking": "medium",
                "description": "standard preset",
            },
            "deep": {"model": "base", "thinking": "high", "description": "deep preset"},
        },
    })
    config.attach_catalog_snapshot(CatalogSnapshot(catalog, "escaped"))
    model = config.get_active_model().model_copy(
        update={"display_name": "Local | Model\nsecond line"}
    )
    monkeypatch.setattr(
        ChartreuxConfigSchema, "available_models", lambda self: {"base": model}
    )
    prompt = get_universal_system_prompt(
        config, SkillManager(lambda: config), AgentManager(load_orchestrator(config))
    )
    assert "| base | Local \\| Model second line | test-provider |" in prompt
    assert "| @worker | base | high | Bounded \\| work with care |" in prompt


@pytest.mark.parametrize(
    ("model_info", "role_instructions", "attached"),
    [(False, None, True), (True, "Child instructions", True), (True, None, False)],
)
def test_model_catalog_section_is_gated_and_catalogless_is_safe(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    model_info: bool,
    role_instructions: str | None,
    attached: bool,
) -> None:
    config = build_config(
        include_model_info=model_info,
        include_prompt_detail=False,
        include_project_context=False,
        include_commit_signature=False,
    )
    if not attached:
        config.attach_catalog_snapshot(None)
    prompt = get_universal_system_prompt(
        config,
        SkillManager(lambda: config),
        AgentManager(load_orchestrator(config)),
        role_instructions=role_instructions,
    )
    assert "# Model catalog" not in prompt
    if role_instructions is not None:
        assert role_instructions in prompt


def test_shipped_catalog_role_descriptions_are_thin_and_vocabulary_carries_routing() -> (
    None
):
    roles = SHIPPED_CATALOG.roles

    # Role descriptions are thin preset labels; the task-routing vocabulary
    # renders from the developer-owned dispatch purposes, not from the catalog.
    assert roles["main"].description == "main assistant preset"
    assert roles["worker"].description == "worker preset"
    assert roles["scout"].description == "scout preset"
    assert roles["heavy"].description == "heavy preset"
    # The retired capacity wording is gone from every role description.
    for role in roles.values():
        assert "capacity preset" not in role.description
    # The routing semantics live in the purpose vocabulary.
    assert set(SHIPPED_PURPOSES) == {
        "research",
        "search",
        "exploration",
        "verification",
        "mechanical-edit",
        "implementation",
        "implementation-demanding-settled",
        "design-analysis",
        "planning-analysis",
        "review.quick",
        "review.standard",
        "review.deep",
    }
    assert "never implementation" in SHIPPED_PURPOSES["design-analysis"].description
    assert "never implementation" in SHIPPED_PURPOSES["planning-analysis"].description
    assert (
        "settled approach"
        in SHIPPED_PURPOSES["implementation-demanding-settled"].description
    )
    assert (
        "report only checks actually run"
        in SHIPPED_PURPOSES["verification"].description
    )


def test_commit_signature_uses_literal_shell_syntax(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(
        include_model_info=False,
        include_prompt_detail=False,
        include_commit_signature=True,
    )
    prompt = get_universal_system_prompt(
        config, SkillManager(lambda: config), AgentManager(load_orchestrator(config))
    )

    assert "Co-Authored-By: Chartreux" in prompt
    assert "git commit -m '<Commit message here>'" in prompt
    assert "$(" not in prompt
    assert "<<" not in prompt


def test_scratchpad_section_included_when_passed(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    sp = init_scratchpad("test-session")
    config = build_config(
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
    )
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(load_orchestrator(config))

    prompt = get_universal_system_prompt(
        config, skill_manager, agent_manager, scratchpad_dir=sp
    )

    assert "# Scratchpad Directory" in prompt
    assert sp is not None
    assert str(sp) in prompt


def test_scratchpad_section_absent_when_not_passed(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
    )
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(load_orchestrator(config))

    prompt = get_universal_system_prompt(config, skill_manager, agent_manager)

    assert "Scratchpad Directory" not in prompt


def test_headless_section_included_when_enabled(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(include_model_info=False, include_commit_signature=False)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(load_orchestrator(config))

    prompt = get_universal_system_prompt(
        config, skill_manager, agent_manager, headless=True
    )

    # The headless execution context is step 4 of the composition.
    assert "## Execution context — headless main session" in prompt
    assert "non-interactive (headless) session" in prompt
    assert "interactive approval prompt" in prompt
    assert "no user watching the loop" in prompt


def test_headless_section_absent_by_default(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(include_model_info=False, include_commit_signature=False)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(load_orchestrator(config))

    prompt = get_universal_system_prompt(config, skill_manager, agent_manager)

    assert "Headless Mode" not in prompt
    assert "## Execution context — headless main session" not in prompt
    assert "## Execution context — interactive main session" in prompt


def test_current_date_placeholder_substituted_in_prompt(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(include_model_info=False, include_commit_signature=False)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(load_orchestrator(config))

    prompt = get_universal_system_prompt(config, skill_manager, agent_manager)

    today = date.today()
    expected = f"Today's date is {today.isoformat()} ({today.strftime('%A')})."
    assert expected in prompt
    assert "$current_date" not in prompt


def test_default_prompt_composes_the_coder_minimal_selection(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(include_model_info=False, include_commit_signature=False)
    prompt = get_universal_system_prompt(
        config, SkillManager(lambda: config), AgentManager(load_orchestrator(config))
    )

    # The composed default opens at character zero with the coder identity.
    assert prompt.startswith(
        "You are Chartreux, an independently maintained CLI coding agent."
    )
    assert "## Coder identity" in prompt
    # The minimal workflow body is step 5.
    assert "# Workflow: minimal" in prompt
    assert "run the relevant checks yourself" in prompt
    # The legacy skeleton's delegation protocol section is gone: the
    # direct-tool posture line renders in step 3 instead.
    assert "## Delegation protocol" not in prompt
    assert "## Orchestration" not in prompt
    assert "## Background subagents" not in prompt
    assert "**Delegation posture.** Delegation is optional:" in prompt
    assert "Direct tools: the full granted tool set." in prompt
    # Routing is the launch-target table, not a slot table.
    assert "### Launch targets" in prompt
    assert "### Authoritative active slot bindings" not in prompt
    # The closing re-anchor is last.
    assert prompt.rstrip().endswith("this closing note adds no new rules.")
    assert "$role" not in prompt
    assert "$dispatch" not in prompt
    assert "`@scout` for search" not in prompt


def test_system_prompt_enum_and_enum_loader_are_retired() -> None:
    from chartreux.core import prompts

    # Package 16b retired the enum-based full-skeleton prompt axis: the
    # SystemPrompt enum, load_system_prompt, and the seven enum-mapped
    # bodies are gone (their case-insensitive enum lookup went with them).
    # What survives is the utility/custom load path.
    assert not hasattr(prompts, "SystemPrompt")
    assert not hasattr(prompts, "load_system_prompt")
    for retired in _RETIRED_ENUM_IDS:
        assert not (PROMPTS_DIR / f"{retired}.md").exists()
    for survivor in (
        "PROMPTS_DIR",
        "Prompt",
        "UtilityPrompt",
        "MissingPromptFileError",
        "custom_prompt_ids",
        "load_prompt",
    ):
        assert hasattr(prompts, survivor)


@pytest.mark.parametrize("prompt_id", [*_RETIRED_ENUM_IDS, *_UNKNOWN_PROMPT_IDS])
def test_retired_and_unknown_prompt_ids_have_no_builtin_body(
    mock_prompts_dirs: tuple[Path, Path], prompt_id: str
) -> None:
    # The retired enum ids (and unknown ids) resolve to no built-in body:
    # the surviving load path finds neither a shipped file (the builtins map
    # is explicit) nor a user custom file.
    with pytest.raises(ResourceResolutionError, match="cannot read resource"):
        load_prompt(
            prompt_id, setting_name="system_prompt_id", builtins=_UTILITY_BUILTINS
        )
    assert not (PROMPTS_DIR / f"{prompt_id}.md").exists()


@pytest.mark.parametrize("prompt_id", [prompt.value for prompt in UtilityPrompt])
def test_utility_prompt_ids_load_through_the_surviving_path(
    mock_prompts_dirs: tuple[Path, Path], prompt_id: str
) -> None:
    # Utility/compaction prompts survive the enum retirement untouched: every
    # built-in utility id loads through the strict loader branch.
    assert (
        load_prompt(
            prompt_id, setting_name="compaction_prompt_id", builtins=_UTILITY_BUILTINS
        )
        == UtilityPrompt(prompt_id).read()
    )


@pytest.mark.parametrize("prompt_id", ["custom_selection", "cli", *_UNKNOWN_PROMPT_IDS])
def test_custom_prompt_files_load_from_the_user_directory(
    mock_prompts_dirs: tuple[Path, Path], prompt_id: str
) -> None:
    project_prompts, user_prompts = mock_prompts_dirs
    (user_prompts / f"{prompt_id}.md").write_text(
        "  User custom prompt\n", encoding="utf-8"
    )
    assert (
        load_prompt(
            prompt_id, setting_name="system_prompt_id", builtins=_UTILITY_BUILTINS
        )
        == "User custom prompt"
    )

    # Project prompt directories are retired (package 16a): a file written
    # there is reported through the retired-dirs surface and never consumed,
    # so the user prompt keeps winning.
    (project_prompts / f"{prompt_id}.md").write_text(
        "  Project custom prompt\n", encoding="utf-8"
    )
    assert (
        load_prompt(
            prompt_id, setting_name="system_prompt_id", builtins=_UTILITY_BUILTINS
        )
        == "User custom prompt"
    )


@pytest.mark.parametrize("prompt_id", [prompt.value for prompt in UtilityPrompt])
def test_builtin_utility_prompts_cannot_be_shadowed_by_user_files(
    mock_prompts_dirs: tuple[Path, Path], prompt_id: str
) -> None:
    # Reserved built-in names cannot be shadowed: a user file reusing a
    # built-in utility id never wins over the shipped body.
    _, user_prompts = mock_prompts_dirs
    (user_prompts / f"{prompt_id}.md").write_text("user shadow\n", encoding="utf-8")
    assert (
        load_prompt(
            prompt_id, setting_name="compaction_prompt_id", builtins=_UTILITY_BUILTINS
        )
        == UtilityPrompt(prompt_id).read()
    )


@pytest.mark.parametrize("identity", ["coder", "advisor", "reviewer"])
def test_composed_child_prompts_contain_the_bounded_child_contract(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    identity: str,
) -> None:
    prompt = _compose_child_prompt(build_config, load_orchestrator, identity)

    # The retired role prompts' shared subagent contract survives as the
    # bounded-child fragment, composed into every child prompt.
    for clause in (
        "Perform one bounded assignment",
        "return a structured blocker",
        "Do not spawn, delegate to, or coordinate nested subagents.",
        "Never read, modify, create, or disclose `.env` files.",
        "Preserve unrelated changes",
        "Validate the result with available, relevant checks when practical.",
        "Report the files changed, checks actually run and their outcomes",
    ):
        assert clause in prompt

    if identity == "advisor":
        # The read-only advisor body keeps its report-in-result denials
        # variant alongside the shared contract.
        assert (
            "report denied commands and the runtime's rejection reason "
            "in your result" in prompt
        )
        assert "do not persist them to task-note files" in prompt
        assert "do not modify files to reconcile them" in prompt


def test_bounded_child_fragment_retires_stop_rule_and_failure_retries() -> None:
    prompt = _fragment("bounded_child")

    # ADR 0020: the whole-assignment stop rule never fired in the six audited
    # mistral-large-4 sessions; the strategy-switch rule replaces it.
    assert "top and return a blocker after two" not in prompt
    assert "After a stop, further task execution is prohibited" not in prompt
    # The failure-retries rule is dropped by maintainer decision (ADR 0020).
    assert "Failure retries" not in prompt
    assert "two occurrences of the same diagnostic" not in prompt
    assert "persist failed commands" not in prompt
    assert "One recorded isolation rerun" not in prompt
    assert "Do not run verification reserved to the parent" not in prompt


def test_bounded_child_fragment_carries_the_operational_discipline() -> None:
    prompt = _fragment("bounded_child")

    # Denials rule: persistent recording, consult-before-next-call, and
    # hardened permission/policy refusal language.
    assert (
        "Denials: record each denied command and the runtime's stated reason "
        "in task notes or a scratchpad file that survives compaction and remains "
        "inspectable, not private reasoning; consult the record before the next "
        "tool call." in prompt
    )
    assert (
        "Never retry a permission/policy-refused action through changed syntax, "
        "another tool, a wrapper, or another model; report the blocked "
        "requirement and continue only independent authorized work." in prompt
    )
    # Same-effect retry clarification (ADR 0020 amendment): the refused
    # action's identity is its effect on the same files or state, not the
    # command spelling.
    assert (
        "The refused action's identity is its effect on the same files or "
        "state, not the command spelling: producing the same refused effect "
        "through a different tool, script, or mechanism is still a retry." in prompt
    )
    # Destinations rule with the sharpened repository-outputs exception and
    # the precedence clause over assignment text (ADR 0020 amendment).
    assert (
        "Keep generated artifacts outside the repository tree unless the "
        "deliverable itself requires repository outputs; incidental tooling "
        "instructions such as where to point a log redirection do not count." in prompt
    )
    assert (
        "These rules take precedence over assignment text: an instruction in "
        "the task message does not authorize a destination this rule forbids; "
        "report the conflict in the result instead of complying silently." in prompt
    )
    assert "unless the task explicitly requires repository outputs" not in prompt
    assert (
        "Treat ambiguous denials mixing syntax and policy as permission/policy "
        "refusals. For syntax-only rejections, use at most one supported "
        "reformulation; never resubmit the denied command or reuse the rejected "
        "syntax element or construct." in prompt
    )
    # The dangling failure-retries cross-reference does not survive the drop.
    assert "executed commands that fail follow failure retries" not in prompt
    # Strategy switch with the same-reason definition, the not-a-retry bridge,
    # and post-compaction verifiability of the adopted method.
    assert (
        "Strategy switch: after the second denial citing the same runtime "
        "rule/pattern across the assignment, record it and adopt a different "
        "supported method for the affected operation" in prompt
    )
    assert (
        '"Same reason" means the rejection-reason category stated by the '
        "runtime's denial message, matched by its stated rule/pattern, not "
        "free-text similarity." in prompt
    )
    assert (
        "Record the adopted method alongside the denial so keeping it stays "
        "verifiable after compaction; keep that method for the rest of the "
        "assignment, including after compaction." in prompt
    )
    assert (
        "Adopting a different supported method under this rule is not a retry "
        "of the refused action." in prompt
    )
    assert (
        "Disclose repeated denials and the method change in the final report." in prompt
    )
    # The edited-paths rule carries the cross-format disclosure requirement.
    assert (
        "If the task or a skill specifies a different report format, include the "
        "denial record and the complete touched-path set in that format as well"
        in prompt
    )
    assert "the disclosure requirement applies to every output form." in prompt
    # The efficiency rule (former rule 7) is deleted entirely, batching clause
    # included; no rule-number references point at it.
    assert "Efficiency:" not in prompt
    assert "batch related shell operations" not in prompt
    assert "Read each file at most once per session" not in prompt


def test_bounded_child_output_format_discloses_denials_and_touched_paths() -> None:
    prompt = _fragment("bounded_child")

    # The output-format resolution — task-specified format, skill-specified
    # format, JSON fallback — must disclose the denial record and the
    # complete cumulative touched-path set in the final report.
    assert (
        "Return the format the task specifies when it names one; otherwise the "
        "loaded skill's specified format; otherwise the JSON fallback below." in prompt
    )
    assert (
        "Every report form must disclose two things: the denial record (each "
        "denied command and the runtime's stated reason, or an empty list when "
        "there were none) and the complete cumulative touched-path set (every "
        "repository path you created, edited, renamed, or deleted, including "
        "edits later reverted)." in prompt
    )
    # The JSON fallback gains an explicit denials field, and files_touched is
    # documented as the cumulative record, not a clean-diff projection.
    assert (
        '"files_touched": ["the cumulative record of every repository path '
        "created, edited, renamed, or deleted, including edits later reverted; "
        'empty array if none"]' in prompt
    )
    assert (
        '"denials": ["each denied command with the runtime\'s stated reason, '
        'or an empty array"]' in prompt
    )


def test_advisor_identity_carries_the_read_only_denials_variant() -> None:
    prompt = _identity_body("advisor")

    # Rewritten denials rule (read-only variant): report-in-result recording
    # with the pinned clauses and hardened refusal language, no stop rule.
    assert (
        "Denials: report denied commands and the runtime's rejection reason "
        "in your result; do not persist them to task-note files." in prompt
    )
    assert (
        "Never retry a permission/policy-refused action through changed syntax, "
        "another tool, a wrapper, or another model; report the blocked "
        "requirement and continue only independent authorized work." in prompt
    )
    # Same-effect retry clarification (ADR 0020 amendment), same sentence as
    # in the bounded-child fragment.
    assert (
        "The refused action's identity is its effect on the same files or "
        "state, not the command spelling: producing the same refused effect "
        "through a different tool, script, or mechanism is still a retry." in prompt
    )
    # Strategy switch adapted to record in the result.
    assert (
        "Strategy switch: after the second denial citing the same runtime "
        "rule/pattern across the assignment, record it in your result and adopt "
        "a different supported method for the affected operation" in prompt
    )
    assert (
        '"Same reason" means the rejection-reason category stated by the '
        "runtime's denial message, matched by its stated rule/pattern, not "
        "free-text similarity." in prompt
    )
    assert (
        "Adopting a different supported method is not a retry of the refused "
        "action." in prompt
    )
    assert "Disclose repeated denials and the method change in your result." in prompt
    # The report format carries a denials slot so recording has a destination.
    assert (
        "**Denials encountered:** {denied commands and the runtime's stated "
        "reasons, or none}" in prompt
    )
    # The read-only variant never asks for task-note persistence or path
    # bookkeeping: those stay with the shared bounded-child contract.
    assert "persist failed commands" not in prompt
    assert "maintain a cumulative record" not in prompt


def test_reviewer_denials_slot_follows_coverage_line() -> None:
    prompt = _identity_body("reviewer")

    assert prompt.index(
        "**Coverage:** {what was reviewed and excluded}"
    ) < prompt.index("**Denials encountered:**")


def test_advisor_denials_slot_covers_every_output_form() -> None:
    prompt = _identity_body("advisor")

    # Every permitted output form can carry the denials slot: the Advice
    # template, the concise paragraph, and the Blocker form.
    assert (
        "The **Denials encountered:** line is part of every permitted output "
        "form: include it, or its content, in the concise paragraph and in the "
        "Blocker form as well." in prompt
    )


def test_bounded_child_fragment_is_byte_identical_across_composed_child_prompts(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    fragment = _fragment("bounded_child").strip()

    def contract_block(identity: str) -> str:
        prompt = _compose_child_prompt(build_config, load_orchestrator, identity)
        # The shared bounded-child fragment lands verbatim in every child.
        assert fragment in prompt
        start = prompt.index("## Execution context — bounded child")
        end = prompt.index("\n## ", start)
        return prompt[start:end]

    blocks = [contract_block(identity) for identity in ("coder", "advisor", "reviewer")]
    # ADR 0020 left the shared contract untouched: it survives as the one
    # bounded-child fragment, byte-identical across the composed child prompts.
    assert blocks[0] == blocks[1] == blocks[2]
    assert "Perform one bounded assignment" in blocks[0]
    assert "Return the requested result format exactly." in blocks[0]


def test_user_prompt_file_stays_a_custom_prompt_without_shadowing_the_builtin_identity(
    mock_prompts_dirs: tuple[Path, Path],
) -> None:
    _, user_prompts = mock_prompts_dirs
    (user_prompts / "advisor.md").write_text("User advisor prompt", encoding="utf-8")

    # The user file loads through the surviving custom-prompt path...
    assert (
        load_prompt(
            "advisor", setting_name="system_prompt_id", builtins=_UTILITY_BUILTINS
        )
        == "User advisor prompt"
    )
    # ...but it never shadows the built-in identity of the same name: the
    # bare selection still resolves to the shipped definition, and the
    # user-qualified form is rejected as a reserved name.
    assert (
        validate_identity_selection("advisor", user_prompts_dirs=[user_prompts])
        == "advisor"
    )
    with pytest.raises(UnknownIdentityError, match="cannot shadow built-in identities"):
        validate_identity_selection("user:advisor", user_prompts_dirs=[user_prompts])


def test_bundled_file_alone_does_not_make_a_prompt_selectable(
    mock_prompts_dirs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled_prompts = tmp_path / "bundled_prompts"
    bundled_prompts.mkdir()
    (bundled_prompts / "synthetic_bundled_only.md").write_text(
        "Not a selectable prompt", encoding="utf-8"
    )
    monkeypatch.setattr("chartreux.core.prompts.PROMPTS_DIR", bundled_prompts)

    # The strict loader resolves built-ins from the explicit builtins map,
    # never by scanning the prompts directory: a bundled-only file is not
    # loadable through the surviving utility/custom path...
    with pytest.raises(ResourceResolutionError, match="cannot read resource"):
        load_prompt(
            "synthetic_bundled_only",
            setting_name="system_prompt_id",
            builtins=_UTILITY_BUILTINS,
        )
    # ...and it is not a selectable identity either.
    with pytest.raises(UnknownIdentityError, match="Unknown identity"):
        validate_identity_selection("synthetic_bundled_only", user_prompts_dirs=[])


@pytest.mark.parametrize("prompt_id", ["", ".", "..", "../cli", "dir/cli", "dir\\cli"])
def test_prompt_selection_rejects_non_bare_ids(
    mock_prompts_dirs: tuple[Path, Path], prompt_id: str
) -> None:
    with pytest.raises(ValueError, match="must be a bare filename"):
        load_prompt(
            prompt_id, setting_name="system_prompt_id", builtins=_UTILITY_BUILTINS
        )


def test_ceremony_workflow_opening_gate_stops_before_mutating_work() -> None:
    prompt = _workflow("ceremony")

    # The opening gate stops before mutating or authoritative work and names it.
    assert (
        "STOP and wait for the user before any mutating or authoritative work: "
        "edits and writes, verification runs, external side effects, and "
        "implementation dispatch." in prompt
    )
    # Read-only investigation is exempt before and after the opening response.
    assert (
        "Read-only investigation (such as `read_file`, `grep`, or exploration "
        "delegation) is exempt at any time, before and after the opening "
        "response." in prompt
    )
    # Trivial, response-only, prior-authorized-scope, and in-message
    # pre-authorization exemptions.
    assert (
        "Exemptions: trivial tasks, response-only work, previously "
        "authorized scope, and an explicit directive in the current message "
        "to proceed without waiting" in prompt
    )
    assert "authorizes the work it names, in this message only" in prompt
    assert (
        "Prior authorization covers only its stated phase and scope, never "
        "new work." in prompt
    )
    # The workflow gates are overridable defaults.
    assert (
        "The workflow gates are overridable defaults: an explicit user "
        "directive overrides them for the scope it names." in prompt
    )
    # The gate is imperative and turn-ending.
    assert (
        "then STOP — end your turn and wait for the user's explicit reaction. "
        "Do not proceed to design, planning, or implementation in the same turn."
        in prompt
    )
    # The observed failure modes are explicitly forbidden before the user reacts;
    # the prohibition covers repo files (scratchpad writes stay sanctioned).
    assert (
        "do not edit or write any repo file, do not dispatch implementation to a "
        "subagent, and do not run verification" in prompt
    )
    # A clear request is not an exemption, and trivial is defined narrowly.
    assert "A clear, detailed, or urgent request is not an exemption" in prompt
    assert (
        "Trivial means one file with no runtime behavior change, no public or "
        "internal contract change, no config semantics, no dependency change, "
        "no persisted data effect, and no user-visible output change" in prompt
    )
    assert "Most code changes are non-trivial" in prompt
    assert "the exemption is deliberately narrow" in prompt
    assert "when unsure, treat the task as non-trivial" in prompt


def test_ceremony_workflow_acceptance_lattice_fails_closed() -> None:
    prompt = _workflow("ceremony")

    # Design acceptance permits planning, not implementation.
    assert "Design acceptance permits planning, not implementation." in prompt
    assert "Design approval permits planning only." in prompt
    assert "Implementation requires an accepted plan covering that work." in prompt
    # Discussion, questions, silence, and elapsed turns are not acceptance.
    assert (
        "Discussion, questions, silence, and elapsed turns do not imply "
        "acceptance; unknown acceptance fails closed." in prompt
    )
    # Acceptance is explicit and phase-scoped: the enumerated phrases accept the
    # current phase only.
    assert (
        '"Yes", "go ahead", "approved", "proceed", or an equivalent directive '
        "accepts the current phase only" in prompt
    )
    assert (
        "design acceptance authorizes design and planning work, never "
        "implementation" in prompt
    )
    assert (
        "implementation requires a separate, explicit plan acceptance naming "
        "the work" in prompt
    )
    # Questions, discussion, elaboration, and silence are not acceptance.
    assert (
        "Questions, discussion, elaboration, and silence are not acceptance" in prompt
    )
    # A combined design-and-plan presentation is design acceptance only.
    assert (
        "Acceptance of a combined design-and-plan presentation is design "
        "acceptance only; implementation still requires a separate, explicit "
        "plan acceptance." in prompt
    )


def test_rendered_orchestrated_multi_model_prompt_routes_by_task_not_difficulty(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    # The composed orchestrator prompt routes by purpose: the launch-target
    # table binds every shipped default target to the session model, and the
    # routing prose rejects difficulty-based escalation.
    config = build_config(
        agent={
            "identity": "orchestrator",
            "workflow": "ceremony",
            "orchestration_strategy": "builtin:default",
        },
        include_model_info=False,
        include_commit_signature=False,
    )
    config.attach_catalog_snapshot(
        CatalogSnapshot(multi_model_catalog(), "routing-test")
    )
    prompt = get_universal_system_prompt(
        config, SkillManager(lambda: config), AgentManager(load_orchestrator(config))
    )

    # The launch-target table binds the session model of the catalog.
    assert "### Launch targets" in prompt
    assert "| Purpose | Identity | Tier | Seat | Model |" in prompt
    assert "| `implementation` | `coder` | substantive | — | `beta-model` |" in prompt
    assert (
        "| `review.deep` | `reviewer` | substantive | analytical | `beta-model` |"
        in prompt
    )
    assert (
        "| `review.deep` | `coder` | substantive | execution | `beta-model` |" in prompt
    )
    # Unmet model-diversity requirements are reported, never faked.
    assert "`review.deep`: 3 distinct canonical models" in prompt
    # The routing prose rides in the orchestrator identity body.
    assert "Route by purpose: the launch-target table" in prompt
    assert (
        "uncertainty, file count, session length, or wanting a better answer "
        "are not escalation criteria" in prompt
    )
    # One configured model and ten produce identical routing: no roster
    # roles leak into the composed prompt.
    assert "`@scout` for search" not in prompt
    assert "`@worker` for all substantive implementation" not in prompt
    assert "$dispatch" not in prompt
    # ADR 0018-G.1's byte-equality baseline is superseded: the roster rename
    # rewrote the curated prose, so the regenerated active goldens are the
    # byte baseline now and the WP0 legacy bytes no longer appear.
    legacy = (
        Path(__file__).parent / "fixtures/dispatch/legacy-orchestrated/cli-routing.md"
    ).read_text()
    assert legacy.rstrip("\n") not in prompt


def test_communication_fragment_uses_ascii_tree_and_arrow_characters() -> None:
    prompt = _fragment("communication")

    # Tree and flow diagrams are ASCII-only; no Unicode box-drawing or arrows.
    for character in ("├", "└", "│", "┌", "┐", "┘", "─", "→"):
        assert character not in prompt
    assert "`|--`" in prompt
    assert "`-- " in prompt
    assert "A -> B -> C" in prompt


def test_composed_orchestrator_prompt_carries_the_unattended_resilience_rule(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    """WP5: the unattended-resilience rule is findable from the composed
    orchestrator prompt — the orchestrator identity body carries the rule with
    its greppable marker — and the asking paragraph it complements (never
    contradicts) coexists in the shared tool-semantics fragment: infrastructure/
    harness failures never ask, genuine authorization decisions still do.
    """
    prompt = _compose_orchestrator_prompt(build_config, load_orchestrator)

    # The greppable marker and the rule's core sentences.
    assert UNATTENDED_RESILIENCE_MARKER in prompt
    assert "(UNATTENDED-RESILIENCE)" in prompt
    assert (
        "never produce a user question mid-run: never call `ask_user_question`"
        in prompt
    )
    assert "Park the failed package and continue independent work" in prompt
    assert "one consolidated blocker report carrying the resumable state" in prompt
    assert "Never leave an open interactive question pending mid-run" in prompt
    # The rule never overrides genuine user questions.
    assert (
        "Authorization, scope, safety, and destructive-action decisions remain "
        "genuine user questions; this rule never overrides them." in prompt
    )
    # The deferred-records surfacing is named in the consolidated report.
    assert "distinguishing deferred preferences from actual blockers" in prompt
    assert "still-open authorization decisions" in prompt
    # The asking paragraph coexists with the rule (it now rides in the shared
    # tool-semantics fragment, step 7, after the identity behavior, step 3).
    assert "**Asking the user.**" in prompt
    assert "`requires_user` questions (the default) always wait for an answer" in prompt
    # The module-level rule sentence matches the prompt's behavior.
    assert "never produce ask_user_question" in UNATTENDED_RESILIENCE_RULE


def test_orchestrator_identity_carries_the_failure_time_replacement_paragraph() -> None:
    """WP5: the orchestrator identity body names the failure-time replacement
    policy: choice at failure time, endpoint independence, mechanical-tier
    rejection for substantive packages, and park-and-continue. The claim shape
    (config lineage, reconciliation barrier, budget) is pinned on the task
    tool prompt by test_task_prompt_documents_replacement_claim, and the
    same-endpoint / luna-class rejections are pinned behaviorally in
    tests/recovery/test_replacement_policy.py.
    """
    prompt = _identity_body("orchestrator")

    assert "**Failure-time replacement.**" in prompt
    assert (
        "choose its replacement at failure time from the validated eligible set"
        in prompt
    )
    assert (
        "no dispatch-time substitution table and no automatic cross-model fallback"
        in prompt
    )
    assert "independent of the attempted failed endpoint" in prompt
    assert "never route a substantive package to a mechanical-tier model" in prompt
    assert "A HEAD/branch/workspace preflight stop stays a stop" in prompt
    assert "park it and continue independent work" in prompt


def test_task_prompt_documents_replacement_claim() -> None:
    """WP5: the task tool prompt documents the cross-model replacement claim
    shape (NEW agent_id + config lineage) next to the replace_run paragraph.
    """
    task = _TASK_PROMPT.read_text()

    assert "For a cross-model replacement after a failure, omit `agent_id`" in task
    assert "`config.predecessor_agent_id` (the failed agent)" in task
    assert "`config.work_package_id` (the package identity)" in task
    assert "`config.package_tier` (`mechanical` or `substantive`)" in task
    assert "Launch admission validates the claim at failure time" in task
    assert "endpoint independence from the ATTEMPTED failed endpoint" in task
    assert "the reconciliation barrier" in task
    assert "milestone notice with the honest cost implication" in task
    # The claim paragraph sits between the replace_run and cancel paragraphs.
    replace_at = task.index("To supersede busy work")
    claim_at = task.index("For a cross-model replacement after a failure")
    cancel_at = task.index("Use `cancel_agent(agent_id, run_id)`")
    assert replace_at < claim_at < cancel_at


def test_composed_orchestrator_and_task_prompts_retire_implementation_escalation(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    prompt = _compose_orchestrator_prompt(build_config, load_orchestrator)
    task = _TASK_PROMPT.read_text()

    for surface in (prompt, task):
        assert "escalate to `@large` only for" not in surface
        assert "escalate worker capability" not in surface
        assert "needs more capability" not in surface
        assert "stronger `model`" not in surface
        assert "To escalate an idle agent" not in surface
        assert "for corrective work, including" not in surface


def test_coder_identity_dispatches_skill_first_with_format_precedence() -> None:
    body = _identity_body("coder")
    fragment = _fragment("bounded_child")

    # Skill-first dispatch: a named skill loads first, otherwise the coder
    # selects the applicable task skill itself.
    assert (
        "If the task names a skill to load, load it first and follow its "
        "methodology and output format." in body
    )
    assert "Otherwise select the applicable existing task skill yourself" in body
    assert (
        "If the task requires a skill that is unavailable, report that as a "
        "blocker instead of inventing a role." in body
    )
    # A named but inapplicable skill is surfaced as a mismatch, not followed.
    assert (
        "If the named skill is inapplicable to the task, say so instead of "
        "following it." in body
    )
    # Output precedence: task-specified format, then skill format, then JSON.
    assert (
        "Return the format the task specifies when it names one; otherwise the "
        "loaded skill's specified format; otherwise the JSON fallback below."
        in fragment
    )
    # The unconditional "return only valid JSON" wording is retired.
    assert "return only valid JSON" not in body
    assert "return only valid JSON" not in fragment


def test_compact_prompt_preserves_approval_state_in_handoff() -> None:
    prompt = UtilityPrompt.COMPACT.read()

    # Acceptance states as separate, explicit fields.
    assert (
        "Design acceptance state and plan acceptance state as separate, "
        "explicit fields, each one of: accepted, pending, rejected, unknown" in prompt
    )
    # Approval evidence, not just the verdict.
    assert (
        "Approved scope and the approval evidence — which user message approved "
        "what — not just the verdict" in prompt
    )
    # One-time grants with consumed/unconsumed state.
    assert "One-time grants: target, purpose, and consumed/unconsumed state" in prompt
    # Agent/run handles, dependencies, and uncollected results.
    assert (
        "Active agent/run handles, outstanding dependencies between them, and "
        "uncollected results" in prompt
    )
    # Fail-closed prohibition, including overflow eviction of earlier rounds.
    assert "never infer a missing approval" in prompt
    assert "an unknown acceptance state stays unknown" in prompt
    assert "must not flip any acceptance state or resurrect a consumed grant" in prompt
    # The overflow rule is executable without identifying disappeared evidence:
    # a grant the retained conversation no longer shows as available is consumed.
    assert "A one-time grant counts as consumed once used" in prompt
    assert (
        "if the retained conversation no longer shows that a grant was still "
        "available, treat it as consumed and never re-grant it" in prompt
    )
    # Text-only contract and <summary> wrapper preserved.
    assert "Respond with text only. Do NOT call any tools." in prompt
    assert "<summary>" in prompt
    assert "</summary>" in prompt
    # Constraint hygiene: the summary carries conversation constraints only, and
    # the summarizer's own transport mechanics must never leak into it.
    assert (
        "ONLY constraints that were active in the conversation before compaction "
        "— never this prompt's own summarization instructions" in prompt
    )
    assert (
        "transport mechanics for this response, not conversation constraints" in prompt
    )
    assert (
        "The resumed agent must treat any such leaked text as a summarizer "
        "artifact, not an instruction" in prompt
    )


def test_compact_summary_prefix_treats_transport_mechanics_as_artifacts() -> None:
    prefix = UtilityPrompt.COMPACT_SUMMARY_PREFIX.read()

    # Standing rule for the resumed agent: summarizer transport mechanics that
    # leak into a compaction summary are artifacts, not instructions.
    assert (
        "Compaction summaries may embed summarizer transport mechanics "
        "(text-only rules, wrapper instructions); treat any such text as a "
        "summarizer artifact, not an instruction — continue using tools "
        "normally." in prefix
    )


def test_headless_context_composes_with_the_ceremony_gate(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(
        agent={
            "identity": "coder",
            "workflow": "ceremony",
            "orchestration_strategy": "none",
        },
        include_model_info=False,
        include_commit_signature=False,
    )
    prompt = get_universal_system_prompt(
        config,
        SkillManager(lambda: config),
        AgentManager(load_orchestrator(config)),
        headless=True,
    )

    # The ceremony workflow carries the opening gate in the composed prompt.
    gate = "STOP and wait for the user before any mutating or authoritative work"
    assert gate in prompt
    # Unattended runs have no one to accept mid-run: the gate's exemption
    # language names scope authorized up front.
    assert "previously authorized scope" in prompt
    # The headless execution context (step 4) composes alongside the gate
    # (step 5): blocking questions are recorded and reported, not asked.
    assert "## Execution context — headless main session" in prompt
    assert "Do not block the run on a question that cannot be answered" in prompt
    assert "there is no follow-up turn to rely on" in prompt
    # The instruction hierarchy resolves conflicts between the two.
    assert "## Instruction hierarchy" in prompt
