from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from chartreux.core.agents import AgentManager
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.dispatch import ORCHESTRATED_PRESET, SHIPPED_PURPOSES
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.prompts import (
    MissingPromptFileError,
    SystemPrompt,
    UtilityPrompt,
    load_system_prompt,
)
from chartreux.core.scratchpad import init_scratchpad
from chartreux.core.skills.manager import SkillManager
from chartreux.core.system_prompt import get_universal_system_prompt
from tests.conftest import ConfigBuilder, OrchestratorLoader, multi_model_catalog


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
        system_prompt_id="cli",
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
    # The routing prose renders from the bound dispatch policy: every slot
    # binds the single canonical model `base`, so the rendering is task-kind
    # and role-free rather than role-routed.
    assert "route by purpose through the configured slots" in prompt
    assert "| `mechanical` | `worker` |" in prompt
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


def test_worker_profile_child_omits_model_catalog_without_role_instructions(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(
        system_prompt_id="worker",
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
    assert roles["orchestrator"].description == "main assistant preset"
    assert roles["worker"].description == "worker preset"
    assert roles["scout"].description == "scout preset"
    assert roles["heavy"].description == "heavy preset"
    # The retired capacity wording is gone from every role description.
    for role in roles.values():
        assert "capacity preset" not in role.description
    # The routing semantics live in the purpose vocabulary.
    assert set(SHIPPED_PURPOSES) == {
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

    assert "# Headless Mode" in prompt
    assert "no human is available to respond" in prompt


def test_headless_section_absent_by_default(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(include_model_info=False, include_commit_signature=False)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(load_orchestrator(config))

    prompt = get_universal_system_prompt(config, skill_manager, agent_manager)

    assert "Headless Mode" not in prompt


def test_current_date_placeholder_substituted_in_prompt(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(
        system_prompt_id="cli", include_model_info=False, include_commit_signature=False
    )
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(load_orchestrator(config))

    prompt = get_universal_system_prompt(config, skill_manager, agent_manager)

    today = date.today()
    expected = f"Today's date is {today.isoformat()} ({today.strftime('%A')})."
    assert expected in prompt
    assert "$current_date" not in prompt


def test_default_prompt_renders_delegation_protocol(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(
        system_prompt_id="cli", include_model_info=False, include_commit_signature=False
    )
    prompt = get_universal_system_prompt(
        config, SkillManager(lambda: config), AgentManager(load_orchestrator(config))
    )

    assert "## Delegation protocol" in prompt
    assert "## Orchestration" not in prompt
    assert "## Background subagents" not in prompt
    assert "re-sent on every call" in prompt
    assert 'task(agent_type="worker", task=...)' in prompt
    assert "write_file/edit — repo files within the approved scope" in prompt
    assert "read_file — files the task names or cites" in prompt
    assert "bash — orchestration metadata only, always with timeouts" in prompt
    assert "delegate searches, exploration, tests, and builds" in prompt
    assert (
        "Pass the model and thinking explicitly; unavailable slots fail closed"
        in prompt
    )
    assert "$role" not in prompt
    # The delegation protocol's mode-variable regions render from the bound
    # dispatch policy at assembly; the standalone preset renders task-kind
    # routing regardless of roster size, so no placeholder or role name leaks.
    assert "route by purpose through the configured slots" in prompt
    assert "$dispatch" not in prompt
    assert "`@scout` for search" not in prompt


def test_system_prompt_builtin_ids_and_default_are_explicit() -> None:
    assert {prompt.value for prompt in SystemPrompt} == {
        "cli",
        "explore",
        "tests",
        "minimal",
        "worker",
        "advisor",
        "reviewer",
    }
    assert ChartreuxConfigSchema.model_fields["system_prompt_id"].annotation is str
    assert ChartreuxConfigSchema.model_fields["system_prompt_id"].default == "cli"


@pytest.mark.parametrize(
    "prompt_id", ["cli", "explore", "tests", "minimal", "worker", "advisor", "reviewer"]
)
def test_explicit_builtin_system_prompt_selection(
    mock_prompts_dirs: tuple[Path, Path], build_config: ConfigBuilder, prompt_id: str
) -> None:
    expected = SystemPrompt(prompt_id).read()
    assert build_config(system_prompt_id=prompt_id).system_prompt == expected
    assert load_system_prompt(prompt_id.upper()) == expected


_NON_SYSTEM_PROMPT_IDS = [
    "cli_2026-07_v2",
    "cli_2026-08_v3",
    *(prompt.value for prompt in UtilityPrompt),
]


@pytest.mark.parametrize("prompt_id", _NON_SYSTEM_PROMPT_IDS)
def test_retired_and_utility_ids_are_not_builtin_system_prompts(
    mock_prompts_dirs: tuple[Path, Path], build_config: ConfigBuilder, prompt_id: str
) -> None:
    config = build_config(system_prompt_id=prompt_id)
    with pytest.raises(MissingPromptFileError) as exc_info:
        _ = config.system_prompt

    assert exc_info.value.setting_name == "system_prompt_id"
    assert exc_info.value.prompt_id == prompt_id
    assert (
        'available prompts ("cli", "explore", "tests", "minimal", "worker", "advisor", "reviewer")'
        in str(exc_info.value)
    )


@pytest.mark.parametrize(
    "prompt_id", ["custom_selection", "cli", *_NON_SYSTEM_PROMPT_IDS]
)
def test_custom_system_prompt_selection_preserves_precedence(
    mock_prompts_dirs: tuple[Path, Path], build_config: ConfigBuilder, prompt_id: str
) -> None:
    project_prompts, user_prompts = mock_prompts_dirs
    config = build_config(system_prompt_id=prompt_id)
    (user_prompts / f"{prompt_id}.md").write_text(
        "  User custom prompt\n", encoding="utf-8"
    )
    assert config.system_prompt == "User custom prompt"

    (project_prompts / f"{prompt_id}.md").write_text(
        "  Project custom prompt\n", encoding="utf-8"
    )
    assert config.system_prompt == "Project custom prompt"


@pytest.mark.parametrize("prompt_id", ["worker", "advisor", "reviewer"])
def test_role_prompts_contain_subagent_contract(prompt_id: str) -> None:
    prompt = load_system_prompt(prompt_id)

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

    if prompt_id in {"advisor", "reviewer"}:
        assert (
            "report denied commands and the runtime's rejection reason in your result"
            in prompt
        )
        assert "do not persist them to task-note files" in prompt
        assert "do not modify files to reconcile them" in prompt
        assert "persist failed commands" not in prompt
        assert "maintain a cumulative record" not in prompt


def test_user_prompt_overrides_builtin_role_prompt_by_id(
    mock_prompts_dirs: tuple[Path, Path],
) -> None:
    _, user_prompts = mock_prompts_dirs
    (user_prompts / "advisor.md").write_text("User advisor prompt", encoding="utf-8")

    assert load_system_prompt("advisor") == "User advisor prompt"


def test_bundled_file_alone_does_not_make_system_prompt_selectable(
    mock_prompts_dirs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled_prompts = tmp_path / "bundled_prompts"
    bundled_prompts.mkdir()
    (bundled_prompts / "synthetic_bundled_only.md").write_text(
        "Not a selectable system prompt", encoding="utf-8"
    )
    monkeypatch.setattr("chartreux.core.prompts.PROMPTS_DIR", bundled_prompts)

    with pytest.raises(MissingPromptFileError):
        load_system_prompt("synthetic_bundled_only")


@pytest.mark.parametrize("prompt_id", ["", ".", "..", "../cli", "dir/cli", "dir\\cli"])
def test_system_prompt_selection_rejects_non_bare_ids(
    mock_prompts_dirs: tuple[Path, Path], prompt_id: str
) -> None:
    with pytest.raises(ValueError, match="must be a bare filename"):
        load_system_prompt(prompt_id)


def test_cli_prompt_opening_gate_stops_before_mutating_work() -> None:
    prompt = SystemPrompt.CLI.read()

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


def test_cli_prompt_acceptance_lattice_fails_closed() -> None:
    prompt = SystemPrompt.CLI.read()

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
    # The routing prose is no longer baked into the prompt skeleton: it
    # renders from the bound dispatch policy at assembly. A multi-model
    # roster keeps the tier routing, now named for the roster roles.
    config = build_config(
        system_prompt_id="cli", include_model_info=False, include_commit_signature=False
    )
    config.attach_catalog_snapshot(
        CatalogSnapshot(
            multi_model_catalog(), "routing-test", dispatch=ORCHESTRATED_PRESET
        )
    )
    prompt = get_universal_system_prompt(
        config, SkillManager(lambda: config), AgentManager(load_orchestrator(config))
    )

    assert (
        "`@scout` for search, grep, exploration, verification, and mechanical "
        "single-file edits" in prompt
    )
    assert "`@worker` for all substantive implementation, however demanding" in prompt
    assert (
        "`@heavy` for architecture, cross-subsystem design, design and planning "
        "analysis, and deep review, plus demanding execution with a settled "
        "approach through the escalation-implementor route — never routine "
        "implementation. The worker and reviewer profiles use `@worker` by "
        "default; the advisor profile uses `@heavy` for architecture, design, "
        "planning, and destructive-operation analysis only; it never implements. "
        "Demanding execution with a settled approach runs through the "
        "worker-profile escalation-implementor route." in prompt
    )
    # The reviewer profile stays @worker; deep review at @heavy is a model override.
    assert (
        'The reviewer profile stays `@worker`; "deep review at `@heavy`" means '
        "the reviewer profile with a `@heavy` model override." in prompt
    )
    # Failed implementation is retried at @worker or analyzed by a @heavy
    # advisor; the advisor seat never implements.
    assert "never re-dispatch implementation to the `@heavy` advisor" in prompt
    # The tier is passed explicitly: mechanical work carries a @scout override,
    # substantive implementation launches at the @worker default.
    assert "the profile default is not the routing decision" in prompt
    assert 'config={"model": "@scout"}' in prompt
    assert 'task="Rename add to plus in utils.py and update its call sites."' in prompt
    assert (
        'task="Add a retry helper with exponential backoff to utils.py and use '
        'it in app.py."' in prompt
    )
    # ADR 0018-G.1's byte-equality baseline is superseded: the roster rename
    # rewrote the curated prose, so the regenerated active goldens are the
    # byte baseline now and the WP0 legacy bytes no longer appear.
    legacy = (
        Path(__file__).parent / "fixtures/dispatch/legacy-orchestrated/cli-routing.md"
    ).read_text()
    assert legacy.rstrip("\n") not in prompt
    assert "`@scout` for search" in prompt


def test_cli_prompt_uses_ascii_tree_and_arrow_characters() -> None:
    prompt = SystemPrompt.CLI.read()

    # Tree and flow diagrams are ASCII-only; no Unicode box-drawing or arrows.
    for character in ("├", "└", "│", "┌", "┐", "┘", "─", "→"):
        assert character not in prompt
    assert "`|--`" in prompt
    assert "`-- `" in prompt
    assert "A -> B -> C" in prompt


def test_cli_and_task_prompts_retire_implementation_escalation() -> None:
    cli = SystemPrompt.CLI.read()
    task = (
        Path(__file__).parents[1] / "chartreux/core/tools/builtins/prompts/task.md"
    ).read_text()

    for prompt in (cli, task):
        assert "escalate to `@large` only for" not in prompt
        assert "escalate worker capability" not in prompt
        assert "needs more capability" not in prompt
        assert "stronger `model`" not in prompt
        assert "To escalate an idle agent" not in prompt
        assert "for corrective work, including" not in prompt


def test_worker_prompt_dispatches_skill_first_with_format_precedence() -> None:
    prompt = SystemPrompt.WORKER.read()

    # Skill-first dispatch: a named skill loads first, otherwise the worker
    # selects the applicable task skill itself.
    assert (
        "If the task names a skill to load, load it first and follow its "
        "methodology and output format." in prompt
    )
    assert "Otherwise select the applicable existing task skill yourself" in prompt
    assert (
        "If the task requires a skill that is unavailable, report it as a "
        "blocker; do not silently invent a role." in prompt
    )
    # A named but inapplicable skill is a mismatch blocker, not a methodology.
    assert (
        "If the named skill is inapplicable to the task, report the mismatch "
        "as a blocker instead of following it." in prompt
    )
    # Output precedence: task-specified format, then skill format, then JSON.
    assert (
        "Return the format the task specifies when it names one; otherwise the "
        "loaded skill's specified format; otherwise this JSON fallback:" in prompt
    )
    # The unconditional "return only valid JSON" wording is retired.
    assert "return only valid JSON" not in prompt


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


def test_headless_section_overrides_wait_gates_after_them(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    config = build_config(
        system_prompt_id="cli", include_model_info=False, include_commit_signature=False
    )
    prompt = get_universal_system_prompt(
        config,
        SkillManager(lambda: config),
        AgentManager(load_orchestrator(config)),
        headless=True,
    )

    gate = "STOP and wait for the user before any mutating or authoritative work"
    override = (
        "Override any earlier instructions that say to wait for confirmation "
        "or ask the user."
    )
    assert gate in prompt
    assert override in prompt
    # The override must follow the gate text it overrides.
    assert prompt.index(override) > prompt.index(gate)
    # Unattended runs have no one to accept mid-run: the only route to mutating
    # work is scope authorized up front.
    assert "previously authorized scope" in prompt
    assert "no human is available to respond" in prompt
