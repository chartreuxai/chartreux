from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from chartreux.core.agents import AgentManager
from chartreux.core.config import ChartreuxConfigSchema
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
from tests.conftest import ConfigBuilder, OrchestratorLoader


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


def test_model_catalog_section_uses_live_eligible_models_and_roles(
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
                "description": "Do bounded work",
            },
            "unavailable": {
                "model": "disabled",
                "thinking": "high",
                "description": "Unavailable",
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
    assert "| @worker | base | high | Do bounded work |" in prompt
    assert "| disabled |" not in prompt
    assert "@unavailable" not in prompt

    restricted = config.model_copy(update={"allowed_models": ["base"]})
    restricted_prompt = get_universal_system_prompt(
        restricted,
        SkillManager(lambda: restricted),
        AgentManager(load_orchestrator(restricted)),
    )
    assert "| backup |" not in restricted_prompt
    assert "| @worker | base | high | Do bounded work |" in restricted_prompt


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
            }
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
    assert "every subsequent API call" in prompt
    for tool in (
        "task",
        "read_file",
        "write_file",
        "edit",
        "bash",
        "skill",
        "todo",
        "web_search",
        "web_fetch",
    ):
        assert f"`{tool}`" in prompt
    assert "scratchpad only, never repo edits" in prompt
    assert "read-only orchestration metadata only" in prompt
    assert "selects that role's one model and thinking level" in prompt
    assert "launch separate tasks" in prompt
    assert "$role" not in prompt


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
