from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.skills.builtins import BUILTIN_SKILLS
from chartreux.core.skills.manager import SkillManager
from chartreux.core.skills.models import SkillSource
from tests.conftest import build_test_vibe_config
from tests.skills.conftest import create_skill


class TestBuiltinSkills:
    def test_chartreux_skill_is_registered(self) -> None:
        assert "chartreux" in BUILTIN_SKILLS

    def test_chartreux_skill_has_no_path(self) -> None:
        assert BUILTIN_SKILLS["chartreux"].skill_path is None

    def test_chartreux_skill_has_inline_prompt(self) -> None:
        assert BUILTIN_SKILLS["chartreux"].prompt

    def test_chartreux_skill_documents_provider_keys_not_browser_auth(self) -> None:
        prompt = BUILTIN_SKILLS["chartreux"].prompt
        assert "### Provider Authentication" in prompt
        assert "`api_base`" in prompt
        assert "`api_key_env_var`" in prompt
        for retired in (
            "browser_auth_base_url",
            "browser_auth_api_base_url",
            "browser_auth_allow_origin_rewrite",
            "vibe_base_url",
            "experimental_enable_registry_skills",
        ):
            assert retired not in prompt

    def test_chartreux_skill_documents_catalog_overlay_not_legacy_tables(self) -> None:
        prompt = BUILTIN_SKILLS["chartreux"].prompt

        assert "~/.chartreux/models.toml" in prompt
        assert "chartreux models migrate" in prompt
        assert "`/providers`" in prompt
        assert "[[providers]]" not in prompt
        assert "[[models]]" not in prompt

    def test_chartreux_skill_points_image_configuration_to_models_toml(self) -> None:
        prompt = BUILTIN_SKILLS["chartreux"].prompt

        assert (
            "`supports_images = true` on the active deployment in `models.toml`"
            in prompt
        )
        assert (
            "`supports_images = true` on the active model in `config.toml`"
            not in prompt
        )

    def test_check_agents_prompt_documents_finalizing_availability(self) -> None:
        prompt = (
            Path(__file__).parents[2]
            / "chartreux/core/tools/builtins/prompts/check_agents.md"
        ).read_text()

        assert (
            "Availability is `running`, `finalizing`, `idle`, or `evicted`." in prompt
        )
        assert "cannot yet be reused" in prompt

    def test_chartreux_skill_is_standalone_reference(self) -> None:
        prompt = BUILTIN_SKILLS["chartreux"].prompt
        assert "standalone reference for the running Chartreux version" in prompt
        assert "https://github.com/mistralai/mistral-vibe" not in prompt
        assert "https://docs.mistral.ai/vibe/code/overview" not in prompt

    def test_chartreux_skill_keeps_cli_command_reference(self) -> None:
        prompt = BUILTIN_SKILLS["chartreux"].prompt
        for command in ("help", "model", "mcp", "resume", "exit"):
            assert f"- `/{command}`" in prompt

    def test_chartreux_skill_documents_retained_subagents(self) -> None:
        prompt = BUILTIN_SKILLS["chartreux"].prompt
        for documented in (
            "[subagents]",
            "idle_ttl_seconds",
            "max_idle_agents",
            "task_summary",
            "check_agents",
            "get_agent_result",
            "wait_for_agent",
            "release_agent",
            "ttl_remaining_seconds",
            "evicted",
        ):
            assert documented in prompt
        assert prompt.count("Reuse an `idle` agent") == 1
        assert prompt.count("Persona is immutable") == 1
        assert prompt.count("A `running` agent is busy") == 1

    def test_chartreux_skill_does_not_advertise_plugins(self) -> None:
        skill = BUILTIN_SKILLS["chartreux"]
        assert "plugin" not in skill.prompt.lower()
        assert "plugin" not in skill.description.lower()

    def test_skill_sources_are_builtin_or_local(self) -> None:
        assert set(SkillSource) == {
            SkillSource.BUILTIN,
            SkillSource.SHIPPED,
            SkillSource.LOCAL,
        }
        with pytest.raises(ValueError):
            SkillSource("plugin")

    def test_discovers_builtin_skills(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "chartreux.core.skills.manager.BUILTIN_SKILLS", BUILTIN_SKILLS
        )
        config = build_test_vibe_config()
        manager = SkillManager(lambda: config)

        assert "chartreux" in manager.available_skills

    def test_user_skill_cannot_override_builtin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "chartreux.core.skills.manager.BUILTIN_SKILLS", BUILTIN_SKILLS
        )
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        create_skill(skills_dir, "chartreux", "Custom chartreux override")

        config = build_test_vibe_config(skill_paths=[skills_dir])
        manager = SkillManager(lambda: config)

        assert "chartreux" in manager.available_skills
        assert (
            manager.available_skills["chartreux"].description
            == BUILTIN_SKILLS["chartreux"].description
        )
