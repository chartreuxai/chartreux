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
            "Availability is `running`, `finalizing`, `idle`, `suspended`, "
            "`restoring`, `evicted`, or `closed`." in prompt
        )
        assert "cannot yet be reused" in prompt
        # Closed (completed without retention) and evicted (retained, then
        # removed) are distinct terminal states; both keep results retrievable.
        assert "A `closed` agent completed without retention" in prompt
        assert "An `evicted` agent was retained but its runtime was removed" in prompt
        assert (
            "remain available through `get_agent_result` until result expiry" in prompt
        )

    def test_bash_prompt_documents_dangerous_command_approval_flow(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/bash.md"
        ).read_text()

        assert "`dangerous_command` denial" in prompt
        assert (
            "the runtime will prompt the user when the session is interactive" in prompt
        )
        assert (
            "A refusal means the user declined: do not retry that exact command"
            in prompt
        )
        assert (
            "Conversational approval does not grant dangerous-command authority"
            in prompt
        )
        assert "hard guards" in prompt
        assert "git reset --hard" in prompt
        assert "--config-env" in prompt
        assert "absolute veto" in prompt
        assert "permission authority, not standing intent" in prompt

    def test_git_workflow_skill_recommends_switch_and_runtime_approval_push(
        self,
    ) -> None:
        skill = (
            Path(__file__).parents[2] / "chartreux/skills/main-git-workflow/SKILL.md"
        ).read_text()

        assert "`git switch -c <branch-name>`" in skill
        assert "`git switch <branch>`" in skill
        assert "git checkout -b" not in skill
        assert "switching branches is fine" not in skill
        assert "dangerous-command approval flow" in skill
        assert "An explicit `[tools.bash]` denylist entry is an absolute veto" in skill
        assert "`git stash apply" in skill

    def test_design_and_plan_skills_gate_on_explicit_acceptance(self) -> None:
        root = Path(__file__).parents[2]
        design = (root / "chartreux/skills/main-design/SKILL.md").read_text()
        plan = (root / "chartreux/skills/main-plan/SKILL.md").read_text()

        assert (
            "Obtain the user's explicit design acceptance before proceeding; "
            "continued discussion, questions, or silence are not acceptance."
        ) in design
        assert (
            "Obtain the user's explicit plan acceptance before proceeding to "
            "implementation; continued discussion, questions, or silence are not "
            "acceptance."
        ) in plan
        for skill in (design, plan):
            assert "wait for the user to react" not in skill.lower()

    def test_chartreux_skill_is_standalone_reference(self) -> None:
        prompt = BUILTIN_SKILLS["chartreux"].prompt
        assert "standalone reference for the running Chartreux version" in prompt
        assert "https://github.com/mistralai/mistral-vibe" not in prompt
        assert "https://docs.mistral.ai/vibe/code/overview" not in prompt

        assert "@main" in prompt
        assert "@orchestrator" in prompt
        assert "Bound dispatch role removed" not in prompt

    def test_chartreux_skill_keeps_cli_command_reference(self) -> None:
        prompt = BUILTIN_SKILLS["chartreux"].prompt
        for command in ("help", "model", "mcp", "resume", "exit"):
            assert f"- `/{command}`" in prompt

    def test_chartreux_skill_documents_retained_subagents(self) -> None:
        prompt = BUILTIN_SKILLS["chartreux"].prompt
        for documented in (
            "[subagents]",
            "retain_agents",
            "idle_ttl_seconds",
            "max_idle_agents",
            "tombstone_ttl_seconds",
            "task_summary",
            "check_agents",
            "get_agent_result",
            "wait_for_agent",
            "release_agent",
            "ttl_remaining_seconds",
            "evicted",
            "closed",
            "still_running",
            "1500",
            "display-only",
        ):
            assert documented in prompt
        assert prompt.count("Reuse an `idle` agent") == 1
        assert prompt.count("Persona is immutable") == 1
        assert prompt.count("A `running` agent is busy") == 1
        # Retention is opt-in with launch-argument precedence, and reuse
        # inherits the instance's committed policy.
        assert "close when their run completes unless they are retained" in prompt
        assert (
            "precedence is the launch argument, then the global `[subagents].retain_agents` setting"
            in prompt
        )
        assert (
            "omit `retain` to inherit the instance's committed retention policy"
            in prompt
        )
        assert "no identity-specific retention setting" in prompt

    def test_chartreux_skill_documents_dangerous_command_approval(self) -> None:
        prompt = BUILTIN_SKILLS["chartreux"].prompt
        for documented in (
            "dangerous git commands",
            "Deny default",
            "never always",
            "absolute veto",
            "not standing intent to publish",
            "Settings > Shell Permissions",
            "no public grant endpoint",
        ):
            assert documented in prompt, documented

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
