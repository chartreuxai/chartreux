from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from chartreux import CHARTREUX_ROOT
from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    get_harness_files_manager,
)
from chartreux.core.skills.builtins import BUILTIN_SKILLS
from chartreux.core.skills.compatibility import (
    ConversationPolicyFacts,
    SkillEligibility,
    check_skill_eligibility,
    format_incompatibility_diagnostic,
)
from chartreux.core.skills.models import (
    ParsedSkillCommand,
    SkillConfigIssue,
    SkillInfo,
    SkillMetadata,
    SkillScope,
    SkillSource,
)
from chartreux.core.skills.parser import SkillParseError, parse_skill_markdown
from chartreux.core.utils import name_matches
from chartreux.observability.logging import logger
from chartreux.utils.io import read_safe

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema


SHIPPED_SKILLS_DIR = CHARTREUX_ROOT / "skills"


class SkillManager:
    def __init__(
        self,
        config_getter: Callable[[], ChartreuxConfigSchema],
        *,
        harness_files: HarnessFilesManager | None = None,
        policy_facts: ConversationPolicyFacts | None = None,
    ) -> None:
        self._config_getter = config_getter
        self._harness_files = harness_files or get_harness_files_manager()
        # The conversation's frozen policy facts (design §3, §6): the bound
        # identity/workflow selections, snapshotted ONCE here. Eligibility
        # reads these frozen facts — a pending saved selection change (one
        # that applies to new conversations) never re-derives a bound
        # conversation's eligibility. Callers that already hold the frozen
        # facts (the agent loop binding the conversation's policy) pass them
        # explicitly; otherwise they are read from the bound config here.
        self._policy_facts = (
            policy_facts
            if policy_facts is not None
            else ConversationPolicyFacts(
                identity=self._config.agent.identity,
                workflow=self._config.agent.workflow,
            )
        )
        self._search_paths = self._compute_search_paths(self._config)
        self._config_issues: list[SkillConfigIssue] = []
        self._discovered_skills = self._discover_skills()
        self.available_skills: Mapping[str, SkillInfo] = MappingProxyType(
            self._apply_filters(self._discovered_skills)
        )

        if self.available_skills:
            logger.info(
                "Discovered %d skill(s) from %d search path(s)",
                len(self.available_skills),
                len(self._search_paths),
            )

    @property
    def _config(self) -> ChartreuxConfigSchema:
        return self._config_getter()

    @property
    def policy_facts(self) -> ConversationPolicyFacts:
        """The conversation's frozen policy facts the eligibility rule reads."""
        return self._policy_facts

    @property
    def settings_inventory(self) -> list[str]:
        """Discovered skill names, including those excluded by current filters.

        Compatibility filtering is NOT automatic catalog pruning: discovery
        still lists every skill, including shipped skills that are ineligible
        under the conversation's frozen workflow (marked via
        :meth:`skill_eligibility` / :attr:`workflow_incompatibilities`).
        """
        return sorted(self._discovered_skills)

    @property
    def config_issues(self) -> tuple[SkillConfigIssue, ...]:
        return tuple(self._config_issues)

    def skill_eligibility(self, name: str) -> SkillEligibility:
        """The marked eligibility of any discovered skill (pruned or not)."""
        info = self._discovered_skills.get(name)
        if info is None:
            return SkillEligibility(
                eligible=False, requirement="the skill is not discovered"
            )
        return check_skill_eligibility(
            name, source=info.source, facts=self._policy_facts
        )

    @property
    def workflow_incompatibilities(self) -> Mapping[str, str]:
        """Discovered skills withheld by the workflow rule, marked by requirement."""
        marks: dict[str, str] = {}
        for name, info in self._discovered_skills.items():
            eligibility = check_skill_eligibility(
                name, source=info.source, facts=self._policy_facts
            )
            if not eligibility.eligible:
                marks[name] = eligibility.requirement or ""
        return MappingProxyType(marks)

    def incompatibility_diagnostic(self, name: str) -> str | None:
        """The actionable diagnostic when an explicit invocation is withheld.

        Returns None when the skill is not discovered or is eligible; the
        text names the workflow/identity requirement and the remedy.
        """
        info = self._discovered_skills.get(name)
        if info is None:
            return None
        eligibility = check_skill_eligibility(
            name, source=info.source, facts=self._policy_facts
        )
        if eligibility.eligible:
            return None
        return format_incompatibility_diagnostic(
            name, source=info.source, facts=self._policy_facts
        )

    def _apply_filters(self, skills: dict[str, SkillInfo]) -> dict[str, SkillInfo]:
        # The workflow-compatibility rule (design §5): ceremony-bearing
        # shipped skills are exposed/loaded only in conversations whose
        # frozen workflow carries the ceremony. One rule, applied here, wires
        # the skills catalog, the `skill` tool, and slash invocation.
        workflow_eligible = {
            name: info
            for name, info in skills.items()
            if check_skill_eligibility(
                name, source=info.source, facts=self._policy_facts
            ).eligible
        }
        if self._config.enabled_skills:
            return {
                name: info
                for name, info in workflow_eligible.items()
                if name_matches(name, self._config.enabled_skills)
            }
        if self._config.disabled_skills:
            return {
                name: info
                for name, info in workflow_eligible.items()
                if not name_matches(name, self._config.disabled_skills)
            }
        return dict(workflow_eligible)

    def _compute_search_paths(
        self, config: ChartreuxConfigSchema
    ) -> list[tuple[Path, SkillScope, SkillSource]]:
        paths: list[tuple[Path, SkillScope, SkillSource]] = []

        for path in config.skill_paths:
            if path.is_dir():
                paths.append((path, SkillScope.GLOBAL, SkillSource.LOCAL))

        paths.append((SHIPPED_SKILLS_DIR, SkillScope.BUILTIN, SkillSource.SHIPPED))

        mgr = self._harness_files
        paths.extend(
            (p, SkillScope.PROJECT, SkillSource.LOCAL) for p in mgr.project_skills_dirs
        )
        paths.extend(
            (p, SkillScope.GLOBAL, SkillSource.LOCAL) for p in mgr.user_skills_dirs
        )

        global_paths = {p.resolve() for p in mgr.user_skills_dirs}
        global_paths.update(
            p.resolve() for p, scope, _ in paths if scope is SkillScope.GLOBAL
        )

        unique: list[tuple[Path, SkillScope, SkillSource]] = []
        seen: set[Path] = set()
        for p, scope, source in paths:
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            if scope is SkillScope.PROJECT and rp in global_paths:
                scope = SkillScope.GLOBAL
            unique.append((rp, scope, source))

        return unique

    def _discover_skills(self) -> dict[str, SkillInfo]:
        skills: dict[str, SkillInfo] = {**BUILTIN_SKILLS}
        for base, scope, source in self._search_paths:
            if not base.is_dir():
                continue
            for name, info in self._discover_skills_in_dir(base, scope, source).items():
                existing = skills.get(name)
                if existing is None or (
                    existing.source is SkillSource.SHIPPED
                    and info.source is SkillSource.LOCAL
                ):
                    skills[name] = info
                else:
                    logger.debug(
                        "Skipping duplicate skill '%s' at %s (already loaded from %s)",
                        name,
                        info.skill_path,
                        existing.skill_path,
                    )
        return skills

    def _discover_skills_in_dir(
        self, base: Path, scope: SkillScope, source: SkillSource
    ) -> dict[str, SkillInfo]:
        skills: dict[str, SkillInfo] = {}
        for skill_dir in base.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.is_file():
                continue
            skill_info = self._try_load_skill(skill_file, source=source, scope=scope)
            if skill_info is None:
                continue
            if skill_info.name in BUILTIN_SKILLS:
                logger.debug(
                    "Skipping skill '%s' at %s because builtin skill names are reserved",
                    skill_info.name,
                    skill_info.skill_path,
                )
                continue
            if skill_info.name in skills:
                logger.debug(
                    "Skipping duplicate skill '%s' at %s (already loaded from %s)",
                    skill_info.name,
                    skill_info.skill_path,
                    skills[skill_info.name].skill_path,
                )
                continue
            skills[skill_info.name] = skill_info
        return skills

    def _try_load_skill(
        self,
        skill_file: Path,
        *,
        source: SkillSource,
        scope: SkillScope,
        check_dir_name: bool = True,
    ) -> SkillInfo | None:
        try:
            skill_info = self._parse_skill_file(
                skill_file, source=source, scope=scope, check_dir_name=check_dir_name
            )
        except Exception as e:
            logger.warning("Failed to parse skill at %s: %s", skill_file, e)
            self._config_issues.append(
                SkillConfigIssue(file=skill_file, message=f"Failed to load: {e}")
            )
            return None
        return skill_info

    def _parse_skill_file(
        self,
        skill_path: Path,
        *,
        source: SkillSource,
        scope: SkillScope,
        check_dir_name: bool = True,
    ) -> SkillInfo:
        try:
            content = read_safe(skill_path).text
        except OSError as e:
            raise SkillParseError(f"Cannot read file: {e}") from e

        frontmatter, body = parse_skill_markdown(content)
        metadata = SkillMetadata.model_validate(frontmatter)

        if check_dir_name:
            skill_name_from_dir = skill_path.parent.name
            if metadata.name != skill_name_from_dir:
                logger.warning(
                    "Skill name '%s' doesn't match directory name '%s' at %s",
                    metadata.name,
                    skill_name_from_dir,
                    skill_path,
                )

        return SkillInfo.from_metadata(
            metadata, skill_path, prompt=body.strip(), source=source, scope=scope
        )

    @property
    def custom_skills_count(self) -> int:
        return sum(
            info.source is SkillSource.LOCAL for info in self.available_skills.values()
        )

    def get_skill(self, name: str) -> SkillInfo | None:
        return self.available_skills.get(name)

    def parse_skill_command(self, text_prompt: str) -> ParsedSkillCommand | None:
        stripped = text_prompt.strip()
        if not stripped.startswith("/"):
            return None

        parts = stripped[1:].split(None, 1)
        if not parts:
            return None

        skill_name = parts[0].lower()
        skill_info = self.get_skill(skill_name)
        if skill_info is None or not skill_info.user_invocable:
            return None

        extra_instructions = parts[1] if len(parts) > 1 else None

        return ParsedSkillCommand(
            name=skill_name,
            content=skill_info.prompt,
            extra_instructions=extra_instructions,
        )
