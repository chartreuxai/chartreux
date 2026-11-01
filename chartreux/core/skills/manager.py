from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import difflib
from enum import StrEnum, auto
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
        self._load_failures: list[SkillLoadFailure] = []
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
    def load_failures(self) -> tuple[SkillLoadFailure, ...]:
        """Typed records of the skill files discovery could not load.

        The single source of load-failure information: the shadow inventory
        carries these same records and :attr:`config_issues` projects them,
        so there is no second failure list to drift.
        """
        return tuple(self._load_failures)

    @property
    def config_issues(self) -> tuple[SkillConfigIssue, ...]:
        """The TUI-facing projection of :attr:`load_failures`."""
        return tuple(
            SkillConfigIssue(
                file=failure.path, message=f"Failed to load: {failure.diagnostic}"
            )
            for failure in self.load_failures
        )

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
        except Exception as exc:
            # The raw exception stays in the log; the record keeps only the
            # bounded, content-free diagnostic a surface may display.
            logger.warning("Failed to parse skill at %s: %s", skill_file, exc)
            self._load_failures.append(
                SkillLoadFailure(
                    path=skill_file,
                    scope=scope,
                    source=source,
                    diagnostic=sanitized_failure_diagnostic(exc),
                )
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


_MAX_DIAGNOSTIC_ERRORS = 3
_MAX_DIAGNOSTIC_CHARS = 200


@dataclass(frozen=True, slots=True)
class SkillLoadFailure:
    """One skill file discovery could not load.

    The record carries the file, where discovery found it, and a bounded
    content-free diagnostic. It never carries a parsed name: a file that
    fails to load may have no valid name to report, so surfaces must render
    it from the path alone.
    """

    path: Path
    scope: SkillScope
    source: SkillSource
    diagnostic: str

    @property
    def origin(self) -> str:
        """The short origin label a surface renders for the failure."""
        if self.source is SkillSource.SHIPPED:
            return "shipped"
        return self.scope.value


def sanitized_failure_diagnostic(exc: BaseException) -> str:
    """A bounded, content-free diagnostic for a failed skill load.

    The exception text is never reused: YAML errors quote the offending
    source and pydantic validation errors include the input value, so either
    would surface raw frontmatter values. Only declared metadata field names
    and error kinds survive.
    """
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        details: list[str] = []
        for error in exc.errors(include_url=False, include_input=False)[
            :_MAX_DIAGNOSTIC_ERRORS
        ]:
            field = _metadata_field(error.get("loc", ()))
            kind = str(error.get("type", "invalid"))
            details.append(f"{field}: {kind}" if field else kind)
        summary = "; ".join(details)
        return _bounded_diagnostic(
            f"invalid skill metadata: {summary}"
            if summary
            else "invalid skill metadata"
        )
    if isinstance(exc, SkillParseError):
        reason = exc.reason
    elif isinstance(exc, OSError):
        reason = "cannot read the file"
    else:
        return f"unexpected {type(exc).__name__}"
    # The parser's detailed reasons append source-derived text (YAML errors
    # quote the failing snippet); keep only the static leading statement.
    return _bounded_diagnostic(reason.split(":", 1)[0])


def _metadata_field(loc: Sequence[object]) -> str:
    """The declared metadata field an error location names, if any.

    Arbitrary frontmatter keys are never echoed; only names from the skill
    metadata schema survive sanitization.
    """
    for part in loc:
        name = str(part)
        if name in _KNOWN_METADATA_FIELDS:
            return name
    return ""


def _bounded_diagnostic(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _MAX_DIAGNOSTIC_CHARS:
        return collapsed
    return collapsed[: _MAX_DIAGNOSTIC_CHARS - 3].rstrip() + "..."


def _known_metadata_fields() -> frozenset[str]:
    names = set(SkillMetadata.model_fields)
    for field in SkillMetadata.model_fields.values():
        alias = field.validation_alias
        if isinstance(alias, str):
            names.add(alias)
    return frozenset(names)


_KNOWN_METADATA_FIELDS = _known_metadata_fields()


class ShadowState(StrEnum):
    """How a user skill copy relates to the copy the loader selects."""

    IDENTICAL = auto()
    DIVERGED = auto()
    INACTIVE = auto()
    UNVERIFIED = auto()


@dataclass(frozen=True, slots=True)
class ShadowEntry:
    """One user skill copy found in a USER skill directory."""

    name: str
    skill_path: Path
    user_dir: Path
    state: ShadowState
    shipped_path: Path | None = None
    winner_path: Path | None = None
    winner_scope: str | None = None
    diff: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ShadowInventory:
    """The one read-only skill snapshot doctor and `skills diff` render."""

    entries: tuple[ShadowEntry, ...]
    user_dirs: tuple[Path, ...]
    load_failures: tuple[SkillLoadFailure, ...] = ()


def build_shadow_inventory(
    config_getter: Callable[[], ChartreuxConfigSchema],
    *,
    harness_files: HarnessFilesManager | None = None,
) -> ShadowInventory:
    """Build the single read-only skill snapshot both surfaces render.

    Only the two USER skill directories (``~/.chartreux/skills`` and
    ``~/.agents/skills``) are scanned for user copies. The loader's own
    discovery decides the winning copy, so the inventory reports selection
    without changing it: a user copy that wins is compared byte-for-byte with
    its shipped copy, and a copy the loader does not select is reported
    inactive with the winning copy named. The load-failure records come from
    that same discovery pass — one snapshot, no second scan.
    """
    manager = SkillManager(config_getter, harness_files=harness_files)
    user_dirs = tuple(manager._harness_files.user_skills_dirs)
    selected = manager._discovered_skills
    entries = [
        entry
        for user_dir in user_dirs
        for skill_path in _user_skill_files(user_dir)
        if (entry := _shadow_entry(skill_path, user_dir, selected)) is not None
    ]
    entries.sort(key=lambda entry: (entry.name, str(entry.skill_path)))
    failures = sorted(manager.load_failures, key=lambda failure: str(failure.path))
    return ShadowInventory(
        entries=tuple(entries), user_dirs=user_dirs, load_failures=tuple(failures)
    )


def _user_skill_files(user_dir: Path) -> list[Path]:
    try:
        children = sorted(user_dir.iterdir())
    except OSError:
        logger.debug("Shadow inventory cannot list %s", user_dir)
        return []
    return [
        skill_dir / "SKILL.md"
        for skill_dir in children
        if skill_dir.is_dir() and (skill_dir / "SKILL.md").is_file()
    ]


def _shadow_entry(
    skill_path: Path, user_dir: Path, selected: Mapping[str, SkillInfo]
) -> ShadowEntry | None:
    """Classify one user copy; None when it is no shadow of anything.

    A copy that does not parse never loads, so it shadows nothing — the
    typed load-failure record is a different surface. A winning copy that has
    no shipped counterpart is a plain local skill, not a shadow either.
    """
    name = _user_skill_name(skill_path)
    if name is None:
        return None
    shipped_path = SHIPPED_SKILLS_DIR / name / "SKILL.md"
    shipped = shipped_path if shipped_path.is_file() else None
    winner = selected.get(name)
    if winner is None:
        return ShadowEntry(
            name=name,
            skill_path=skill_path,
            user_dir=user_dir,
            state=ShadowState.UNVERIFIED,
            shipped_path=shipped,
            reason="the loader does not select this copy",
        )
    if winner.skill_dir != skill_path.parent.resolve():
        return ShadowEntry(
            name=name,
            skill_path=skill_path,
            user_dir=user_dir,
            state=ShadowState.INACTIVE,
            shipped_path=shipped,
            winner_path=winner.skill_path,
            winner_scope=_scope_label(winner),
        )
    if shipped is None:
        return None
    return _compared_entry(name, skill_path, user_dir, shipped)


def _user_skill_name(skill_path: Path) -> str | None:
    try:
        frontmatter, _body = parse_skill_markdown(read_safe(skill_path).text)
        return SkillMetadata.model_validate(frontmatter).name
    except Exception:
        logger.debug("Shadow inventory skipped unparsable skill at %s", skill_path)
        return None


def _scope_label(info: SkillInfo) -> str:
    if info.source is SkillSource.BUILTIN:
        return "builtin"
    if info.scope is SkillScope.BUILTIN:
        return "shipped"
    return str(info.scope)


def _compared_entry(
    name: str, skill_path: Path, user_dir: Path, shipped_path: Path
) -> ShadowEntry:
    try:
        shipped_bytes = shipped_path.read_bytes()
        user_bytes = skill_path.read_bytes()
    except OSError as exc:
        return ShadowEntry(
            name=name,
            skill_path=skill_path,
            user_dir=user_dir,
            state=ShadowState.UNVERIFIED,
            shipped_path=shipped_path,
            reason=f"cannot read the skill copy: {exc}",
        )
    if shipped_bytes == user_bytes:
        return ShadowEntry(
            name=name,
            skill_path=skill_path,
            user_dir=user_dir,
            state=ShadowState.IDENTICAL,
            shipped_path=shipped_path,
        )
    try:
        user_text = user_bytes.decode("utf-8")
        shipped_text = shipped_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return ShadowEntry(
            name=name,
            skill_path=skill_path,
            user_dir=user_dir,
            state=ShadowState.UNVERIFIED,
            shipped_path=shipped_path,
            reason=f"skill text is not valid UTF-8 ({exc.reason}); cannot diff",
        )
    return ShadowEntry(
        name=name,
        skill_path=skill_path,
        user_dir=user_dir,
        state=ShadowState.DIVERGED,
        shipped_path=shipped_path,
        diff=_unified_diff(shipped_path, skill_path, shipped_text, user_text),
    )


def _unified_diff(
    shipped_path: Path, skill_path: Path, shipped_text: str, user_text: str
) -> str:
    """The shipped copy is the diff's old side; the user copy is the new side."""
    return "".join(
        difflib.unified_diff(
            shipped_text.splitlines(keepends=True),
            user_text.splitlines(keepends=True),
            fromfile=f"shipped:{shipped_path}",
            tofile=f"user:{skill_path}",
        )
    )
