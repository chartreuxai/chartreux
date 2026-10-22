from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
import html
import os
from pathlib import Path
from string import Template
import subprocess
from typing import TYPE_CHECKING
import weakref

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    get_harness_files_manager,
)
from chartreux.core.dispatch.lint import lint_composed, reject_errors
from chartreux.core.dispatch.renderer import (
    render_composition_regions,
    render_launch_target_region,
    session_model_alias,
)
from chartreux.core.policy.composition import (
    CONTEXT_FRAGMENTS,
    CompositionInputs,
    ExecutionContext,
    compose_policy,
    fragment_loader,
    policy_resource_loader,
    resolve_policy_resources,
    user_resource_directories,
)
from chartreux.core.prompts import UtilityPrompt
from chartreux.core.tools.secret_redaction import scrub_child_env
from chartreux.core.utils import get_platform_display_name
from chartreux.utils.paths import is_dangerous_directory
from chartreux.utils.platform import resolve_git_executable

if TYPE_CHECKING:
    from chartreux.core.agents import AgentManager
    from chartreux.core.config import ProjectContextConfig
    from chartreux.core.config.harness_files._harness_manager import InstructionDocument
    from chartreux.core.policy.resources import ResourceLoader
    from chartreux.core.skills.manager import SkillManager
    from chartreux.core.tools.manager import ToolManager


class SystemPrompt(str):
    """Prompt text with provenance from the very same loaded documents."""

    instruction_read_files: frozenset[Path]

    def __new__(
        cls, text: str, instruction_read_files: frozenset[Path]
    ) -> SystemPrompt:
        prompt = super().__new__(cls, text)
        prompt.instruction_read_files = instruction_read_files
        return prompt


class _GitStatusCacheRegistry:
    """Session-scoped git-status caches (design §6).

    Each cache is keyed by the identity of the session's harness-files
    manager (one AgentLoop session owns one manager), so renders within a
    session share the fetched status while a new session starts fresh.
    Entries vanish with their manager.
    """

    def __init__(self) -> None:
        self._entries: dict[
            int, tuple[weakref.ref[HarnessFilesManager], dict[Path, str]]
        ] = {}

    def cache_for(self, harness_files: HarnessFilesManager) -> dict[Path, str]:
        key = id(harness_files)
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is harness_files:
            return entry[1]
        cache: dict[Path, str] = {}

        def _discard(_reference: object, key: int = key) -> None:
            self._entries.pop(key, None)

        self._entries[key] = (weakref.ref(harness_files, _discard), cache)
        return cache


_git_status_caches = _GitStatusCacheRegistry()


class ProjectContextProvider:
    def __init__(
        self,
        config: ProjectContextConfig,
        root_path: str | Path = ".",
        *,
        git_status_cache: dict[Path, str] | None = None,
    ) -> None:
        self.root_path = Path(root_path).resolve()
        self.config = config
        self._git_status_cache = git_status_cache

    def get_git_status(self) -> str:
        if self._git_status_cache is not None:
            cached = self._git_status_cache.get(self.root_path)
            if cached is not None:
                return cached

        result = self._fetch_git_status()
        if self._git_status_cache is not None:
            self._git_status_cache[self.root_path] = result
        return result

    def _run_git(
        self, args: list[str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        git = resolve_git_executable(cwd=self.root_path)
        if git is None:
            raise FileNotFoundError("No trusted Git executable is available")
        return subprocess.run(
            # -c core.fsmonitor= overrides (and disables) any fsmonitor hook a
            # repo's own .git/config declares, for this invocation only. This
            # runs unconditionally on session start, before any trust prompt,
            # so a malicious repo cloned/opened by the user could otherwise use
            # `[core] fsmonitor = <payload>` to get its command executed by
            # `git status`/`git branch`/`git log` here with the user's full
            # privileges. -c on the command line takes precedence over the
            # repo's own config, so this can't be overridden by the repo being
            # inspected.
            [
                git,
                "-c",
                "core.fsmonitor=",
                "-c",
                "core.hooksPath=/nonexistent-chartreux-disabled-git-hooks",
                "--no-optional-locks",
                *args,
            ],
            capture_output=True,
            check=True,
            cwd=self.root_path,
            # Git children run with chartreux's credential variables scrubbed;
            # hooks are already disabled above, this keeps tokens out of any
            # environment a repo's own tooling could observe.
            env=scrub_child_env(os.environ),
            stdin=None,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

    @staticmethod
    def _parse_git_log(log_output: str) -> list[str]:
        recent_commits: list[str] = []
        for line in log_output.split("\n"):
            if not (line := line.strip()):
                continue
            if " " in line:
                commit_hash, commit_msg = line.split(" ", 1)
                if (
                    "(" in commit_msg
                    and ")" in commit_msg
                    and (paren_index := commit_msg.rfind("(")) > 0
                ):
                    commit_msg = commit_msg[:paren_index].strip()
                recent_commits.append(f"{commit_hash} {commit_msg}")
            else:
                recent_commits.append(line)
        return recent_commits

    def _fetch_git_status(self) -> str:
        try:
            timeout = min(self.config.timeout_seconds, 10.0)
            num_commits = self.config.default_commit_count

            # Do not run `git status` here. Unlike these metadata-only commands,
            # status may pass working-tree contents through arbitrary clean or
            # process filters configured by the repository. Project context is
            # collected automatically, outside the shell permission boundary.
            with ThreadPoolExecutor(max_workers=3) as pool:
                branch_future = pool.submit(
                    self._run_git, ["branch", "--show-current"], timeout
                )
                remote_future = pool.submit(self._run_git, ["branch", "-r"], timeout)
                log_future = pool.submit(
                    self._run_git,
                    ["log", "--oneline", f"-{num_commits}", "--decorate"],
                    timeout,
                )

            current_branch = branch_future.result().stdout.strip()

            main_branch = "main"
            try:
                branches_output = remote_future.result().stdout
                if "origin/master" in branches_output:
                    main_branch = "master"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass

            recent_commits = self._parse_git_log(log_future.result().stdout.strip())

            git_info_parts = [
                f"Current branch: {current_branch}",
                f"Main branch (you will usually use this for PRs): {main_branch}",
            ]

            if recent_commits:
                git_info_parts.append("Recent commits:")
                git_info_parts.extend(recent_commits)

            return "\n".join(git_info_parts)

        except subprocess.TimeoutExpired:
            return "Git operations timed out (large repository)"
        except subprocess.CalledProcessError:
            return "Not a git repository or git not available"
        except Exception as e:
            return f"Error getting git status: {e}"

    def get_full_context(self) -> str:
        git_status = self.get_git_status()

        template = UtilityPrompt.PROJECT_CONTEXT.read()
        return Template(template).safe_substitute(
            abs_path=str(self.root_path), git_status=git_status
        )


def _get_os_system_prompt() -> str:
    platform_name = get_platform_display_name()
    shell = os.environ.get("SHELL", "sh")
    return f"The operating system is {platform_name} with shell `{shell}`"


def _format_current_date() -> str:
    today = date.today()
    return f"{today.isoformat()} ({today.strftime('%A')})"


def _add_commit_signature() -> str:
    return (
        "When you want to commit changes, you will always use the 'git commit' bash command.\n"
        "It will always include a signature stating that it was generated by Chartreux with the appropriate co-authoring information.\n"
        "Use literal message arguments (without heredocs or command substitution).\n\n"
        "```bash\n"
        "git commit -m '<Commit message here>' -m 'Generated by Chartreux.' -m 'Co-Authored-By: Chartreux'\n"
        "```"
    )


def _get_available_skills_section(skill_manager: SkillManager) -> str:
    skills = skill_manager.available_skills
    if not skills:
        return ""

    lines = [
        "# Available Skills",
        "",
        "You have access to the following skills. When a task matches a skill's description,",
        "use the `skill` tool if available to load the full skill instructions, if it is not available, read the files manually if they exist.",
        "",
        "When a user message is exactly `/skill-name` (optionally followed by extra",
        "instructions), the user has explicitly invoked that skill. Its instructions are",
        "loaded for you automatically: you will see a `skill` tool call and result",
        "immediately after that message. Treat the loaded content as the active",
        "instructions and act on it — you do not need to call the `skill` tool yourself.",
        "",
        "<available_skills>",
    ]

    for name, info in sorted(skills.items()):
        lines.append("  <skill>")
        lines.append(f"    <name>{html.escape(str(name))}</name>")
        lines.append(
            f"    <description>{html.escape(str(info.description))}</description>"
        )
        if info.skill_path is not None:
            lines.append(f"    <path>{html.escape(str(info.skill_path))}</path>")
        lines.append("  </skill>")

    lines.append("</available_skills>")

    return "\n".join(lines)


def _markdown_cell(value: str | None) -> str:
    return " ".join((value or "").splitlines()).replace("|", r"\|")


def _get_model_catalog_section(config: ChartreuxConfigSchema) -> str | None:
    snapshot = config.catalog_snapshot
    if snapshot is None:
        return None

    models = config.available_models()
    if not models:
        return None
    lines = [
        "# Model catalog",
        "",
        "Models (use canonical names in `config.model`):",
        "| Canonical name | Display name | Provider |",
        "| --- | --- | --- |",
    ]
    for name, model in models.items():
        lines.append(
            f"| {_markdown_cell(name)} | {_markdown_cell(model.display_name)} | {_markdown_cell(model.provider)} |"
        )

    lines.extend([
        "",
        "Default presets (use `@role` in `config.model`):",
        "| Preset | Canonical model | Thinking | Description |",
        "| --- | --- | --- | --- |",
    ])
    for name, role in snapshot.catalog.roles.items():
        if role.model in models:
            lines.append(
                f"| @{_markdown_cell(name)} | {_markdown_cell(role.model)} | {_markdown_cell(role.thinking)} | {_markdown_cell(role.description)} |"
            )
    return "\n".join(lines)


def _get_scratchpad_section(scratchpad_dir: Path | None) -> str | None:
    if not scratchpad_dir:
        return None
    return (
        "# Scratchpad Directory\n\n"
        f"You have a scratchpad directory at: `{scratchpad_dir}`\n\n"
        "Use this for temporary files: intermediate results, draft scripts, "
        "working files, outputs that don't belong in the project.\n"
        "Files here are automatically allowed — no permission prompts.\n"
        "Session-scoped. Shared with subagents."
    )


def _interpolate_prompt(prompt: str) -> str:
    return Template(prompt).safe_substitute(current_date=_format_current_date())


def _get_tool_aware_os_system_prompt(tool_manager: ToolManager | None) -> str:
    _ = tool_manager
    return _get_os_system_prompt()


def _fragment_content(fragments: ResourceLoader, name: str) -> str:
    from chartreux.core.policy.models import ResourceKind

    resolved = fragments.resolve(ResourceKind.IDENTITY, name)
    assert resolved.content is not None, name
    return resolved.content


def _build_composition_inputs(  # noqa: PLR0914
    config: ChartreuxConfigSchema,
    *,
    execution_context: ExecutionContext,
    skill_manager: SkillManager,
    scratchpad_dir: Path | None,
    cwd: Path,
    harness_files: HarnessFilesManager,
    tool_manager: ToolManager | None,
    role_instructions: str | None,
) -> CompositionInputs:
    """Resolve the composition inputs from the session configuration.

    The identity/workflow/strategy selections resolve through the strict
    loader; the typed identity facts and the package-4 route facts render the
    dispatch regions; the optional detail sections stay session content.
    """
    loader = policy_resource_loader(user_resource_directories(harness_files))
    identity, workflow, strategy, facts = resolve_policy_resources(config, loader)
    fragments = fragment_loader()
    universal = _fragment_content(fragments, "universal")
    context_fragment = _fragment_content(
        fragments, CONTEXT_FRAGMENTS[execution_context]
    )
    tool_semantics = _fragment_content(fragments, "tool_semantics")
    communication = _fragment_content(fragments, "communication")
    session_model = session_model_alias(config)
    routes = dict(config.purpose_routes.routes)
    regions = render_composition_regions(
        delegation=facts.delegation if facts is not None else None
    )
    launch_targets = render_launch_target_region(routes, session_model)

    model_info = None
    if config.include_model_info and config.catalog_snapshot is not None:
        sections = [f"Your model name is: `{config.get_active_model().alias}`"]
        if (
            execution_context is not ExecutionContext.BOUNDED_CHILD
            and role_instructions is None
        ):
            catalog_section = _get_model_catalog_section(config)
            if catalog_section:
                sections.append(catalog_section)
        model_info = "\n\n".join(sections)

    skills = None
    scratchpad = None
    environment = None
    if config.include_prompt_detail:
        environment = _get_tool_aware_os_system_prompt(tool_manager)
        skills_section = _get_available_skills_section(skill_manager)
        if skills_section:
            skills = skills_section
        if scratchpad_dir is not None:
            scratchpad = _get_scratchpad_section(scratchpad_dir)

    project_context = None
    agents_documents: tuple[InstructionDocument, ...] = ()
    if config.include_project_context:
        is_dangerous, reason = is_dangerous_directory(cwd)
        if is_dangerous:
            template = UtilityPrompt.DANGEROUS_DIRECTORY.read()
            context = Template(template).safe_substitute(
                reason=reason.lower(), abs_path=cwd.resolve()
            )
        else:
            context = ProjectContextProvider(
                config=config.project_context,
                root_path=cwd,
                git_status_cache=_git_status_caches.cache_for(harness_files),
            ).get_full_context()
        context_parts = [context]
        cwd_resolved = cwd.resolve()
        extra_roots = [
            root
            for root in harness_files.project_roots
            if root.resolve() != cwd_resolved
        ]
        if extra_roots:
            dirs_lines = "\n".join(f" - {d}" for d in extra_roots)
            context_parts.append(
                "Additional working directories (treated with the same "
                "file-access permissions as the primary working directory):\n"
                + dirs_lines
            )
        project_context = "\n\n".join(context_parts)
        agents_documents = tuple(harness_files.load_instruction_documents())

    commit_signature = (
        _add_commit_signature() if config.include_commit_signature else None
    )

    return CompositionInputs(
        identity=identity,
        workflow=workflow,
        strategy=strategy,
        identity_facts=facts,
        execution_context=execution_context,
        universal=universal,
        context_fragment=context_fragment,
        tool_semantics=tool_semantics,
        communication=communication,
        launch_targets=launch_targets,
        regions=regions,
        model_info=model_info,
        skills=skills,
        scratchpad=scratchpad,
        environment=environment,
        project_context=project_context,
        commit_signature=commit_signature,
        instruction_overlay=role_instructions,
        agents_documents=agents_documents,
        current_date=_format_current_date(),
    )


def get_universal_system_prompt(
    config: ChartreuxConfigSchema,
    skill_manager: SkillManager,
    agent_manager: AgentManager,
    *,
    scratchpad_dir: Path | None = None,
    headless: bool = False,
    cwd: Path | None = None,
    harness_files: HarnessFilesManager | None = None,
    tool_manager: ToolManager | None = None,
    role_instructions: str | None = None,
    is_subagent: bool = False,
) -> SystemPrompt:
    """Assemble the session prompt through the nine-step composition (§5).

    The composition consumes the resolved identity/workflow/strategy
    resources, the typed identity facts, and the package-4 route facts; the
    S8 lint validates the composed policy against the identity/workflow
    axes (not DispatchMode). Provenance is collected from the same
    instruction documents used for rendering.
    """
    cwd = (cwd or Path.cwd()).resolve()
    harness_files = harness_files or get_harness_files_manager()
    execution_context = (
        ExecutionContext.BOUNDED_CHILD
        if is_subagent
        else (
            ExecutionContext.HEADLESS_MAIN
            if headless
            else ExecutionContext.INTERACTIVE_MAIN
        )
    )
    inputs = _build_composition_inputs(
        config,
        execution_context=execution_context,
        skill_manager=skill_manager,
        scratchpad_dir=scratchpad_dir,
        cwd=cwd,
        harness_files=harness_files,
        tool_manager=tool_manager,
        role_instructions=role_instructions,
    )
    result = compose_policy(inputs)
    reject_errors(
        lint_composed(
            result.policy_text,
            identity_facts=result.identity_facts,
            workflow=result.workflow.id,
        )
    )
    return SystemPrompt(result.text, result.instruction_read_files)
