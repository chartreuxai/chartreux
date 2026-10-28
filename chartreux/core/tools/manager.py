from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sys
import threading
from typing import TYPE_CHECKING, Any, TypeGuard
import weakref

from chartreux.core.agents.identities import (
    AgentIdentity,
    ToolPosture,
    get_builtin_identity,
)
from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    get_harness_files_manager,
)
from chartreux.core.dangerous_commands import (
    OPERATION_REGISTRY,
    DangerousCommandAuthority,
    GitRepositoryRelocation,
)
from chartreux.core.dispatch.renderer import (
    contains_dispatch_placeholder,
    render_task_description_for_config,
)
from chartreux.core.llm_models import AvailableFunction
from chartreux.core.paths import DEFAULT_TOOL_DIR
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    ToolPermission,
    registry_provenance_classification,
)
from chartreux.core.tools.classification import (
    ToolClass,
    _stamp_location_trusted_user_tool,
    classification_revision,
    classify_tool,
    shipped_builtin_name,
)
from chartreux.core.tools.remote import MCPTool
from chartreux.core.tools.utils import PathAuthority
from chartreux.core.utils import name_matches, run_sync
from chartreux.core.workspace import Workspace
from chartreux.observability.logging import logger
from chartreux.utils.io import read_safe

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema
    from chartreux.core.config._restrictions import SourceRestrictions
    from chartreux.core.tools.builtins._shell_permission_resolver import (
        DangerousCommandAuthorityProvider,
    )
    from chartreux.core.tools.mcp.registry import MCPRegistry


def _try_canonical_module_name(path: Path) -> str | None:
    """Extract canonical module name for vibe package files.

    Prevents Pydantic class identity mismatches when the same module
    is imported via dynamic discovery and regular imports.
    """
    try:
        parts = path.resolve().parts
    except (OSError, ValueError):
        return None

    package_indices = [
        idx
        for idx, part in enumerate(parts)
        if part == "chartreux"
        and idx + 1 < len(parts)
        and (
            parts[idx + 1] in {"acp", "cli", "core", "setup"}
            or parts[idx + 1].endswith(".py")
        )
    ]
    if not package_indices:
        return None

    vibe_idx = package_indices[-1]
    if vibe_idx + 1 >= len(parts):
        return None

    module_parts = [p.removesuffix(".py") for p in parts[vibe_idx:]]
    return ".".join(module_parts)


def _compute_module_name(path: Path) -> str:
    """Return canonical module name for vibe files, hash-based synthetic name otherwise."""
    if canonical := _try_canonical_module_name(path):
        return canonical

    resolved = path.resolve()
    path_hash = hashlib.md5(str(resolved).encode()).hexdigest()[:8]
    stem = re.sub(r"[^0-9A-Za-z_]", "_", path.stem) or "mod"
    return f"chartreux_tools_discovered_{stem}_{path_hash}"


class NoSuchToolError(Exception):
    """Exception raised when a tool is not found."""


_POSTURE_CLASSES: dict[ToolPosture, frozenset[ToolClass]] = {
    ToolPosture.ALL_GRANTED: frozenset(ToolClass),
    # Control-plane tools (task, agent lifecycle, user questions) ride with a
    # read-only posture: they coordinate work without granting write authority,
    # so an orchestrator can delegate without gaining direct write access.
    ToolPosture.NON_MUTATING: frozenset({
        ToolClass.NON_MUTATING,
        ToolClass.CONTROL_PLANE,
    }),
}


def posture_allows_class(posture: ToolPosture, tool_class: ToolClass) -> bool:
    """Whether a posture selector admits a reviewed tool class."""
    return tool_class in _POSTURE_CLASSES[posture]


def posture_allows_name(posture: ToolPosture, tool_name: str) -> bool:
    """Admission-level posture check by reviewed classification.

    The central table is name-keyed, so an unknown name fails closed as
    MUTATING. The authoritative check is the manager's registry-provenance
    classification (``get_authority_classification``); this guard exists so
    launch admission rejects loosening attempts before construction.
    """
    return posture_allows_class(posture, classify_tool(tool_name, builtin=True))


def identity_allows_name(identity: AgentIdentity, tool_name: str) -> bool:
    """Admission-level personal check, honoring the execution carve-out.

    The launch-admission mirror of ``ToolManager._personal_posture_allows``:
    a carved-out name is admitted as the identity's own reviewed exception,
    everything else follows the posture's class set. The admission check is
    name-keyed by design; the runtime boundary stays exact-class — a custom
    variant of a carved-out name still fails closed at the shipped-builtin
    check inside the manager.
    """
    if tool_name in identity.execution_carve_out:
        return True
    return posture_allows_name(identity.personal_tools, tool_name)


class ToolManager:  # noqa: PLR0904 - reviewed authority lookup joins the full method budget
    """Manages tool discovery and instantiation for an Agent.

    Discovers available tools from the provided search paths. Each Agent
    should have its own ToolManager instance.
    """

    def __init__(  # noqa: PLR0913 - runtime-owned dependency injection
        self,
        config_getter: Callable[[], ChartreuxConfigSchema],
        mcp_registry: MCPRegistry | None = None,
        *,
        defer_mcp: bool = False,
        discovery_source: ToolManager | None = None,
        cwd: Path | None = None,
        harness_files: HarnessFilesManager | None = None,
        scratchpad_dir: Path | None = None,
        plan_file_write_scope_getter: Callable[[], Path | None] | None = None,
        restriction_getter: Callable[[], tuple[SourceRestrictions, ...]] | None = None,
        inherited_restrictions: tuple[SourceRestrictions, ...] = (),
        inherited_workspace: Workspace | None = None,
        inherited_plan_write_scopes: tuple[tuple[Path, Path | None], ...] = (),
        parent_authority_getter: Callable[[], ToolManager] | None = None,
        parent_authority_revision_getter: Callable[[], int] | None = None,
        accepted_token_getter: Callable[[], object] | None = None,
        task_description: str | None = None,
        session_root_grants: object | None = None,
        dangerous_command_authority_provider: DangerousCommandAuthorityProvider
        | None = None,
        bound_identity: str | None = None,
    ) -> None:
        self._config_getter = config_getter
        # The conversation's identity binding (design §3, decision log item 3):
        # when the owning loop supplies its construction-time binding, posture
        # derives from the bound identity — never from a later live config
        # change. A reload or in-place edit that changes [agent] identity is a
        # pending selection for new conversations, not a mid-conversation
        # re-posture of an existing manager or child.
        self._bound_identity = bound_identity
        self._cwd = (cwd or Path.cwd()).resolve()
        self._harness_files = harness_files or get_harness_files_manager()
        self._scratchpad_dir = scratchpad_dir
        self._plan_file_write_scope_getter = plan_file_write_scope_getter
        self._session_root_grants = session_root_grants
        # Runtime-owned dangerous-command grants (session/once lifetimes); the
        # runtime approval lifecycle implements this port.
        self._dangerous_command_authority_provider = (
            dangerous_command_authority_provider
        )
        # Only accepted runtime contributions belong here, never a preview getter.
        # Fixed parent authority is independent of later local profile changes.
        self._restriction_getter = restriction_getter
        self._inherited_workspace = inherited_workspace
        self._inherited_restrictions = tuple(inherited_restrictions)
        self._inherited_plan_write_scopes = tuple(inherited_plan_write_scopes)
        # Parent authority is resolved live so a child never retains a superseded
        # manager after the parent refreshes or tightens policy.
        self._parent_authority_getter = parent_authority_getter
        self._parent_authority_revision_getter = parent_authority_revision_getter
        # Managers without an accepted-revision signal retain live, uncached getters.
        self._accepted_token_getter = accepted_token_getter
        self._authority_generation = 0
        self._instruction_read_files: frozenset[Path] = frozenset()
        self._registry_generation = 0
        self._authority_cache: dict[
            str, tuple[tuple[object, ...], type[BaseToolConfig], str]
        ] = {}
        self._selection_cache: dict[
            str, tuple[tuple[object, ...], type[BaseTool] | None]
        ] = {}
        # Keyed by name for this manager's own delegation enumeration, or by
        # (name, id(implementation)) when the decision carries a descendant's
        # exact implementation through the ancestor chain; the entry's third
        # element verifies that implementation by identity (weak reference),
        # so a warm decision is never served for a different implementation
        # (S2 round 3: the ancestor chain re-resolves by name).
        self._delegation_selection_cache: dict[
            str | tuple[str, int],
            tuple[
                tuple[object, ...],
                type[BaseTool] | None,
                weakref.ref[type[BaseTool]] | None,
            ],
        ] = {}
        self._workspace_cache: tuple[tuple[object, ...], Workspace] | None = None
        self._mcp_registry = mcp_registry
        self._instances: dict[str, BaseTool] = {}
        # Keyed by (name, caller implementation), mirroring the delegation
        # selection cache's carried-implementation keying: the instance's
        # liveness closure classifies the EXACT implementation the caller
        # runs, so a warm instance built for one caller's class is never
        # served for another's (S2 final warning; package 3b amendment —
        # the flip corner: without the class key, an admitted second
        # caller's invocation reused the first caller's stale baked-class
        # instance and failed closed).
        self._delegated_instances: dict[
            tuple[str, type[BaseTool] | None], BaseTool
        ] = {}
        self._authority_retired = False
        self._search_paths: list[Path] = self._compute_search_paths(self._config)
        self._lock = threading.Lock()
        self._available_tool_specs_cache_lock = threading.Lock()
        self._available_tool_specs_cache_key: (
            frozenset[tuple[str, type[BaseTool], type[Any], str, str | None]] | None
        ) = None
        self._available_tool_specs_cache: dict[str, str] = {}
        self._mcp_integrated = False

        self._tool_variants_by_name: dict[str, list[type[BaseTool]]] = {}
        self._custom_tool_variants_by_name: dict[str, list[bool]] = {}
        # Historical one-class-per-name registry. When multiple classes publish the
        # same name, this is the fallback used if no variant is active.
        self._all_tools: dict[str, type[BaseTool]] = {}
        if discovery_source is not None:
            # Policy-only staging reuses accepted discovery, never live registries
            # or files. Copy containers, but not instances bound to old authority.
            with discovery_source._lock:
                self._tool_variants_by_name = {
                    name: list(variants)
                    for name, variants in discovery_source._tool_variants_by_name.items()
                }
                self._custom_tool_variants_by_name = {
                    name: list(origins)
                    for name, origins in discovery_source._custom_tool_variants_by_name.items()
                }
                self._all_tools = dict(discovery_source._all_tools)
                self._tool_descriptions = dict(discovery_source._tool_descriptions)
                self._mcp_integrated = discovery_source._mcp_integrated
        else:
            for (
                tool_class,
                is_custom,
                source_file,
            ) in self._iter_tool_classes_with_origin(self._search_paths):
                self._register_discovered_tool_variant(
                    tool_class, is_custom=is_custom, source_file=source_file
                )
            self._tool_descriptions: dict[str, str] = dict(
                self._iter_tool_descriptions(self._search_paths)
            )
        discovered_task = self._tool_descriptions.get("task")
        self._custom_task_description = (
            discovery_source._custom_task_description
            if discovery_source is not None
            else discovered_task is not None
            and not contains_dispatch_placeholder(discovered_task)
        )
        self._install_task_description(task_description)
        if not defer_mcp:
            self.integrate_all()

    def _install_task_description(self, task_description: str | None) -> None:
        """Install the policy-bound `task` tool description.

        The shipped ``prompts/task.md`` is a skeleton carrying the dispatch
        placeholders; the rendered description replaces it, bypassing the
        process-global ``Task.get_tool_prompt`` cache. A custom
        ``prompts/task.md`` without placeholders is user-owned and always wins.
        The per-instance spec cache already keys on the description, so a
        different policy re-keys it.
        """
        if self._custom_task_description:
            return
        if task_description is None:
            skeleton = self._tool_descriptions.get("task")
            if skeleton is None or not contains_dispatch_placeholder(skeleton):
                return
            task_description = render_task_description_for_config(
                self._config_getter(), skeleton
            )
        self._tool_descriptions["task"] = task_description

    def _bind_identity(self, identity: str) -> None:
        """Rebind the conversation's identity binding (design §3, decision 3).

        The owning loop calls this in the SAME transaction that rebinds its
        own identity (a clear or an in-place resume), so effective tool
        authority always matches the loop's current identity — a rebound
        conversation never keeps the previous identity's posture. The
        authority token carries the bound identity, so the rebind
        invalidates every token-keyed cache.
        """
        self._bound_identity = identity

    def set_mcp_registry(self, mcp_registry: MCPRegistry | None) -> None:
        self._mcp_registry = mcp_registry
        self._registry_generation += 1

    def _get_mcp_registry(self) -> MCPRegistry:
        if self._mcp_registry is None:
            from chartreux.core.tools.mcp.registry import MCPRegistry

            self._mcp_registry = MCPRegistry()
        return self._mcp_registry

    @property
    def _config(self) -> ChartreuxConfigSchema:
        return self._config_getter()

    def _compute_search_paths(self, config: ChartreuxConfigSchema) -> list[Path]:
        paths: list[Path] = [DEFAULT_TOOL_DIR.path]

        paths.extend(config.tool_paths)

        mgr = self._harness_files
        paths.extend(mgr.project_tools_dirs)
        paths.extend(mgr.user_tools_dirs)

        unique: list[Path] = []
        seen: set[Path] = set()
        for p in paths:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                unique.append(rp)
        return unique

    @staticmethod
    def _iter_tool_classes(search_paths: list[Path]) -> Iterator[type[BaseTool]]:
        """Iterate over all search_paths to find tool classes.

        A search path is either a directory of tool files (``<dir>/*.py``, e.g.
        ``.chartreux/tools/``) or a single ``.py`` file. Tool files sit directly in
        the directory — the same flat layout as the builtins and as the sibling
        ``prompts/*.md`` descriptions (see ``_iter_tool_descriptions``).
        """
        for tool_class, _, _ in ToolManager._iter_tool_classes_with_origin(
            search_paths
        ):
            yield tool_class

    @staticmethod
    def _iter_tool_classes_with_origin(
        search_paths: list[Path],
    ) -> Iterator[tuple[type[BaseTool], bool, Path]]:
        """Yield ``(tool_class, is_custom, source_file)`` per discovered class.

        ``source_file`` is the ``.py`` file the class was loaded from — the
        file-location provenance input for the user-tools-directory trust
        contract (P4-1): the trust decision keys on WHERE the tool was
        authored, so registration needs the file, not just the class.
        """
        builtin_dir = DEFAULT_TOOL_DIR.path.resolve()
        for base in search_paths:
            if not base.is_dir() and base.name.endswith(".py"):
                if tools := ToolManager._load_tools_from_file(base):
                    for tool in tools:
                        yield (
                            tool,
                            not base.resolve().is_relative_to(builtin_dir),
                            base,
                        )

            for path in base.glob("*.py"):
                if tools := ToolManager._load_tools_from_file(path):
                    for tool in tools:
                        yield (
                            tool,
                            not path.resolve().is_relative_to(builtin_dir),
                            path,
                        )

    @staticmethod
    def _load_tools_from_file(file_path: Path) -> list[type[BaseTool]] | None:
        if not file_path.is_file():
            return
        name = file_path.name
        if name.startswith("_"):
            return

        module_name = _compute_module_name(file_path)

        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                return

        # Builtin tool modules import shared base classes from
        # sibling files; drop those re-exports so they are not registered twice
        # under the importing module. Custom tool files may legitimately re-export
        # tool classes, so the filter is scoped to the builtin directory only.
        is_builtin = file_path.resolve().is_relative_to(DEFAULT_TOOL_DIR.path.resolve())

        tools = []
        for tool_obj in vars(module).values():
            if not inspect.isclass(tool_obj):
                continue
            if is_builtin and tool_obj.__module__ != module.__name__:
                continue
            if not issubclass(tool_obj, BaseTool) or tool_obj is BaseTool:
                continue
            if inspect.isabstract(tool_obj):
                continue
            tools.append(tool_obj)
        return tools

    @staticmethod
    def _iter_tool_descriptions(search_paths: list[Path]) -> Iterator[tuple[str, str]]:
        """Yield ``(tool_name, description)`` from ``prompts/<name>.md`` files in
        the tool search paths.

        Every tool directory pairs implementations with their descriptions the
        same way the builtins do — ``<tools-dir>/*.py`` alongside
        ``<tools-dir>/prompts/*.md`` (the very layout ``get_tool_prompt`` reads).
        So ``.chartreux/tools/prompts/weather.md`` describes a custom ``weather`` tool
        and ``.chartreux/tools/prompts/bash.md`` re-describes the builtin ``bash``.

        Keyed by file stem (the tool name); later search paths win, matching
        ``.py`` override precedence. A ``.py`` search-path entry is matched
        against the ``prompts/`` dir beside it.
        """
        for base in search_paths:
            if base.is_dir():
                prompts_dir = base / "prompts"
            elif base.name.endswith(".py"):
                prompts_dir = base.parent / "prompts"
            else:
                continue
            if not prompts_dir.is_dir():
                continue
            for md_path in sorted(prompts_dir.glob("*.md")):
                try:
                    text = read_safe(md_path).text
                except OSError:
                    continue
                # Yield the raw text (matching get_full_description), but skip
                # blank files so they fall back instead of blanking a tool.
                if text.strip():
                    yield md_path.stem, text

    @staticmethod
    def discover_tool_defaults(
        search_paths: list[Path] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if search_paths is None:
            search_paths = [DEFAULT_TOOL_DIR.path]

        defaults: dict[str, dict[str, Any]] = {}
        for cls in ToolManager._iter_tool_classes(search_paths):
            try:
                tool_name = cls.get_name()
                config_class = cls._get_tool_config_class()
                defaults[tool_name] = config_class().model_dump(exclude_none=True)
            except Exception as e:
                logger.warning(
                    "Failed to get defaults for tool %s: %s", cls.__name__, e
                )
                continue
        return defaults

    def _is_user_tools_file(self, source_file: Path) -> bool:
        """Whether a discovered tool file lives under the user's tools directory.

        The file-location trust contract (P4-1) keys on the user's OWN tools
        directory (``~/.chartreux/tools/`` — the user's home config, not
        project files): project-discovered tools and config ``tool_paths``
        entries stay fail-closed. The comparison resolves both sides, so a
        symlinked spelling of the same directory still matches and a
        different directory that merely shares a name never does.
        """
        try:
            resolved = source_file.resolve()
        except (OSError, ValueError):
            return False
        return any(
            resolved.is_relative_to(directory.resolve())
            for directory in self._harness_files.user_tools_dirs
        )

    def _register_discovered_tool_variant(
        self,
        tool_class: type[BaseTool],
        *,
        is_custom: bool,
        source_file: Path | None = None,
    ) -> None:
        name = tool_class.get_name()
        self._tool_variants_by_name.setdefault(name, []).append(tool_class)
        self._custom_tool_variants_by_name.setdefault(name, []).append(is_custom)
        self._all_tools[name] = tool_class
        if (
            source_file is not None
            and not self._is_remote_tool_class(tool_class)
            and self._is_user_tools_file(source_file)
        ):
            # File-location provenance (P4-1, charter decision 4): a tool
            # authored under the user's OWN tools directory is the user's own
            # code, so it is trusted by where it lives and admitted to the
            # non-mutating posture — a deliberate, separate contract recorded
            # in the harness-side location-trust side table, never as a class
            # attribute. MCP proxies keep their own explicit fail-closed
            # decision even when a user file wraps one, and a class
            # registered through this path WITHOUT a user-tools-directory
            # source file (direct registration, re-export, reuse staging)
            # carries no location trust and fails closed as before.
            _stamp_location_trusted_user_tool(tool_class)
        self._registry_generation += 1

    @property
    def registered_tools(self) -> dict[str, type[BaseTool]]:
        with self._lock:
            selected_tools: dict[str, type[BaseTool]] = {}
            for name, fallback_tool_class in self._all_tools.items():
                selected_tools[name] = self._select_registered_variant(
                    name, fallback_tool_class
                )
            return selected_tools

    def _parent_authority(self) -> ToolManager | None:
        if self._parent_authority_getter is None:
            return None
        try:
            parent = self._parent_authority_getter()
        except Exception:
            return None
        return parent if parent is not self else None

    def _identity_posture(
        self,
    ) -> tuple[ToolPosture | None, ToolPosture | None, frozenset[str]]:
        """(personal, delegation, carve-out) selectors of this manager's identity.

        Derived from the ``[agent]`` selection below the tool-config layer: a
        custom identity carries no harness-known posture (None = unrestricted
        on that axis), with the delegation set defaulting to the full granted
        ceiling (design decision 8). When the owning loop bound the
        conversation's identity at construction (design §3, decision log item
        3), posture derives from the BOUND identity: a later live config
        change is a pending selection for new conversations and never
        re-postures an existing manager. The carve-out is the identity's
        execution exception to its personal posture (settlement decision 3);
        the delegation axis never receives it.
        """
        if self._bound_identity is not None:
            reference = self._bound_identity
        else:
            reference = self._config.agent.identity
        if reference.startswith("user:"):
            return None, None, frozenset()
        identity = get_builtin_identity(reference.removeprefix("builtin:"))
        if identity is None:
            return None, None, frozenset()
        return (
            identity.personal_tools,
            identity.delegation_tools,
            identity.execution_carve_out,
        )

    def _personal_posture_allows(self, tool_class: type[BaseTool]) -> bool:
        """Whether this identity may execute the SELECTED implementation itself.

        The classification is computed from the exact selected class's origin
        — never re-resolved by name: with several variants of one name, a
        name-keyed lookup can classify a DIFFERENT class than the one the
        availability decision actually selected, letting a foreign custom
        ride a read-only posture (S2 round 2: posture reselects by name). The
        identity's execution carve-out admits a carved-out name only for a
        class defined by a shipped builtin module, so a foreign custom variant
        of that name fails closed exactly as any other custom does.
        """
        personal, _, carve_out = self._identity_posture()
        if personal is None:
            return True
        classification = registry_provenance_classification(tool_class)
        if posture_allows_class(personal, classification):
            return True
        if not carve_out:
            return False
        builtin = shipped_builtin_name(tool_class)
        return builtin is not None and builtin in carve_out

    def _delegation_posture_allows(self, tool_class: type[BaseTool]) -> bool:
        """Whether this identity may pass the SELECTED implementation to children.

        Classifies the exact selected class, exactly as the personal axis does
        (S2 round 2: posture reselects by name). The execution carve-out is
        personal-axis only: a read-only identity's delegation set stays
        read-only, so a reviewer cannot hand its shell to a child.
        """
        _, delegation, _ = self._identity_posture()
        if delegation is None:
            return True
        return posture_allows_class(
            delegation, registry_provenance_classification(tool_class)
        )

    @property
    def authority_revision(self) -> int | None:
        if self._parent_authority_revision_getter is None:
            return None
        try:
            return self._parent_authority_revision_getter()
        except Exception:
            return None

    def _tool_filter_digest(self) -> tuple[object, ...]:
        """Live digest of the granted-axis tool filters.

        The authority token is the sanctioned commit signal, but an
        unsanctioned in-place config edit (no accepted-token bump) must still
        invalidate every token-keyed cache: the digest covers the tool
        allow/deny lists and the per-tool config mapping (S2 W2).
        """
        config = self._config
        entries: list[tuple[str, str]] = []
        for name, entry in config.tools.items():
            if isinstance(entry, Mapping):
                try:
                    entries.append((
                        name,
                        json.dumps(entry, sort_keys=True, default=repr),
                    ))
                except (TypeError, ValueError):
                    entries.append((name, repr(entry)))
            else:
                entries.append((name, repr(entry)))
        entries.sort()
        return (
            tuple(config.enabled_tools),
            tuple(config.disabled_tools),
            tuple(entries),
        )

    def _effective_authority_token(  # noqa: PLR0911 - each missing signal fails closed
        self, seen: set[int] | None = None
    ) -> tuple[object, ...] | None:
        """A revision for accepted inputs along the *entire* authority chain.

        An unavailable parent or revision signal disables caching; never reuse an
        entry obtained while the chain was healthy. The token also carries the
        live identity selection and the granted-axis tool filters, so an
        unsanctioned in-place edit invalidates cached authority even without an
        accepted-token bump.
        """
        if self._authority_retired or self._accepted_token_getter is None:
            return None
        seen = set() if seen is None else seen
        if id(self) in seen:
            return None
        seen.add(id(self))
        try:
            accepted = self._accepted_token_getter()
            if accepted is None:
                return None
            parent_token = None
            parent_identity = None
            if self._parent_authority_getter is not None:
                parent = self._parent_authority()
                if parent is None:
                    return None
                parent_identity = id(parent)
                parent_token = parent._effective_authority_token(seen)
                if parent_token is None:
                    return None
            return (
                id(self),
                self._authority_generation,
                accepted,
                self._inherited_restrictions,
                self._inherited_workspace,
                self._cwd,
                self._registry_generation,
                classification_revision(),
                self._bound_identity,
                self._config.agent.identity,
                self._tool_filter_digest(),
                parent_identity,
                parent_token,
            )
        except Exception:
            return None
        finally:
            seen.remove(id(self))

    def _name_versionable(self, name: str, seen: set[int] | None = None) -> bool:
        seen = set() if seen is None else seen
        if id(self) in seen:
            return False
        seen.add(id(self))
        policy_names = (name, "bash") if name == "bash_start" else (name,)
        with self._lock:
            variants = tuple(
                cls
                for policy_name in policy_names
                for cls in self._tool_variants_by_name.get(policy_name, ())
            )
        if any(
            getattr(cls.is_available, "__func__", None)
            is not BaseTool.is_available.__func__
            for cls in variants
        ):
            return False
        if self._parent_authority_getter is None:
            return True
        parent = self._parent_authority()
        return parent is not None and parent._name_versionable(name, seen)

    def _select_available_variant_or_none(
        self, name: str, fallback: type[BaseTool] | None
    ) -> type[BaseTool] | None:
        with self._lock:
            return (
                self._select_available_variant(name, fallback)
                if fallback is not None
                else None
            )

    def _live_tool_policy_allows(self, tool_name: str) -> bool:
        """Whether the live per-tool config entry is well-formed for a tool.

        The schema types ``tools`` entries as config mappings; an entry that is
        not a mapping is a broken config, and broken config fails closed — the
        tool is never served from a warm selection. An ABSENT entry means "no
        per-tool policy" (allowed); a PRESENT but malformed entry — an
        explicit ``None`` (malformed per the schema type) or any other
        non-mapping value — fails closed on every surface, never a silent
        reset to defaults (S2 round 2: explicit null tool entry fails open).
        Well-formed entries keep the pinned execution-time denial semantics
        (``permission = "never"`` denies at the execution decision, not at
        selection).
        """
        name = "bash" if tool_name == "bash_start" else tool_name
        tools = self._config.tools
        if name not in tools:
            return True
        return isinstance(tools[name], Mapping)

    def _apply_live_tool_filters(
        self,
        name: str,
        selected: type[BaseTool],
        *,
        personal: bool,
        chain_class: type[BaseTool] | None = None,
    ) -> type[BaseTool] | None:
        """Live granted-axis and posture filters for one availability decision.

        These filters re-run on EVERY decision, cache hits included: the
        selection caches key on the authority token, but an unsanctioned
        in-place config edit (no accepted-token bump) must not leave a warm
        selection serving a tool the current policy excludes (S2 W2).

        The posture check classifies the EXACT selected class — never a
        name-keyed re-resolution, which can pick a different variant of the
        same name than the one this decision selected (S2 round 2: posture
        reselects by name).

        ``chain_class`` is the exact implementation the ANCESTOR chain must
        classify: the caller's selected implementation when the caller is a
        descendant, or the caller's selected implementation carried through a
        delegation check. It defaults to ``selected`` — this manager's own
        variant of the name. The ancestor chain never re-resolves the name
        inside an ancestor's registry: that selects the ANCESTOR's variant of
        the name and can admit a mutating custom implementation through a
        read-only ancestor (S2 round 3: the ancestor chain re-resolves by
        name).
        """
        if chain_class is None:
            chain_class = selected
        config = self._config
        disabled, per_source = self._build_source_disable_index(config)
        if (
            self._is_source_disabled(selected, disabled, per_source)  # noqa: PLR0916
            or (config.enabled_tools and not name_matches(name, config.enabled_tools))
            or (config.disabled_tools and name_matches(name, config.disabled_tools))
            or not self._live_tool_policy_allows(name)
            or not (
                self._personal_posture_allows(selected)
                if personal
                else self._delegation_posture_allows(chain_class)
            )
            or not (
                self._parent_allows_tool(name, chain_class)
                if personal
                else self._parent_delegation_allows(name, chain_class)
            )
        ):
            return None
        return selected

    def _available_tool(self, name: str) -> type[BaseTool] | None:
        """Resolve only one name; custom availability callbacks remain live."""
        if self._authority_retired:
            return None
        token = self._effective_authority_token()
        with self._lock:
            fallback = self._all_tools.get(name)
        versionable = self._name_versionable(name)
        selected: type[BaseTool] | None
        if token is not None and versionable:
            cached = self._selection_cache.get(name)
            if cached is not None and cached[0] == token:
                selected = cached[1]
            else:
                selected = self._select_available_variant_or_none(name, fallback)
        else:
            selected = self._select_available_variant_or_none(name, fallback)
        if selected is not None:
            # Live filters run on cache hits too (S2 W2): a warm selection
            # cannot serve a tool the current policy excludes.
            selected = self._apply_live_tool_filters(name, selected, personal=True)
        if token is not None:
            if self._effective_authority_token() != token:
                return None
            if versionable:
                self._selection_cache[name] = (token, selected)
        return selected

    def _parent_allows_tool(
        self, tool_name: str, selected_class: type[BaseTool] | None = None
    ) -> bool:
        """Chain-walk: every ancestor's DELEGATION ceiling admits the implementation.

        The child ceiling is the parent's delegation set, not its personal
        inventory: personal posture and delegation are independent axes, and
        the granted ceiling never widens down the tree. The chain classifies
        the EXACT implementation this decision selected — never a name-keyed
        re-resolution inside an ancestor, which selects the ANCESTOR's variant
        of the name and can admit a mutating custom implementation through a
        read-only ancestor (S2 round 3: the ancestor chain re-resolves by
        name).
        """
        if tool_name == "bash_start" and self._available_tool("bash") is None:
            return False
        if self._parent_authority_getter is None:
            return True
        parent = self._parent_authority()
        return parent is not None and parent._delegation_allows_tool(
            tool_name, selected_class
        )

    def _parent_delegation_allows(
        self, tool_name: str, selected_class: type[BaseTool] | None = None
    ) -> bool:
        """Ancestor chain only: every ancestor's delegation ceiling admits it."""
        if self._parent_authority_getter is None:
            return True
        parent = self._parent_authority()
        return parent is not None and parent._delegation_allows_tool(
            tool_name, selected_class
        )

    def _delegation_allows_tool(
        self, tool_name: str, selected_class: type[BaseTool] | None = None
    ) -> bool:
        """This identity's full delegation ceiling admits the implementation.

        Granted configuration ∩ this identity's delegation posture ∩ every
        ancestor's delegation ceiling. The personal posture never narrows
        what an identity may delegate: an orchestrator delegates writes it
        cannot perform itself.

        ``selected_class`` is the exact implementation the CALLER will run:
        every delegation posture along the chain classifies THAT class, never
        a name-keyed re-resolution inside this identity's registry (S2 round
        3: the ancestor chain re-resolves by name). When omitted, this
        identity's own selected variant of the name is judged — the
        delegation-axis enumeration of what this identity may pass.
        """
        return self._delegated_available_tool(tool_name, selected_class) is not None

    def _delegated_available_tool(
        self, name: str, selected_class: type[BaseTool] | None = None
    ) -> type[BaseTool] | None:
        """Delegation-axis availability, cached like ``_available_tool``.

        Mirrors the personal path without the personal posture: the class
        resolves under the delegation ceiling (own granted config, own
        delegation posture, ancestors' delegation). Live filters re-run on
        cache hits too, so an in-place policy edit cannot serve a warm
        selection (S2 W2).

        ``selected_class`` carries the caller's EXACT implementation through
        the ancestor chain: every ancestor's delegation posture classifies
        THAT class, never a name-keyed re-resolution inside the ancestor's
        registry (S2 round 3: the ancestor chain re-resolves by name). When
        omitted, this identity's own selected variant of the name is carried
        instead — the delegation-axis enumeration judges the class it would
        pass. Cached decisions are keyed on the carried implementation as
        well (verified by identity through a weak reference), so a warm entry
        computed for one implementation is never served for another.
        """
        if self._authority_retired:
            return None
        token = self._effective_authority_token()
        with self._lock:
            fallback = self._all_tools.get(name)
        versionable = self._name_versionable(name)
        cache_key: str | tuple[str, int] = (
            name if selected_class is None else (name, id(selected_class))
        )
        selected: type[BaseTool] | None
        if token is not None and versionable:
            cached = self._delegation_selection_cache.get(cache_key)
            if (
                cached is not None
                and cached[0] == token
                and (
                    selected_class is None
                    or cached[2] is not None
                    and cached[2]() is selected_class
                )
            ):
                selected = cached[1]
            else:
                selected = self._select_available_variant_or_none(name, fallback)
        else:
            selected = self._select_available_variant_or_none(name, fallback)
        if selected is not None:
            if (
                name == "bash_start"
                and self._delegated_available_tool("bash", selected_class) is None
            ):
                selected = None
            else:
                selected = self._apply_live_tool_filters(
                    name, selected, personal=False, chain_class=selected_class
                )
        if token is not None:
            if self._effective_authority_token() != token:
                return None
            if versionable:
                self._delegation_selection_cache[cache_key] = (
                    token,
                    selected,
                    None if selected_class is None else weakref.ref(selected_class),
                )
        return selected

    def path_authority(self, tool_name: str) -> PathAuthority:
        """Capture local policy with live chain-walked runtime capabilities."""
        if self._authority_retired:
            raise NoSuchToolError("Tool manager authority retired")
        if tool_name == "bash_start":
            # Availability is still name-specific; execution paths use bash policy.
            if self._available_tool(tool_name) is None:
                raise NoSuchToolError(f"Unknown or disabled tool: {tool_name}")
            tool_name = "bash"
        config = self.get_tool_config(tool_name)
        tool_class = self._available_tool(tool_name)
        if tool_class is None:
            raise NoSuchToolError(f"Unknown or disabled tool: {tool_name}")
        return PathAuthority(
            tool_name,
            config.permission,
            tuple(config.allowlist),
            tuple(config.denylist),
            tool_class.path_sensitive_patterns(config),
            self.workspace,
            self._scratchpad_dir,
            self.scratchpad_roots,
            self.instruction_read_files,
            plan_file_write_scope=(
                self._plan_file_write_scope_getter()
                if tool_name in {"write_file", "edit"}
                and self._plan_file_write_scope_getter is not None
                else None
            ),
            inherited_plan_write_scopes=(
                self._inherited_plan_write_scopes
                if tool_name in {"write_file", "edit"}
                else ()
            ),
        )

    def _delegated_path_authority(
        self, tool_name: str, selected_class: type[BaseTool] | None = None
    ) -> PathAuthority:
        """Delegation-axis path policy for ancestor policy resolution.

        Mirrors ``path_authority`` under the delegation ceiling, so an
        ancestor whose personal posture excludes the tool still contributes
        its granted path policy to delegated execution. ``selected_class``
        keys the liveness exactly as in ``_delegated_tool`` (S2 round 4:
        invocation/availability implementation consistency).
        """
        if self._authority_retired:
            raise NoSuchToolError("Tool manager authority retired")
        if tool_name == "bash_start":
            # Availability is still name-specific; execution paths use bash policy.
            if self._delegated_available_tool(tool_name, selected_class) is None:
                raise NoSuchToolError(f"Unknown or disabled tool: {tool_name}")
            tool_name = "bash"
        config = self.get_tool_config(tool_name)
        tool_class = self._delegated_available_tool(tool_name, selected_class)
        if tool_class is None:
            raise NoSuchToolError(f"Unknown or disabled tool: {tool_name}")
        return PathAuthority(
            tool_name,
            config.permission,
            tuple(config.allowlist),
            tuple(config.denylist),
            tool_class.path_sensitive_patterns(config),
            self.workspace,
            self._scratchpad_dir,
            self.scratchpad_roots,
            self.instruction_read_files,
            plan_file_write_scope=(
                self._plan_file_write_scope_getter()
                if tool_name in {"write_file", "edit"}
                and self._plan_file_write_scope_getter is not None
                else None
            ),
            inherited_plan_write_scopes=(
                self._inherited_plan_write_scopes
                if tool_name in {"write_file", "edit"}
                else ()
            ),
        )

    def snapshot_path_authority(
        self, tool_name: str, resolver: Callable[..., Any]
    ) -> tuple[tuple[object, ...] | None, PathAuthority | None]:
        """Conjoin builtin path policies only when the entire chain is versioned.

        Unavailable or custom policies require live invocation resolution, never
        an approximation that could turn missing authority into permission.
        """
        from dataclasses import replace

        token = self._effective_authority_token()
        if token is None:
            return None, None
        states: list[PathAuthority] = []
        try:
            roots = self.scratchpad_roots
            files = self.instruction_read_files
            cursor: ToolManager | None = self
            while cursor is not None:
                if cursor._authority_retired or not cursor._name_versionable(tool_name):
                    return None, None
                tool = cursor.get(tool_name)
                if type(tool).resolve_permission is not resolver:
                    return token, None
                states.append(
                    replace(
                        cursor.path_authority(tool_name),
                        scratchpad_roots=roots,
                        instruction_read_files=files,
                    )
                )
                cursor = (
                    cursor._parent_authority()
                    if cursor._parent_authority_getter is not None
                    else None
                )
        except Exception:
            return None, None
        if self._effective_authority_token() != token:
            return None, None
        return token, replace(states[0], parents=tuple(states[1:]))

    def _shell_execution_permission(
        self,
        args: Any,
        local: Any = None,
        execution_class: type[BaseTool] | None = None,
    ) -> Any:
        # Launch arguments are not foreground arguments. Custom resolvers
        # must receive the canonical schema, under the same request ceilings.
        from chartreux.core.tools.builtins.bash import Bash, BashArgs
        from chartreux.core.tools.builtins.bash_start import BashStart
        from chartreux.core.tools.permissions import PermissionContext
        from chartreux.core.tools.utils import (
            instruction_read_ceiling,
            scratchpad_ceiling,
        )

        try:
            canonical = self.get("bash")
            if type(canonical) is Bash and type(self.get("bash_start")) is BashStart:
                # The builtin adapter already analyzed canonical Bash config and
                # the complete path-authority chain under these request ceilings.
                context = local
            else:
                with (
                    instruction_read_ceiling(self.instruction_read_files),
                    scratchpad_ceiling(self.scratchpad_roots),
                ):
                    context = canonical.resolve_permission(
                        BashArgs(command=args.command)
                    )
            if self.get_tool_config("bash").permission == ToolPermission.NEVER:
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason="Shell execution is disabled by canonical bash policy",
                )
            if context is not None and context.permission == ToolPermission.NEVER:
                if context.denial_kind == "dangerous_command":
                    # The parent ceiling is checked even when local dangerous
                    # authority is missing: a parent dangerous-authority gap
                    # is never locally grantable, so it classifies as the
                    # parent ceiling instead of a grantable local denial.
                    parent = self._parent_permission(
                        "bash_start", args, execution_class
                    )
                    if parent is not None and parent.permission == ToolPermission.NEVER:
                        return parent
                return context
            return self._parent_permission("bash_start", args, execution_class)
        except Exception:
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="Canonical shell authority is unavailable",
            )

    def _delegated_shell_execution_permission(
        self,
        args: Any,
        local: Any = None,
        execution_class: type[BaseTool] | None = None,
    ) -> Any:
        """Delegation-axis shell guard: canonical bash policy under the delegation ceiling.

        Mirrors ``_shell_execution_permission`` for instances that resolve an
        ancestor's policy for a tool the ancestor's personal posture excludes.
        """
        # Launch arguments are not foreground arguments. Custom resolvers
        # must receive the canonical schema, under the same request ceilings.
        from chartreux.core.tools.builtins.bash import Bash, BashArgs
        from chartreux.core.tools.builtins.bash_start import BashStart
        from chartreux.core.tools.permissions import PermissionContext
        from chartreux.core.tools.utils import (
            instruction_read_ceiling,
            scratchpad_ceiling,
        )

        try:
            canonical = self._delegated_tool("bash")
            if (
                type(canonical) is Bash
                and type(self._delegated_tool("bash_start")) is BashStart
            ):
                # The builtin adapter already analyzed canonical Bash config and
                # the complete path-authority chain under these request ceilings.
                context = local
            else:
                with (
                    instruction_read_ceiling(self.instruction_read_files),
                    scratchpad_ceiling(self.scratchpad_roots),
                ):
                    context = canonical.resolve_permission(
                        BashArgs(command=args.command)
                    )
            if self.get_tool_config("bash").permission == ToolPermission.NEVER:
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason="Shell execution is disabled by canonical bash policy",
                )
            if context is not None and context.permission == ToolPermission.NEVER:
                if context.denial_kind == "dangerous_command":
                    # The parent ceiling is checked even when local dangerous
                    # authority is missing: a parent dangerous-authority gap
                    # is never locally grantable, so it classifies as the
                    # parent ceiling instead of a grantable local denial.
                    parent = self._parent_permission(
                        "bash_start", args, execution_class
                    )
                    if parent is not None and parent.permission == ToolPermission.NEVER:
                        return parent
                return context
            return self._parent_permission("bash_start", args, execution_class)
        except Exception:
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="Canonical shell authority is unavailable",
            )

    def _parent_permission(
        self, tool_name: str, args: Any, execution_class: type[BaseTool] | None = None
    ) -> Any:
        """Return a parent invocation denial, including tool-specific guards.

        The parent ceiling is the DELEGATION axis: the ancestor lends its
        policy for a tool its own personal posture excludes (an orchestrator
        delegates writes it cannot perform itself).

        ``execution_class`` is the exact implementation THIS invocation will
        run. The parent's delegation posture classifies THAT class — exactly
        like the availability chain — never a name-keyed re-resolution inside
        the parent's registry, which selects the PARENT's variant of the name
        and can deny an admitted child implementation (or admit one the
        availability chain denies) when the two registries select different
        variants of the same name (S2 round 4: invocation/availability
        implementation consistency). The POLICY computation stays the parent's
        own: the policy instance is built from the parent's own selected
        variant under the parent's config, so a child-forged
        ``resolve_permission`` never weakens the parent's ceiling.

        Name-vs-class coherence (round-4 W2/W3): this runtime check is
        class-exact — it classifies ``execution_class``, the exact
        implementation this invocation runs — while the admission ceiling
        (``delegable_tools``) is name-keyed on the parent's own variant.
        Both derive posture from the conversation's bound identity
        (``_bound_identity``), never from a live ``[agent]`` identity
        re-read.
        """
        from chartreux.core.tools.permissions import PermissionContext

        if self._parent_authority_getter is None:
            return None
        parent = self._parent_authority()
        if parent is None or not parent._delegation_allows_tool(
            tool_name, execution_class
        ):
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="Tool is unavailable under parent authority",
            )
        try:
            parent_tool = parent._delegated_tool(tool_name, execution_class)
            from chartreux.core.tools.utils import (
                instruction_read_ceiling,
                scratchpad_ceiling,
            )

            with (
                instruction_read_ceiling(self.instruction_read_files),
                scratchpad_ceiling(self.scratchpad_roots),
            ):
                parent_context = parent_tool.resolve_permission(args)
            if parent.get_tool_config(tool_name).permission == ToolPermission.NEVER:
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason="Tool is disabled by parent policy",
                )
            if (
                parent_context is not None
                and parent_context.permission == ToolPermission.NEVER
            ):
                # A parent's out-of-root denial is this agent's ceiling, not a
                # grantable gap: no session grant can lift the live parent.
                if parent_context.denial_kind == "out_of_root":
                    parent_context = parent_context.model_copy(
                        update={"denial_kind": "parent_ceiling"}
                    )
                elif parent_context.denial_kind == "dangerous_command":
                    # A parent's dangerous-authority gap is likewise a ceiling:
                    # the parent re-resolves the call under its own authority,
                    # so no local grant can lift it, and the grantable proposal
                    # never travels up to the child.
                    parent_context = parent_context.model_copy(
                        update={
                            "denial_kind": "parent_ceiling",
                            "dangerous_command_proposal": None,
                        }
                    )
                return parent_context
        except Exception:
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="Parent tool authority is unavailable",
            )
        return None

    @property
    def settings_inventory(self) -> list[str]:
        """Runtime-capable tool names before the user's allow/deny filters."""
        with self._lock:
            candidates = {
                name: selected
                for name, fallback in self._all_tools.items()
                if (selected := self._select_available_variant(name, fallback))
                is not None
            }
        return sorted(
            name
            for name, selected in self._apply_per_source_filtering(candidates).items()
            if self._parent_allows_tool(name, selected)
        )

    def _config_available_tools(self) -> dict[str, type[BaseTool]]:
        """Runtime-capable tools after per-source and allow/deny filters.

        Shared base of the personal and delegation surfaces: no posture and
        no ancestor chain are applied here.
        """
        with self._lock:
            runtime_available: dict[str, type[BaseTool]] = {}
            for name, fallback_tool_class in self._all_tools.items():
                selected_tool_class = self._select_available_variant(
                    name, fallback_tool_class
                )
                if selected_tool_class is None:
                    continue
                runtime_available[name] = selected_tool_class

        # Per-source filtering first (MCP server/connector disabled flags).
        result = self._apply_per_source_filtering(runtime_available)

        # Global allowlist narrows the candidate set; denylist is always final.
        if self._config.enabled_tools:
            result = {
                name: cls
                for name, cls in result.items()
                if name_matches(name, self._config.enabled_tools)
            }
        if self._config.disabled_tools:
            result = {
                name: cls
                for name, cls in result.items()
                if not name_matches(name, self._config.disabled_tools)
            }
        return result

    @property
    def available_tools(self) -> dict[str, type[BaseTool]]:
        """Tools this identity may execute itself (personal axis).

        Granted configuration ∩ the identity's personal posture ∩ every
        ancestor's delegation ceiling. The live per-tool policy check runs on
        this enumeration too, so a malformed in-place ``tools`` entry fails
        closed on every surface — inventory, specs, and admission ceilings —
        exactly as on the ``get()`` path (S2 W2). The posture check classifies
        the exact selected class of each entry, and the ancestor chain
        classifies that same class (S2 round 2: posture reselects by name;
        S2 round 3: the ancestor chain re-resolves by name).
        """
        return {
            name: cls
            for name, cls in self._config_available_tools().items()
            if self._live_tool_policy_allows(name)
            and self._personal_posture_allows(cls)
            and self._parent_allows_tool(name, cls)
        }

    @property
    def delegable_tools(self) -> dict[str, type[BaseTool]]:
        """Tools this identity may pass to children (delegation axis).

        Granted configuration ∩ the identity's delegation posture ∩ every
        ancestor's delegation ceiling. This is the admission ceiling for
        child launches; the personal posture never narrows it.

        Each name is resolved ONCE on the delegation axis
        (``_delegated_available_tool``), and the listed class is exactly the
        class that decision validated: the inventory never lists a class a
        separate name-keyed re-resolution failed to check (S2 round 2:
        posture reselects by name). The live per-tool policy check applies
        here as well (S2 W2).

        Name-vs-class coherence (round-4 W2/W3): the ADMISSION ceiling is
        name-keyed — each name judged on THIS manager's own selected variant
        of it — while the RUNTIME authority check (``_parent_permission``)
        is class-exact, classifying the exact implementation the invocation
        runs. Both axes derive their posture from the conversation's bound
        identity (``_bound_identity``), never from a live ``[agent]``
        identity re-read, so a later record re-key cannot re-introduce an
        identity read that diverges from the bound identity.
        """
        result: dict[str, type[BaseTool]] = {}
        with self._lock:
            names = list(self._all_tools)
        for name in names:
            delegated = self._delegated_available_tool(name)
            if delegated is not None:
                result[name] = delegated
        return result

    @property
    def custom_tool_names(self) -> set[str]:
        available_names = set(self.available_tools)
        custom_names: set[str] = set()
        with self._lock:
            for name in available_names:
                fallback_tool_class = self._all_tools.get(name)
                if fallback_tool_class is None:
                    continue
                selected = self._select_available_variant_with_index(
                    name, fallback_tool_class
                )
                if selected is None:
                    continue
                _, discovery_index = selected
                custom_variants = self._custom_tool_variants_by_name.get(name, [])
                if (
                    discovery_index < len(custom_variants)
                    and custom_variants[discovery_index]
                ):
                    custom_names.add(name)
        return custom_names

    def _is_tool_available(self, cls: type[BaseTool]) -> bool:
        return self._is_tool_runtime_available(cls)

    def _is_tool_runtime_available(self, cls: type[BaseTool]) -> bool:
        # Backwards-compatibility check to avoid breaking
        # existing custom tools that call is_available without parameters
        if inspect.signature(cls.is_available).parameters:
            return cls.is_available(self._config)
        return cls.is_available()

    def _tool_variants_for_name(
        self, name: str, fallback: type[BaseTool]
    ) -> list[type[BaseTool]]:
        return self._tool_variants_by_name.get(name) or [fallback]

    def _select_available_variant(
        self, name: str, fallback: type[BaseTool]
    ) -> type[BaseTool] | None:
        selected = self._select_available_variant_with_index(name, fallback)
        if selected is None:
            return None
        return selected[0]

    def _select_available_variant_with_index(
        self, name: str, fallback: type[BaseTool]
    ) -> tuple[type[BaseTool], int] | None:
        selected_tool_class: type[BaseTool] | None = None
        selected_discovery_index = 0
        selected_rank: tuple[int, int] | None = None

        for discovery_index, tool_class in enumerate(
            self._tool_variants_for_name(name, fallback)
        ):
            if not self._is_tool_available(tool_class):
                continue

            rank = (self._tool_selection_priority(tool_class), discovery_index)
            if selected_rank is not None and rank <= selected_rank:
                continue

            selected_tool_class = tool_class
            selected_discovery_index = discovery_index
            selected_rank = rank

        if selected_tool_class is None:
            return None
        return selected_tool_class, selected_discovery_index

    @staticmethod
    def _tool_selection_priority(tool_class: type[BaseTool]) -> int:
        return tool_class.selection_priority

    def _select_registered_variant(
        self, name: str, fallback: type[BaseTool]
    ) -> type[BaseTool]:
        selected_tool_class = self._select_available_variant(name, fallback)
        if selected_tool_class is not None:
            return selected_tool_class
        return fallback

    def _tool_class_for_config(self, tool_name: str) -> type[BaseTool] | None:
        fallback_tool_class = self._all_tools.get(tool_name)
        if fallback_tool_class is None:
            return None
        return self._select_registered_variant(tool_name, fallback_tool_class)

    def _apply_per_source_filtering(
        self, tools: dict[str, type[BaseTool]]
    ) -> dict[str, type[BaseTool]]:
        """Filter out MCP/connector tools disabled at the server or connector level."""
        disabled_sources, per_source_disabled = self._build_source_disable_index()
        if not disabled_sources and not per_source_disabled:
            return tools

        return {
            name: cls
            for name, cls in tools.items()
            if not self._is_source_disabled(cls, disabled_sources, per_source_disabled)
        }

    def _build_source_disable_index(
        self, config: ChartreuxConfigSchema | None = None
    ) -> tuple[set[str], dict[str, set[str]]]:
        """Return (fully_disabled, per_tool_disabled) keyed by source name."""
        disabled_sources: set[str] = set()
        per_source_disabled: dict[str, set[str]] = {}

        for srv in (config if config is not None else self._config).mcp_servers:
            key = srv.name
            if srv.disabled:
                disabled_sources.add(key)
            elif srv.disabled_tools:
                per_source_disabled[key] = set(srv.disabled_tools)

        return disabled_sources, per_source_disabled

    @staticmethod
    def _is_source_disabled(
        tool_cls: type[BaseTool],
        disabled_sources: set[str],
        per_source_disabled: dict[str, set[str]],
    ) -> bool:
        if not ToolManager._is_remote_tool_class(tool_cls):
            return False
        server_name = tool_cls.get_server_name()
        if server_name is None:
            return False
        key = server_name
        if key in disabled_sources:
            return True
        return tool_cls.get_remote_name() in per_source_disabled.get(key, set())

    @staticmethod
    def _is_remote_tool_class(tool_cls: type[BaseTool]) -> TypeGuard[type[MCPTool]]:
        return issubclass(tool_cls, MCPTool)

    def integrate_mcp(self, *, raise_on_failure: bool = False) -> None:
        """Discover and register MCP tools (sync wrapper).

        Idempotent: subsequent calls after a successful integration are
        no-ops to avoid redundant MCP discovery.
        """
        run_sync(self._integrate_mcp_async(raise_on_failure=raise_on_failure))

    async def _integrate_mcp_async(self, *, raise_on_failure: bool = False) -> None:
        """Async MCP discovery — canonical implementation."""
        if self._mcp_integrated:
            return
        if not self._config.mcp_servers:
            if self._mcp_registry is not None:
                self._mcp_registry.sync_active_servers([])
            return

        try:
            mcp_tools = await self._get_mcp_registry().get_tools_async(
                self._config.mcp_servers
            )
        except Exception as exc:
            logger.warning("MCP integration failed: %s", exc)
            if raise_on_failure:
                raise
            return

        with self._lock:
            for name, tool_class in mcp_tools.items():
                self._tool_variants_by_name.setdefault(name, []).append(tool_class)
                self._custom_tool_variants_by_name.setdefault(name, []).append(False)
                # Preserve the discovered fallback so an MCP collision can be
                # selected alongside it and safely withdrawn on reconfiguration.
                self._all_tools.setdefault(name, tool_class)
            self._registry_generation += 1
        self._mcp_integrated = True
        logger.info(
            "MCP integration registered %d tools (via registry)", len(mcp_tools)
        )

    def _purge_mcp_state(self) -> None:
        """Remove stale MCP tool classes and cached instances (under _lock)."""
        self._registry_generation += 1
        for name, variants in tuple(self._tool_variants_by_name.items()):
            origins = self._custom_tool_variants_by_name[name]
            retained = [
                (tool_class, is_custom)
                for tool_class, is_custom in zip(variants, origins, strict=True)
                if not self._is_remote_tool_class(tool_class)
            ]
            if retained:
                if len(retained) != len(variants):
                    self._instances.pop(name, None)
                self._tool_variants_by_name[name] = [item[0] for item in retained]
                self._custom_tool_variants_by_name[name] = [
                    item[1] for item in retained
                ]
            else:
                self._tool_variants_by_name.pop(name)
                self._custom_tool_variants_by_name.pop(name)

        stale_keys = [
            name
            for name, cls in self._all_tools.items()
            if self._is_remote_tool_class(cls)
        ]
        for key in stale_keys:
            self._all_tools.pop(key, None)
            self._instances.pop(key, None)

    async def refresh_remote_tools_async(self) -> None:
        """Force MCP re-discovery for the current config."""
        with self._lock:
            if self._mcp_registry is not None:
                self._mcp_registry.clear()
            self._purge_mcp_state()
            self._mcp_integrated = False

        await self._integrate_all_async()

    async def reconfigure_mcp_async(self) -> None:
        """Rebuild MCP tool visibility while retaining valid registry descriptors."""
        with self._lock:
            self._purge_mcp_state()
            self._mcp_integrated = False
            if self._mcp_registry is not None:
                self._mcp_registry.sync_active_servers(self._config.mcp_servers)
        await self._integrate_mcp_async()

    def suspend_mcp(self, name: str, tool_name: str | None = None) -> None:
        """Withdraw one source or remote tool before reducing its authority."""
        with self._lock:
            self._registry_generation += 1
            for key, variants in tuple(self._tool_variants_by_name.items()):
                origins = self._custom_tool_variants_by_name[key]
                retained = [
                    (tool_class, is_custom)
                    for tool_class, is_custom in zip(variants, origins, strict=True)
                    if not (
                        self._is_remote_tool_class(tool_class)
                        and tool_class.get_server_name() == name
                        and (
                            tool_name is None
                            or tool_class.get_remote_name() == tool_name
                        )
                    )
                ]
                if retained:
                    if len(retained) != len(variants):
                        self._instances.pop(key, None)
                    self._tool_variants_by_name[key] = [item[0] for item in retained]
                    self._custom_tool_variants_by_name[key] = [
                        item[1] for item in retained
                    ]
                else:
                    self._tool_variants_by_name.pop(key)
                    self._custom_tool_variants_by_name.pop(key)

            stale_keys = [
                key
                for key, tool_class in self._all_tools.items()
                if self._is_remote_tool_class(tool_class)
                and tool_class.get_server_name() == name
                and (tool_name is None or tool_class.get_remote_name() == tool_name)
            ]
            for key in stale_keys:
                self._all_tools.pop(key, None)
                self._instances.pop(key, None)

    def refresh_remote_tools(self) -> None:
        """Sync wrapper for :meth:`refresh_remote_tools_async`."""
        run_sync(self.refresh_remote_tools_async())

    def integrate_all(self, *, raise_on_mcp_failure: bool = False) -> None:
        """Discover MCP tools.

        Runs the async discovery path via ``run_sync``.
        """
        run_sync(self._integrate_all_async(raise_on_mcp_failure=raise_on_mcp_failure))

    async def _integrate_all_async(self, *, raise_on_mcp_failure: bool = False) -> None:
        """Run MCP discovery.

        Uses ``return_exceptions=True`` so that a failing MCP server does not
        cancel in-flight discovery.
        """
        await self._integrate_mcp_async(raise_on_failure=raise_on_mcp_failure)

    def available_tool_specs(self) -> list[AvailableFunction]:
        """Model-facing definitions for every available tool: name, resolved
        description, and parameters.

        Cache JSON schemas by the currently available tool classes, their argument
        models, resolved descriptions, and (for MCP tools) live input schemas. The
        active set is recomputed first, so configuration filters, MCP changes, and
        parent authority changes remain visible. Copy schemas on return because
        consumers may mutate them. The key captures tool names, classes,
        argument-model classes, descriptions, and live remote parameter-schema
        content; future ``get_parameters`` dependencies on other mutable state must
        extend the key to avoid serving stale schemas.
        """
        tool_inputs = {
            name: (
                cls,
                cls._get_tool_args_results()[0],
                self._tool_descriptions.get(name) or cls.get_full_description(),
                (
                    json.dumps(
                        cls.get_parameters(), sort_keys=True, separators=(",", ":")
                    )
                    if self._is_remote_tool_class(cls)
                    else None
                ),
            )
            for name, cls in self.available_tools.items()
        }
        cache_key = frozenset(
            (name, cls, args_model, description, schema_state)
            for name, (
                cls,
                args_model,
                description,
                schema_state,
            ) in tool_inputs.items()
        )
        with self._available_tool_specs_cache_lock:
            if cache_key != self._available_tool_specs_cache_key:
                self._available_tool_specs_cache = {
                    name: json.dumps(cls.get_parameters(), separators=(",", ":"))
                    for name, (cls, _, _, _) in tool_inputs.items()
                }
                self._available_tool_specs_cache_key = cache_key
            cached_specs = self._available_tool_specs_cache
        return [
            AvailableFunction(
                name=name,
                description=description,
                parameters=json.loads(cached_specs[name]),
            )
            for name, (_, _, description, _) in tool_inputs.items()
        ]

    def get_tool_config(  # noqa: PLR0912, PLR0915 - restriction sources and cache guards
        self, tool_name: str
    ) -> BaseToolConfig:
        # Managed launches have exactly one permission source: [tools.bash].
        if tool_name == "bash_start":
            return self.get_tool_config("bash")
        if self._authority_retired:
            raise NoSuchToolError("Tool manager authority retired")
        token = self._effective_authority_token()
        # Unversioned availability predicates must be consulted on every lookup.
        versionable = self._name_versionable(tool_name)
        if token is not None and versionable:
            cached = self._authority_cache.get(tool_name)
            if cached is not None and cached[0] == token:
                return cached[1].model_validate_json(cached[2])
        with self._lock:
            tool_class = self._tool_class_for_config(tool_name)

        if tool_class:
            config_class = tool_class._get_tool_config_class()
            default_config = config_class()
        else:
            config_class = BaseToolConfig
            default_config = BaseToolConfig()

        tools = self._config.tools
        if tool_name in tools:
            raw_entry = tools[tool_name]
            if not isinstance(raw_entry, Mapping):
                # A PRESENT but malformed tools entry (an explicit null, or
                # any other non-mapping value) fails closed: the tool's
                # effective permission is NEVER on this surface too — never
                # a silent reset to defaults, which would flip a warm
                # "never" entry to "always" (S2 round 2: explicit null tool
                # entry fails open).
                merged_dict = {
                    **default_config.model_dump(),
                    "permission": ToolPermission.NEVER,
                }
            else:
                merged_dict = {**default_config.model_dump(), **raw_entry}
        else:
            # An ABSENT entry means "no per-tool policy": the defaults apply.
            merged_dict = default_config.model_dump()
        config = (
            tool_class.validate_tool_config(merged_dict)
            if tool_class is not None
            else config_class.model_validate(merged_dict)
        )
        parent = self._parent_authority()
        if self._parent_authority_getter is not None:
            try:
                parent_denied = (
                    parent is None
                    or parent.get_tool_config(tool_name).permission
                    == ToolPermission.NEVER
                )
            except Exception:
                parent_denied = True
            if parent_denied:
                config.permission = ToolPermission.NEVER
        sources = self._inherited_restrictions + (
            self._restriction_getter() if self._restriction_getter else ()
        )
        for source in sources:
            for restriction in source.tools:
                if restriction.tool_name != tool_name:
                    continue
                if restriction.denied:
                    config.permission = ToolPermission.NEVER
                config.denylist.extend(
                    pattern
                    for pattern in restriction.denylist
                    if pattern not in config.denylist
                )
                config.sensitive_patterns.extend(
                    pattern
                    for pattern in restriction.sensitive_patterns
                    if pattern not in config.sensitive_patterns
                )
        resolved = config_class.model_validate(config.model_dump())
        if token is not None:
            if self._effective_authority_token() != token:
                raise NoSuchToolError("Tool authority changed during resolution")
            if versionable:
                # A custom config may contain non-JSON extras; those retain the
                # live resolver instead of changing its validation semantics.
                try:
                    frozen = resolved.model_dump_json()
                except Exception:
                    pass
                else:
                    self._authority_cache[tool_name] = (token, config_class, frozen)
        return resolved

    def _invalidate_session_root_grants(self) -> None:
        """Invalidate cached authority after a runtime session grant."""
        self._authority_generation += 1
        self._workspace_cache = None
        self._authority_cache.clear()

    def _invalidate_dangerous_command_authority(self) -> None:
        """Invalidate cached authority after a runtime dangerous-command grant."""
        self._authority_generation += 1
        self._workspace_cache = None
        self._authority_cache.clear()

    def _dangerous_command_authorities(self) -> tuple[DangerousCommandAuthority, ...]:
        """Assemble the immutable dangerous-command authority snapshot (read-only).

        Combines accepted saved grants from restriction sources (the protected
        config read side) with runtime-owned grants from the provider port.
        The assembly is a pure snapshot: no grant mutation, no persistence, no
        execution. A saved grant contributes authority only for the repository
        scope governing its registered project, and only for persistable
        operations; a project whose repository cannot be located contributes
        nothing.
        """
        from chartreux.core.tools.builtins._shell_command_policy import (
            resolve_repository_scope,
        )

        if self._authority_retired:
            raise NoSuchToolError("Tool manager authority retired")
        authorities: list[DangerousCommandAuthority] = []
        sources = self._inherited_restrictions + (
            self._restriction_getter() if self._restriction_getter else ()
        )
        for source in sources:
            for grant in source.dangerous_command_grants:
                scope = resolve_repository_scope(
                    grant.project, GitRepositoryRelocation()
                )
                if scope is None:
                    continue
                operations = frozenset(
                    operation_id
                    for operation_id in grant.operations
                    if (operation := OPERATION_REGISTRY.get(operation_id)) is not None
                    and operation.persistable
                )
                if operations:
                    authorities.append(
                        DangerousCommandAuthority(
                            operations=operations,
                            scope=scope,
                            source="protected_config",
                        )
                    )
        provider = self._dangerous_command_authority_provider
        if provider is not None:
            try:
                authorities.extend(provider.dangerous_command_authorities())
            except Exception:
                # A broken provider fails closed: no runtime authority.
                logger.warning(
                    "Dangerous-command authority provider failed", exc_info=True
                )
        return tuple(authorities)

    @property
    def workspace(self) -> Workspace:
        """Current accepted file authority, independent of effective/discovery data."""
        if self._authority_retired:
            raise NoSuchToolError("Tool manager authority retired")
        token = self._effective_authority_token()
        if token is not None and self._workspace_cache is not None:
            if self._workspace_cache[0] == token:
                return self._workspace_cache[1]
        workspace = Workspace.from_restrictions(
            self._cwd,
            self._inherited_restrictions
            + (self._restriction_getter() if self._restriction_getter else ()),
            ceiling=self._inherited_workspace,
        )
        if self._session_root_grants is not None:
            roots = tuple(getattr(self._session_root_grants, "roots", ()))
            workspace = Workspace(
                workspace.cwd,
                tuple(dict.fromkeys((*workspace.authorized_roots, *roots))),
                workspace.ceiling,
            )
        if token is not None:
            if self._effective_authority_token() != token:
                raise NoSuchToolError("Tool authority changed during resolution")
            self._workspace_cache = (token, workspace)
        return workspace

    def _tool_authority_is_current(
        self, tool_name: str, tool_class: type[BaseTool]
    ) -> bool:
        try:
            if tool_name in {"read_file", "read_image", "write_file", "edit"}:
                # Validate the chain before availability recursively visits parents.
                _ = self.scratchpad_roots
            return (
                not self._authority_retired
                and self._available_tool(tool_name) is tool_class
            )
        except Exception:
            return False

    def _delegated_tool_authority_is_current(
        self,
        tool_name: str,
        tool_class: type[BaseTool],
        selected_class: type[BaseTool] | None = None,
    ) -> bool:
        """Delegation-axis liveness for ancestor policy-resolution instances.

        ``selected_class`` is the exact implementation the CALLER will run;
        the liveness decision classifies THAT class, exactly as the
        availability chain does (S2 round 4: invocation/availability
        implementation consistency). When omitted, this manager's own selected
        variant of the name is judged — the delegation-axis enumeration.
        """
        try:
            if tool_name in {"read_file", "read_image", "write_file", "edit"}:
                # Validate the chain before availability recursively visits parents.
                _ = self.scratchpad_roots
            return (
                not self._authority_retired
                and self._delegated_available_tool(tool_name, selected_class)
                is tool_class
            )
        except Exception:
            return False

    def get_authority_classification(self, tool_name: str) -> ToolClass:
        """Reviewed class of the registered implementation for ``tool_name``.

        Computed MANAGER-SIDE from the selected class's origin: a class
        defined by a shipped builtin module takes its reviewed table entry,
        while anything else (custom replacements, MCP proxies, subclasses)
        fails closed as MUTATING regardless of its name or remote hints. The
        selected class's ``get_authority_classification`` classmethod is
        deliberately NOT consulted for authority — it is overridable, and a
        self-declared class would widen a read-only posture (S2
        classification self-declaration blocker). Unknown names fail closed.
        """
        with self._lock:
            fallback = self._all_tools.get(tool_name)
        if fallback is None:
            return classify_tool(tool_name)
        selected = self._select_registered_variant(tool_name, fallback)
        return registry_provenance_classification(selected)

    def get(self, tool_name: str) -> BaseTool:
        """Get a tool instance, creating it lazily on first call.

        Raises:
            NoSuchToolError: If the requested tool is not available.
        """
        if self._authority_retired:
            raise NoSuchToolError("Tool manager authority retired")
        if tool_name in {"read_file", "read_image", "write_file", "edit"}:
            _ = self.scratchpad_roots
        tool_class = self._available_tool(tool_name)
        if tool_class is None:
            raise NoSuchToolError(f"Unknown or disabled tool: {tool_name}")
        cached = self._instances.get(tool_name)
        if cached is not None and type(cached) is tool_class:
            return cached
        instance = self._build_instance(tool_name, tool_class, delegated=False)
        self._instances[tool_name] = instance
        return instance

    def _delegated_tool(
        self, tool_name: str, selected_class: type[BaseTool] | None = None
    ) -> BaseTool:
        """Delegation-axis instance resolving this manager's policy for a tool.

        Used for ancestor policy checks when the ancestor's personal posture
        excludes the tool: delegation is an independent axis, so the
        ancestor's config (permission, denylist, workspace) still governs
        delegated execution.

        ``selected_class`` is the exact implementation the CALLER will run.
        It keys the LIVENESS of this policy instance — this manager's
        delegation posture classifies THAT class, never a name-keyed
        re-resolution (S2 round 4: invocation/availability implementation
        consistency) — while the instance itself is still built from THIS
        manager's own selected variant under this manager's own config: the
        caller's implementation never resolves the ancestor's policy, so a
        child-forged ``resolve_permission`` cannot weaken the ancestor's
        ceiling.

        The instance cache is keyed by (name, caller implementation): the
        liveness closure bakes ``selected_class``, so the cache never serves
        one caller's instance — whose liveness classifies that caller's
        class — to a different caller (S2 final warning; package 3b
        amendment). Admission (``delegable_tools``) stays name-keyed on this
        manager's own variant; the runtime stays class-exact — see
        ``_parent_permission``.

        Raises:
            NoSuchToolError: If the tool is outside the delegation ceiling.
        """
        if self._authority_retired:
            raise NoSuchToolError("Tool manager authority retired")
        if tool_name in {"read_file", "read_image", "write_file", "edit"}:
            _ = self.scratchpad_roots
        tool_class = self._delegated_available_tool(tool_name, selected_class)
        if tool_class is None:
            raise NoSuchToolError(f"Unknown or disabled tool: {tool_name}")
        cached = self._delegated_instances.get((tool_name, selected_class))
        if cached is not None and type(cached) is tool_class:
            return cached
        instance = self._build_instance(
            tool_name, tool_class, delegated=True, selected_class=selected_class
        )
        self._delegated_instances[(tool_name, selected_class)] = instance
        return instance

    def _build_instance(
        self,
        tool_name: str,
        tool_class: type[BaseTool],
        *,
        delegated: bool,
        selected_class: type[BaseTool] | None = None,
    ) -> BaseTool:
        """Construct one tool instance bound to this manager's live authority.

        ``delegated`` instances resolve the delegation axis (ancestor policy
        checks): their availability, path authority, and shell guard follow
        the delegation ceiling instead of the personal posture.

        Every invocation guard bound here receives the EXECUTION
        implementation — ``tool_class``, the exact class this instance runs —
        so the ancestor chain classifies the implementation the invocation
        will actually execute, exactly as the availability chain classifies
        the child's selected class (S2 round 4: invocation/availability
        implementation consistency). For delegated instances,
        ``selected_class`` is the CALLER's execution implementation: it keys
        the instance's delegation-axis liveness (the ancestor's posture
        classifies THAT class), while the instance itself — the policy
        computation — stays this manager's own variant under this manager's
        own config, so a caller-forged ``resolve_permission`` never weakens
        the ancestor's ceiling.
        """
        instance = tool_class.from_config(
            lambda: self.get_tool_config(tool_name),
            cwd=self._cwd,
            harness_files=self._harness_files,
            scratchpad_dir=self._scratchpad_dir,
        )
        instance.workspace = self.workspace
        instance._workspace_getter = lambda: self.workspace
        instance.instruction_read_files_getter = lambda: self.instruction_read_files
        instance.scratchpad_roots_getter = lambda: self.scratchpad_roots
        if delegated:
            instance.path_authority_getter = lambda: self._delegated_path_authority(
                tool_name, selected_class
            )
        else:
            instance.path_authority_getter = lambda: self.path_authority(tool_name)
        instance.owned_scratchpad_getter = lambda: self._scratchpad_dir
        instance.dangerous_command_authorities_getter = lambda: (
            self._dangerous_command_authorities()
        )
        runtime_config_setter = getattr(instance, "_set_runtime_config_getter", None)
        if runtime_config_setter is not None:
            runtime_config_setter(self._config_getter)
        if self._plan_file_write_scope_getter is not None:
            instance.plan_file_write_scope_getter = self._plan_file_write_scope_getter
        instance.inherited_plan_write_scopes = self._inherited_plan_write_scopes
        if delegated:
            is_current = lambda: self._delegated_tool_authority_is_current(
                tool_name, tool_class, selected_class
            )
            local_permission_guard = (
                (
                    lambda args, local: self._delegated_shell_execution_permission(
                        args, local, tool_class
                    )
                )
                if tool_name == "bash_start"
                else None
            )
        else:
            is_current = lambda: self._tool_authority_is_current(tool_name, tool_class)
            local_permission_guard = (
                (
                    lambda args, local: self._shell_execution_permission(
                        args, local, tool_class
                    )
                )
                if tool_name == "bash_start"
                else None
            )
        instance._bind_authority(
            is_current,
            None
            if tool_name == "bash_start"
            else lambda args: self._parent_permission(tool_name, args, tool_class),
            local_permission_guard=local_permission_guard,
        )
        if tool_name == "grep" and not delegated:
            # Optional per-invocation snapshot hook for the builtin grep tool.
            setattr(instance, "_grep_authority_manager", self)  # noqa: B010 - optional hook
        return instance

    def pop_mcp_errors(self) -> dict[str, str]:
        """Return and clear pending MCP discovery errors (server name -> message)."""
        if self._mcp_registry is None:
            return {}
        return self._mcp_registry.pop_failed()

    @property
    def instruction_read_files(self) -> frozenset[Path]:
        """Per-agent prompt files plus live inherited read capabilities.

        These exact-file grants never enter Workspace.authorized_roots. Ancestor
        denials are checked separately against the requesting agent's manifest.
        """
        files: set[Path] = set()
        seen: set[int] = set()
        cursor: ToolManager | None = self
        while cursor is not None:
            if cursor._authority_retired or id(cursor) in seen:
                raise NoSuchToolError("Instruction read authority unavailable")
            seen.add(id(cursor))
            files.update(cursor._instruction_read_files)
            if cursor._parent_authority_getter is None:
                break
            cursor = cursor._parent_authority()
            if cursor is None:
                raise NoSuchToolError("Parent instruction read authority unavailable")
        return frozenset(files)

    def _instruction_read_token(self) -> tuple[tuple[int, int], ...]:
        """Version runtime manifest publications even in uncached policy chains."""
        tokens: list[tuple[int, int]] = []
        seen: set[int] = set()
        cursor: ToolManager | None = self
        while cursor is not None:
            if cursor._authority_retired or id(cursor) in seen:
                raise NoSuchToolError("Instruction read authority unavailable")
            seen.add(id(cursor))
            tokens.append((id(cursor), cursor._authority_generation))
            if cursor._parent_authority_getter is None:
                break
            cursor = cursor._parent_authority()
            if cursor is None:
                raise NoSuchToolError("Parent instruction read authority unavailable")
        return tuple(tokens)

    def set_instruction_read_files(self, files: frozenset[Path]) -> None:
        """Publish canonical loader identities without re-resolving or persisting."""
        if files != self._instruction_read_files:
            self._instruction_read_files = frozenset(files)
            self._authority_generation += 1

    @property
    def scratchpad_roots(self) -> frozenset[Path]:
        """Live inherited scratch access; never transfers directory ownership.

        Identities are absolute and checked for symlink replacement at invocation.
        Each ancestor still applies its own denial and Plan ceilings.
        """
        roots: set[Path] = set()
        seen: set[int] = set()
        cursor: ToolManager | None = self
        while cursor is not None:
            if cursor._authority_retired or id(cursor) in seen:
                raise NoSuchToolError("Scratchpad authority unavailable")
            seen.add(id(cursor))
            if cursor._scratchpad_dir is not None:
                roots.add(cursor._scratchpad_dir.absolute())
            if cursor._parent_authority_getter is None:
                break
            cursor = cursor._parent_authority()
            if cursor is None:
                raise NoSuchToolError("Parent scratchpad authority unavailable")
        return frozenset(roots)

    def set_scratchpad_dir(self, scratchpad_dir: Path | None) -> None:
        """Replace session scratch authority for both cached and future tools."""
        self._scratchpad_dir = scratchpad_dir
        self._authority_generation += 1
        for instance in self._instances.values():
            instance.scratchpad_dir = scratchpad_dir

    def _retire_authority(self) -> None:
        """Commit-only invalidation of cached AND externally retained instances.

        A prepared replacement must be validated before this irreversible step.
        New authority requires a new manager; reset_all is only cache eviction.
        """
        self._authority_retired = True
        self.reset_all()

    def reset_all(self) -> None:
        self._instances.clear()
        self._delegated_instances.clear()
