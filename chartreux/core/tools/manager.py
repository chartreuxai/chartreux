from __future__ import annotations

from collections.abc import Callable, Iterator
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sys
import threading
from typing import TYPE_CHECKING, Any, TypeGuard

from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    get_harness_files_manager,
)
from chartreux.core.llm_models import AvailableFunction
from chartreux.core.paths import DEFAULT_TOOL_DIR
from chartreux.core.tools.base import BaseTool, BaseToolConfig, ToolPermission
from chartreux.core.tools.remote import MCPTool
from chartreux.core.utils import name_matches, run_sync
from chartreux.core.workspace import Workspace
from chartreux.observability.logging import logger
from chartreux.utils.io import read_safe

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema
    from chartreux.core.config._restrictions import SourceRestrictions
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


class ToolManager:
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
    ) -> None:
        self._config_getter = config_getter
        self._cwd = (cwd or Path.cwd()).resolve()
        self._harness_files = harness_files or get_harness_files_manager()
        self._scratchpad_dir = scratchpad_dir
        self._plan_file_write_scope_getter = plan_file_write_scope_getter
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
        self._registry_generation = 0
        self._authority_cache: dict[
            str, tuple[tuple[object, ...], type[BaseToolConfig], str]
        ] = {}
        self._selection_cache: dict[
            str, tuple[tuple[object, ...], type[BaseTool] | None]
        ] = {}
        self._workspace_cache: tuple[tuple[object, ...], Workspace] | None = None
        self._mcp_registry = mcp_registry
        self._instances: dict[str, BaseTool] = {}
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
            for tool_class, is_custom in self._iter_tool_classes_with_origin(
                self._search_paths
            ):
                self._register_discovered_tool_variant(tool_class, is_custom=is_custom)
            self._tool_descriptions: dict[str, str] = dict(
                self._iter_tool_descriptions(self._search_paths)
            )
        if not defer_mcp:
            self.integrate_all()

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
        for tool_class, _ in ToolManager._iter_tool_classes_with_origin(search_paths):
            yield tool_class

    @staticmethod
    def _iter_tool_classes_with_origin(
        search_paths: list[Path],
    ) -> Iterator[tuple[type[BaseTool], bool]]:
        builtin_dir = DEFAULT_TOOL_DIR.path.resolve()
        for base in search_paths:
            if not base.is_dir() and base.name.endswith(".py"):
                if tools := ToolManager._load_tools_from_file(base):
                    for tool in tools:
                        yield tool, not base.resolve().is_relative_to(builtin_dir)

            for path in base.glob("*.py"):
                if tools := ToolManager._load_tools_from_file(path):
                    for tool in tools:
                        yield tool, not path.resolve().is_relative_to(builtin_dir)

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

    def _register_discovered_tool_variant(
        self, tool_class: type[BaseTool], *, is_custom: bool
    ) -> None:
        name = tool_class.get_name()
        self._tool_variants_by_name.setdefault(name, []).append(tool_class)
        self._custom_tool_variants_by_name.setdefault(name, []).append(is_custom)
        self._all_tools[name] = tool_class
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

    @property
    def authority_revision(self) -> int | None:
        if self._parent_authority_revision_getter is None:
            return None
        try:
            return self._parent_authority_revision_getter()
        except Exception:
            return None

    def _effective_authority_token(  # noqa: PLR0911 - each missing signal fails closed
        self, seen: set[int] | None = None
    ) -> tuple[object, ...] | None:
        """A revision for accepted inputs along the *entire* authority chain.

        An unavailable parent or revision signal disables caching; never reuse an
        entry obtained while the chain was healthy.
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
        with self._lock:
            variants = tuple(self._tool_variants_by_name.get(name, ()))
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

    def _available_tool(self, name: str) -> type[BaseTool] | None:
        """Resolve only one name; custom availability callbacks remain live."""
        if self._authority_retired:
            return None
        token = self._effective_authority_token()
        with self._lock:
            fallback = self._all_tools.get(name)
        versionable = self._name_versionable(name)
        if token is not None and versionable:
            cached = self._selection_cache.get(name)
            if cached is not None and cached[0] == token:
                return cached[1]
        with self._lock:
            selected = (
                self._select_available_variant(name, fallback)
                if fallback is not None
                else None
            )
        if selected is not None:
            config = self._config
            disabled, per_source = self._build_source_disable_index(config)
            if (
                self._is_source_disabled(selected, disabled, per_source)  # noqa: PLR0916
                or (
                    config.enabled_tools
                    and not name_matches(name, config.enabled_tools)
                )
                or (config.disabled_tools and name_matches(name, config.disabled_tools))
                or not self._parent_allows_tool(name)
            ):
                selected = None
        if token is not None:
            if self._effective_authority_token() != token:
                return None
            if versionable:
                self._selection_cache[name] = (token, selected)
        return selected

    def _parent_allows_tool(self, tool_name: str) -> bool:
        parent = self._parent_authority()
        return (
            parent is None
            if self._parent_authority_getter is None
            else (parent is not None and parent._available_tool(tool_name) is not None)
        )

    def _parent_permission(self, tool_name: str, args: Any) -> Any:
        """Return a parent invocation denial, including tool-specific guards."""
        from chartreux.core.tools.permissions import PermissionContext

        if self._parent_authority_getter is None:
            return None
        parent = self._parent_authority()
        if parent is None or parent._available_tool(tool_name) is None:
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="Tool is unavailable under parent authority",
            )
        try:
            parent_tool = parent.get(tool_name)
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
                return parent_context
        except Exception:
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="Parent tool authority is unavailable",
            )
        return None

    @property
    def available_tools(self) -> dict[str, type[BaseTool]]:
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
        return {
            name: cls for name, cls in result.items() if self._parent_allows_tool(name)
        }

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

    def get_tool_config(  # noqa: PLR0912 - restriction sources and cache guards
        self, tool_name: str
    ) -> BaseToolConfig:
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

        user_overrides = self._config.tools.get(tool_name)
        merged_dict = {**default_config.model_dump(), **(user_overrides or {})}
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
        if token is not None:
            if self._effective_authority_token() != token:
                raise NoSuchToolError("Tool authority changed during resolution")
            self._workspace_cache = (token, workspace)
        return workspace

    def get(self, tool_name: str) -> BaseTool:
        """Get a tool instance, creating it lazily on first call.

        Raises:
            NoSuchToolError: If the requested tool is not available.
        """
        if self._authority_retired:
            raise NoSuchToolError("Tool manager authority retired")
        tool_class = self._available_tool(tool_name)
        if tool_class is None:
            raise NoSuchToolError(f"Unknown or disabled tool: {tool_name}")
        cached = self._instances.get(tool_name)
        if cached is not None and type(cached) is tool_class:
            return cached
        instance = tool_class.from_config(
            lambda: self.get_tool_config(tool_name),
            cwd=self._cwd,
            harness_files=self._harness_files,
            scratchpad_dir=self._scratchpad_dir,
        )
        instance.workspace = self.workspace
        instance._workspace_getter = lambda: self.workspace
        runtime_config_setter = getattr(instance, "_set_runtime_config_getter", None)
        if runtime_config_setter is not None:
            runtime_config_setter(self._config_getter)
        if self._plan_file_write_scope_getter is not None:
            instance.plan_file_write_scope_getter = self._plan_file_write_scope_getter
        instance.inherited_plan_write_scopes = self._inherited_plan_write_scopes
        instance._bind_authority(
            lambda: (
                not self._authority_retired
                and self._available_tool(tool_name) is tool_class
            ),
            lambda args: self._parent_permission(tool_name, args),
        )
        if tool_name == "grep":
            # Optional per-invocation snapshot hook for the builtin grep tool.
            setattr(instance, "_grep_authority_manager", self)  # noqa: B010 - optional hook
        self._instances[tool_name] = instance
        return instance

    def pop_mcp_errors(self) -> dict[str, str]:
        """Return and clear pending MCP discovery errors (server name -> message)."""
        if self._mcp_registry is None:
            return {}
        return self._mcp_registry.pop_failed()

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
