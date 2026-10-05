from __future__ import annotations

import ast
from collections import deque
from collections.abc import Iterator
from importlib.util import resolve_name
from pathlib import Path

import pytest

from chartreux.app_server.protocol import SERVER_METHODS
from chartreux.utils.io import read_safe

TEXTUAL_UI_ROOT = Path(__file__).parents[3] / "chartreux" / "cli" / "textual_ui"
CHARTREUX_ROOT = TEXTUAL_UI_ROOT.parents[1]
CORE_ROOT = CHARTREUX_ROOT / "core"
PROGRAMMATIC_PATH = CHARTREUX_ROOT / "cli" / "programmatic.py"
ACP_RUNTIME_PATHS = (
    CHARTREUX_ROOT / "acp" / "agent.py",
    CHARTREUX_ROOT / "acp" / "content.py",
    CHARTREUX_ROOT / "acp" / "image_blocks.py",
    CHARTREUX_ROOT / "acp" / "session.py",
    CHARTREUX_ROOT / "acp" / "session_updates.py",
    CHARTREUX_ROOT / "acp" / "tool_io.py",
    CHARTREUX_ROOT / "acp" / "user_display_content.py",
    CHARTREUX_ROOT / "acp" / "utils.py",
    *sorted((CHARTREUX_ROOT / "acp" / "commands").rglob("*.py")),
)
PUBLIC_APP_SERVER_PATHS = tuple(
    CHARTREUX_ROOT / "app_server" / name
    for name in (
        "__init__.py",
        "_connection_protocol.py",
        "_effect_models.py",
        "_model.py",
        "client.py",
        "client_state.py",
        "client_tools.py",
        "config.py",
        "connection.py",
        "events.py",
        "host.py",
        "models.py",
        "protocol.py",
        "resources.py",
        "session.py",
        "transport.py",
    )
)
SERVER_ONLY_APP_SERVER_MODULES = (
    "chartreux.app_server._",
    "chartreux.app_server.client",
    "chartreux.app_server.client_state",
    "chartreux.app_server.connection",
    "chartreux.app_server.local",
    "chartreux.app_server.server",
    "chartreux.app_server.stdio",
)
# protocol is a public app_server module, so its core dependency is legal.
ALLOWED_CORE_IMPORTS = {
    "chartreux.app_server.protocol": {"chartreux.core.config.settings_catalog"},
    "chartreux.core.config.settings_catalog": {
        "chartreux.core.config.chartreux_schema",
        "chartreux.core.config.models",
    },
}
# Schema/model implementations are leaf boundaries, not an exempt runtime closure.
SCHEMA_LEAVES = ALLOWED_CORE_IMPORTS["chartreux.core.config.settings_catalog"]


def _production_files() -> list[Path]:
    return sorted(TEXTUAL_UI_ROOT.rglob("*.py"))


def _module_name(source_path: Path) -> str:
    parts = list(source_path.relative_to(CHARTREUX_ROOT.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _production_modules() -> dict[str, Path]:
    return {_module_name(path): path for path in CHARTREUX_ROOT.rglob("*.py")}


def _imports(source_path: Path) -> Iterator[tuple[int, str]]:
    tree = ast.parse(read_safe(source_path).text, filename=source_path)
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                modules = [alias.name for alias in names]
            case ast.ImportFrom(module=module) if module is not None:
                modules = [module]
            case _:
                continue

        for module in modules:
            yield node.lineno, module


def test_textual_imports_only_public_app_server_modules() -> None:
    violations = [
        f"{source_path.relative_to(TEXTUAL_UI_ROOT)}:{line}: {module}"
        for source_path in _production_files()
        for line, module in _imports(source_path)
        if module == "chartreux.core"
        or module.startswith("chartreux.core.")
        or module.startswith(SERVER_ONLY_APP_SERVER_MODULES)
    ]

    assert not violations, "\n".join(violations)


def test_textual_has_no_transitive_core_dependency() -> None:
    modules = _production_modules()
    imports = {
        module: {name for _, name in _imports(path)} for module, path in modules.items()
    }
    starts = {
        module
        for module, path in modules.items()
        if path.is_relative_to(TEXTUAL_UI_ROOT)
    }
    pending = deque((module, [module]) for module in starts)
    visited = set(starts)
    violations: list[str] = []

    while pending:
        module, chain = pending.popleft()
        for imported in imports[module]:
            if (
                imported == "chartreux.core" or imported.startswith("chartreux.core.")
            ) and imported not in ALLOWED_CORE_IMPORTS.get(module, set()):
                violations.append(" -> ".join([*chain, imported]))
                continue
            resolved = imported
            while resolved and resolved not in modules:
                resolved = resolved.rpartition(".")[0]
            if not resolved or resolved in visited or resolved in SCHEMA_LEAVES:
                continue
            visited.add(resolved)
            pending.append((resolved, [*chain, resolved]))

    assert not violations, "\n".join(sorted(violations))


def test_boundary_rejects_runtime_import_from_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_imports = _imports
    catalog = CHARTREUX_ROOT / "core" / "config" / "settings_catalog.py"

    def injected_imports(source_path: Path) -> Iterator[tuple[int, str]]:
        yield from original_imports(source_path)
        if source_path == catalog:
            yield 1, "chartreux.core.tools.manager"

    monkeypatch.setattr(__name__ + "._imports", injected_imports)
    with pytest.raises(AssertionError, match="chartreux.core.tools.manager"):
        test_textual_has_no_transitive_core_dependency()


def test_textual_does_not_reference_agent_loop() -> None:
    violations = [
        f"{source_path.relative_to(TEXTUAL_UI_ROOT)}:{node.lineno}"
        for source_path in _production_files()
        for node in ast.walk(
            ast.parse(read_safe(source_path).text, filename=source_path)
        )
        if (isinstance(node, ast.Name) and node.id == "agent_loop")
        or (isinstance(node, ast.Attribute) and node.attr == "agent_loop")
        or (isinstance(node, ast.arg) and node.arg == "agent_loop")
    ]

    assert not violations, "\n".join(violations)


def test_textual_does_not_install_callback_setters_or_observers() -> None:
    violations = [
        f"{source_path.relative_to(TEXTUAL_UI_ROOT)}:{node.lineno}: {node.attr}"
        for source_path in _production_files()
        for node in ast.walk(
            ast.parse(read_safe(source_path).text, filename=source_path)
        )
        if isinstance(node, ast.Attribute)
        and (
            (node.attr.startswith("set_") and node.attr.endswith("_callback"))
            or "observer" in node.attr
        )
    ]

    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    "source_path", [PROGRAMMATIC_PATH, *ACP_RUNTIME_PATHS, *PUBLIC_APP_SERVER_PATHS]
)
def test_app_server_clients_do_not_import_core(source_path: Path) -> None:
    violations = [
        f"{source_path.relative_to(CHARTREUX_ROOT)}:{line}: {module}"
        for line, module in _imports(source_path)
        if (module == "chartreux.core" or module.startswith("chartreux.core."))
        and module not in ALLOWED_CORE_IMPORTS.get(_module_name(source_path), set())
    ]

    assert not violations, "\n".join(violations)


def test_core_does_not_import_app_server_or_textual() -> None:
    violations = [
        f"{source_path.relative_to(CORE_ROOT)}:{line}: {module}"
        for source_path in CORE_ROOT.rglob("*.py")
        for line, module in _imports(source_path)
        if module == "chartreux.app_server"
        or module.startswith("chartreux.app_server.")
        or module == "textual"
        or module.startswith("textual.")
    ]

    assert not violations, "\n".join(sorted(violations))


def _reverse_boundary_imports(source_path: Path) -> set[str]:
    names = {name for _, name in _imports(source_path)}
    module = _module_name(source_path)
    package = module if source_path.name == "__init__.py" else module.rpartition(".")[0]
    tree = ast.parse(read_safe(source_path).text, filename=source_path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        base = node.module or ""
        if node.level:
            base = resolve_name("." * node.level + base, package)
        names.add(base)
        names.update(f"{base}.{alias.name}" for alias in node.names)
    # Importing a submodule also executes its package initializers.
    names.update(
        parent
        for name in list(names)
        for parent in (
            ".".join(name.split(".")[:index])
            for index in range(1, len(name.split(".")))
        )
    )
    return names


def test_core_and_app_server_have_no_transitive_cli_or_textual_dependency() -> None:
    modules = _production_modules()
    imports = {
        module: _reverse_boundary_imports(path) for module, path in modules.items()
    }
    starts = {
        module
        for module, path in modules.items()
        if path.is_relative_to(CORE_ROOT)
        or path.is_relative_to(CHARTREUX_ROOT / "app_server")
    }
    pending = deque((module, [module]) for module in starts)
    visited = set(starts)
    violations: list[str] = []
    forbidden = ("textual", "chartreux.cli")
    while pending:
        module, chain = pending.popleft()
        for imported in imports[module]:
            if any(
                imported == prefix or imported.startswith(prefix + ".")
                for prefix in forbidden
            ):
                violations.append(" -> ".join([*chain, imported]))
                continue
            resolved = imported
            while resolved and resolved not in modules:
                resolved = resolved.rpartition(".")[0]
            if not resolved or resolved in visited:
                continue
            visited.add(resolved)
            pending.append((resolved, [*chain, resolved]))
    assert not violations, "\n".join(sorted(violations))


@pytest.mark.parametrize("forbidden", ["textual.widgets", "chartreux.cli.textual_ui"])
def test_reverse_boundary_rejects_transitive_ui_import(
    monkeypatch: pytest.MonkeyPatch, forbidden: str
) -> None:
    original_imports = _imports
    utils = CHARTREUX_ROOT / "utils" / "io.py"

    def injected_imports(source_path: Path) -> Iterator[tuple[int, str]]:
        yield from original_imports(source_path)
        if source_path == utils:
            yield 1, forbidden

    monkeypatch.setattr(__name__ + "._imports", injected_imports)
    with pytest.raises(AssertionError, match=forbidden):
        test_core_and_app_server_have_no_transitive_cli_or_textual_dependency()


def test_only_app_server_runtime_constructs_agent_loop() -> None:
    allowed = CHARTREUX_ROOT / "app_server" / "_runtime.py"
    violations = [
        f"{source_path.relative_to(CHARTREUX_ROOT)}:{node.lineno}"
        for source_path in CHARTREUX_ROOT.rglob("*.py")
        if source_path != allowed
        for node in ast.walk(
            ast.parse(read_safe(source_path).text, filename=source_path)
        )
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AgentLoop"
    ]

    assert not violations, "\n".join(violations)


def test_protocol_has_no_generic_command_escape_hatch() -> None:
    assert not [method for method in SERVER_METHODS if method.endswith("/command")]


def test_agent_stop_clients_do_not_access_registry() -> None:
    forbidden = {
        "SubagentRegistry",
        "_registry",
        "_subagent_registry",
        "_registry_lock",
        "cancel_run",
    }
    violations = [
        f"{path.relative_to(CHARTREUX_ROOT)}:{node.lineno}"
        for path in [*_production_files(), *PUBLIC_APP_SERVER_PATHS]
        for node in ast.walk(ast.parse(read_safe(path).text, filename=path))
        if (isinstance(node, ast.Name) and node.id in forbidden)
        or (isinstance(node, ast.Attribute) and node.attr in forbidden)
    ]
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    "relative_path,symbols",
    [
        ("cli/textual_ui/app.py", {"CancelOutcome"}),
        (
            "cli/textual_ui/widgets/agent_bar.py",
            {"AgentsCancelResponse", "CancelOutcome"},
        ),
        ("app_server/session.py", {"AgentsCancelParams", "AgentsCancelResponse"}),
    ],
)
def test_agent_stop_uses_public_protocol_models(
    relative_path: str, symbols: set[str]
) -> None:
    tree = ast.parse(read_safe(CHARTREUX_ROOT / relative_path).text)
    imported = {
        alias.asname or alias.name: (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for symbol in symbols:
        assert imported[symbol] == ("chartreux.app_server.protocol", symbol)
