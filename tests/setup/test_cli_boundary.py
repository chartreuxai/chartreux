from __future__ import annotations

import ast
from pathlib import Path

from chartreux.utils.io import read_safe

SETUP_ROOT = Path(__file__).parents[2] / "chartreux" / "setup"


def _imports(source_path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(read_safe(source_path).text, filename=source_path)
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                imports.extend((node.lineno, alias.name) for alias in names)
            case ast.ImportFrom(module=module) if module is not None:
                imports.append((node.lineno, module))
    return imports


def test_setup_does_not_import_cli() -> None:
    violations = [
        f"{source_path.relative_to(SETUP_ROOT)}:{line}: {module}"
        for source_path in sorted(SETUP_ROOT.rglob("*.py"))
        for line, module in _imports(source_path)
        if module == "chartreux.cli" or module.startswith("chartreux.cli.")
    ]

    assert not violations, "\n".join(violations)
