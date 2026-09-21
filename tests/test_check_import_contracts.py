from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import textwrap

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cic = _load_script("_check_import_contracts_under_test", "check_import_contracts.py")


def _write(tree: Path, rel: str, source: str) -> Path:
    p = tree / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(source))
    return p


def test_is_chartreux_module():
    assert cic._is_chartreux_module("chartreux")
    assert cic._is_chartreux_module("chartreux.core")
    assert cic._is_chartreux_module("chartreux.core.tools")
    assert not cic._is_chartreux_module("collections")
    assert not cic._is_chartreux_module("mistralai")


def test_missing_dep_name_for_non_chartreux_module():
    exc = ModuleNotFoundError("No module named 'foo'", name="foo")
    assert cic._missing_dep_name(exc) == "foo"


def test_missing_dep_name_for_chartreux_module():
    exc = ModuleNotFoundError(
        "No module named 'chartreux.missing'", name="chartreux.missing"
    )
    assert cic._missing_dep_name(exc) is None


def test_missing_dep_name_for_non_module_not_found():
    exc = ImportError("boom")
    assert cic._missing_dep_name(exc) is None


def test_resolve_relative_regular_module(tmp_path: Path):
    vibe_pkg = tmp_path / "chartreux"
    f = vibe_pkg / "core" / "tools" / "base.py"
    f.parent.mkdir(parents=True)
    f.touch()
    # `from . import utils` in core/tools/base.py -> chartreux.core.tools.utils
    assert (
        cic._resolve_relative(f, vibe_pkg, level=1, name="utils")
        == "chartreux.core.tools.utils"
    )
    # `from .. import core` in core/tools/base.py -> chartreux.core (drops one level)
    assert cic._resolve_relative(f, vibe_pkg, level=2, name=None) == "chartreux.core"


def test_resolve_relative_init_file(tmp_path: Path):
    vibe_pkg = tmp_path / "chartreux"
    f = vibe_pkg / "core" / "__init__.py"
    f.parent.mkdir(parents=True)
    f.touch()
    # `from . import tools` in core/__init__.py -> chartreux.core.tools
    assert (
        cic._resolve_relative(f, vibe_pkg, level=1, name="tools")
        == "chartreux.core.tools"
    )
    # `from .sub import x` in core/__init__.py -> chartreux.core.sub.x
    assert (
        cic._resolve_relative(f, vibe_pkg, level=1, name="sub.x")
        == "chartreux.core.sub.x"
    )


def test_resolve_relative_too_deep(tmp_path: Path):
    vibe_pkg = tmp_path / "chartreux"
    f = vibe_pkg / "core" / "tools.py"
    f.parent.mkdir(parents=True)
    f.touch()
    assert cic._resolve_relative(f, vibe_pkg, level=5, name=None) is None


def test_import_visitor_collects_from_import(tmp_path: Path):
    vibe_pkg = tmp_path / "chartreux"
    f = _write(vibe_pkg, "mod.py", "from collections import OrderedDict\n")
    tree = cic.ast.parse(f.read_text(), filename=str(f))
    visitor = cic.ImportVisitor(f, vibe_pkg)
    visitor.visit(tree)
    assert ("collections", "OrderedDict", 1) in visitor.imports


def test_import_visitor_skips_star_import_name(tmp_path: Path):
    vibe_pkg = tmp_path / "chartreux"
    f = _write(vibe_pkg, "mod.py", "from os import *\n")
    tree = cic.ast.parse(f.read_text(), filename=str(f))
    visitor = cic.ImportVisitor(f, vibe_pkg)
    visitor.visit(tree)
    assert ("os", "*", 1) in visitor.imports


def test_check_contracts_flags_missing_attr(tmp_path: Path, monkeypatch):
    # Inject a real module object so importlib.import_module succeeds but the
    # requested name is not an attribute, and not a submodule.
    import types

    fake_mod = types.ModuleType("chartreux.bad")
    monkeypatch.setitem(sys.modules, "chartreux.bad", fake_mod)
    monkeypatch.setitem(sys.modules, "chartreux", types.ModuleType("chartreux"))
    pairs: dict[tuple[str, str], set[Path]] = {
        ("chartreux.bad", "DoesNotExist"): {tmp_path / "bad.py"}
    }
    failures, by_sig, env_warnings, checked = cic._check_contracts(pairs, {})
    assert any("DoesNotExist" in f for f in failures)
    assert not by_sig
    assert not env_warnings
    assert checked == 1


def test_check_contracts_tolerates_missing_non_chartreux_dep(
    tmp_path: Path, monkeypatch
):
    # Simulate a chartreux module whose import fails because of a missing non-chartreux
    # transitive dependency (e.g. a native extension not installed in this env).
    import types

    def _boom(name):
        raise ModuleNotFoundError(
            "No module named 'nonexistent_pkg_xyz'", name="nonexistent_pkg_xyz"
        )

    monkeypatch.setitem(sys.modules, "chartreux", types.ModuleType("chartreux"))
    monkeypatch.setattr(cic.importlib, "import_module", _boom)
    pairs: dict[tuple[str, str], set[Path]] = {
        ("chartreux.missing_dep", "thing"): {tmp_path / "missing_dep.py"}
    }
    failures, by_sig, env_warnings, checked = cic._check_contracts(pairs, {})
    assert not failures
    assert not by_sig
    assert any("nonexistent_pkg_xyz" in w for w in env_warnings)
    assert checked == 1


def test_check_contracts_star_module_missing_non_chartreux_dep(
    tmp_path: Path, monkeypatch
):
    def _boom(name):
        raise ModuleNotFoundError(
            "No module named 'nonexistent_pkg_xyz'", name="nonexistent_pkg_xyz"
        )

    monkeypatch.setattr(cic.importlib, "import_module", _boom)
    star: dict[str, set[Path]] = {"chartreux.star": {tmp_path / "star.py"}}
    by_sig, env_warnings, checked = cic._check_star_modules(star)
    assert not by_sig
    assert any("nonexistent_pkg_xyz" in w for w in env_warnings)
    assert checked == 1


def test_check_all_passes_on_clean_tree(tmp_path: Path, monkeypatch):
    repo = tmp_path
    vibe_pkg = repo / "chartreux"
    _write(vibe_pkg, "__init__.py", "")
    _write(
        vibe_pkg,
        "ok.py",
        """
        from collections import OrderedDict
        x = OrderedDict()
        """,
    )
    _write(repo, "tests/__init__.py", "")
    monkeypatch.setattr(sys, "argv", ["check_import_contracts.py", str(repo)])
    rc = cic.check_all()
    assert rc == 0


def test_check_all_flags_cross_file_type_checking_reexport(tmp_path: Path, monkeypatch):
    repo = tmp_path
    vibe_pkg = repo / "chartreux"
    _write(vibe_pkg, "__init__.py", "")
    _write(
        vibe_pkg,
        "bad_export.py",
        """
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from collections import OrderedDict
        """,
    )
    _write(
        vibe_pkg,
        "consumer.py",
        """
        from __future__ import annotations
        from chartreux.bad_export import OrderedDict
        x: OrderedDict = None
        """,
    )
    _write(repo, "tests/__init__.py", "")
    monkeypatch.setattr(sys, "argv", ["check_import_contracts.py", str(repo)])
    rc = cic.check_all()
    assert rc == 1
