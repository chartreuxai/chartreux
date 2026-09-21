from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RETIRED_RUNTIME_ROOTS = {"colorama", "pywin32", "pywin32-ctypes", "pywinpty"}
_RETIRED_BUILD_ROOTS = {
    "altgraph",
    "macholib",
    "pefile",
    "pyinstaller",
    "pyinstaller-hooks-contrib",
    "pywin32-ctypes",
    "setuptools",
}
_RETIRED_DEV_ROOTS = {"pyinstrument"}
_RETIRED_DISTRIBUTION_PATHS = (
    ".github/workflows/build-and-upload.yml",
    ".github/workflows/release.yml",
    "action.yml",
    "chartreux/cli/profiler.py",
    "chartreux/core/prompts/cli_2026-07_v2.md",
    "chartreux/core/prompts/cli_2026-08_v3.md",
    "chartreux.spec",
    "chartreux-acp.spec",
    "chartreux-app-server.spec",
    "flake.nix",
    "flake.lock",
    "pyinstaller/runtime_hook_truststore.py",
    "scripts/ci/build-pyinstaller-binaries.sh",
    "scripts/ci/clear-linux-execstack.sh",
    "scripts/ci/package-pyinstaller-artifacts.sh",
    "scripts/ci/setup-linux-pyinstaller-build.sh",
    "scripts/ci/smoke-pyinstaller-cli.sh",
    "scripts/install.sh",
    "tests/cli/smoke_binary.py",
    "tests/acp/smoke_binary.py",
    "tests/app_server/smoke_binary.py",
    "tests/test_install_script.py",
)


def _dependency_name(requirement: str) -> str:
    name = re.split(r"[<>=!~;\[\s]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def test_retired_distribution_paths_are_absent_from_source_tree() -> None:
    for relative in _RETIRED_DISTRIBUTION_PATHS:
        path = _PROJECT_ROOT / relative
        assert not path.exists() and not path.is_symlink(), relative


def test_dependency_declarations_are_locked_without_installing(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for filename in ("pyproject.toml", "uv.lock"):
        shutil.copy2(_PROJECT_ROOT / filename, project / filename)

    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    project_config = metadata["project"]
    runtime_roots = {
        _dependency_name(requirement) for requirement in project_config["dependencies"]
    }
    groups = metadata["dependency-groups"]
    assert "build" not in groups
    group_roots = {
        _dependency_name(requirement)
        for requirements in groups.values()
        for requirement in requirements
        if isinstance(requirement, str)
    }
    assert runtime_roots.isdisjoint(_RETIRED_RUNTIME_ROOTS)
    assert (runtime_roots | group_roots).isdisjoint(_RETIRED_BUILD_ROOTS)
    assert (runtime_roots | group_roots).isdisjoint(_RETIRED_DEV_ROOTS)
    assert "debugpy" not in group_roots
    assert "pexpect" not in runtime_roots
    assert "pexpect==4.9.0" in groups["dev"]
    assert "packaging" not in runtime_roots
    assert "packaging==26.2" in groups["dev"]
    assert project_config["scripts"] == {
        "chartreux": "chartreux.cli.entrypoint:main",
        "chartreux-acp": "chartreux.acp.entrypoint:main",
        "chartreux-app-server": "chartreux.app_server.stdio:main",
    }

    uv = shutil.which("uv") or "/home/pav/.local/bin/uv"
    check_env = dict(os.environ)
    for name in ("UV_CONFIG_FILE", "UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_FIND_LINKS"):
        check_env.pop(name, None)
    lock_before = (project / "uv.lock").read_bytes()
    lock_check = subprocess.run(
        [
            uv,
            "lock",
            "--check",
            "--offline",
            "--default-index",
            "https://pypi.org/simple",
        ],
        cwd=project,
        env=check_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "lock-check.log").write_text(lock_check.stdout + lock_check.stderr)
    assert lock_check.returncode == 0, lock_check.stdout + lock_check.stderr
    assert (project / "uv.lock").read_bytes() == lock_before
