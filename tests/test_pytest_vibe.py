from __future__ import annotations

from collections.abc import Generator
import os
from pathlib import Path
import tempfile
from typing import cast

import pytest

from chartreux.core import tmp_ownership as to
import pytest_chartreux


@pytest.fixture(autouse=True)
def _clear_worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.delenv("CI", raising=False)
    for var in pytest_chartreux._CI_RUNNER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    ("cpu_count", "expected_worker_count"),
    [(1, 1), (2, 1), (3, 1), (4, 1), (5, 2), (6, 2), (8, 3)],
)
def test_default_worker_count(cpu_count: int, expected_worker_count: int) -> None:
    assert pytest_chartreux._default_worker_count(cpu_count) == expected_worker_count


def test_local_run_throttles_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pytest_chartreux, "_available_cpu_count", lambda: 8)

    assert pytest_chartreux.pytest_xdist_auto_num_workers() == 3


@pytest.mark.parametrize("runner_var", pytest_chartreux._CI_RUNNER_ENV_VARS)
@pytest.mark.parametrize("runner_value", ["1", "true", "TRUE", "yes"])
def test_ci_run_uses_all_available_cpus(
    monkeypatch: pytest.MonkeyPatch, runner_var: str, runner_value: str
) -> None:
    monkeypatch.setenv(runner_var, runner_value)
    monkeypatch.setattr(pytest_chartreux, "_available_cpu_count", lambda: 8)

    assert pytest_chartreux.pytest_xdist_auto_num_workers() == 8


def test_generic_ci_var_does_not_unthrottle(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``CI`` leaks into ordinary developer shells; it must not trigger the
    # all-cores path, or local runs would saturate the machine.
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(pytest_chartreux, "_available_cpu_count", lambda: 8)

    assert pytest_chartreux.pytest_xdist_auto_num_workers() == 3


def test_worker_count_uses_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "4")

    worker_count = pytest_chartreux.pytest_xdist_auto_num_workers()

    assert worker_count == 4


def test_environment_override_wins_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUILDKITE", "true")
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "2")
    monkeypatch.setattr(pytest_chartreux, "_available_cpu_count", lambda: 8)

    assert pytest_chartreux.pytest_xdist_auto_num_workers() == 2


@pytest.mark.parametrize("configured_worker_count", ["0", "-1", "invalid"])
def test_worker_count_ignores_invalid_environment_override(
    monkeypatch: pytest.MonkeyPatch, configured_worker_count: str
) -> None:
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", configured_worker_count)
    monkeypatch.setattr(pytest_chartreux, "_available_cpu_count", lambda: 8)

    with pytest.warns(UserWarning, match="must be a positive integer"):
        worker_count = pytest_chartreux.pytest_xdist_auto_num_workers()

    assert worker_count == 3


class _StubOption:
    def __init__(self, basetemp: str | None) -> None:
        self.basetemp = basetemp


class _StubFactory:
    def __init__(self) -> None:
        self._basetemp: Path | None = None
        self._given_basetemp: Path | None = None


class _StubConfig:
    def __init__(self, basetemp: str | None, factory: _StubFactory | None) -> None:
        self.option = _StubOption(basetemp)
        if factory is not None:
            self._tmp_path_factory = factory
        self.cleanups: list = []

    def add_cleanup(self, fn) -> None:  # type: ignore[no-untyped-def]
        self.cleanups.append(fn)


@pytest.fixture
def isolated_temproot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path, None, None]:
    """Point tempfile.gettempdir() at an isolated root (TMPDIR + cache reset)."""
    root = tmp_path / "temproot"
    root.mkdir()
    monkeypatch.setenv("TMPDIR", str(root))
    monkeypatch.setattr(tempfile, "tempdir", None)
    yield root
    to.set_ambient_binding(None)


def test_configure_hook_is_trylast_pinned() -> None:
    # The pinned pytest 9.1.1 ordering fact: a -p plugin's plain
    # pytest_configure runs BEFORE config._tmp_path_factory exists, so the
    # managed-root preset must run in a trylast configure hook.
    impl = getattr(pytest_chartreux.pytest_configure, "pytest_impl", None)
    assert impl is not None
    assert impl["trylast"] is True


def test_configure_presets_managed_run_root(isolated_temproot: Path) -> None:
    config = _StubConfig(basetemp=None, factory=_StubFactory())
    factory = config._tmp_path_factory
    pytest_chartreux.pytest_configure(cast(pytest.Config, config))
    try:
        basetemp = factory._basetemp
        assert basetemp is not None
        parent = to.managed_parent(isolated_temproot)
        assert basetemp.parent == parent
        assert basetemp.name.startswith("pytest-run-")
        assert basetemp.is_dir()
        assert factory._given_basetemp is None
        alloc = to.allocation_from_path(basetemp)
        record = to.load_ownership_record(parent, alloc.name)
        assert record is not None
        assert record.kind == to.KIND_PYTEST_RUN_ROOT
        # The controller registered itself in the run-root ledger.
        binding = to.ambient_binding()
        assert binding is not None
        assert binding.allocation.path == basetemp
        entries = to.Ledger(alloc).entries()
        assert any(
            e.get("kind") == "completion" and e.get("pid") == os.getpid()
            for e in entries
        )
    finally:
        for cleanup in config.cleanups:
            cleanup()


def test_configure_explicit_basetemp_is_never_preset(
    isolated_temproot: Path, tmp_path: Path
) -> None:
    explicit = tmp_path / "explicit-base"
    config = _StubConfig(basetemp=str(explicit), factory=_StubFactory())
    pytest_chartreux.pytest_configure(cast(pytest.Config, config))
    try:
        assert config._tmp_path_factory._basetemp is None
        parent = to.managed_parent(isolated_temproot)
        assert not list(parent.glob("pytest-run-*"))
        assert to.ambient_binding() is None
    finally:
        for cleanup in config.cleanups:
            cleanup()


def test_configure_without_tmp_path_factory_is_noop(isolated_temproot: Path) -> None:
    config = _StubConfig(basetemp=None, factory=None)
    pytest_chartreux.pytest_configure(cast(pytest.Config, config))
    parent = to.managed_parent(isolated_temproot)
    assert not list(parent.glob("pytest-run-*"))
