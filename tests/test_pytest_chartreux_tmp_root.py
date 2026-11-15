"""Plugin-contract tests for the managed pytest run root (C4 chunks i/iii).

Each integration test runs a real ``python -m pytest -p pytest_chartreux``
subprocess with ``TMPDIR`` pointed at an isolated root, so the managed parent
``<temproot>/chartreux-managed-<uid>/`` is exercised without touching the real
``/tmp``. The repo default ``tmp_path_retention_policy`` is ``failed`` (activated
at chunk ii); the green-cleanup leg still passes it explicitly per run, as
pinned by B3 §11(i). Chunk (iii) adds the teardown-hardening pins: a test that
globally monkeypatches fs calls must never turn the gated deletion path into a
teardown ERROR.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest

import chartreux
from chartreux.core import tmp_ownership as to
from chartreux.core.scratchpad import reap_stale_allocations

REPO_ROOT = Path(chartreux.__file__).resolve().parent.parent

_SUBPROCESS_TIMEOUT = 180


@pytest.fixture(autouse=True)
def _clear_ambient_binding():
    to.set_ambient_binding(None)
    yield
    to.set_ambient_binding(None)


@pytest.fixture
def temproot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated temp root; TMPDIR redirected so both this process and the
    pytest subprocesses resolve it via tempfile.gettempdir().
    """
    root = tmp_path / "temproot"
    root.mkdir()
    monkeypatch.setenv("TMPDIR", str(root))
    monkeypatch.setattr(tempfile, "tempdir", None)
    return root


def _managed_parent(temproot: Path) -> Path:
    return to.managed_parent(temproot)


def _write_case(case_dir: Path, body: str) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    test_file = case_dir / "test_case.py"
    test_file.write_text(body)
    return test_file


def _subprocess_env(temproot: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["TMPDIR"] = str(temproot)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_DEBUG_TEMPROOT", None)
    return env


def _run_pytest(
    temproot: Path,
    case_dir: Path,
    extra_args: list[str],
    timeout: int = _SUBPROCESS_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "pytest_chartreux",
        "-p",
        "no:cacheprovider",
        "-q",
        *extra_args,
        str(case_dir),
    ]
    return subprocess.run(
        cmd,
        env=_subprocess_env(temproot),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=case_dir,
    )


def _run_roots(parent: Path) -> list[Path]:
    return sorted(p for p in parent.glob("pytest-run-*") if p.is_dir())


def _run_root_markers(parent: Path) -> list[Path]:
    return sorted(parent.glob("pytest-run-*.owner.json"))


GREEN_CASE = """
from pathlib import Path

def test_ok(tmp_path):
    assert tmp_path.is_dir()
    (Path(__file__).parent / "used.txt").write_text(str(tmp_path))
"""

RED_CASE = """
def test_pass(tmp_path):
    (tmp_path / "p.txt").write_text("p")

def test_fail(tmp_path):
    (tmp_path / "f.txt").write_text("f")
    assert False
"""

SLOW_CASE = """
import time

def test_slow(tmp_path):
    time.sleep(4)
"""

# C4 chunk (iii) hardening pins: a test that GLOBALLY monkeypatches an fs call
# hits the per-test gated rmtree at teardown while the patch is still active
# (the autouse fixture depends on monkeypatch so the patch is undone only after
# the tmp_path finalizer runs). shutil.rmtree passes an int dir-fd to
# os.scandir, so a patch that assumes a str path raises TypeError; the mkdir
# patch raises OSError from the gate's own managed_parent() mkdir; the open
# patch raises pytest.fail's BaseException on the ledger read.
_PATCH_HOLDER = """
import pytest

@pytest.fixture(autouse=True)
def _patch_holder(monkeypatch):
    yield
"""

MONKEYPATCHED_SCANDIR_CASE = (
    _PATCH_HOLDER
    + """
import os

def test_ok(tmp_path, monkeypatch):
    real_scandir = os.scandir

    def refusing_scandir(path):
        if isinstance(path, int):
            raise TypeError("argument should be a str, not int")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", refusing_scandir)
    (tmp_path / "used.txt").write_text("used")
    assert tmp_path.is_dir()
"""
)

MONKEYPATCHED_MKDIR_CASE = (
    _PATCH_HOLDER
    + """
import errno
from pathlib import Path

def test_ok(tmp_path, monkeypatch):
    def _enospc(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "mkdir", _enospc)
    (tmp_path / "used.txt").write_text("used")
    assert tmp_path.is_dir()
"""
)

MONKEYPATCHED_OPEN_CASE = (
    _PATCH_HOLDER
    + """
from pathlib import Path
import pytest

def test_ok(tmp_path, monkeypatch):
    (tmp_path / "used.txt").write_text("used")
    real_open = Path.open

    def refusing_open(self, *args, **kwargs):
        if "ledger" in str(self):
            pytest.fail("opened")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refusing_open)
    assert tmp_path.is_dir()
"""
)


class TestManagedRunRoot:
    def test_green_run_with_explicit_failed_policy_cleans_whole_tree(
        self, temproot: Path, tmp_path: Path
    ) -> None:
        case = tmp_path / "case-green"
        _write_case(case, GREEN_CASE)
        result = _run_pytest(temproot, case, ["-o", "tmp_path_retention_policy=failed"])
        assert result.returncode == 0, result.stderr
        # The run used the managed root, never pytest-of-<user>.
        used = (case / "used.txt").read_text()
        assert "chartreux-managed-" in used
        assert "pytest-run-" in used
        assert not list(temproot.glob("pytest-of-*"))
        parent = _managed_parent(temproot)
        # Green + policy=failed: the gated whole-basetemp rmtree removed the
        # whole run root; only the parent-beside marker remains.
        assert _run_roots(parent) == []
        assert len(_run_root_markers(parent)) == 1

    def test_repo_default_policy_all_keeps_tree_then_startup_reaper_collects(
        self, temproot: Path, tmp_path: Path
    ) -> None:
        # Chunk (i) pin: the repo default stays 'all', so native cleanup is
        # inert for ambient runs and a green run retains its whole tree...
        case = tmp_path / "case-ambient"
        _write_case(case, GREEN_CASE)
        result = _run_pytest(temproot, case, [])
        assert result.returncode == 0, result.stderr
        parent = _managed_parent(temproot)
        roots = _run_roots(parent)
        assert len(roots) == 1
        assert any(p.name.startswith("test_ok") for p in roots[0].iterdir())
        # ...and the next startup reaper collects the confirmed-dead
        # controller's allocation (kill-confirmable residue zero).
        results = reap_stale_allocations(temproot)
        assert any(r.reaped for r in results)
        assert _run_roots(parent) == []
        assert _run_root_markers(parent) == []

    def test_explicit_basetemp_takes_precedence(
        self, temproot: Path, tmp_path: Path
    ) -> None:
        case = tmp_path / "case-explicit"
        _write_case(case, GREEN_CASE)
        explicit = tmp_path / "explicit-base"
        result = _run_pytest(temproot, case, ["--basetemp", str(explicit)])
        assert result.returncode == 0, result.stderr
        # The explicit basetemp was used (default policy 'all' retains the
        # test dir) and no managed run root was allocated.
        assert any(p.name.startswith("test_ok") for p in explicit.iterdir())
        parent = _managed_parent(temproot)
        assert _run_roots(parent) == []
        assert _run_root_markers(parent) == []

    def test_red_run_retains_only_failed_call_phase_dirs(
        self, temproot: Path, tmp_path: Path
    ) -> None:
        case = tmp_path / "case-red"
        _write_case(case, RED_CASE)
        result = _run_pytest(temproot, case, ["-o", "tmp_path_retention_policy=failed"])
        assert result.returncode == 1
        parent = _managed_parent(temproot)
        roots = _run_roots(parent)
        assert len(roots) == 1
        names = {p.name for p in roots[0].iterdir()}
        assert any(n.startswith("test_fail") for n in names)
        assert not any(n.startswith("test_pass") for n in names)


class TestConcurrency:
    def test_concurrent_controllers_get_distinct_roots(
        self, temproot: Path, tmp_path: Path
    ) -> None:
        case = tmp_path / "case-slow"
        _write_case(case, SLOW_CASE)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pytest_chartreux",
            "-p",
            "no:cacheprovider",
            "-q",
            str(case),
        ]
        env = _subprocess_env(temproot)
        procs = [
            subprocess.Popen(
                cmd,
                env=env,
                cwd=case,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        try:
            parent = _managed_parent(temproot)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                roots = _run_roots(parent)
                if len(roots) >= 2:
                    break
                time.sleep(0.2)
            roots = _run_roots(parent)
            assert len(roots) == 2
            assert roots[0].name != roots[1].name
        finally:
            for proc in procs:
                try:
                    proc.wait(timeout=_SUBPROCESS_TIMEOUT)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        assert all(proc.returncode == 0 for proc in procs)
        # Both controllers' roots remain (default policy 'all'), distinct.
        parent = _managed_parent(temproot)
        assert len(_run_roots(parent)) == 2

    def test_xdist_workers_share_their_controllers_root(
        self, temproot: Path, tmp_path: Path
    ) -> None:
        case = tmp_path / "case-xdist"
        _write_case(
            case,
            GREEN_CASE + "\n\ndef test_ok2(tmp_path):\n    assert tmp_path.is_dir()\n",
        )
        result = _run_pytest(temproot, case, ["-n", "2"])
        assert result.returncode == 0, result.stderr
        parent = _managed_parent(temproot)
        roots = _run_roots(parent)
        # Exactly one controller run root; both workers' popen-gwN dirs live
        # inside it (no per-worker run roots).
        assert len(roots) == 1
        worker_dirs = sorted(
            p.name for p in roots[0].iterdir() if p.name.startswith("popen-gw")
        )
        assert worker_dirs == ["popen-gw0", "popen-gw1"]


class TestTeardownHardening:
    """C4 chunk (iii) pins: with the retention policy active, a test that
    globally monkeypatches an fs call must never turn the gated deletion path
    into a teardown error — the deletion is retained on ambiguity and only
    warned about (the flagged C4ii open item: 14 teardown errors).
    """

    def _assert_no_teardown_error(
        self, temproot: Path, tmp_path: Path, name: str, body: str
    ) -> None:
        case = tmp_path / name
        _write_case(case, body)
        result = _run_pytest(temproot, case, ["-o", "tmp_path_retention_policy=failed"])
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "ERROR at teardown" not in output
        # The ambiguous deletion was retained, not silently dropped, and the
        # warning names the retained managed path.
        assert "retaining on ambiguity" in output
        assert "chartreux-managed-" in output

    def test_monkeypatched_scandir_never_errors_teardown(
        self, temproot: Path, tmp_path: Path
    ) -> None:
        self._assert_no_teardown_error(
            temproot, tmp_path, "case-scandir", MONKEYPATCHED_SCANDIR_CASE
        )

    def test_monkeypatched_mkdir_never_errors_teardown(
        self, temproot: Path, tmp_path: Path
    ) -> None:
        self._assert_no_teardown_error(
            temproot, tmp_path, "case-mkdir", MONKEYPATCHED_MKDIR_CASE
        )

    def test_monkeypatched_open_never_errors_teardown(
        self, temproot: Path, tmp_path: Path
    ) -> None:
        self._assert_no_teardown_error(
            temproot, tmp_path, "case-open", MONKEYPATCHED_OPEN_CASE
        )
