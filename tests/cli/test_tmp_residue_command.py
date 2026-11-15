"""Tests for the sessionless `chartreux tmp-residue` preview/confirmation UX
(C4 chunk iii, B3 §9 contract).

The preview (`list`) is ALWAYS read-only and works noninteractively; `remove`
deletes only after an interactive confirmation that names the exact paths, and
noninteractive operation stays preview-only. No legacy/unmarked tree is ever
auto-deleted: every deletion in these tests is a probe-created fixture tree
under a tmp temp root, explicitly confirmed.
"""

from __future__ import annotations

import io
from pathlib import Path
import subprocess

import pytest

from chartreux.cli import tmp_residue_command
from chartreux.core import tmp_ownership as to


@pytest.fixture(autouse=True)
def _clear_ambient_binding():
    to.set_ambient_binding(None)
    yield
    to.set_ambient_binding(None)


@pytest.fixture
def temproot(tmp_path: Path) -> Path:
    root = tmp_path / "temproot"
    root.mkdir()
    return root


def _spawned_pid() -> int:
    handle = subprocess.Popen(["true"])
    handle.wait()
    return handle.pid


def _dead_identity(pid: int | None = None) -> to.IdentityTuple:
    pid = _spawned_pid() if pid is None else pid
    return to.IdentityTuple(pid=pid, pgid=pid, starttime=0, boot_id=to.boot_id() or "")


def _legacy_tree(temproot: Path, name: str = "chartreux-scratchpad-probe") -> Path:
    tree = temproot / name
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "data.txt").write_text("probe residue\n")
    return tree


def _managed_alloc(
    temproot: Path, *, owner: to.IdentityTuple | None = None
) -> to.Allocation:
    parent = to.managed_parent(temproot)
    return to.init_allocation(
        parent, to.KIND_SCRATCHPAD, owner=owner or to.current_identity()
    )


class _TtyStdin(io.StringIO):
    def isatty(self) -> bool:
        return True


class _TtyStdout(io.StringIO):
    def isatty(self) -> bool:
        return True


def _exit_code(exc: pytest.ExceptionInfo[SystemExit]) -> int:
    code = exc.value.code
    assert isinstance(code, int)
    return code


def _run_captured(
    capsys: pytest.CaptureFixture[str], *argv: str
) -> tuple[int, str, str]:
    with pytest.raises(SystemExit) as exc:
        tmp_residue_command.run_tmp_residue_cli(list(argv))
    captured = capsys.readouterr()
    return _exit_code(exc), captured.out, captured.err


def _run_interactive(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
    *argv: str,
) -> tuple[int, str, str]:
    stdout = _TtyStdout()
    monkeypatch.setattr("sys.stdin", _TtyStdin(answer))
    monkeypatch.setattr("sys.stdout", stdout)
    with pytest.raises(SystemExit) as exc:
        tmp_residue_command.run_tmp_residue_cli(list(argv))
    captured = capsys.readouterr()
    return _exit_code(exc), stdout.getvalue(), captured.err


class TestPreviewList:
    def test_list_is_read_only_and_names_exact_paths(
        self, temproot: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        legacy = _legacy_tree(temproot)
        alloc = _managed_alloc(temproot, owner=_dead_identity())
        (alloc.path / "scratch.txt").write_text("x")
        code, out, _ = _run_captured(capsys, "list", "--temp-root", str(temproot))
        assert code == 0
        assert "Preview only" in out
        # Exact absolute paths, entry counts, and classes are listed.
        assert str(legacy) in out
        assert str(alloc.path) in out
        assert "legacy-unmarked" in out
        assert "managed" in out
        assert "entries=2" in out  # legacy tree: sub + data.txt
        # Managed allocations show the owner identity summary and gate state.
        assert "owner pid=" in out
        assert "gates:" in out
        # The preview deleted nothing.
        assert legacy.is_dir()
        assert alloc.path.is_dir()

    def test_list_never_includes_the_shared_pytest_root(
        self, temproot: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _legacy_tree(temproot)
        pytest_root = temproot / "pytest-of-pav"
        pytest_root.mkdir()
        (pytest_root / "pytest-0").mkdir()
        code, out, _ = _run_captured(capsys, "list", "--temp-root", str(temproot))
        assert code == 0
        assert "pytest-of-pav" not in out
        assert pytest_root.is_dir()

    def test_list_empty_root(
        self, temproot: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, _ = _run_captured(capsys, "list", "--temp-root", str(temproot))
        assert code == 0
        assert "No residue trees." in out


class TestRemoveGuards:
    def test_remove_noninteractive_is_preview_only_and_deletes_nothing(
        self, temproot: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        legacy = _legacy_tree(temproot)
        code, _, err = _run_captured(
            capsys, "remove", str(legacy), "--temp-root", str(temproot)
        )
        assert code == 1
        assert "interactive confirmation" in err
        assert "preview-only" in err
        # The refusal names the exact path that would have been removed.
        assert str(legacy) in err
        assert "There is no force flag." in err
        assert legacy.is_dir()

    @pytest.mark.parametrize("flag", ["--yes", "--force", "-y"])
    def test_remove_has_no_assume_yes_flag(
        self, temproot: Path, capsys: pytest.CaptureFixture[str], flag: str
    ) -> None:
        legacy = _legacy_tree(temproot)
        with pytest.raises(SystemExit) as exc:
            tmp_residue_command.run_tmp_residue_cli([
                "remove",
                str(legacy),
                flag,
                "--temp-root",
                str(temproot),
            ])
        captured = capsys.readouterr()
        assert exc.value.code == 2
        assert flag in captured.err
        assert legacy.is_dir()

    def test_remove_refuses_a_path_outside_the_inventory(
        self, temproot: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _legacy_tree(temproot)
        outside = temproot / "not-residue"
        outside.mkdir()
        code, _, err = _run_captured(
            capsys, "remove", str(outside), "--temp-root", str(temproot)
        )
        assert code == 1
        assert "not in the residue inventory" in err
        assert outside.is_dir()

    def test_remove_refuses_the_shared_pytest_root(
        self, temproot: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pytest_root = temproot / "pytest-of-pav"
        (pytest_root / "pytest-0").mkdir(parents=True)
        code, _, err = _run_captured(
            capsys, "remove", str(pytest_root), "--temp-root", str(temproot)
        )
        assert code == 1
        assert "never swept" in err
        assert pytest_root.is_dir()

    def test_remove_refuses_a_live_managed_allocation(
        self, temproot: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        alloc = _managed_alloc(temproot)  # live owner = this process
        code, _, err = _run_captured(
            capsys, "remove", str(alloc.path), "--temp-root", str(temproot)
        )
        assert code == 1
        assert "retained by its live-user gates" in err
        assert alloc.path.is_dir()

    def test_remove_refuses_a_symlink(
        self, temproot: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        legacy = _legacy_tree(temproot)
        link = tmp_path / "link"
        link.symlink_to(legacy)
        code, _, err = _run_captured(
            capsys, "remove", str(link), "--temp-root", str(temproot)
        )
        assert code == 1
        assert "not in the residue inventory" in err or "symlink" in err
        assert legacy.is_dir()


class TestRemoveInteractiveConfirmation:
    def test_confirmation_names_exact_paths_and_deletes_only_them(
        self,
        temproot: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        legacy = _legacy_tree(temproot)
        keeper = _legacy_tree(temproot, name="vibe-scratchpad-keeper")
        code, out, _ = _run_interactive(
            monkeypatch,
            capsys,
            "y\n",
            "remove",
            str(legacy),
            "--temp-root",
            str(temproot),
        )
        assert code == 0
        # The confirmation named the exact absolute path and warned that
        # liveness cannot be established for unmarked trees.
        assert str(legacy) in out
        assert "liveness cannot be established for unmarked trees" in out
        assert "Remove these 1 exact path(s)? [y/N]" in out
        assert f"Removed legacy-unmarked tree {legacy}." in out
        assert not legacy.exists()
        # The sibling tree is untouched.
        assert keeper.is_dir()

    def test_confirmation_default_answer_is_no(
        self,
        temproot: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        legacy = _legacy_tree(temproot)
        code, out, _ = _run_interactive(
            monkeypatch,
            capsys,
            "\n",
            "remove",
            str(legacy),
            "--temp-root",
            str(temproot),
        )
        assert code == 1
        assert "Removal cancelled" in out
        assert legacy.is_dir()

    def test_interactive_remove_of_a_reapable_managed_allocation(
        self,
        temproot: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        alloc = _managed_alloc(temproot, owner=_dead_identity())
        (alloc.path / "scratch.txt").write_text("x")
        code, out, _ = _run_interactive(
            monkeypatch,
            capsys,
            "yes\n",
            "remove",
            str(alloc.path),
            "--temp-root",
            str(temproot),
        )
        assert code == 0
        assert str(alloc.path) in out
        assert not alloc.path.exists()
        assert not alloc.owner_record_path.exists()


class TestSurfaceWiring:
    def test_slash_command_is_registered(self) -> None:
        from chartreux.cli.commands import CommandRegistry

        command = CommandRegistry().get("tmp-residue")
        assert command is not None
        assert "/tmp-residue" in command.aliases
        assert command.handler == "_show_tmp_residue"

    def test_entrypoint_dispatches_tmp_residue_without_a_session(
        self,
        temproot: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from chartreux.cli import entrypoint

        _legacy_tree(temproot)

        def _never(*args: object, **kwargs: object) -> None:
            raise AssertionError("tmp-residue commands must not start a session")

        monkeypatch.setattr("chartreux.cli.cli.run_cli", _never)
        monkeypatch.setattr(
            "sys.argv",
            ["chartreux", "tmp-residue", "list", "--temp-root", str(temproot)],
        )
        with pytest.raises(SystemExit) as exc:
            entrypoint.main()
        assert _exit_code(exc) == 0
        out = capsys.readouterr().out
        assert "legacy-unmarked" in out
