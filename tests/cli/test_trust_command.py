from __future__ import annotations

import io
from pathlib import Path
import tomllib

import pytest
import tomli_w

from chartreux.cli import trust_command


def _store_path() -> Path:
    from chartreux.core.paths import TRUSTED_FOLDERS_FILE

    return TRUSTED_FOLDERS_FILE.path


def _write_store(*, trusted: list[str], untrusted: list[str]) -> Path:
    path = _store_path()
    path.write_text(tomli_w.dumps({"trusted": trusted, "untrusted": untrusted}))
    return path


def _stored() -> dict[str, list[str]]:
    return tomllib.loads(_store_path().read_text(encoding="utf-8"))


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
        trust_command.run_trust_cli(list(argv))
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
        trust_command.run_trust_cli(list(argv))
    captured = capsys.readouterr()
    return _exit_code(exc), stdout.getvalue(), captured.err


def test_list_prints_both_stored_decision_kinds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trusted_dir = tmp_path / "trusted"
    untrusted_dir = tmp_path / "untrusted"
    _write_store(trusted=[str(trusted_dir)], untrusted=[str(untrusted_dir)])
    code, out, err = _run_captured(capsys, "list")
    assert code == 0
    assert err == ""
    trusted_rows = [line for line in out.splitlines() if str(trusted_dir) in line]
    untrusted_rows = [line for line in out.splitlines() if str(untrusted_dir) in line]
    assert len(trusted_rows) == 1
    assert len(untrusted_rows) == 1
    assert "trusted" in trusted_rows[0]
    assert "stored" in trusted_rows[0]
    assert "untrusted" in untrusted_rows[0]
    assert "stored" in untrusted_rows[0]


def test_list_reports_no_stored_decisions(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = _run_captured(capsys, "list")
    assert code == 0
    assert err == ""
    assert "No stored trust decisions." in out


def test_list_explains_what_trust_authorizes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Trust covers config, harness resources, and the project shell environment."""
    _write_store(trusted=["/p"], untrusted=[])
    code, out, _ = _run_captured(capsys, "list")
    assert code == 0
    assert "project configuration" in out
    assert "harness resources" in out
    assert "project shell environment" in out
    assert "authorized_roots_by_project" in out
    assert "does not grant workspace roots" in out


def test_list_resolves_the_closest_recorded_ancestor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    sibling = parent / "sibling"
    _write_store(trusted=[str(parent)], untrusted=[str(child)])
    code, out, _ = _run_captured(capsys, "list", str(child))
    assert code == 0
    assert f"{child}: untrusted (recorded at this path)" in out
    code, out, _ = _run_captured(capsys, "list", str(child / "nested"))
    assert code == 0
    assert f"{child / 'nested'}: untrusted (inherited from {child})" in out
    code, out, _ = _run_captured(capsys, "list", str(sibling))
    assert code == 0
    assert f"{sibling}: trusted (inherited from {parent})" in out
    code, out, _ = _run_captured(capsys, "list", str(tmp_path / "elsewhere"))
    assert code == 0
    assert "undecided" in out


def test_list_marks_a_nested_entry_as_overriding_its_ancestor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    _write_store(trusted=[str(parent)], untrusted=[str(child)])
    code, out, _ = _run_captured(capsys, "list")
    assert code == 0
    child_rows = [line for line in out.splitlines() if str(child) in line]
    assert len(child_rows) == 1
    assert f"overrides trusted {parent}" in child_rows[0]


def test_remove_deletes_only_the_exact_recorded_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    parent = tmp_path / "p"
    vendor = parent / "vendor"
    nested = vendor / "nested"
    sibling = parent / "sibling"
    _write_store(
        trusted=[str(parent), str(nested)], untrusted=[str(vendor), str(sibling)]
    )
    code, out, err = _run_interactive(monkeypatch, capsys, "y\n", "remove", str(vendor))
    assert code == 0
    assert err == ""
    assert f"Removed stored decision untrusted {vendor}." in out
    assert _stored() == {
        "trusted": [str(parent), str(nested)],
        "untrusted": [str(sibling)],
    }
    assert (
        f"After removal, {vendor} resolves to trusted (inherited from {parent})." in out
    )


def test_remove_refuses_a_path_that_only_inherits_trust(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Removal acts on recorded entries, never on a resolution result."""
    parent = tmp_path / "p"
    child = parent / "child"
    _write_store(trusted=[str(parent)], untrusted=[])
    code, _, err = _run_captured(capsys, "remove", str(child))
    assert code == 1
    assert "no stored decision is recorded" in err
    assert _stored()["trusted"] == [str(parent)]


def test_remove_confirmation_default_answer_is_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "p"
    _write_store(trusted=[str(target)], untrusted=[])
    code, out, _ = _run_interactive(monkeypatch, capsys, "\n", "remove", str(target))
    assert code == 1
    assert "[y/N]" in out
    assert str(target) in out
    assert "project configuration" in out
    assert "harness resources" in out
    assert "Removal cancelled" in out
    assert _stored()["trusted"] == [str(target)]


def test_remove_is_denied_without_an_interactive_terminal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "p"
    _write_store(trusted=[str(target)], untrusted=[])
    code, _, err = _run_captured(capsys, "remove", str(target))
    assert code == 1
    assert "interactive confirmation" in err
    assert "There is no force flag." in err
    assert _stored()["trusted"] == [str(target)]


@pytest.mark.parametrize("flag", ["--yes", "--force", "-y"])
def test_remove_has_no_assume_yes_flag(
    flag: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "p"
    _write_store(trusted=[str(target)], untrusted=[])
    with pytest.raises(SystemExit) as exc:
        trust_command.run_trust_cli(["remove", str(target), flag])
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert flag in captured.err
    assert _stored()["trusted"] == [str(target)]


def test_remove_prints_a_restart_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "p"
    _write_store(trusted=[str(target)], untrusted=[])
    code, out, _ = _run_interactive(monkeypatch, capsys, "yes\n", "remove", str(target))
    assert code == 0
    assert "Restart any running Chartreux session" in out
    assert "/reload" not in out


def test_entrypoint_dispatches_trust_without_a_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`chartreux trust` runs sessionless: dispatch never reaches the session CLI."""
    from chartreux.cli import entrypoint

    _write_store(trusted=["/p"], untrusted=["/q"])

    def _never(*args: object, **kwargs: object) -> None:
        raise AssertionError("trust commands must not start a session")

    monkeypatch.setattr("chartreux.cli.cli.run_cli", _never)
    monkeypatch.setattr("sys.argv", ["chartreux", "trust", "list"])
    with pytest.raises(SystemExit) as exc:
        entrypoint.main()
    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "Stored trust decisions" in captured.out
    assert "/p" in captured.out
    assert "/q" in captured.out


@pytest.mark.parametrize(
    ("contents", "state"),
    [("[malformed", "malformed"), ('trusted = "invalid"', "invalid")],
    ids=["malformed", "invalid"],
)
def test_list_refuses_an_unusable_store(
    contents: str, state: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _store_path()
    path.write_text(contents)
    code, _, err = _run_captured(capsys, "list")
    assert code == 2
    assert f"trust store {state}" in err
    assert path.read_text(encoding="utf-8") == contents


def test_remove_refuses_an_unusable_store_and_never_rewrites_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _store_path()
    path.write_text("[malformed")
    code, _, err = _run_interactive(monkeypatch, capsys, "y\n", "remove", "/p")
    assert code == 2
    assert "trust store malformed" in err
    assert "nothing was removed" in err
    assert path.read_text(encoding="utf-8") == "[malformed"


def test_no_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    trust_command.run_trust_cli([])
    captured = capsys.readouterr()
    assert "chartreux trust" in captured.out
    assert "list" in captured.out
    assert "remove" in captured.out
