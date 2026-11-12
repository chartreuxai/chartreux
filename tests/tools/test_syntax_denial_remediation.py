"""Every shell syntax denial names the compliant alternative inline (P2-e).

The recognition-contract treatment: a worker denied five or eight times for
the same construct was given no instruction, so each audit category's denial
must carry its specific remediation in the rendered message. One pin per
audit category asserts that the actual denial message for a representative
command contains the compliant alternative.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_diagnostics import DenialClass, Diagnostic
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.utils import resolve_file_tool_permission
from chartreux.core.workspace import Workspace

# One remediation pin per audit category: (representative command, the
# denial's recognized marker, the compliant alternative the message must
# name). The glob command denies through the expansion inspection's
# unsupported-syntax failure, the one path-glob shape the resolver classifies
# as a syntax denial; caller-side rejections (an option-like word, a match
# outside the workspace) stay policy denials and name no alternative.
_PINS: tuple[tuple[str, str, str], ...] = (
    (
        "echo $HOME",
        "variable expansion",
        "write the literal value directly as the argument",
    ),
    (
        "grep src/**/x.py target.py",
        "path glob candidate",
        "list the directory with ls to see the matching names, or write "
        "the exact paths instead of the glob",
    ),
    ("echo =ls", "Zsh equals expansion", "quote the marker"),
    (
        "cat <<EOF\nhi\nEOF",
        "unquoted heredoc delimiter",
        "use write_file to create the content",
    ),
    (
        "for i in 1 2; do echo $i; done",
        "for loop",
        "split the work into separate simple commands",
    ),
)


def _denied_reason(cwd: Path, command: str) -> str:
    tool = Bash(config_getter=BashToolConfig, state=BaseToolState(), cwd=cwd)
    result = tool.resolve_permission(BashArgs(command=command))
    assert result is not None, command
    assert result.permission == ToolPermission.NEVER, (command, result)
    return result.reason or ""


@pytest.mark.parametrize(("command", "marker", "alternative"), _PINS, ids=range(5))
def test_syntax_denial_names_the_compliant_alternative(
    tmp_path: Path, command: str, marker: str, alternative: str, zsh_executor
) -> None:
    (tmp_path / "target.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("x = 2\n", encoding="utf-8")
    reason = _denied_reason(tmp_path, command)
    assert marker in reason, reason
    assert "Instead," in reason, reason
    assert alternative in reason, reason


@pytest.mark.parametrize("command", ["echo =ls", "echo ===X==="])
def test_unquoted_equals_marker_stays_denied_with_the_quoting_remediation(
    tmp_path: Path, command: str, zsh_executor
) -> None:
    # S1 admits the quoted and escaped marker only; under the zsh executor the
    # unquoted form keeps its denial and names the exact remediation (quote
    # the marker). The shipped bash default never expands the marker, so the
    # family is pinned here.
    reason = _denied_reason(tmp_path, command)
    assert "Zsh equals expansion" in reason, reason
    assert "quote the marker" in reason, reason
    assert "Instead," in reason, reason


def test_syntax_denial_without_a_compliant_alternative_is_rejected() -> None:
    # The recognition contract is a constructor invariant: a syntax denial
    # that names no remediation never renders, and a policy denial never
    # names one.
    with pytest.raises(ValueError, match="lacks a compliant alternative"):
        Diagnostic("analysis", detail="unmodeled", denial_class=DenialClass.SYNTAX)
    related = Diagnostic(
        "analysis",
        detail="unmodeled",
        denial_class=DenialClass.SYNTAX,
        alternative="write the literal value directly as the argument",
    )
    aggregate = Diagnostic(
        "analysis", related=(related,), denial_class=DenialClass.SYNTAX
    )
    assert aggregate.alternative is None
    with pytest.raises(ValueError, match="lacks a compliant alternative"):
        Diagnostic(
            "analysis",
            related=(Diagnostic("analysis", detail="unknown"),),
            denial_class=DenialClass.SYNTAX,
        )
    with pytest.raises(ValueError, match="only a syntax denial carries one"):
        Diagnostic(
            "analysis",
            detail="denied",
            alternative="write the literal value directly as the argument",
        )


# The shape-conservatism codes: an inspection gap in the analyzer, not a
# deliberate policy decision, so each denial names the compliant form.
_SHAPE_CONSERVATISM_PINS: tuple[tuple[str, str, str], ...] = (
    (
        "unset PATH",
        "unsetting shell environment variables cannot be safely inspected",
        "run the command without unsetting environment variables",
    ),
    (
        "X=$Y git status",
        "a non-literal environment assignment",
        "write the assignment as NAME=value with a literal value",
    ),
    (
        "echo hi > $TMP/out.txt",
        "a non-literal redirection target",
        "write the redirection destination as a literal path",
    ),
    (
        "echo hi > -x",
        "an invalid redirection path target",
        "write the redirection destination as a literal path",
    ),
)


@pytest.mark.parametrize(
    ("command", "marker", "alternative"),
    _SHAPE_CONSERVATISM_PINS,
    ids=range(len(_SHAPE_CONSERVATISM_PINS)),
)
def test_shape_conservatism_denials_name_the_compliant_form(
    tmp_path: Path, command: str, marker: str, alternative: str
) -> None:
    reason = _denied_reason(tmp_path, command)
    assert "unsupported shell syntax:" in reason, reason
    assert marker in reason, reason
    assert "Instead," in reason, reason
    assert alternative in reason, reason


def test_unknown_and_mixed_denials_stay_policy(tmp_path: Path) -> None:
    # The fail-closed default survives the re-tagging: an unclassified node
    # and a construct mixing unmodeled syntax with a policy violation name no
    # alternative.
    unknown = _denied_reason(tmp_path, "cat <<<x")
    assert "herestring" in unknown, unknown
    assert "Instead," not in unknown, unknown
    mixed = _denied_reason(tmp_path, "PYTHONPATH=.. echo hi > $TMP/x")
    assert "dangerous environment assignment" in mixed, mixed
    assert "Instead," not in mixed, mixed


def test_sensitive_path_denial_stays_bare_policy(tmp_path: Path) -> None:
    # Suggesting an alternative for a secrets denial would invite the refused
    # access through another form: the message stays as shipped.
    target = tmp_path / ".env"
    target.write_text("SECRET=1", encoding="utf-8")
    decision = resolve_file_tool_permission(
        str(target),
        tool_name="read_file",
        allowlist=[],
        denylist=[],
        config_permission=ToolPermission.ALWAYS,
        sensitive_patterns=["**/.env"],
        workspace=Workspace.for_session(tmp_path),
    )
    assert decision is not None
    assert decision.permission is ToolPermission.NEVER
    assert decision.denial_kind == "sensitive"
    assert decision.reason == "Sensitive file access denied (read_file)"
    assert "Instead," not in (decision.reason or "")
