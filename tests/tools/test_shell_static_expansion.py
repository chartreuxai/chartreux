"""Standalone unit tests for the static-expansion modules (AM-E1).

The autouse fixture patches every process-execution entry point to raise,
so every test in this file doubles as part of the execution-impossibility
proof: no recognition path ever executes anything while analyzing a
construct. Reading input files is allowed (read-only) and asserted only
through the analyzers' results.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import subprocess

import pytest

from chartreux.core.tools.builtins._shell_argument_positions import (
    ARGUMENT_POSITION_RULES,
    classify_argument_position,
    is_read_only_argument_position,
    read_only_positions,
)
from chartreux.core.tools.builtins._shell_static_expansion import (
    _SIMULATION_READ_CAP_BYTES,
    evaluate_expansion_for_position,
    evaluate_static_expansion,
    recognize_sed_locate_print,
    resolve_equals_command,
    sed_locate_print_source,
)

# --- execution-impossibility proof (autouse for every test below) -------------


def _execution_forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("static analysis must never execute a process")


_SUBPROCESS_ENTRYPOINTS = ("run", "call", "check_call", "check_output", "Popen")
_OS_EXEC_ENTRYPOINTS = (
    "system",
    "popen",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "posix_spawn",
    "posix_spawnp",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawn",
    "spawnvp",
    "spawnvpe",
)


@pytest.fixture(autouse=True)
def _no_process_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ban every process-execution entry point for the whole test module."""
    for name in _SUBPROCESS_ENTRYPOINTS:
        monkeypatch.setattr(subprocess, name, _execution_forbidden)
    for name in _OS_EXEC_ENTRYPOINTS:
        if hasattr(os, name):
            monkeypatch.setattr(os, name, _execution_forbidden)


def test_execution_patch_is_active() -> None:
    """Canary: the patched entry points really raise, so pass = no call."""
    with pytest.raises(AssertionError):
        subprocess.run(["true"])  # type: ignore[call-arg]
    with pytest.raises(AssertionError):
        os.system("true")  # type: ignore[call-arg]
    with pytest.raises(AssertionError):
        os.popen("true")  # type: ignore[call-arg]
    with pytest.raises(AssertionError):
        os.execvp("true", ["true"])
    with pytest.raises(AssertionError):
        os.posix_spawn("true", ["true"], {})  # type: ignore[call-arg]
    with pytest.raises(AssertionError):
        os.spawnvp(os.P_WAIT, "true", ["true"])  # type: ignore[call-arg]


# --- fixtures -----------------------------------------------------------------


@pytest.fixture()
def single_match_file(tmp_path: Path) -> str:
    path = tmp_path / "single.txt"
    path.write_text("alpha\nneedle here\nbeta\nend\n")
    return str(path)


@pytest.fixture()
def multi_match_file(tmp_path: Path) -> str:
    path = tmp_path / "multi.txt"
    path.write_text("alpha\nneedle here\nbeta\nneedle again\nend\n")
    return str(path)


def idiom(file: str, suffix: str, pipeline: str, sed_file: str | None = None) -> str:
    return f'sed -n "$({pipeline}){suffix}" {sed_file or file}'


# --- construct (a): recognized idiom and variants -----------------------------


def test_observed_idiom_head_cut_range_suffix(single_match_file: str) -> None:
    result = recognize_sed_locate_print(
        idiom(
            single_match_file,
            ",+45p",
            f"grep -n 'needle' {single_match_file} | head -1 | cut -d: -f1",
        )
    )
    assert result is not None
    assert result.expanded_script == "2,+45p"
    assert result.address == "2"
    assert result.suffix == ",+45p"
    assert result.addresses == (2,)
    assert result.grep_pattern == "needle"
    assert result.source_file == single_match_file
    assert result.sed_file == single_match_file


def test_pattern_with_spaces_and_head_n_form(single_match_file: str) -> None:
    result = recognize_sed_locate_print(
        idiom(
            single_match_file,
            ",+45p",
            f"grep -n 'needle here' {single_match_file} | head -n 3 | cut -d: -f1",
        )
    )
    assert result is not None
    assert result.expanded_script == "2,+45p"


def test_no_head_variant(single_match_file: str) -> None:
    result = recognize_sed_locate_print(
        idiom(
            single_match_file,
            ",+45p",
            f"grep -n 'needle' {single_match_file} | cut -d: -f1",
        )
    )
    assert result is not None
    assert result.expanded_script == "2,+45p"
    assert result.addresses == (2,)


def test_tail_selector_picks_last_match(multi_match_file: str) -> None:
    result = recognize_sed_locate_print(
        idiom(
            multi_match_file,
            ",+45p",
            f"grep -n 'needle' {multi_match_file} | tail -1 | cut -d: -f1",
        )
    )
    assert result is not None
    assert result.expanded_script == "4,+45p"
    assert result.addresses == (4,)


def test_absolute_range_suffix(multi_match_file: str) -> None:
    result = recognize_sed_locate_print(
        idiom(
            multi_match_file,
            ",12p",
            f"grep -n 'needle' {multi_match_file} | head -1 | cut -d: -f1",
        )
    )
    assert result is not None
    assert result.expanded_script == "2,12p"


def test_plain_p_suffix(single_match_file: str) -> None:
    result = recognize_sed_locate_print(
        idiom(
            single_match_file,
            "p",
            f"grep -n 'needle' {single_match_file} | head -1 | cut -d: -f1",
        )
    )
    assert result is not None
    assert result.expanded_script == "2p"


def test_cut_before_head_ordering(multi_match_file: str) -> None:
    result = recognize_sed_locate_print(
        idiom(
            multi_match_file,
            "p",
            f"grep -n 'needle' {multi_match_file} | cut -d: -f1 | head -1",
        )
    )
    assert result is not None
    assert result.expanded_script == "2p"
    assert result.addresses == (2,)


def test_split_cut_spelling(single_match_file: str) -> None:
    result = recognize_sed_locate_print(
        idiom(
            single_match_file,
            "p",
            f"grep -n 'needle' {single_match_file} | head -1 | cut -d : -f 1",
        )
    )
    assert result is not None
    assert result.expanded_script == "2p"


def test_head_and_tail_selectors_together(single_match_file: str) -> None:
    result = recognize_sed_locate_print(
        idiom(
            single_match_file,
            "p",
            f"grep -n 'needle' {single_match_file} | head -3 | tail -1 | cut -d: -f1",
        )
    )
    assert result is not None
    assert result.expanded_script == "2p"


def test_inner_and_outer_files_may_differ(
    single_match_file: str, tmp_path: Path
) -> None:
    other = tmp_path / "other.txt"
    other.write_text("irrelevant\n")
    result = recognize_sed_locate_print(
        idiom(
            single_match_file,
            ",+5p",
            f"grep -n 'needle' {single_match_file} | head -1 | cut -d: -f1",
            sed_file=str(other),
        )
    )
    assert result is not None
    assert result.expanded_script == "2,+5p"
    assert result.source_file == single_match_file
    assert result.sed_file == str(other)


def test_multi_match_expansion_is_digits_and_newlines_only(
    multi_match_file: str,
) -> None:
    """Multi-match is a runtime sed error (probed), never a write vector."""
    result = recognize_sed_locate_print(
        idiom(
            multi_match_file, "p", f"grep -n 'needle' {multi_match_file} | cut -d: -f1"
        )
    )
    assert result is not None
    assert result.addresses == (2, 4)
    assert re.fullmatch(r"\d+(\n\d+)*", result.address)
    assert result.expanded_script == "2\n4p"


def test_no_match_expands_to_empty_address(single_match_file: str) -> None:
    """No match: the address is empty; the script degrades to the suffix."""
    for suffix, expected in ((",+45p", ",+45p"), ("p", "p")):
        result = recognize_sed_locate_print(
            idiom(
                single_match_file,
                suffix,
                f"grep -n 'absent' {single_match_file} | head -1 | cut -d: -f1",
            )
        )
        assert result is not None
        assert result.addresses == ()
        assert result.address == ""
        assert result.expanded_script == expected


# --- construct (a): adversarial non-recognizable shapes -----------------------


def test_nested_substitution_inside_construct(single_match_file: str) -> None:
    nested = idiom(
        single_match_file,
        "p",
        f"grep -n $(echo needle) {single_match_file} | head -1 | cut -d: -f1",
    )
    assert recognize_sed_locate_print(nested) is None


def test_nested_substitution_in_arithmetic_shape(single_match_file: str) -> None:
    """zsh performs command substitution inside $((...)) — unmodeled, rejected."""
    command = f'sed -n "$(( $(echo 2) ))p" {single_match_file}'
    assert recognize_sed_locate_print(command) is None


def test_redirect_inside_substitution(single_match_file: str, tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    command = idiom(
        single_match_file,
        "p",
        f"grep -n 'needle' {single_match_file} > {out} | head -1 | cut -d: -f1",
    )
    assert recognize_sed_locate_print(command) is None


def test_redirect_after_outer_command(single_match_file: str, tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    command = (
        f"sed -n \"$(grep -n 'needle' {single_match_file} "
        f'| head -1 | cut -d: -f1),+45p" {single_match_file} > {out}'
    )
    assert recognize_sed_locate_print(command) is None


def test_head_without_cut_is_not_recognizable(tmp_path: Path) -> None:
    """The probed write vector: raw file content reaches the sed script.

    Under real zsh this exact shape executes the `w` write command and
    creates the marker file; the recognizer must reject it outright.
    """
    marker = "/tmp/chartreux-static-expansion-marker"
    evil = tmp_path / "evil.txt"
    evil.write_text(f"w {marker}\n")
    target = tmp_path / "target.txt"
    target.write_text("zzz\n")
    command = idiom(
        str(evil), "p", f"grep -n 'w {marker}' {evil} | head -1", sed_file=str(target)
    )
    assert recognize_sed_locate_print(command) is None
    assert not os.path.exists(marker)


def test_non_literal_grep_arguments(single_match_file: str) -> None:
    variable = idiom(
        single_match_file,
        "p",
        f"grep -n $PAT {single_match_file} | head -1 | cut -d: -f1",
    )
    assert recognize_sed_locate_print(variable) is None
    double_quoted = idiom(
        single_match_file,
        "p",
        f'grep -n "needle" {single_match_file} | head -1 | cut -d: -f1',
    )
    assert recognize_sed_locate_print(double_quoted) is None


def test_grep_pattern_with_regex_metacharacter(single_match_file: str) -> None:
    metachar = idiom(
        single_match_file,
        "p",
        f"grep -n 'ndl.' {single_match_file} | head -1 | cut -d: -f1",
    )
    assert recognize_sed_locate_print(metachar) is None


def test_grep_without_line_numbers(single_match_file: str) -> None:
    command = idiom(
        single_match_file,
        "p",
        f"grep 'needle' {single_match_file} | head -1 | cut -d: -f1",
    )
    assert recognize_sed_locate_print(command) is None


def test_injection_shape_in_suffix(single_match_file: str) -> None:
    command = (
        f"sed -n \"$(grep -n 'needle' {single_match_file} "
        f'| head -1 | cut -d: -f1),+45p; w /tmp/x" {single_match_file}'
    )
    assert recognize_sed_locate_print(command) is None


def test_literal_injection_without_substitution(single_match_file: str) -> None:
    command = f'sed -n "2p; w /tmp/x" {single_match_file}'
    assert recognize_sed_locate_print(command) is None


def test_further_substitution_layer(single_match_file: str) -> None:
    sub = idiom(
        single_match_file,
        ",$(echo 5)p",
        f"grep -n 'needle' {single_match_file} | head -1 | cut -d: -f1",
    )
    assert recognize_sed_locate_print(sub) is None
    arith = idiom(
        single_match_file,
        ",+$((N))p",
        f"grep -n 'needle' {single_match_file} | head -1 | cut -d: -f1",
    )
    assert recognize_sed_locate_print(arith) is None


def test_second_substitution_segment(single_match_file: str) -> None:
    command = (
        f"sed -n \"$(grep -n 'needle' {single_match_file} | head -1 "
        f'| cut -d: -f1)$(echo 5)p" {single_match_file}'
    )
    assert recognize_sed_locate_print(command) is None


def test_unknown_pipeline_stage(single_match_file: str) -> None:
    command = idiom(
        single_match_file, "p", f"grep -n 'needle' {single_match_file} | head -1 | sort"
    )
    assert recognize_sed_locate_print(command) is None


def test_missing_grep_file_operand(single_match_file: str) -> None:
    command = idiom(single_match_file, "p", "grep -n 'needle' | head -1 | cut -d: -f1")
    assert recognize_sed_locate_print(command) is None


def test_unreadable_grep_source_is_not_recognizable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    command = idiom(
        str(missing), "p", f"grep -n 'needle' {missing} | head -1 | cut -d: -f1"
    )
    assert recognize_sed_locate_print(command) is None


# --- F2/W2: the simulation read is gated, bounded, and binary-refused -----------


def test_peek_source_parses_the_shape_without_reading(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    command = idiom(
        str(missing), "p", f"grep -n 'needle' {missing} | head -1 | cut -d: -f1"
    )
    # The peek is pure parsing: it names the source even though the file does
    # not exist, so the wiring can pre-check the path before any read.
    assert sed_locate_print_source(command) == str(missing)
    assert sed_locate_print_source("sed -n '2p' x.txt") is None
    assert sed_locate_print_source("echo hi") is None
    assert (
        sed_locate_print_source(idiom(str(missing), "p", "grep -n 'x' | cut -d: -f1"))
        is None
    )


def test_fifo_source_is_refused_without_blocking(tmp_path: Path) -> None:
    # A FIFO source would block the synchronous analysis (probed: reading
    # /dev/stdin on a FIFO hangs it); the stat gate refuses it before any
    # open, so the analysis cannot hang.
    fifo = tmp_path / "pipe.txt"
    os.mkfifo(fifo)
    command = idiom(str(fifo), "p", f"grep -n 'needle' {fifo} | head -1 | cut -d: -f1")
    assert recognize_sed_locate_print(command) is None


def test_oversized_source_is_refused_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * (_SIMULATION_READ_CAP_BYTES + 1))
    command = idiom(str(big), "p", f"grep -n 'needle' {big} | head -1 | cut -d: -f1")

    def _no_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("the simulation must not read an oversized source")

    monkeypatch.setattr(Path, "read_text", _no_read)
    monkeypatch.setattr(Path, "open", _no_read)
    assert recognize_sed_locate_print(command) is None


def test_nul_byte_source_is_refused_as_binary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # GNU grep prints "binary file matches" to stderr and nothing on stdout
    # for a binary source, so the runtime pipeline never produces the address
    # this simulation would authorize (W2); the refusal is drift-logged.
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"needle\n\x00binary\n")
    command = idiom(
        str(binary), "p", f"grep -n 'needle' {binary} | head -1 | cut -d: -f1"
    )
    with caplog.at_level(
        "DEBUG", logger="chartreux.core.tools.builtins._shell_static_expansion"
    ):
        assert recognize_sed_locate_print(command) is None
    assert any("NUL byte" in message for message in caplog.messages)


# --- construct (b): equals expansion ------------------------------------------


@pytest.fixture()
def path_dir(tmp_path: Path) -> str:
    directory = tmp_path / "bin"
    directory.mkdir()
    tool = directory / "mytool"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    return str(directory)


def test_equals_resolves_found_command(path_dir: str) -> None:
    assert resolve_equals_command("=mytool", path=[path_dir]) == f"{path_dir}/mytool"


def test_equals_rejects_missing_command(path_dir: str) -> None:
    assert resolve_equals_command("=notatool", path=[path_dir]) is None
    assert resolve_equals_command("=mytool", path=[]) is None


def test_equals_rejects_directories_and_non_executables(tmp_path: Path) -> None:
    directory = tmp_path / "bin"
    directory.mkdir()
    (directory / "adir").mkdir()
    locked = directory / "locked"
    locked.write_text("#!/bin/sh\n")
    locked.chmod(0o644)
    assert resolve_equals_command("=adir", path=[str(directory)]) is None
    assert resolve_equals_command("=locked", path=[str(directory)]) is None


def test_equals_word_grammar() -> None:
    assert resolve_equals_command("=ls") is not None
    assert resolve_equals_command("=definitely-not-a-real-command-xyz") is None
    # Probed: exactly one '=' is stripped — '==ls' looks up '=ls' and fails.
    assert resolve_equals_command("==ls", path=["/usr/bin"]) is None
    # Probed: words containing '/' never resolve.
    assert resolve_equals_command("=./foo") is None
    assert resolve_equals_command("=/bin/sh") is None
    assert resolve_equals_command("=") is None
    assert resolve_equals_command("ls") is None
    assert resolve_equals_command("=with space", path=[]) is None
    assert resolve_equals_command("=-dashed", path=[]) is None


def test_equals_resolution_never_executes(path_dir: str) -> None:
    """Runs under the autouse execution ban; a stat-only resolution passes."""
    resolved = resolve_equals_command("=mytool", path=[path_dir])
    assert resolved is not None
    assert os.stat(resolved).st_mode & stat.S_IXUSR


# --- construct (c): static expansion evaluation -------------------------------


def test_plain_literal_resolves_to_itself() -> None:
    assert evaluate_static_expansion("head", {}) == "head"
    assert evaluate_static_expansion("", {}) == ""
    assert evaluate_static_expansion("a.b/c-d", {}) == "a.b/c-d"


def test_variable_expansion_forms() -> None:
    variables = {"N": "5"}
    assert evaluate_static_expansion("$N", variables) == "5"
    assert evaluate_static_expansion("${N}", variables) == "5"
    assert evaluate_static_expansion("x${N}y", variables) == "x5y"
    assert evaluate_static_expansion("-n$N", variables) == "-n5"
    assert evaluate_static_expansion("$((N))p", variables) == "5p"


def test_arithmetic_evaluation() -> None:
    variables = {"N": "5"}
    assert evaluate_static_expansion("$((1+2))", variables) == "3"
    assert evaluate_static_expansion("$((N+1))", variables) == "6"
    assert evaluate_static_expansion("$(( (N+1)*2 ))", variables) == "12"
    assert evaluate_static_expansion("$((5/2))", variables) == "2"
    # zsh/C truncating division and remainder, as probed.
    assert evaluate_static_expansion("$((0-7))", variables) == "-7"
    assert evaluate_static_expansion("$((-7/2))", variables) == "-3"
    assert evaluate_static_expansion("$((5%3))", variables) == "2"


def test_arithmetic_rejects_unknown_and_malformed() -> None:
    assert evaluate_static_expansion("$((M+1))", {"N": "5"}) is None
    assert evaluate_static_expansion("$((N+1))", {"N": "x"}) is None
    assert evaluate_static_expansion("$((1/0))", {}) is None
    assert evaluate_static_expansion("$((1%0))", {}) is None
    assert evaluate_static_expansion("$((1+))", {}) is None
    assert evaluate_static_expansion("$(( ))", {}) is None
    assert evaluate_static_expansion("$((N", {"N": "1"}) is None
    # A well-formed $((...)) followed by unmodeled literal text still
    # resolves the arithmetic part faithfully.
    assert evaluate_static_expansion("$((1))x)", {}) == "1x)"


def test_unknown_variables_are_not_resolvable() -> None:
    assert evaluate_static_expansion("$UNSET", {}) is None
    assert evaluate_static_expansion("${UNSET}", {}) is None


def test_command_substitution_is_never_resolvable() -> None:
    assert evaluate_static_expansion("$(echo hi)", {}) is None
    assert evaluate_static_expansion("`id`", {}) is None
    assert evaluate_static_expansion("$(( $(x) ))", {}) is None
    assert evaluate_static_expansion("$(( $((1+1)) ))", {}) is None
    assert evaluate_static_expansion("${X:-y}", {"X": "1"}) is None
    assert evaluate_static_expansion("${X#y}", {"X": "1"}) is None


def test_special_dollar_forms_are_not_resolvable() -> None:
    for expression in ("$?", "$$", "$1", "$@", "$'lit'", "$[1+1]", "$"):
        assert evaluate_static_expansion(expression, {}) is None


# --- F1: zsh modifier/subscript suffixes after an expansion --------------------


@pytest.mark.parametrize(
    "expression",
    [
        # A bare `$NAME` followed by `:` is a zsh history modifier and
        # followed by `[` a subscript (probed): the resolved word would
        # diverge from what the shell runs.
        "$X:s,a,../x,",
        "$X:gs,a,b,",
        "$X:h",
        "$X:t",
        "$X:r",
        "$X:e",
        "$X:q:s,a,b,",
        "$X:q",
        "$X:u",
        "$X:l",
        "$X:A",
        "$X:a",
        "$X[1]",
        "$X[1,2]",
        # Defense in depth: the braced and arithmetic forms treat the suffix
        # as literal text in zsh (probed), but it is rejected anyway.
        "${X}:h",
        "${X}[1]",
        "$((N)):h",
        "$((N))[1]",
    ],
)
def test_expansion_suffix_modifiers_are_not_resolvable(expression: str) -> None:
    assert evaluate_static_expansion(expression, {"X": "a", "N": "5"}) is None


def test_expansion_suffix_controls_stay_resolvable() -> None:
    # Only `:` and `[` are re-interpreted after a bare `$NAME` (probed);
    # literal suffixes and further expansions resolve faithfully.
    assert evaluate_static_expansion("$X", {"X": "a"}) == "a"
    assert evaluate_static_expansion("$X.txt", {"X": "a"}) == "a.txt"
    assert evaluate_static_expansion("$X-suffix", {"X": "a"}) == "a-suffix"
    assert evaluate_static_expansion("$X$Y", {"X": "a", "Y": "b"}) == "ab"
    assert evaluate_static_expansion("x${N}y", {"N": "5"}) == "x5y"
    assert evaluate_static_expansion("$((N))p", {"N": "5"}) == "5p"


def test_conservative_variable_value_policy() -> None:
    for bad in (
        "1; rm",
        "$X",
        ">out",
        "=ls",
        "~home",
        "a|b",
        "x'y",
        'x"y',
        "a\\b",
        "a\nb",
        # Whitespace is forbidden too: the one-argv-element guarantee is
        # only sound for splitting-free values (H1).
        "a b",
        "a\tb",
        "a\rb",
        "a\vb",
        "a\fb",
    ):
        assert evaluate_static_expansion("$V", {"V": bad}) is None
    assert evaluate_static_expansion("$V", {"V": "-5"}) == "-5"
    # An empty value still resolves at this layer; the caller-facing
    # resolver refuses it (H1: zsh drops an empty unquoted word entirely).
    assert evaluate_static_expansion("$V", {"V": ""}) == ""


def test_empty_resolution_is_refused_for_a_position() -> None:
    # zsh drops an empty unquoted word entirely, so authorizing it would
    # describe a different argv than the runtime runs (H1).
    assert (
        evaluate_expansion_for_position("cat", "positional:1", "$V", {"V": ""}) is None
    )
    assert (
        evaluate_expansion_for_position("cat", "positional:1", "$V$V", {"V": ""})
        is None
    )
    # A non-empty concatenation around an empty value stays resolvable.
    assert (
        evaluate_expansion_for_position("cat", "positional:1", "$V.txt", {"V": ""})
        == ".txt"
    )


# --- argument-position table (module 2) --------------------------------------


def test_table_entries_are_well_formed() -> None:
    for command, table in ARGUMENT_POSITION_RULES.items():
        for position, rule in table.items():
            assert rule.command == command
            assert rule.position == position
            assert rule.rationale


def test_read_only_positions_accept_resolvable_literals() -> None:
    assert (
        evaluate_expansion_for_position("head", "option-arg:-n", "$((N+1))", {"N": "5"})
        == "6"
    )
    assert (
        evaluate_expansion_for_position("tail", "option-arg:-n", "$N", {"N": "3"})
        == "3"
    )
    assert (
        evaluate_expansion_for_position("cut", "option-arg:-d", "$D", {"D": ":"}) == ":"
    )
    assert (
        evaluate_expansion_for_position("grep", "positional:1", "$P", {"P": "needle"})
        == "needle"
    )
    assert (
        evaluate_expansion_for_position("head", "positional:1", "$F", {"F": "x.txt"})
        == "x.txt"
    )


def test_positional_open_ranges_cover_later_indices() -> None:
    assert is_read_only_argument_position("grep", "positional:3")
    rule = classify_argument_position("grep", "positional:3")
    assert rule is not None
    assert rule.position == "positional:2+"
    assert is_read_only_argument_position("head", "positional:2")
    assert (
        evaluate_expansion_for_position("grep", "positional:5", "$F", {"F": "x.txt"})
        == "x.txt"
    )


def test_write_positions_reject_everything() -> None:
    for position in (
        "positional:1",
        "option-arg:-e",
        "option-arg:--expression",
        "option-arg:-f",
        "option-arg:--file",
        "positional:2",
    ):
        assert not is_read_only_argument_position("sed", position)
        # Write positions reject even plain literals and valid arithmetic.
        assert evaluate_expansion_for_position("sed", position, "5", {}) is None
        assert (
            evaluate_expansion_for_position("sed", position, "$((N+1))", {"N": "5"})
            is None
        )


def test_unknown_commands_and_positions_fail_closed() -> None:
    assert classify_argument_position("sort", "positional:1") is None
    assert not is_read_only_argument_position("sort", "positional:1")
    assert classify_argument_position("head", "option-arg:--unset") is None
    assert not is_read_only_argument_position("head", "positional:0")
    assert evaluate_expansion_for_position("tee", "positional:1", "x", {}) is None
    assert evaluate_expansion_for_position("head", "made-up:1", "x", {}) is None


def test_read_only_positions_listing() -> None:
    assert read_only_positions("sed") == frozenset()
    assert "option-arg:-n" in read_only_positions("head")
    assert "positional:1" in read_only_positions("grep")
    assert "positional:1+" in read_only_positions("cat")
    assert read_only_positions("unknown-cmd") == frozenset()


# --- the group execution-impossibility proof -----------------------------------


def test_all_recognition_paths_run_under_the_execution_ban(
    single_match_file: str, multi_match_file: str, path_dir: str
) -> None:
    """Every modeled construct, every path, with all executors banned."""
    recognized = [
        idiom(
            single_match_file,
            ",+45p",
            f"grep -n 'needle' {single_match_file} | head -1 | cut -d: -f1",
        ),
        idiom(
            multi_match_file,
            "p",
            f"grep -n 'needle' {multi_match_file} | tail -1 | cut -d: -f1",
        ),
    ]
    rejected = [
        idiom(single_match_file, "p", f"grep -n 'zzz' {single_match_file} | head -1"),
        f'sed -n "$(( $(echo 2) ))p" {single_match_file}',
        "sed -n \"$(grep -n 'x' /nope | cut -d: -f1)p\" /nope",
    ]
    for command in recognized:
        assert recognize_sed_locate_print(command) is not None
    for command in rejected:
        assert recognize_sed_locate_print(command) is None
    assert resolve_equals_command("=mytool", path=[path_dir]) is not None
    assert resolve_equals_command("=missing-tool", path=[path_dir]) is None
    assert evaluate_static_expansion("$((N+1))", {"N": "5"}) == "6"
    assert (
        evaluate_expansion_for_position("head", "option-arg:-n", "$((N+1))", {"N": "5"})
        == "6"
    )
