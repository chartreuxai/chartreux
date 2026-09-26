from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    analyze_shell_command_policy,
    git_repository_config_risk,
    inline_interpreter_switch,
    matches_command_prefix,
    path_candidates,
)
from chartreux.core.tools.builtins.bash import Bash, BashToolConfig


def test_inline_interpreter_switch_scans_only_real_options() -> None:
    assert (
        inline_interpreter_switch(["/usr/bin/python3", "-B", "-uc", "print(1)"])
        == "python3 -c"
    )
    assert inline_interpreter_switch(["node", "--eval=1"]) == "node -e"
    assert (
        inline_interpreter_switch(["node", "--no-warnings", "--eval", "1"]) == "node -e"
    )
    assert inline_interpreter_switch(["ruby", "-we", "puts 1"]) == "ruby -e"
    assert inline_interpreter_switch(["python3", "script.py", "-c"]) is None
    assert inline_interpreter_switch(["python3", "--", "-c"]) is None
    assert inline_interpreter_switch(["echo", "curl"]) is None


@pytest.mark.parametrize(
    "tokens",
    [
        ["python3", "-W", "ignore", "-c", "print(1)"],
        ["python", "-X", "dev", "-c", "print(1)"],
        ["pypy", "-Wignore", "-c", "print(1)"],
        ["pypy3", "-Xdev", "-c", "print(1)"],
        ["node", "--require=module", "--eval=1"],
        ["node", "-r", "module", "-e", "1"],
        ["perl", "-I", "lib", "-e", "1"],
        ["ruby", "-I", "lib", "-e", "puts 1"],
    ],
)
def test_inline_interpreter_options_consume_values(tokens: list[str]) -> None:
    assert (
        inline_interpreter_switch(tokens)
        == f"{tokens[0]} -{'c' if tokens[0].startswith(('python', 'pypy')) else 'e'}"
    )


@pytest.mark.parametrize(
    "tokens",
    [
        ["python", "-u", "script.py", "-c", "arg"],
        ["python", "-m", "module", "-c", "argument"],
        ["python", "-W", "ignore::DeprecationWarning", "script.py"],
        ["python3", "-Xdev", "script.py", "-c", "argument"],
        ["node", "--require=module", "script.js", "--eval=1"],
        ["perl", "-Ilib", "script.pl", "-e", "argument"],
        ["ruby", "-Ilib", "script.rb", "-e", "argument"],
    ],
)
def test_inline_interpreter_options_stop_at_target(tokens: list[str]) -> None:
    assert inline_interpreter_switch(tokens) is None


def test_unknown_interpreter_option_before_inline_code_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported python option"):
        inline_interpreter_switch(["python", "--unknown=value", "-c", "print(1)"])


def test_prefix_match_accepts_extra_arguments() -> None:
    assert matches_command_prefix(["git", "status", "--short"], ["git", "status"])


def test_prefix_match_accepts_exact_command() -> None:
    assert matches_command_prefix(["git", "status"], ["git", "status"])


def test_prefix_match_normalizes_executable_basename_in_command() -> None:
    assert matches_command_prefix(["/usr/bin/git", "status"], ["git", "status"])
    assert matches_command_prefix(["./git", "status"], ["git", "status"])


def test_prefix_match_normalizes_executable_basename_in_pattern() -> None:
    assert matches_command_prefix(["gdb", "-x"], ["/usr/bin/gdb"])


def test_prefix_match_does_not_normalize_argument_tokens() -> None:
    # Only the executable token is basename-normalized; arguments compare literally.
    assert not matches_command_prefix(["git", "/usr/bin/status"], ["git", "status"])


def test_prefix_match_requires_exact_argument_tokens() -> None:
    # Argument tokens must match exactly: "statusx" is not the argument "status".
    assert not matches_command_prefix(["git", "statusx"], ["git", "status"])


def test_prefix_match_rejects_different_executable() -> None:
    assert not matches_command_prefix(["got", "status"], ["git", "status"])
    assert not matches_command_prefix(["gdbx"], ["gdb"])


def test_prefix_match_rejects_command_shorter_than_pattern() -> None:
    assert not matches_command_prefix(["git"], ["git", "status"])


def test_prefix_match_rejects_empty_tokens_or_pattern() -> None:
    assert not matches_command_prefix([], ["x"])
    assert not matches_command_prefix(["x"], [])


def test_prefix_match_short_option_matches_combined_short_options() -> None:
    # getopt parses "-ic" as "-i -c", so the pattern "bash -i" must catch the
    # equivalent combined form of the interactive shell.
    assert matches_command_prefix(["bash", "-ic", "x"], ["bash", "-i"])
    assert matches_command_prefix(["bash", "-ci"], ["bash", "-i"])


def test_prefix_match_short_option_does_not_match_long_options() -> None:
    assert not matches_command_prefix(["bash", "--interactive"], ["bash", "-i"])
    assert not matches_command_prefix(["cmd", "--ignore"], ["cmd", "-i"])


def test_prefix_match_short_option_does_not_match_plain_words() -> None:
    assert not matches_command_prefix(["bash", "interactive"], ["bash", "-i"])
    assert not matches_command_prefix(["bash", "i"], ["bash", "-i"])


def test_prefix_match_combined_flags_outside_any_pattern_pass_through() -> None:
    # Benign combined flags that appear in no denylist pattern are unaffected.
    assert not matches_command_prefix(["ls", "-la"], ["bash", "-i"])
    assert not matches_command_prefix(["grep", "-in", "x"], ["bash", "-i"])


def test_single_token_pattern_matches_command_with_arguments() -> None:
    # A bare-name denylist pattern is a prefix, so it denies every invocation.
    assert matches_command_prefix(["gdb", "-x", "file"], ["gdb"])
    assert matches_command_prefix(["gdb"], ["gdb"])


@pytest.fixture
def bash_tool(tmp_path, monkeypatch) -> Bash:
    # Explicit entries only: the default denylists are extended elsewhere and
    # must not be pinned by these tests.
    monkeypatch.chdir(tmp_path)
    config = BashToolConfig(denylist=["bash -i"], denylist_standalone=["python"])
    return Bash(config_getter=lambda: config, state=BaseToolState())


def test_with_args_denylist_denies_matching_prefix(bash_tool) -> None:
    assert bash_tool._find_denylist_match("bash -i") == "bash -i"
    assert bash_tool._find_denylist_match("bash -i -x") == "bash -i"


def test_with_args_denylist_normalizes_executable_basename(bash_tool) -> None:
    assert bash_tool._find_denylist_match("/usr/bin/bash -i") == "bash -i"


def test_with_args_denylist_denies_combined_short_options(bash_tool) -> None:
    # "bash -ic x" is the interactive shell the "bash -i" pattern blocks.
    assert bash_tool._find_denylist_match("bash -ic x") == "bash -i"


def test_with_args_denylist_allows_benign_combined_flags(bash_tool) -> None:
    assert bash_tool._find_denylist_match("ls -la") is None
    assert bash_tool._find_denylist_match("grep -in pattern file") is None


def test_with_args_denylist_does_not_deny_bare_command(bash_tool) -> None:
    # "bash -i" is a prefix pattern: bare "bash" does not carry the -i argument.
    assert bash_tool._find_denylist_match("bash") is None


def test_standalone_denylist_denies_bare_command(bash_tool) -> None:
    assert bash_tool._is_standalone_denylisted("python")


def test_standalone_denylist_normalizes_executable_basename(bash_tool) -> None:
    assert bash_tool._is_standalone_denylisted("/usr/bin/python")


def test_standalone_denylist_allows_command_with_arguments(bash_tool) -> None:
    assert not bash_tool._is_standalone_denylisted("python script.py")


def test_standalone_denylist_matches_exact_names_only(bash_tool) -> None:
    # Standalone entries are exact names, not prefixes: python3 is a different binary.
    assert not bash_tool._is_standalone_denylisted("python3")


def test_guardrail_denies_with_args_denylist_match(bash_tool) -> None:
    context = bash_tool._resolve_guardrail_permission(["/usr/bin/bash -i"])

    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "matches denylist pattern" in context.reason


def test_guardrail_denies_standalone_denylist_match(bash_tool) -> None:
    context = bash_tool._resolve_guardrail_permission(["python"])

    assert context is not None
    assert context.permission is ToolPermission.NEVER
    assert "not allowed as a standalone command" in context.reason


def test_guardrail_allows_non_denylisted_commands(bash_tool) -> None:
    assert bash_tool._resolve_guardrail_permission(["echo hi"]) is None
    assert bash_tool._resolve_guardrail_permission(["python script.py"]) is None


def test_rm_plain_file_needs_no_approval() -> None:
    assert not analyze_shell_command_policy(["rm", "file"]).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["rm", "-r", "dir"],
        ["rm", "-R", "dir"],
        ["rm", "-rf", "/"],
        ["rm", "-fr", "dir"],
        ["rm", "--recursive", "dir"],
        ["rm", "--rec", "dir"],
    ],
)
def test_rm_recursive_flags_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


def test_rm_double_dash_stops_option_scanning() -> None:
    # After "--", "-r" is a filename, not a recursive flag.
    assert not analyze_shell_command_policy(["rm", "--", "-r"]).requires_approval


def test_rm_option_after_positional_still_requires_approval() -> None:
    # GNU option permutation: "dir" does not end option scanning.
    assert analyze_shell_command_policy(["rm", "dir", "--recursive"]).requires_approval


def test_chmod_non_recursive_permissive_mode_needs_no_approval() -> None:
    assert not analyze_shell_command_policy(["chmod", "777", "dir"]).requires_approval


@pytest.mark.parametrize(
    "mode", ["777", "0007", "00007", "666", "7", "77", "07", "6", "2"]
)
def test_chmod_recursive_permissive_numeric_mode_requires_approval(mode: str) -> None:
    # GNU chmod zero-pads octal modes on the left (any number of digits up to
    # its 07777 value ceiling), so the last digit is always the "others"
    # class: "chmod -R 00007 dir" is mode 0007.
    assert analyze_shell_command_policy(["chmod", "-R", mode, "dir"]).requires_approval


def test_chmod_recursive_too_large_numeric_mode_needs_no_approval() -> None:
    # Beyond GNU's 07777 ceiling chmod fails at runtime ("mode too large");
    # the policy treats it like any other non-permissive mode.
    assert not analyze_shell_command_policy([
        "chmod",
        "-R",
        "10000",
        "f",
    ]).requires_approval


@pytest.mark.parametrize("mode", ["755", "444", "664", "4", "44", "5", "644"])
def test_chmod_recursive_read_only_numeric_mode_needs_no_approval(mode: str) -> None:
    assert not analyze_shell_command_policy([
        "chmod",
        "-R",
        mode,
        "dir",
    ]).requires_approval


@pytest.mark.parametrize("mode", ["a+w", "o+w", "=rw", "u+w,o+w"])
def test_chmod_recursive_permissive_symbolic_mode_requires_approval(mode: str) -> None:
    assert analyze_shell_command_policy(["chmod", "-R", mode, "dir"]).requires_approval


@pytest.mark.parametrize("mode", ["u+w", "g+w", "ug+w"])
def test_chmod_recursive_owner_or_group_symbolic_mode_needs_no_approval(
    mode: str,
) -> None:
    assert not analyze_shell_command_policy([
        "chmod",
        "-R",
        mode,
        "dir",
    ]).requires_approval


def test_chown_non_recursive_needs_no_approval() -> None:
    assert not analyze_shell_command_policy(["chown", "user", "file"]).requires_approval


def test_chown_recursive_requires_approval() -> None:
    assert analyze_shell_command_policy([
        "chown",
        "-R",
        "user",
        "dir",
    ]).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["find", ".", "-delete"],
        ["find", ".", "-exec", "rm", "{}", ";"],
        ["find", ".", "-execdir", "ls"],
        ["find", ".", "-fprint", "/tmp/out"],
        ["find", ".", "-ok", "rm", "{}", ";"],
    ],
)
def test_find_side_effect_predicates_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


def test_find_read_only_predicates_need_no_approval() -> None:
    assert not analyze_shell_command_policy([
        "find",
        ".",
        "-name",
        "x",
    ]).requires_approval
    assert not analyze_shell_command_policy([
        "find",
        ".",
        "-printf",
        "y",
    ]).requires_approval


def test_find_value_predicate_consumes_its_argument() -> None:
    # "-delete" is the pattern argument of -name, not a side-effect predicate.
    assert not analyze_shell_command_policy([
        "find",
        ".",
        "-name",
        "-delete",
    ]).requires_approval
    assert analyze_shell_command_policy([
        "find",
        ".",
        "-name",
        "x",
        "-delete",
    ]).requires_approval


def test_find_newer_variants_consume_their_argument() -> None:
    assert analyze_shell_command_policy([
        "find",
        ".",
        "-newermt",
        "x",
        "-delete",
    ]).requires_approval


def test_find_leading_option_tokens_are_skipped() -> None:
    assert analyze_shell_command_policy([
        "find",
        "-L",
        ".",
        "-name",
        "x",
        "-delete",
    ]).requires_approval
    assert analyze_shell_command_policy([
        "find",
        "-O3",
        ".",
        "-delete",
    ]).requires_approval


def test_git_reset_hard_requires_approval() -> None:
    assert analyze_shell_command_policy(["git", "reset", "--hard"]).requires_approval


def test_git_reset_soft_needs_no_approval() -> None:
    assert not analyze_shell_command_policy([
        "git",
        "reset",
        "--soft",
    ]).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "clean", "-f"],
        ["git", "clean", "-fd"],
        ["git", "clean", "-fdx"],
        ["git", "clean", "-f", "-e", "pattern"],
    ],
)
def test_git_clean_force_requires_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "clean", "-n"],
        ["git", "clean", "-n", "-f"],
        ["git", "clean", "-ndx"],
        ["git", "clean", "--dry-run", "-f"],
    ],
)
def test_git_clean_dry_run_needs_no_approval(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).requires_approval


def test_git_global_options_precede_subcommand() -> None:
    assert analyze_shell_command_policy([
        "git",
        "-C",
        "/repo",
        "reset",
        "--hard",
    ]).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "diff", "--output", "/tmp/x"],
        ["git", "diff", "--output=/tmp/x"],
        ["git", "show", "--ext-diff"],
        ["git", "show", "--textconv"],
    ],
)
def test_git_diff_output_and_external_tools_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "status"],
        ["git", "log"],
        ["git", "--no-pager", "diff"],
        ["git", "diff", "--no-ext-diff"],
        ["git", "--help"],
        ["git", "--version"],
    ],
)
def test_git_read_only_commands_need_no_approval(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "-c", "core.editor=x", "status"],
        ["git", "--config-env", "x", "status"],
        ["git", "--unknown-global", "status"],
    ],
)
def test_git_config_override_and_unknown_global_fail_closed(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


def test_git_dir_values_are_collected_as_paths() -> None:
    policy = analyze_shell_command_policy(["git", "--git-dir", "/x", "log"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("/x",)

    attached = analyze_shell_command_policy(["git", "--git-dir=/x", "log"])
    assert attached.option_path_values == ("/x",)


def test_git_diff_no_index_inspects_positional_paths() -> None:
    policy = analyze_shell_command_policy(["git", "diff", "--no-index", "a", "b"])
    assert not policy.requires_approval
    assert policy.inspect_positional_paths
    assert policy.positional_values == ("a", "b")


def test_xargs_always_requires_approval() -> None:
    assert analyze_shell_command_policy(["xargs", "echo"]).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["sort", "-o", "out", "in"],
        ["sort", "--output=out", "in"],
        ["sort", "--compress-program", "gzip"],
        ["sort", "-T", "dir", "in"],
        ["sort", "--files0-from", "list"],
        ["sort", "--temporary-directory", "dir"],
    ],
)
def test_sort_side_effecting_options_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


def test_sort_random_source_is_a_path_value_not_an_approval() -> None:
    policy = analyze_shell_command_policy(["sort", "--random-source", "f", "in"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("f",)


@pytest.mark.parametrize(
    "tokens", [["date", "-s", "2020-01-01"], ["date", "--set", "2020-01-01"]]
)
def test_date_set_requires_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


def test_date_reference_options_are_path_values() -> None:
    policy = analyze_shell_command_policy(["date", "-f", "stampfile"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("stampfile",)

    reference = analyze_shell_command_policy(["date", "-r", "stampfile"])
    assert reference.option_path_values == ("stampfile",)


@pytest.mark.parametrize(
    "tokens",
    [
        ["less", "-o", "log", "file"],
        ["less", "--log-file", "x"],
        ["less", "--LOG-FILE", "x"],
    ],
)
def test_less_log_file_options_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize("tokens", [["tree", "-o", "out"], ["tree", "--output", "out"]])
def test_tree_output_requires_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["tree", "-Ho", "host", "out"],
        ["tree", "-Io", "pat", "out"],
        ["tree", "-Lo", "2", "out"],
        ["tree", "-To", "title", "out"],
        ["tree", "-Po", "pat", "out"],
    ],
)
def test_tree_cluster_output_file_requires_approval(tokens: list[str]) -> None:
    # tree takes a value-taking short's operand from the next argv token and
    # resumes scanning the same cluster, so the "o" in each cluster is the
    # output-file option plain `tree -o` already gates.
    assert analyze_shell_command_policy(tokens).requires_approval


def test_tree_inspects_positional_paths_by_default() -> None:
    policy = analyze_shell_command_policy(["tree", "."])
    assert not policy.requires_approval
    assert policy.inspect_positional_paths
    assert policy.positional_values == (".",)


def test_grep_file_option_value_is_collected() -> None:
    policy = analyze_shell_command_policy(["grep", "-f", "patterns", "file"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("patterns",)

    attached = analyze_shell_command_policy(["grep", "--file=patterns", "x"])
    assert attached.option_path_values == ("patterns",)


@pytest.mark.parametrize(
    "tokens", [["du", "--files0-from", "list"], ["wc", "--files0-from", "list"]]
)
def test_files0_from_option_gates_and_names_its_list_file(tokens: list[str]) -> None:
    # The list file's own path is nominated, but the paths it lists come from
    # file content, which static path analysis cannot see.
    policy = analyze_shell_command_policy(tokens)
    assert policy.requires_approval
    assert policy.option_path_values == ("list",)


def test_file_files_from_splits_colon_separated_lists() -> None:
    policy = analyze_shell_command_policy(["file", "--files-from", "a:b", "x"])
    assert policy.requires_approval
    assert policy.option_path_values == ("a", "b")


def test_diff_from_file_is_a_path_value() -> None:
    policy = analyze_shell_command_policy(["diff", "--from-file", "a", "b"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("a",)


def test_policy_dispatch_normalizes_executable_basename() -> None:
    assert analyze_shell_command_policy(["/bin/rm", "-r", "dir"]).requires_approval


@pytest.mark.parametrize("tokens", [["echo", "hi"], ["frobnicate", "--dangerous"], []])
def test_unknown_commands_have_no_policy(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).requires_approval


def test_path_candidates_include_option_values_without_inspect() -> None:
    assert path_candidates(
        ["grep", "-f", "patterns", "file"], inspect_positional_paths=False
    ) == ("patterns",)


def test_path_candidates_add_positionals_when_inspecting() -> None:
    assert path_candidates(
        ["grep", "-f", "patterns", "file"], inspect_positional_paths=True
    ) == ("patterns", "file")
    assert path_candidates(["cat", "-n", "a"], inspect_positional_paths=True) == ("a",)


def test_path_candidates_treat_tokens_after_double_dash_as_positional() -> None:
    assert path_candidates(["cat", "--", "-n"], inspect_positional_paths=True) == (
        "-n",
    )


def test_path_candidates_skip_chmod_mode_tokens() -> None:
    assert path_candidates(["chmod", "+x", "file"], inspect_positional_paths=True) == (
        "file",
    )


def test_path_candidates_exclude_sort_output_file() -> None:
    # "out" is consumed by -o; only the input file is a path candidate.
    assert path_candidates(
        ["sort", "-o", "out", "in"], inspect_positional_paths=True
    ) == ("in",)


def test_path_candidates_inspect_tree_positionals_via_policy() -> None:
    # tree declares inspect_positional_paths itself, so callers need not opt in.
    assert path_candidates(["tree", "."], inspect_positional_paths=False) == (".",)


def test_path_candidates_include_git_no_index_positionals() -> None:
    assert path_candidates(
        ["git", "diff", "--no-index", "a", "b"], inspect_positional_paths=False
    ) == ("a", "b")


def test_path_candidates_include_git_global_path_values() -> None:
    assert path_candidates(
        ["git", "--git-dir", "/repo", "log"], inspect_positional_paths=False
    ) == ("/repo",)


def test_path_candidates_empty_tokens() -> None:
    assert path_candidates([], inspect_positional_paths=True) == ()


@pytest.mark.parametrize(
    "tokens",
    [
        ["git", "diff"],
        ["git", "log"],
        ["git", "show"],
        ["git", "status"],
        ["git", "blame"],
        ["git", "whatchanged"],
    ],
)
def test_git_readers_inspect_the_repository(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).inspect_git_repository


@pytest.mark.parametrize(
    "tokens", [["git", "commit", "-m", "x"], ["git", "init"], ["git"], ["cat", "file"]]
)
def test_non_readers_do_not_inspect_the_repository(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).inspect_git_repository


def _repository_with_config(root: Path, config: str) -> Path:
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(config)
    return root


def test_git_repository_config_risk_names_the_active_vector(tmp_path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n\tpager = ./pager\n")
    risk = git_repository_config_risk(["git", "log"], cwd=root)
    assert risk is not None
    assert risk.endswith("via core.pager")


def test_git_repository_config_risk_is_none_for_clean_repositories(tmp_path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n\trepositoryformatversion = 0\n")
    assert git_repository_config_risk(["git", "status"], cwd=root) is None


def test_git_repository_config_risk_ignores_inactive_values(tmp_path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n\tpager = false\n")
    assert git_repository_config_risk(["git", "log"], cwd=root) is None


def test_git_repository_config_risk_fails_closed_on_unreadable_config(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").mkdir()
    risk = git_repository_config_risk(["git", "log"], cwd=tmp_path)
    assert risk is not None
    assert "unreadable" in risk


def test_git_repository_config_risk_is_none_outside_any_repository(tmp_path) -> None:
    assert git_repository_config_risk(["git", "log"], cwd=tmp_path) is None


def test_git_repository_config_risk_reads_worktree_pointer(tmp_path) -> None:
    git_dir = tmp_path / "real-git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n\tpager = ./pager\n")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {git_dir}")
    risk = git_repository_config_risk(["git", "log"], cwd=worktree)
    assert risk is not None
    assert risk.endswith("via core.pager")


def test_git_repository_config_risk_treats_empty_config_value_as_unset(
    tmp_path,
) -> None:
    # `core.pager =` (empty value) is FALSE to git, not the implicit true of
    # the valueless bare-key form.
    root = _repository_with_config(tmp_path, "[core]\n\tpager =\n")
    assert git_repository_config_risk(["git", "log"], cwd=root) is None


def test_git_repository_config_risk_treats_bare_config_key_as_true(tmp_path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n\tpager\n")
    risk = git_repository_config_risk(["git", "log"], cwd=root)
    assert risk is not None
    assert risk.endswith("via core.pager")


def test_git_repository_config_risk_fails_closed_on_unreadable_commondir(
    tmp_path,
) -> None:
    # A directory where git expects the commondir file makes the read fail;
    # falling back to the worktree gitdir's own (nonexistent) config would
    # hide the active core.pager in the shared config.
    worktree_git_dir = tmp_path / "real-git" / "worktrees" / "wt"
    worktree_git_dir.mkdir(parents=True)
    (tmp_path / "real-git" / "config").write_text("[core]\n\tpager = ./pager\n")
    (worktree_git_dir / "commondir").mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {worktree_git_dir}")
    risk = git_repository_config_risk(["git", "log"], cwd=worktree)
    assert risk is not None
    assert "unreadable" in risk


def test_git_repository_config_risk_reads_separate_git_dir_config(tmp_path) -> None:
    # A --separate-git-dir checkout has no commondir; its gitdir's own config
    # is the repository config, so the missing file must not fail closed.
    git_dir = tmp_path / "separate-git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n\tpager = ./pager\n")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text(f"gitdir: {git_dir}")
    risk = git_repository_config_risk(["git", "log"], cwd=checkout)
    assert risk is not None
    assert risk.endswith("via core.pager")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            "[include]\n\tpath = ./included\n",
            "via include.path; audit or remove the [include] directive in .git/config",
        ),
        (
            '[includeIf "gitdir:."]\n\tpath = ./included\n',
            "via includeif.path; "
            "audit or remove the [includeIf] directive in .git/config",
        ),
        (
            "[core]\n\tfsmonitor = ./monitor\n",
            "via core.fsmonitor; "
            "audit or remove the core.fsmonitor setting in .git/config",
        ),
        (
            '[filter "lfs"]\n\tclean = ./clean\n',
            "via filter.clean; audit or remove the [filter] command in .git/config",
        ),
        (
            '[filter "lfs"]\n\tprocess = ./filter\n',
            "via filter.process; audit or remove the [filter] command in .git/config",
        ),
    ],
)
def test_git_repository_config_risk_names_remediation_for_broad_vectors(
    tmp_path, config: str, expected: str
) -> None:
    # The resolver is deny-only with no passthrough, so the broadest vectors
    # must tell the reader what to do about them, not just name the key.
    root = _repository_with_config(tmp_path, config)
    risk = git_repository_config_risk(["git", "status"], cwd=root)
    assert risk is not None
    assert risk.endswith(expected)


@pytest.mark.parametrize(
    "tokens",
    [
        ["du", "--files0-from", "list"],
        ["du", "--files0-from=list"],
        ["wc", "--files0-from", "list"],
        ["md5sum", "--check", "sums"],
        ["md5sum", "-c", "sums"],
        ["sha1sum", "-c", "sums"],
        ["sha256sum", "-c", "sums"],
        ["shasum", "-c", "sums"],
        ["uniq", "a", "b"],
        ["uniq", "-f", "1", "a", "b"],
        ["less", "+!curl", "file"],
        ["less", "-k", "keys", "file"],
        ["less", "--lesskey-file", "keys"],
        ["less", "-t", "tag"],
        ["less", "--tag-file", "tags"],
        ["less", "--pattern=foo$-k", "file"],
        ["date", "06010203"],
        ["date", "-f", "%Y", "06010203"],
        ["file", "--files-from", "list"],
        ["file", "-f", "list"],
        ["file", "-z", "archive.gz"],
        ["file", "--compile"],
        ["git", "log", "--remerge-diff"],
        ["git", "log", "--show-signature"],
        ["git", "log", "--help"],
        ["git", "log", "-p", "--diff-merges=remerge"],
        ["git", "log", "--diff-merges", "r"],
    ],
)
def test_ported_policies_require_approval(tokens: list[str]) -> None:
    assert analyze_shell_command_policy(tokens).requires_approval


@pytest.mark.parametrize(
    "tokens",
    [
        ["md5sum", "file.txt"],
        ["sha256sum", "file.txt"],
        ["uniq", "file.txt"],
        ["uniq", "-c", "file.txt"],
        ["uniq", "--skip-fields=1", "a"],
        ["less", "file.txt"],
        ["less", "-N", "file.txt"],
        ["less", "+/pattern", "file.txt"],
        ["less", "+G", "file.txt"],
        ["more", "file.txt"],
        ["date"],
        ["date", "+%Y"],
        ["date", "-u"],
        ["date", "-d", "tomorrow"],
        ["date", "-f", "stampfile"],
        ["file", "file.txt"],
        ["file", "-m", "magic"],
        ["git", "log"],
        ["git", "log", "--diff-merges=first-parent"],
    ],
)
def test_ported_policies_allow_benign_forms(tokens: list[str]) -> None:
    assert not analyze_shell_command_policy(tokens).requires_approval


def test_grep_exclude_from_is_a_path_value() -> None:
    policy = analyze_shell_command_policy([
        "grep",
        "--exclude-from",
        "patterns",
        "file",
    ])
    assert not policy.requires_approval
    assert policy.option_path_values == ("patterns",)


def test_diff_exclude_from_is_a_path_value() -> None:
    policy = analyze_shell_command_policy(["diff", "-X", "patterns", "a", "b"])
    assert not policy.requires_approval
    assert policy.option_path_values == ("patterns",)

    attached = analyze_shell_command_policy([
        "diff",
        "--exclude-from=patterns",
        "a",
        "b",
    ])
    assert attached.option_path_values == ("patterns",)


def test_tree_gitfile_is_a_path_value() -> None:
    policy = analyze_shell_command_policy(["tree", "--gitfile", ".gitignore", "."])
    assert not policy.requires_approval
    assert policy.option_path_values == (".gitignore",)


def test_git_pathspec_from_file_is_a_path_value() -> None:
    policy = analyze_shell_command_policy([
        "git",
        "diff",
        "--pathspec-from-file",
        "list",
    ])
    assert not policy.requires_approval
    assert policy.option_path_values == ("list",)

    short = analyze_shell_command_policy(["git", "log", "-O", "list"])
    assert short.option_path_values == ("list",)


def test_policy_dispatch_normalizes_quoted_command_names() -> None:
    assert analyze_shell_command_policy(["'sort'", "-o", "out", "in"]).requires_approval
    assert analyze_shell_command_policy(["./git", "log"]).inspect_git_repository


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (["tar", "-cf-", "--add-file=.env"], ".env"),
        (["dd", "if=.env"], ".env"),
        (["sed", "-n", "p", ".env"], ".env"),
    ],
)
def test_file_operands_are_inspected(tokens: list[str], expected: str) -> None:
    assert expected in path_candidates(tokens, inspect_positional_paths=True)


def test_less_option_resume_has_bounded_depth() -> None:
    assert analyze_shell_command_policy(["less", "-" + "$" * 2100]).requires_approval


@pytest.mark.parametrize(
    "command", ["branch", "tag", "grep", "reflog", "stash", "shortlog"]
)
def test_paging_git_subcommands_inspect_repository(command: str) -> None:
    assert analyze_shell_command_policy(["git", command]).inspect_git_repository


def test_git_config_same_line_and_unsupported_syntax_fail_closed(
    tmp_path: Path,
) -> None:
    root = _repository_with_config(tmp_path, '[core] fsmonitor = "./monitor"\n')
    assert "core.fsmonitor" in (
        git_repository_config_risk(["git", "status"], cwd=root) or ""
    )
    (root / ".git" / "config").write_text("[core]\n  nonsense ! syntax\n")
    assert "unreadable" in (
        git_repository_config_risk(["git", "status"], cwd=root) or ""
    )


def test_git_config_invalid_utf8_fails_closed(tmp_path: Path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n")
    (root / ".git" / "config").write_bytes(b"[core]\nfsmonitor = \xff\n")
    assert "unreadable" in (
        git_repository_config_risk(["git", "status"], cwd=root) or ""
    )


def test_no_pager_disables_only_pager_config_vector(tmp_path: Path) -> None:
    root = _repository_with_config(tmp_path, "[core]\n pager = ./pager\n")
    assert git_repository_config_risk(["git", "--no-pager", "log"], cwd=root) is None
    assert git_repository_config_risk(["git", "-P", "log"], cwd=root) is None
    assert git_repository_config_risk(["git", "--no-pager", "-p", "log"], cwd=root)
