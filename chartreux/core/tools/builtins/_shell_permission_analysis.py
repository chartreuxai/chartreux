from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
import re

from tree_sitter import Language, Node, Parser
import tree_sitter_bash as tsbash

_SUPPORTED_COMMAND_PARTS = {
    "command_name",
    "number",
    "word",
    "string",
    "raw_string",
    "concatenation",
}

_REDIRECTION_NODES = {"heredoc_redirect", "herestring_redirect"}

# Assignment names that can alter command lookup, loading, startup, or runtime
# behavior. Prefix families are checked separately below.
DANGEROUS_ENV_NAMES = frozenset({
    "BASH_ENV",
    "BAT_PAGER",
    "CDPATH",
    "CORE_PAGER",
    "EDITOR",
    "ENV",
    "GEM_HOME",
    "GEM_PATH",
    "GIT_PAGER",
    "HOME",
    "IFS",
    "JAVA_TOOL_OPTIONS",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LESS",
    "LESSHISTFILE",
    "LESSSECURE",
    "LV",
    "MANPAGER",
    "NODE_OPTIONS",
    "PAGER",
    "PERL5LIB",
    "PERL5OPT",
    "PYTHONBREAKPOINT",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONSTARTUP",
    "PYTHONWARNINGS",
    "RIPGREP_CONFIG_PATH",
    "RUBYLIB",
    "RUBYOPT",
    "SHELL",
    "SYSTEMD_PAGER",
    "VISUAL",
    "ZDOTDIR",
    "PATH",
})
_DANGEROUS_ENV_PREFIXES = ("DYLD_", "GIT_")

# Reasons are noun phrases so they read as a list in the approval prompt.
_DYNAMIC_NODES = {
    "ansi_c_string": "ANSI-C quoted arguments",
    "arithmetic_expansion": "arithmetic expansion",
    "brace_expression": "brace expansion",
    "command_substitution": "command substitution",
    "expansion": "parameter expansion",
    "process_substitution": "process substitution",
    "simple_expansion": "variable expansion (unsupported in v0.1 because it can change arguments)",
}

_COMPOUND_NODES = {
    "case_statement": "case statement",
    "compound_statement": "command group",
    "c_style_for_statement": "for loop",
    "for_statement": "for loop",
    "function_definition": "function definition",
    "if_statement": "if statement",
    "subshell": "subshell",
    "test_command": "test expression",
    # The parser does not emit a regular command node for this builtin.
    "unset_command": "unsetting shell environment variables cannot be safely inspected",
    "while_statement": "while loop",
}


def _zsh_sensitive_word_reason(value: bytes) -> str | None:
    """Name Zsh word expansions that the Bash grammar treats as literals."""
    if value.startswith(b"="):
        return "Zsh equals expansion"
    if b"==" in value:
        return "Zsh magic-equals expansion"
    if value.startswith(b"~") and not (value == b"~" or value.startswith(b"~/")):
        return "shell-specific named-directory expansion"
    if b"***" in value:
        return "Zsh symlink-following glob"
    return None


def _zsh_sensitive_node_reason(node: Node) -> str | None:
    if node.type not in {"command_name", "word"} or node.text is None:
        return None
    return _zsh_sensitive_word_reason(node.text)


def _supported_command_part(node: Node) -> str | None:
    if node.type not in _SUPPORTED_COMMAND_PARTS or node.text is None:
        return None
    if _zsh_sensitive_node_reason(node):
        return None
    return node.text.decode("utf-8")


def _node_approval_reason(node: Node) -> str | None:
    if node.type in _REDIRECTION_NODES:
        return (
            "heredocs are unsupported in v0.1"
            if node.type == "heredoc_redirect"
            else "redirection"
        )
    if reason := _DYNAMIC_NODES.get(node.type):
        return reason
    if reason := _zsh_sensitive_node_reason(node):
        return reason
    if node.type == "concatenation" and any(
        child.type == "word" and child.text in {b"{", b"}"} for child in node.children
    ):
        return "brace expansion"
    if compound := _COMPOUND_NODES.get(node.type):
        return compound
    return "background execution" if node.type == "&" else None


def _descendants(node: Node) -> Iterator[Node]:
    for child in node.children:
        yield child
        yield from _descendants(child)


def _literal_token(node: Node) -> str | None:
    if (
        node.type not in {"command_name", "number", "word", "string", "raw_string"}
        or node.text is None
    ):
        return None
    if any(child.type in _DYNAMIC_NODES for child in _descendants(node)):
        return None
    try:
        import shlex

        values = shlex.split(node.text.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return values[0] if len(values) == 1 else None


def _assignment_reason(node: Node) -> str | None:
    if node.text is None:
        return "an environment assignment that cannot be inspected"
    raw = node.text.decode("utf-8")
    name, separator, _ = raw.partition("=")
    if not separator or re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) is None:
        return "an environment assignment with a non-uppercase name"
    if name in DANGEROUS_ENV_NAMES or name.startswith(_DANGEROUS_ENV_PREFIXES):
        return f"dangerous environment assignment ({name})"
    value = node.child_by_field_name("value")
    if value is None:
        literal_value = ""
    else:
        literal_value = _literal_token(value)
        if literal_value is None:
            return "a non-literal environment assignment"
    if name == "PYTHONPATH":
        from pathlib import PurePosixPath

        paths = literal_value.split(":")
        if any(
            not entry
            or PurePosixPath(entry).is_absolute()
            or entry.startswith("~")
            or ".." in PurePosixPath(entry).parts
            for entry in paths
        ):
            return "dangerous environment assignment (PYTHONPATH)"
    return None


def _redirect_reason(node: Node) -> str | None:
    descriptor = node.child_by_field_name("descriptor")
    destination = node.child_by_field_name("destination")
    operators = [
        child.type for child in node.children if child.type in {">", ">>", ">&"}
    ]
    operator = operators[0] if len(operators) == 1 else None
    descriptor_text = (
        descriptor.text.decode("ascii") if descriptor and descriptor.text else None
    )
    destination_text = _literal_token(destination) if destination is not None else None

    if operator == ">&" and descriptor_text == "2" and destination_text == "1":
        return None
    if operator not in {">", ">>"} or descriptor_text not in {None, "2"}:
        return "unsupported redirection (only >, >>, 2>, and 2>&1 are allowed)"
    if destination_text is None:
        return "a non-literal redirection target"
    if destination_text.startswith("-") or destination_text.isdigit():
        return "an invalid redirection path target"
    from pathlib import PurePosixPath

    if PurePosixPath(destination_text).is_absolute():
        return "an absolute redirection target"
    return None


def _executable_parts(node: Node) -> list[str]:
    if node.type != "command":
        return []
    parts = [
        token for child in node.children if (token := _literal_token(child)) is not None
    ]
    # Unwrap each executable position, including repeated command/builtin
    # prefixes. Arguments of the final executable are never reinterpreted.
    while parts and (command := parts[0].rsplit("/", 1)[-1]) in {
        "command",
        "builtin",
        "exec",
    }:
        index = 1
        while index < len(parts):
            if parts[index] == "--":
                index += 1
                break
            if command == "exec" and parts[index] == "-a":
                index += 2
                continue
            if parts[index].startswith("-"):
                index += 1
                continue
            break
        parts = parts[index:]
    return parts


def _command_name(node: Node) -> str | None:
    parts = _executable_parts(node)
    return parts[0].rsplit("/", 1)[-1] if parts else None


_COMMAND_LOOKUP_MUTATORS = frozenset({
    "hash",
    "alias",
    "unalias",
    "enable",
    "shopt",
    "set",
    "source",
    ".",
})


def _lookup_mutation_reason(node: Node, nested_source: bool) -> str | None:
    parts = _executable_parts(node)
    if not parts:
        return None
    name = parts[0].rsplit("/", 1)[-1]
    if name in {"export", "declare", "typeset", "readonly"}:
        assignments = parts[1:] + [
            child.text.decode("utf-8")
            for child in node.children
            if child.type == "concatenation" and child.text is not None
        ]
        for arg in assignments:
            variable, separator, _ = arg.partition("=")
            if separator and (
                variable in DANGEROUS_ENV_NAMES
                or variable.startswith(_DANGEROUS_ENV_PREFIXES)
            ):
                return f"dangerous environment assignment ({variable})"
    # A top-level hash -p or alias definition affects later command resolution;
    # nested shells can also change lookup via the remaining builtins.
    mutates_lookup = (
        (nested_source and name in _COMMAND_LOOKUP_MUTATORS)
        or (
            name == "hash"
            and any(option == "-p" or option.startswith("-p") for option in parts[1:])
        )
        or (name == "alias" and any("=" in value for value in parts[1:]))
        or (
            name == "alias"
            and any(
                child.type == "concatenation"
                and child.text is not None
                and b"=" in child.text
                for child in node.children
            )
        )
    )
    if mutates_lookup:
        return f"unsupported command lookup modification ({name})"
    return None


@dataclass(frozen=True)
class ShellPermissionAnalysis:
    command_parts: tuple[str, ...]
    approval_reasons: tuple[str, ...]

    @property
    def requires_approval(self) -> bool:
        return bool(self.approval_reasons)

    @property
    def approval_label(self) -> str:
        """Prompt text naming what made the command unsafe to auto-approve."""
        return f"shell syntax requiring approval: {', '.join(self.approval_reasons)}"


@lru_cache(maxsize=1)
def _get_parser() -> Parser:
    return Parser(Language(tsbash.language()))


def _analyze_shell_command(
    command: str, *, nested_source: bool = False
) -> ShellPermissionAnalysis:
    """Extract commands and fail closed on syntax the policy cannot model."""
    if "\0" in command:
        raise ValueError("shell command contains a NUL byte")
    command = re.sub(r"(?<!\\)\\\n", "", command)
    tree = _get_parser().parse(command.encode("utf-8"))
    commands: list[str] = []
    approval_reasons: set[str] = set()

    if tree.root_node.has_error:
        approval_reasons.add("a syntax error")

    all_nodes = (tree.root_node, *tuple(_descendants(tree.root_node)))
    has_file_redirect = any(node.type == "file_redirect" for node in all_nodes)
    has_command = any(_command_name(node) is not None for node in all_nodes)
    has_cwd_change = any(
        _command_name(node) in {"cd", "pushd", "popd"} for node in all_nodes
    )
    if has_file_redirect and not has_command:
        approval_reasons.add("redirect-only statements are not permitted")
    if (
        any(node.type == "variable_assignment" for node in all_nodes)
        and not has_command
    ):
        approval_reasons.add("assignment-only statements are not permitted")
    if has_file_redirect and has_cwd_change:
        approval_reasons.add(
            "redirection in a command chain containing cd"
            if any(_command_name(node) == "cd" for node in all_nodes)
            else "redirection in a command chain changing the working directory"
        )

    def find_commands(node: Node) -> None:  # noqa: PLR0912
        if node.type == "variable_assignment":
            if reason := _assignment_reason(node):
                approval_reasons.add(reason)
        elif node.type == "file_redirect":
            if reason := _redirect_reason(node):
                approval_reasons.add(reason)
        elif reason := _node_approval_reason(node):
            approval_reasons.add(reason)

        if node.type == "command":
            if reason := _lookup_mutation_reason(node, nested_source):
                approval_reasons.add(reason)
            parts: list[str] = []
            for child in node.children:
                if part := _supported_command_part(child):
                    parts.append(part)
                elif child.type == "ansi_c_string":
                    # Preserve the token for guardrails such as find's execution
                    # predicate while requiring approval because shlex does not
                    # decode Bash ANSI-C quoting.
                    if child.text is not None:
                        parts.append(child.text.decode("utf-8"))
                elif child.type == "variable_assignment":
                    pass
                elif (
                    child.type not in _DYNAMIC_NODES
                    and not _zsh_sensitive_node_reason(child)
                ):
                    # The executor receives the original shell string. If policy
                    # extraction omits a semantic command child, the two views can
                    # differ, so the command must not be auto-approved. The node
                    # type is kept in the reason so an unexpected construct is
                    # identifiable from the approval prompt alone. Children already
                    # named by _DYNAMIC_NODES are skipped here because the walk
                    # records their specific reason when it recurses into them.
                    approval_reasons.add(f"unsupported syntax ({child.type})")

            # A redirect is a sibling of the command under redirected_statement.
            # Keep the marker so standalone-command denylist behavior stays intact.
            if parts and node.parent and node.parent.type == "redirected_statement":
                parts.append("<redirect>")
            if parts:
                commands.append(" ".join(parts))

        for child in node.children:
            find_commands(child)

    find_commands(tree.root_node)
    return ShellPermissionAnalysis(
        command_parts=tuple(commands), approval_reasons=tuple(sorted(approval_reasons))
    )


def analyze_shell_command(
    command: str, *, nested_source: bool = False
) -> ShellPermissionAnalysis:
    """Analyze a command, denying safely when parsing or traversal fails."""
    try:
        return _analyze_shell_command(command, nested_source=nested_source)
    except Exception:
        return ShellPermissionAnalysis(
            command_parts=(), approval_reasons=("shell analysis failed",)
        )
