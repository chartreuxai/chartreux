from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from functools import lru_cache
import logging
from pathlib import Path
import re
import shlex
from typing import cast

from tree_sitter import Language, Node, Parser
import tree_sitter_bash as tsbash

from chartreux.core.tools.builtins._destination import (
    _PROXY_ENV_KEYS,
    _proxy_value_loopback,
)
from chartreux.core.tools.builtins._shell_argv import ShellToken, split_shell_tokens
from chartreux.core.tools.builtins._shell_command_policy import (
    executor_boundary,
    package_argument_indexes,
)
from chartreux.core.tools.builtins._shell_diagnostics import (
    DenialClass,
    Diagnostic,
    aggregate_denial_class,
)
from chartreux.core.tools.builtins._shell_static_expansion import (
    evaluate_expansion_for_position,
    recognize_sed_locate_print,
    resolve_equals_command,
    sed_locate_print_source,
)
from chartreux.core.tools.secret_redaction import redact
from chartreux.core.tools.utils import (
    ambient_workspace,
    is_canonical_scratch_path,
    resolve_tool_path,
)

logger = logging.getLogger(__name__)

_SUPPORTED_COMMAND_PARTS = {
    "command_name",
    "number",
    "word",
    "string",
    "raw_string",
    "concatenation",
}

_REDIRECTION_NODES = {"heredoc_redirect", "herestring_redirect"}

# Standard-device sinks are not filesystem targets: a redirect to one never
# opens a file, so it never counts as a file redirect for policy purposes.
_REDIRECTION_DEVICES = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})

# Assignment names that can alter command lookup, loading, startup, or runtime
# behavior. Prefix families are checked separately below. CURL_HOME and
# WGETRC name the modeled network clients' implicit startup-config
# discovery — curl reads "$CURL_HOME/.curlrc" first, wget reads the
# WGETRC startup file — so an agent-set value points the client at an
# uninspected config whose proxy/output lines control egress and writes.
DANGEROUS_ENV_NAMES = frozenset({
    "SSH_ASKPASS",
    "SSH_ASKPASS_REQUIRE",
    "RUSTC_WRAPPER",
    "RUSTC",
    "CC",
    "CXX",
    "RUSTFLAGS",
    "GOFLAGS",
    "MAKEFLAGS",
    "GNUMAKEFLAGS",
    "PERLLIB",
    "_JAVA_OPTIONS",
    "BASH_ENV",
    "BUN_INSTALL",
    "CARGO_HOME",
    "GOBIN",
    "GOPATH",
    "PIPX_HOME",
    "PIPX_BIN_DIR",
    "PIPX_MAN_DIR",
    "PIPX_COMPLETION_DIR",
    "PIPX_SHARED_LIBS",
    "PIPX_GLOBAL_HOME",
    "PIPX_GLOBAL_BIN_DIR",
    "PIPX_GLOBAL_MAN_DIR",
    "PIPX_GLOBAL_COMPLETION_DIR",
    "POETRY_CONFIG_DIR",
    "POETRY_CACHE_DIR",
    "POETRY_DATA_DIR",
    "POETRY_HOME",
    "POETRY_PYTHON_INSTALLATION_DIR",
    "PIP_TARGET",
    "PIP_PREFIX",
    "PIP_CONFIG_FILE",
    "PIP_CACHE_DIR",
    "PIP_ROOT",
    "PIP_SRC",
    "PIP_SOURCE",
    "PIP_SOURCE_DIR",
    "PIP_SOURCE_DIRECTORY",
    "PIP_DEST",
    "PIP_DESTINATION_DIR",
    "PIP_DESTINATION_DIRECTORY",
    "PIP_REQUIREMENTS_FROM_SCRIPT",
    "PIP_USER",
    "PIP_LOG",
    "PIP_REPORT",
    "PIP_BUILD_TRACKER",
    "PIP_DOWNLOAD_DIR",
    "PIP_WHEEL_DIR",
    "PIP_REQUIREMENT",
    "PIP_CONSTRAINT",
    "PIP_BUILD_CONSTRAINT",
    "UV_CONFIG_FILE",
    "UV_PROJECT",
    "UV_CREDENTIALS_DIR",
    "UV_INSTALL_DIR",
    "UV_UNMANAGED_INSTALL",
    "UV_PYTHON_BIN_DIR",
    "UV_PYTHON_CACHE_DIR",
    "UV_ENV_FILE",
    "UV_BUILD_CONSTRAINT",
    "UV_CONSTRAINT",
    "UV_OVERRIDE",
    "UV_EXCLUDE",
    "TRACING_DURATIONS_FILE",
    "CONDA_PREFIX",
    "CURL_HOME",
    "XDG_BIN_HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CONFIG_DIRS",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SYSTEMDRIVE",
    "UV_PYTHON_DOWNLOADS_JSON_URL",
    "APPDATA",
    "LOCALAPPDATA",
    "USERPROFILE",
    "UV_TARGET",
    "UV_CACHE_DIR",
    "UV_PROJECT_ENVIRONMENT",
    "UV_PYTHON_INSTALL_DIR",
    "UV_TOOL_DIR",
    "UV_TOOL_BIN_DIR",
    "UV_WORKING_DIR",
    "UV_WORKING_DIRECTORY",  # Supported by uv as a backwards-compatible alias.
    "POETRY_VIRTUALENVS_PATH",
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
    "KSH_ENV",
    "LESS",
    "LESSOPEN",
    "LESSCLOSE",
    "LESSEDIT",
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
    "SYSTEM_WGETRC",
    "VISUAL",
    "WGETRC",
    "ZDOTDIR",
    "PATH",
})
# Package configuration can move installation/cache destinations or load an
# uninspected config file. Gate these assignments before argv extraction drops them.
_DANGEROUS_ENV_PREFIXES = ("DYLD_", "GIT_", "NPM_CONFIG_")


def _syntax_reason(detail: str, alternative: str) -> Diagnostic:
    """An unmodeled v0.1 construct; the denial names the compliant alternative."""
    return Diagnostic(
        "analysis",
        detail=detail,
        denial_class=DenialClass.SYNTAX,
        alternative=alternative,
    )


def _policy_reason(detail: str) -> Diagnostic:
    """A deliberate security decision; the denial suggests nothing."""
    return Diagnostic("analysis", detail=detail, denial_class=DenialClass.POLICY)


# Shared compliant alternatives for SYNTAX denials, phrased against what the
# dedicated tools actually accept (chartreux/core/tools/builtins/prompts/):
# read_file pages with offset and limit, grep returns line numbers, edit
# replaces exact strings, write_file creates files, and bash_start runs
# managed long-running jobs.
_FILE_TOOLS_ALTERNATIVE = (
    "use read_file with offset and limit to read files; "
    "use grep to locate content and line numbers; "
    "use the edit tool to modify files"
)
_STEPWISE_ALTERNATIVE = (
    "split the work into separate simple commands, or use read_file with "
    "offset and limit, grep, and the edit tool for file work"
)
_HEREDOC_ALTERNATIVE = (
    "use write_file to create the content, or use read_file with offset and "
    "limit, grep, and the edit tool for existing files"
)
_LITERAL_VALUE_ALTERNATIVE = "write the literal value directly as the argument"

# Reasons are noun phrases so they read as a list in the approval prompt.
# Each entry is created with its denial class: unmodeled v0.1 constructs are
# SYNTAX denials whose message names the compliant alternative.
_DYNAMIC_NODES: dict[str, Diagnostic] = {
    "ansi_c_string": _syntax_reason(
        "ANSI-C quoted arguments", "write the argument as an ordinary quoted string"
    ),
    "arithmetic_expansion": _syntax_reason(
        "arithmetic expansion",
        "compute the value and write the literal result as the argument",
    ),
    "brace_expression": _syntax_reason(
        "brace expansion", "write each expanded value literally as its own argument"
    ),
    "command_substitution": _syntax_reason(
        "command substitution", _FILE_TOOLS_ALTERNATIVE
    ),
    "expansion": _syntax_reason("parameter expansion", _LITERAL_VALUE_ALTERNATIVE),
    "process_substitution": _syntax_reason(
        "process substitution",
        "write the command output to a file first, then pass the file path",
    ),
    "simple_expansion": _syntax_reason(
        "variable expansion (unsupported in v0.1 because it can change arguments)",
        _LITERAL_VALUE_ALTERNATIVE,
    ),
}

# Compound statements (SYNTAX) name the stepwise alternative. Unsetting
# environment variables is a policy decision (POLICY), not a vocabulary gap:
# the denial states the requirement and names no alternative.
_COMPOUND_NODES: dict[str, Diagnostic] = {
    "case_statement": _syntax_reason("case statement", _STEPWISE_ALTERNATIVE),
    "compound_statement": _syntax_reason("command group", _STEPWISE_ALTERNATIVE),
    "c_style_for_statement": _syntax_reason("for loop", _STEPWISE_ALTERNATIVE),
    "for_statement": _syntax_reason("for loop", _STEPWISE_ALTERNATIVE),
    "function_definition": _syntax_reason("function definition", _STEPWISE_ALTERNATIVE),
    "if_statement": _syntax_reason("if statement", _STEPWISE_ALTERNATIVE),
    "subshell": _syntax_reason("subshell", _STEPWISE_ALTERNATIVE),
    "test_command": _syntax_reason("test expression", _STEPWISE_ALTERNATIVE),
    # The parser does not emit a regular command node for this builtin.
    "unset_command": _policy_reason(
        "unsetting shell environment variables cannot be safely inspected"
    ),
    "while_statement": _syntax_reason("while loop", _STEPWISE_ALTERNATIVE),
}


def _zsh_sensitive_word_reason(value: bytes) -> Diagnostic | None:
    """Name Zsh word expansions that the Bash grammar treats as literals."""
    if value.startswith(b"="):
        return _syntax_reason(
            "Zsh equals expansion",
            "drop the leading equals sign and invoke the command by its name "
            "or absolute path",
        )
    if b"==" in value:
        return _syntax_reason(
            "Zsh magic-equals expansion",
            "drop the leading equals signs and invoke the command by its "
            "name or absolute path",
        )
    if value.startswith(b"~") and not (value == b"~" or value.startswith(b"~/")):
        return _syntax_reason(
            "shell-specific named-directory expansion",
            "spell the absolute directory path instead of the tilde-name form",
        )
    if b"***" in value:
        return _syntax_reason(
            "Zsh symlink-following glob",
            "use a plain star glob or spell the path literally",
        )
    return None


def _zsh_sensitive_node_reason(node: Node) -> Diagnostic | None:
    if node.type not in {"command_name", "word"} or node.text is None:
        return None
    return _zsh_sensitive_word_reason(node.text)


# --- tier-2 static expansion wiring -------------------------------------------
#
# The strict-shape recognizers in _shell_static_expansion turn three whitelisted
# idioms into literals BEFORE the syntax walk denies them. Each recognized
# construct is replaced in the emitted command parts by exactly one literal
# argv element, and that replacement goes through every downstream layer
# (denylist, standalone, path, metadata, dangerous-command) exactly like a
# hand-written command:
#
# (a) sed locate-then-print: the quoted script argument is replaced by the
#     statically expanded numeric script and then analyzed by the normal sed
#     script policy;
# (b) zsh equals expansion at the command position: a resolvable ``=command``
#     word is replaced by the resolved executable's basename — the name the
#     policy keys every command surface by — and then goes through the
#     normal command policy exactly like the hand-written name; the lookup
#     only proves the word names a real executable (zsh's own = semantics);
# (c) variable/arithmetic expansion in a provably read-only argument
#     position of a table-whitelisted command, with variables bound ONLY by
#     literal assignment STATEMENTS that precede the command in the same
#     invocation (each shell call is fresh; cross-call state does not exist).
#     A binding survives only while no later statement reassigns the name in
#     a form the collection does not model: an export/typeset/readonly/
#     declare/local argument (``export X=/etc/passwd`` parses as a
#     declaration_command, not a statement), an &&/||-guarded reassignment
#     that DOES bind at runtime, a command-prefix assignment, or a
#     non-literal value (``X=$Y``). Any such reassignment invalidates the
#     binding, so resolution uses only bindings proven unchanged through the
#     whole invocation.
#     A command-prefix assignment is NOT a source: every POSIX shell expands
#     argument words before the command's own assignments take effect, so
#     ``X=5 grep -m $X`` never sees the 5 (probed in zsh, bash, and sh) —
#     authorizing from a prefix assignment would diverge from what actually
#     executes.
#
# Everything else keeps its denial: an unrecognized shape denies with the same
# class and message as before, and a recognized-but-rejected shape keeps its
# proper class. Authorizing the sed idiom never depends on the file snapshot:
# the mandatory numeric-extraction stage yields a digits-only field for ANY
# file content, so a file changed between analysis and execution can at worst
# turn into a different numeric address or a sed runtime error, never a write.
# The simulation's READ is gated before the recognizer runs: the grep source
# must resolve inside the pinned scratchpad roots or the workspace roots, so
# the analysis never reads an out-of-workspace path at all.

_SED_LOCATE_QUOTED_PREFIX_RE = re.compile(r'^sed\s+-n\s+"')
_DYNAMIC_ELEMENT_NODES = frozenset({
    "arithmetic_expansion",
    "expansion",
    "simple_expansion",
})
_DRIFT_PREVIEW_LENGTH = 160
# A name assigned through a reassignment form the binding collection does
# not model (see the tier-2 comment above).
_BINDING_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _log_static_expansion_drift(idiom: str, text: str) -> None:
    """Log an unrecognized variant of a modeled idiom (tier-3 candidates).

    Debug level, redacted, and bounded: recurring shapes are the data, while
    secrets and bulk stay out of the log. Logging must never influence the
    permission outcome.
    """
    try:
        logger.debug(
            "unrecognized %s shape: %s", idiom, redact(text)[:_DRIFT_PREVIEW_LENGTH]
        )
    except Exception:
        pass


def _assignment_binding(node: Node) -> tuple[str, str] | None:
    """The (name, literal value) a safe assignment binds, or None.

    The value participates only when it is a single literal token; dynamics
    make the variable unresolvable, which denies downstream. Callers must
    have established the assignment itself is allowed (no policy reason).
    """
    if node.text is None:
        return None
    raw = node.text.decode("utf-8", errors="replace")
    name, separator, _ = raw.partition("=")
    if not separator:
        return None
    value = node.child_by_field_name("value")
    literal = _literal_token(value) if value is not None else ""
    if literal is None:
        return None
    return name, literal


def _assignment_name(node: Node) -> str | None:
    """The name a variable_assignment node assigns, or None."""
    if node.text is None:
        return None
    name, separator, _ = node.text.decode("utf-8", errors="replace").partition("=")
    if not separator or _BINDING_NAME_RE.fullmatch(name) is None:
        return None
    return name


def _invalidate_assigned_names(texts: Iterable[str], variables: dict[str, str]) -> None:
    """Invalidate bindings for names assigned by unmodeled reassignment text."""
    for text in texts:
        name, separator, _ = text.partition("=")
        if separator and _BINDING_NAME_RE.fullmatch(name) is not None:
            variables.pop(name, None)


# The declaration builtins whose NAME=value arguments the shell applies as
# persistent assignments. A prefix assignment demotes the form to a plain
# command node (`X=1 export NAME=value`), so the command branch checks its
# command name against this set too.
_DECLARATION_COMMAND_NAMES = frozenset({
    "export",
    "declare",
    "typeset",
    "readonly",
    "local",
})


def _invalidate_declaration_bindings(node: Node, variables: dict[str, str]) -> None:
    """Invalidate bindings reassigned by an export/typeset/... argument.

    ``export X=/etc/passwd`` parses as a declaration_command, so the
    assignment is not a top-level statement the collection models — but the
    shell applies it, and a quoted argument (``export 'X=/etc/passwd'``) is
    not even a variable_assignment node. Any ``NAME=value`` argument of an
    export/typeset/readonly/declare/local therefore invalidates an earlier
    binding for NAME. A bare ``export X`` carries no value and changes
    nothing, so it invalidates nothing.
    """
    for child in node.children:
        if child.type == "variable_assignment":
            continue  # Handled by the variable_assignment branch.
        if child.text is None:
            continue
        raw = child.text.decode("utf-8", errors="replace")
        literal = _literal_token(child)
        texts = (raw, literal) if literal is not None else (raw,)
        _invalidate_assigned_names(texts, variables)


def _record_declaration_binding(
    name: str,
    value: str,
    persistent: dict[str, str],
    candidates: dict[str, tuple[str, ...]],
    *,
    conditional: bool,
) -> None:
    """Record one literal NAME=value a declaration argument binds.

    An unconditional declaration binds exactly where the shell applies it
    and replaces any earlier binding for the name. A conditional declaration
    (an &&/||-guarded one, whose guard may be false at runtime) must not
    clear or overwrite a binding an earlier unconditional declaration
    established — that would let a loopback candidate un-gate a known
    non-loopback proxy the runtime never replaced. It records an additional
    CANDIDATE binding instead, and the egress proxy inspection gates when
    any candidate value in any reachable branch is a non-loopback proxy
    (union semantics, fail-closed).
    """
    if conditional:
        candidates[name] = (*candidates.get(name, ()), value)
    else:
        persistent[name] = value
        candidates.pop(name, None)


def _persist_declaration_bindings(
    node: Node,
    persistent: dict[str, str],
    candidates: dict[str, tuple[str, ...]],
    *,
    conditional: bool,
) -> None:
    """Bind the persistent env bindings an export/... argument leaves.

    The egress proxy inspection reads statement-level bindings that persist
    for later parts of the same command text, so an export/declare/... form
    binds exactly where the shell applies it. A literal ``NAME=value``
    argument binds — bare or quoted, since ``export 'X=v'`` assigns through
    its quoting exactly like the bare form. A bare ``export X`` re-exports
    the current value and changes nothing here. Any other argument
    (``export $x`` and friends) may overwrite a name at runtime with an
    unestablishable value; the last known binding stays in force — the
    fail-closed direction for a gate. variable_assignment arguments are
    handled by the variable_assignment branch when the walk recurses into
    them.
    """
    for child in node.children:
        if child.type == "variable_assignment":
            continue  # Handled by the variable_assignment branch.
        if child.text is None:
            continue
        literal = _literal_token(child)
        if literal is None:
            continue
        name, separator, value = literal.partition("=")
        if separator and _BINDING_NAME_RE.fullmatch(name) is not None:
            _record_declaration_binding(
                name, value, persistent, candidates, conditional=conditional
            )


def _union_binding_pairs(
    persistent: dict[str, str], candidates: dict[str, tuple[str, ...]]
) -> tuple[tuple[str, str], ...]:
    """The statement-level bindings in force, one pair per name.

    A name bound by an unconditional declaration carries its persistent
    value plus every candidate a conditional declaration recorded since;
    the egress proxy inspection must gate when ANY of those values is a
    non-loopback proxy, so a proxy-family name emits its non-loopback
    value when one exists (union semantics, fail-closed). Every other
    name carries the persistent value, or its first candidate when no
    unconditional binding exists. A non-proxy value never gates: the
    loopback classification applies to proxy-family keys only.
    """
    pairs: list[tuple[str, str]] = []
    for name in persistent.keys() | candidates.keys():
        values = (persistent[name],) if name in persistent else ()
        values += candidates.get(name, ())
        non_loopback = (
            next(
                (
                    value
                    for value in values
                    if value and not _proxy_value_loopback(value)
                ),
                None,
            )
            if name.casefold() in _PROXY_ENV_KEYS
            else None
        )
        pairs.append((name, non_loopback if non_loopback is not None else values[0]))
    return tuple(pairs)


def _within_read_roots(source: str, roots: frozenset[Path]) -> bool:
    """Whether the simulation's read of ``source`` stays inside the roots.

    The source resolves exactly as the read does — against the analysis-time
    working directory — and must land inside a pinned root (component-wise,
    never a string prefix), mirroring the resolver's own workspace root
    check, so the simulation never reads an out-of-workspace path.
    """
    if not roots:
        return False
    try:
        resolved = resolve_tool_path(source, Path.cwd())
    except (OSError, RuntimeError, ValueError):
        return False
    return any(
        root == root.resolve() and resolved.is_relative_to(root) for root in roots
    )


def _simulation_read_roots(
    scratchpad_roots: frozenset[Path], workspace_roots: frozenset[Path] | None
) -> frozenset[Path]:
    """The roots the sed simulation may read inside (workspace+scratchpad).

    ``workspace_roots`` defaults to the ambient workspace; a caller that
    cannot resolve one gets no roots, and the simulation reads nothing.
    """
    if workspace_roots is None:
        try:
            workspace_roots = frozenset(ambient_workspace().authorized_roots)
        except (OSError, RuntimeError, ValueError):
            workspace_roots = frozenset()
    return frozenset(scratchpad_roots) | frozenset(workspace_roots)


def _sed_locate_print_replacement(
    node: Node,
    text: str,
    authorized: list[tuple[int, int]],
    add_reason: Callable[[Diagnostic], None],
    read_roots: frozenset[Path],
) -> tuple[tuple[int, int], str] | None:
    """The (byte range, literal script) replacing the sed construct, or None.

    The grep source is pre-checked against ``read_roots`` BEFORE the
    recognizer runs, so the simulation never reads an out-of-workspace path;
    an unrecognized shape keeps its ordinary denial (fail closed).
    """
    source = sed_locate_print_source(text)
    if source is not None and not _within_read_roots(source, read_roots):
        _log_static_expansion_drift("sed locate-then-print", text)
        return None
    recognized = recognize_sed_locate_print(text)
    if recognized is None:
        _log_static_expansion_drift("sed locate-then-print", text)
        return None
    if len(recognized.addresses) > 1:
        # A multi-match address can never be a valid single sed script
        # (the newline-joined address is a sed runtime error), so the
        # expanded command would not run; reject it as unexecutable.
        add_reason(
            _syntax_reason(
                "a sed locate-then-print expansion locating multiple lines",
                "narrow the grep pattern or select one located line "
                "(for example with head -1) so the sed script has a "
                "single numeric address",
            )
        )
        return None
    script_child = next(
        (
            child
            for child in node.children
            if child.type == "string"
            and any(
                descendant.type == "command_substitution"
                for descendant in _descendants(child)
            )
        ),
        None,
    )
    if script_child is None:
        return None
    key = (script_child.start_byte, script_child.end_byte)
    authorized.append(key)
    return key, shlex.quote(recognized.expanded_script)


def _dynamic_argument_element(child: Node) -> bool:
    """Whether child is one unquoted argument element carrying expansion syntax.

    Double-quoted elements are excluded: their expansion text contains quotes,
    which the static evaluator rejects, so they keep the walk's denial.
    """
    if child.type in _DYNAMIC_ELEMENT_NODES:
        return True
    return child.type in {"concatenation", "word"} and any(
        descendant.type in _DYNAMIC_ELEMENT_NODES for descendant in _descendants(child)
    )


def _argument_position_key(argument_tokens: list[str]) -> str:
    """The position key of the next argument element (conservative).

    An element following an option token is classified as that option's value;
    otherwise it is the next positional. Miscounts are safe: the table's
    read-only positions belong exclusively to pure readers, and every unknown
    or unclassified position fails closed in classify_argument_position.
    """
    if not argument_tokens:
        return "positional:1"
    if argument_tokens[-1].startswith("-"):
        return f"option-arg:{argument_tokens[-1]}"
    positional = 0
    index = 0
    while index < len(argument_tokens):
        if argument_tokens[index].startswith("-"):
            index += 2  # An option plus the value it consumes.
            continue
        positional += 1
        index += 1
    return f"positional:{positional + 1}"


def _static_expansion_replacements(
    node: Node,
    authorized: list[tuple[int, int]],
    add_reason: Callable[[Diagnostic], None],
    variables: dict[str, str],
    read_roots: frozenset[Path],
) -> dict[tuple[int, int], str]:
    """Literal replacements for this command's statically expanded constructs.

    Maps a child's byte range to the literal argv element that replaces it.
    ``authorized`` collects the byte ranges whose dynamic-syntax denial is
    consumed by the replacement (suppressed only for expansion node types and
    ``=``-leading command names — never for redirects, braces, or heredocs).
    ``read_roots`` are the roots the sed simulation may read inside.
    """
    text = (node.text or b"").decode("utf-8", errors="replace")
    replacements: dict[tuple[int, int], str] = {}

    # (a) sed locate-then-print: the expanded script goes through the normal
    # sed script policy exactly like a hand-written literal script.
    if _SED_LOCATE_QUOTED_PREFIX_RE.match(text) and "$(" in text:
        replaced = _sed_locate_print_replacement(
            node, text, authorized, add_reason, read_roots
        )
        if replaced is not None:
            key, replacement = replaced
            replacements[key] = replacement

    # (b) zsh equals expansion at the command position. Equals words in
    # argument positions stay denied: no argument position models them.
    for child in node.children:
        if (
            child.type != "command_name"
            or child.text is None
            or not child.text.startswith(b"=")
        ):
            continue
        resolved = resolve_equals_command(child.text.decode("utf-8", errors="replace"))
        if resolved is None:
            _log_static_expansion_drift("zsh equals expansion", text)
            continue
        # Inline the resolved executable's BASENAME, not its path: the policy
        # is argv-shaped and keys every command surface (denylist, tables,
        # guardrails) by basename, and zsh's ``=name`` executes the same PATH
        # lookup as a plain ``name`` invocation — the lookup here only proves
        # the word names a real executable. An absolute command token would be
        # re-inspected as a path operand and deny like a hand-written one.
        key = (child.start_byte, child.end_byte)
        authorized.append(key)
        replacements[key] = resolved.rsplit("/", 1)[-1]

    # (c) variable/arithmetic expansion in read-only argument positions.
    command_name: str | None = None
    argument_tokens: list[str] = []
    for child in node.children:
        if child.type == "variable_assignment":
            continue
        key = (child.start_byte, child.end_byte)
        if child.type == "command_name":
            token = replacements.get(key)
            if token is None:
                token = _literal_token(child) or (
                    (child.text or b"").decode("utf-8", errors="replace")
                )
            command_name = token.rsplit("/", 1)[-1]
            continue
        if not _dynamic_argument_element(child) or child.text is None:
            token = _supported_command_part(child)
            if token is not None:
                argument_tokens.append(token)
            continue
        expression = child.text.decode("utf-8", errors="replace")
        value = None
        if command_name is not None:
            value = evaluate_expansion_for_position(
                command_name,
                _argument_position_key(argument_tokens),
                expression,
                variables,
            )
        if value is None:
            # Unresolved: the walk denies the construct exactly as before.
            argument_tokens.append(expression)
            continue
        authorized.append(key)
        replacements[key] = shlex.quote(value)
        argument_tokens.append(value)
    return replacements


def _supported_command_part(node: Node) -> str | None:
    if node.type not in _SUPPORTED_COMMAND_PARTS or node.text is None:
        return None
    if _zsh_sensitive_node_reason(node):
        return None
    return node.text.decode("utf-8")


def _node_approval_reason(node: Node) -> Diagnostic | None:
    if node.type in _REDIRECTION_NODES:
        detail, alternative = (
            ("heredocs are unsupported in v0.1", _HEREDOC_ALTERNATIVE)
            if node.type == "heredoc_redirect"
            else ("redirection", "pass the value as a literal argument")
        )
        return _syntax_reason(detail, alternative)
    if reason := _DYNAMIC_NODES.get(node.type):
        return reason
    if reason := _zsh_sensitive_node_reason(node):
        return reason
    if node.type == "concatenation" and any(
        child.type == "word" and child.text in {b"{", b"}"} for child in node.children
    ):
        return _syntax_reason(
            "brace expansion", "write each expanded value literally as its own argument"
        )
    if compound := _COMPOUND_NODES.get(node.type):
        return compound
    return (
        _syntax_reason(
            "background execution",
            "run the command in the foreground, or use bash_start for a "
            "long-running job",
        )
        if node.type == "&"
        else None
    )


def _descendants(node: Node) -> Iterator[Node]:
    for child in node.children:
        yield child
        yield from _descendants(child)


def _literal_token(node: Node) -> str | None:
    if node.type not in _SUPPORTED_COMMAND_PARTS or node.text is None:
        return None
    if any(child.type in _DYNAMIC_NODES for child in _descendants(node)):
        return None
    try:
        import shlex

        values = shlex.split(node.text.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return values[0] if len(values) == 1 else None


def _literal_shell_token(node: Node) -> ShellToken | None:
    """The quote-provenance literal of a redirect destination, or None.

    Mirrors ``_literal_token`` but tokenizes with ``split_shell_tokens`` so
    the redirect seam learns which glob metacharacters the shell keeps
    literal (quoted) and which it pathname-expands (unquoted). Token values
    match ``shlex.split`` exactly; tokens split by other means are plain
    ``str`` and read as "no provenance" (fail closed).
    """
    if node.type not in _SUPPORTED_COMMAND_PARTS or node.text is None:
        return None
    if any(child.type in _DYNAMIC_NODES for child in _descendants(node)):
        return None
    try:
        values = split_shell_tokens(node.text.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    # split_shell_tokens is annotated list[str] (its elements are str
    # subclasses); the runtime value always carries the provenance flags.
    return cast("ShellToken | None", values[0] if len(values) == 1 else None)


def _assignment_reason(node: Node) -> Diagnostic | None:  # noqa: PLR0911
    # Environment-assignment restrictions are policy decisions: the denial
    # states the blocked requirement and names no alternative.
    if node.text is None:
        return _policy_reason("an environment assignment that cannot be inspected")
    raw = node.text.decode("utf-8")
    name, separator, _ = raw.partition("=")
    if not separator or re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) is None:
        return _policy_reason("an environment assignment with a non-uppercase name")
    if name in DANGEROUS_ENV_NAMES or name.startswith(_DANGEROUS_ENV_PREFIXES):
        return _policy_reason(f"dangerous environment assignment ({name})")
    value = node.child_by_field_name("value")
    if value is None:
        literal_value = ""
    else:
        literal_value = _literal_token(value)
        if literal_value is None:
            return _policy_reason("a non-literal environment assignment")
        if value.type not in {"string", "raw_string"} and any(
            syntax in literal_value for syntax in ("|", "`", "$(")
        ):
            return _policy_reason(
                f"dangerous environment assignment ({name}): command-execution syntax"
            )
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
            return _policy_reason("dangerous environment assignment (PYTHONPATH)")
    return None


def _is_scratchpad_target(destination: str, roots: frozenset[Path]) -> bool:
    """Whether an absolute redirect target resolves inside a scratchpad root.

    The match is component-wise against the pinned roots over the full
    realpath — never a string prefix — so a sibling directory that merely
    spells like a scratchpad root, or a symlink escaping one, stays denied.
    """
    if not roots:
        return False
    try:
        resolved = resolve_tool_path(destination, Path.cwd())
    except (OSError, RuntimeError, ValueError):
        return False
    return any(is_canonical_scratch_path(resolved, root) for root in roots)


def _redirect_reason(  # noqa: PLR0911
    node: Node, *, scratchpad_roots: frozenset[Path] = frozenset()
) -> Diagnostic | None:
    descriptor = node.child_by_field_name("descriptor")
    destination = node.child_by_field_name("destination")
    operators = [
        child.type for child in node.children if child.type in {">", ">>", ">&"}
    ]
    operator = operators[0] if len(operators) == 1 else None
    descriptor_text = (
        descriptor.text.decode("ascii") if descriptor and descriptor.text else None
    )
    destination_text = (
        _literal_shell_token(destination) if destination is not None else None
    )

    if operator == ">&" and descriptor_text == "2" and destination_text == "1":
        return None
    if operator not in {">", ">>"} or descriptor_text not in {None, "2"}:
        # An unmodeled redirection form (SYNTAX): the denial names the
        # compliant form. Input redirection has no allowed shell form, so the
        # file tools are the alternative there.
        if any(child.type == "<" for child in node.children):
            return _syntax_reason(
                "unsupported redirection (only >, >>, 2>, and 2>&1 are allowed)",
                "read the file with read_file, or pass its content as a "
                "literal argument",
            )
        return _syntax_reason(
            "unsupported redirection (only >, >>, 2>, and 2>&1 are allowed)",
            "use an allowed redirection form instead: plain output, append, "
            "stderr, or combined stderr to stdout",
        )
    if destination_text is None:
        # A non-literal target cannot be inspected: fail closed (POLICY).
        return _policy_reason("a non-literal redirection target")
    if destination_text.startswith("-") or destination_text.isdigit():
        return _policy_reason("an invalid redirection path target")
    if destination_text in _REDIRECTION_DEVICES:
        # Device sinks are not filesystem targets (ADR 0021).
        return None
    from pathlib import PurePosixPath

    if PurePosixPath(destination_text).is_absolute():
        # Absolute targets are authorized only inside the session scratchpad
        # roots. The resolver's path layer independently gates every literal
        # redirect target as a WRITE, so this exemption is friction relief,
        # never an authorization beyond workspace+scratchpad.
        if _is_scratchpad_target(destination_text, scratchpad_roots):
            return None
        # An absolute target outside the scratchpad is a workspace boundary.
        return _policy_reason("an absolute redirection target")
    return None


def _literal_redirect_target(node: Node) -> str | None:
    """The literal filesystem target of a file_redirect, or None.

    fd duplication (2>&1) and non-literal destinations are not filesystem
    targets; device sinks are targets (the resolver filters them from
    synthetic operands).
    """
    destination = node.child_by_field_name("destination")
    if destination is None:
        return None
    target = _literal_shell_token(destination)
    if target is None:
        return None
    duplication = any(child.type in {">&", "<&"} for child in node.children)
    if duplication and (target.isdigit() or target == "-"):
        return None  # Descriptor duplication is not filesystem access.
    return target


def _redirect_anchor(node: Node) -> Node | None:
    """The command a redirect applies to, for occurrence-local resolution.

    A trailing redirect on a list or pipeline applies to its last command
    (``cd sub && make > out.log`` redirects make's output); a redirect-only
    statement anchors to nothing and resolves against the next command.
    """
    parent = node.parent
    if parent is None:
        return None
    if parent.type == "command":
        # Redirect-first and assignment forms keep the redirect inside the
        # command node; the analysis denies those shapes, but the target is
        # still attributed so it stays checked.
        return parent
    if parent.type != "redirected_statement":
        return None
    statement = next(
        (
            child
            for child in parent.children
            if child.type
            not in {"file_redirect", "heredoc_redirect", "herestring_redirect"}
        ),
        None,
    )
    if statement is None:
        return None  # Redirect-only statement: no command to anchor to.
    if statement.type == "command":
        return statement
    anchored = [child for child in _descendants(statement) if child.type == "command"]
    return anchored[-1] if anchored else None


def _executable_parts(node: Node) -> list[str]:
    if node.type != "command":
        return []
    parts = [
        token for child in node.children if (token := _literal_token(child)) is not None
    ]
    # Follow registry-owned argv positions only, with the same depth bound as
    # Bash. env stays visible so its original assignment spelling is checked.
    for _ in range(8):
        if parts and parts[0].rsplit("/", 1)[-1] == "env":
            break
        try:
            boundary = executor_boundary(parts)
        except ValueError:
            break  # Bash reports the grammar failure after extracting argv.
        if boundary is None or boundary.inner_command_range is None:
            break
        start, end = boundary.inner_command_range
        parts = parts[start:end]
    return parts


def _has_active_bracket_glob(value: bytes) -> bool:
    """Recognize complete character classes, respecting shell quotes and escapes."""
    characters: list[tuple[str, bool]] = []
    text = value.decode("utf-8")
    quote = ""
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\\" and quote != "'" and index + 1 < len(text):
            index += 1
            characters.append((text[index], False))
        elif character in {'"', "'"} and (not quote or character == quote):
            quote = "" if quote else character
        else:
            characters.append((character, not quote))
        index += 1
    for index, (character, active) in enumerate(characters):
        if character != "[" or not active:
            continue
        end = index + 1
        if end < len(characters) and characters[end][0] in {"!", "^"}:
            end += 1
        start = end
        # A leading ] is a member of the class, not its terminator.
        if end < len(characters) and characters[end][0] == "]":
            end += 1
        while end < len(characters):
            if characters[end] == ("]", True):
                if end > start:
                    return True
                break
            end += 1
    return False


def _command_allowance_reasons(node: Node) -> Iterator[Diagnostic]:
    """Check syntax before argv normalization discards quoting information."""
    parts = _executable_parts(node)
    # Only an actual env executable introduces assignment operands. Wrapper
    # commands are inspected separately with their original shell spelling.
    if parts and parts[0].rsplit("/", 1)[-1] == "env":
        # env arguments are not assignment nodes in the original AST. Reparse
        # their original spelling, rather than shlex-quoting normalized values.
        env_seen = False
        for child in node.children:
            if not env_seen:
                token = _literal_token(child)
                env_seen = token is not None and token.rsplit("/", 1)[-1] == "env"
                continue
            if child.text is None:
                continue
            raw = child.text.decode("utf-8")
            if re.match(r"[A-Za-z_][A-Za-z0-9_]*=", raw) is None:
                break
            tree = _get_parser().parse(f"{raw} true".encode())
            for assignment in _descendants(tree.root_node):
                if assignment.type == "variable_assignment":
                    if reason := _assignment_reason(assignment):
                        yield reason
        parts = parts[1:]
        while parts and re.match(r"[A-Za-z_][A-Za-z0-9_]*=", parts[0]):
            parts = parts[1:]
    argv_nodes = [child for child in node.children if _literal_token(child) is not None]
    try:
        indexes = package_argument_indexes([
            _literal_token(child) or "" for child in argv_nodes
        ])
    except ValueError:
        indexes = set()  # The recursive pipeline denies invalid executor syntax.
    if any(
        index in indexes
        and child.text is not None
        and _has_active_bracket_glob(child.text)
        for index, child in enumerate(argv_nodes)
    ):
        yield _syntax_reason(
            "unquoted package bracket syntax may expand as a shell glob",
            "quote the package argument with single quotes so the shell "
            "cannot expand it",
        )


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


def argv_permission_diagnostics(
    parts: list[str], *, nested_source: bool = False
) -> tuple[Diagnostic, ...]:
    """Classed approval findings for an actual executable argv.

    Environment-assignment and command-lookup restrictions are policy
    decisions: their findings carry the POLICY class and name no alternative.
    """
    if not parts:
        return ()
    name = parts[0].rsplit("/", 1)[-1]
    reasons: list[Diagnostic] = []
    if name == "env":
        try:
            boundary = executor_boundary(parts)
        except ValueError:
            return ()  # The pipeline reports this invalid grammar on entry.
        end = (
            boundary.inner_command_range[0]
            if boundary and boundary.inner_command_range
            else len(parts)
        )
        for assignment in parts[1:end]:
            if assignment.startswith("-"):
                # An env option operand (the attached spelling --unset=BAR
                # or --chdir=.), which the boundary engine consumed as an
                # option, not applied as an assignment: env's operand scan
                # excludes dash-prefixed operands for exactly this reason
                # (see _env_assignment_operand), so the attached spelling
                # is not policy-checked as an assignment either — it
                # matches its separate counterpart (--unset BAR, --chdir .).
                continue
            key, separator, value = assignment.partition("=")
            if not separator:
                continue
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) is None:
                reasons.append(
                    _policy_reason(
                        "an environment assignment with a non-uppercase name"
                    )
                )
            elif key in DANGEROUS_ENV_NAMES or key.startswith(_DANGEROUS_ENV_PREFIXES):
                reasons.append(
                    _policy_reason(f"dangerous environment assignment ({key})")
                )
            elif key == "PYTHONPATH" and any(
                not entry or entry.startswith(("/", "~")) or ".." in entry.split("/")
                for entry in value.split(":")
            ):
                reasons.append(
                    _policy_reason("dangerous environment assignment (PYTHONPATH)")
                )
    if name in {"export", "declare", "typeset", "readonly"}:
        for assignment in parts[1:]:
            key, separator, _ = assignment.partition("=")
            if separator and (
                key in DANGEROUS_ENV_NAMES or key.startswith(_DANGEROUS_ENV_PREFIXES)
            ):
                reasons.append(
                    _policy_reason(f"dangerous environment assignment ({key})")
                )
    mutates_lookup = nested_source and name in _COMMAND_LOOKUP_MUTATORS
    mutates_lookup |= name == "hash" and any(
        value.startswith("-p") for value in parts[1:]
    )
    mutates_lookup |= name == "alias" and any("=" in value for value in parts[1:])
    if mutates_lookup:
        reasons.append(
            _policy_reason(f"unsupported command lookup modification ({name})")
        )
    return tuple(reasons)


def argv_permission_reasons(
    parts: list[str], *, nested_source: bool = False
) -> tuple[str, ...]:
    """Inspect an actual executable argv without interpreting its arguments as shell."""
    return tuple(
        reason.detail or ""
        for reason in argv_permission_diagnostics(parts, nested_source=nested_source)
    )


def _lookup_mutation_reason(node: Node, nested_source: bool) -> Diagnostic | None:
    return next(
        iter(
            argv_permission_diagnostics(
                _executable_parts(node), nested_source=nested_source
            )
        ),
        None,
    )


@dataclass(frozen=True)
class ShellPermissionAnalysis:
    command_parts: tuple[str, ...]
    approval_reasons: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...] = ()
    # Literal redirect targets grouped by occurrence position: group i holds
    # the targets whose synthetic WRITE operands resolve against the working
    # directory in force before command part i runs, and the final group
    # holds targets that follow the last part. Redirections are set up before
    # their command executes, so a redirect attached to a cd resolves against
    # the pre-cd working directory.
    redirect_operands: tuple[tuple[str, ...], ...] = ()
    # The literal (name, value) environment bindings of command part i,
    # aligned with command_parts: the statement-level bindings in force
    # where the part runs (an earlier ``export NAME=value`` or standalone
    # ``NAME=value`` statement of the same command text) followed by the
    # part's own command-prefix assignments, which win (POSIX prefix
    # semantics). The part text drops its assignment children, but the shell
    # applies all of these to the command's subprocess, so consumers that
    # must see the command's effective environment (the egress proxy
    # inspection) read them here. Only provably-literal bindings are
    # recorded; anything else is denied upstream as uninspectable.
    command_assignments: tuple[tuple[tuple[str, str], ...], ...] = ()

    @property
    def approval_diagnostic(self) -> Diagnostic:
        # Fail closed: a mixed command (unmodeled syntax plus a policy
        # violation) aggregates to a policy denial that names no alternative.
        return aggregate_approval_diagnostic(self)

    @property
    def requires_approval(self) -> bool:
        return bool(self.approval_reasons)

    @property
    def approval_label(self) -> str:
        """Prompt text naming what made the command unsafe to auto-approve."""
        # The prefix follows the aggregated class: a policy restriction is not
        # unsupported syntax, so it must not be introduced as one.
        prefix = (
            "unsupported shell syntax: "
            if self.approval_diagnostic.denial_class is DenialClass.SYNTAX
            else "shell policy restriction: "
        )
        return f"{prefix}{', '.join(self.approval_reasons)}"


def aggregate_approval_diagnostic(
    analysis: ShellPermissionAnalysis, *policy_reasons: Diagnostic
) -> Diagnostic:
    """The whole-call denial aggregate, including additional policy-class reasons.

    Follows the fail-closed ``aggregate_denial_class`` convention: the
    aggregate is a syntax denial only when every reason — the analysis
    diagnostics and the additional policy-class ones — is one. A mixed call
    is a policy denial that states the blocked requirements and names no
    alternative.
    """
    related = analysis.diagnostics or tuple(
        Diagnostic("analysis", detail=reason) for reason in analysis.approval_reasons
    )
    related = (*related, *policy_reasons)
    return Diagnostic(
        "analysis", related=related, denial_class=aggregate_denial_class(related)
    )


@lru_cache(maxsize=1)
def _get_parser() -> Parser:
    return Parser(Language(tsbash.language()))


def _analyze_shell_command(  # noqa: PLR0915, PLR0914
    command: str,
    *,
    nested_source: bool = False,
    scratchpad_roots: frozenset[Path] = frozenset(),
    workspace_roots: frozenset[Path] | None = None,
) -> ShellPermissionAnalysis:
    """Extract commands and fail closed on syntax the policy cannot model."""
    if "\0" in command:
        raise ValueError("shell command contains a NUL byte")
    command = re.sub(r"(?<!\\)\\\n", "", command)
    tree = _get_parser().parse(command.encode("utf-8"))
    commands: list[str] = []
    # Literal redirect targets grouped by the command part they precede; the
    # trailing group collects targets that follow the last part (redirect-only
    # statements anchor to the next command, or to the end of the chain).
    redirect_groups: list[list[str]] = []
    pending_targets: list[str] = []
    command_assignment_groups: list[tuple[tuple[str, str], ...]] = []
    part_index_by_command: dict[tuple[int, int, str], int] = {}
    approval_reasons: set[str] = set()
    reason_diagnostics: dict[str, Diagnostic] = {}
    diagnostics: dict[str, Diagnostic] = {}
    # Byte ranges of constructs consumed by a recognized static expansion:
    # their dynamic-syntax denial is suppressed because the emitted command
    # parts already carry the expanded literal.
    authorized: list[tuple[int, int]] = []
    # Variables bound by safe literal assignment statements of this
    # invocation, in statement order. Only top-level `;`-separated statements
    # bind: a reassignment in any other form (an export/typeset/readonly/
    # declare/local argument, an &&/||-guarded statement that may bind at
    # runtime, a command prefix, a subshell, or a non-literal value) may
    # overwrite the name, so it invalidates the binding instead.
    invocation_variables: dict[str, str] = {}
    # The statement-level environment bindings in force at each point of
    # this invocation, in statement order: a top-level standalone
    # `NAME=value` statement or an `export`/`declare`/... `NAME=value`
    # argument persists for every LATER part of the same command text, so
    # the egress proxy inspection gates on it exactly like a command-prefix
    # assignment. A later literal statement rebinds the name; a
    # reassignment in any form the collection does not model (an
    # &&/||-guarded statement, a command prefix, a subshell, `unset`)
    # leaves the last known binding in force — the fail-closed direction
    # for a gate, which must never un-gate a known non-loopback proxy.
    # Residuals, documented: a name whose FIRST binding is an unmodeled
    # form never enters the map, and bindings across separate commands
    # are out of scope — each tool call is a fresh subprocess. A nested
    # shell payload records only its own bindings here; the resolver's
    # expansion merges the enclosing statement's bindings (and the
    # wrapper's command-prefix assignments) into the payload's parts, so
    # the payload's clients see the environment the nested shell passes
    # on, exactly as the real shell would.
    persistent_assignments: dict[str, str] = {}
    # Candidate bindings a CONDITIONAL declaration (an &&/||-guarded
    # export/declare/... whose guard may be false at runtime) establishes.
    # A conditional declaration never clears or overwrites a binding an
    # earlier unconditional declaration established — that would let a
    # loopback candidate un-gate a known non-loopback proxy the runtime
    # never replaced. Each part's egress snapshot merges the persistent
    # bindings with these candidates under union semantics: the gate
    # fires when ANY value in any reachable branch is a non-loopback
    # proxy (fail-closed).
    conditional_candidates: dict[str, tuple[str, ...]] = {}
    # Roots the sed locate-then-print simulation may read inside: the pinned
    # scratchpad roots plus the workspace roots (the ambient workspace when
    # the caller supplies none). The recognizer never reads outside them.
    read_roots = _simulation_read_roots(scratchpad_roots, workspace_roots)

    def add_reason(diagnostic: Diagnostic) -> None:
        detail = diagnostic.detail or ""
        approval_reasons.add(detail)
        existing = reason_diagnostics.get(detail)
        if existing is None:
            reason_diagnostics[detail] = diagnostic
        elif existing.denial_class is not diagnostic.denial_class:
            # One reason never carries two classes: fail closed to policy.
            reason_diagnostics[detail] = Diagnostic("analysis", detail=detail)

    def _consumed_by_static_expansion(node: Node) -> bool:
        """Whether a recognized strict shape already replaced this construct."""
        if node.type in _DYNAMIC_NODES:
            return any(
                start <= node.start_byte and node.end_byte <= end
                for start, end in authorized
            )
        if (
            node.text
            and node.text.startswith(b"=")
            and node.type in {"command_name", "word"}
        ):
            # An equals command_name and its word child share one byte range;
            # an exact-range match suppresses exactly that pair.
            return any(
                start == node.start_byte and node.end_byte == end
                for start, end in authorized
            )
        return False

    if tree.root_node.has_error:
        add_reason(
            _syntax_reason(
                "a syntax error", "write the command as well-formed shell syntax"
            )
        )

    all_nodes = (tree.root_node, *tuple(_descendants(tree.root_node)))
    has_file_redirect = any(node.type == "file_redirect" for node in all_nodes)
    has_command = any(_command_name(node) is not None for node in all_nodes)
    if has_file_redirect and not has_command:
        add_reason(
            _syntax_reason(
                "redirect-only statements are not permitted",
                "attach the redirection to a command, or use write_file or the "
                "edit tool to create or change the file",
            )
        )
    if (
        any(node.type == "variable_assignment" for node in all_nodes)
        and not has_command
    ):
        add_reason(
            _syntax_reason(
                "assignment-only statements are not permitted",
                "attach the assignment to a command in the same statement",
            )
        )

    def find_commands(  # noqa: PLR0912, PLR0915
        node: Node, *, conditional: bool = False
    ) -> None:
        previous_reasons = approval_reasons.copy()
        if node.type == "variable_assignment":
            reason = _assignment_reason(node)
            if reason is not None:
                add_reason(reason)
            else:
                if node.parent is not None and node.parent.type in {
                    "program",
                    "variable_assignments",
                }:
                    if binding := _assignment_binding(node):
                        invocation_variables[binding[0]] = binding[1]
                    elif name := _assignment_name(node):
                        # A top-level reassignment whose value is not a
                        # provably-literal token: the collection cannot prove
                        # an earlier binding unchanged, so it no longer
                        # resolves.
                        invocation_variables.pop(name, None)
                elif name := _assignment_name(node):
                    # A reassignment in a form the collection does not model
                    # — an export/typeset/readonly/declare/local argument, an
                    # &&/||-guarded statement, a command prefix, a subshell —
                    # may bind at runtime, so an earlier binding is unproven.
                    invocation_variables.pop(name, None)
                # The persistent environment bindings the egress proxy
                # inspection reads: a top-level standalone statement or an
                # export/declare/... argument binds (a later literal
                # statement rebinds). A command prefix applies to its own
                # part only, and a reassignment in an unmodeled form
                # leaves the last known binding in force.
                if node.parent is not None and node.parent.type in {
                    "program",
                    "variable_assignments",
                    "declaration_command",
                }:
                    if binding := _assignment_binding(node):
                        _record_declaration_binding(
                            binding[0],
                            binding[1],
                            persistent_assignments,
                            conditional_candidates,
                            conditional=conditional,
                        )
        elif node.type == "declaration_command":
            _invalidate_declaration_bindings(node, invocation_variables)
            _persist_declaration_bindings(
                node,
                persistent_assignments,
                conditional_candidates,
                conditional=conditional,
            )
        elif node.type == "file_redirect":
            if reason := _redirect_reason(node, scratchpad_roots=scratchpad_roots):
                add_reason(reason)
            if target := _literal_redirect_target(node):
                anchor = _redirect_anchor(node)
                index = (
                    part_index_by_command.get((
                        anchor.start_byte,
                        anchor.end_byte,
                        anchor.type,
                    ))
                    if anchor is not None
                    else None
                )
                if index is None:
                    # No emitted part precedes this redirect yet (a
                    # redirect-only statement, or an anchor that produced no
                    # command part): it resolves against the next part.
                    pending_targets.append(target)
                else:
                    redirect_groups[index].append(target)
        elif reason := _node_approval_reason(node):
            if not _consumed_by_static_expansion(node):
                add_reason(reason)

        if node.type == "command":
            for reason in _command_allowance_reasons(node):
                add_reason(reason)
            if reason := _lookup_mutation_reason(node, nested_source):
                add_reason(reason)
            if _command_name(node) in _DECLARATION_COMMAND_NAMES:
                # A prefix assignment demotes the export/typeset/... form to
                # a plain command node (`X=1 export NAME=value`), but the
                # runtime declaration semantics are unchanged: the arguments
                # bind exactly where a declaration_command's would, so the
                # walk collects them here.
                _invalidate_declaration_bindings(node, invocation_variables)
                _persist_declaration_bindings(
                    node,
                    persistent_assignments,
                    conditional_candidates,
                    conditional=conditional,
                )
            replacements = _static_expansion_replacements(
                node, authorized, add_reason, invocation_variables, read_roots
            )
            parts: list[str] = []
            prefix_assignments: list[tuple[str, str]] = []
            for child in node.children:
                replacement = replacements.get((child.start_byte, child.end_byte))
                if replacement is not None:
                    # A recognized strict-shape expansion: the construct is
                    # emitted as exactly one literal argv element and its
                    # dynamic-syntax denial is consumed.
                    parts.append(replacement)
                    continue
                if part := _supported_command_part(child):
                    parts.append(part)
                elif child.type == "ansi_c_string":
                    # Preserve the token for guardrails such as find's execution
                    # predicate while requiring approval because shlex does not
                    # decode Bash ANSI-C quoting.
                    if child.text is not None:
                        parts.append(child.text.decode("utf-8"))
                elif child.type == "variable_assignment":
                    # The part text drops command-prefix assignments, but the
                    # shell applies them to this command's subprocess; record
                    # the literal bindings so the egress proxy inspection sees
                    # the command's effective environment.
                    if binding := _assignment_binding(child):
                        prefix_assignments.append(binding)
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
                    # An unrecognized construct is an unknown denial: it fails
                    # closed to POLICY and names no alternative.
                    add_reason(_policy_reason(f"unsupported syntax ({child.type})"))

            # A redirect is a sibling of the command under redirected_statement.
            # Keep the marker so standalone-command denylist behavior stays intact.
            if parts and node.parent and node.parent.type == "redirected_statement":
                parts.append("<redirect>")
            if parts:
                part_index_by_command[(node.start_byte, node.end_byte, node.type)] = (
                    len(commands)
                )
                commands.append(" ".join(parts))
                # Redirects that precede this part (redirect-only statements,
                # or anchors that emitted no part) resolve against the
                # working directory in force before this part runs.
                redirect_groups.append(list(pending_targets))
                pending_targets.clear()
                # The part's effective environment: the statement-level
                # bindings in force where it runs first, then its own
                # command-prefix assignments, which win (POSIX prefix
                # semantics — dict() keeps the last binding of a name).
                command_assignment_groups.append(
                    _union_binding_pairs(persistent_assignments, conditional_candidates)
                    + tuple(prefix_assignments)
                )

        for reason in approval_reasons - previous_reasons:
            original = (node.text or b"").decode("utf-8")
            parent = node
            while parent.parent is not None and parent.type != "command":
                parent = parent.parent
            diagnostics[reason] = replace(
                reason_diagnostics[reason],
                offending_token=original,
                command_part=(parent.text or b"").decode("utf-8"),
            )
        if node.type == "list":
            # A statement after an &&/|| connector runs conditionally — the
            # guard's exit status decides — so its declarations record
            # candidate bindings instead of overwrites. The first statement
            # and every `;`/newline-separated statement inherit this walk's
            # own conditionality.
            seen_connector = False
            for child in node.children:
                find_commands(child, conditional=conditional or seen_connector)
                if child.type in {"&&", "||"}:
                    seen_connector = True
        else:
            for child in node.children:
                find_commands(child, conditional=conditional)

    find_commands(tree.root_node)
    return ShellPermissionAnalysis(
        command_parts=tuple(commands),
        approval_reasons=tuple(sorted(approval_reasons)),
        diagnostics=tuple(
            diagnostics.get(reason)
            or replace(
                reason_diagnostics.get(reason) or Diagnostic("analysis", detail=reason),
                command_part=command,
            )
            for reason in sorted(approval_reasons)
        ),
        redirect_operands=tuple(tuple(group) for group in redirect_groups)
        + (tuple(pending_targets),),
        command_assignments=tuple(command_assignment_groups),
    )


def analyze_shell_command(
    command: str,
    *,
    nested_source: bool = False,
    scratchpad_roots: frozenset[Path] = frozenset(),
    workspace_roots: frozenset[Path] | None = None,
) -> ShellPermissionAnalysis:
    """Analyze a command, denying safely when parsing or traversal fails.

    ``scratchpad_roots`` and ``workspace_roots`` are the roots the sed
    locate-then-print simulation may read inside (workspace+scratchpad);
    ``workspace_roots`` defaults to the ambient workspace. They gate only
    that read, never an authorization.
    """
    try:
        return _analyze_shell_command(
            command,
            nested_source=nested_source,
            scratchpad_roots=scratchpad_roots,
            workspace_roots=workspace_roots,
        )
    except Exception:
        return ShellPermissionAnalysis(
            command_parts=(), approval_reasons=("shell analysis failed",)
        )
