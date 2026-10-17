"""The git inspection half of the shell-policy family: the family's only I/O module.

Repository and work-tree resolution, repository-owned config discovery and
parsing, remote-name resolution, and the push-destination and index-hook risk
vectors. Every filesystem touch in the git family lives here (``Path.cwd``,
``read_text``, ``stat``, ``is_dir``/``is_file``/``exists``); git itself is never
invoked, and every helper fails closed (None) when the inspection cannot be
completed. The module imports only ``_shell_argv``, ``_shell_git_support``,
``chartreux.core.dangerous_commands``, and ``chartreux.core.tools.utils`` --
never the facade, so the import graph stays acyclic.
``_shell_command_policy`` (the facade) re-exports every name defined here, so
import paths and object identity are unchanged by the split.
"""

from __future__ import annotations

from pathlib import Path
import re
import stat
from typing import TYPE_CHECKING

from chartreux.core.dangerous_commands import GitRepositoryRelocation, RepositoryScope
from chartreux.core.tools.builtins._shell_argv import _command_name
from chartreux.core.tools.builtins._shell_git_support import (
    _GIT_SUBCOMMAND_INDEX,
    _scan_git_globals,
    canonical_git_command,
    push_destination_operands,
)
from chartreux.core.tools.utils import resolve_tool_path

if TYPE_CHECKING:
    from chartreux.core.dangerous_commands import ModeledGitCall


def resolve_repository_scope(
    cwd: Path, relocation: GitRepositoryRelocation
) -> RepositoryScope | None:
    """Resolve the repository/worktree identity one git invocation operates on.

    Pure filesystem inspection; git is never invoked. ``cwd`` is the statically
    tracked working directory of the invocation and ``relocation`` carries
    git's global -C/--git-dir/--work-tree operands (each -C applies in turn,
    before the override paths are resolved, exactly as git applies them).
    Returns None when a repository marker is present but its identity cannot
    be determined, so callers fail closed. A directory governed by no
    repository yields that directory's nominal scope: git itself refuses such
    a call, so the nominal identity only ever matters for grant scoping.
    """
    current = (cwd if cwd.is_absolute() else Path.cwd() / cwd).resolve()
    for value in relocation.cwd_paths:
        current = resolve_tool_path(value, current)
    work_tree: Path | None = None
    if relocation.work_tree is not None:
        work_tree = resolve_tool_path(relocation.work_tree, current)
    if relocation.git_dir is not None:
        return RepositoryScope(
            git_dir=resolve_tool_path(relocation.git_dir, current), work_tree=work_tree
        )
    scope: RepositoryScope | None = None
    for directory in (current, *current.parents):
        marker = directory / ".git"
        if marker.is_dir():
            scope = RepositoryScope(
                git_dir=marker.resolve(), work_tree=work_tree or directory
            )
            break
        if (
            (directory / "HEAD").is_file()
            and (directory / "config").is_file()
            and (directory / "objects").is_dir()
            and (directory / "refs").is_dir()
        ):
            # A bare repository (or a git dir used directly) has no work tree.
            scope = RepositoryScope(git_dir=directory.resolve(), work_tree=work_tree)
            break
        if not marker.is_file():
            continue
        try:
            marker_value = marker.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        prefix, separator, value = marker_value.partition(":")
        if not separator or prefix.casefold() != "gitdir":
            # Not a usable repository pointer; git itself would refuse it, so
            # the repository identity cannot be determined.
            return None
        git_dir = Path(value.strip()).expanduser()
        if not git_dir.is_absolute():
            git_dir = directory / git_dir
        # A linked worktree's own git dir is its identity: sibling worktrees
        # of one repository never share a scope.
        scope = RepositoryScope(
            git_dir=git_dir.resolve(), work_tree=work_tree or directory
        )
        break
    if scope is None:
        scope = RepositoryScope(
            git_dir=(current / ".git").resolve(), work_tree=work_tree or current
        )
    return scope


# Routine git subcommands that write work-tree files: tracked paths under the
# work tree (checkout/switch/restore/merge/cherry-pick/revert/rebase/am and
# the whole stash subcommand, whose push/pop/apply forms all touch the work
# tree) or patch paths (apply), plus pull, whose merge step writes the work
# tree. A relocation that resolves the work tree inside the git metadata dir
# turns each of them into a metadata writer, so the resolver denies them
# there. fetch never writes the work tree and stays out; reset --keep/--merge,
# git rm/mv, bisect, and submodule stay routine under such a relocation (a
# documented residual).
_GIT_TREE_WRITING_SUBCOMMANDS = frozenset({
    "am",
    "apply",
    "cherry-pick",
    "checkout",
    "merge",
    "pull",
    "rebase",
    "restore",
    "revert",
    "stash",
    "switch",
})


def git_work_tree_writing_subcommand(tokens: list[str]) -> str | None:
    """The subcommand when one git invocation writes work-tree files.

    Returns the subcommand name for the routine tree-writing subcommands, or
    None for every other invocation — readers, unmodeled or unrecognized
    forms (already denied or governed elsewhere), and non-git commands.
    """
    canonical = canonical_git_command(tokens)
    if canonical is None or len(canonical) <= _GIT_SUBCOMMAND_INDEX:
        return None
    subcommand = canonical[1]
    return subcommand if subcommand in _GIT_TREE_WRITING_SUBCOMMANDS else None


def resolve_git_work_tree(tokens: list[str], cwd: Path) -> Path | None:
    """Resolve the work tree one git invocation writes, without invoking git.

    Mirrors git's own resolution order: the ``-C`` paths apply in turn, then
    ``--work-tree``, then ``core.worktree`` from the repository's own config
    (relative values resolve against the git dir, as git resolves them), then
    discovery — the directory holding the ``.git`` marker, or the current
    directory itself when it is a git dir (git then uses the cwd as the work
    tree). Returns None when the work tree cannot be determined, so callers
    fail closed.
    """
    globals_scan = _scan_git_globals(tokens[1:])
    if globals_scan.error is not None:
        return None
    relocation = globals_scan.relocation
    current = (cwd if cwd.is_absolute() else Path.cwd() / cwd).resolve()
    for value in relocation.cwd_paths:
        current = resolve_tool_path(value, current)
    if relocation.work_tree is not None:
        # GIT_WORK_TREE wins over core.worktree and discovery.
        return resolve_tool_path(relocation.work_tree, current)
    scope = resolve_repository_scope(
        current, GitRepositoryRelocation(git_dir=relocation.git_dir)
    )
    if scope is None:
        return None
    # An explicit git dir without --work-tree, or a git dir discovered as the
    # cwd itself, leaves git using the cwd as the work tree.
    work_tree = scope.work_tree if scope.work_tree is not None else current
    config_paths = _git_dir_config_paths(scope.git_dir)
    entries = (
        _parse_git_config_entries(config_paths) if config_paths is not None else None
    )
    if entries is None:
        # An unreadable commondir pointer or an unparsable config: the work
        # tree cannot be verified, so fail closed.
        return None
    configured = next(
        (
            value
            for section, key, value in entries
            if section == "core" and key == "worktree"
        ),
        None,
    )
    if configured is None:
        return work_tree
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = scope.git_dir / path
    return path.resolve()


def git_cwd_relocations(tokens: list[str], cwds: set[Path]) -> set[Path] | None:
    """The directories git's own -C sequence relocates each candidate cwd to.

    Git applies its -C operands in turn — each relative to the directory the
    previous one produced — before the subcommand runs, so every relative
    operand of the subcommand, raw write operands included, resolves against
    the final directory (mirroring how ``resolve_git_work_tree`` models the
    same sequence). Returns None when the invocation carries no -C, so those
    operands resolve against the tracked cwds unchanged. An unscannable global
    sequence never reaches operand resolution — the policy layer denies it
    before the operand guard runs — and an empty -C value is a no-op
    relocation git itself accepts, so the tracked cwds stay correct there too.
    """
    globals_scan = _scan_git_globals(tokens[1:])
    if globals_scan.error is not None or not globals_scan.relocation.cwd_paths:
        return None
    relocated: set[Path] = set()
    for cwd in cwds:
        current = (cwd if cwd.is_absolute() else Path.cwd() / cwd).resolve()
        for value in globals_scan.relocation.cwd_paths:
            current = resolve_tool_path(value, current)
        relocated.add(current)
    return relocated


_GIT_CONFIG_ENTRY = re.compile(
    r"^(?P<key>[A-Za-z][A-Za-z0-9-]*)\s*(?:=\s*(?P<value>.*))?$"
)
_FALSE_GIT_CONFIG_VALUES = frozenset({"", "0", "false", "no", "off"})
# The section name is casefolded on parse; the remediation text restores the
# spelling git documents for the directive.
_INCLUDE_DIRECTIVE_NAMES = {"include": "include", "includeif": "includeIf"}


def _git_config_paths(cwd: Path) -> tuple[Path, ...] | None:  # noqa: PLR0911
    """Locate repository-owned config without invoking Git.

    Returns None when a repository is present but its config location cannot be
    determined, so callers fail closed. Returns () when no repository governs
    ``cwd``. Worktrees point at the shared config through ``commondir``; the
    worktree-specific ``config.worktree`` is read alongside it.
    """
    for directory in (cwd, *cwd.parents):
        marker = directory / ".git"
        if marker.is_dir():
            return marker / "config", marker / "config.worktree"
        if (
            (directory / "HEAD").is_file()
            and (directory / "config").is_file()
            and (directory / "objects").is_dir()
            and (directory / "refs").is_dir()
        ):
            return directory / "config", directory / "config.worktree"
        if not marker.is_file():
            continue
        try:
            marker_value = marker.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        prefix, separator, value = marker_value.partition(":")
        if not separator or prefix.casefold() != "gitdir":
            # Not a usable repository pointer; git itself would refuse it.
            return ()
        git_dir = Path(value.strip()).expanduser()
        if not git_dir.is_absolute():
            git_dir = directory / git_dir
        try:
            common_value = (git_dir / "commondir").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            # No commondir: a --separate-git-dir checkout, whose gitdir's own
            # config is the repository config. A linked worktree always has a
            # commondir, and git refuses to operate in one without it.
            return git_dir / "config", git_dir / "config.worktree"
        except (OSError, UnicodeError):
            # An unreadable commondir would fall back to the worktree gitdir's
            # own config, which does not exist in a linked worktree; fail
            # closed like the .git-pointer and config reads.
            return None
        common_dir = Path(common_value)
        if not common_dir.is_absolute():
            common_dir = git_dir / common_dir
        return common_dir / "config", git_dir / "config.worktree"
    return ()


def git_metadata_paths(tokens: list[str], *, cwd: Path) -> tuple[Path, ...] | None:
    """Resolve repository metadata identity without running Git or its helpers.

    Include the ambient repository as well as global-option overrides. An explicit
    git-dir is protected even if it is not yet a complete repository.
    """
    roots = {cwd}
    explicit_dirs: set[Path] = set()
    override_values: list[tuple[str, str]] = []
    if tokens and _command_name(tokens[0]) == "git":
        current = cwd
        index = 1
        while index < len(tokens) and tokens[index].startswith("-"):
            token = tokens[index]
            name, separator, value = token.partition("=")
            if token == "-C" or name in {"--git-dir", "--work-tree"}:
                if not separator:
                    index += 1
                    if index >= len(tokens):
                        return None
                    value = tokens[index]
                path = Path(value).expanduser()
                path = (path if path.is_absolute() else current / path).resolve()
                if token == "-C":
                    current = path
                elif name == "--git-dir":
                    explicit_dirs.add(path)
                    override_values.append((name, value))
                else:
                    roots.add(path)
                    override_values.append((name, value))
            elif token.startswith("-C"):
                current = (current / token[2:]).resolve()
            index += 1
        roots.add(current)
        # Git applies -C before resolving repository override paths, even when
        # the override appeared earlier on the command line.
        for name, value in override_values:
            path = Path(value).expanduser()
            path = (path if path.is_absolute() else current / path).resolve()
            (explicit_dirs if name == "--git-dir" else roots).add(path)
    metadata = set(explicit_dirs)
    for root in roots:
        configs = _git_config_paths(root)
        if configs is None:
            return None
        metadata.update(config.parent.resolve() for config in configs)
    # Follow commondir for explicit git-dir and symlinked .git directories too.
    for git_dir in tuple(metadata):
        try:
            value = (git_dir / "commondir").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError):
            return None
        common = Path(value)
        metadata.add((common if common.is_absolute() else git_dir / common).resolve())
    return tuple(metadata)


def _parse_git_config_entries(
    config_paths: tuple[Path, ...],
) -> tuple[tuple[str, str, str], ...] | None:
    """Parse config file entries without invoking Git.

    None means one of the files exists but could not be read or parsed;
    callers fail closed on that result.
    """
    entries: list[tuple[str, str, str]] = []
    for config_path in config_paths:
        if not config_path.exists():
            continue
        try:
            lines = config_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return None
        section = ""
        for raw_line in lines:
            line = raw_line.strip().lstrip("\ufeff")
            if not line or line.startswith(("#", ";")):
                continue
            if line.startswith("["):
                header, separator, line = line.partition("]")
                if not separator or not header[1:].strip():
                    return None
                section = (
                    header[1:].split(maxsplit=1)[0].split(".", maxsplit=1)[0].casefold()
                )
                line = line.strip()
                if not line or line.startswith(("#", ";")):
                    continue
            if match := _GIT_CONFIG_ENTRY.fullmatch(line):
                value = match.group("value")
                entries.append((
                    section,
                    match.group("key").casefold(),
                    # A valueless bare key is true; an explicit empty value
                    # after "=" is unset, which git parses as false.
                    ("true" if value is None else value).strip().strip('"'),
                ))
            else:
                # Continuations and unknown syntax can conceal executable keys.
                return None
    return tuple(entries)


def _git_config_entries(cwd: Path) -> tuple[tuple[str, str, str], ...] | None:
    """Parse repository-owned config entries without invoking Git.

    None means a repository governs ``cwd`` but its config could not be read;
    callers fail closed on that result.
    """
    config_paths = _git_config_paths(cwd)
    if config_paths is None:
        return None
    return _parse_git_config_entries(config_paths)


# One [section] / [section "sub"] / [section.sub] header with the
# subsection captured: remote names live in the subsection, which the entry
# parser's top-level section fold drops.
_GIT_CONFIG_SECTION_HEADER = re.compile(
    r"\[\s*(?P<section>[A-Za-z][A-Za-z0-9-]*)"
    r"(?:\s+\"(?P<quoted>[^\"]*)\"|\.(?P<dotted>[^\]\s]+))?"
    r"\s*\]"
)


def _git_dir_config_paths(git_dir: Path) -> tuple[Path, ...] | None:
    """Locate the config files one git dir reads, without invoking Git.

    A linked worktree's git dir shares the common repository's config
    through ``commondir`` and adds its own ``config.worktree``; every other
    git dir reads its own ``config`` (plus ``config.worktree`` when the
    worktree-config extension is in use). Returns None when a commondir
    pointer exists but cannot be read, so callers fail closed.
    """
    try:
        common_value = (git_dir / "commondir").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return git_dir / "config", git_dir / "config.worktree"
    except (OSError, UnicodeError):
        # An unreadable commondir would fall back to the worktree gitdir's
        # own config, which does not exist in a linked worktree; fail closed
        # like the config reads.
        return None
    common_dir = Path(common_value)
    if not common_dir.is_absolute():
        common_dir = git_dir / common_dir
    return common_dir / "config", git_dir / "config.worktree"


def _git_config_remote_url_names(
    config_paths: tuple[Path, ...],
) -> frozenset[str] | None:
    """Remote names carrying a configured ``remote.<name>.url``, or None.

    Parses the same repository-owned config grammar as
    ``_git_config_entries``, keeping the section subsection the top-level
    fold drops. A remote name resolves only when its ``url`` key is
    present, because git resolves any destination that is not a configured
    remote name as a URL. None means a config could not be read or parsed,
    so callers fail closed.
    """
    names: set[str] = set()
    for config_path in config_paths:
        if not config_path.exists():
            continue
        try:
            lines = config_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return None
        remote_name: str | None = None
        for raw_line in lines:
            line = raw_line.strip().lstrip("\ufeff")
            if not line or line.startswith(("#", ";")):
                continue
            if line.startswith("["):
                header, separator, line = line.partition("]")
                if not separator:
                    return None
                match = _GIT_CONFIG_SECTION_HEADER.fullmatch(f"{header}]")
                if match is None:
                    # Unknown header syntax can conceal a remote definition.
                    return None
                section = match.group("section").casefold()
                subsection = match.group("quoted")
                if subsection is None:
                    subsection = match.group("dotted")
                remote_name = subsection if section == "remote" else None
                line = line.strip()
                if not line or line.startswith(("#", ";")):
                    continue
            if match := _GIT_CONFIG_ENTRY.fullmatch(line):
                if remote_name is not None and match.group("key").casefold() == "url":
                    names.add(remote_name)
            else:
                # Continuations and unknown syntax can conceal a url key.
                return None
    return frozenset(names)


def git_configured_remote_names(scope: RepositoryScope) -> frozenset[str] | None:
    """Remote names one repository scope resolves from its own config.

    Pure filesystem inspection; git is never invoked. A push destination
    operand resolves to a remote only when the repository's config carries
    ``remote.<name>.url`` for it: git resolves any destination that is not
    a configured remote name as a URL — a path, for a bare name — so a
    remote with only ``remote.<name>.mirror`` or ``remote.<name>.push``
    config is NOT resolvable. None means the repository's config could not
    be read, so callers fail closed. The scope's git dir is the config
    source: global and system config are not consulted, because the grant
    scopes the repository whose own config defines its remotes.
    """
    config_paths = _git_dir_config_paths(scope.git_dir)
    if config_paths is None:
        return None
    return _git_config_remote_url_names(config_paths)


def git_push_destination_risk(
    call: ModeledGitCall, scope: RepositoryScope
) -> str | None:
    """The unconfigured-destination vector of one push call in one scope.

    A push destination operand that is a bare name resolves only through the
    repository's configured remotes: git resolves any destination that is
    not a configured remote name as a URL — a path, for a bare name — so a
    push to an unconfigured name runs the destination's receive-pack and
    hooks, outside the repository the grant covers. The first positional is
    always the repository and ``--repo`` overrides it, so both operands are
    verified.

    Returns None when the call is not a push, carries no destination
    operand (git then resolves the destination from configuration the
    agent cannot write), or every destination resolves to a configured
    remote of the scope. A scope whose git dir does not exist governs no
    repository — git itself refuses the push there — so it is not verified.
    """
    if call.subcommand != "push":
        return None
    destinations = push_destination_operands(call)
    if not destinations:
        return None
    if not scope.git_dir.is_dir():
        # No repository governs this scope: git itself refuses the push
        # from such a directory, so no destination can run there.
        return None
    remotes = git_configured_remote_names(scope)
    if remotes is None:
        repository = scope.work_tree or scope.git_dir
        return (
            f"the configuration of the repository at {repository} could not "
            "be read, so the push destination cannot be verified as a "
            "configured remote; failing closed"
        )
    for destination in destinations:
        if destination not in remotes:
            repository = scope.work_tree or scope.git_dir
            return (
                f"destination {destination!r} is not a configured remote of "
                f"the repository at {repository}: git resolves a push "
                "destination that is not a configured remote name as a URL, "
                "so a push to it runs the destination's receive-pack and "
                "hooks, outside the repository the grant covers; no grant "
                "can authorize an unconfigured destination"
            )
    return None


def _git_value_is_active(value: str) -> bool:
    return value.casefold() not in _FALSE_GIT_CONFIG_VALUES


def _git_index_hook_risk(
    cwd: Path, entries: tuple[tuple[str, str, str], ...]
) -> str | None:
    # Readers such as status/diff can refresh and write the index. Conservatively
    # inspect every modeled reader, including stash's index-writing operations.
    # Do not interpret hooksPath quoting/expansion: any override fails closed.
    if any(section == "core" and key == "hookspath" for section, key, _ in entries):
        return "repository git config may execute arbitrary code via core.hooksPath"
    config_paths = _git_config_paths(cwd)
    if config_paths is None:
        return "repository git hooks cannot be inspected; failing closed"
    for config_path in config_paths:
        hook = config_path.parent / "hooks" / "post-index-change"
        try:
            mode = hook.stat().st_mode
        except FileNotFoundError:
            continue
        except OSError:
            return "repository git hooks cannot be inspected; failing closed"
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            return (
                "repository git hooks may execute arbitrary code via post-index-change"
            )
    return None
