"""Git-state assessment for the recursive-`rm` operand analysis (v0.5.9 P2-c).

The pure-filesystem inspection family (``_shell_git_inspection``) never
invokes git. This module deliberately does — for exactly three facts that
filesystem inspection cannot establish: whether a repository's work tree is
clean (``git status --porcelain --untracked-files=all`` reports nothing),
whether it carries commits no remote-tracking ref reaches (``git rev-list
--count --all --not --remotes``), and whether any remote is configured to
hold a recovery copy (``git remote`` plus ``git remote get-url`` for each
configured remote). The plan's saved/unsaved split is stated in ``git
status`` terms, and re-deriving status semantics over the index, the object
database and the ignore rules would be a re-derivation, not a reuse. The
remote check gates the unpushed criterion: a remote-tracking ref is local
metadata any writer can author with ``git update-ref``, so it counts as
evidence of a recovery copy only when a remote exists to hold one — and a
remote whose fetch URL is a LOCAL path (a plain path or ``file://``) counts
only when git can actually fetch from it (a sanitized ``git ls-remote``
against the resolved target succeeds — a repository or a bundle file, never
a plain file or a non-repository directory), because a dead or non-remote
local-path target leaves stale remote-tracking refs reading as
recovery-established with no copy behind them. A local-scheme fetch target
must also survive the deletion itself: recovery evidence that lies inside
(or equals) any operand path the command deletes — the operand holding the
at-risk repository and every other operand alike — is not recovery
evidence, so the gate rejects a remote whose resolved target the deletion
footprint covers. Remote-scheme URLs (``http(s)://``, ``ssh://``, ``git@``
and friends) count as evidence exactly as a name-only check counted them:
they cannot be verified locally, and cannot lie inside a filesystem
deletion, so the gate is asymmetric by construction — local-scheme URLs are
fetchability-checked with git itself and footprint-checked against the
resolved operand paths, remote-scheme URLs are trusted unverified.

Repository discovery recognizes bare repositories as well as work-tree
repositories: a directory carrying a ``HEAD`` file plus ``objects/`` and
``refs/`` directories (git's own heuristic; all three required) is a
repository whose git dir is the directory itself and which has no work tree
to keep clean — its recovery is assessed through its own git dir, its own
rev-list, and its own configured remotes exactly like a nested work-tree
repository. Metadata inside the operand requires recovery established or
the command resolves handoff, fail-closed on anything unestablishable.

Every invocation is sanitized the way ``chartreux.core.git.repo`` sanitizes
its own pre-trust commands (hooks path neutralized, fsmonitor disabled, no
optional locks so the repository is never written) plus a from-scratch
environment that disables system and global config, so no user- or
repository-owned configuration can select a program, hook, or pager through
it. ``git status`` is not a metadata-only command — it can pass work-tree
contents through repository-configured clean/process filters — so a
repository whose own config selects any program, or includes another config
file the pure parser does not follow, is never inspected by subprocess at
all: it fails closed, exactly like the reader flow's config-risk guard.

Every helper fails closed (None/False) when the state cannot be established.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
import subprocess

from chartreux.core.dangerous_commands import GitRepositoryRelocation, RepositoryScope
from chartreux.core.tools.builtins._shell_git_inspection import (
    _git_config_entries,
    _git_dir_config_paths,
    _git_value_is_active,
    _parse_git_config_entries,
    resolve_repository_scope,
)
from chartreux.utils.platform import resolve_git_executable

_GIT_STATE_TIMEOUT_SECONDS = 10.0
# The same neutralizations as chartreux.core.git.repo.sanitized_git: the
# hooks path is absolute and absent, so it cannot resolve inside the
# repository; an empty fsmonitor value disables the daemon hook.
_NO_HOOKS_CONFIG = "core.hooksPath=/nonexistent-chartreux-disabled-git-hooks"
# A git dir that holds no configuration. The fetchability probe runs with it
# as --git-dir so that NO repository configuration — the assessed
# repository's own included — can rewrite the probed URL
# (url.<base>.insteadOf) into a different target, least of all a remote one:
# the probe contacts exactly the resolved local URL a fetch would contact,
# and never the network.
_NO_CONFIG_GIT_DIR = "/nonexistent-chartreux-remote-fetchability-probe"
# Filesystem markers of an in-progress rebase/am/merge/cherry-pick/revert,
# a bisect, or a sequencer todo. A leftover marker after a crashed operation
# fails closed exactly like a live one.
_IN_PROGRESS_MARKERS = (
    "rebase-merge",
    "rebase-apply",
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "BISECT_LOG",
    "sequencer",
)
# Repository-config keys that select a program git may run while reading
# work-tree state. A repository setting any of them to an active value is
# never inspected by subprocess; include directives fail closed because git
# would follow them and the pure parser does not.
_PROGRAM_SELECTION_KEYS = frozenset({
    ("core", "hookspath"),
    ("core", "editor"),
    ("core", "fsmonitor"),
    ("core", "pager"),
    ("sequence", "editor"),
    ("filter", "clean"),
    ("filter", "process"),
    ("merge", "driver"),
    ("diff", "command"),
    ("diff", "external"),
    ("diff", "textconv"),
    ("gpg", "program"),
})
# A walk budget for discovering repositories inside one operand: a crafted
# tree must not turn permission resolution into an unbounded traversal.
_MAX_STATE_WALK_ENTRIES = 4096


@dataclass(frozen=True)
class RepositoryState:
    """One repository's resolved identity for the rm operand analysis."""

    # None for a bare repository: it has no work tree, so there is no
    # clean-tree check to run — its recovery is assessed through its git
    # dir, its rev-list, and its remotes alone.
    work_tree: Path | None
    git_dir: Path
    # Every path whose deletion would damage this repository: its git dir,
    # the linked-worktree common dir, and the work-tree .git marker (a
    # directory for a plain repository, a gitdir pointer file otherwise).
    # A bare repository's git dir is the directory itself, so its own path
    # is the metadata; it has no work-tree marker to add.
    metadata_dirs: frozenset[Path]


def repository_state(scope: RepositoryScope) -> RepositoryState | None:
    """Resolve one repository scope into the state the rm analysis needs.

    None means the metadata layout cannot be established (no work tree, a git
    dir that is not a directory, or an unreadable commondir pointer), so
    callers fail closed.
    """
    if scope.work_tree is None or not scope.git_dir.is_dir():
        return None
    config_paths = _git_dir_config_paths(scope.git_dir)
    if config_paths is None:
        return None
    common_dir = config_paths[0].parent
    return RepositoryState(
        work_tree=scope.work_tree,
        git_dir=scope.git_dir,
        metadata_dirs=frozenset({scope.git_dir, common_dir, scope.work_tree / ".git"}),
    )


def _path_exists(path: Path) -> bool | None:
    """Whether a path exists; None means the check itself failed.

    A component that is not a directory (ENOTDIR) reads as absent: no marker
    can live under a file.
    """
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return None
    return True


def _path_is_file(path: Path) -> bool | None:
    """Whether a path exists as a regular file; None means the probe failed."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return None


def _path_is_dir(path: Path) -> bool | None:
    """Whether a path exists as a directory; None means the probe failed."""
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return None


def _in_progress_operation(state: RepositoryState) -> bool | None:
    """Whether any metadata directory carries an in-progress operation marker.

    None means a marker's presence cannot be established; callers fail closed.
    """
    for directory in sorted(state.metadata_dirs):
        for marker in _IN_PROGRESS_MARKERS:
            present = _path_exists(directory / marker)
            if present is None:
                return None
            if present:
                return True
    return False


def _config_selects_program(state: RepositoryState) -> bool | None:
    """Whether the repository's own config can run a program in our invocation.

    None means the config could not be read or parsed; callers fail closed.
    A bare repository has no work tree to walk: its config is its git dir's
    own config (plus the linked-worktree common config when a commondir
    pointer says so), read directly.
    """
    if state.work_tree is None:
        config_paths = _git_dir_config_paths(state.git_dir)
        entries = (
            _parse_git_config_entries(config_paths)
            if config_paths is not None
            else None
        )
    else:
        entries = _git_config_entries(state.work_tree)
    if entries is None:
        return None
    for section, _key, _value in entries:
        if section in {"include", "includeif"}:
            return True
    for section, key, value in entries:
        if (section, key) in _PROGRAM_SELECTION_KEYS and _git_value_is_active(value):
            return True
    return False


def _git_state_environment() -> dict[str, str]:
    """The from-scratch environment every git-state invocation runs under.

    System and global config are disabled, so no user-owned configuration can
    select a program, hook, or pager through the invocation.
    """
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": os.environ.get("HOME", str(Path.home())),
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }


def _run_git_state(
    state: RepositoryState, args: list[str], *, work_tree: bool
) -> bytes | None:
    """Run one sanitized git command; None means failure (fail closed).

    The invocation is the sanitized form of a metadata-only read: an explicit
    git dir, no optional locks (the repository is never written), a disabled
    hooks path and fsmonitor, and a from-scratch environment that disables
    system and global config, so no repository- or user-owned configuration
    can select a program, hook, or pager through it.
    """
    executable = resolve_git_executable(cwd=state.work_tree)
    if executable is None:
        return None
    command = [
        executable,
        "--git-dir",
        str(state.git_dir),
        "--no-optional-locks",
        "-c",
        _NO_HOOKS_CONFIG,
        "-c",
        "core.fsmonitor=",
    ]
    if work_tree and state.work_tree is not None:
        command += ["--work-tree", str(state.work_tree)]
    command += args
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=_GIT_STATE_TIMEOUT_SECONDS,
            env=_git_state_environment(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _work_tree_is_clean(state: RepositoryState) -> bool | None:
    """Whether git status reports no change, no staged work, no untracked file.

    Ignored artifacts are invisible to git status; losing them on a clean-tree
    run is the accepted residual risk of the saved/unsaved split.
    """
    output = _run_git_state(
        state, ["status", "--porcelain", "--untracked-files=all"], work_tree=True
    )
    if output is None:
        return None
    return output.strip() == b""


def _unpushed_commit_count(state: RepositoryState) -> int | None:
    """Commits no remote-tracking ref reaches, or None when unestablishable."""
    output = _run_git_state(
        state, ["rev-list", "--count", "--all", "--not", "--remotes"], work_tree=False
    )
    if output is None:
        return None
    try:
        return int(output.decode("ascii").strip())
    except (ValueError, UnicodeError):
        return None


_REMOTE_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _remote_url_is_local(url: str) -> bool:
    """Whether a remote's fetch URL names a local path.

    A plain path (relative or absolute) and ``file://`` URLs are local; any
    other scheme (``http(s)://``, ``ssh://``, ``git://``, ...) names a remote
    host, and so does scp-like syntax (``git@host:path`` — a colon in the
    first path component, exactly as git resolves it).
    """
    if url.startswith("file://"):
        return True
    if _REMOTE_URL_SCHEME.match(url) is not None:
        return False
    head = url.partition("/")[0]
    return ":" not in head


# The length of one percent-encoded sequence ("%XX"): git's
# url.c:url_decode_internal decodes a '%' only when at least this many
# characters remain from it.
_PERCENT_SEQUENCE_LENGTH = 3


def _git_percent_decode(text: str) -> str:
    """Percent-decode a ``file://`` URL path exactly as git decodes it.

    git's connect-time URL handling (``connect.c:parse_connect_url``) runs
    the whole URL through ``url.c:url_decode`` before splitting host and
    path, so the fetch target of a ``file://`` remote is the DECODED path.
    The decoder reproduces ``url.c:url_decode_internal`` exactly: a ``%``
    followed by two hex digits (either case) decodes to that byte in a
    single pass, ``+`` stays literal (``decode_plus`` is 0), and anything
    else — an invalid or truncated sequence, or ``%00`` (a NUL byte,
    rejected by git's ``0 < val`` test) — stays literal. Verified
    empirically against git 2.55.0: ``git ls-remote`` on
    ``file://<root>/%76endor/nested/.git`` succeeds and its failure messages
    print the decoded path, while ``%zz``/trailing ``%`` stay literal and
    ``%2576`` decodes once to ``%76``. The decoded bytes are read as UTF-8
    with surrogateescape so byte-exact paths compare equal to the
    filesystem paths the deletion footprint resolves.
    """
    if "%" not in text:
        return text
    raw = bytearray()
    index = 0
    length = len(text)
    while index < length:
        if text[index] == "%" and length - index >= _PERCENT_SEQUENCE_LENGTH:
            high = _hex_digit_value(text[index + 1])
            low = _hex_digit_value(text[index + 2])
            if high >= 0 and low >= 0 and (value := (high << 4) | low) > 0:
                raw.append(value)
                index += _PERCENT_SEQUENCE_LENGTH
                continue
        raw.extend(text[index].encode("utf-8", "surrogateescape"))
        index += 1
    return raw.decode("utf-8", "surrogateescape")


def _hex_digit_value(char: str) -> int:
    """The value of one hex digit (either case), or -1 when not a digit."""
    if "0" <= char <= "9":
        return ord(char) - ord("0")
    if "a" <= char <= "f":
        return ord(char) - ord("a") + 10
    if "A" <= char <= "F":
        return ord(char) - ord("A") + 10
    return -1


def _local_remote_is_fetchable(url: str, state: RepositoryState) -> bool:
    """Whether a local-scheme remote URL names a genuinely fetchable remote.

    A ``file://`` URL names its path after the prefix (an absolute path; a
    host-qualified remainder is unestablishable and reads as absent, fail
    closed). A plain relative path resolves against the repository itself —
    the work tree, or the bare directory for a bare repository — which is
    the base git's own relative-URL resolution approximates. The resolved
    target is then verified with a sanitized ``git ls-remote``: git accepts
    only a repository (bare or with a work tree) or a bundle file, so a
    plain file or a directory that is not a repository is not a remote any
    fetch could read — the recovery gate never approves a copy behind a
    target no fetch can contact. The probe runs with a config-free git dir
    (``_NO_CONFIG_GIT_DIR``) so no repository configuration can rewrite the
    probed URL, and only local-scheme URLs are ever probed, so the check
    never touches the network. Any failure reads as not fetchable, fail
    closed. The probe hands the URL itself to git, and git percent-decodes
    ``file://`` URLs at connect time (see ``_git_percent_decode``), so the
    fetchability side already sees the decoded target — only the survival
    containment in ``_local_remote_target_paths`` must decode it too.
    """
    if url.startswith("file://"):
        path_text = url[len("file://") :]
        if not path_text.startswith("/"):
            return False
        target = url
    else:
        path = Path(url).expanduser()
        if not path.is_absolute():
            path = (state.work_tree or state.git_dir) / path
        target = str(path)
    executable = resolve_git_executable(cwd=state.work_tree)
    if executable is None:
        return False
    command = [
        executable,
        "--git-dir",
        _NO_CONFIG_GIT_DIR,
        "--no-optional-locks",
        "-c",
        _NO_HOOKS_CONFIG,
        "-c",
        "core.fsmonitor=",
        "ls-remote",
        target,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=_GIT_STATE_TIMEOUT_SECONDS,
            env=_git_state_environment(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _local_remote_target_paths(
    url: str, state: RepositoryState
) -> tuple[Path, Path] | None:
    """The (lexical, symlink-resolved) filesystem paths a local remote names.

    The resolution mirrors ``_local_remote_is_fetchable`` exactly: a
    ``file://`` URL names its path after the prefix (a host-qualified
    remainder is unestablishable and reads as None, fail closed); a plain
    relative path resolves against the repository itself — the work tree,
    or the bare directory for a bare repository. A ``file://`` path is
    percent-decoded exactly as git decodes it at connect time
    (``_git_percent_decode``), so the survival containment sees the same
    path git actually fetches from — an encoded component
    (``file://<root>/%76endor/...``) names ``<root>/vendor/...``, not a
    literal ``%76endor`` directory. The lexical path is normalized without
    following symlinks and the resolved path follows them, so a target
    reached through a symlinked component is caught on either side. None
    means the URL names no establishable path.
    """
    if url.startswith("file://"):
        path_text = url[len("file://") :]
        if not path_text.startswith("/"):
            return None
        path = Path(_git_percent_decode(path_text))
    else:
        path = Path(url).expanduser()
        if not path.is_absolute():
            path = (state.work_tree or state.git_dir) / path
    lexical = Path(os.path.normpath(str(path)))
    # Resolve the original spelling: lexical normalization first would erase
    # symlink/.. traversal and inspect a different target from git's fetch.
    return lexical, path.resolve()


def _remote_evidence_survives_deletion(
    url: str, state: RepositoryState, deletion_footprint: frozenset[Path]
) -> bool:
    """Whether a local remote's fetch target survives the whole deletion.

    Recovery evidence the deletion itself destroys is not recovery
    evidence: the target may not lie inside (or equal) any operand path
    being deleted — neither the operand holding the at-risk repository
    nor any other operand of the same command. Both the lexical and the
    symlink-resolved location are checked against the resolved operand
    paths. A URL whose path cannot be established fails closed.
    """
    paths = _local_remote_target_paths(url, state)
    if paths is None:
        return False
    return not any(
        location.is_relative_to(operand)
        for location in paths
        for operand in deletion_footprint
    )


def _remote_urls_hold_copy(
    urls: list[str], state: RepositoryState, deletion_footprint: frozenset[Path]
) -> bool:
    """Whether one remote's fetch URLs can hold a surviving recovery copy.

    A remote-scheme URL counts as evidence (it cannot be verified locally,
    and cannot lie inside a filesystem deletion — the documented
    asymmetry of the gate); a local-scheme URL counts only when git can
    actually fetch from it AND its fetch target survives the entire
    deletion footprint. A remote with no URL at all holds nothing.
    """
    for url in urls:
        if _remote_url_is_local(url):
            if _local_remote_is_fetchable(
                url, state
            ) and _remote_evidence_survives_deletion(url, state, deletion_footprint):
                return True
        else:
            return True
    return False


def _remote_fetch_urls(state: RepositoryState, name: str) -> list[str] | None:
    """One remote's fetch URLs, or None when the read fails closed.

    ``git remote get-url`` applies the repository's own insteadOf
    rewrites, so the URLs are the ones git would actually contact. A
    nonzero exit, an unparseable read, or a remote with no fetch URL at
    all reads as None; callers fail closed on it.
    """
    output = _run_git_state(state, ["remote", "get-url", name], work_tree=False)
    if output is None:
        return None
    try:
        urls = [
            line.strip() for line in output.decode("utf-8").splitlines() if line.strip()
        ]
    except UnicodeError:
        return None
    return urls or None


def _has_configured_remote(
    state: RepositoryState, deletion_footprint: frozenset[Path]
) -> bool | None:
    """Whether at least one remote is configured to hold a recovery copy.

    A remote-tracking ref is local metadata: a fixture can author one with
    ``git update-ref`` and no remote copy of the commits exists anywhere, so
    the unpushed criterion may read ``--remotes`` only when a remote is
    configured to hold one. The gate inspects each configured remote's FETCH
    URL (``git remote get-url``, sanitized like every invocation here), not
    just its name: a remote whose fetch URL is a local path counts as
    evidence only when git can actually fetch from it (a sanitized
    ``git ls-remote`` against the resolved target succeeds), so a dead
    local-path remote — or a local path that is not a repository at all —
    leaves stale remote-tracking refs reading as recovery-established with
    no copy behind them, and a fetchable local target counts only when it
    survives the entire deletion footprint, so a copy the deletion itself
    destroys reads as no copy at all. None means the check failed — a
    nonzero or unparseable URL read fails closed — so callers fail closed.
    """
    names = _run_git_state(state, ["remote"], work_tree=False)
    if names is None:
        return None
    try:
        remote_names = names.decode("utf-8").split()
    except UnicodeError:
        return None
    for name in remote_names:
        urls = _remote_fetch_urls(state, name)
        if urls is None:
            return None
        if _remote_urls_hold_copy(urls, state, deletion_footprint):
            return True
    return False


def repository_recovery(
    state: RepositoryState,
    *,
    commits_at_risk: bool,
    deletion_footprint: frozenset[Path],
) -> bool:
    """Whether this repository's recovery is established for one deletion.

    Recovery is established when the repository shows no in-progress
    rebase/merge/cherry-pick/revert/bisect, its work tree is clean (tracked
    content is recoverable from git), and — when the operand would destroy
    its Git metadata — at least one remote is configured to hold a recovery
    copy that survives the entire deletion footprint (no operand path being
    deleted may contain or equal the remote's fetch target) and no commit
    exists that no remote-tracking ref reaches. A bare repository has no
    work tree, so the clean-tree check is vacuous for it; every other axis
    applies unchanged. Every unestablishable aspect fails closed.
    """
    in_progress = _in_progress_operation(state)
    if in_progress is None or in_progress:
        return False
    selects_program = _config_selects_program(state)
    if selects_program is None or selects_program:
        return False
    if state.work_tree is not None:
        clean = _work_tree_is_clean(state)
        if clean is None or not clean:
            return False
    if commits_at_risk:
        remote = _has_configured_remote(state, deletion_footprint)
        if remote is None or not remote:
            return False
        unpushed = _unpushed_commit_count(state)
        if unpushed is None or unpushed > 0:
            return False
    return True


def _repository_state_at(root: Path) -> RepositoryState | None:
    """The repository rooted exactly at ``root``, or None.

    None covers both readings — no repository is rooted here, and a marker
    that exists but cannot be resolved — because the caller distinguishes
    them with ``_path_exists`` and fails closed on the second.
    """
    scope = resolve_repository_scope(root, GitRepositoryRelocation())
    if scope is None:
        return None
    return repository_state(scope)


def _bare_markers_present(root: Path) -> bool | None:
    """Whether the bare-repository markers are all present at ``root``.

    Git's own heuristic — a ``HEAD`` file plus ``objects/`` and ``refs/``
    directories, all three required — reads True; a directory missing any
    marker reads False (no bare repository is rooted there); None means a
    probe itself failed, so callers fail closed.
    """
    head = _path_is_file(root / "HEAD")
    objects = _path_is_dir(root / "objects")
    refs = _path_is_dir(root / "refs")
    if head is None or objects is None or refs is None:
        return None
    return head and objects and refs


def _bare_repository_state(root: Path) -> RepositoryState | None:
    """The bare repository rooted exactly at ``root``, or None.

    A bare repository carries its Git metadata directly in the directory
    itself — git's own heuristic, a ``HEAD`` file plus ``objects/`` and
    ``refs/`` directories, all three required — so its git dir is the
    directory and it has no work tree. None means the metadata layout
    cannot be established (an unreadable commondir pointer); the caller has
    already probed the three markers, so None here fails closed.
    """
    config_paths = _git_dir_config_paths(root)
    if config_paths is None:
        return None
    common_dir = config_paths[0].parent
    return RepositoryState(
        work_tree=None, git_dir=root, metadata_dirs=frozenset({root, common_dir})
    )


def repositories_intersecting(  # noqa: PLR0911
    operand: Path, assessed: RepositoryState
) -> list[RepositoryState] | None:
    """Repositories inside the assessed work tree whose work tree intersects
    the operand: rooted at the operand or deeper, or enclosing it.

    Bare repositories join the analysis like any nested repository: a
    directory carrying a ``HEAD`` file plus ``objects/`` and ``refs/``
    directories is a repository whose git dir is the directory itself, and
    metadata inside the operand requires its recovery established or the
    command resolves handoff. The assessed repository itself is excluded
    (its recovery is established by the caller: its metadata is protected
    by the overlap check, so its commits are not at risk). Repositories
    above the assessed work tree are out of scope: the operand lies wholly
    within the assessed work tree, where an ancestor sees only untracked,
    ignored, or gitlinked content whose loss is the accepted residual of
    the split. None means the discovery failed closed (an unreadable
    marker, an unresolvable repository identity, or a walk budget
    exceeded).
    """
    work_tree = assessed.work_tree
    if work_tree is None:
        # The assessed repository always carries a work tree; a state
        # without one cannot scope the enclosure walk, so fail closed.
        return None
    found: dict[tuple[Path, Path | None], RepositoryState] = {}
    bare_dirs: set[Path] = set()

    def discover(root: Path) -> bool:
        """Record the repository rooted at ``root``; False fails closed."""
        marker = _path_exists(root / ".git")
        if marker is None:
            return False
        if not marker:
            # No .git marker: a bare repository carries its metadata
            # directly in the directory itself (git's own heuristic, all
            # three markers required, conservative).
            bare = _bare_markers_present(root)
            if bare is None:
                return False
            if bare:
                state = _bare_repository_state(root)
                if state is None:
                    return False
                found[(state.git_dir, state.work_tree)] = state
                bare_dirs.add(root)
            return True
        state = _repository_state_at(root)
        if state is None:
            return False
        found[(state.git_dir, state.work_tree)] = state
        return True

    # Repositories enclosing the operand, up to the assessed work tree.
    current = operand
    while current != work_tree:
        if not discover(current):
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent

    # Repositories contained in the operand.
    try:
        operand_status = os.lstat(operand)
    except FileNotFoundError:
        return list(found.values())
    except OSError:
        return None
    if not stat.S_ISDIR(operand_status.st_mode):
        return list(found.values())
    walk_failed = False

    def on_walk_error(_error: OSError) -> None:
        nonlocal walk_failed
        walk_failed = True

    enumerated = 0
    for dirpath, dirnames, _filenames in os.walk(
        operand, onerror=on_walk_error, followlinks=False
    ):
        if walk_failed:
            return None
        enumerated += 1
        if enumerated > _MAX_STATE_WALK_ENTRIES:
            return None
        # A git directory holds no work trees; never descend into one. A
        # bare repository's directory IS its git dir, so the same rule
        # prunes its whole interior once it is recorded.
        dirnames[:] = [name for name in dirnames if name != ".git"]
        root = Path(dirpath)
        if root in bare_dirs:
            dirnames[:] = []
        if not discover(root):
            return None
    if walk_failed:
        return None
    return list(found.values())
