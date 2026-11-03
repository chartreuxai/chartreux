from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import stat
import tomllib
from typing import TYPE_CHECKING

import tomli_w

from chartreux.core.paths import (
    AGENTS_MD_FILENAME,
    TRUSTED_FOLDERS_FILE,
    find_local_config_dirs,
)
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.core.config.layers._checked_store import CheckedWriteResult


class WorkspaceTrustDecision(StrEnum):
    TRUST_REPO = "trust_repo"
    TRUST_CWD = "trust_cwd"
    TRUST_SESSION = "trust_session"
    DECLINE = "decline"


class WorkspaceTrustStatus(StrEnum):
    TRUSTED = "trusted"
    SESSION = "session"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True)
class WorkspaceTrustPrompt:
    cwd: Path
    cwd_identity: tuple[int, int]
    repo_root: Path | None
    repo_root_identity: tuple[int, int] | None
    detected_files: list[str]
    repo_detected_files: list[str]
    offer_repo_trust: bool
    repo_explicitly_untrusted: bool


def filesystem_identity(path: Path) -> tuple[int, int]:
    """Identity of a real directory without following the final path component."""
    path_stat = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(path_stat.st_mode):
        raise ValueError(f"Workspace path is no longer a directory: {path}")
    return path_stat.st_dev, path_stat.st_ino


def _identity_matches(path: Path, expected: tuple[int, int]) -> bool:
    try:
        return filesystem_identity(path) == expected
    except (OSError, ValueError):
        return False


def has_agents_md_file(path: Path) -> bool:
    agents_md = path / AGENTS_MD_FILENAME
    try:
        return agents_md.is_file()
    except OSError as e:
        logger.warning("Skipping unreadable path=%s: %s", agents_md, e)
        return False


def _is_git_repo_root(path: Path) -> bool:
    git_dir = path / ".git"
    try:
        return git_dir.is_dir() and (git_dir / "HEAD").is_file()
    except OSError as e:
        logger.warning("Skipping unreadable git dir=%s: %s", git_dir, e)
        return False


def find_git_repo_ancestor(path: Path) -> Path | None:
    """Closest ancestor (or *path*) with a real ``.git/HEAD``.

    Excludes the home directory and the filesystem root.
    """
    resolved = path.expanduser().resolve()
    home = Path.home().resolve()
    current = resolved
    while current not in {home, current.parent}:
        if _is_git_repo_root(current):
            return current
        current = current.parent
    return None


def find_trustable_files(path: Path) -> list[str]:
    """Relative paths of files/dirs under *path* that would modify agent behavior."""
    resolved = path.resolve()
    found: list[str] = []

    if has_agents_md_file(path):
        found.append(AGENTS_MD_FILENAME)

    for config_dir in find_local_config_dirs(path).config_dirs:
        label = f"{config_dir.relative_to(resolved)}/"
        if label not in found:
            found.append(label)

    return sorted(found)


def find_repo_trustable_files_for_cwd(cwd: Path, repo_root: Path | None) -> list[str]:
    """Repo-context files that influence *cwd* when inside a git repository.

    Includes:
    - all trustable files at ``repo_root``
    - all ``AGENTS.md`` files on ancestors between ``cwd`` and ``repo_root``
    """
    if repo_root is None:
        return []

    resolved_cwd = cwd.resolve()
    resolved_repo_root = repo_root.resolve()
    if resolved_repo_root not in resolved_cwd.parents:
        return []

    found = set(find_trustable_files(resolved_repo_root))

    current = resolved_cwd.parent
    while current != resolved_repo_root:
        if has_agents_md_file(current):
            relative_path = (current / AGENTS_MD_FILENAME).relative_to(
                resolved_repo_root
            )
            found.add(relative_path.as_posix())
        current = current.parent

    return sorted(found)


def find_untrusted_config_dirs(
    cwd: Path, *, manager: TrustedFoldersManager | None = None
) -> list[Path]:
    """Config dirs under a trusted *cwd* that are explicitly untrusted.

    These are the folders left in a broken state by the fixed bug where trusting
    a folder also marked its nested ``.chartreux``/``.agents`` untrusted. Returns an
    empty list unless *cwd* itself is trusted.
    """
    manager = manager or trusted_folders_manager
    resolved_cwd = cwd.resolve()
    if manager.is_trusted(resolved_cwd) is not True:
        return []

    return sorted(
        config_dir
        for config_dir in find_local_config_dirs(resolved_cwd).config_dirs
        if manager.is_explicitly_untrusted(config_dir)
    )


def maybe_build_workspace_trust_prompt(
    cwd: Path,
    *,
    include_explicitly_untrusted: bool = False,
    manager: TrustedFoldersManager | None = None,
    cwd_identity: tuple[int, int] | None = None,
) -> WorkspaceTrustPrompt | None:
    manager = manager or trusted_folders_manager
    resolved_cwd = cwd.resolve()
    if resolved_cwd == Path.home().resolve():
        return None

    if manager.is_trusted(cwd) is True:
        return None
    if not include_explicitly_untrusted and manager.is_explicitly_untrusted(cwd):
        return None

    repo_root = find_git_repo_ancestor(cwd)
    detected_files = find_trustable_files(cwd)
    repo_detected_files = find_repo_trustable_files_for_cwd(cwd, repo_root)
    if not detected_files and not repo_detected_files:
        return None

    resolved_repo_root = repo_root.resolve() if repo_root else None
    offer_repo_trust = (
        resolved_repo_root is not None
        and resolved_repo_root in resolved_cwd.parents
        and manager.is_trusted(resolved_repo_root) is not True
        and (
            include_explicitly_untrusted
            or not manager.is_explicitly_untrusted(resolved_repo_root)
        )
    )
    repo_explicitly_untrusted = (
        resolved_repo_root is not None
        and manager.is_explicitly_untrusted(resolved_repo_root)
    )

    return WorkspaceTrustPrompt(
        cwd=resolved_cwd,
        cwd_identity=(
            cwd_identity
            if cwd_identity is not None
            else filesystem_identity(resolved_cwd)
        ),
        repo_root=resolved_repo_root,
        repo_root_identity=(
            filesystem_identity(resolved_repo_root)
            if resolved_repo_root is not None
            else None
        ),
        detected_files=detected_files,
        repo_detected_files=repo_detected_files,
        offer_repo_trust=offer_repo_trust,
        repo_explicitly_untrusted=repo_explicitly_untrusted,
    )


def available_workspace_trust_decisions(
    prompt: WorkspaceTrustPrompt, *, include_session: bool = False
) -> list[WorkspaceTrustDecision]:
    decisions = [WorkspaceTrustDecision.TRUST_CWD, WorkspaceTrustDecision.DECLINE]
    if include_session:
        decisions.insert(1, WorkspaceTrustDecision.TRUST_SESSION)
    if prompt.offer_repo_trust:
        decisions.insert(0, WorkspaceTrustDecision.TRUST_REPO)
    return decisions


def apply_workspace_trust_decision(
    prompt: WorkspaceTrustPrompt,
    decision: WorkspaceTrustDecision,
    *,
    manager: TrustedFoldersManager | None = None,
) -> None:
    manager = manager or trusted_folders_manager
    match decision:
        case WorkspaceTrustDecision.TRUST_REPO if (
            prompt.offer_repo_trust
            and prompt.repo_root is not None
            and prompt.repo_root_identity is not None
        ):
            if not _identity_matches(prompt.cwd, prompt.cwd_identity):
                raise ValueError("Workspace changed before trust could be persisted")
            if not _identity_matches(prompt.repo_root, prompt.repo_root_identity):
                raise ValueError("Workspace changed before trust could be persisted")
            manager.add_trusted(
                prompt.repo_root, expected_identity=prompt.repo_root_identity
            )
        case WorkspaceTrustDecision.TRUST_CWD:
            if not _identity_matches(prompt.cwd, prompt.cwd_identity):
                raise ValueError("Workspace changed before trust could be persisted")
            manager.add_trusted(prompt.cwd, expected_identity=prompt.cwd_identity)
        case WorkspaceTrustDecision.TRUST_SESSION:
            if not _identity_matches(prompt.cwd, prompt.cwd_identity):
                raise ValueError("Workspace changed before trust could be persisted")
            manager.trust_for_session(prompt.cwd, expected_identity=prompt.cwd_identity)
        case WorkspaceTrustDecision.DECLINE:
            if not _identity_matches(prompt.cwd, prompt.cwd_identity):
                raise ValueError("Workspace changed before trust could be persisted")
            manager.add_untrusted(prompt.cwd, expected_identity=prompt.cwd_identity)
        case _:
            raise ValueError(f"Unsupported trust decision: {decision}")


class TrustStoreState(StrEnum):
    """Explicit read-path classification of the backing trust store."""

    MISSING = "missing"
    UNREADABLE = "unreadable"
    MALFORMED = "malformed"
    INVALID = "invalid"
    READY = "ready"


def parse_trust_store(data: bytes) -> dict[str, list[str]]:
    """Parse and validate trust-store *data*.

    ``UnicodeDecodeError``/``tomllib.TOMLDecodeError`` mark malformed bytes; the
    shape check raises ``ValueError`` for a well-formed but invalid store.
    """
    document = tomllib.loads(data.decode("utf-8"))
    trusted = document.get("trusted", [])
    untrusted = document.get("untrusted", [])
    if not all(
        isinstance(entries, list) and all(isinstance(entry, str) for entry in entries)
        for entries in (trusted, untrusted)
    ):
        raise ValueError("invalid trust decisions")
    return {"trusted": list(trusted), "untrusted": list(untrusted)}


def classify_trust_store(data: bytes | None) -> tuple[TrustStoreState, str | None]:
    """Classify raw store bytes explicitly as missing/malformed/invalid/ready."""
    if data is None:
        return TrustStoreState.MISSING, None
    try:
        parse_trust_store(data)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        return TrustStoreState.MALFORMED, str(exc)
    except ValueError as exc:
        return TrustStoreState.INVALID, str(exc)
    return TrustStoreState.READY, None


class TrustedFoldersManager:
    def __init__(self) -> None:
        self._file_path = TRUSTED_FOLDERS_FILE.path
        self._trusted: list[str] = []
        self._untrusted: list[str] = []
        self._session_trusted: list[str] = []
        self._session_trust_grants: dict[str, list[str]] = {}
        self.load_error: str | None = None
        self.store_state: TrustStoreState = TrustStoreState.MISSING
        self._load()

    def trust_for_session(
        self, path: Path, *, expected_identity: tuple[int, int] | None = None
    ) -> None:
        normalized = self._normalize_path(path)
        self._validate_normalized_identity(normalized, expected_identity)
        self._session_trusted.append(normalized)
        self._session_trust_grants.setdefault(self._session_grant_key(path), []).append(
            normalized
        )

    def revoke_session_trust(self, path: Path) -> None:
        """Undo one ``trust_for_session`` grant.

        Grants are counted rather than deduplicated, so revoking removes the
        one the caller made and leaves any other in place. Trust that came from
        the stored lists is untouched. A saved caller key is used before
        normalizing again so a redirected path cannot revoke another target.
        An unmatched caller key is a no-op.
        """
        caller_key = self._session_grant_key(path)
        grants = self._session_trust_grants.get(caller_key)
        if grants:
            granted_path = grants.pop(0)
            if not grants:
                del self._session_trust_grants[caller_key]
            if granted_path in self._session_trusted:
                self._session_trusted.remove(granted_path)
            return

    def revoke_session_trusts(self, paths: tuple[Path, ...]) -> None:
        """Revoke the session grants represented by *paths*.

        Callers retain their own grant list, so concurrent sessions can revoke
        exactly their counted grants without disturbing another session's trust.
        """
        for path in paths:
            self.revoke_session_trust(path)

    def _session_grant_key(self, path: Path) -> str:
        """Absolute caller path without resolving a redirectable final component."""
        return str(path.expanduser().absolute())

    def _normalize_path(self, path: Path) -> str:
        return str(path.expanduser().resolve())

    def _validate_normalized_identity(
        self, normalized: str, expected_identity: tuple[int, int] | None
    ) -> None:
        if expected_identity is not None and not _identity_matches(
            Path(normalized), expected_identity
        ):
            raise ValueError("Workspace changed before trust could be persisted")

    def _load(self) -> None:
        # Imports and diagnostic config loading must never create or repair files.
        # Only explicit trust decisions persist through the checked write.
        self.load_error = None
        self._trusted = []
        self._untrusted = []
        try:
            data = self._file_path.read_bytes()
        except FileNotFoundError:
            self.store_state = TrustStoreState.MISSING
            self.load_error = "trust store missing"
            return
        except OSError as exc:
            self.store_state = TrustStoreState.UNREADABLE
            self.load_error = f"trust store unreadable: {exc}"
            return
        state, _detail = classify_trust_store(data)
        self.store_state = state
        if state is not TrustStoreState.READY:
            # Read-only load: a malformed or invalid store is never rewritten here.
            self.load_error = "trust store invalid or unreadable"
            return
        content = parse_trust_store(data)
        self._trusted = content["trusted"]
        self._untrusted = content["untrusted"]

    def _refresh_store_state(self) -> None:
        """Re-classify the on-disk store after a write attempt (read-only)."""
        try:
            data = self._file_path.read_bytes()
        except FileNotFoundError:
            self.store_state = TrustStoreState.MISSING
            return
        except OSError:
            self.store_state = TrustStoreState.UNREADABLE
            return
        state, _detail = classify_trust_store(data)
        self.store_state = state

    def _persist_decision(
        self, normalized: str, *, grant_trusted: bool
    ) -> CheckedWriteResult:
        """Checked write of one trust decision onto the backing store.

        The candidate state is computed inside the primitive's ``edit`` from
        the store's current content and is published to the live in-memory
        lists ONLY when the replace is confirmed (``SAVED``). A refused write
        (malformed/invalid store, stale-revision conflict, lock timeout, or a
        pre-replace failure) never publishes the candidate, so a failed grant
        is never live in the process. A ``DURABILITY_UNCERTAIN`` result means
        the replace happened but its durability is unconfirmed — the reported
        outcome is that caveat (never "definitely not saved"), and the
        in-memory grant is still not applied.
        """
        # Imported lazily: chartreux.core.config.layers.__init__ imports the
        # project layer, which imports this module (circular at import time).
        from chartreux.core.config.layers._checked_store import (
            CheckedWriteOutcome,
            CheckedWriteResult,
            checked_write_store,
            store_revision,
        )

        candidate_box: dict[str, dict[str, list[str]]] = {}

        def edit(current: dict[str, list[str]]) -> bytes:
            candidate = {
                "trusted": list(current.get("trusted", [])),
                "untrusted": list(current.get("untrusted", [])),
            }
            target = candidate["trusted"] if grant_trusted else candidate["untrusted"]
            other = candidate["untrusted"] if grant_trusted else candidate["trusted"]
            if normalized not in target:
                target.append(normalized)
            if normalized in other:
                other.remove(normalized)
            candidate_box["candidate"] = candidate
            return tomli_w.dumps(candidate).encode("utf-8")

        try:
            revision = store_revision(self._file_path)
            result = checked_write_store(
                self._file_path,
                edit,
                parse=parse_trust_store,
                missing_value={"trusted": [], "untrusted": []},
                expected_revision=revision,
            )
        except OSError as exc:
            # An unreadable store fails closed without raising into callers.
            return CheckedWriteResult(
                CheckedWriteOutcome.NOT_SAVED,
                error=f"trust store unreadable, decision not saved: {exc}",
            )
        if result.outcome is CheckedWriteOutcome.SAVED:
            candidate = candidate_box["candidate"]
            self._trusted = candidate["trusted"]
            self._untrusted = candidate["untrusted"]
            self.store_state = TrustStoreState.READY
            self.load_error = None
        else:
            self._refresh_store_state()
        return result

    def _closest_decision(self, path: Path) -> tuple[bool, Path] | None:
        """``(trusted, ancestor)`` for the closest decision, ``None`` if undecided."""
        current = Path(self._normalize_path(path))
        while True:
            s = str(current)
            if s in self._trusted or s in self._session_trusted:
                return True, current
            if s in self._untrusted:
                return False, current
            if current.parent == current:
                return None
            current = current.parent

    def is_trusted(self, path: Path) -> bool | None:
        """Tri-state closest decision; ``None`` when no ancestor has one."""
        match self._closest_decision(path):
            case (trusted, _):
                return trusted
            case None:
                return None

    def trust_status(self, path: Path) -> WorkspaceTrustStatus:
        current = Path(self._normalize_path(path))
        while True:
            s = str(current)
            if s in self._session_trusted:
                return WorkspaceTrustStatus.SESSION
            if s in self._trusted:
                return WorkspaceTrustStatus.TRUSTED
            if s in self._untrusted:
                return WorkspaceTrustStatus.UNTRUSTED
            if current.parent == current:
                return WorkspaceTrustStatus.UNTRUSTED
            current = current.parent

    def is_explicitly_untrusted(self, path: Path) -> bool:
        """*path* literally in the untrusted list (no ancestor walk)."""
        return self._normalize_path(path) in self._untrusted

    def find_trust_root(self, path: Path) -> Path | None:
        """Closest explicitly trusted ancestor; ``None`` if a closer untrust blocks."""
        match self._closest_decision(path):
            case (True, root):
                return root
            case _:
                return None

    def add_trusted(
        self, path: Path, *, expected_identity: tuple[int, int] | None = None
    ) -> CheckedWriteResult:
        normalized = self._normalize_path(path)
        self._validate_normalized_identity(normalized, expected_identity)
        return self._persist_decision(normalized, grant_trusted=True)

    def add_untrusted(
        self, path: Path, *, expected_identity: tuple[int, int] | None = None
    ) -> CheckedWriteResult:
        normalized = self._normalize_path(path)
        self._validate_normalized_identity(normalized, expected_identity)
        return self._persist_decision(normalized, grant_trusted=False)


trusted_folders_manager = TrustedFoldersManager()
