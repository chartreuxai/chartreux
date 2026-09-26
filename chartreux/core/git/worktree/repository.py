from __future__ import annotations

from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import timedelta
from enum import StrEnum, auto
from functools import cached_property
import os
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

from chartreux.core.git.errors import GitError
from chartreux.core.git.repo import (
    GitRepo,
    GitStatus,
    RepoPaths,
    _git_python,
    sanitized_git,
)
from chartreux.core.git.worktree.naming import (
    worktree_name_from_text,
    worktree_name_with_suffix,
)
from chartreux.core.git.worktree.record import (
    AcquireResult,
    HolderKind,
    OwnershipToken,
    WorktreeClaim,
    WorktreeRecord,
    WorktreeRecordError,
    acquire_holder,
    inspect_holders,
    managed_bucket_name,
    release_holder,
    worktree_prune_lock,
)
from chartreux.core.paths import WORKTREES_DIR
from chartreux.core.utils.slug import create_slug
from chartreux.core.utils.time import utc_now
from chartreux.observability.logging import logger

_INVALID_WORKTREE_NAME_CHARS = frozenset('<>:"/\\|?*')
_WORKTREE_REV_PARSE_PARTS = 2
_AUTO_WORKTREE_BRANCH_PREFIX = "chartreux/"
_MAX_AUTO_WORKTREE_ATTEMPTS = 100
# Long enough to cover a checkout of a large repo, so a session that is still
# starting is never mistaken for one that died.
_CLAIM_SWEEP_GRACE = timedelta(minutes=10)
# Under `refs/vibe/` rather than `refs/heads/`, so a snapshot never appears in
# the branch picker, in `git branch`, or as something to push. It is a way back
# to work the reaper removed, not a branch anybody is meant to develop on.
SNAPSHOT_REF_PREFIX = "refs/vibe/reaped"
_RESERVED_WORKTREE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class WorktreeError(GitError): ...


@dataclass(frozen=True)
class PreparedWorktree:
    name: str
    branch: str
    root: Path
    path: Path
    repo_root: Path
    base_commit: str
    created: bool
    branch_created: bool
    pending_token: OwnershipToken | None = None

    def inspect_for_cleanup(self) -> WorktreeCleanupState:
        """Inspect worktree state relative to the session-start HEAD.

        Commit counts intentionally use the worktree's current HEAD instead of
        the named branch so detached-HEAD commits still block cleanup.
        """
        git = _git_python()
        try:
            repo = git.repo(self.root)
            status_lines = (
                _unhooked(repo)
                .status("--porcelain", "--untracked-files=all")
                .splitlines()
            )
            new_commit_count = int(
                repo.git.rev_list("--count", f"{self.base_commit}..HEAD").strip()
            )
        except (
            git.invalid_git_repository_error,
            git.git_command_error,
            ValueError,
        ) as e:
            raise WorktreeError(f"Failed to inspect worktree {self.name!r}: {e}") from e

        return WorktreeCleanupState(
            has_uncommitted_changes=any(
                not line.startswith("??") for line in status_lines
            ),
            has_untracked_files=any(line.startswith("??") for line in status_lines),
            new_commit_count=new_commit_count,
        )

    def snapshot(self) -> str:
        """Commit this worktree's whole state to a ref nothing checks out.

        Returns the ref. Raises when the state could not be saved, which is
        the caller's signal to keep the worktree instead of removing it.

        The reaper deletes a worktree whose session is gone even when work was
        left behind, and that is only defensible because the work survives
        somewhere. Untracked files are in: an agent that wrote a file and
        never staged it produced exactly the state a user would most regret
        losing, and here it is the common case rather than the corner. Ignored
        files are out, for the same reason `git add` leaves them -- a snapshot
        that swept up `node_modules` would cost more than the work it saved.

        Written through a second index, so the worktree's own is untouched: a
        failure here aborts the removal, and the worktree the user is then left
        with must be the one they had. That index lives in the worktree's own
        git directory, which the removal takes with it, rather than in a
        temporary directory that would outlive a crash.
        """
        git = _git_python()
        ref = f"{SNAPSHOT_REF_PREFIX}/{self.name}"
        try:
            repo = git.repo(self.root)
            existing_ref = (
                _unhooked(repo).for_each_ref(ref, "--format=%(refname)").strip()
            )
            if existing_ref:
                raise WorktreeError(
                    f"Cannot snapshot worktree {self.name!r}: its recovery name "
                    "conflicts with a previous recovery snapshot"
                )
            index = Path(repo.git.rev_parse("--absolute-git-dir").strip())
            with repo.git.custom_environment(
                GIT_INDEX_FILE=str(index / "index.vibe-snapshot")
            ):
                _unhooked(repo).read_tree("HEAD")
                _unhooked(repo).add("--all", ".")
                tree = _unhooked(repo).write_tree().strip()
                head = _unhooked(repo).rev_parse("HEAD").strip()
                commit = (
                    _unhooked(repo)
                    .commit_tree(
                        tree,
                        "-p",
                        head,
                        "-m",
                        f"vibe: state of worktree {self.name} before it was reaped",
                    )
                    .strip()
                )
            _unhooked(repo).update_ref(ref, commit)
        except (git.invalid_git_repository_error, git.git_command_error) as e:
            raise WorktreeError(
                f"Failed to snapshot worktree {self.name!r}: {e}"
            ) from e
        return ref

    def remove(self, *, delete_branch: bool = True) -> None:
        git = _git_python()
        try:
            repo = git.repo(self.repo_root)
            _unhooked(repo).worktree("remove", "--force", str(self.root))
            if delete_branch:
                _unhooked(repo).branch("-D", self.branch)
        except (git.invalid_git_repository_error, git.git_command_error) as e:
            raise WorktreeError(f"Failed to remove worktree {self.name!r}: {e}") from e

    # Callers that own the process working directory must invoke this before
    # remove(); Windows refuses to delete a directory that is any process's cwd.
    # remove() itself must not chdir, because the app-server serves concurrent
    # sessions and resolves relative paths against the process cwd.
    def leave_if_current_directory(self) -> None:
        try:
            cwd = Path.cwd().resolve()
        except FileNotFoundError:
            os.chdir(self.repo_root)
            return
        if cwd.is_relative_to(self.root.resolve()):
            os.chdir(self.repo_root)


@dataclass(frozen=True)
class LinkedWorktree:
    name: str
    branch: str
    root: Path
    path: Path
    repo_root: Path


class RemovalPolicy(StrEnum):
    """Controls whether a created checkout or only its reservation may be removed."""

    NORMAL = auto()
    FAILED_START = auto()


@dataclass(frozen=True)
class TransferAttempt:
    source_token: OwnershipToken
    target_token: OwnershipToken


@dataclass(frozen=True)
class SweepResult:
    removed: int = 0
    repaired: int = 0
    reservations_reclaimed: int = 0


class WorktreeReleaseOutcome(StrEnum):
    REMOVED = auto()
    KEPT_DIRTY = auto()
    KEPT_IN_USE = auto()
    KEPT_UNMANAGED = auto()
    NOT_FOUND = auto()


@dataclass(frozen=True)
class WorktreeRelease:
    outcome: WorktreeReleaseOutcome
    root: Path | None = None
    branch: str | None = None
    branch_deleted: bool = False
    reasons: tuple[str, ...] = field(default_factory=tuple)
    # Where the work went when a removed worktree still had some. None when it
    # had none, so a caller can tell "nothing to recover" from "recover here".
    snapshot_ref: str | None = None


@dataclass(frozen=True)
class WorktreeCleanupState:
    has_uncommitted_changes: bool
    has_untracked_files: bool
    new_commit_count: int

    @property
    def is_clean(self) -> bool:
        return (
            not self.has_uncommitted_changes
            and not self.has_untracked_files
            and self.new_commit_count == 0
        )

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.has_uncommitted_changes:
            reasons.append("uncommitted changes")
        if self.has_untracked_files:
            reasons.append("untracked files")
        if self.new_commit_count:
            noun = "commit" if self.new_commit_count == 1 else "commits"
            reasons.append(f"{self.new_commit_count} {noun} added during this session")
        return tuple(reasons)


class WorktreeRepository:
    """The managed worktrees of one git repository.

    Opened against a path inside the repository rather than the repository
    root, because that path decides two things at once: which repository is
    meant, and which subdirectory of a prepared worktree the caller lands in.
    """

    def __init__(self, git: GitRepo, base: Path) -> None:
        self._git = git
        self._base = base

    # GitPython keeps `git cat-file --batch` children alive holding handles into
    # .git until the Repo is closed, which is why closing is the context
    # manager's job rather than each caller's: a background sweep that forgot
    # would hold those handles past app-server teardown.
    @classmethod
    @contextmanager
    def open(cls, base: Path) -> Iterator[WorktreeRepository]:
        git = GitRepo.open(base)
        try:
            yield cls(git, base)
        finally:
            git.close()

    # Identifies the repository a path belongs to, for a caller telling two
    # repositories apart that has no reason to care whether the path is in one.
    @classmethod
    def bucket_for(cls, base: Path) -> str | None:
        try:
            with cls.open(base) as repository:
                return repository.bucket
        except GitError:
            return None

    @classmethod
    def sweep_claims(
        cls,
        base: Path,
        *,
        resumable_snapshot_provider: Callable[[], Collection[Path]] | None = None,
        in_use: Collection[Path] | None = None,
    ) -> SweepResult:
        provider = resumable_snapshot_provider
        if provider is None:
            provider = lambda: () if in_use is None else in_use
        try:
            with cls.open(base) as repository:
                return repository.sweep(resumable_snapshot_provider=provider)
        except GitError:
            return SweepResult()

    @property
    def root(self) -> Path:
        return self._paths.repo_root

    @property
    def repository_counterpart(self) -> Path | None:
        """Where the base sits when mapped onto the main checkout.

        ``linked()`` reports this same mapping for each worktree, preserving the
        subdirectory the caller opened at. The main checkout has no entry there,
        so a caller comparing against both needs this to complete the set.

        None when there is no usable counterpart. It goes through the checks
        ``linked()`` applies rather than resolving the path directly, so a
        symlink that leaves the checkout is refused instead of being reported
        as a position inside it.
        """
        try:
            return _target_cwd(self._paths.repo_root, self._relative_base)
        except WorktreeError:
            return None

    @property
    def bucket(self) -> str:
        paths = self._paths
        return managed_bucket_name(paths.repo_root, paths.common_git_dir)

    # Derived on access, not eagerly: _worktree_root enforces containment under
    # WORKTREES_DIR and raises, so callers that only read git metadata must not
    # pay for a check they never needed.
    @property
    def worktree_root(self) -> Path:
        paths = self._paths
        return _worktree_root(paths.repo_root, paths.common_git_dir)

    @property
    def _paths(self) -> RepoPaths:
        return self._git.paths

    @cached_property
    def _relative_base(self) -> Path:
        return self._git.relative_base(self._base)

    def prepare(self, name: str, *, branch: str | None = None) -> PreparedWorktree:
        _validate_worktree_name(name)
        branch = name if branch is None else branch
        self._git.validate_branch(branch)

        paths = self._paths
        target = self.worktree_root / name

        claim = WorktreeClaim(bucket=self.bucket, name=name)
        try:
            target.mkdir(parents=True)
        except FileExistsError:
            acquired = self.acquire_for_attachment(target)
            if acquired.outcome not in {AcquireResult.SUCCESS, AcquireResult.UNMANAGED}:
                raise WorktreeError(f"Worktree {name!r} is currently unavailable.")
            token = acquired.token
            try:
                _validate_existing_worktree(target, branch, paths.common_git_dir)
                return self._build_prepared(
                    name,
                    branch,
                    target,
                    created=False,
                    branch_created=False,
                    pending_token=token,
                )
            except BaseException:
                if token is not None:
                    release_holder(token)
                raise
        except OSError as e:
            raise WorktreeError(
                f"Failed to claim worktree directory {target}: {e}"
            ) from e

        branch_created = not self._git.branch_exists(branch)
        record = WorktreeRecord.new(
            name=name,
            branch=branch,
            repo_root=paths.repo_root,
            branch_created=branch_created,
        )
        starting = self._publish_starting(claim, record, target)
        return self._create(
            claim, record, target, branch_created=branch_created, starting=starting
        )

    def prepare_auto(
        self, *, prompt: str | None = None, suggested_name: str | None = None
    ) -> PreparedWorktree:
        paths = self._paths
        name, branch, target = self._claim_auto_name(
            _auto_worktree_name(prompt, suggested_name)
        )
        claim = WorktreeClaim(bucket=self.bucket, name=name)
        record = WorktreeRecord.new(
            name=name, branch=branch, repo_root=paths.repo_root, branch_created=True
        )
        starting = self._publish_starting(claim, record, target)
        return self._create(
            claim, record, target, branch_created=True, starting=starting
        )

    @staticmethod
    def _publish_starting(
        claim: WorktreeClaim, record: WorktreeRecord, target: Path
    ) -> OwnershipToken:
        generation = record.claimed_at.isoformat()
        with worktree_prune_lock():
            try:
                claim.write(record)
                acquired = acquire_holder(
                    claim,
                    "starting",
                    kind=HolderKind.STARTING,
                    expected_generation=generation,
                )
                if (
                    acquired.outcome is not AcquireResult.SUCCESS
                    or acquired.token is None
                ):
                    raise WorktreeError("Unable to protect worktree creation.")
                return acquired.token
            except BaseException:
                claim.delete()
                with suppress(OSError):
                    target.rmdir()
                raise

    @staticmethod
    def acquire_for_attachment(
        cwd: Path, *, expected_generation: str | None = None
    ) -> AcquireResult:
        claim = WorktreeClaim.locate(cwd)
        if claim is None:
            return AcquireResult(AcquireResult.UNMANAGED)
        with worktree_prune_lock():
            record = claim.read()
            if record is None:
                return AcquireResult(AcquireResult.UNMANAGED)
            generation = record.claimed_at.isoformat()
            if expected_generation is not None and expected_generation != generation:
                return AcquireResult(AcquireResult.STALE)
            return acquire_holder(
                claim,
                f"attach-{uuid4().hex}",
                kind=HolderKind.PENDING_ATTACHMENT,
                expected_generation=generation,
            )

    def status(self) -> GitStatus:
        """This repository's own checkout, off the open this object holds.

        The listing and the status answer about the same repository, and
        opening it is what costs: each open leaves ``git cat-file --batch``
        children holding handles into .git until it is closed. Read together
        they cost one open instead of two.
        """
        return self._git.status()

    def checkouts(self) -> tuple[tuple[Path, str | None], ...]:
        """Every checkout git reports for this repository, and its branch.

        Wider than ``linked()``, which keeps only the managed worktrees a
        session may be moved into. A session can be *sitting* in one that is
        detached, prunable, or fails validation, and such a checkout is absent
        from that listing: asked from it alone, the repository looks as though
        it does not hold the session at all, and the branch reported for it is
        the main checkout's rather than the one the session is on.

        Branch is None for a detached checkout, which is a real answer here
        rather than a missing one.

        Read off the same open as ``linked()``, so a caller needing both pays
        for one repository rather than two.
        """
        return tuple(
            (record.root.resolve(), record.branch) for record in self._git.records()
        )

    def linked(self) -> tuple[LinkedWorktree, ...]:
        paths = self._paths
        # Resolved up front, not inside the loop: a base that escapes the
        # checkout is the caller's error, and reading it lazily would let the
        # per-record except swallow it as "skip this worktree" - or, in a
        # repository with no linked worktrees, never check it at all.
        relative_base = self._relative_base
        records = self._git.records()
        linked: list[LinkedWorktree] = []

        for record in records[1:]:
            if record.branch is None or record.prunable:
                continue
            try:
                _validate_listed_worktree(record.root, paths.common_git_dir)
                root = record.root.resolve()
                path = _target_cwd(root, relative_base)
            except WorktreeError:
                continue
            linked.append(
                LinkedWorktree(
                    name=root.name,
                    branch=record.branch,
                    root=root,
                    path=path,
                    repo_root=paths.repo_root,
                )
            )

        return tuple(sorted(linked, key=lambda worktree: str(worktree.path)))

    def sweep(
        self, *, resumable_snapshot_provider: Callable[[], Collection[Path]]
    ) -> SweepResult:
        """Recover abandoned reservations and remove non-resumable claims."""
        with worktree_prune_lock():
            try:
                worktree_root = self.worktree_root
                kept = {path.resolve() for path in resumable_snapshot_provider()}
            except (GitError, OSError) as exc:
                logger.warning("Skipping worktree sweep: snapshot failed: %s", exc)
                return SweepResult()
            cutoff = utc_now() - _CLAIM_SWEEP_GRACE
            removed = repaired = reclaimed = 0
            for claim in WorktreeClaim.in_bucket(self.bucket):
                record = claim.read()
                if record is None:
                    continue
                holders = inspect_holders(claim)
                if holders.outcome is AcquireResult.ERROR:
                    continue
                if record.claimed_at > cutoff or holders.starting or holders.holders:
                    continue
                target = worktree_root / claim.name
                if record.base_commit is not None:
                    root = target.resolve()
                    if not any(cwd.is_relative_to(root) for cwd in kept):
                        before = target.exists()
                        self._reap(claim, target)
                        removed += int(before and not target.exists())
                    continue
                if (target / ".git").is_file():
                    try:
                        _validate_existing_worktree(
                            target, record.branch, self._paths.common_git_dir
                        )
                    except WorktreeError:
                        pass
                    # Never invent a baseline from current HEAD: it may conceal
                    # valuable commits created after the interrupted publication.
                    continue
                if target.is_symlink() or not _is_empty_dir(target):
                    continue
                self._discard_claim(
                    target, record.branch, branch_created=record.branch_created
                )
                reclaimed += 1
                logger.info("Discarded an abandoned worktree reservation at %s", target)
            return SweepResult(removed, repaired, reclaimed)

    def _reap(self, claim: WorktreeClaim, target: Path) -> None:
        try:
            release = ManagedWorktree(claim=claim).release()
        except (GitError, OSError) as exc:
            logger.debug("Failed to reap worktree %s: %s", target, exc)
            return
        if release.outcome is WorktreeReleaseOutcome.REMOVED:
            logger.info(
                "Reaped the worktree of a session that no longer exists: %s", target
            )

    def _base_ref(self) -> str | None:
        """The ref a newly created worktree branch starts from.

        The remote's default branch rather than the invoking checkout's HEAD,
        which is whatever the user happened to have checked out and, for a
        `main` that is rarely pulled, weeks behind.

        None leaves the start point off the `git worktree add` entirely, so a
        repository with no remote keeps git's own default.
        """
        ref = self._git.remote_default_branch_ref()
        if ref is None:
            return None
        remote, _, branch = ref.partition("/")
        try:
            self._git.fetch_branch(remote, branch)
        except GitError as e:
            # Offline, or the remote refused. The remote-tracking ref is still
            # a better base than HEAD even when stale, because it advances on
            # any fetch while a local branch only moves when pulled.
            logger.warning("Could not refresh %s before branching: %s", ref, e)
        return ref

    def _create(
        self,
        claim: WorktreeClaim,
        record: WorktreeRecord,
        target: Path,
        *,
        branch_created: bool,
        starting: OwnershipToken,
    ) -> PreparedWorktree:
        branch = record.branch
        start_point = self._base_ref() if branch_created else None
        try:
            self._git.add_worktree(
                target, branch, branch_created=branch_created, start_point=start_point
            )
        except Exception as e:
            try:
                ManagedWorktree(claim=claim).remove_if_unheld(
                    expected_generation=starting.claim_generation,
                    policy=RemovalPolicy.FAILED_START,
                    retiring_token=starting,
                )
            except (WorktreeError, OSError) as cleanup_error:
                e.add_note(
                    f"Failed to clean up worktree {target.name!r} after creation "
                    f"failure: {cleanup_error}"
                )
            raise
        try:
            prepared = self._build_prepared(
                record.name, branch, target, created=True, branch_created=branch_created
            )
        except Exception as e:
            try:
                ManagedWorktree(claim=claim).remove_if_unheld(
                    expected_generation=starting.claim_generation,
                    policy=RemovalPolicy.FAILED_START,
                    retiring_token=starting,
                )
            except (WorktreeError, OSError) as cleanup_error:
                e.add_note(
                    f"Failed to clean up worktree {target.name!r} after prepare "
                    f"failure: {cleanup_error}"
                )
            raise
        completed = record.model_copy(update={"base_commit": prepared.base_commit})
        acquired: AcquireResult | None = None
        try:
            with worktree_prune_lock():
                claim.write(completed)
                acquired = acquire_holder(
                    claim,
                    f"attach-{uuid4().hex}",
                    kind=HolderKind.PENDING_ATTACHMENT,
                    expected_generation=completed.claimed_at.isoformat(),
                )
                if (
                    acquired.outcome is not AcquireResult.SUCCESS
                    or acquired.token is None
                ):
                    raise WorktreeError(
                        "Unable to protect prepared worktree attachment."
                    )
                release_holder(starting)
        except BaseException as exc:
            if acquired is not None and acquired.token is not None:
                try:
                    release_holder(acquired.token)
                except (OSError, WorktreeRecordError) as release_error:
                    exc.add_note(
                        f"Failed to release pending attachment for worktree "
                        f"{target.name!r}: {release_error}"
                    )
            try:
                ManagedWorktree(claim=claim).remove_if_unheld(
                    expected_generation=starting.claim_generation,
                    policy=RemovalPolicy.FAILED_START,
                    retiring_token=None if starting.consumed else starting,
                )
            except (WorktreeError, WorktreeRecordError, OSError) as cleanup_error:
                exc.add_note(
                    f"Failed to clean up worktree {target.name!r} after publication "
                    f"failure: {cleanup_error}"
                )
            raise
        return replace(prepared, pending_token=acquired.token)

    def _build_prepared(
        self,
        name: str,
        branch: str,
        target: Path,
        *,
        created: bool,
        branch_created: bool,
        pending_token: OwnershipToken | None = None,
    ) -> PreparedWorktree:
        # base_commit is the worktree's own HEAD at session start, so cleanup
        # counts only commits added during this session (not commits an attached
        # or reused branch already carried, which the invoking checkout's HEAD
        # would miss).
        return PreparedWorktree(
            name=name,
            branch=branch,
            root=target,
            path=_target_cwd(target, self._relative_base),
            repo_root=self._paths.repo_root,
            base_commit=GitRepo.head_commit_at(target),
            created=created,
            branch_created=branch_created,
            pending_token=pending_token,
        )

    def _claim_auto_name(self, base_name: str) -> tuple[str, str, Path]:
        # mkdir is the atomic claim: git worktree add cannot report *why* it
        # failed (a taken path and an invalid ref both exit 128), so a lost race
        # has to be detected before git runs or it is indistinguishable from a
        # real failure.
        worktree_root = self.worktree_root
        for name in _auto_worktree_candidates(base_name):
            branch = f"{_AUTO_WORKTREE_BRANCH_PREFIX}{name}"
            if self._git.branch_exists(branch):
                continue
            target = worktree_root / name
            try:
                target.mkdir(parents=True)
            except FileExistsError:
                continue
            except OSError as e:
                raise WorktreeError(
                    f"Failed to claim worktree directory {target}: {e}"
                ) from e
            return name, branch, target

        raise WorktreeError(
            f"Unable to find an unused worktree name for {base_name!r} after "
            f"{_MAX_AUTO_WORKTREE_ATTEMPTS} attempts."
        )

    def _discard_claim(
        self, target: Path, branch: str, *, branch_created: bool
    ) -> None:
        _delete_record_for(target)
        # rmdir refuses a populated directory, so a partial checkout is never
        # lost, and git's own remove_junk may have removed it already.
        with suppress(OSError):
            target.rmdir()
        # Only a branch this reservation brought into being. A named worktree
        # can ask for a branch that already existed and holds the user's work,
        # and that one has to survive the reservation being discarded.
        if not branch_created:
            return
        # `worktree add -b` creates the branch before it validates the path, so
        # a failed add leaves it behind. _claim_auto_name would then skip this
        # name for every later call, walking the suffix chain until it reports
        # exhaustion instead of the real failure.
        #
        # A safe delete rather than a forced one: the branch this call created
        # still points at HEAD, so it goes, while a racing branch carrying
        # commits is refused. That is a partial guard, not a full one -- a
        # branch someone else created at HEAD between the branch_exists check
        # and the add is merged too, so it is deleted (recoverable from the
        # reflog). Proving ownership would mean leaving the branch and burning
        # the name on every failure, which costs more than the race it closes.
        with suppress(GitError):
            self._git.delete_branch(branch)


@dataclass(frozen=True)
class ManagedWorktree:
    """A worktree Chartreux created, addressed by a path inside it.

    Wraps the claim so holding, releasing and removing all go through one
    value. `at` returning None is the whole "is this one of ours?" question: an
    unmanaged directory has no claim, and each caller decides what that means
    rather than having a silent no-op decide for it.
    """

    claim: WorktreeClaim

    # Deliberately not conditioned on the record existing: the CLI deletes the
    # record with forget() and only then drops its holder, and that second call
    # still has to find the claim to clean up after itself.
    @classmethod
    def at(cls, cwd: Path) -> ManagedWorktree | None:
        claim = WorktreeClaim.locate(cwd)
        return None if claim is None else cls(claim=claim)

    @property
    def name(self) -> str:
        return self.claim.name

    @property
    def root(self) -> Path:
        return WORKTREES_DIR.path.resolve() / self.claim.bucket / self.claim.name

    def hold(
        self, session_id: str, pending_token: OwnershipToken | None = None
    ) -> AcquireResult:
        record = self.claim.read()
        if record is None:
            if pending_token is not None:
                release_holder(pending_token)
            return AcquireResult(AcquireResult.UNMANAGED)
        if pending_token is not None and pending_token.claim_identity != (
            self.claim.bucket,
            self.claim.name,
        ):
            release_holder(pending_token)
            raise WorktreeError("Pending attachment does not match the session cwd.")
        acquired = acquire_holder(
            self.claim,
            session_id,
            kind=HolderKind.SESSION,
            expected_generation=record.claimed_at.isoformat(),
        )
        if pending_token is not None:
            release_holder(pending_token)
        return acquired

    def release_holder(self, token: OwnershipToken | str) -> None:
        if isinstance(token, OwnershipToken):
            release_holder(token)
        else:
            self.claim.remove_holder(token)

    def holders(self) -> frozenset[str]:
        snapshot = inspect_holders(self.claim)
        return frozenset(snapshot.holders)

    @staticmethod
    def begin_transfer(
        source_token: OwnershipToken, target_cwd: Path
    ) -> TransferAttempt:
        acquired = WorktreeRepository.acquire_for_attachment(target_cwd)
        if acquired.outcome is not AcquireResult.SUCCESS or acquired.token is None:
            raise WorktreeError("Unable to reserve transfer target.")
        return TransferAttempt(source_token, acquired.token)

    @staticmethod
    def commit_transfer(attempt: TransferAttempt) -> OwnershipToken:
        release_holder(attempt.source_token)
        return attempt.target_token

    @staticmethod
    def abort_transfer(attempt: TransferAttempt) -> None:
        release_holder(attempt.target_token)

    # For a caller that removed the worktree itself, like the CLI's interactive
    # exit cleanup. Leaving the record behind would keep a claim for a directory
    # that no longer exists.
    def forget(self) -> None:
        with worktree_prune_lock():
            self.claim.delete()

    def remove_if_unheld(  # noqa: PLR0911
        self,
        *,
        expected_generation: str | None = None,
        policy: RemovalPolicy = RemovalPolicy.NORMAL,
        retiring_token: OwnershipToken | None = None,
        delete_branch: bool | None = None,
    ) -> WorktreeRelease:
        with worktree_prune_lock():
            record = self.claim.read()
            if record is None:
                return WorktreeRelease(WorktreeReleaseOutcome.KEPT_UNMANAGED)
            generation = record.claimed_at.isoformat()
            if expected_generation is not None and expected_generation != generation:
                return WorktreeRelease(WorktreeReleaseOutcome.KEPT_UNMANAGED)
            if retiring_token is not None:
                if retiring_token.claim_generation != generation:
                    return WorktreeRelease(WorktreeReleaseOutcome.KEPT_UNMANAGED)
                release_holder(retiring_token)
            holders = inspect_holders(self.claim)
            if holders.outcome is not AcquireResult.SUCCESS:
                return WorktreeRelease(
                    WorktreeReleaseOutcome.KEPT_IN_USE, branch=record.branch
                )
            if holders.starting or holders.holders:
                return WorktreeRelease(
                    WorktreeReleaseOutcome.KEPT_IN_USE, branch=record.branch
                )
            if policy is RemovalPolicy.FAILED_START:
                if record.base_commit is None and not (self.root / ".git").is_file():
                    self.claim.delete()
                    with suppress(OSError):
                        self.root.rmdir()
                    if record.branch_created:
                        with (
                            suppress(GitError),
                            WorktreeRepository.open(record.repo_root) as repository,
                        ):
                            repository._git.delete_branch(record.branch)
                else:
                    prepared = PreparedWorktree(
                        name=self.claim.name,
                        branch=record.branch,
                        root=self.root,
                        path=self.root,
                        repo_root=record.repo_root,
                        base_commit=record.base_commit or "",
                        created=True,
                        branch_created=record.branch_created,
                    )
                    prepared.remove(delete_branch=record.branch_created)
                    self.claim.delete()
                return WorktreeRelease(
                    WorktreeReleaseOutcome.REMOVED,
                    root=self.root,
                    branch=record.branch,
                    branch_deleted=record.branch_created,
                )
            return self._release_unheld(
                record,
                delete_branch=record.branch_created
                if delete_branch is None
                else delete_branch,
            )

    @classmethod
    def abort_prepared(
        cls, prepared: PreparedWorktree, pending_token: OwnershipToken
    ) -> WorktreeRelease:
        managed = cls.at(prepared.root)
        release_holder(pending_token)
        if managed is None or not prepared.created:
            return WorktreeRelease(
                WorktreeReleaseOutcome.KEPT_UNMANAGED,
                root=prepared.root,
                branch=prepared.branch,
            )
        return managed.remove_if_unheld(
            expected_generation=pending_token.claim_generation,
            policy=RemovalPolicy.NORMAL,
        )

    def release(self, session_id: str | None = None) -> WorktreeRelease:
        with worktree_prune_lock():
            return self._release_locked(session_id)

    def _release_locked(self, session_id: str | None = None) -> WorktreeRelease:
        record = self.claim.read()
        if record is None:
            return WorktreeRelease(WorktreeReleaseOutcome.KEPT_UNMANAGED)

        # None means the caller never held it - a delete arriving after the
        # session already closed. Every other holder still has to be gone.
        if session_id is not None:
            self.claim.remove_holder(session_id)
        if remaining := self.claim.holders():
            logger.debug(
                "Keeping worktree %s: still held by %d session(s)",
                self.claim.name,
                len(remaining),
            )
            return WorktreeRelease(
                WorktreeReleaseOutcome.KEPT_IN_USE, branch=record.branch
            )

        return self._release_unheld(record)

    def _release_unheld(
        self, record: WorktreeRecord, *, delete_branch: bool | None = None
    ) -> WorktreeRelease:
        if delete_branch is None:
            delete_branch = record.branch_created
        root = self.root
        if not root.is_dir():
            self.claim.delete()
            return WorktreeRelease(WorktreeReleaseOutcome.NOT_FOUND)

        if record.base_commit is None:
            # The claim never became a worktree, or the second record write was
            # lost. Either way there is no baseline to judge cleanliness
            # against, so leave it for the sweep rather than guess.
            return WorktreeRelease(WorktreeReleaseOutcome.KEPT_UNMANAGED)

        prepared = PreparedWorktree(
            name=self.claim.name,
            branch=record.branch,
            root=root,
            path=root,
            repo_root=record.repo_root,
            base_commit=record.base_commit,
            created=True,
            branch_created=record.branch_created,
        )
        state = prepared.inspect_for_cleanup()
        snapshot: str | None = None
        if not state.is_clean:
            # Work left behind is a reason to save it, not a reason to keep the
            # directory. Keeping was safe and unbounded: a session deleted after
            # writing one uncommitted file left a worktree nothing would ever
            # collect, and dogfooding grew a pile of them. So the removal is
            # made recoverable instead, and only a snapshot that will not
            # write is still worth stopping for.
            try:
                snapshot = prepared.snapshot()
            except (WorktreeError, OSError) as exc:
                logger.warning(
                    "Keeping worktree %s on branch %s: %s could not be saved (%s)",
                    root,
                    record.branch,
                    ", ".join(state.reasons),
                    exc,
                )
                return WorktreeRelease(
                    WorktreeReleaseOutcome.KEPT_DIRTY,
                    root=root,
                    branch=record.branch,
                    reasons=state.reasons,
                )

        # Re-checked immediately before the destructive step. Inspecting a
        # worktree shells out to git, which is long enough for a session in
        # another process to claim it. This narrows the window rather than
        # closing it: without a cross-process lock, a holder written during
        # remove() itself still loses. Losing means a live session in a deleted
        # directory, so the check is worth the extra stat even though it is not
        # a guarantee.
        if late := self.claim.holders():
            logger.info(
                "Keeping worktree %s: %d session(s) joined during inspection",
                root,
                len(late),
            )
            return WorktreeRelease(
                WorktreeReleaseOutcome.KEPT_IN_USE, root=root, branch=record.branch
            )

        prepared.remove(delete_branch=delete_branch)
        self.claim.delete()
        if snapshot is not None:
            # At INFO with the command spelled out, because this is the only
            # trace of the work left and a ref nobody can name is not a way
            # back. `git branch -D` above orphans any commits the session made;
            # the ref is what keeps them reachable.
            logger.info(
                "Removed worktree %s, which still had %s. Recover it with: "
                "git -C %s switch -c %s %s",
                root,
                ", ".join(state.reasons),
                record.repo_root,
                self.claim.name,
                snapshot,
            )
        return WorktreeRelease(
            WorktreeReleaseOutcome.REMOVED,
            root=root,
            branch=record.branch,
            branch_deleted=delete_branch,
            snapshot_ref=snapshot,
        )


def acquire_for_attachment(
    cwd: Path, *, expected_generation: str | None = None
) -> AcquireResult:
    return WorktreeRepository.acquire_for_attachment(
        cwd, expected_generation=expected_generation
    )


def begin_transfer(source_token: OwnershipToken, target_cwd: Path) -> TransferAttempt:
    return ManagedWorktree.begin_transfer(source_token, target_cwd)


def commit_transfer(attempt: TransferAttempt) -> OwnershipToken:
    return ManagedWorktree.commit_transfer(attempt)


def abort_transfer(attempt: TransferAttempt) -> None:
    ManagedWorktree.abort_transfer(attempt)


def remove_if_unheld(
    cwd: Path,
    *,
    expected_generation: str | None = None,
    policy: RemovalPolicy = RemovalPolicy.NORMAL,
    retiring_token: OwnershipToken | None = None,
    delete_branch: bool | None = None,
) -> WorktreeRelease:
    managed = ManagedWorktree.at(cwd)
    if managed is None:
        return WorktreeRelease(WorktreeReleaseOutcome.KEPT_UNMANAGED)
    return managed.remove_if_unheld(
        expected_generation=expected_generation,
        policy=policy,
        retiring_token=retiring_token,
        delete_branch=delete_branch,
    )


def abort_prepared(
    prepared: PreparedWorktree, pending_token: OwnershipToken
) -> WorktreeRelease:
    return ManagedWorktree.abort_prepared(prepared, pending_token)


def sweep_claims(
    base: Path, *, resumable_snapshot_provider: Callable[[], Collection[Path]]
) -> SweepResult:
    return WorktreeRepository.sweep_claims(
        base, resumable_snapshot_provider=resumable_snapshot_provider
    )


def _auto_worktree_name(prompt: str | None, suggested: str | None = None) -> str:
    # The model's answer and the raw prompt go through the same slugifier, so a
    # name that reaches git is portable however it was produced. Either can
    # reduce to "" or to a reserved device name such as "nul", neither of which
    # survives _validate_worktree_name, hence the walk down to a random slug.
    for text in (suggested, prompt):
        if text is None:
            continue
        name = worktree_name_from_text(text)
        if _is_portable_worktree_name(name):
            return name
    return create_slug()


def _auto_worktree_candidates(base_name: str) -> Iterator[str]:
    yield base_name
    for suffix in range(2, _MAX_AUTO_WORKTREE_ATTEMPTS + 1):
        yield worktree_name_with_suffix(base_name, suffix)


def _is_empty_dir(target: Path) -> bool:
    try:
        return not any(target.iterdir())
    except OSError:
        return False


def _unhooked(repo: Any) -> Any:
    """Return the repository's command handle sanitized for one command."""
    return sanitized_git(repo)


def _validate_worktree_name(name: str) -> None:
    if not _is_portable_worktree_name(name):
        raise WorktreeError(
            "--worktree NAME must be a single path segment with a portable filename."
        )


def _is_portable_worktree_name(name: str) -> bool:
    if not name or name in {".", ".."}:
        return False
    if name[-1] in {" ", "."} or not name.isprintable():
        return False
    if any(char in _INVALID_WORKTREE_NAME_CHARS for char in name):
        return False
    if name.split(".", 1)[0].upper() in _RESERVED_WORKTREE_NAMES:
        return False
    return Path(name).parts == (name,) and PureWindowsPath(name).parts == (name,)


def _delete_record_for(target: Path) -> None:
    if claim := WorktreeClaim.locate(target):
        claim.delete()


def _worktree_root(repo_root: Path, common_git_dir: Path) -> Path:
    repo_dir = managed_bucket_name(repo_root, common_git_dir)
    managed_root = WORKTREES_DIR.path.resolve()
    target_root = (managed_root / repo_dir).resolve()
    if not target_root.is_relative_to(managed_root):
        raise WorktreeError(
            f"Managed worktree root {target_root} resolves outside {managed_root}."
        )
    return target_root


def _validate_existing_worktree(
    target: Path, expected_branch: str, expected_common_git_dir: Path
) -> None:
    git = _git_python()
    if _has_linked_path_component(target):
        raise WorktreeError(
            f"Path {target} contains a symbolic link or junction, "
            "not a stable git worktree path."
        )
    if not (target / ".git").is_file():
        raise WorktreeError(f"Path {target} already exists but is not a git worktree.")

    try:
        existing_repo = git.repo(target)
    except git.invalid_git_repository_error as e:
        raise WorktreeError(
            f"Path {target} already exists but is not a git worktree."
        ) from e

    try:
        rev_parse_parts = existing_repo.git.rev_parse(
            "--git-common-dir", "--abbrev-ref", "HEAD"
        ).splitlines()
    except git.git_command_error as e:
        raise WorktreeError(f"Failed to inspect worktree {target}: {e}") from e
    if len(rev_parse_parts) != _WORKTREE_REV_PARSE_PARTS:
        raise WorktreeError(
            f"Failed to inspect worktree {target}: expected git rev-parse to return "
            f"{_WORKTREE_REV_PARSE_PARTS} lines, got {len(rev_parse_parts)}."
        )
    common_dir_value, branch = rev_parse_parts

    existing_common_git_dir = GitRepo(existing_repo).resolve_git_dir(common_dir_value)
    if existing_common_git_dir != expected_common_git_dir:
        raise WorktreeError(f"Path {target} belongs to a different git repository.")

    if branch == "HEAD" or branch != expected_branch:
        actual = "detached HEAD" if branch == "HEAD" else branch
        raise WorktreeError(
            f"Path {target} is checked out on {actual!r}, expected {expected_branch!r}."
        )


def _validate_listed_worktree(target: Path, expected_common_git_dir: Path) -> None:
    """The part of :func:`_validate_existing_worktree` a *listing* still needs.

    The callers that adopt a directory have to establish what it is: the path
    merely exists, and it has to be proven a worktree of this repository on the
    branch expected. A listing starts from the other end. Every record here came
    out of this repository's own ``git worktree list``, so the repository it
    belongs to and the branch it is on are things git has already answered.

    Re-asking costs a GitPython repository object plus a ``git rev-parse``
    subprocess *per worktree*, and that is the entire cost of listing: roughly
    29ms x N worktrees, paid on every listing a session's project resolution
    walks.

    The filesystem checks stay. They cost a symlink check per path component
    and one ``.is_file()``, and they catch the one thing the record cannot: a
    path removed or swapped for a symlink since git wrote it. The ``.git``
    pointer adds the identity half of the full validation at the same
    filesystem-only price: a linked worktree of this repository names a gitdir
    under the common git dir's ``worktrees/`` directory, so a directory whose
    ``.git`` file points at another repository's worktree is refused without a
    subprocess, stale branch label notwithstanding.
    """
    if _has_linked_path_component(target):
        raise WorktreeError(
            f"Path {target} contains a symbolic link or junction, "
            "not a stable git worktree path."
        )
    git_marker = target / ".git"
    if not git_marker.is_file():
        raise WorktreeError(f"Path {target} already exists but is not a git worktree.")
    try:
        pointer = git_marker.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as e:
        raise WorktreeError(f"Failed to inspect worktree {target}: {e}") from e
    prefix, separator, value = pointer.partition(":")
    value = value.strip()
    if not separator or prefix.casefold() != "gitdir" or not value:
        # Not a usable worktree pointer; git itself would refuse it.
        raise WorktreeError(f"Path {target} already exists but is not a git worktree.")
    git_dir = Path(value).expanduser()
    if not git_dir.is_absolute():
        git_dir = target / git_dir
    try:
        resolved_git_dir = git_dir.resolve(strict=True)
        expected_worktrees = (expected_common_git_dir / "worktrees").resolve(
            strict=True
        )
    except (OSError, RuntimeError) as e:
        raise WorktreeError(f"Failed to inspect worktree {target}: {e}") from e
    if not resolved_git_dir.is_dir():
        raise WorktreeError(f"Path {target} has no usable worktree git directory.")
    if not resolved_git_dir.is_relative_to(expected_worktrees):
        raise WorktreeError(f"Path {target} belongs to a different git repository.")


def _has_linked_path_component(path: Path) -> bool:
    current = Path(path.anchor) if path.anchor else Path()
    parts = path.parts[1:] if path.anchor else path.parts
    for index, part in enumerate(parts):
        current /= part
        # Root-level aliases such as macOS /tmp -> /private/tmp are controlled by
        # the operating system, not by the worktree directory hierarchy.
        if path.anchor and index == 0:
            continue
        is_junction = getattr(current, "is_junction", None)
        if current.is_symlink() or (is_junction is not None and is_junction()):
            return True
    return False


def _target_cwd(target: Path, relative_base: Path) -> Path:
    root = target.resolve()
    target_cwd = root / relative_base
    try:
        resolved_cwd = target_cwd.resolve(strict=True)
    except (OSError, RuntimeError) as e:
        raise WorktreeError(
            f"Worktree path {target_cwd} does not exist after checkout."
        ) from e
    if not resolved_cwd.is_dir():
        raise WorktreeError(f"Worktree path {target_cwd} is not a directory.")
    try:
        resolved_cwd.relative_to(root)
    except ValueError as e:
        raise WorktreeError(
            f"Worktree path {target_cwd} resolves outside worktree {root}."
        ) from e
    # Landing inside the worktree is not enough. `root` is already resolved, so
    # anything the join still moves is a link below it, and one pointing at a
    # sibling satisfies the check above while naming a directory nobody asked
    # for. That directory becomes the session's position, its write root and a
    # trust grant, so the redirect is refused rather than followed. Committing
    # the link would otherwise make this the branch's choice, not the user's.
    if resolved_cwd != target_cwd:
        raise WorktreeError(
            f"Worktree path {target_cwd} is reached through a symbolic link to "
            f"{resolved_cwd}."
        )
    current = resolved_cwd
    while current != root:
        git_marker = current / ".git"
        if git_marker.exists() or git_marker.is_symlink():
            raise WorktreeError(
                f"Worktree path {resolved_cwd} belongs to a different git repository."
            )
        current = current.parent
    return resolved_cwd
