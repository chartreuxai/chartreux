from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator, Sequence
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import errno
import getpass
import hashlib
import io
import itertools
import json
import logging
import os
from pathlib import Path
import subprocess
from threading import Lock
import time
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4
import warnings

from chartreux.core.llm_models import (
    CONTEXT_INJECTION_ORIGIN_KEY,
    CONTEXT_INJECTION_TOOL_PAIR_V1,
    AvailableFunction,
    LLMMessage,
    Role,
    archived_replacement,
)
from chartreux.core.message_list import MessageList
from chartreux.core.session.session_loader import (
    MESSAGES_FILENAME,
    METADATA_FILENAME,
    SessionLoader,
)
from chartreux.core.session.session_permissions import (
    ensure_private_directory,
    restrict_private_file,
    restrict_session_permissions,
)
from chartreux.core.session.transcript_archive import (
    ArchivedPrefixV1,
    ArchiveIntegrityError,
    ArchiveSaveSnapshot,
    ArchiveSealError,
    ArchiveSealReceipt,
    ReboundHydrationProof,
    TrimAttemptKey,
    archived_prefix_chunks,
    eligible_archive_prefix,
    eligible_pre_seal_prefix,
    iter_archived_records,
    pre_seal_endpoint,
    prove_rebound_hydration,
    trim_attempt_key,
    verify_archived_prefix_on_disk,
)
from chartreux.core.session_types import (
    AgentStats,
    LaunchMetadata,
    PlanModeRecord,
    SessionMetadata,
    WorktreeContext,
)
from chartreux.core.tools.secret_redaction import scrub_child_env
from chartreux.core.utils import utc_now
from chartreux.utils.durable_io import (
    durable_append,
    durable_replace,
    durable_replace_stream,
    read_chunks,
)
from chartreux.utils.io import read_safe, read_safe_async
from chartreux.utils.platform import resolve_git_executable
from chartreux.utils.session_id import shorten_session_id

if TYPE_CHECKING:
    from chartreux.core.agent_loop._transcript_integrity import ArchivedPrefixAuthority
    from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
    from chartreux.core.tools.manager import ToolManager


TMP_CLEANUP_INTERVAL = timedelta(seconds=5)
TRANSCRIPT_VERIFY_INTERVAL = 64
# Cooperative GIL release cadence for the persisted-transcript verification
# (P6). The verify runs on the save worker thread while the event loop keeps
# rendering the stream, so a 20k-record compare starves frame flushes unless
# it hands the GIL back periodically. 128 is the measured optimum: tuning on
# the P0d paired-trial harness recorded 64 -> 41.6% and 256 -> 41.7% median
# flush-p95 degradation against 128's best point.
PERSISTED_VERIFY_YIELD_INTERVAL = 128
# Disk-full save failures are logged at most this often while the condition
# persists; a successful save resets the rate limit so the next occurrence is
# reported immediately.
DISK_FULL_LOG_INTERVAL = timedelta(seconds=60)
logger = logging.getLogger(__name__)
# Fields a M1 tombstone retains in its serialized record: role, message_id
# (None drops out of the record via exclude_none), injected. The private
# archive reference never serializes, so it cannot appear in a record.
_TOMBSTONE_RETAINED_KEYS: frozenset[str] = frozenset({"role", "message_id", "injected"})
# The resolved archive-marker attribute NAME (P6). A module-level import of
# ``chartreux.core.agent_loop._transcript_integrity`` would cycle
# (agent_loop -> _loop -> this module), so the function-level import runs
# ONCE per process and the resolved name is cached here; every per-record
# ``_message_record`` call then probes ``getattr(message, name, None)``
# directly instead of paying a function call per record on the save path.
_archive_reference_attribute_name: str | None = None


class SessionDiskFullError(RuntimeError):
    """A session persistence write failed with ENOSPC: the disk is full."""

    code = "session_disk_full"

    def __init__(self, path: Path, operation: str) -> None:
        self.path = path
        self.operation = operation
        super().__init__(
            f"Disk is full: failed to {operation} at {path}. "
            "The session continues in memory; persistence retries on the next save."
        )


def _is_enospc(exc: BaseException) -> bool:
    """Detect ENOSPC (disk full), unwrapping exception groups."""
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_enospc(child) for child in exc.exceptions)
    return isinstance(exc, OSError) and exc.errno == errno.ENOSPC


class ArchiveUnavailableError(RuntimeError):
    """A retained archive descriptor could not be honored on the save path.

    Raised on the save/verify path when an authoritative archive descriptor
    is retained but its archived bytes cannot be proven on disk, or the
    retained prefix does not match the declared trim (P4a3, M1 matrix row
    "missing/truncated/changed archived bytes" and the authority-state
    class). ``save_interaction``'s non-raising facade surfaces it as a
    FAILED save with the diagnostic recorded (``last_save_diagnostic``);
    while a descriptor is retained the session never re-enters the legacy
    rewrite path. ``recoverable_by_restart`` splits the failure classes:
    authority-state failures recover by SESSION RESTART; missing, truncated,
    or changed archived bytes are UNRECOVERABLE in place — the archived
    history is gone and a new conversation is required (never rebuilt from
    trimmed messages or tombstones).
    """

    code = "archive_unavailable"

    def __init__(
        self, path: Path, reason: str, *, recoverable_by_restart: bool
    ) -> None:
        self.path = path
        self.reason = reason
        self.recoverable_by_restart = recoverable_by_restart
        if recoverable_by_restart:
            message = (
                f"Archive authority failure at {path}: {reason}. The session "
                "does not re-enter the legacy rewrite path while a descriptor "
                "is retained; recover by session restart."
            )
        else:
            message = (
                f"Archived transcript prefix at {path} is missing, truncated, "
                f"or changed: {reason}. The archived history is unrecoverable "
                "in place; a new conversation is required."
            )
        super().__init__(message)


class _PersistedTranscriptError(ValueError):
    """A persisted transcript line could not be parsed as a JSON record.

    The streaming full-verify reader raises this instead of returning a
    partial result; the caller treats it as a corrupted transcript
    (quarantine + rewrite), exactly like the bulk reader's failure result.
    """


def _transcript_encoding(prefix: bytes) -> str:
    """The codec the transcript's byte-order mark selects.

    Applies the same BOM table as ``decode_safe``, the pre-streaming bulk
    reader's decode path: a UTF-16/UTF-32 transcript that reader accepted
    must keep decoding rather than classify as corrupt. The 32-bit marks are
    checked first because the UTF-32-LE mark starts with the UTF-16-LE one.
    Without a mark the transcript is UTF-8.
    """
    if prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if prefix.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return "utf-32"
    if prefix.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    return "utf-8"


class _SizedByteStream:
    """A byte iterator whose ``__length_hint__`` sizes ``bytes()`` exactly.

    ``bytes()`` pre-sizes its buffer from the source's length hint when one
    is available; without it the buffer grows with over-allocation, which
    on a multi-megabyte transcript payload is measurable residency.
    """

    __slots__ = ("_source", "_total")

    def __init__(self, source: Iterator[int], total: int) -> None:
        self._source = source
        self._total = total

    def __iter__(self) -> Iterator[int]:
        return self._source

    def __length_hint__(self) -> int:
        return self._total


@dataclass(frozen=True)
class _TranscriptCursor:
    # A future MessageList.__setitem__ could bypass the boundary check; periodic
    # full verification is the backstop for same-size interior edits.
    count: int
    boundary_digest: str | None
    file_size: int
    metadata_published: bool


@dataclass(frozen=True)
class _PromptSerialization:
    """The metadata blobs a content hash gates, cached in logger memory.

    P2 keeps the previous hash of the system_prompt/tools_available inputs
    here — in process memory only, never in persisted metadata — so a restarted
    logger re-serializes once and then rebuilds the hash from that first save.
    """

    content_hash: bytes
    system_prompt: dict[str, Any] | None
    tools_available: list[dict[str, Any]]


#: The explicit outcome of one post-turn trim completion attempt (T1b). Every
#: attempt reports exactly one of these; nothing is silently skipped.
PostTurnTrimOutcome = Literal[
    # The pending trim was drained, fenced, sealed and published.
    "published",
    # A seal/publication for the same boundary is already in flight.
    "excluded_concurrent",
    # The boundary was already sealed and published (e.g. a later seam).
    "excluded_already_published",
    "refused_disabled",
    "refused_session_mismatch",
    # The pending's message epoch does not match the live list (stale after a
    # rebind): refused at the attempt boundary, never sealed.
    "refused_epoch_mismatch",
    "refused_boundary_mismatch",
    # A retained descriptor pairs with a different list epoch: stale authority.
    "refused_stale_authority",
    # Sealing was refused or failed: no receipt, originals retained.
    "failed_seal",
    # The seal succeeded but publication failed: the pre-seal archived bytes
    # and descriptor remain effective; the receipt is carried for reporting.
    "failed_publish",
]


@dataclass(frozen=True, slots=True)
class PostTurnTrimResult:
    """The reported outcome of a ``complete_post_turn_trim`` attempt (T1b)."""

    outcome: PostTurnTrimOutcome
    detail: str = ""
    receipt: ArchiveSealReceipt | None = None


#: The explicit outcome of one warm-resume clear-and-reseal proof/transaction
#: attempt (T2a). Every attempt reports exactly one of these; nothing is
#: silently skipped.
WarmResumeOutcome = Literal[
    # The hydration proof held and the staged repair + authority transition
    # completed under the save lock: the retained descriptor is CLEARED
    # (clear-and-reseal), never re-pinned.
    "proved_and_transitioned",
    "refused_disabled",
    "refused_session_mismatch",
    # The hydration proof failed closed: archive authority and the originals
    # stay — the on-disk bytes are byte-identical, nothing was staged, and no
    # descriptor re-pin fallback exists.
    "refused_proof",
    # The proof held but the staged repair failed: the rollback transition
    # completed (original bytes restored / retained) before the fence moved.
    "failed_repair_rolled_back",
]


@dataclass(frozen=True, slots=True)
class WarmResumeTransactionResult:
    """The reported outcome of a ``clear_and_reseal_warm_resume`` attempt (T2a)."""

    outcome: WarmResumeOutcome
    detail: str = ""
    proof: ReboundHydrationProof | None = None
    artifact_path: Path | None = None


# Over the method limit, as AgentLoop and ChartreuxApp already are: most of thesethese
# are one-line accessors onto the metadata record this owns, and hiding them
# behind a second object would put a hop between the log and everything that
# reads it.
class SessionLogger:  # noqa: PLR0904
    def __init__(
        self,
        session_config: SessionLoggingConfig,
        session_id: str,
        *,
        cwd: Path | None = None,
        session_dir: Path | None = None,
    ) -> None:
        self.session_config = session_config
        self.cwd = (cwd or Path.cwd()).resolve()
        self._title: str | None = None
        self.enabled = session_config.enabled
        self._last_tmp_cleanup_at: datetime | None = None
        self._tmp_cleanup_lock = Lock()
        self._persisted = False
        self._transcript_cursor: _TranscriptCursor | None = None
        self._transcript_saves_since_verify = 0
        self._transcript_cursor_generation = 0
        # M1 descriptor retention: the archive descriptor is retained
        # independently of _TranscriptCursor — cursor invalidation discards
        # append acceleration, never archive authority.
        self._archive_descriptor: ArchivedPrefixV1 | None = None
        self._last_save_archive_snapshot: ArchiveSaveSnapshot | None = None
        # P4a4 reset generation: bumped by every logger rebind (reset_session /
        # apply_resumed_session). A trim publication planned against an older
        # reset generation is stale and never overwrites the current authority.
        self._archive_generation = 0
        # P4a3 rev J: the fail-closed diagnostic of the most recent failed
        # save (an ArchiveUnavailableError surfaced through the non-raising
        # save facade). Cleared by the next successful save.
        self._last_save_diagnostic: ArchiveUnavailableError | None = None
        self._launch_config_dirty = False
        self._launch_config_generation = 0
        # P2: the last prompt/tool serialization and the content hash it came
        # from. Logger memory only, so a restart re-serializes once.
        self._prompt_serialization: _PromptSerialization | None = None
        # Rate-limits repeated disk-full warnings so saves cannot spam the log.
        self._disk_full_last_logged_at: datetime | None = None
        # Serializes writes so concurrent saves cannot interleave appends to
        # messages.jsonl or race on the metadata read-modify-write.
        self._save_lock = asyncio.Lock()
        # T1b post-turn trim completion: exclusivity per release-attempt key.
        # A key is in ``_trim_in_flight_boundaries`` while its
        # drain/fence/seal/publication attempt runs and in
        # ``_trim_published_boundaries`` once published; a second attempt for
        # the same key is excluded, never a concurrent seal/publication. The
        # key is the widened seam key (``trim_attempt_key``): a real compaction
        # boundary keys as ("boundary", session, epoch, boundary id), a
        # pre-seal release as ("pre-seal", session, epoch, endpoint) — never a
        # bare None key.
        self._trim_in_flight_boundaries: set[TrimAttemptKey] = set()
        self._trim_published_boundaries: set[TrimAttemptKey] = set()

        # Repair every configured store independently, including when new logging
        # is disabled. The repair recognizes session directories and leaves other
        # files in the configured root alone.
        restrict_session_permissions(session_config)

        if not self.enabled:
            self.save_dir: Path | None = None
            self.session_prefix: str | None = None
            self.session_id: str = "disabled"
            self.session_start_time: str = "N/A"
            self.session_dir: Path | None = None
            self.session_metadata: SessionMetadata | None = None
            return

        self.save_dir = Path(session_config.save_dir)
        self.session_prefix = session_config.session_prefix
        self.session_id = session_id
        self.session_start_time = utc_now().isoformat()

        ensure_private_directory(Path(session_config.permission_repair_dir))
        if session_dir is not None:
            self.resume_existing_session(session_id, session_dir)
            return
        self.session_dir = self.save_folder
        self.session_metadata = self._initialize_session_metadata()

    @property
    def save_folder(self) -> Path:
        if self.save_dir is None or self.session_prefix is None:
            raise RuntimeError(
                "Cannot get session save folder when logging is disabled"
            )

        timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
        folder_name = (
            f"{self.session_prefix}_{timestamp}_{shorten_session_id(self.session_id)}"
        )
        return self.save_dir / folder_name

    @property
    def persisted(self) -> bool:
        return self._persisted

    def invalidate_transcript_cursor(self) -> None:
        """Force the next interaction save to verify the full transcript.

        Discards append acceleration only: the retained archive descriptor
        (M1) lives independently of the cursor and is NOT cleared here.
        """
        self._transcript_cursor_generation += 1
        self._transcript_cursor = None
        self._transcript_saves_since_verify = 0

    @property
    def archive_descriptor(self) -> ArchivedPrefixV1 | None:
        """The retained archive descriptor (M1), or None when unarchived."""
        return self._archive_descriptor

    @property
    def transcript_save_generation(self) -> int:
        """The current save generation; moves on every cursor invalidation.

        Read by the trim wiring to plan a ``fence_and_publish_archive_trim``
        call against the generations it validates.
        """
        return self._transcript_cursor_generation

    @property
    def archive_reset_generation(self) -> int:
        """The current reset generation; moves on every logger rebind."""
        return self._archive_generation

    @property
    def last_save_archive_snapshot(self) -> ArchiveSaveSnapshot | None:
        """The archive side of the most recent accepted save snapshot (M1)."""
        return self._last_save_archive_snapshot

    @property
    def last_save_diagnostic(self) -> ArchiveUnavailableError | None:
        """The fail-closed diagnostic of the most recent failed save (P4a3).

        ``save_interaction`` never raises an archive-unavailable diagnostic;
        it records it here and reports a FAILED save instead.
        """
        return self._last_save_diagnostic

    def _record_save_failure(self, diagnostic: ArchiveUnavailableError) -> None:
        """Record a fail-closed save diagnostic on the non-raising facade."""
        self._last_save_diagnostic = diagnostic
        logger.warning(
            "Session save failed closed (%s): %s", diagnostic.code, diagnostic
        )

    def _active_archive_snapshot(self) -> ArchiveSaveSnapshot:
        """The archive side of the save the worker thread is executing.

        Read under the save lock: ``save_interaction`` (and ``seal_archive``)
        publish the snapshot before spawning the persist worker, so the
        worker sees exactly the snapshot of the save it runs. With no
        published snapshot a retained descriptor pairs with no epoch, which
        is non-authoritative (P4a2: only sealing refuses that pairing).
        """
        snapshot = self._last_save_archive_snapshot
        if snapshot is not None:
            return snapshot
        return ArchiveSaveSnapshot(
            descriptor=self._archive_descriptor, message_epoch=None
        )

    def retain_archive_descriptor(self, descriptor: ArchivedPrefixV1) -> None:
        """Retain the archive descriptor matching this session (M1).

        Retention is independent of the transcript cursor: a later cursor
        invalidation does not delete this authority. A descriptor naming a
        different session is rejected fail-closed, and so is a STALE one: a
        descriptor whose epoch is older than the retained one never overwrites
        the current authority (P4a4).
        """
        if self.enabled and descriptor.session_id != self.session_id:
            raise ValueError(
                f"archive descriptor session_id {descriptor.session_id!r} does not "
                f"match this logger's session {self.session_id!r}"
            )
        current = self._archive_descriptor
        if current is not None and descriptor.message_epoch < current.message_epoch:
            raise ValueError(
                f"stale archive descriptor epoch {descriptor.message_epoch} would "
                f"overwrite the retained epoch {current.message_epoch}; a stale "
                "descriptor never overwrites the current one"
            )
        self._archive_descriptor = descriptor

    def _drop_mismatched_archive_authority(self, session_id: str) -> None:
        """Re-validate retained archive authority on a logger rebind (P4a2).

        Retention is session-scoped: a rebind onto a different session drops a
        retained descriptor naming the old one, fail-closed, while a matching
        descriptor (a warm continuation onto the same session) is retained.
        """
        descriptor = self._archive_descriptor
        if descriptor is not None and descriptor.session_id != session_id:
            self._archive_descriptor = None

    def _archive_authority_for(
        self, messages: Sequence[LLMMessage]
    ) -> ArchivedPrefixAuthority | None:
        """The validation-side archive authority for the snapshot being saved.

        Minted (P4a4, M1 §4) only when a retained descriptor's epoch matches
        the saved list's epoch — the authoritative pairing. Without it the
        validators get no authority, so any archive marker in the snapshot
        fails closed there. A PRE-SEAL descriptor (``boundary_message_id``
        None, P3) is widened to the snapshot's current non-system count: the
        whole live list is covered by the verified transcript, so the
        authority carries no tail that would need a compaction boundary.
        Function-level import: a top-level import of the agent_loop validator
        would cycle (agent_loop -> _loop -> this module).
        """
        descriptor = self._archive_descriptor
        if descriptor is None:
            return None
        message_epoch = (
            messages.message_epoch if isinstance(messages, MessageList) else None
        )
        if message_epoch is None or descriptor.message_epoch != message_epoch:
            return None
        if descriptor.boundary_message_id is None:
            from chartreux.core.agent_loop._transcript_integrity import (
                ARCHIVE_REFERENCE_ATTRIBUTE,
                ArchivedPrefixAuthority,
            )

            if any(
                getattr(message, ARCHIVE_REFERENCE_ATTRIBUTE, None) is not None
                for message in messages
            ):
                # A boundary-None descriptor over MARKED slots is not the
                # cover-all pre-seal shape — the pre-seal publication replaces
                # every slot with an UNMARKED copy — it is a partial seal
                # (e.g. handed off across a fork/rewind). Mint the plain
                # authority over the descriptor's own archived_count so the
                # markers still match the sealed prefix exactly.
                return self._mint_archive_authority(descriptor)
            non_system_count = sum(
                1 for message in messages if message.role is not Role.system
            )
            return ArchivedPrefixAuthority(
                archived_count=non_system_count,
                boundary_message_id=None,
                message_epoch=descriptor.message_epoch,
            )
        return self._mint_archive_authority(descriptor)

    @staticmethod
    def _mint_archive_authority(
        descriptor: ArchivedPrefixV1,
    ) -> ArchivedPrefixAuthority:
        """The validation token for an ALREADY authoritative descriptor.

        Used by the save boundary (via ``_archive_authority_for``) and by the
        sealing planner, which holds a plain message snapshot — no list epoch
        to re-derive. Every caller must have established the authoritative
        pairing first. Function-level import: a top-level import of the
        agent_loop validator would cycle (agent_loop -> _loop -> this module).
        """
        from chartreux.core.agent_loop._transcript_integrity import (
            ArchivedPrefixAuthority,
        )

        return ArchivedPrefixAuthority(
            archived_count=descriptor.archived_count,
            boundary_message_id=descriptor.boundary_message_id,
            message_epoch=descriptor.message_epoch,
        )

    @property
    def active_model(self) -> str | None:
        metadata = self.session_metadata
        if metadata is None or metadata.config is None:
            return None
        active_model = metadata.config.get("active_model")
        return active_model if isinstance(active_model, str) and active_model else None

    def _get_session_info(self) -> tuple[Path, SessionMetadata] | None:
        if (
            not self.enabled
            or self.session_dir is None
            or self.session_metadata is None
        ):
            return None
        return (self.session_dir, self.session_metadata)

    @property
    def metadata_filepath(self) -> Path:
        if self.session_dir is None:
            raise RuntimeError(
                "Cannot get session metadata filepath when logging is disabled"
            )
        return self.session_dir / METADATA_FILENAME

    @property
    def messages_filepath(self) -> Path:
        if self.session_dir is None:
            raise RuntimeError(
                "Cannot get session messages filepath when logging is disabled"
            )
        return self.session_dir / MESSAGES_FILENAME

    def _fetch_git_metadata(self) -> tuple[str | None, str | None]:
        """Fetch git commit and branch in a single subprocess call."""
        git_executable = resolve_git_executable(cwd=self.cwd)
        if git_executable is None:
            return None, None
        try:
            result = subprocess.run(
                [git_executable, "rev-parse", "HEAD", "--abbrev-ref", "HEAD"],
                capture_output=True,
                stdin=None,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5.0,
                cwd=self.cwd,
                env=scrub_child_env(os.environ),
            )
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().splitlines()
                commit = lines[0] if len(lines) > 0 else None
                branch = lines[1] if len(lines) > 1 else None
                return commit, branch
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            pass
        return None, None

    @property
    def git_commit(self) -> str | None:
        return self._fetch_git_metadata()[0]

    @property
    def git_branch(self) -> str | None:
        return self._fetch_git_metadata()[1]

    @property
    def username(self) -> str:
        try:
            return getpass.getuser()
        except Exception:
            return "unknown"

    def _initialize_session_metadata(self) -> SessionMetadata:
        git_commit, git_branch = self._fetch_git_metadata()
        user_name = self.username

        return SessionMetadata(
            session_id=self.session_id,
            start_time=self.session_start_time,
            end_time=None,
            git_commit=git_commit,
            git_branch=git_branch,
            username=user_name,
            environment={"working_directory": str(self.cwd)},
            origin_directory=str(self.cwd),
            title=None,
            title_source="auto",
        )

    def _set_title_state(
        self, title: str | None, *, source: Literal["auto", "manual"]
    ) -> None:
        self._title = title
        if self.session_metadata is None:
            return

        self.session_metadata.title = title
        self.session_metadata.title_source = source

    def relocated_to(self, cwd: Path) -> None:
        """Record that the session now sits at *cwd*.

        ``environment.working_directory`` is what the interface shows as the
        session's directory, so it names where the agent is working rather than
        where it started. ``origin_directory`` is left where it was: it is what
        keeps the session findable from the place the user began it.

        A session that predates ``origin_directory``, or that was imported, has
        only the environment entry, and that entry is where it started. The
        first move has to promote it, or overwriting the entry would leave the
        session with no record of its origin at all, which is exactly the
        disappearance this pair of fields exists to prevent.
        """
        self.cwd = cwd.resolve()
        if self.session_metadata is None:
            return

        environment = self.session_metadata.environment
        if self.session_metadata.origin_directory is None:
            self.session_metadata.origin_directory = environment.get(
                "working_directory"
            )
        environment["working_directory"] = str(self.cwd)

    def _set_title(self, title: str | None) -> None:
        if title is None:
            self._set_title_state(None, source="auto")
            return

        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("Session title cannot be empty.")

        self._set_title_state(normalized_title, source="manual")

    @property
    def title(self) -> str | None:
        if self.session_metadata is not None:
            return self.session_metadata.title
        return self._title

    @property
    def title_source(self) -> Literal["auto", "manual"]:
        if self.session_metadata is not None:
            return self.session_metadata.title_source
        return "auto"

    def needs_initial_auto_title(self) -> bool:
        return self.title is None

    def set_initial_auto_title(self, title: str) -> bool:
        if not self.needs_initial_auto_title():
            return False

        normalized_title = title.strip()
        if not normalized_title:
            return False

        self._set_title_state(normalized_title, source="auto")
        return True

    # Two writers touch the title: the background auto-refresh
    # (``refresh_auto_title``) and a manual ``/rename`` (``apply_manual_title``),
    # possibly concurrently. The invariant that keeps them correct:
    #   1. Both mutate title state and persist only while holding ``_save_lock``.
    #   2. ``refresh_auto_title`` re-checks ``title_source`` under the lock and
    #      bails if a manual rename already won, so a manual title is never
    #      clobbered by an auto one.
    #   3. Both persist first and flip memory only after the write succeeds, so a
    #      failed read or write leaves memory and disk consistent (old value).
    # A manual title is therefore always the winner of any race.
    async def refresh_auto_title(
        self, title: str, *, expected_session_id: str | None = None
    ) -> bool:
        """Replace an auto-generated title and persist it to disk.

        Never overrides a manual rename. ``expected_session_id`` guards against a
        background generation that outlived a ``/new`` or ``/clear`` reset: the
        logger is reset in place, so a title generated for the old conversation
        must not land on the new one. Returns whether the title changed.
        """
        if self.session_metadata is None or self.title_source == "manual":
            return False

        normalized_title = title.strip()
        if not normalized_title or normalized_title == self.title:
            return False

        async with self._save_lock:
            # Re-check under the lock: a /rename may have landed and must win.
            if self.title_source == "manual":
                return False
            # A reset swaps this logger to a new session id; bail so the old
            # title isn't persisted onto it. The check is immediately before the
            # persist (which captures the session dir synchronously), so no reset
            # can slip between the guard and the write.
            if (
                expected_session_id is not None
                and self.session_id != expected_session_id
            ):
                return False
            # Persist first, flip memory after: a failed persist then leaves
            # memory and disk consistent (both still the old auto title).
            await self._persist_title_fields_locked(normalized_title, "auto")
            self._set_title_state(normalized_title, source="auto")
        return True

    async def apply_manual_title(self, title: str) -> str | None:
        """Flip the title to a manual rename and persist under the save lock.

        Persists first and flips memory only after the write succeeds, so a
        failed read or write never leaves memory manual while disk stays auto. A
        concurrent auto-refresh rechecks title_source under the same lock and
        bails. Returns the persisted end_time, or None when there is no on-disk
        metadata yet.
        """
        async with self._save_lock:
            session_info = self._get_session_info()
            if session_info is None:
                self._set_title(title)
                return None
            session_dir, _ = session_info
            metadata_path = session_dir / METADATA_FILENAME
            if not metadata_path.exists():
                self._set_title(title)
                return None
            normalized_title = title.strip()
            if not normalized_title:
                raise ValueError("Session title cannot be empty.")
            try:
                raw = (await read_safe_async(metadata_path)).text
                metadata = json.loads(raw)
            except (OSError, json.JSONDecodeError) as e:
                raise RuntimeError(
                    f"Failed to read session metadata at {metadata_path}: {e}"
                ) from e
            metadata["title"] = normalized_title
            metadata["title_source"] = "manual"
            await self._persist_metadata_locked(metadata, session_dir)
            self._set_title_state(normalized_title, source="manual")
        end_time = metadata.get("end_time")
        return end_time if isinstance(end_time, str) else None

    @staticmethod
    def _persist_metadata_sync(metadata: Any, session_dir: Path) -> None:
        metadata_filepath = session_dir / METADATA_FILENAME
        try:
            durable_replace(
                metadata_filepath,
                json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8"),
            )
        except Exception as e:
            if _is_enospc(e):
                raise SessionDiskFullError(
                    metadata_filepath, "persist session metadata"
                ) from e
            raise RuntimeError(
                f"Failed to persist session metadata to {metadata_filepath}: {e}"
            ) from e

    @staticmethod
    async def persist_metadata(metadata: Any, session_dir: Path) -> None:
        await asyncio.to_thread(
            SessionLogger._persist_metadata_sync, metadata, session_dir
        )

    async def _persist_metadata_locked(self, metadata: Any, session_dir: Path) -> None:
        """Finish a metadata write before cancellation can release the save lock."""
        persistence = asyncio.create_task(self.persist_metadata(metadata, session_dir))
        try:
            await asyncio.shield(persistence)
        except asyncio.CancelledError as cancellation:
            while not persistence.done():
                try:
                    await asyncio.shield(persistence)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not persistence.cancelled():
                with contextlib.suppress(Exception):
                    persistence.result()
            raise cancellation

    @staticmethod
    def _persist_messages_sync(messages: Iterable[dict], session_dir: Path) -> int:
        messages_filepath = session_dir / MESSAGES_FILENAME
        try:
            durable_append(
                messages_filepath,
                SessionLogger._transcript_bytes(messages),
                ensure_newline=True,
            )
            return messages_filepath.stat().st_size
        except Exception as e:
            if _is_enospc(e):
                raise SessionDiskFullError(
                    messages_filepath, "persist session messages"
                ) from e
            raise RuntimeError(
                f"Failed to persist session messages to {messages_filepath}: {e}"
            ) from e

    @staticmethod
    async def persist_messages(messages: list[dict], session_dir: Path) -> None:
        await asyncio.to_thread(
            SessionLogger._persist_messages_sync, messages, session_dir
        )

    @staticmethod
    def _overwrite_messages_sync(
        messages: Iterable[dict[str, Any]], session_dir: Path
    ) -> None:
        """Rewrite the transcript from records produced one at a time (P4b).

        The records stream into the replacement, so neither the record list
        nor the serialized payload is resident whole; the caller may pass a
        generator over the live messages.
        """
        messages_filepath = session_dir / MESSAGES_FILENAME
        try:
            durable_replace_stream(
                messages_filepath, SessionLogger._transcript_chunks(messages)
            )
        except Exception as e:
            if _is_enospc(e):
                raise SessionDiskFullError(
                    messages_filepath, "overwrite session messages"
                ) from e
            raise RuntimeError(
                f"Failed to overwrite session messages at {messages_filepath}: {e}"
            ) from e

    @staticmethod
    def _message_record(
        message: LLMMessage, *, allow_archived: bool = False
    ) -> dict[str, Any]:
        """The explicit session-transcript record for one message (round-5 Y).

        Starts with the ordinary ``model_dump`` (which NEVER emits the
        private origin marker — private attributes are absent without an
        exclusion) and adds the recognized provenance key ONLY when the
        message is an assistant tool-call block whose private marker is an
        exact string equal to the recognized constant. The proof therefore
        travels atomically with its owning message in ``messages.jsonl``, on
        every serialization/fingerprint path (the fingerprint, the durable
        repaired-transcript serialization, the incremental boundary
        comparison and tail serialization, and the full-save
        serialization/comparison).

        M1: a message carrying the private archive reference is an
        archived/tombstone replacement, and that marker must never be
        persisted — the archived bytes on disk are the only authoritative
        record of a sealed slot. Persist-path calls therefore REJECT marked
        messages fail-closed (the default); only the streaming
        verify/comparison paths pass ``allow_archived=True``, because they
        compare the retained trimmed record against disk and never write.
        """
        global _archive_reference_attribute_name
        if _archive_reference_attribute_name is None:
            from chartreux.core.agent_loop._transcript_integrity import (
                ARCHIVE_REFERENCE_ATTRIBUTE,
            )

            _archive_reference_attribute_name = ARCHIVE_REFERENCE_ATTRIBUTE
        if (
            not allow_archived
            and getattr(message, _archive_reference_attribute_name, None) is not None
        ):
            raise ValueError(
                "an archived/tombstone message marker must never be persisted "
                "to the session transcript; the archived bytes on disk are the "
                "only authoritative record of a sealed slot"
            )
        record = message.model_dump(exclude_none=True, mode="json")
        if (
            message.role == Role.assistant
            and message.tool_calls
            and type(message._context_injection_origin) is str
            and message._context_injection_origin == CONTEXT_INJECTION_TOOL_PAIR_V1
        ):
            record[CONTEXT_INJECTION_ORIGIN_KEY] = CONTEXT_INJECTION_TOOL_PAIR_V1
        return record

    @staticmethod
    def _message_fingerprint(message: LLMMessage) -> str:
        payload = json.dumps(SessionLogger._message_record(message), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # Keep cancellation draining and error handling in one lock scope.
    async def save_interaction(  # noqa: PLR0915, PLR0912 - one serialized commit boundary
        self,
        messages: Sequence[LLMMessage],
        stats: AgentStats,
        config: ChartreuxConfigSchema,
        tool_manager: ToolManager,
        *,
        allow_empty: bool = False,
    ) -> None:
        """Save an ordinary snapshot, discarding stale snapshots."""
        session_info = self._get_session_info()
        if session_info is None:
            return
        session_dir, session_metadata = session_info
        cursor_generation_snapshot = self._transcript_cursor_generation

        non_system_messages = [m for m in messages if m.role != Role.system]

        # Empty conversations are only persisted on explicit opt-in (rewind to
        # the first message); otherwise an empty log would be unloadable.
        if not non_system_messages and not allow_empty:
            return

        # Serialization and fsync are too slow for the UI thread, so snapshot
        # here and hand off to a worker thread.
        messages_snapshot = list(messages)
        # M1: the save snapshot captures the retained archive descriptor WITH
        # the message epoch of the list being saved. Dormant until a later
        # subwave seals an archive: with no retained descriptor this captures
        # as ArchiveSaveSnapshot(None, <epoch>).
        archive_snapshot = ArchiveSaveSnapshot(
            descriptor=self._archive_descriptor,
            message_epoch=(
                messages.message_epoch if isinstance(messages, MessageList) else None
            ),
        )
        config_snapshot = config.model_dump(mode="json")
        # P5 (S1-C, protocol v2.3): the archived-prefix authority argument is
        # captured on-loop pre-lock, exactly as the on-loop validation call
        # captured it.
        archived_prefix_authority = self._archive_authority_for(messages)
        async with self._save_lock:
            # PI1/T (S1 save boundary): a transcript defect is never silently
            # persisted. The append/insert boundary already refuses to create
            # one; this backstop fails closed with the typed local diagnostic
            # so no unknown path can re-poison the log the load-time repair
            # just healed. P4a4: with an authoritative retained descriptor the
            # save recognizes its authorized archived prefix (M1 §4); without
            # one, any archive marker in the snapshot fails closed. The scan
            # itself runs in a worker thread while the save lock is held, so
            # the event loop stays free; the typed TranscriptIntegrityError
            # still raises from inside the lock BEFORE the staleness fence
            # below — exactly the on-loop call's observable failure, only on
            # another thread.
            await asyncio.to_thread(
                self._check_transcript_persistence,
                non_system_messages,
                archived_prefix=archived_prefix_authority,
            )
            # Fence the entire write (transcript and metadata), not only the
            # cursor update. Rewind/reset invalidates snapshots queued earlier.
            if self._transcript_cursor_generation != cursor_generation_snapshot:
                logger.debug("Discarding stale interaction save for %s", session_dir)
                return
            session_metadata.config = config_snapshot
            launch_config_generation = self._launch_config_generation
            metadata_snapshot = session_metadata.model_copy(deep=True)
            cursor_snapshot = self._transcript_cursor
            saves_since_verify = self._transcript_saves_since_verify
            self._last_save_archive_snapshot = archive_snapshot

            def persist() -> tuple[_TranscriptCursor | None, int] | None:
                return self._save_interaction_sync(
                    messages_snapshot,
                    stats,
                    tool_manager,
                    session_dir,
                    metadata_snapshot,
                    allow_empty,
                    self._launch_config_dirty,
                    cursor_snapshot,
                    saves_since_verify,
                )

            persistence = asyncio.create_task(asyncio.to_thread(persist))
            try:
                save_result = await asyncio.shield(persistence)
            except SessionDiskFullError as disk_full:
                # Fail soft: a full disk must not crash the agent loop. The
                # session stays authoritative in memory and the next save
                # retries persistence.
                self._note_disk_full(disk_full)
                return
            except ArchiveUnavailableError as unavailable:
                # P4a3 rev J: fail closed through the non-raising facade — a
                # FAILED save with the diagnostic recorded, never a legacy
                # rewrite while a descriptor is retained.
                self._record_save_failure(unavailable)
                return
            except asyncio.CancelledError as cancellation:
                # ``to_thread`` keeps running after its awaiter is cancelled. Keep
                # the save lock until that worker has finished so a subsequent
                # save cannot append to the same transcript concurrently. Further
                # cancellations must not interrupt that drain.
                while not persistence.done():
                    try:
                        await asyncio.shield(persistence)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not persistence.cancelled():
                    try:
                        save_result = persistence.result()
                    except SessionDiskFullError as disk_full:
                        self._note_disk_full(disk_full)
                    except ArchiveUnavailableError as unavailable:
                        self._record_save_failure(unavailable)
                    except Exception:
                        # The caller's cancellation remains the observable outcome,
                        # but retrieving the exception avoids abandoning the task.
                        pass
                    else:
                        if (
                            save_result is not None
                            and self._transcript_cursor_generation
                            == cursor_generation_snapshot
                        ):
                            (
                                self._transcript_cursor,
                                self._transcript_saves_since_verify,
                            ) = save_result
                raise cancellation
            if (
                save_result is not None
                and self._transcript_cursor_generation == cursor_generation_snapshot
            ):
                self._transcript_cursor, self._transcript_saves_since_verify = (
                    save_result
                )
            self._persisted = True
            # A successful save is a state change: report the next disk-full
            # occurrence immediately instead of staying rate-limited, and
            # clear any recorded fail-closed save diagnostic (P4a3).
            self._disk_full_last_logged_at = None
            self._last_save_diagnostic = None
            if self._launch_config_generation == launch_config_generation:
                self._launch_config_dirty = False

    @staticmethod
    def _check_transcript_persistence(
        non_system_messages: Sequence[LLMMessage],
        *,
        archived_prefix: ArchivedPrefixAuthority | None = None,
    ) -> None:
        """S1 save boundary: fail closed, never silently persist a defect.

        ``archived_prefix`` (P4a4, M1 §4) is the authorized archived prefix of
        the snapshot, minted by :meth:`_archive_authority_for` only for an
        authoritative descriptor/epoch pairing: the validators recognize the
        sealed head by that authority — never by a compaction marker alone —
        and any archive marker without matching authority fails closed here.

        Function-level import: a top-level import of the agent_loop validator
        would cycle (agent_loop -> _loop -> this module).
        """
        from chartreux.core.agent_loop._transcript_integrity import (
            TranscriptIntegrityError,
            find_persistence_violations,
        )

        violations = find_persistence_violations(
            non_system_messages, archived_prefix=archived_prefix
        )
        if violations:
            raise TranscriptIntegrityError(violations, context="session save boundary")

    def _note_disk_full(self, disk_full: SessionDiskFullError) -> None:
        """Record a disk-full save failure, logging it at a bounded rate."""
        now = utc_now()
        last = self._disk_full_last_logged_at
        if last is not None and now - last < DISK_FULL_LOG_INTERVAL:
            return
        self._disk_full_last_logged_at = now
        logger.warning(
            "Session persistence failed: the disk is full (%s). The session "
            "continues in memory; persistence retries on the next save.",
            disk_full.path,
        )

    def _plan_archive_seal(
        self,
        messages_snapshot: list[LLMMessage],
        message_epoch: int | None,
        trim_mode: Literal["payloads", "tombstones"],
        messages_path: Path,
    ) -> ArchivedPrefixV1:
        """Validate the closed prefix and build its descriptor (M1 §4 item 1).

        Every refusal raises ``ArchiveSealError`` and issues no receipt: stale
        retained authority, a missing message epoch, no eligible prefix, a
        malformed assistant/tool structure in the closed prefix, or a retained
        prefix whose disk bytes no longer verify. A RE-SEAL over an already
        sealed prefix extends the descriptor from the OLD PREFIX DISK BYTES
        plus the newly eligible records (see ``_extend_archive_descriptor``):
        after a real trim the RAM copies no longer carry the original
        payloads, so re-serializing them from RAM would desynchronize the
        descriptor from the authoritative disk bytes.
        """
        retained = self._archive_descriptor
        archive_snapshot = ArchiveSaveSnapshot(
            descriptor=retained, message_epoch=message_epoch
        )
        # P4a2 open question (a): a retained descriptor paired with a different
        # list epoch is stale authority. Ordinary saves do NOT hard-fail on it
        # (the live-list save stays correct); sealing refuses — no receipt.
        if retained is not None and not archive_snapshot.is_authoritative:
            raise ArchiveSealError(
                f"retained archive descriptor epoch {retained.message_epoch} does "
                f"not match the saved list epoch {message_epoch}; sealing refused"
            )
        if message_epoch is None:
            raise ArchiveSealError(
                "cannot seal a plain message sequence: no message epoch to pin "
                "the receipt to"
            )
        eligible = eligible_archive_prefix(messages_snapshot)
        if eligible is None:
            # P3 (S1-B) pre-seal arm: no compaction boundary exists yet, so
            # the post-seal prefix is not eligible. Release the settled
            # pre-compaction prefix's reasoning payloads instead (disk
            # unchanged) — or refuse when nothing is eligible pre-seal either.
            return self._plan_pre_seal_release(
                messages_snapshot, message_epoch, trim_mode, retained, messages_path
            )
        prefix_messages = list(eligible.messages)
        # M1 §4 required-contract item 1: strictly validate the closed prefix
        # BEFORE sealing — a malformed assistant/tool structure yields no receipt.
        from chartreux.core.agent_loop._transcript_integrity import validate_transcript

        # A re-seal validates a prefix whose head is already sealed: the
        # retained authority is the ONLY way the validators may recognize that
        # contiguous archived head (M1 §4 — never skip everything before a
        # marker). Without a retained descriptor this is None and any marker
        # fails closed here. The stale-retained check above has already
        # established the authoritative pairing, and ``messages_snapshot`` is a
        # plain sequence (no list epoch to re-derive). A retained PRE-SEAL
        # descriptor (boundary None) is widened to the whole closed prefix:
        # pre-seal slots carry no markers and the prefix has no compaction
        # boundary in its tail.
        archived_prefix_authority = (
            self._widened_pre_seal_authority(retained, len(prefix_messages))
            if retained is not None and retained.boundary_message_id is None
            else self._mint_archive_authority(retained)
            if retained is not None
            else None
        )
        violations = validate_transcript(
            prefix_messages,
            live_tail_exempt=False,
            archived_prefix=archived_prefix_authority,
        )
        if violations:
            raise ArchiveSealError(
                "the closed prefix failed the strict settled-transcript "
                "validation; no receipt",
                violations=tuple(violations),
            )
        non_system_messages = [
            message for message in messages_snapshot if message.role != Role.system
        ]
        # The sealing persist must not write a defect the save boundary would
        # refuse: the same S1 check over the whole snapshot, recognizing the
        # retained prefix's authority when the pairing is authoritative (P4a4).
        self._check_transcript_persistence(
            non_system_messages, archived_prefix=archived_prefix_authority
        )
        if retained is not None:
            # A re-seal over the same (authoritative) list: the first
            # ``retained.archived_count`` prefix slots are the payload-trimmed
            # replacements, so their authoritative bytes are the OLD PREFIX
            # DISK BYTES. Extend from those plus the newly eligible records
            # instead of re-serializing the trimmed RAM copies.
            return self._extend_archive_descriptor(
                retained,
                prefix_messages,
                message_epoch,
                trim_mode,
                boundary_message_id=eligible.boundary_message_id,
                messages_path=messages_path,
            )
        prefix_records = [SessionLogger._message_record(m) for m in prefix_messages]
        archived_bytes = SessionLogger._transcript_bytes(prefix_records)
        return ArchivedPrefixV1(
            session_id=self.session_id,
            message_epoch=message_epoch,
            archived_count=len(prefix_records),
            prefix_end_byte=len(archived_bytes),
            prefix_sha256=hashlib.sha256(archived_bytes).hexdigest(),
            boundary_message_id=eligible.boundary_message_id,
            trim_mode=trim_mode,
        )

    @staticmethod
    def _widened_pre_seal_authority(
        retained: ArchivedPrefixV1, covered_count: int
    ) -> Any:
        """The validation authority for a PRE-SEAL (boundary-None) archive.

        A pre-seal descriptor covers the whole non-system snapshot it was
        sealed over, and its released slots carry no archive markers, so the
        authority is widened to ``covered_count`` records: the validators then
        see no tail that would need a compaction boundary. Function-level
        import: a top-level import of the agent_loop validator would cycle.
        """
        from chartreux.core.agent_loop._transcript_integrity import (
            ArchivedPrefixAuthority,
        )

        return ArchivedPrefixAuthority(
            archived_count=covered_count,
            boundary_message_id=None,
            message_epoch=retained.message_epoch,
        )

    def _plan_pre_seal_release(
        self,
        messages_snapshot: list[LLMMessage],
        message_epoch: int,
        trim_mode: Literal["payloads", "tombstones"],
        retained: ArchivedPrefixV1 | None,
        messages_path: Path,
    ) -> ArchivedPrefixV1:
        """Plan a PRE-SEAL payload release (P3, S1-B): descriptor with
        ``boundary_message_id=None``.

        The pre-seal descriptor covers the WHOLE non-system snapshot: the
        released prefix and the retained rounds all live in the durably
        verified transcript, so the cover-all descriptor leaves no live tail
        that the admission validators would require to carry a compaction
        boundary. The strict settled validation runs over the eligible
        pre-seal prefix (settled by construction); the persistence check runs
        over the whole snapshot. A re-release over a retained pre-seal
        descriptor extends from the OLD PREFIX DISK BYTES plus the newly
        appended records (the released slots' RAM copies no longer carry the
        original payloads). Every refusal raises ``ArchiveSealError``.
        """
        pre = eligible_pre_seal_prefix(messages_snapshot)
        if pre is None:
            raise ArchiveSealError(
                "no eligible archive prefix: no compaction boundary with a "
                "complete settled round strictly before it, and no eligible "
                "pre-seal prefix (settled rounds before the newest "
                "retained user rounds carrying a reasoning payload)"
            )
        from chartreux.core.agent_loop._transcript_integrity import validate_transcript

        non_system_messages = [
            message for message in messages_snapshot if message.role != Role.system
        ]
        # The pre-seal descriptor covers the whole non-system snapshot; the
        # authority is widened accordingly (pre-seal slots carry no markers).
        archived_prefix_authority = (
            self._widened_pre_seal_authority(retained, len(non_system_messages))
            if retained is not None
            else None
        )
        violations = validate_transcript(
            list(pre.messages),
            live_tail_exempt=False,
            archived_prefix=archived_prefix_authority,
        )
        if violations:
            raise ArchiveSealError(
                "the pre-seal prefix failed the strict settled-transcript "
                "validation; no receipt",
                violations=tuple(violations),
            )
        self._check_transcript_persistence(
            non_system_messages, archived_prefix=archived_prefix_authority
        )
        if retained is not None:
            return self._extend_archive_descriptor(
                retained,
                non_system_messages,
                message_epoch,
                trim_mode,
                boundary_message_id=None,
                messages_path=messages_path,
            )
        prefix_records = [SessionLogger._message_record(m) for m in non_system_messages]
        archived_bytes = SessionLogger._transcript_bytes(prefix_records)
        return ArchivedPrefixV1(
            session_id=self.session_id,
            message_epoch=message_epoch,
            archived_count=len(prefix_records),
            prefix_end_byte=len(archived_bytes),
            prefix_sha256=hashlib.sha256(archived_bytes).hexdigest(),
            boundary_message_id=None,
            trim_mode=trim_mode,
        )

    def _extend_archive_descriptor(
        self,
        retained: ArchivedPrefixV1,
        prefix_messages: Sequence[LLMMessage],
        message_epoch: int,
        trim_mode: Literal["payloads", "tombstones"],
        *,
        boundary_message_id: str | None,
        messages_path: Path,
    ) -> ArchivedPrefixV1:
        """Extend an already sealed prefix's descriptor (P4d, P4b open item a).

        The OLD PREFIX DISK BYTES are authoritative, so the extension is built
        from exactly those bytes plus the newly eligible records' serialized
        form — never from the trimmed RAM copies. The retained prefix is
        re-verified against its descriptor in the SAME streaming read that
        feeds the new hash: changed archived bytes are a refusal (no receipt),
        so a re-seal can never re-anchor mutated history. A trim-mode change
        is refused the same way, because the retained copies' declared
        transformation would no longer describe the retained fields.
        """
        if trim_mode != retained.trim_mode:
            raise ArchiveSealError(
                f"cannot extend a {retained.trim_mode!r} archived prefix with a "
                f"{trim_mode!r} trim; no receipt"
            )
        if len(prefix_messages) < retained.archived_count:
            raise ArchiveSealError(
                f"the eligible prefix holds {len(prefix_messages)} records but the "
                f"retained archive already sealed {retained.archived_count}; no receipt"
            )
        hasher = hashlib.sha256()

        def hashed_chunks() -> Iterator[bytes]:
            # One read serves both proofs: iter_archived_records validates the
            # exact chunks the extension hash is computed over.
            for chunk in archived_prefix_chunks(
                messages_path, retained.prefix_end_byte
            ):
                hasher.update(chunk)
                yield chunk

        try:
            for _record in iter_archived_records(retained, hashed_chunks()):
                pass
        except ArchiveIntegrityError as exc:
            raise ArchiveSealError(
                "the retained archived prefix no longer verifies against its "
                f"descriptor, so it cannot be extended: {exc}"
            ) from exc
        except OSError as exc:
            raise ArchiveSealError(
                f"cannot read the retained archived prefix at {messages_path}: {exc}"
            ) from exc
        extension_records = [
            SessionLogger._message_record(message)
            for message in prefix_messages[retained.archived_count :]
        ]
        extension_bytes = SessionLogger._transcript_bytes(extension_records)
        hasher.update(extension_bytes)
        return ArchivedPrefixV1(
            session_id=self.session_id,
            message_epoch=message_epoch,
            archived_count=len(prefix_messages),
            prefix_end_byte=retained.prefix_end_byte + len(extension_bytes),
            prefix_sha256=hasher.hexdigest(),
            boundary_message_id=boundary_message_id,
            trim_mode=trim_mode,
        )

    def _seal_persist_sync(
        self,
        messages_snapshot: list[LLMMessage],
        stats: AgentStats,
        tool_manager: ToolManager,
        session_dir: Path,
        session_metadata: SessionMetadata,
        config_snapshot: dict[str, Any],
        launch_config_dirty: bool,
        cursor: _TranscriptCursor | None,
        saves_since_verify: int,
    ) -> tuple[_TranscriptCursor | None, int]:
        """The sealing durable persist (runs on a worker thread under the lock)."""
        session_metadata.config = config_snapshot
        metadata_snapshot = session_metadata.model_copy(deep=True)
        return self._save_interaction_sync(
            messages_snapshot,
            stats,
            tool_manager,
            session_dir,
            metadata_snapshot,
            False,
            launch_config_dirty,
            cursor,
            saves_since_verify,
        )

    async def seal_archive(
        self,
        messages: Sequence[LLMMessage],
        stats: AgentStats,
        config: ChartreuxConfigSchema,
        tool_manager: ToolManager,
        *,
        trim_mode: Literal["payloads", "tombstones"] = "payloads",
    ) -> ArchiveSealReceipt:
        """Seal the eligible closed prefix and return its durable receipt (M1).

        A returned save call is NOT authorization: this operation runs its own
        durable persist under ``_save_lock`` and issues the receipt only after
        that persist AND the on-disk verification of the sealed prefix bytes.
        Every refusal raises and issues NO receipt: ``ArchiveSealError`` for an
        ineligible or malformed prefix or stale retained authority,
        ``TranscriptIntegrityError`` from the save boundary, and
        ``SessionDiskFullError`` propagates from the persist itself — the
        fail-soft catch-and-continue of ``save_interaction`` is deliberately
        NOT applied here, which is exactly the gap the receipt closes.
        """
        session_info = self._get_session_info()
        if session_info is None:
            raise ArchiveSealError(
                "session logging is disabled; no archive can be sealed"
            )
        session_dir, session_metadata = session_info
        cursor_generation_snapshot = self._transcript_cursor_generation
        messages_snapshot = list(messages)
        message_epoch = (
            messages.message_epoch if isinstance(messages, MessageList) else None
        )
        retained_before = self._archive_descriptor
        # P4d: a re-seal reads the retained prefix's disk bytes, so the plan
        # runs off the event loop (the first seal stays RAM-only).
        descriptor = await asyncio.to_thread(
            self._plan_archive_seal,
            messages_snapshot,
            message_epoch,
            trim_mode,
            session_dir / MESSAGES_FILENAME,
        )
        config_snapshot = config.model_dump(mode="json")
        launch_config_generation = self._launch_config_generation
        async with self._save_lock:
            # A rewind/reset between the snapshot and the lock invalidates the
            # sealing snapshot itself: refuse, no receipt.
            if self._transcript_cursor_generation != cursor_generation_snapshot:
                raise ArchiveSealError(
                    "the transcript was rewound or reset while sealing; no receipt"
                )
            if self._archive_descriptor is not retained_before:
                raise ArchiveSealError(
                    "the retained archive authority changed while sealing; no receipt"
                )
            # P4a3: the sealing persist runs against the PRE-seal retained
            # authority — publish its snapshot so the worker's save path
            # verifies (and preserves) the old archived prefix on a re-seal.
            # A refused seal restores the previous snapshot: no receipt, no
            # visible save-snapshot change.
            previous_snapshot = self._last_save_archive_snapshot
            self._last_save_archive_snapshot = ArchiveSaveSnapshot(
                descriptor=retained_before, message_epoch=message_epoch
            )
            try:
                persistence = asyncio.create_task(
                    asyncio.to_thread(
                        self._seal_persist_sync,
                        messages_snapshot,
                        stats,
                        tool_manager,
                        session_dir,
                        session_metadata,
                        config_snapshot,
                        self._launch_config_dirty,
                        self._transcript_cursor,
                        self._transcript_saves_since_verify,
                    )
                )
                try:
                    save_result = await asyncio.shield(persistence)
                except asyncio.CancelledError as cancellation:
                    # The filesystem worker outlives cancellation: keep the save
                    # lock until it finishes so no later save interleaves. No
                    # receipt is issued either way.
                    while not persistence.done():
                        try:
                            await asyncio.shield(persistence)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    if not persistence.cancelled():
                        with contextlib.suppress(Exception):
                            persistence.result()
                    raise cancellation
                # SessionDiskFullError from the persist propagates: the sealing
                # path is NOT fail-soft, and no receipt is issued.
                if (
                    save_result is not None
                    and self._transcript_cursor_generation == cursor_generation_snapshot
                ):
                    self._transcript_cursor, self._transcript_saves_since_verify = (
                        save_result
                    )
                messages_path = session_dir / MESSAGES_FILENAME
                # The receipt is issued only after the durable persist AND the
                # on-disk verification of the sealed prefix bytes.
                await asyncio.to_thread(
                    verify_archived_prefix_on_disk, descriptor, messages_path
                )
            except BaseException:
                self._last_save_archive_snapshot = previous_snapshot
                raise
            receipt = ArchiveSealReceipt(
                descriptor=descriptor, sealed_at=utc_now().isoformat()
            )
            self.retain_archive_descriptor(descriptor)
            self._last_save_archive_snapshot = ArchiveSaveSnapshot(
                descriptor=descriptor, message_epoch=message_epoch
            )
            self._persisted = True
            self._disk_full_last_logged_at = None
            if self._launch_config_generation == launch_config_generation:
                self._launch_config_dirty = False
            return receipt

    def fence_and_publish_archive_trim(
        self,
        messages: MessageList,
        descriptor: ArchivedPrefixV1,
        replacements: Sequence[Any],
        *,
        save_generation: int,
        reset_generation: int,
    ) -> None:
        """The P4a4 generation fence + synchronous trim publication (M1 §4 step 6).

        Fences QUEUED STALE SAVES — the save generation moves, so every save
        that snapshotted before this call discards itself under the save lock
        before publishing anything (a cancelled in-flight save drains under the
        lock and its post-drain publication is discarded by the same fence) —
        then publishes the replacements and the DESCRIPTOR in ONE synchronous
        block: this method is a plain ``def``, so no ``await`` can intervene
        between the fence and the publish, and no ``reset()`` is called and no
        reset hook fires.

        ``replacements`` holds one replacement per sealed slot (TEST-DOUBLE
        objects at P4a4 — the P4b replacement model lands later): exactly
        ``descriptor.archived_count`` of them, applied same-slot by ABSOLUTE
        index over the non-system sequence, so no slot is inserted or deleted
        and system messages are preserved.

        Save/reset/cancellation generations are all handled: a descriptor
        planned against a departed ``save_generation`` or ``reset_generation``,
        one whose epoch predates the retained authority, or one that does not
        match the list's epoch is STALE — it raises fail-closed and never
        overwrites the current descriptor.
        """
        if not self.enabled:
            raise ValueError("session logging is disabled; no archive can be published")
        if descriptor.session_id != self.session_id:
            raise ValueError(
                f"archive descriptor session_id {descriptor.session_id!r} does not "
                f"match this logger's session {self.session_id!r}"
            )
        if reset_generation != self._archive_generation:
            raise ValueError(
                f"stale reset generation {reset_generation} != current "
                f"{self._archive_generation}: the logger was rebound since this "
                "descriptor was planned; a stale descriptor never overwrites"
            )
        if save_generation != self._transcript_cursor_generation:
            raise ValueError(
                f"stale save generation {save_generation} != current "
                f"{self._transcript_cursor_generation}: the transcript was "
                "invalidated since this descriptor was planned; a stale "
                "descriptor never overwrites"
            )
        current = self._archive_descriptor
        if current is not None and descriptor.message_epoch < current.message_epoch:
            raise ValueError(
                f"stale archive descriptor epoch {descriptor.message_epoch} would "
                f"overwrite the retained epoch {current.message_epoch}; a stale "
                "descriptor never overwrites the current one"
            )
        message_epoch = messages.message_epoch
        if descriptor.message_epoch != message_epoch:
            raise ValueError(
                f"archive descriptor epoch {descriptor.message_epoch} does not "
                f"match the list epoch {message_epoch}"
            )
        non_system_indexes = [
            index
            for index, message in enumerate(messages)
            if message.role is not Role.system
        ]
        if descriptor.archived_count > len(non_system_indexes):
            raise ValueError(
                f"descriptor archives {descriptor.archived_count} records but the "
                f"list holds only {len(non_system_indexes)} non-system messages"
            )
        if len(replacements) != descriptor.archived_count:
            raise ValueError(
                f"trim publication replaces exactly one slot per archived record: "
                f"got {len(replacements)} replacements for "
                f"{descriptor.archived_count} sealed slots"
            )
        # M1 §4 step 6: fence first, then publish — one synchronous block, no
        # intervening await, no reset(), no reset hooks. The list applies the
        # replacements same-slot over the non-system sequence and takes the
        # descriptor in its own lock scope (P4b's MessageList publication).
        self.invalidate_transcript_cursor()
        messages.publish_archive_trim(descriptor, replacements)
        self._archive_descriptor = descriptor
        self._last_save_archive_snapshot = ArchiveSaveSnapshot(
            descriptor=descriptor, message_epoch=message_epoch
        )

    def _rollback_post_turn_trim(
        self,
        descriptor_before: ArchivedPrefixV1 | None,
        snapshot_before: ArchiveSaveSnapshot | None,
    ) -> None:
        """Restore the pre-seal archive authority after a failed publication.

        Called only from the synchronous post-seal block of
        ``complete_post_turn_trim`` (no await between the seal's return and
        this call), so the restore is atomic: no save can interleave. The
        original (pre-seal) archived bytes and descriptor remain effective.
        """
        self._archive_descriptor = descriptor_before
        self._last_save_archive_snapshot = snapshot_before

    async def complete_post_turn_trim(  # noqa: PLR0911, PLR0912, PLR0915 - one explicit outcome per refusal/failure arm
        self,
        messages: MessageList,
        stats: AgentStats,
        config: ChartreuxConfigSchema,
        tool_manager: ToolManager,
        *,
        session_id: str,
        message_epoch: int,
        boundary_message_id: str | None,
        trim_mode: Literal["payloads", "tombstones"] = "payloads",
    ) -> PostTurnTrimResult:
        """Drain, fence, seal and publish a pending post-turn trim (T1b).

        This is the completion half of the guarded post-finalization entry:
        the compaction manager records the pending trim at the exclusive idle
        seam and calls this with the recorded key. The path, in order:

        1. validate the key (session, message epoch, boundary or pre-seal
           endpoint) against the live list — a stale key is refused at this
           attempt boundary;
        2. DRAIN queued/in-flight saves and take the exclusivity + fence
           snapshot under ``_save_lock``: every save queued before this point
           completes first, the attempt key enters the in-flight set (no
           concurrent seal/publication for the same key), and the save /
           reset generations are snapshotted for the publication fence;
        3. SEAL via ``seal_archive`` (its own durable persist runs under the
           save lock, after the drained saves, and issues the receipt only
           after on-disk verification);
        4. FENCE + PUBLISH in one synchronous call to
           ``fence_and_publish_archive_trim`` — the fence and the publication
           never split as active behavior.

        ``boundary_message_id=None`` selects the P3 pre-seal release: the
        attempt key is the progressive pre-seal key (stable eligible-prefix
        endpoint + session/epoch — never a bare None key), the seal plans a
        boundary-None descriptor covering the whole non-system snapshot, and
        the publication releases only the eligible pre-seal prefix's
        reasoning payloads (the retained rounds keep theirs; disk is
        unchanged). A real boundary coalesces: post-seal alone owns the
        release once one exists.

        Failure semantics, each reported as an explicit outcome: a refused or
        failed seal issues no receipt and retains the originals; a seal that
        succeeds but cannot publish rolls the retained descriptor and save
        snapshot back to their pre-seal values, so the original archived bytes
        and descriptor remain effective and nothing is partially published; a
        cancellation during the seal is drained inside ``seal_archive`` (no
        receipt, no publication) and re-raised. Every attempt returns a
        ``PostTurnTrimResult``; the caller decides what to swallow.
        """
        if not self.enabled:
            return PostTurnTrimResult("refused_disabled", "session logging is disabled")
        if session_id != self.session_id:
            return PostTurnTrimResult(
                "refused_session_mismatch",
                f"pending session {session_id!r} != logger session {self.session_id!r}",
            )
        if message_epoch != messages.message_epoch:
            return PostTurnTrimResult(
                "refused_epoch_mismatch",
                f"pending message epoch {message_epoch} != list epoch "
                f"{messages.message_epoch}: stale after a rebind, never sealed",
            )
        if boundary_message_id is not None:
            eligible = eligible_archive_prefix(list(messages))
            if eligible is None or eligible.boundary_message_id != boundary_message_id:
                return PostTurnTrimResult(
                    "refused_boundary_mismatch",
                    f"pending boundary {boundary_message_id!r} is not the live "
                    "eligible boundary",
                )
            endpoint: str | None = None
        else:
            pre = eligible_pre_seal_prefix(list(messages))
            if pre is None:
                return PostTurnTrimResult(
                    "refused_boundary_mismatch",
                    "no eligible pre-seal prefix: nothing settled before the "
                    "retained user rounds carries a reasoning payload",
                )
            endpoint = pre_seal_endpoint(pre)
        # The widened seam key — a real boundary keys as ("boundary", ...), a
        # pre-seal release as ("pre-seal", ...) from the stable endpoint.
        attempt_key = trim_attempt_key(
            session_id, message_epoch, boundary_message_id, endpoint
        )
        # Drain + exclusivity + fence snapshot under the save lock. Releasing
        # the lock before the seal is safe: seal_archive re-acquires it, so any
        # save queued in between still completes before the sealing persist,
        # and the publication fence discards anything snapshotted earlier.
        async with self._save_lock:
            if attempt_key in self._trim_published_boundaries:
                retained_now = self._archive_descriptor
                non_system_now = sum(
                    1 for message in messages if message.role is not Role.system
                )
                if (
                    retained_now is not None
                    and retained_now.boundary_message_id is None
                    and retained_now.archived_count < non_system_now
                ):
                    # A published pre-seal key with a coverage gap: the list
                    # grew since the release — proceed to extend the cover-all
                    # descriptor over the grown list (replacements idempotent).
                    pass
                else:
                    return PostTurnTrimResult(
                        "excluded_already_published",
                        f"release key {attempt_key!r} was already sealed and published",
                    )
            if attempt_key in self._trim_in_flight_boundaries:
                return PostTurnTrimResult(
                    "excluded_concurrent",
                    f"a seal/publication for release key {attempt_key!r} "
                    "is already in flight",
                )
            retained_before = self._archive_descriptor
            snapshot_before = self._last_save_archive_snapshot
            if (
                retained_before is not None
                and retained_before.message_epoch != message_epoch
            ):
                return PostTurnTrimResult(
                    "refused_stale_authority",
                    f"retained descriptor epoch {retained_before.message_epoch} "
                    f"!= list epoch {message_epoch}: stale authority, no receipt",
                )
            save_generation = self._transcript_cursor_generation
            reset_generation = self._archive_generation
            self._trim_in_flight_boundaries.add(attempt_key)
        try:
            try:
                receipt = await self.seal_archive(
                    messages, stats, config, tool_manager, trim_mode=trim_mode
                )
            except asyncio.CancelledError:
                # seal_archive drained its persist worker under the save lock
                # and issued no receipt: nothing is published. The retained
                # authority is untouched (no receipt path was reached).
                raise
            except Exception as exc:
                # No receipt: seal_archive already restored its save snapshot
                # and retained no descriptor — the originals stay effective.
                return PostTurnTrimResult("failed_seal", f"{type(exc).__name__}: {exc}")
            # The publish block is fully synchronous: no await can interleave
            # between the seal's receipt and the fence+publication.
            try:
                if boundary_message_id is not None:
                    eligible_now = eligible_archive_prefix(list(messages))
                    if (
                        eligible_now is None
                        or eligible_now.boundary_message_id != boundary_message_id
                        or len(eligible_now.messages)
                        != receipt.descriptor.archived_count
                    ):
                        raise ArchiveSealError(
                            "the list moved between sealing and publication; no publish"
                        )
                    replacements = [
                        archived_replacement(
                            message,
                            receipt.descriptor,
                            trim_mode=receipt.descriptor.trim_mode,
                        )
                        for message in eligible_now.messages
                    ]
                else:
                    pre_now = eligible_pre_seal_prefix(list(messages))
                    non_system_now = [
                        message
                        for message in messages
                        if message.role is not Role.system
                    ]
                    if (
                        pre_now is None
                        or pre_seal_endpoint(pre_now) != endpoint
                        or len(non_system_now) != receipt.descriptor.archived_count
                    ):
                        raise ArchiveSealError(
                            "the list moved between sealing and publication; no publish"
                        )
                    # Pre-seal publication through publish_archive_trim: the
                    # cover-all descriptor replaces every non-system slot — the
                    # eligible pre-seal prefix with payload-trimmed copies
                    # (unmarked: the model context still carries them), the
                    # retained rounds with their full selves.
                    released = len(pre_now.messages)
                    replacements = [
                        message.model_copy(update={"reasoning_payloads": None})
                        if index < released
                        else message
                        for index, message in enumerate(non_system_now)
                    ]
                self.fence_and_publish_archive_trim(
                    messages,
                    receipt.descriptor,
                    replacements,
                    save_generation=save_generation,
                    reset_generation=reset_generation,
                )
            except BaseException as exc:
                # The seal succeeded but the trim could not be published:
                # restore the pre-seal descriptor/snapshot so the original
                # archived bytes and descriptor remain effective. Nothing is
                # partially published (the fence either ran whole or not at
                # all — it is synchronous).
                self._rollback_post_turn_trim(retained_before, snapshot_before)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return PostTurnTrimResult(
                    "failed_publish", f"{type(exc).__name__}: {exc}", receipt=receipt
                )
            self._trim_published_boundaries.add(attempt_key)
            return PostTurnTrimResult("published", receipt=receipt)
        finally:
            self._trim_in_flight_boundaries.discard(attempt_key)

    async def clear_and_reseal_warm_resume(
        self,
        messages: MessageList,
        *,
        session_id: str,
        repair_messages: Sequence[LLMMessage] | None = None,
    ) -> WarmResumeTransactionResult:
        """The warm-resume clear-and-reseal PROOF/TRANSACTION half (T2a).

        Everything runs under ONE ``_save_lock`` hold, which DRAINS every
        queued/in-flight save: in-flight persist workers finish before the
        proof reads, and every save that snapshotted before this attempt
        discards itself under the lock once the fence moves the save
        generation — so the drain/generation fence stays effective THROUGH
        the synchronous rebind performed below in the same lock hold.

        Order:
        1. PROOF — the rebound list is proven FULLY HYDRATED from the target
           disk (archived bytes, session/path identity, COMPLETE loaded
           records) through the shared archive read interface. Archive
           authority is retained UNTIL this proof AND the fence both hold; a
           failed proof returns ``refused_proof`` with the on-disk bytes
           untouched (byte-identical), the retained descriptor in place, and
           nothing staged — there is NO descriptor re-pin fallback.
        2. STAGED REPAIR (only when ``repair_messages`` is given) — the
           original is preserved as a verified recovery artifact BEFORE the
           replacement, the live path is replaced with the exact old prefix
           bytes plus the repaired tail, and the staged result is re-proven
           against the complete repaired records: the exact archived prefix
           bytes are preserved (asserted, not assumed). A failure after
           replacement rolls back — the artifact's verified bytes are
           restored over the live path.
        3. AUTHORITY TRANSITION — synchronous in the same lock hold: the save
           generation moves (the fence), the reset generation bumps (the
           rebind), and both archive authorities are CLEARED —
           clear-and-reseal, never re-pinned — so the next seal mints a
           fresh receipt at the new epoch.
        """
        from chartreux.core.agent_loop._transcript_integrity import (
            TranscriptIntegrityError,
        )

        if not self.enabled:
            return WarmResumeTransactionResult(
                "refused_disabled", "session logging is disabled"
            )
        if session_id != self.session_id:
            return WarmResumeTransactionResult(
                "refused_session_mismatch",
                f"resume session {session_id!r} != logger session {self.session_id!r}",
            )
        session_info = self._get_session_info()
        if session_info is None:
            return WarmResumeTransactionResult(
                "refused_disabled", "no session is bound to this logger"
            )
        session_dir, _session_metadata = session_info
        messages_path = session_dir / MESSAGES_FILENAME
        async with self._save_lock:
            # Drain complete at this acquisition: in-flight workers finished;
            # saves queued earlier discard under the lock once the fence below
            # moves the save generation (same hold as the rebind).
            retained = self._archive_descriptor
            loaded_records = [
                SessionLogger._message_record(message, allow_archived=True)
                for message in messages
                if message.role is not Role.system
            ]
            try:
                proof = await asyncio.to_thread(
                    prove_rebound_hydration,
                    retained,
                    session_id=session_id,
                    messages_path=messages_path,
                    loaded_records=loaded_records,
                )
            except ArchiveIntegrityError as exc:
                # Fail closed: authority and originals stay. The proof is
                # read-only, so the on-disk bytes are byte-identical; nothing
                # was staged and no re-pin fallback exists.
                return WarmResumeTransactionResult(
                    "refused_proof", f"hydration proof failed closed: {exc}"
                )
            artifact_path: Path | None = None
            if repair_messages is not None:
                staging = asyncio.create_task(
                    asyncio.to_thread(
                        SessionLogger._stage_repaired_transcript,
                        messages_path,
                        repair_messages,
                        retained,
                        self.session_id,
                    )
                )
                try:
                    artifact_path = await asyncio.shield(staging)
                except asyncio.CancelledError:
                    # Drain the staging worker under the save lock, then prove
                    # the rollback/authority transition: a completed staging
                    # replaced the live path, so restore the artifact's
                    # verified original bytes; the retained authority stays
                    # (the transition below was never reached).
                    while not staging.done():
                        try:
                            await asyncio.shield(staging)
                        except asyncio.CancelledError:
                            continue
                        except BaseException:
                            break
                    if staging.exception() is None:
                        try:
                            SessionLogger._restore_staged_artifact(
                                messages_path, staging.result()
                            )
                        except OSError:
                            pass
                    raise
                except (OSError, ValueError, TranscriptIntegrityError) as exc:
                    # The staging refused: pre-replacement failures leave the
                    # original live path intact; post-replacement failures
                    # already rolled back inside _stage_repaired_transcript.
                    return WarmResumeTransactionResult(
                        "failed_repair_rolled_back",
                        f"staged repair failed; original restored/retained: {exc}",
                        proof=proof,
                    )
            # Synchronous authority transition in the same lock hold: fence,
            # rebind generation, clear-and-reseal (never re-pinned).
            self.invalidate_transcript_cursor()
            self._archive_generation += 1
            messages.clear_archive_descriptor()
            self._archive_descriptor = None
            self._last_save_archive_snapshot = ArchiveSaveSnapshot(
                descriptor=None, message_epoch=messages.message_epoch
            )
            return WarmResumeTransactionResult(
                "proved_and_transitioned", proof=proof, artifact_path=artifact_path
            )

    @staticmethod
    def _quarantine_corrupt_transcript(messages_path: Path) -> Path:
        """Preserve an unreadable transcript before replacing its live path."""
        timestamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        quarantine_path = messages_path.with_name(
            f"{messages_path.name}.corrupt-{timestamp}-{uuid4().hex}"
        )
        while quarantine_path.exists():
            quarantine_path = messages_path.with_name(
                f"{messages_path.name}.corrupt-{timestamp}-{uuid4().hex}"
            )
        try:
            os.replace(messages_path, quarantine_path)
            restrict_private_file(quarantine_path)
        except OSError as exc:
            if _is_enospc(exc):
                # The corruption itself must still be reported: the quarantine
                # failing on a full disk must not mask why it was attempted.
                logger.warning(
                    "Session transcript was corrupted at %s and could not be "
                    "quarantined because the disk is full; a new transcript "
                    "will be written.",
                    messages_path,
                )
                raise SessionDiskFullError(
                    messages_path, "quarantine corrupted session transcript"
                ) from exc
            raise RuntimeError(
                f"Failed to quarantine corrupted session transcript at {messages_path}: {exc}"
            ) from exc
        message = (
            f"Session transcript was corrupted and quarantined at {quarantine_path}; "
            "a new transcript will be written."
        )
        logger.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return quarantine_path

    @staticmethod
    def _stage_recovery_artifact(messages_path: Path, artifact_path: Path) -> None:
        """Copy the original transcript to its recovery artifact, streamed (P4b).

        The original is never held whole: the copy hashes and writes one
        bounded chunk at a time, and the read-back verification hashes the
        artifact off disk the same way. The digest is an exact identity
        check, so the read-back proves the same thing the whole-byte
        comparison did. The artifact is published atomically, so a failed
        staging write leaves only the original.
        """
        digest = hashlib.sha256()

        def chunks() -> Iterator[bytes]:
            for chunk in read_chunks(messages_path):
                digest.update(chunk)
                yield chunk

        durable_replace_stream(artifact_path, chunks())
        restrict_private_file(artifact_path)
        read_back = hashlib.sha256()
        for chunk in read_chunks(artifact_path):
            read_back.update(chunk)
        if read_back.digest() != digest.digest():
            raise OSError(f"recovery artifact read-back mismatch at {artifact_path}")

    @staticmethod
    def migrate_repaired_transcript(
        messages_path: Path, repaired_messages: Sequence[LLMMessage]
    ) -> Path:
        """Durably install an in-memory transcript repair (PI1/T S4).

        Mirrors ``_quarantine_corrupt_transcript``: the ORIGINAL transcript is
        preserved as a recovery artifact and the backup is verified by
        read-back BEFORE the live path is replaced, so a failed migration
        never destroys the original. Any failure raises the typed
        transcript-integrity diagnostic — the write-owning resume install
        fails closed, never a silent partial migration. The repair never
        deletes genuine structured tool results; the serialization matches a
        normal full save, so the transcript cursor stays consistent. Neither
        side is materialized (P4b): the artifact copies off disk in chunks and
        the repaired records stream into the replacement one line at a time.
        """
        from chartreux.core.agent_loop._transcript_integrity import (
            TranscriptIntegrityError,
        )

        non_system_messages = [
            message for message in repaired_messages if message.role != Role.system
        ]
        timestamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        artifact_path = messages_path.with_name(
            f"{messages_path.name}.recovered-{timestamp}-{uuid4().hex}"
        )
        while artifact_path.exists():
            artifact_path = messages_path.with_name(
                f"{messages_path.name}.recovered-{timestamp}-{uuid4().hex}"
            )
        try:
            SessionLogger._stage_recovery_artifact(messages_path, artifact_path)
            durable_replace_stream(
                messages_path,
                SessionLogger._transcript_chunks(
                    SessionLogger._message_record(message)
                    for message in non_system_messages
                ),
            )
        except OSError as exc:
            raise TranscriptIntegrityError.migration_failed(messages_path, exc) from exc
        message = (
            f"Session transcript repaired in place at {messages_path}; the "
            f"original transcript was preserved at {artifact_path}."
        )
        logger.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return artifact_path

    @staticmethod
    def _restore_staged_artifact(messages_path: Path, artifact_path: Path) -> None:
        """Restore a staged recovery artifact over the replaced live path (T2a).

        The artifact was verified by read-back at staging time, so streaming
        it back over ``messages_path`` reproduces the ORIGINAL bytes exactly —
        the proven rollback of a post-replacement failure or cancellation.
        """
        durable_replace_stream(messages_path, read_chunks(artifact_path))

    @staticmethod
    def _stage_repaired_transcript(
        messages_path: Path,
        repaired_messages: Sequence[LLMMessage],
        retained: ArchivedPrefixV1 | None,
        session_id: str,
    ) -> Path:
        """Stage a warm-resume repair, preserving the exact archived prefix bytes (T2a).

        The original transcript is preserved as a verified recovery artifact
        BEFORE the live path is replaced (same staging as
        ``migrate_repaired_transcript``). The replacement streams the OLD
        PREFIX BYTES verbatim — the first ``prefix_end_byte`` bytes of the
        current live file — followed by the repaired tail records one line at
        a time, so the archived prefix is byte-identical BY CONSTRUCTION
        (reading the live path mid-stream is safe: the rename publishes only
        after the stream completes). The staged result is then RE-PROVEN
        through the shared read interface against the complete repaired
        records: any deviation — a repair that would have altered the sealed
        prefix — fails closed and rolls back, restoring the artifact's
        verified original bytes over the live path. Returns the artifact
        path.
        """
        from chartreux.core.agent_loop._transcript_integrity import (
            TranscriptIntegrityError,
        )

        prefix_end_byte = retained.prefix_end_byte if retained is not None else 0
        archived_count = retained.archived_count if retained is not None else 0
        non_system_repaired = [
            message for message in repaired_messages if message.role != Role.system
        ]
        repaired_records = [
            SessionLogger._message_record(message) for message in non_system_repaired
        ]
        timestamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        artifact_path = messages_path.with_name(
            f"{messages_path.name}.recovered-{timestamp}-{uuid4().hex}"
        )
        while artifact_path.exists():
            artifact_path = messages_path.with_name(
                f"{messages_path.name}.recovered-{timestamp}-{uuid4().hex}"
            )
        try:
            SessionLogger._stage_recovery_artifact(messages_path, artifact_path)
            tail_records = repaired_records[archived_count:]

            def replacement_chunks() -> Iterator[bytes]:
                # The exact old prefix bytes stream through verbatim; only the
                # repaired tail is re-serialized.
                yield from archived_prefix_chunks(messages_path, prefix_end_byte)
                yield from SessionLogger._transcript_chunks(tail_records)

            durable_replace_stream(messages_path, replacement_chunks())
        except OSError as exc:
            raise TranscriptIntegrityError.migration_failed(messages_path, exc) from exc
        try:
            prove_rebound_hydration(
                retained,
                session_id=session_id,
                messages_path=messages_path,
                loaded_records=repaired_records,
            )
        except (ArchiveIntegrityError, OSError) as exc:
            # Post-replacement failure: roll back to the staged original.
            try:
                SessionLogger._restore_staged_artifact(messages_path, artifact_path)
            except OSError as rollback_exc:
                raise TranscriptIntegrityError.migration_failed(
                    messages_path, rollback_exc
                ) from rollback_exc
            raise TranscriptIntegrityError.migration_failed(messages_path, exc) from exc
        message = (
            f"Session transcript repair staged at {messages_path} preserving the "
            f"archived prefix; the original transcript was preserved at {artifact_path}."
        )
        logger.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return artifact_path

    @staticmethod
    def _read_persisted_messages(
        messages_path: Path,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Bulk reader: every persisted record at once, plus validity.

        The full-verify no longer uses this (P4a streams the comparison via
        ``_stream_persisted_records``); it remains the shared bulk entry for
        callers that genuinely want the whole transcript.
        """
        persisted_messages: list[dict[str, Any]] = []
        try:
            persisted_messages.extend(
                SessionLogger._stream_persisted_records(messages_path)
            )
        except _PersistedTranscriptError:
            return [], False
        return persisted_messages, True

    @staticmethod
    def _stream_persisted_records(messages_path: Path) -> Iterator[dict[str, Any]]:
        """Stream persisted transcript records one at a time (P4a).

        The full-verify comparison must not materialize the transcript, so
        each line is parsed and yielded individually; the comparison holds
        one persisted record and one live record at a time. The raw bytes
        stream through a text wrapper using the codec the BOM selects, so a
        UTF-16/UTF-32 transcript decodes as it did through the bulk reader's
        ``read_safe``; the wrapper's incremental decoder never splits a
        multi-byte sequence across lines, and universal-newline iteration
        keeps the documented line-splitting behavior. A line that fails to
        parse — or parses to a non-record — raises
        ``_PersistedTranscriptError``, the per-line analogue of the bulk
        reader's ``(list, False)`` result. A missing file yields nothing.
        """
        try:
            if not messages_path.exists():
                return
            with messages_path.open("rb") as raw:
                encoding = _transcript_encoding(raw.peek(4)[:4])
                with io.TextIOWrapper(raw, encoding=encoding) as lines:
                    for line in lines:
                        message = json.loads(line)
                        if not isinstance(message, dict):
                            raise _PersistedTranscriptError(
                                f"{messages_path}: transcript line is not a JSON object"
                            )
                        yield message
        except _PersistedTranscriptError:
            raise
        except (OSError, ValueError) as exc:
            raise _PersistedTranscriptError(
                f"{messages_path}: unreadable transcript: {exc}"
            ) from exc

    @staticmethod
    def _stream_persisted_records_from(
        messages_path: Path, offset: int
    ) -> Iterator[dict[str, Any]]:
        """Stream persisted transcript records after ``offset`` bytes (P4a3).

        Same contract as ``_stream_persisted_records`` — BOM-selected codec,
        one parsed record at a time, ``_PersistedTranscriptError`` on a bad
        line, a missing file yields nothing — but the stream starts at a
        byte offset, so the archive-aware verify compares the live tail
        without re-reading the archived prefix. The offset is a record
        boundary (``prefix_end_byte`` sits immediately after a newline), so
        the incremental decoder never starts inside a multi-byte sequence.
        """
        try:
            if not messages_path.exists():
                return
            with messages_path.open("rb") as raw:
                encoding = _transcript_encoding(raw.peek(4)[:4])
                raw.seek(offset)
                with io.TextIOWrapper(raw, encoding=encoding) as lines:
                    for line in lines:
                        message = json.loads(line)
                        if not isinstance(message, dict):
                            raise _PersistedTranscriptError(
                                f"{messages_path}: transcript line is not a JSON object"
                            )
                        yield message
        except _PersistedTranscriptError:
            raise
        except (OSError, ValueError) as exc:
            raise _PersistedTranscriptError(
                f"{messages_path}: unreadable transcript: {exc}"
            ) from exc

    @staticmethod
    def _transcript_line_digest(message_data: dict[str, Any]) -> str:
        line = json.dumps(message_data, ensure_ascii=False)
        return hashlib.sha256(line.encode("utf-8")).hexdigest()

    @classmethod
    def _cursor_for_messages(
        cls, live_count: int, last_live_message: LLMMessage | None, messages_path: Path
    ) -> _TranscriptCursor:
        """The verified-boundary cursor for the live transcript."""
        file_size = messages_path.stat().st_size if messages_path.exists() else 0
        return _TranscriptCursor(
            count=live_count,
            boundary_digest=(
                cls._transcript_line_digest(cls._message_record(last_live_message))
                if last_live_message is not None
                else None
            ),
            file_size=file_size,
            metadata_published=True,
        )

    @staticmethod
    def _tail_records(
        messages: Sequence[LLMMessage], verified_prefix: int
    ) -> Iterator[dict[str, Any]]:
        """The live records past the verified boundary, streamed (P4a).

        The verified prefix is skipped by position, without serializing it
        again; the tail records are dumped one at a time.
        """
        live_index = 0
        for message in messages:
            if message.role == Role.system:
                continue
            if live_index >= verified_prefix:
                yield SessionLogger._message_record(message)
            live_index += 1

    def _interaction_metadata_dump(
        self,
        messages: list[LLMMessage],
        non_system_messages: list[LLMMessage],
        stats: AgentStats,
        tool_manager: ToolManager,
        session_metadata: SessionMetadata,
    ) -> dict[str, Any]:
        """Serialize the prompt/tool blobs and assemble the metadata payload."""
        tools_available = [
            {"type": "function", "function": fn.model_dump()}
            for fn in tool_manager.available_tool_specs()
        ]
        system_prompt = (
            messages[0].model_dump()
            if messages and messages[0].role == Role.system
            else None
        )
        return self._assemble_metadata_dump(
            non_system_messages, stats, session_metadata, system_prompt, tools_available
        )

    def _assemble_metadata_dump(
        self,
        non_system_messages: list[LLMMessage],
        stats: AgentStats,
        session_metadata: SessionMetadata,
        system_prompt: dict[str, Any] | None,
        tools_available: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the metadata payload around already-serialized blobs."""
        last_message_fingerprint = (
            self._message_fingerprint(non_system_messages[-1])
            if non_system_messages
            else None
        )
        metadata_dump = {
            **session_metadata.model_dump(exclude={"launch_config"}),
            "end_time": utc_now().isoformat(),
            "stats": stats.model_dump(),
            "total_messages": len(non_system_messages),
            "last_message_fingerprint": last_message_fingerprint,
            "tools_available": tools_available,
            "system_prompt": system_prompt,
        }
        if session_metadata.launch_config is not None:
            metadata_dump["launch_config"] = session_metadata.launch_config.model_dump(
                mode="json"
            )
        return metadata_dump

    @staticmethod
    def _prompt_content_hash(
        system_message: LLMMessage | None, tool_specs: Sequence[AvailableFunction]
    ) -> bytes:
        """Content hash of the inputs the prompt/tool serialization renders.

        Each input is rendered by pydantic's JSON encoder and folded in by its
        Python content hash: a rendered blob is unchanged exactly when its hash
        is, so a prompt edit, an added or dropped tool, a changed description,
        and a changed parameter schema each change the digest. Rendering through
        pydantic-core and hashing the rendered text in place (rather than
        encoding a second bytes copy of it) is the cheapest faithful render
        available — noticing that the content changed costs one pass over it.
        The digest is compared only within this process, so per-process hash
        seeding is harmless.
        """
        digest = hashlib.blake2b(digest_size=16)
        if system_message is not None:
            digest.update(b"+prompt")
            digest.update(str(hash(system_message.model_dump_json())).encode())
        for spec in tool_specs:
            digest.update(b"\x00tool")
            digest.update(str(hash(spec.model_dump_json())).encode())
        return digest.digest()

    def _gated_metadata_dump(
        self,
        messages: list[LLMMessage],
        non_system_messages: list[LLMMessage],
        stats: AgentStats,
        tool_manager: ToolManager,
        session_metadata: SessionMetadata,
    ) -> dict[str, Any]:
        """Serialize the metadata payload, gated by the prompt/tool content hash.

        The system prompt and the tool roster change rarely, so their
        serializations are reused while the hash of their inputs is unchanged;
        the per-save fields (end time, stats, message count and fingerprint) are
        always rebuilt. The hash lives in logger memory only (P2's hash storage
        pin), so a restarted logger re-serializes once and then rebuilds it.
        """
        system_message = (
            messages[0] if messages and messages[0].role == Role.system else None
        )
        content_hash = SessionLogger._prompt_content_hash(
            system_message, tool_manager.available_tool_specs()
        )
        cached = self._prompt_serialization
        if cached is not None and cached.content_hash == content_hash:
            system_prompt = (
                dict(cached.system_prompt) if cached.system_prompt is not None else None
            )
            return self._assemble_metadata_dump(
                non_system_messages,
                stats,
                session_metadata,
                system_prompt,
                [dict(item) for item in cached.tools_available],
            )
        metadata_dump = self._interaction_metadata_dump(
            messages, non_system_messages, stats, tool_manager, session_metadata
        )
        self._prompt_serialization = _PromptSerialization(
            content_hash=content_hash,
            system_prompt=metadata_dump["system_prompt"],
            tools_available=list(metadata_dump["tools_available"]),
        )
        return metadata_dump

    def _save_interaction_sync(
        self,
        messages: list[LLMMessage],
        stats: AgentStats,
        tool_manager: ToolManager,
        session_dir: Path,
        session_metadata: SessionMetadata,
        allow_empty: bool,
        launch_config_dirty: bool,
        cursor: _TranscriptCursor | None,
        saves_since_verify: int,
    ) -> tuple[_TranscriptCursor | None, int]:
        non_system_messages = [
            message for message in messages if message.role != Role.system
        ]
        if not non_system_messages and not allow_empty:
            return cursor, saves_since_verify

        if (
            cursor is None
            or allow_empty
            or saves_since_verify + 1 >= TRANSCRIPT_VERIFY_INTERVAL
        ):
            verified_cursor = self._save_full_verify(
                messages,
                stats,
                tool_manager,
                session_dir,
                session_metadata,
                launch_config_dirty,
            )
            return verified_cursor, 0

        messages_path = session_dir / MESSAGES_FILENAME
        try:
            file_size = messages_path.stat().st_size
        except OSError:
            file_size = -1
        if (
            file_size != cursor.file_size
            or cursor.count < 0
            or len(non_system_messages) < cursor.count
            or (cursor.count == 0 and cursor.boundary_digest is not None)
        ):
            verified_cursor = self._save_full_verify(
                messages,
                stats,
                tool_manager,
                session_dir,
                session_metadata,
                launch_config_dirty,
            )
            return verified_cursor, 0

        if cursor.count:
            boundary_data = SessionLogger._message_record(
                non_system_messages[cursor.count - 1]
            )
            if (
                cursor.boundary_digest is None
                or self._transcript_line_digest(boundary_data) != cursor.boundary_digest
            ):
                verified_cursor = self._save_full_verify(
                    messages,
                    stats,
                    tool_manager,
                    session_dir,
                    session_metadata,
                    launch_config_dirty,
                )
                return verified_cursor, 0

        tail = [
            SessionLogger._message_record(message)
            for message in non_system_messages[cursor.count :]
        ]
        updated_cursor = cursor
        try:
            if tail:
                file_size = SessionLogger._persist_messages_sync(tail, session_dir)
                updated_cursor = _TranscriptCursor(
                    count=len(non_system_messages),
                    boundary_digest=self._transcript_line_digest(tail[-1]),
                    file_size=file_size,
                    metadata_published=False,
                )

            if tail or not (cursor.metadata_published and not launch_config_dirty):
                metadata_dump = self._gated_metadata_dump(
                    messages, non_system_messages, stats, tool_manager, session_metadata
                )
                SessionLogger._persist_metadata_sync(metadata_dump, session_dir)
                updated_cursor = _TranscriptCursor(
                    count=updated_cursor.count,
                    boundary_digest=updated_cursor.boundary_digest,
                    file_size=updated_cursor.file_size,
                    metadata_published=True,
                )
            else:
                return updated_cursor, saves_since_verify + 1
        except SessionDiskFullError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to save session to {session_dir}: {e}") from e
        finally:
            self.maybe_cleanup_tmp_files()
        return updated_cursor, saves_since_verify + 1

    @staticmethod
    def _transcript_chunks(messages: Iterable[dict[str, Any]]) -> Iterator[bytes]:
        """Yield the transcript payload one encoded record line at a time (P4b).

        The streamed rewrite writes each line as it is produced, so the
        payload is never resident whole.
        """
        for message in messages:
            yield (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")

    @staticmethod
    def _transcript_bytes(messages: Iterable[dict[str, Any]]) -> bytes:
        """Serialize records into the transcript payload, one line at a time.

        Each record is encoded and folded into the payload buffer
        immediately, so the intermediate full-text copy the join-then-encode
        form held is gone (P4a); the accepted iterable may be a generator,
        keeping the caller's record construction streaming too. When the
        records are re-iterable the encoded size is measured first so the
        buffer is allocated exactly once instead of growing with
        over-allocation slack. The rewrite path streams instead (P4b).
        """
        if isinstance(messages, Sequence):
            total = sum(
                len(line) for line in SessionLogger._transcript_chunks(messages)
            )
            return bytes(
                _SizedByteStream(
                    itertools.chain.from_iterable(
                        SessionLogger._transcript_chunks(messages)
                    ),
                    total,
                )
            )
        return bytes(
            itertools.chain.from_iterable(SessionLogger._transcript_chunks(messages))
        )

    def _verify_persisted_boundary(
        self, messages: list[LLMMessage], messages_path: Path
    ) -> tuple[int, bool, bool, LLMMessage | None]:
        """Stream-compare the persisted transcript against the live records.

        Returns the verified prefix count, whether the persisted transcript
        parsed cleanly, whether its records are a prefix of the live
        records, and the last verified live message. Neither side is
        materialized: the persisted records stream off disk and the live
        records dump one at a time, so the residency is one record pair
        (P4a).

        Both phases — the comparison and the drain that follows it — release
        the GIL every ``PERSISTED_VERIFY_YIELD_INTERVAL`` persisted records
        (P6): the verify occupies the save worker thread while the event
        loop keeps rendering the stream, and an uninterrupted 20k-record
        pass starves that render.
        """
        persisted_count = 0
        transcript_valid = True
        boundary_unchanged = True
        last_live_message: LLMMessage | None = None
        try:
            persisted_records = SessionLogger._stream_persisted_records(messages_path)
            comparison_count = 0
            for live_message in (
                message for message in messages if message.role != Role.system
            ):
                persisted_record = next(persisted_records, None)
                if persisted_record is None:
                    break
                # The cadence counts records consumed, so it is identical in
                # both phases and independent of the match outcome.
                comparison_count += 1
                if comparison_count % PERSISTED_VERIFY_YIELD_INTERVAL == 0:
                    time.sleep(0)
                if persisted_record != SessionLogger._message_record(
                    live_message, allow_archived=True
                ):
                    boundary_unchanged = False
                    break
                persisted_count += 1
                last_live_message = live_message
            else:
                if next(persisted_records, None) is not None:
                    # The persisted transcript is longer than the live one:
                    # the boundary is not a prefix, so the rewrite replaces it.
                    boundary_unchanged = False
            # Drain the remainder: the comparison stops early, but a corrupt
            # line anywhere in the transcript must still quarantine it, so
            # every line is parsed exactly once across compare and drain.
            drained_count = 0
            for _ in persisted_records:
                drained_count += 1
                if drained_count % PERSISTED_VERIFY_YIELD_INTERVAL == 0:
                    time.sleep(0)
        except _PersistedTranscriptError:
            transcript_valid = False
            boundary_unchanged = False
        return persisted_count, transcript_valid, boundary_unchanged, last_live_message

    @staticmethod
    def _retained_prefix_record_matches(
        live_record: dict[str, Any],
        archived_record: dict[str, Any],
        trim_mode: Literal["payloads", "tombstones"],
    ) -> bool:
        """Whether a retained prefix record is the declared trim of the archived one.

        M1's full verify compares the retained prefix FIELDS against the
        declared trim transformation, never against the raw archived record:
        an untrimmed retained message equals the archived record; ``payloads``
        may only drop ``reasoning_payloads``; ``tombstones`` may only keep
        role/message_id/injected (the private archive reference never
        serializes). Every retained field must equal the archived value.
        """
        if live_record == archived_record:
            return True
        if trim_mode == "payloads":
            dropped = set(archived_record) - set(live_record)
            return dropped <= {"reasoning_payloads"} and all(
                archived_record[key] == value for key, value in live_record.items()
            )
        if trim_mode == "tombstones":
            return (
                set(live_record) <= _TOMBSTONE_RETAINED_KEYS
                and live_record.get("role") == archived_record.get("role")
                and all(
                    archived_record.get(key) == value
                    for key, value in live_record.items()
                    if key != "role"
                )
            )
        return False

    def _verify_archived_prefix(
        self,
        live_prefix: Sequence[LLMMessage],
        messages_path: Path,
        descriptor: ArchivedPrefixV1,
    ) -> None:
        """Phase A: stream-hash the archived bytes and check the retained fields.

        Streams the archived byte prefix through the shared read interface
        (``iter_archived_records`` over ``archived_prefix_chunks``) — byte
        count, sha256, record count, and per-line structure all validated —
        while comparing each retained prefix record against the declared trim
        transformation. Byte-level integrity is proven FIRST (exhausting the
        stream runs the descriptor's checks); only a hash-clean archive has
        its retained fields compared, so a retained mismatch is raised after
        the drain. Raises ``ArchiveUnavailableError`` fail-closed:
        missing/truncated/changed archived bytes are unrecoverable in place;
        a retained-field mismatch is an authority-state failure.
        """
        # A retained mismatch is recorded and raised after the drain, so the
        # byte/hash validation always classifies a damaged prefix first.
        prefix_mismatch: int | None = None
        try:
            archived_records = iter_archived_records(
                descriptor,
                archived_prefix_chunks(messages_path, descriptor.prefix_end_byte),
            )
            live_iter = iter(live_prefix)
            index = 0
            for archived_record in archived_records:
                index += 1
                if index % PERSISTED_VERIFY_YIELD_INTERVAL == 0:
                    time.sleep(0)
                live_message = next(live_iter, None)
                if live_message is None:
                    # The live list is shorter than the archived prefix: the
                    # remaining archived records drain (and validate) below.
                    break
                if prefix_mismatch is None and not (
                    SessionLogger._retained_prefix_record_matches(
                        SessionLogger._message_record(
                            live_message, allow_archived=True
                        ),
                        archived_record,
                        descriptor.trim_mode,
                    )
                ):
                    prefix_mismatch = index - 1
            drained = 0
            for _record in archived_records:
                drained += 1
                if drained % PERSISTED_VERIFY_YIELD_INTERVAL == 0:
                    time.sleep(0)
        except ArchiveIntegrityError as exc:
            raise ArchiveUnavailableError(
                messages_path, str(exc), recoverable_by_restart=False
            ) from exc
        except OSError as exc:
            raise ArchiveUnavailableError(
                messages_path,
                f"cannot read the archived prefix: {exc}",
                recoverable_by_restart=False,
            ) from exc
        if prefix_mismatch is not None:
            raise ArchiveUnavailableError(
                messages_path,
                f"retained prefix record {prefix_mismatch} does not match the "
                f"declared {descriptor.trim_mode!r} trim of its archived record",
                recoverable_by_restart=True,
            )

    def _verify_archived_live_tail(
        self,
        live_tail: Sequence[LLMMessage],
        messages_path: Path,
        descriptor: ArchivedPrefixV1,
    ) -> tuple[int, bool]:
        """Phase B: compare the disk records after the prefix exactly against
        the live tail (same pair-at-a-time contract and P6 yield cadence as
        the legacy verify). Returns the matched tail count and whether the
        tail boundary is unchanged. A corrupt tail line fails closed as an
        authority-state ``ArchiveUnavailableError`` — never quarantined.
        """
        tail_persisted = 0
        tail_boundary_unchanged = True
        try:
            tail_records = SessionLogger._stream_persisted_records_from(
                messages_path, descriptor.prefix_end_byte
            )
            comparison = 0
            for live_message in live_tail:
                persisted_record = next(tail_records, None)
                if persisted_record is None:
                    break
                comparison += 1
                if comparison % PERSISTED_VERIFY_YIELD_INTERVAL == 0:
                    time.sleep(0)
                if persisted_record != SessionLogger._message_record(
                    live_message, allow_archived=True
                ):
                    tail_boundary_unchanged = False
                    break
                tail_persisted += 1
            else:
                if next(tail_records, None) is not None:
                    # The persisted tail is longer than the live one.
                    tail_boundary_unchanged = False
            drained = 0
            for _record in tail_records:
                drained += 1
                if drained % PERSISTED_VERIFY_YIELD_INTERVAL == 0:
                    time.sleep(0)
        except _PersistedTranscriptError as exc:
            raise ArchiveUnavailableError(
                messages_path,
                f"the transcript tail is unreadable with a retained archive "
                f"descriptor: {exc}",
                recoverable_by_restart=True,
            ) from exc
        return tail_persisted, tail_boundary_unchanged

    def _verify_archived_prefix_and_tail(
        self,
        non_system_messages: Sequence[LLMMessage],
        messages_path: Path,
        descriptor: ArchivedPrefixV1,
    ) -> tuple[int, bool, LLMMessage | None]:
        """Archive-aware streaming verify (M1 full-verify row, P4a3).

        Phase A stream-hashes the archived bytes and compares the retained
        prefix fields against the declared trim transformation; phase B
        compares the remaining disk records exactly against the live tail.
        Returns the verified record count, whether the boundary is unchanged,
        and the last verified live message. Neither side is materialized.

        Raises ``ArchiveUnavailableError`` fail-closed — never a partial
        result, never a quarantine: missing/truncated/changed archived bytes
        are unrecoverable in place; a retained-prefix or live-tail defect is
        an authority-state failure that recovers by session restart.
        """
        live_count = len(non_system_messages)
        self._verify_archived_prefix(
            non_system_messages[: descriptor.archived_count], messages_path, descriptor
        )
        tail_persisted, tail_boundary_unchanged = self._verify_archived_live_tail(
            non_system_messages[descriptor.archived_count :], messages_path, descriptor
        )
        persisted_count = descriptor.archived_count + tail_persisted
        boundary_unchanged = (
            live_count >= descriptor.archived_count and tail_boundary_unchanged
        )
        verified = min(persisted_count, live_count)
        last_live_message = non_system_messages[verified - 1] if verified else None
        return persisted_count, boundary_unchanged, last_live_message

    def _save_full_verify(
        self,
        messages: list[LLMMessage],
        stats: AgentStats,
        tool_manager: ToolManager,
        session_dir: Path,
        session_metadata: SessionMetadata,
        launch_config_dirty: bool,
    ) -> _TranscriptCursor:
        # If the session directory does not exist, create it owner-only. The
        # mode applies to the leaf directory; mkdir(parents=True) does not
        # apply it to every ancestor. Existing directories retain their mode.
        try:
            session_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        except OSError as e:
            if _is_enospc(e):
                raise SessionDiskFullError(
                    session_dir, "create session directory"
                ) from e
            raise RuntimeError(
                f"Failed to create session directory at {session_dir}: {type(e).__name__}: {e}"
            ) from e

        # messages.jsonl is authoritative for the persisted boundary. Metadata
        # is published separately and may lag after an interrupted save.
        messages_path = session_dir / MESSAGES_FILENAME

        # P4a3: with an AUTHORITATIVE retained archive descriptor (the
        # descriptor's epoch matches the saved list's) the archived byte
        # prefix is authority — the verify goes archive-aware and never
        # re-enters the legacy rewrite path. A retained but non-authoritative
        # descriptor keeps the ordinary save on the legacy path (P4a2: only
        # sealing refuses that pairing).
        archive_snapshot = self._active_archive_snapshot()
        if (
            archive_snapshot.descriptor is not None
            and archive_snapshot.is_authoritative
        ):
            return self._save_full_verify_archived(
                messages,
                stats,
                tool_manager,
                session_dir,
                session_metadata,
                launch_config_dirty,
                archive_snapshot.descriptor,
            )

        # P4a: the verification compares the persisted transcript against the
        # live records one PAIR at a time — neither side is materialized. The
        # persisted records stream off disk; the live records dump lazily.
        # Each pair compares as dicts, the same per-record equality the bulk
        # reader's list comparison applied, so numeric-type drift (1 vs 1.0,
        # True vs 1) and key-order differences still compare equal, and a
        # NaN record still fails its own equality, exactly as before. The
        # full-rewrite repair below streams too (P4b): the live records are
        # dumped and encoded one line at a time into the replacement.
        persisted_count, transcript_valid, boundary_unchanged, last_live_message = (
            self._verify_persisted_boundary(messages, messages_path)
        )
        live_count = sum(1 for message in messages if message.role != Role.system)

        if not transcript_valid and messages_path.exists():
            self._quarantine_corrupt_transcript(messages_path)
            persisted_count = 0
            boundary_unchanged = False

        # Preserve the existing no-op behavior only when the metadata cache
        # confirms the transcript boundary. If publication failed previously,
        # the transcript still prevents duplicate appends and this retry repairs
        # the stale metadata.
        metadata_path = session_dir / METADATA_FILENAME
        if (
            not launch_config_dirty
            and boundary_unchanged
            and live_count == persisted_count
        ):
            try:
                cached_metadata = json.loads(read_safe(metadata_path).text)
                if cached_metadata.get(
                    "total_messages"
                ) == persisted_count and cached_metadata.get(
                    "last_message_fingerprint"
                ) == (
                    self._message_fingerprint(last_live_message)
                    if last_live_message is not None
                    else None
                ):
                    return self._cursor_for_messages(
                        live_count, last_live_message, messages_path
                    )
            except (OSError, json.JSONDecodeError):
                pass

        try:
            if (
                live_count > persisted_count
                and boundary_unchanged
                and messages_path.exists()
            ):
                SessionLogger._persist_messages_sync(
                    SessionLogger._tail_records(messages, persisted_count), session_dir
                )
            elif (
                live_count != persisted_count
                or not boundary_unchanged
                or not messages_path.exists()
            ):
                SessionLogger._overwrite_messages_sync(
                    (
                        SessionLogger._message_record(message)
                        for message in messages
                        if message.role != Role.system
                    ),
                    session_dir,
                )

            non_system_messages = [
                message for message in messages if message.role != Role.system
            ]
            metadata_dump = self._gated_metadata_dump(
                messages, non_system_messages, stats, tool_manager, session_metadata
            )
            SessionLogger._persist_metadata_sync(metadata_dump, session_dir)
        except SessionDiskFullError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to save session to {session_dir}: {e}") from e
        finally:
            self.maybe_cleanup_tmp_files()
        return self._cursor_for_messages(
            live_count,
            non_system_messages[-1] if non_system_messages else None,
            messages_path,
        )

    def _save_full_verify_archived(
        self,
        messages: list[LLMMessage],
        stats: AgentStats,
        tool_manager: ToolManager,
        session_dir: Path,
        session_metadata: SessionMetadata,
        launch_config_dirty: bool,
        descriptor: ArchivedPrefixV1,
    ) -> _TranscriptCursor:
        """The archive-aware full verify (M1 save-behavior matrix, P4a3).

        Stream-hashes the archived bytes, compares the retained prefix fields
        against the declared trim transformation, and compares the remaining
        disk records exactly against the live tail. A live-tail mismatch (or
        intentional truncation) publishes a temporary replacement carrying
        the ORIGINAL archived bytes plus the authoritative live suffix,
        atomically, and the published file is BYTE-CHECKED to contain the
        original archived prefix. Missing, truncated, or changed archived
        bytes raise ``ArchiveUnavailableError`` fail-closed: the
        corrupt-transcript quarantine is never reused for an
        archive-descriptor session, and the prefix is never rebuilt from
        trimmed messages or tombstones.
        """
        messages_path = session_dir / MESSAGES_FILENAME
        non_system_messages = [
            message for message in messages if message.role != Role.system
        ]
        live_count = len(non_system_messages)
        persisted_count, boundary_unchanged, last_live_message = (
            self._verify_archived_prefix_and_tail(
                non_system_messages, messages_path, descriptor
            )
        )

        # Same metadata-cache no-op contract as the legacy verify, over the
        # total persisted count (archived prefix plus matched tail).
        metadata_path = session_dir / METADATA_FILENAME
        if (
            not launch_config_dirty
            and boundary_unchanged
            and live_count == persisted_count
        ):
            try:
                cached_metadata = json.loads(read_safe(metadata_path).text)
                if cached_metadata.get(
                    "total_messages"
                ) == persisted_count and cached_metadata.get(
                    "last_message_fingerprint"
                ) == (
                    self._message_fingerprint(last_live_message)
                    if last_live_message is not None
                    else None
                ):
                    return self._cursor_for_messages(
                        live_count, last_live_message, messages_path
                    )
            except (OSError, json.JSONDecodeError):
                pass

        try:
            if (
                live_count > persisted_count
                and boundary_unchanged
                and messages_path.exists()
            ):
                # Append only the live tail; the archived prefix is untouched.
                SessionLogger._persist_messages_sync(
                    SessionLogger._tail_records(messages, persisted_count), session_dir
                )
            elif (
                live_count != persisted_count
                or not boundary_unchanged
                or not messages_path.exists()
            ):
                SessionLogger._rewrite_messages_preserving_archive_sync(
                    descriptor, messages, session_dir
                )
            metadata_dump = self._gated_metadata_dump(
                messages, non_system_messages, stats, tool_manager, session_metadata
            )
            SessionLogger._persist_metadata_sync(metadata_dump, session_dir)
        except SessionDiskFullError:
            raise
        except ArchiveUnavailableError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to save session to {session_dir}: {e}") from e
        finally:
            self.maybe_cleanup_tmp_files()
        return self._cursor_for_messages(
            live_count,
            non_system_messages[-1] if non_system_messages else None,
            messages_path,
        )

    @staticmethod
    def _rewrite_messages_preserving_archive_sync(
        descriptor: ArchivedPrefixV1, messages: Sequence[LLMMessage], session_dir: Path
    ) -> None:
        """Publish the archive-preserving replacement atomically (M1 row 3).

        The temporary replacement carries the ORIGINAL archived bytes —
        streamed off the live transcript through the shared read interface —
        followed by the authoritative live suffix serialized one record at a
        time; ``durable_replace_stream`` publishes it with the same
        temp+fsync+rename+dirsync barriers as every durable write. The
        published file is then BYTE-CHECKED through the shared read
        interface: its first ``prefix_end_byte`` bytes must hash to
        ``descriptor.prefix_sha256``, proving the replacement contains the
        original archived prefix (a RAM-only rewrite cannot pass this check).
        """
        messages_path = session_dir / MESSAGES_FILENAME

        def chunks() -> Iterator[bytes]:
            yield from archived_prefix_chunks(messages_path, descriptor.prefix_end_byte)
            yield from SessionLogger._transcript_chunks(
                SessionLogger._tail_records(messages, descriptor.archived_count)
            )

        try:
            durable_replace_stream(messages_path, chunks())
        except Exception as e:
            if _is_enospc(e):
                raise SessionDiskFullError(
                    messages_path, "rewrite session messages preserving the archive"
                ) from e
            raise RuntimeError(
                f"Failed to rewrite session messages at {messages_path}: {e}"
            ) from e
        try:
            verify_archived_prefix_on_disk(descriptor, messages_path)
        except ArchiveIntegrityError as exc:
            raise ArchiveUnavailableError(
                messages_path,
                f"the published replacement failed the archived-prefix byte "
                f"check: {exc}",
                recoverable_by_restart=True,
            ) from exc

    def install_launch_config(self, launch_config: LaunchMetadata) -> int:
        """Synchronously make an accepted envelope authoritative for later saves."""
        session_info = self._get_session_info()
        if session_info is None:
            return self._launch_config_generation
        _, session_metadata = session_info
        session_metadata.launch_config = launch_config
        self._launch_config_generation += 1
        self._launch_config_dirty = True
        return self._launch_config_generation

    async def persist_launch_config(self, launch_config: LaunchMetadata) -> None:
        """Install committed child launch state and serialize its durable write."""
        # This deliberately precedes lock acquisition: cancellation while waiting
        # for an ordinary save must not leave the previous envelope authoritative.
        launch_config_generation = self.install_launch_config(launch_config)
        async with self._save_lock:
            session_info = self._get_session_info()
            if session_info is None:
                return
            session_dir, session_metadata = session_info
            payload = launch_config.model_dump(mode="json")
            try:
                session_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            except OSError as e:
                if _is_enospc(e):
                    raise SessionDiskFullError(
                        session_dir, "create session directory"
                    ) from e
                raise
            metadata_path = session_dir / METADATA_FILENAME
            if metadata_path.exists():
                try:
                    raw = (await read_safe_async(metadata_path)).text
                    metadata = json.loads(raw)
                except (OSError, json.JSONDecodeError) as e:
                    raise RuntimeError(
                        f"Failed to read session metadata at {metadata_path}: {e}"
                    ) from e
            else:
                metadata = session_metadata.model_dump(mode="json")
            metadata["launch_config"] = payload
            persistence = asyncio.create_task(
                SessionLogger.persist_metadata(metadata, session_dir)
            )
            try:
                await asyncio.shield(persistence)
            except asyncio.CancelledError as cancellation:
                # The filesystem worker outlives cancellation. Retain the lock and
                # ownership until it finishes so a later save cannot be overwritten.
                while not persistence.done():
                    try:
                        await asyncio.shield(persistence)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not persistence.cancelled():
                    try:
                        persistence.result()
                    except Exception:
                        pass
                raise cancellation
            if self._launch_config_generation == launch_config_generation:
                self._launch_config_dirty = False
            self._persisted = True

    async def persist_active_model(self, active_model: str) -> bool:
        """Persist a changed model alias and report whether storage was updated."""
        async with self._save_lock:
            session_info = self._get_session_info()
            if session_info is None:
                return False
            session_dir, session_metadata = session_info
            if self.active_model == active_model:
                return False
            config = dict(session_metadata.config or {})
            config["active_model"] = active_model
            metadata_path = session_dir / METADATA_FILENAME
            if metadata_path.exists():
                try:
                    raw = (await read_safe_async(metadata_path)).text
                    metadata = json.loads(raw)
                except (OSError, json.JSONDecodeError) as e:
                    raise RuntimeError(
                        f"Failed to read session metadata at {metadata_path}: {e}"
                    ) from e
                persisted_config = metadata.get("config")
                if not isinstance(persisted_config, dict):
                    persisted_config = {}
                persisted_config["active_model"] = active_model
                metadata["config"] = persisted_config
                await self._persist_metadata_locked(metadata, session_dir)
            session_metadata.config = config
            return True

    async def persist_plan_mode(self, record: PlanModeRecord) -> None:
        """Record the active plan mode (design §7); written once at entry.

        The in-memory field is set as well as the file patched, for the same
        reason ``created_worktree`` does both: entry can precede the session's
        first message, so on a fresh session there is no metadata file to patch
        yet and only the in-memory record can reach the first full save.
        """
        session_info = self._get_session_info()
        if session_info is None:
            return
        _, session_metadata = session_info
        session_metadata.plan_mode = record
        await self._persist_metadata_field("plan_mode", record.model_dump(mode="json"))

    async def clear_plan_mode(self) -> None:
        """Remove the plan-mode record; the session is no longer planning."""
        session_info = self._get_session_info()
        if session_info is None:
            return
        _, session_metadata = session_info
        session_metadata.plan_mode = None
        await self._persist_metadata_field("plan_mode", None)

    async def persist_loops(self) -> None:
        session_info = self._get_session_info()
        if session_info is None:
            return
        _, session_metadata = session_info
        await self._persist_metadata_field(
            "loops", [loop.model_dump(mode="json") for loop in session_metadata.loops]
        )

    async def persist_created_worktree(self, worktree: WorktreeContext) -> None:
        session_info = self._get_session_info()
        if session_info is None:
            return
        _, session_metadata = session_info
        # Both halves are load-bearing. At session start the metadata file does
        # not exist yet - only the in-memory object does - so setting the field
        # here is what the first full save writes out; on a resume the file is
        # already there and the next full save may be a whole turn away, so the
        # patch below is what reaches disk in time.
        session_metadata.created_worktree = worktree
        await self._persist_metadata_field(
            "created_worktree", worktree.model_dump(mode="json")
        )

    async def persist_child_sessions(self) -> None:
        session_info = self._get_session_info()
        if session_info is None:
            return
        _, session_metadata = session_info
        await self._persist_metadata_field(
            "child_sessions",
            [link.model_dump(mode="json") for link in session_metadata.child_sessions],
        )

    async def _persist_title_fields_locked(
        self, title: str, source: Literal["auto", "manual"]
    ) -> None:
        session_info = self._get_session_info()
        if session_info is None:
            return
        session_dir, _ = session_info
        metadata_path = session_dir / METADATA_FILENAME
        if not metadata_path.exists():
            return
        try:
            raw = (await read_safe_async(metadata_path)).text
            metadata = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to read session metadata at {metadata_path}: {e}"
            ) from e
        metadata["title"] = title
        metadata["title_source"] = source
        await self._persist_metadata_locked(metadata, session_dir)

    async def _persist_metadata_field(self, field: str, value: Any) -> None:
        async with self._save_lock:
            await self._persist_metadata_field_locked(field, value)

    async def _persist_metadata_field_locked(self, field: str, value: Any) -> None:
        session_info = self._get_session_info()
        if session_info is None:
            return
        session_dir, _ = session_info
        metadata_path = session_dir / METADATA_FILENAME
        if not metadata_path.exists():
            return
        try:
            raw = (await read_safe_async(metadata_path)).text
            metadata = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to read session metadata at {metadata_path}: {e}"
            ) from e
        metadata[field] = value
        await self._persist_metadata_locked(metadata, session_dir)

    def reset_session(
        self, session_id: str, *, parent_session_id: str | None = None
    ) -> None:
        """Clear existing session info and setup a new session."""
        if not self.enabled:
            return

        self.session_id = session_id
        # P4a2 open question (b): the retained descriptor's session-id check is
        # re-validated on the rebind — a descriptor naming the departed
        # session is dropped fail-closed.
        self._drop_mismatched_archive_authority(session_id)
        # P4a4: every rebind starts a new reset generation — a trim publication
        # planned before it is stale and never overwrites the current authority.
        self._archive_generation += 1
        self.session_start_time = utc_now().isoformat()
        self.session_dir = self.save_folder
        self.session_metadata = self._initialize_session_metadata()
        self._persisted = False
        self.invalidate_transcript_cursor()
        if parent_session_id is not None:
            self.session_metadata.parent_session_id = parent_session_id

    def resume_existing_session(self, session_id: str, session_dir: Path) -> None:
        if not self.enabled:
            return
        self.apply_resumed_session(
            session_id, session_dir, SessionLoader.load_metadata(session_dir)
        )

    def apply_resumed_session(
        self, session_id: str, session_dir: Path, metadata: SessionMetadata
    ) -> None:
        """Bind to an already-loaded session. Infallible: no disk reads.

        Used by the in-place resume commit, where the metadata was loaded during
        the (fallible) prepare step so the commit itself cannot raise.
        """
        if not self.enabled:
            return

        self.session_id = session_id
        # P4a2 open question (b): re-validate the retained descriptor's
        # session-id check on the rebind — a foreign descriptor is dropped
        # fail-closed; a matching one (warm continuation) is retained.
        self._drop_mismatched_archive_authority(session_id)
        # P4a4: every rebind starts a new reset generation — a trim publication
        # planned before it is stale and never overwrites the current authority.
        self._archive_generation += 1
        self.session_dir = session_dir
        self.session_metadata = metadata
        self._title = metadata.title
        self._persisted = True
        self.invalidate_transcript_cursor()

        if metadata.start_time:
            self.session_start_time = metadata.start_time

    def cleanup_tmp_files(self) -> None:
        """Delete temporary files created more than 5 minutes ago"""
        if not self.enabled or not self.save_dir:
            return

        now = utc_now()
        ago = now - timedelta(minutes=5)

        tmp_files = self.save_dir.glob("**/*.json.tmp")  # Recursive search

        for file_path in tmp_files:
            if file_path.is_file():
                try:
                    file_mtime = datetime.fromtimestamp(
                        file_path.stat().st_mtime, tz=UTC
                    )
                    if file_mtime < ago:
                        file_path.unlink()
                except Exception:
                    continue

    def maybe_cleanup_tmp_files(self) -> None:
        if not self.enabled or not self.save_dir:
            return

        if not self._tmp_cleanup_lock.acquire(blocking=False):
            return
        try:
            now = utc_now()
            if (
                self._last_tmp_cleanup_at is not None
                and now - self._last_tmp_cleanup_at < TMP_CLEANUP_INTERVAL
            ):
                return

            self.cleanup_tmp_files()
            self._last_tmp_cleanup_at = now
        finally:
            self._tmp_cleanup_lock.release()
