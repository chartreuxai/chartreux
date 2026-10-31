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
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4
import warnings

from chartreux.core.llm_models import (
    CONTEXT_INJECTION_ORIGIN_KEY,
    CONTEXT_INJECTION_TOOL_PAIR_V1,
    LLMMessage,
    Role,
)
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
from chartreux.core.session_types import (
    AgentStats,
    LaunchMetadata,
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
    from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
    from chartreux.core.tools.manager import ToolManager


TMP_CLEANUP_INTERVAL = timedelta(seconds=5)
TRANSCRIPT_VERIFY_INTERVAL = 64
# Disk-full save failures are logged at most this often while the condition
# persists; a successful save resets the rate limit so the next occurrence is
# reported immediately.
DISK_FULL_LOG_INTERVAL = timedelta(seconds=60)
logger = logging.getLogger(__name__)


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
        self._launch_config_dirty = False
        self._launch_config_generation = 0
        # Rate-limits repeated disk-full warnings so saves cannot spam the log.
        self._disk_full_last_logged_at: datetime | None = None
        # Serializes writes so concurrent saves cannot interleave appends to
        # messages.jsonl or race on the metadata read-modify-write.
        self._save_lock = asyncio.Lock()

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
        """Force the next interaction save to verify the full transcript."""
        self._transcript_cursor_generation += 1
        self._transcript_cursor = None
        self._transcript_saves_since_verify = 0

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
    def _message_record(message: LLMMessage) -> dict[str, Any]:
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
        """
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
    async def save_interaction(  # noqa: PLR0915 - one serialized commit boundary
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

        # PI1/T (S1 save boundary): a transcript defect is never silently
        # persisted. The append/insert boundary already refuses to create one;
        # this backstop fails closed with the typed local diagnostic so no
        # unknown path can re-poison the log the load-time repair just healed.
        self._check_transcript_persistence(non_system_messages)

        # Serialization and fsync are too slow for the UI thread, so snapshot
        # here and hand off to a worker thread.
        messages_snapshot = list(messages)
        config_snapshot = config.model_dump(mode="json")
        async with self._save_lock:
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
            # occurrence immediately instead of staying rate-limited.
            self._disk_full_last_logged_at = None
            if self._launch_config_generation == launch_config_generation:
                self._launch_config_dirty = False

    @staticmethod
    def _check_transcript_persistence(
        non_system_messages: Sequence[LLMMessage],
    ) -> None:
        """S1 save boundary: fail closed, never silently persist a defect.

        Function-level import: a top-level import of the agent_loop validator
        would cycle (agent_loop -> _loop -> this module).
        """
        from chartreux.core.agent_loop._transcript_integrity import (
            TranscriptIntegrityError,
            find_persistence_violations,
        )

        violations = find_persistence_violations(non_system_messages)
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
        tools_available = [
            {"type": "function", "function": fn.model_dump()}
            for fn in tool_manager.available_tool_specs()
        ]
        system_prompt = (
            messages[0].model_dump()
            if messages and messages[0].role == Role.system
            else None
        )
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
                metadata_dump = self._interaction_metadata_dump(
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
        """
        persisted_count = 0
        transcript_valid = True
        boundary_unchanged = True
        last_live_message: LLMMessage | None = None
        try:
            persisted_records = SessionLogger._stream_persisted_records(messages_path)
            for live_message in (
                message for message in messages if message.role != Role.system
            ):
                persisted_record = next(persisted_records, None)
                if persisted_record is None:
                    break
                if persisted_record != SessionLogger._message_record(live_message):
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
            for _ in persisted_records:
                pass
        except _PersistedTranscriptError:
            transcript_valid = False
            boundary_unchanged = False
        return persisted_count, transcript_valid, boundary_unchanged, last_live_message

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
            metadata_dump = self._interaction_metadata_dump(
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
