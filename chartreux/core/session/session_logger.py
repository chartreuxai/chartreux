from __future__ import annotations

import asyncio
from collections.abc import Sequence
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import errno
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4
import warnings

from chartreux.core.llm_models import LLMMessage, Role
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
from chartreux.utils.io import read_safe, read_safe_async
from chartreux.utils.platform import resolve_git_executable
from chartreux.utils.session_id import shorten_session_id

if TYPE_CHECKING:
    from chartreux.core.agents.models import AgentProfile
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
        temp_metadata_filepath = None
        metadata_filepath = session_dir / METADATA_FILENAME
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json.tmp",
                dir=str(session_dir),
                delete=False,
                encoding="utf-8",
            ) as f:
                temp_metadata_filepath = Path(f.name)
                f.write(json.dumps(metadata, indent=2, ensure_ascii=False))
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_metadata_filepath, str(metadata_filepath))
        except Exception as e:
            if _is_enospc(e):
                raise SessionDiskFullError(
                    metadata_filepath, "persist session metadata"
                ) from e
            raise RuntimeError(
                f"Failed to persist session metadata to {metadata_filepath}: {e}"
            ) from e
        finally:
            if (
                temp_metadata_filepath
                and temp_metadata_filepath.exists()
                and temp_metadata_filepath.is_file()
            ):
                temp_metadata_filepath.unlink()

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
    def _persist_messages_sync(messages: list[dict], session_dir: Path) -> int:
        messages_filepath = session_dir / "messages.jsonl"
        try:
            # Session logs hold raw tool results, so new files are owner-only.
            # An existing file (a resumed session) keeps its current mode.
            descriptor = os.open(
                messages_filepath, os.O_APPEND | os.O_CREAT | os.O_RDWR, 0o600
            )
            if os.lseek(descriptor, 0, os.SEEK_END) > 0:
                os.lseek(descriptor, -1, os.SEEK_END)
                if os.read(descriptor, 1) != b"\n":
                    os.write(descriptor, b"\n")
            with os.fdopen(descriptor, "a", encoding="utf-8") as f:
                for message in messages:
                    f.write(json.dumps(message, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
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
    def _overwrite_messages_sync(messages: list[dict], session_dir: Path) -> None:
        messages_filepath = session_dir / MESSAGES_FILENAME
        temp_filepath = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".jsonl.tmp",
                dir=str(session_dir),
                delete=False,
                encoding="utf-8",
            ) as f:
                temp_filepath = Path(f.name)
                for message in messages:
                    f.write(json.dumps(message, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_filepath, str(messages_filepath))
        except Exception as e:
            if _is_enospc(e):
                raise SessionDiskFullError(
                    messages_filepath, "overwrite session messages"
                ) from e
            raise RuntimeError(
                f"Failed to overwrite session messages at {messages_filepath}: {e}"
            ) from e
        finally:
            if temp_filepath and temp_filepath.exists() and temp_filepath.is_file():
                temp_filepath.unlink()

    @staticmethod
    def _message_fingerprint(message: LLMMessage) -> str:
        payload = json.dumps(
            message.model_dump(exclude_none=True, mode="json"), sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def save_interaction(
        self,
        messages: Sequence[LLMMessage],
        stats: AgentStats,
        config: ChartreuxConfigSchema,
        tool_manager: ToolManager,
        agent_profile: AgentProfile | None,
        *,
        allow_empty: bool = False,
    ) -> None:
        session_info = self._get_session_info()
        if session_info is None:
            return
        session_dir, session_metadata = session_info

        non_system_messages = [m for m in messages if m.role != Role.system]

        # Empty conversations are only persisted on explicit opt-in (rewind to
        # the first message); otherwise an empty log would be unloadable.
        if not non_system_messages and not allow_empty:
            return

        # Serialization and fsync are too slow for the UI thread, so snapshot
        # here and hand off to a worker thread.
        messages_snapshot = list(messages)
        config_snapshot = config.model_dump(mode="json")
        async with self._save_lock:
            session_metadata.config = config_snapshot
            launch_config_generation = self._launch_config_generation
            metadata_snapshot = session_metadata.model_copy(deep=True)
            cursor_snapshot = self._transcript_cursor
            saves_since_verify = self._transcript_saves_since_verify
            cursor_generation_snapshot = self._transcript_cursor_generation
            persistence = asyncio.create_task(
                asyncio.to_thread(
                    self._save_interaction_sync,
                    messages_snapshot,
                    stats,
                    tool_manager,
                    agent_profile,
                    session_dir,
                    metadata_snapshot,
                    allow_empty,
                    self._launch_config_dirty,
                    cursor_snapshot,
                    saves_since_verify,
                )
            )
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
    def _read_persisted_messages(
        messages_path: Path,
    ) -> tuple[list[dict[str, Any]], bool]:
        persisted_messages: list[dict[str, Any]] = []
        try:
            if messages_path.exists():
                for line in read_safe(messages_path).text.splitlines():
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        return [], False
                    persisted_messages.append(message)
        except (OSError, json.JSONDecodeError):
            return [], False
        return persisted_messages, True

    @staticmethod
    def _transcript_line_digest(message_data: dict[str, Any]) -> str:
        line = json.dumps(message_data, ensure_ascii=False)
        return hashlib.sha256(line.encode("utf-8")).hexdigest()

    @classmethod
    def _cursor_for_messages(
        cls, messages: list[dict[str, Any]], messages_path: Path
    ) -> _TranscriptCursor:
        file_size = messages_path.stat().st_size if messages_path.exists() else 0
        boundary_digest = (
            cls._transcript_line_digest(messages[-1]) if messages else None
        )
        return _TranscriptCursor(
            count=len(messages),
            boundary_digest=boundary_digest,
            file_size=file_size,
            metadata_published=True,
        )

    def _interaction_metadata_dump(
        self,
        messages: list[LLMMessage],
        non_system_messages: list[LLMMessage],
        stats: AgentStats,
        tool_manager: ToolManager,
        agent_profile: AgentProfile | None,
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
            "agent_profile": (
                {"name": agent_profile.name, "overrides": agent_profile.overrides}
                if agent_profile is not None
                else None
            ),
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
        agent_profile: AgentProfile | None,
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
                agent_profile,
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
                agent_profile,
                session_dir,
                session_metadata,
                launch_config_dirty,
            )
            return verified_cursor, 0

        if cursor.count:
            boundary_data = non_system_messages[cursor.count - 1].model_dump(
                exclude_none=True, mode="json"
            )
            if (
                cursor.boundary_digest is None
                or self._transcript_line_digest(boundary_data) != cursor.boundary_digest
            ):
                verified_cursor = self._save_full_verify(
                    messages,
                    stats,
                    tool_manager,
                    agent_profile,
                    session_dir,
                    session_metadata,
                    launch_config_dirty,
                )
                return verified_cursor, 0

        tail = [
            message.model_dump(exclude_none=True, mode="json")
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
                    messages,
                    non_system_messages,
                    stats,
                    tool_manager,
                    agent_profile,
                    session_metadata,
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

    def _save_full_verify(
        self,
        messages: list[LLMMessage],
        stats: AgentStats,
        tool_manager: ToolManager,
        agent_profile: AgentProfile | None,
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
            raise RuntimeError(
                f"Failed to create session directory at {session_dir}: {type(e).__name__}: {e}"
            ) from e

        # messages.jsonl is authoritative for the persisted boundary. Metadata
        # is published separately and may lag after an interrupted save.
        messages_path = session_dir / MESSAGES_FILENAME
        persisted_messages, transcript_valid = self._read_persisted_messages(
            messages_path
        )

        non_system_messages = [m for m in messages if m.role != Role.system]

        if not transcript_valid and messages_path.exists():
            self._quarantine_corrupt_transcript(messages_path)
            persisted_messages = []

        persisted_count = len(persisted_messages)
        current_data = [
            message.model_dump(exclude_none=True, mode="json")
            for message in non_system_messages
        ]
        boundary_unchanged = transcript_valid and persisted_count <= len(current_data)
        if boundary_unchanged:
            boundary_unchanged = persisted_messages == current_data[:persisted_count]

        # Preserve the existing no-op behavior only when the metadata cache
        # confirms the transcript boundary. If publication failed previously,
        # the transcript still prevents duplicate appends and this retry repairs
        # the stale metadata.
        metadata_path = session_dir / METADATA_FILENAME
        if (
            not launch_config_dirty
            and boundary_unchanged
            and len(current_data) == persisted_count
        ):
            try:
                cached_metadata = json.loads(read_safe(metadata_path).text)
                if cached_metadata.get(
                    "total_messages"
                ) == persisted_count and cached_metadata.get(
                    "last_message_fingerprint"
                ) == (
                    self._message_fingerprint(non_system_messages[-1])
                    if non_system_messages
                    else None
                ):
                    return self._cursor_for_messages(current_data, messages_path)
            except (OSError, json.JSONDecodeError):
                pass

        try:
            if (
                len(current_data) > persisted_count
                and boundary_unchanged
                and messages_path.exists()
            ):
                SessionLogger._persist_messages_sync(
                    current_data[persisted_count:], session_dir
                )
            elif (
                len(current_data) != persisted_count
                or not boundary_unchanged
                or not messages_path.exists()
            ):
                SessionLogger._overwrite_messages_sync(current_data, session_dir)

            metadata_dump = self._interaction_metadata_dump(
                messages,
                non_system_messages,
                stats,
                tool_manager,
                agent_profile,
                session_metadata,
            )
            SessionLogger._persist_metadata_sync(metadata_dump, session_dir)
        except SessionDiskFullError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to save session to {session_dir}: {e}") from e
        finally:
            self.maybe_cleanup_tmp_files()
        return self._cursor_for_messages(current_data, messages_path)

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
            session_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
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
        self._transcript_cursor = None
        self._transcript_saves_since_verify = 0
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
        self._transcript_cursor = None
        self._transcript_saves_since_verify = 0

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
