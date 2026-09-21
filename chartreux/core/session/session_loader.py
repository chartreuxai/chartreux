from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import errno
import json
import os
from pathlib import Path
import stat
from typing import TYPE_CHECKING, Any, BinaryIO
import warnings

from chartreux.core.llm_models import LLMMessage
from chartreux.core.session.session_index import (
    MESSAGES_FILENAME,
    METADATA_FILENAME,
    SessionInfo,
    session_index_for,
)
from chartreux.core.session_types import SessionMetadata
from chartreux.utils.io import read_safe
from chartreux.utils.session_id import shorten_session_id

if TYPE_CHECKING:
    from chartreux.core.config import SessionLoggingConfig


__all__ = [
    "MESSAGES_FILENAME",
    "METADATA_FILENAME",
    "BoundedSessionLoad",
    "BoundedSessionLoadState",
    "SessionFileContainmentError",
    "SessionInfo",
    "SessionLoader",
]

# The transcript viewer must never load arbitrarily large saved sessions.
BOUNDED_SESSION_READ_LIMIT = 16 * 1024 * 1024


class SessionFileContainmentError(OSError):
    """A bounded session file was not a stable, regular directory entry."""


class BoundedSessionLoadState(StrEnum):
    AVAILABLE = "available"
    MISSING = "missing"
    EXCEEDS_LIMIT = "exceeds_limit"
    INVALID = "invalid"


@dataclass(frozen=True)
class BoundedSessionLoad:
    """Raw saved-session data read without exceeding an input byte ceiling."""

    state: BoundedSessionLoadState
    messages: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


# Upper bound for a first-message preview used as a fallback session label.
_PREVIEW_MAX_CHARS = 200


def _preview_snippet(text: str) -> str:
    # Cap length: the fallback label lands in the resume picker and ACP session
    # lists, which a long first-message paste would otherwise blow up. Newlines
    # are already collapsed by _clean_text.
    if len(text) > _PREVIEW_MAX_CHARS:
        return text[:_PREVIEW_MAX_CHARS].rstrip() + "…"
    return text


class SessionLoader:
    @staticmethod
    def load_session_bounded(  # noqa: PLR0911
        filepath: Path,
        *,
        byte_limit: int = BOUNDED_SESSION_READ_LIMIT,
        dir_fd: int | None = None,
    ) -> BoundedSessionLoad:
        """Read saved JSONL and metadata with a strict in-read byte ceiling.

        This opt-in path intentionally returns a typed outcome rather than
        changing ``load_session``'s legacy exceptions and unbounded behavior.
        """
        metadata_path = filepath / METADATA_FILENAME
        messages_path = filepath / MESSAGES_FILENAME
        try:
            metadata_bytes = SessionLoader._read_bounded_file(
                metadata_path, byte_limit, dir_fd=dir_fd
            )
            if metadata_bytes is None:
                return BoundedSessionLoad(BoundedSessionLoadState.EXCEEDS_LIMIT)
            metadata = json.loads(metadata_bytes)
            if not isinstance(metadata, dict):
                return BoundedSessionLoad(BoundedSessionLoadState.INVALID)
            messages, recovered, exceeds = SessionLoader._read_bounded_message_lines(
                messages_path, byte_limit, dir_fd=dir_fd
            )
        except SessionFileContainmentError:
            raise
        except FileNotFoundError:
            return BoundedSessionLoad(BoundedSessionLoadState.MISSING)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return BoundedSessionLoad(BoundedSessionLoadState.INVALID)

        if exceeds:
            return BoundedSessionLoad(BoundedSessionLoadState.EXCEEDS_LIMIT)
        if messages is None or (recovered and not messages):
            return BoundedSessionLoad(BoundedSessionLoadState.INVALID)
        if not SessionLoader._log_is_loadable(messages, metadata):
            return BoundedSessionLoad(BoundedSessionLoadState.INVALID)
        return BoundedSessionLoad(BoundedSessionLoadState.AVAILABLE, messages, metadata)

    @staticmethod
    def _open_bounded_file(path: Path, *, dir_fd: int | None = None) -> BinaryIO:
        target: str | Path = path.name if dir_fd is not None else path
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        before = None
        if not nofollow:
            before = os.stat(target, dir_fd=dir_fd, follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                raise SessionFileContainmentError(
                    f"Session file is not regular: {path.name}"
                )
        try:
            fd = os.open(target, os.O_RDONLY | os.O_NONBLOCK | nofollow, dir_fd=dir_fd)
        except OSError as exc:
            if nofollow and exc.errno == errno.ELOOP:
                raise SessionFileContainmentError(
                    f"Session file is a symbolic link: {path.name}"
                ) from exc
            raise
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise SessionFileContainmentError(
                    f"Session file is not regular: {path.name}"
                )
            if before is not None and (before.st_dev, before.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise SessionFileContainmentError(
                    f"Session file changed while opening: {path.name}"
                )
            return os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _read_bounded_file(
        path: Path, byte_limit: int, *, dir_fd: int | None = None
    ) -> bytes | None:
        """Return file contents, or ``None`` after reading only an overflow byte."""
        with SessionLoader._open_bounded_file(path, dir_fd=dir_fd) as stream:
            content = stream.read(byte_limit + 1)
        return content if len(content) <= byte_limit else None

    @staticmethod
    def _read_bounded_message_lines(  # noqa: PLR0911
        path: Path, byte_limit: int, *, dir_fd: int | None = None
    ) -> tuple[list[dict[str, Any]] | None, bool, bool]:
        """Parse JSONL while bounding both aggregate bytes and one record."""
        messages: list[dict[str, Any]] = []
        consumed = 0
        with SessionLoader._open_bounded_file(path, dir_fd=dir_fd) as stream:
            while True:
                remaining = byte_limit - consumed
                if remaining < 0:
                    return None, False, True
                line = stream.readline(remaining + 1)
                if not line:
                    return messages, False, False
                if len(line) > remaining:
                    return None, False, True
                consumed += len(line)
                # A record exactly filling the budget may be complete only at EOF.
                if consumed == byte_limit and not line.endswith(b"\n"):
                    if stream.read(1):
                        return None, False, True
                is_final = not line.endswith(b"\n")
                try:
                    message = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if is_final or not stream.read(1):
                        return messages, True, False
                    return None, False, False
                if not isinstance(message, dict):
                    if is_final or not stream.read(1):
                        return messages, True, False
                    return None, False, False
                messages.append(message)
                if is_final:
                    return messages, False, False

    @staticmethod
    def _parse_message_lines_with_recovery(
        text: str,
    ) -> tuple[list[dict[str, Any]] | None, bool]:
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()

        messages: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    return messages, True
                return None, False
            if not isinstance(message, dict):
                if index == len(lines) - 1:
                    return messages, True
                return None, False
            messages.append(message)
        return messages, False

    @staticmethod
    def _parse_message_lines(text: str) -> list[dict[str, Any]] | None:
        messages, _recovered = SessionLoader._parse_message_lines_with_recovery(text)
        return messages

    @staticmethod
    def _same_working_directory(stored: Any, working_directory: Path) -> bool:
        if not isinstance(stored, str):
            return False
        if stored == str(working_directory):
            return True
        try:
            return Path(stored).resolve() == working_directory.resolve()
        except OSError:
            return False

    @staticmethod
    def _session_reaches(metadata: dict[str, Any], working_directory: Path) -> bool:
        """Whether this session should be offered from *working_directory*.

        The environment entry follows a move and ``origin_directory`` does not,
        so a session that has moved is findable both where it began and where
        it now sits. Sessions written before the origin was recorded carry only
        the environment entry, which is where they started and stayed.
        """
        current = (metadata.get("environment") or {}).get("working_directory")
        return any(
            SessionLoader._same_working_directory(stored, working_directory)
            for stored in (metadata.get("origin_directory"), current)
        )

    @staticmethod
    def _read_validated_session(
        session_dir: Path, working_directory: Path | None = None
    ) -> dict[str, Any] | None:
        metadata_path = session_dir / METADATA_FILENAME
        messages_path = session_dir / MESSAGES_FILENAME

        if not metadata_path.is_file() or not messages_path.is_file():
            return None

        try:
            metadata = json.loads(read_safe(metadata_path).text)
            if not isinstance(metadata, dict):
                return None
            if working_directory is not None and not SessionLoader._session_reaches(
                metadata, working_directory
            ):
                return None

            messages, recovered = SessionLoader._parse_message_lines_with_recovery(
                read_safe(messages_path).text
            )
        except (OSError, json.JSONDecodeError):
            return None

        # A malformed final record is recoverable only when a valid prefix exists;
        # otherwise an entirely invalid transcript could masquerade as empty.
        if (recovered and not messages) or not SessionLoader._log_is_loadable(
            messages, metadata
        ):
            return None

        return metadata

    @staticmethod
    def _log_is_loadable(
        messages: list[dict[str, Any]] | None, metadata: dict[str, Any]
    ) -> bool:
        if messages is None:
            return False
        # An empty log is valid only when metadata records an empty session.
        return bool(messages) or metadata.get("total_messages") == 0

    @staticmethod
    def _is_valid_session(
        session_dir: Path, working_directory: Path | None = None
    ) -> bool:
        return (
            SessionLoader._read_validated_session(session_dir, working_directory)
            is not None
        )

    @staticmethod
    def latest_session(
        session_dirs: list[Path], working_directory: Path | None = None
    ) -> Path | None:
        sessions_with_mtime: list[tuple[Path, float]] = []
        for session in session_dirs:
            messages_path = session / MESSAGES_FILENAME
            if not messages_path.is_file():
                continue
            try:
                mtime = messages_path.stat().st_mtime
                sessions_with_mtime.append((session, mtime))
            except OSError:
                continue

        if not sessions_with_mtime:
            return None

        sessions_with_mtime.sort(key=lambda x: x[1], reverse=True)

        for session, _mtime in sessions_with_mtime:
            if SessionLoader._is_valid_session(
                session, working_directory=working_directory
            ):
                return session

        return None

    @staticmethod
    def find_latest_session(
        config: SessionLoggingConfig, working_directory: Path | None = None
    ) -> Path | None:
        save_dir = Path(config.save_dir)
        if not save_dir.exists():
            return None

        pattern = f"{config.session_prefix}_*"
        session_dirs = list(save_dir.glob(pattern))

        return SessionLoader.latest_session(
            session_dirs, working_directory=working_directory
        )

    @staticmethod
    def find_session_by_id(
        session_id: str,
        config: SessionLoggingConfig,
        working_directory: Path | None = None,
    ) -> Path | None:
        matches = SessionLoader._find_session_dirs_by_short_id(session_id, config)

        return SessionLoader.latest_session(
            matches, working_directory=working_directory
        )

    @staticmethod
    def does_session_exist(
        session_id: str, config: SessionLoggingConfig
    ) -> Path | None:
        for session_dir in SessionLoader._find_session_dirs_by_short_id(
            session_id, config
        ):
            if (session_dir / MESSAGES_FILENAME).is_file():
                return session_dir
        return None

    @staticmethod
    def _find_session_dirs_by_short_id(
        session_id: str, config: SessionLoggingConfig
    ) -> list[Path]:
        save_dir = Path(config.save_dir)
        if not save_dir.exists():
            return []

        short_id = shorten_session_id(session_id)
        return list(save_dir.glob(f"{config.session_prefix}_*_{short_id}"))

    @staticmethod
    def list_sessions(
        config: SessionLoggingConfig, cwd: str | None = None, *, strict_io: bool = False
    ) -> list[SessionInfo]:
        sessions = session_index_for(config).list(cwd, strict_io=strict_io)
        # The index yields arbitrary order; callers expect most-recent first.
        # updated_at is normalized UTC ISO, so a lexicographic sort is
        # chronological; sessions without one sort last.
        sessions.sort(key=lambda item: item["updated_at"], reverse=True)
        return sessions

    @staticmethod
    def load_metadata(session_dir: Path) -> SessionMetadata:
        metadata_path = session_dir / METADATA_FILENAME
        if not metadata_path.exists():
            raise ValueError(f"Session metadata not found at {session_dir}")

        try:
            metadata_content = read_safe(metadata_path).text
            return SessionMetadata.model_validate_json(metadata_content)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to load session metadata at {session_dir}: {e}"
            ) from e

    @staticmethod
    def load_session(filepath: Path) -> tuple[list[LLMMessage], dict[str, Any]]:
        metadata_filepath = filepath / METADATA_FILENAME
        if metadata_filepath.exists():
            try:
                metadata = json.loads(read_safe(metadata_filepath).text)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Session metadata contains invalid JSON (may have been corrupted): "
                    f"{filepath}\nDetails: {e}"
                ) from e
        else:
            metadata = {}

        messages_filepath = filepath / MESSAGES_FILENAME
        try:
            transcript = read_safe(messages_filepath).text
        except Exception as e:
            raise ValueError(
                f"Error reading session messages at {filepath}: {e}"
            ) from e

        data, recovered = SessionLoader._parse_message_lines_with_recovery(transcript)
        if data is None:
            raise ValueError(
                "Session messages contain invalid JSON in an interior record "
                f"(may have been corrupted): {filepath}"
            )
        if recovered and not data:
            raise ValueError(
                "Session messages contain invalid JSON: no valid records precede the "
                f"malformed final record: {filepath}"
            )

        # An empty log is valid only when metadata records an empty session.
        if not data and metadata.get("total_messages") != 0:
            raise ValueError(
                f"Session messages file is empty (may have been corrupted by interruption): "
                f"{filepath}"
            )

        if recovered:
            warnings.warn(
                f"Omitted malformed final session message record from {filepath}",
                RuntimeWarning,
                stacklevel=2,
            )

        messages = [
            LLMMessage.model_validate(msg) for msg in data if msg["role"] != "system"
        ]

        return messages, metadata

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.strip().replace("\n", " ")
        return text or "(empty message)"

    @staticmethod
    def _extract_text_from_content(content: Any) -> str | None:
        if isinstance(content, list):
            parts = [
                p["text"]
                for p in content
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            ]
            content = "\n".join(parts)
        if not isinstance(content, str) or not content:
            return None
        return SessionLoader._clean_text(content)

    @staticmethod
    def _latest_matching_session_dir(
        session_id: str, config: SessionLoggingConfig
    ) -> Path | None:
        candidates: list[tuple[Path, float]] = []
        for session_dir in SessionLoader._find_session_dirs_by_short_id(
            session_id, config
        ):
            messages_path = session_dir / MESSAGES_FILENAME
            try:
                candidates.append((session_dir, messages_path.stat().st_mtime))
            except OSError:
                continue

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0]

    @staticmethod
    def get_first_user_message(session_id: str, config: SessionLoggingConfig) -> str:
        """Get the first user message from a session for preview.
        Streams the transcript and stops at the first user message; never
        parses the whole conversation or runs Pydantic validation. The result is
        length-capped so an untitled session's fallback label (resume picker, ACP
        session list) can't be blown up by a long first-message paste.
        """
        session_path = SessionLoader._latest_matching_session_dir(session_id, config)
        if not session_path:
            return "(session not found)"

        try:
            content = read_safe(session_path / MESSAGES_FILENAME).text
        except OSError:
            return "(error reading session)"

        for line in content.split("\n"):
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                return "(corrupted session)"
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            text = SessionLoader._extract_text_from_content(message.get("content"))
            if text:
                return _preview_snippet(text)

        return "(no user messages)"
