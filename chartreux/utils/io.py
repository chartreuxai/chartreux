from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
import contextlib
from contextlib import asynccontextmanager
import hashlib
import locale
import os
from pathlib import Path
import shutil
import time
from typing import NamedTuple

import anyio


class ReadSafeResult(NamedTuple):
    r"""Text decoded from a file, the codec used, and the detected newline style.

    ``text`` is always normalized to use ``\n`` line endings regardless of the
    original file. ``newline`` records the original style (``"\n"``, ``"\r\n"``,
    or ``"\r"``) so callers can round-trip writes via ``open(..., newline=...)``.
    When no newline is present, defaults to ``os.linesep`` to match Python's
    default text-mode write behavior.
    """

    text: str
    encoding: str
    newline: str = os.linesep


def _detect_newline(text: str) -> str:
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    cr = text.count("\r") - crlf
    counts = {"\r\n": crlf, "\n": lf, "\r": cr}
    best = max(counts, key=lambda nl: counts[nl])
    return best if counts[best] > 0 else os.linesep


def _encodings_from_bom(raw: bytes) -> str | None:
    # Endianless codecs (utf-16/utf-32) auto-detect byte order from the BOM and
    # strip it, symmetric with utf-8-sig; the explicit -le/-be variants leave a
    # stray U+FEFF in the decoded text. Check the 32-bit BOMs first: the
    # utf-32-le BOM has the utf-16-le BOM as a prefix.
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return "utf-32"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    return None


def _encoding_from_best_match(raw: bytes) -> str | None:
    from charset_normalizer import from_bytes

    if not (match := from_bytes(raw).best()):
        return None
    return match.encoding


def _get_candidate_encodings(
    raw: bytes, preferred_encoding: str | None = None
) -> Iterator[str]:
    """Yield candidate encodings lazily — expensive detection runs only if needed."""
    seen: set[str] = set()

    def _emit(encoding: str | None) -> Iterator[str]:
        if encoding and (key := encoding.lower()) not in seen:
            seen.add(key)
            yield encoding

    yield from _emit(_encodings_from_bom(raw))
    yield from _emit("utf-8")
    yield from _emit(preferred_encoding)
    yield from _emit(locale.getpreferredencoding(False))
    yield from _emit(_encoding_from_best_match(raw))


def normalize_newlines(text: str) -> tuple[str, str]:
    r"""Return ``text`` with ``\n`` newlines and the detected original style."""
    if "\r" not in text:
        newline = "\n" if "\n" in text else os.linesep
        return text, newline
    newline = _detect_newline(text)
    return text.replace("\r\n", "\n").replace("\r", "\n"), newline


def _decode(
    raw: bytes, *, raise_on_error: bool, preferred: str | None
) -> tuple[str, str]:
    for encoding in _get_candidate_encodings(raw, preferred):
        try:
            return raw.decode(encoding), encoding
        except (LookupError, UnicodeDecodeError, ValueError):
            pass
    errors = "strict" if raise_on_error else "replace"
    return raw.decode("utf-8", errors=errors), "utf-8"


def decode_safe(raw: bytes, *, raise_on_error: bool = False) -> ReadSafeResult:
    """Decode file ``raw`` like :func:`read_safe` after ``read_bytes``.

    Tries BOM, UTF-8, locale, charset-normalizer, then UTF-8 (strict or
    replace). ``UnicodeDecodeError`` can only occur in that last step when
    ``raise_on_error`` is true.
    """
    text, encoding = _decode(raw, raise_on_error=raise_on_error, preferred=None)
    text, newline = normalize_newlines(text)
    return ReadSafeResult(text, encoding, newline)


def decode_console_safe(raw: bytes, *, raise_on_error: bool = False) -> str:
    r"""Decode console output with the same fallback chain as :func:`decode_safe`.

    Newlines are left verbatim: a lone ``\r`` redraws the line rather than ending one,
    ``\r\n`` here would break a reader that pages the output in byte windows,
    because a window can end between the two bytes. Collapsing carriage returns
    is a rendering decision, so it belongs to whoever draws the text.
    """
    text, _encoding = _decode(raw, raise_on_error=raise_on_error, preferred=None)
    return text


def encode_safe(text: str, *, encoding: str = "utf-8", newline: str = "\n") -> bytes:
    r"""Inverse of :func:`decode_safe`: translate ``\n`` line endings to
    ``newline`` and encode with ``encoding``, falling back to UTF-8 when the codec
    cannot represent the text. The in-memory sibling of ``atomic_replace``'s
    ``open(..., newline=...)`` write.
    """
    body = text if newline == "\n" else text.replace("\n", newline)
    try:
        return body.encode(encoding)
    except (LookupError, UnicodeError):
        return body.encode("utf-8")


def read_safe(path: Path, *, raise_on_error: bool = False) -> ReadSafeResult:
    """Read ``path`` and decode with :func:`decode_safe`."""
    return decode_safe(path.read_bytes(), raise_on_error=raise_on_error)


async def read_safe_async(
    path: Path, *, raise_on_error: bool = False
) -> ReadSafeResult:
    """Async :func:`read_safe` (``anyio``)."""
    raw = await anyio.Path(path).read_bytes()
    return await asyncio.to_thread(decode_safe, raw, raise_on_error=raise_on_error)


class BoundedReadResult(NamedTuple):
    r"""A bounded slice of a file's lines plus truncation metadata.

    ``lines`` are decoded and ``\n``-normalized, without trailing newlines.
    ``total_lines`` is ``None`` when the read stopped early at the line or byte
    budget (the true total is unknown without scanning the rest of the file);
    otherwise it is the number of lines in the file. ``was_truncated`` is true
    when the read stopped before reaching end of file.
    """

    lines: list[str]
    total_lines: int | None
    was_truncated: bool


def read_lines_safe(
    path: Path, *, start_line: int = 1, limit: int, max_bytes: int
) -> BoundedReadResult:
    r"""Read up to ``limit`` lines from ``start_line`` (1-indexed) bounded by bytes.

    Streams the file line-by-line in binary and stops once ``limit`` lines or
    ``max_bytes`` of selected content have been collected, so large files are
    never loaded whole. The collected bytes are decoded once via
    :func:`decode_safe`, which also normalizes ``\r\n``/``\r`` to ``\n``.
    """
    raw_lines: list[bytes] = []
    bytes_read = 0
    line_number = 0
    was_truncated = True

    with path.open("rb") as f:
        while raw_line := f.readline():
            line_number += 1
            if line_number < start_line:
                continue
            if len(raw_lines) >= limit:
                break
            if bytes_read + len(raw_line) > max_bytes:
                remaining = max_bytes - bytes_read
                if remaining > 0:
                    raw_lines.append(raw_line[:remaining])
                break
            raw_lines.append(raw_line)
            bytes_read += len(raw_line)
        else:
            was_truncated = False

    total_lines = None if was_truncated else line_number
    lines = decode_safe(b"".join(raw_lines)).text.splitlines()
    return BoundedReadResult(lines, total_lines, was_truncated)


async def read_lines_safe_async(
    path: Path, *, start_line: int = 1, limit: int, max_bytes: int
) -> BoundedReadResult:
    """Async :func:`read_lines_safe` (runs the blocking read in a thread)."""
    return await asyncio.to_thread(
        read_lines_safe, path, start_line=start_line, limit=limit, max_bytes=max_bytes
    )


CONTENT_REVISION_ALGORITHM = "blake2b-128"
_CONTENT_REVISION_DIGEST_BYTES = 16


class ContentRevisionHasher:
    r"""Incremental hasher producing the content-revision identifier form.

    blake2b is stable across processes and platforms (unlike the built-in
    ``hash``), cheap to feed while streaming a file, and a 128-bit digest is
    ample for change detection. The identifier is ``"blake2b-128:<hex>"``.
    """

    def __init__(self) -> None:
        self._hash = hashlib.blake2b(digest_size=_CONTENT_REVISION_DIGEST_BYTES)

    def update(self, data: bytes) -> None:
        self._hash.update(data)

    def identifier(self) -> str:
        return f"{CONTENT_REVISION_ALGORITHM}:{self._hash.hexdigest()}"


def content_revision(data: bytes) -> str:
    """Return the stable content-revision identifier for exact file bytes."""
    hasher = ContentRevisionHasher()
    hasher.update(data)
    return hasher.identifier()


class WindowReadResult(NamedTuple):
    r"""A whole-line window of a file plus content-revision metadata.

    ``lines`` are decoded and ``\n``-normalized, without trailing newlines, and
    are never cut mid-line: a line that would exceed the remaining byte budget
    starts the next window instead. The first selected line is exempt (it is
    collected whole, bounded by ``max_line_bytes``) so callers can slice an
    oversized line explicitly; ``first_line_truncated`` reports that this
    elision happened. ``last_line_may_be_partial`` marks windows produced by a
    transport that cuts its tail line at a byte cap (the tool I/O port), so
    callers can avoid serving the possibly-cut line.

    ``revision`` is the content revision of the entire file and is present only
    when the read consumed the whole file: either a full scan that reached EOF
    (any ``start_line``), or a bounded window that reached EOF starting from
    line 1. Otherwise it is ``None`` (the window is a partial view of the
    file, which is not a stable file-state identifier).

    ``total_lines`` is the exact line count when the scan reached EOF (always
    true for a full scan), otherwise ``None``. ``was_truncated`` is true when
    the window stopped before end of file. ``raw_bytes`` is the number of raw
    bytes the scan consumed, or ``None`` when unknown (tool I/O port).
    """

    lines: list[str]
    total_lines: int | None
    was_truncated: bool
    revision: str | None
    raw_bytes: int | None
    first_line_truncated: bool
    last_line_may_be_partial: bool


def read_lines_hashed(
    path: Path,
    *,
    start_line: int = 1,
    limit: int,
    max_bytes: int,
    full_scan: bool = False,
    max_line_bytes: int | None = None,
) -> WindowReadResult:
    r"""Read up to ``limit`` whole lines from ``start_line`` (1-indexed), bounded by bytes.

    Streams the file line-by-line in binary. Unlike :func:`read_lines_safe`, a
    line is never cut mid-way: when the next line would exceed the remaining
    byte budget the window ends before it, so no served line is silently
    truncated. The first selected line is collected whole (up to
    ``max_line_bytes``, default ``max(4 * max_bytes, 256 KiB)``) so oversized
    lines can be sliced explicitly by the caller.

    Every consumed byte feeds a content-revision hash. With ``full_scan`` the
    scan continues past the window to EOF, so ``total_lines`` and ``revision``
    describe the whole file even though only the window is retained; memory
    stays bounded by the window plus one line.
    """
    if max_line_bytes is None:
        max_line_bytes = max(4 * max_bytes, 256 * 1024)
    raw_lines: list[bytes] = []
    bytes_read = 0
    line_number = 0
    raw_bytes = 0
    was_truncated = True
    first_line_truncated = False
    collecting = True
    scan_reached_eof = False
    hasher = ContentRevisionHasher()

    with path.open("rb") as f:
        while raw_line := f.readline():
            line_number += 1
            raw_bytes += len(raw_line)
            hasher.update(raw_line)
            if not collecting or line_number < start_line:
                continue
            if len(raw_lines) >= limit or (
                raw_lines and bytes_read + len(raw_line) > max_bytes
            ):
                # The window ends before this line; it starts the next window.
                if not full_scan:
                    break
                collecting = False
                continue
            if not raw_lines and len(raw_line) > max_line_bytes:
                # Oversized first line: keep a bounded slice and report the
                # elision; the caller surfaces it explicitly.
                raw_lines.append(raw_line[:max_line_bytes])
                bytes_read += max_line_bytes
                first_line_truncated = True
                if not full_scan:
                    break
                collecting = False
                continue
            raw_lines.append(raw_line)
            bytes_read += len(raw_line)
        else:
            scan_reached_eof = True
            if collecting:
                was_truncated = False

    total_lines = line_number if scan_reached_eof else None
    revision = (
        hasher.identifier()
        if scan_reached_eof and (full_scan or start_line == 1)
        else None
    )
    lines = decode_safe(b"".join(raw_lines)).text.splitlines()
    return WindowReadResult(
        lines,
        total_lines,
        was_truncated,
        revision,
        raw_bytes,
        first_line_truncated,
        False,
    )


async def read_lines_hashed_async(
    path: Path,
    *,
    start_line: int = 1,
    limit: int,
    max_bytes: int,
    full_scan: bool = False,
    max_line_bytes: int | None = None,
) -> WindowReadResult:
    """Async :func:`read_lines_hashed` (runs the blocking read in a thread)."""
    return await asyncio.to_thread(
        read_lines_hashed,
        path,
        start_line=start_line,
        limit=limit,
        max_bytes=max_bytes,
        full_scan=full_scan,
        max_line_bytes=max_line_bytes,
    )


_FILE_WRITE_LOCKS: dict[str, asyncio.Lock] = {}
_FILE_WRITE_LOCK_LOOP: asyncio.AbstractEventLoop | None = None


def _get_lock(path: Path) -> asyncio.Lock:
    global _FILE_WRITE_LOCK_LOOP
    loop = asyncio.get_running_loop()
    if _FILE_WRITE_LOCK_LOOP is not loop:
        _FILE_WRITE_LOCKS.clear()
        _FILE_WRITE_LOCK_LOOP = loop
    key = str(path.resolve())
    lock = _FILE_WRITE_LOCKS.get(key)
    if lock is None:
        lock = _FILE_WRITE_LOCKS[key] = asyncio.Lock()
    return lock


@asynccontextmanager
async def file_write_lock(path: Path) -> AsyncIterator[None]:
    async with _get_lock(path):
        yield


async def atomic_replace(
    path: Path, content: str, *, encoding: str = "utf-8", newline: str | None = None
) -> None:
    target = Path(path)
    tmp = target.parent / f".{target.name}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        async with await anyio.Path(tmp).open(
            mode="w", encoding=encoding, newline=newline
        ) as f:
            await f.write(content)
        with contextlib.suppress(FileNotFoundError):
            shutil.copymode(target, tmp)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
