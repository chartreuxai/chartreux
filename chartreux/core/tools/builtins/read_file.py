from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, final

from humanize import naturalsize
from pydantic import BaseModel, Field

from chartreux.core.events import ToolStreamEvent
from chartreux.core.file_revision import (
    ReadTraversal,
    TraversalRegistry,
    new_traversal_id,
)
from chartreux.core.scratchpad import is_scratchpad_display_path
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.io_port import ToolIOPort
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from chartreux.core.tools.utils import (
    DEFAULT_SENSITIVE_PATTERNS,
    ToolPath,
    display_file_path,
    resolve_file_tool_permission,
    resolve_tool_path,
)
from chartreux.utils import VIBE_WARNING_TAG
from chartreux.utils.io import WindowReadResult, read_lines_hashed_async
from chartreux.utils.tool_presentation import ToolEffectKind

if TYPE_CHECKING:
    from chartreux.core.events import ToolResultEvent

_KB = 1024
DEFAULT_LINE_LIMIT = 2000
MAX_BYTES = 50 * _KB
# Continuation contract: one traversal chunk targets 32 KiB of rendered
# output (line-number prefix included), well under the 50 KiB output cap, so
# the traversal notice always fits inside the cap as well.
CHUNK_TARGET_BYTES = 32 * _KB
# Rendered overhead per line: 9-digit right-justified number, the U+2192 arrow
# (3 bytes in UTF-8), and the newline joiner.
_LINE_PREFIX_BYTES = 13
# Reserved room for the traversal notice block / the mid-line continuation
# marker when sizing chunks and slices.
_NOTICE_ALLOWANCE = 512
_LONG_LINE_MARKER_ALLOWANCE = 160
# Byte budget for one slice of a line longer than a whole chunk.
_LINE_SLICE_BUDGET = (
    CHUNK_TARGET_BYTES - _LINE_PREFIX_BYTES - _LONG_LINE_MARKER_ALLOWANCE
)


def _add_line_numbers(lines: list[str], *, start: int) -> str:
    return "\n".join(
        f"{str(n).rjust(9)}\u2192{line}" for n, line in enumerate(lines, start=start)
    )


def _warning(message: str) -> str:
    return f"<{VIBE_WARNING_TAG}>{message}</{VIBE_WARNING_TAG}>"


def _display_relative(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def _rendered_line_bytes(line: str) -> int:
    """Rendered size of one numbered output line, in bytes."""
    return len(line.encode("utf-8")) + _LINE_PREFIX_BYTES


def _select_chunk(
    lines: list[str],
    *,
    start_line: int,
    was_truncated: bool,
    last_line_may_be_partial: bool,
) -> tuple[list[str], int | None]:
    r"""Pick the next whole-line chunk within the rendered-output budget.

    Returns ``(chunk, next_offset)``; ``next_offset`` is ``None`` when the
    chunk runs through end of file. A window tail line that the transport may
    have cut mid-line is never served: it starts the next chunk instead, so no
    content is silently skipped or duplicated.
    """
    eligible = len(lines)
    if last_line_may_be_partial and was_truncated and eligible > 1:
        eligible -= 1
    chunk: list[str] = []
    used = 0
    for line in lines[:eligible]:
        cost = _rendered_line_bytes(line)
        if chunk and used + cost > CHUNK_TARGET_BYTES:
            break
        if not chunk and cost > CHUNK_TARGET_BYTES:
            break  # oversized single line: the caller slices it explicitly
        chunk.append(line)
        used += cost
    if not chunk:
        return [], None
    if len(chunk) >= eligible and not was_truncated:
        return chunk, None
    return chunk, start_line + len(chunk)


def _slice_line(line: str, *, start_chars: int, budget: int) -> tuple[str, bool]:
    """Return ``(slice, complete)`` for up to ``budget`` UTF-8 bytes of a line.

    The slice starts at ``start_chars`` and never splits a character, so no
    content is lost at a slice boundary.
    """
    remaining = line[start_chars:]
    if len(remaining.encode("utf-8")) <= budget:
        return remaining, True
    low, high = 0, len(remaining)
    while low < high:
        mid = (low + high + 1) // 2
        if len(remaining[:mid].encode("utf-8")) <= budget:
            low = mid
        else:
            high = mid - 1
    return remaining[:low], False


def _traversal_notice(
    *,
    total_lines: int | None,
    start_line: int,
    end_line: int,
    next_offset: int | None,
    traversal_id: str,
    first_chunk: bool,
    extra_note: str | None,
    line_continues: bool,
) -> str:
    """Continuation notice prepended to a traversal chunk's content."""
    parts: list[str] = []
    if extra_note:
        parts.append(extra_note)
    total = f" of {total_lines}" if total_lines is not None else ""
    if line_continues:
        parts.append(
            f"Line {start_line} is longer than one chunk; this chunk ends "
            f"mid-line. Continue with offset={start_line} and "
            f"traversal_id={traversal_id} to read the rest of line {start_line}."
        )
    elif next_offset is None:
        parts.append(
            f"Lines {start_line}\u2013{end_line}{total}: end of file; the "
            f"traversal is complete."
        )
    else:
        opening = (
            "This is the first chunk of a multi-part read."
            if first_chunk
            else "Continuing the multi-part read."
        )
        parts.append(
            f"{opening} Lines {start_line}\u2013{end_line}{total}. Continue "
            f"with offset={next_offset} and traversal_id={traversal_id} for "
            f"the next chunk; the traversal ends when no next_offset is "
            f"returned."
        )
    return _warning(" ".join(parts))


class _TraversalContext(NamedTuple):
    """Continuation state carried into one chunk of a traversal."""

    traversal_id: str | None
    opening_offset: int
    opening_revision: str | None
    progress: int | None
    restarted: bool
    first_chunk: bool
    extra_note: str | None


class ReadFileArgs(BaseModel):
    file_path: ToolPath = Field(
        description=(
            "The absolute path to the file to read. Paths outside the session's authorized roots "
            "may still be attempted: in interactive sessions the runtime asks the user to grant "
            "access and the operation proceeds on approval; the tool result states when access is "
            "unavailable or was declined, and only then is config or manual fallback appropriate."
        )
    )
    offset: int | None = Field(
        default=None,
        ge=1,
        description="The line number to start reading from (1-indexed)",
    )
    limit: int = Field(
        default=DEFAULT_LINE_LIMIT,
        gt=0,
        description="The maximum number of lines to read",
    )
    traversal_id: str | None = Field(
        default=None,
        description=(
            "Continuation token returned by a previous oversized read of this "
            "file. Pass it together with offset=<next_offset> to continue the "
            "same traversal; omit it for a fresh read."
        ),
    )


class ReadFileResult(BaseModel):
    file_path: str
    content: str
    num_lines: int
    start_line: int
    requested_offset: int | None = None
    requested_limit: int = DEFAULT_LINE_LIMIT
    total_lines: int | None = None
    was_truncated: bool = False
    end_line: int | None = None
    next_offset: int | None = None
    traversal_id: str | None = None
    revision: str | None = None
    total_chunks: int | None = None
    restarted: bool = False
    line_continues: bool = False


class ReadFileConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    sensitive_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SENSITIVE_PATTERNS),
        description="File patterns that deny execution even when permission is ALWAYS.",
    )
    max_read_bytes: int = Field(
        default=MAX_BYTES,
        gt=0,
        description="Maximum selected/output bytes to return in one call.",
    )


class ReadFileState(BaseToolState):
    injected_agents_md: set[str] = Field(default_factory=set)
    traversals: TraversalRegistry = Field(default_factory=TraversalRegistry)


class ReadFile(
    BaseTool[ReadFileArgs, ReadFileResult, ReadFileConfig, ReadFileState],
    ToolUIData[ReadFileArgs, ReadFileResult],
):
    effect_kind = ToolEffectKind.FILE_READ

    def resolve_permission(self, args: ReadFileArgs) -> PermissionContext | None:
        return resolve_file_tool_permission(
            args.file_path,
            tool_name=self.get_name(),
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
            config_permission=self.config.permission,
            sensitive_patterns=self.config.sensitive_patterns,
            workspace=self.workspace,
            scratchpad_roots=self.scratchpad_roots,
            instruction_read_files=self.instruction_read_files,
        )

    def _find_undiscovered_agents_md(self, file_path: Path) -> list[tuple[Path, str]]:
        docs = self.harness_files.find_subdirectory_agents_md(file_path)
        return [
            (d, c)
            for d, c in docs
            if str(d.resolve()) not in self.state.injected_agents_md
        ]

    def get_result_extra(self, result: ReadFileResult) -> str | None:
        new_docs = self._find_undiscovered_agents_md(Path(result.file_path))
        if not new_docs:
            return None
        for d, _ in new_docs:
            self.state.injected_agents_md.add(str(d.resolve()))
        sections = [
            f"Contents of {d}/AGENTS.md (project instructions for this directory):\n\n{c.strip()}"
            for d, c in new_docs
        ]
        return f"<{VIBE_WARNING_TAG}>\n{'\n\n'.join(sections)}\n</{VIBE_WARNING_TAG}>"

    async def _read_window(
        self,
        file_path: Path,
        *,
        start_line: int,
        limit: int,
        tool_io: ToolIOPort | None,
        full_scan: bool,
    ) -> WindowReadResult:
        """Read one window through the tool's own read path.

        Client-mediated reads (tool I/O port) cannot provide a full-state
        content revision, so their result carries ``revision=None``; the
        traversal then runs without change detection (a documented
        degradation, not an error).
        """
        try:
            if tool_io is not None and tool_io.supports_read:
                result = await tool_io.read_lines(
                    file_path,
                    start_line=start_line,
                    limit=limit,
                    max_bytes=self.config.max_read_bytes,
                )
                return WindowReadResult(
                    lines=result.lines,
                    total_lines=result.total_lines,
                    was_truncated=result.was_truncated,
                    revision=None,
                    raw_bytes=None,
                    first_line_truncated=False,
                    last_line_may_be_partial=True,
                )
            return await read_lines_hashed_async(
                file_path,
                start_line=start_line,
                limit=limit,
                max_bytes=self.config.max_read_bytes,
                full_scan=full_scan,
            )
        except Exception as exc:
            raise ToolError(f"Error reading {file_path}: {exc}") from exc

    def _render_window(self, read: WindowReadResult, start_line: int) -> str:
        """Render a window (or its empty/offset warnings) as result content."""
        if read.lines:
            return _add_line_numbers(read.lines, start=start_line)
        if read.total_lines == 0:
            return _warning("Warning: the file exists but the contents are empty.")
        if read.total_lines is None:
            return _warning(f"Warning: no content returned for offset {start_line}.")
        return _warning(
            f"Warning: the file exists but is shorter than the provided "
            f"offset ({start_line}). The file has {read.total_lines} lines."
        )

    def _check_output_size(self, content: str) -> None:
        size = len(content.encode("utf-8"))
        if size > self.config.max_read_bytes:
            raise ToolError(
                f"Output ({naturalsize(size, binary=True)}) exceeds maximum "
                f"allowed size ({naturalsize(self.config.max_read_bytes, binary=True)}). "
                f"Use offset and limit to read a smaller portion of the file."
            )

    def _register_traversal(
        self,
        *,
        traversal_id: str,
        file_path: Path,
        opening_offset: int,
        opening_revision: str | None,
        next_offset: int | None,
        in_progress_line: int | None,
        in_progress_chars: int,
    ) -> None:
        if next_offset is None:
            # The traversal reached end of file; drop its record.
            self.state.traversals.discard(traversal_id)
            return
        self.state.traversals.open(
            ReadTraversal(
                traversal_id=traversal_id,
                file_path=str(file_path),
                opening_offset=opening_offset,
                opening_revision=opening_revision,
                next_offset=next_offset,
                in_progress_line=in_progress_line,
                in_progress_chars=in_progress_chars,
            )
        )

    async def _read_request(
        self, args: ReadFileArgs, file_path: Path, tool_io: ToolIOPort | None
    ) -> ReadFileResult:
        """Serve a fresh (non-continuation) read request."""
        start_line = args.offset or 1
        explicit = "offset" in args.model_fields_set or "limit" in args.model_fields_set
        read = await self._read_window(
            file_path,
            start_line=start_line,
            limit=args.limit,
            tool_io=tool_io,
            full_scan=False,
        )
        if explicit:
            # A bounded request is honored exactly as given; an oversized
            # bounded read is an error, never a continuation traversal.
            content = self._render_window(read, start_line)
            self._check_output_size(content)
            return ReadFileResult(
                file_path=str(file_path),
                content=content,
                num_lines=len(read.lines),
                start_line=start_line,
                requested_offset=args.offset,
                requested_limit=args.limit,
                total_lines=read.total_lines,
                was_truncated=read.was_truncated,
                end_line=start_line + len(read.lines) - 1 if read.lines else None,
                revision=read.revision,
            )
        if read.total_lines == 0:
            return ReadFileResult(
                file_path=str(file_path),
                content=_warning(
                    "Warning: the file exists but the contents are empty."
                ),
                num_lines=0,
                start_line=1,
                requested_offset=None,
                requested_limit=args.limit,
                total_lines=0,
                was_truncated=False,
                revision=read.revision,
            )
        content = _add_line_numbers(read.lines, start=1)
        if (
            not read.was_truncated
            and len(content.encode("utf-8")) <= self.config.max_read_bytes
        ):
            # The whole file fits in one result.
            return ReadFileResult(
                file_path=str(file_path),
                content=content,
                num_lines=len(read.lines),
                start_line=1,
                requested_offset=None,
                requested_limit=args.limit,
                total_lines=read.total_lines,
                was_truncated=False,
                end_line=len(read.lines),
                revision=read.revision,
            )
        # Oversized (or line-capped) unbounded read: serve it as one
        # intentional traversal instead of an error.
        return await self._begin_traversal(
            args,
            file_path,
            tool_io,
            start_line=1,
            ctx=_TraversalContext(
                traversal_id=None,
                opening_offset=1,
                opening_revision=None,
                progress=None,
                restarted=False,
                first_chunk=True,
                extra_note=None,
            ),
        )

    async def _begin_traversal(
        self,
        args: ReadFileArgs,
        file_path: Path,
        tool_io: ToolIOPort | None,
        *,
        start_line: int,
        ctx: _TraversalContext,
    ) -> ReadFileResult:
        """Open a traversal: full scan for the revision, then serve a chunk."""
        read = await self._read_window(
            file_path,
            start_line=start_line,
            limit=args.limit,
            tool_io=tool_io,
            full_scan=True,
        )
        return self._serve_chunk(
            args,
            file_path,
            read,
            start_line=start_line,
            ctx=ctx._replace(opening_revision=read.revision),
        )

    async def _continue_traversal(
        self,
        args: ReadFileArgs,
        file_path: Path,
        tool_io: ToolIOPort | None,
        *,
        traversal_id: str,
    ) -> ReadFileResult:
        """Serve one continuation chunk of an in-progress traversal."""
        record = self.state.traversals.get(traversal_id)
        if record is None or record.file_path != str(file_path):
            start_line = args.offset or 1
            return await self._begin_traversal(
                args,
                file_path,
                tool_io,
                start_line=start_line,
                ctx=_TraversalContext(
                    traversal_id=None,
                    opening_offset=start_line,
                    opening_revision=None,
                    progress=None,
                    restarted=False,
                    first_chunk=True,
                    extra_note=(
                        f"Unknown or expired traversal_id {traversal_id!r}; "
                        f"starting a new traversal from line {start_line}."
                    ),
                ),
            )
        start_line = args.offset or record.next_offset
        progress = (
            record.in_progress_chars
            if record.in_progress_line == start_line and record.in_progress_chars > 0
            else None
        )
        read = await self._read_window(
            file_path,
            start_line=start_line,
            limit=args.limit,
            tool_io=tool_io,
            full_scan=True,
        )
        if (
            record.opening_revision is not None
            and read.revision is not None
            and read.revision != record.opening_revision
        ):
            # The content changed under the traversal: restart it from its
            # opening offset with the revision now observed.
            restart_read = await self._read_window(
                file_path,
                start_line=record.opening_offset,
                limit=args.limit,
                tool_io=tool_io,
                full_scan=True,
            )
            return self._serve_chunk(
                args,
                file_path,
                restart_read,
                start_line=record.opening_offset,
                ctx=_TraversalContext(
                    traversal_id=traversal_id,
                    opening_offset=record.opening_offset,
                    opening_revision=restart_read.revision,
                    progress=None,
                    restarted=True,
                    first_chunk=False,
                    extra_note=(
                        f"The file changed during this traversal (content "
                        f"revision {record.opening_revision} \u2192 "
                        f"{read.revision}); beginning again from line "
                        f"{record.opening_offset}."
                    ),
                ),
            )
        return self._serve_chunk(
            args,
            file_path,
            read,
            start_line=start_line,
            ctx=_TraversalContext(
                traversal_id=traversal_id,
                opening_offset=record.opening_offset,
                opening_revision=record.opening_revision,
                progress=progress,
                restarted=False,
                first_chunk=False,
                extra_note=None,
            ),
        )

    def _serve_chunk(
        self,
        args: ReadFileArgs,
        file_path: Path,
        read: WindowReadResult,
        *,
        start_line: int,
        ctx: _TraversalContext,
    ) -> ReadFileResult:
        """Serve one whole-line chunk of a traversal and advance its state."""
        lines = read.lines
        if not lines:
            # A continuation landed past end of file: the traversal is over.
            self.state.traversals.discard(ctx.traversal_id or "")
            return ReadFileResult(
                file_path=str(file_path),
                content=_warning(
                    f"End of file: no content at offset {start_line}; the "
                    f"traversal is complete."
                ),
                num_lines=0,
                start_line=start_line,
                requested_offset=args.offset,
                requested_limit=args.limit,
                total_lines=read.total_lines,
                was_truncated=False,
                next_offset=None,
                traversal_id=ctx.traversal_id,
                revision=read.revision,
                restarted=ctx.restarted,
            )
        traversal_id = ctx.traversal_id or new_traversal_id()
        if (
            _rendered_line_bytes(lines[0]) > CHUNK_TARGET_BYTES - _NOTICE_ALLOWANCE
            or ctx.progress is not None
        ):
            # Single line longer than the chunk budget (or mid-line
            # continuation): slice it explicitly, never loop, never skip
            # silently.
            return self._serve_line_slice(
                args,
                file_path,
                read,
                start_line=start_line,
                ctx=ctx._replace(traversal_id=traversal_id),
            )
        chunk, next_offset = _select_chunk(
            lines,
            start_line=start_line,
            was_truncated=read.was_truncated,
            last_line_may_be_partial=read.last_line_may_be_partial,
        )
        end_line = start_line + len(chunk) - 1
        notice = _traversal_notice(
            total_lines=read.total_lines,
            start_line=start_line,
            end_line=end_line,
            next_offset=next_offset,
            traversal_id=traversal_id,
            first_chunk=ctx.first_chunk,
            extra_note=ctx.extra_note,
            line_continues=False,
        )
        content = f"{notice}\n{_add_line_numbers(chunk, start=start_line)}"
        self._check_output_size(content)
        self._register_traversal(
            traversal_id=traversal_id,
            file_path=file_path,
            opening_offset=ctx.opening_offset,
            opening_revision=ctx.opening_revision,
            next_offset=next_offset,
            in_progress_line=None,
            in_progress_chars=0,
        )
        return ReadFileResult(
            file_path=str(file_path),
            content=content,
            num_lines=len(chunk),
            start_line=start_line,
            requested_offset=args.offset,
            requested_limit=args.limit,
            total_lines=read.total_lines,
            was_truncated=next_offset is not None,
            end_line=end_line,
            next_offset=next_offset,
            traversal_id=traversal_id,
            revision=read.revision,
            total_chunks=None,
            restarted=ctx.restarted,
            line_continues=False,
        )

    def _serve_line_slice(
        self,
        args: ReadFileArgs,
        file_path: Path,
        read: WindowReadResult,
        *,
        start_line: int,
        ctx: _TraversalContext,
    ) -> ReadFileResult:
        """Serve one slice of a line longer than the chunk budget."""
        traversal_id = ctx.traversal_id or new_traversal_id()
        full_line = read.lines[0]
        progress = ctx.progress or 0
        text, complete = _slice_line(
            full_line, start_chars=progress, budget=_LINE_SLICE_BUDGET
        )
        elision: str | None = None
        if complete:
            marker = ""
            next_offset = (
                start_line + 1 if read.was_truncated or len(read.lines) > 1 else None
            )
            in_progress_line: int | None = None
            in_progress_chars = 0
            line_continues = False
            if read.first_line_truncated:
                elision = (
                    f"Line {start_line} is longer than this tool reads in one "
                    f"pass; content beyond the read limit was elided."
                )
        else:
            rest_bytes = len(full_line[progress + len(text) :].encode("utf-8"))
            marker = (
                f" \u2026[+{rest_bytes} bytes in this line; continue with "
                f"offset={start_line} and traversal_id={traversal_id} for the "
                f"rest of line {start_line}]"
            )
            next_offset = start_line
            in_progress_line = start_line
            in_progress_chars = progress + len(text)
            line_continues = True
        extra_note = " ".join(part for part in (ctx.extra_note, elision) if part)
        extra_note = extra_note or None
        notice = _traversal_notice(
            total_lines=read.total_lines,
            start_line=start_line,
            end_line=start_line,
            next_offset=next_offset,
            traversal_id=traversal_id,
            first_chunk=ctx.first_chunk,
            extra_note=extra_note,
            line_continues=line_continues,
        )
        content = f"{notice}\n{_add_line_numbers([text + marker], start=start_line)}"
        self._check_output_size(content)
        self._register_traversal(
            traversal_id=traversal_id,
            file_path=file_path,
            opening_offset=ctx.opening_offset,
            opening_revision=ctx.opening_revision,
            next_offset=next_offset,
            in_progress_line=in_progress_line,
            in_progress_chars=in_progress_chars,
        )
        return ReadFileResult(
            file_path=str(file_path),
            content=content,
            num_lines=1,
            start_line=start_line,
            requested_offset=args.offset,
            requested_limit=args.limit,
            total_lines=read.total_lines,
            was_truncated=next_offset is not None,
            end_line=start_line,
            next_offset=next_offset,
            traversal_id=traversal_id,
            revision=read.revision,
            total_chunks=None,
            restarted=ctx.restarted,
            line_continues=line_continues,
        )

    @final
    async def run(
        self, args: ReadFileArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | ReadFileResult, None]:
        file_path = self._resolve_path(args.file_path)
        tool_io = ctx.tool_io if ctx is not None else None

        if args.traversal_id is not None:
            result = await self._continue_traversal(
                args, file_path, tool_io, traversal_id=args.traversal_id
            )
        else:
            result = await self._read_request(args, file_path, tool_io)

        if ctx is not None:
            new_docs = self._find_undiscovered_agents_md(file_path)
            if new_docs:
                parts = [
                    f"{_display_relative(d, self.cwd)}/AGENTS.md" for d, _ in new_docs
                ]
                yield ToolStreamEvent(
                    tool_name=self.get_name(),
                    tool_call_id=ctx.tool_call_id,
                    message=f"Discovered {', '.join(parts)}",
                )

        yield result

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ToolError("file_path cannot be empty")

        path = resolve_tool_path(raw_path, self.cwd)

        if not path.exists():
            raise ToolError(f"File not found at: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory, not a file: {path}")
        return path

    @classmethod
    def format_call_display(cls, args: ReadFileArgs) -> ToolCallDisplay:
        suffix = "(scratchpad)" if is_scratchpad_display_path(args.file_path) else ""
        message = display_file_path(args.file_path)
        extras: list[str] = []
        if args.offset:
            extras.append(f"from line {args.offset}")
        if args.limit != DEFAULT_LINE_LIMIT:
            extras.append(f"limit {args.limit} lines")
        if args.traversal_id:
            extras.append("continuing traversal")
        if extras:
            message += f" ({', '.join(extras)})"
        return ToolCallDisplay(
            summary=f"Reading {message}",
            suffix=suffix,
            verb="Reading",
            message=message,
            settled_verb="Read",
            settled_message=message,
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ReadFileResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        n = event.result.num_lines
        word = "line" if n == 1 else "lines"
        message = f"{n} {word} from {display_file_path(event.result.file_path)}"
        suffix_parts: list[str] = []
        if is_scratchpad_display_path(event.result.file_path):
            suffix_parts.append("(scratchpad)")
        if event.result.was_truncated or (
            event.result.total_lines is not None
            and event.result.start_line + event.result.num_lines - 1
            < event.result.total_lines
        ):
            suffix_parts.append("(truncated)")

        return ToolResultDisplay(
            success=True, verb="Read", message=message, suffix=" ".join(suffix_parts)
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Reading file"
