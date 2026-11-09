from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import NamedTuple, final

from pydantic import BaseModel, Field, JsonValue, PrivateAttr, field_validator

from chartreux.core.checkpoints import FileSnapshot
from chartreux.core.events import ToolResultEvent, ToolStreamEvent
from chartreux.core.file_revision import FileRevision
from chartreux.core.scratchpad import is_scratchpad_display_path
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.builtins._edit_result_context import (
    EDIT_DIFF_MAX_SITES,
    AppliedSite,
    EditEvidence,
    _expand_whole_lines,
    _read_post_edit_revision,
    build_edit_evidence,
    render_applied_diff,
    render_context_window,
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
from chartreux.utils.io import (
    ReadSafeResult,
    atomic_replace,
    file_write_lock,
    read_safe_async,
)
from chartreux.utils.text import line_contexts
from chartreux.utils.tool_presentation import ToolEffectKind


class AnchorEdit(BaseModel):
    """One ordered replacement in the multi-anchor edit mode."""

    old_string: str = Field(
        description="The text this anchor matches; it must occur exactly once in the file"
    )
    new_string: str = Field(description="The replacement for this anchor")


class EditArgs(BaseModel):
    file_path: ToolPath = Field(
        description=(
            "The absolute path to the file to modify. Paths outside the session's authorized "
            "roots may still be attempted: in interactive sessions the runtime asks the user to "
            "grant access and the operation proceeds on approval; the tool result states when "
            "access is unavailable or was declined, and only then is config or manual fallback "
            "appropriate."
        )
    )
    old_string: str | None = Field(
        default=None,
        description=(
            "The text to replace; may be empty only when populating an empty "
            "existing file. Omit when using multi_anchor"
        ),
    )
    new_string: str | None = Field(
        default=None,
        description=(
            "The text to replace it with (must be different from old_string). "
            "Omit when using multi_anchor"
        ),
    )
    replace_all: bool = Field(
        default=False,
        description=(
            "Replace all occurrences of old_string (default false). Not valid "
            "together with multi_anchor"
        ),
    )
    expected_site_count: int | None = Field(
        default=None,
        description=(
            "Requires replace_all=true: the anchor must match exactly this many "
            "sites; any other count writes nothing and returns a count-mismatch "
            "error naming the first sites by line number. Omitted means no check"
        ),
    )
    multi_anchor: list[AnchorEdit] | None = Field(
        default=None,
        description=(
            "Ordered list of replacements, used instead of old_string/new_string: "
            "each anchor must match exactly once, in file order, non-overlapping. "
            "All anchors are validated before any write and the assembled result "
            "lands in one atomic write. Not valid with replace_all=true or "
            "expected_site_count"
        ),
    )

    @field_validator("file_path")
    @classmethod
    def normalize_file_path(cls, value: ToolPath) -> ToolPath:
        normalized = value.strip()
        if value and not normalized:
            raise ValueError("File path cannot be empty")
        return normalized


class EditResult(BaseModel):
    file: str
    message: str
    old_string: str
    new_string: str
    # Post-edit evidence, computed from the file as it stands after the
    # applied edit and serialized to the model (see _edit_result_context):
    # applied_diff renders each replaced site as whole lines (old, then new)
    # under its current line number; context is a numbered window of the
    # current content around the first site; revision identifies the
    # post-edit file state (None for client-mediated writes); replaced_count
    # is the number of applied replacement sites.
    applied_diff: str | None = None
    context: str | None = None
    revision: FileRevision | None = None
    replaced_count: int = 0
    # UI hint for the diff renderer; not part of the serialized result contract.
    # One entry per replaced occurrence (replace_all yields several), each as
    # (start_line, old_lines, new_lines) where old/new are the changed snippet
    # expanded to whole lines so the diff shows full lines per occurrence.
    _ui_occurrences: list[tuple[int, str, str]] = PrivateAttr(default_factory=list)

    @property
    def ui_start_lines(self) -> list[int]:
        return [start for start, _, _ in self._ui_occurrences]

    @property
    def ui_occurrences(self) -> list[tuple[int | None, str, str]]:
        if self._ui_occurrences:
            return list(self._ui_occurrences)
        return [(None, self.old_string, self.new_string)]


class EditConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    sensitive_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SENSITIVE_PATTERNS),
        description="File patterns that deny execution even when permission is ALWAYS.",
    )


class _EditPlan(NamedTuple):
    """The validated outcome of one edit call, before anything is written.

    ``sites`` is populated only for a multi-anchor edit, whose evidence is
    rendered from those sites; ``contexts`` are the UI occurrence hints as
    ``(start_line, old_text, new_text)``.
    """

    modified: str
    old_string: str
    new_string: str
    replace_all: bool
    sites: list[AppliedSite] | None
    contexts: list[tuple[int, str, str]]


def _occurrence_positions(content: str, snippet: str) -> list[int]:
    """Non-overlapping occurrence offsets of ``snippet``, mirroring ``str.replace``."""
    if not snippet:
        return []
    positions: list[int] = []
    position = content.find(snippet)
    while position != -1:
        positions.append(position)
        position = content.find(snippet, position + len(snippet))
    return positions


def _bounded_line_numbers(lines: list[int]) -> str:
    """Render line numbers capped at the tuned site bound, with an explicit remainder."""
    shown = lines[:EDIT_DIFF_MAX_SITES]
    text = ", ".join(str(line) for line in shown)
    omitted = len(lines) - len(shown)
    if omitted:
        text += f" (+{omitted} more)"
    return text


class Edit(
    BaseTool[EditArgs, EditResult, EditConfig, BaseToolState],
    ToolUIData[EditArgs, EditResult],
):
    effect_kind = ToolEffectKind.FILE_EDIT

    def resolve_permission(self, args: EditArgs) -> PermissionContext | None:
        return resolve_file_tool_permission(
            args.file_path,
            tool_name=self.get_name(),
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
            config_permission=self.config.permission,
            sensitive_patterns=self.config.sensitive_patterns,
            workspace=self.workspace,
            scratchpad_dir=self.owned_scratchpad_dir,
            scratchpad_roots=self.scratchpad_roots,
            instruction_read_files=self.instruction_read_files,
        )

    def get_file_snapshot(self, args: EditArgs) -> FileSnapshot | None:
        return self.get_file_snapshot_for_path(args.file_path)

    @classmethod
    def format_call_display(cls, args: EditArgs) -> ToolCallDisplay:
        suffix = "(scratchpad)" if is_scratchpad_display_path(args.file_path) else ""
        shown = display_file_path(args.file_path)
        if args.multi_anchor is not None:
            content = "\n".join(
                f"multi_anchor[{index}]: {anchor.old_string!r} -> {anchor.new_string!r}"
                for index, anchor in enumerate(args.multi_anchor)
            )
        else:
            content = (
                f"old_string: {args.old_string!r}\nnew_string: {args.new_string!r}"
            )
        return ToolCallDisplay(
            summary=f"Editing {shown}",
            content=content,
            suffix=suffix,
            verb="Editing",
            message=shown,
            settled_verb="Edited",
            settled_message=shown,
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, EditResult):
            suffix = (
                "(scratchpad)" if is_scratchpad_display_path(event.result.file) else ""
            )
            return ToolResultDisplay(
                success=True,
                verb="Edited",
                message=display_file_path(event.result.file),
                suffix=suffix,
            )
        return ToolResultDisplay(
            success=False, message=event.error or event.skip_reason or "No result"
        )

    @classmethod
    def project_result(cls, result: EditResult) -> JsonValue:
        return {
            "file": result.file,
            "old_string": result.old_string,
            "new_string": result.new_string,
            "occurrences": [
                {"start_line": start_line, "old_text": old_text, "new_text": new_text}
                for start_line, old_text, new_text in result.ui_occurrences
            ],
        }

    @classmethod
    def get_status_text(cls) -> str:
        return "Editing files"

    async def _read_file(
        self, file_path: Path, tool_io: ToolIOPort | None = None
    ) -> ReadSafeResult:
        if tool_io is not None and tool_io.supports_read:
            return await tool_io.read_text(file_path)
        return await read_safe_async(file_path, raise_on_error=True)

    async def _write_file(
        self,
        file_path: Path,
        content: str,
        encoding: str,
        newline: str,
        tool_io: ToolIOPort | None = None,
    ) -> None:
        if tool_io is not None and tool_io.supports_write:
            await tool_io.write_text(
                file_path, content, encoding=encoding, newline=newline
            )
            return
        await atomic_replace(file_path, content, encoding=encoding, newline=newline)

    @final
    async def run(
        self, args: EditArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | EditResult, None]:
        file_path = self._validate_args(args)
        tool_io = ctx.tool_io if ctx is not None else None
        host_written = tool_io is None or not tool_io.supports_write

        try:
            async with file_write_lock(file_path):
                read = await self._read_file(file_path, tool_io)
                original = read.text

                plan = self._plan_edit(original, args)

                if plan.modified != original:
                    await self._write_file(
                        file_path, plan.modified, read.encoding, read.newline, tool_io
                    )

                if plan.sites is not None:
                    evidence = await self._multi_anchor_evidence(
                        file_path=file_path,
                        modified=plan.modified,
                        sites=plan.sites,
                        host_written=host_written,
                    )
                else:
                    evidence = await build_edit_evidence(
                        file_path=file_path,
                        original=original,
                        modified=plan.modified,
                        old_string=plan.old_string,
                        new_string=plan.new_string,
                        replace_all=plan.replace_all,
                        host_written=host_written,
                    )
        except UnicodeDecodeError as e:
            raise ToolError(
                f"Cannot edit {file_path}: file is not valid text "
                f"({e.encoding}, byte {e.start})"
            ) from e
        except PermissionError as e:
            raise ToolError(f"Permission denied accessing file: {file_path}") from e
        except OSError as e:
            raise ToolError(f"OS error accessing {file_path}: {e}") from e

        if args.multi_anchor is not None:
            message = (
                "The file has been updated. Every anchor was replaced exactly once"
            )
        elif args.replace_all:
            message = (
                "The file has been updated. All occurrences were successfully replaced"
            )
        else:
            message = "The file has been updated successfully."

        result = EditResult(
            file=str(file_path),
            message=message,
            old_string=plan.old_string,
            new_string=plan.new_string,
            applied_diff=evidence.applied_diff,
            context=evidence.context,
            revision=evidence.revision,
            replaced_count=evidence.replaced_count,
        )
        result._ui_occurrences = plan.contexts
        yield result

    @final
    def _validate_args(self, args: EditArgs) -> Path:
        if not args.file_path:
            raise ToolError("File path cannot be empty")

        self._validate_edit_mode(args)

        if args.multi_anchor is None and args.old_string == args.new_string:
            raise ToolError(
                "No changes to make — old_string and new_string are identical"
            )

        file_path = resolve_tool_path(args.file_path, self.cwd)

        if not file_path.exists():
            raise ToolError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise ToolError(f"Path is not a file: {file_path}")

        return file_path

    @staticmethod
    @final
    def _validate_edit_mode(args: EditArgs) -> None:
        """Enforce the mode matrix before the file is touched.

        multi_anchor is mutually exclusive with replace_all and
        expected_site_count, and expected_site_count composes only with
        replace_all=true. Every invalid combination is a parameter error.
        """
        if args.multi_anchor is not None:
            if args.old_string is not None or args.new_string is not None:
                raise ToolError(
                    "multi_anchor replaces old_string/new_string: pass every "
                    "replacement as a multi_anchor entry and omit old_string and "
                    "new_string"
                )
            if args.replace_all:
                raise ToolError(
                    "multi_anchor cannot be combined with replace_all=true; each "
                    "anchor is replaced exactly once"
                )
            if args.expected_site_count is not None:
                raise ToolError(
                    "multi_anchor cannot be combined with expected_site_count; each "
                    "anchor must match exactly once"
                )
            if not args.multi_anchor:
                raise ToolError("multi_anchor must contain at least one anchor")
            for index, anchor in enumerate(args.multi_anchor):
                if not anchor.old_string:
                    raise ToolError(f"multi_anchor[{index}] old_string cannot be empty")
                if anchor.old_string == anchor.new_string:
                    raise ToolError(
                        f"multi_anchor[{index}] makes no change: old_string and "
                        f"new_string are identical"
                    )
            return

        if args.old_string is None or args.new_string is None:
            raise ToolError(
                "old_string and new_string are required unless multi_anchor is used"
            )
        if args.expected_site_count is not None:
            if not args.replace_all:
                raise ToolError(
                    "expected_site_count requires replace_all=true; without it the "
                    "edit replaces only the first occurrence"
                )
            if args.expected_site_count < 1:
                raise ToolError("expected_site_count must be at least 1")

    def _plan_edit(self, original: str, args: EditArgs) -> _EditPlan:
        """Validate the whole edit and compute its outcome without writing."""
        if args.multi_anchor is not None:
            modified, sites = self._apply_multi_anchor(original, args.multi_anchor)
            return _EditPlan(
                modified=modified,
                old_string="",
                new_string="",
                replace_all=False,
                sites=sites,
                contexts=[(site.line, site.old_text, site.new_text) for site in sites],
            )

        # _validate_edit_mode guarantees both strings outside multi-anchor mode.
        old_string = args.old_string if args.old_string is not None else ""
        new_string = args.new_string if args.new_string is not None else ""

        if not old_string and original:
            raise ToolError(
                "old_string can be empty only when populating an empty file"
            )
        if old_string not in original:
            raise ToolError(
                f"String to replace not found in file.\nString: {old_string}"
            )
        occurrences = original.count(old_string)
        positions = _occurrence_positions(original, old_string)
        if (
            args.expected_site_count is not None
            and occurrences != args.expected_site_count
        ):
            raise self._count_mismatch_error(
                args.expected_site_count,
                occurrences,
                [original.count("\n", 0, position) + 1 for position in positions],
            )
        if occurrences > 1 and not args.replace_all:
            raise ToolError(
                f"Found {occurrences} matches of the string to replace, "
                f"but replace_all is false. To replace all occurrences, "
                f"set replace_all to true. To replace only one occurrence, "
                f"please provide more context to uniquely identify the "
                f"instance.\nString: {old_string}"
            )

        contexts = [
            (start, prefix + old_string + suffix, prefix + new_string + suffix)
            for start, prefix, suffix in line_contexts(original, old_string)
        ]
        if not args.replace_all:
            contexts = contexts[:1]

        return _EditPlan(
            modified=self._apply_edit(
                original, old_string, new_string, args.replace_all
            ),
            old_string=old_string,
            new_string=new_string,
            replace_all=args.replace_all,
            sites=None,
            contexts=contexts,
        )

    @classmethod
    def _apply_multi_anchor(
        cls, original: str, anchors: list[AnchorEdit]
    ) -> tuple[str, list[AppliedSite]]:
        """Assemble every anchor's replacement, validating all anchors first.

        Each anchor must match exactly once, anchors must be non-overlapping
        and listed in file order. Any failure raises before content is
        assembled, so nothing is ever written partially.
        """
        positions: list[int] = []
        for index, anchor in enumerate(anchors):
            found = _occurrence_positions(original, anchor.old_string)
            if len(found) != 1:
                raise cls._anchor_count_error(
                    index,
                    len(found),
                    [original.count("\n", 0, position) + 1 for position in found],
                )
            positions.append(found[0])

        for index in range(1, len(positions)):
            if positions[index] < positions[index - 1]:
                raise ToolError(
                    f"multi_anchor[{index}] starts before multi_anchor[{index - 1}], "
                    f"so the anchors are not in file order; no changes were made. "
                    f"List the anchors in the order they appear in the file."
                )
            previous_end = positions[index - 1] + len(anchors[index - 1].old_string)
            if positions[index] < previous_end:
                raise ToolError(
                    f"multi_anchor[{index}] overlaps multi_anchor[{index - 1}] and "
                    f"no changes were made: anchors must not overlap"
                )

        pieces: list[str] = []
        cursor = 0
        for anchor, position in zip(anchors, positions, strict=True):
            pieces.append(original[cursor:position])
            pieces.append(anchor.new_string)
            cursor = position + len(anchor.old_string)
        pieces.append(original[cursor:])
        modified = "".join(pieces)

        sites: list[AppliedSite] = []
        line_shift = 0
        char_shift = 0
        for anchor, position in zip(anchors, positions, strict=True):
            post_position = position + char_shift
            sites.append(
                AppliedSite(
                    line=original.count("\n", 0, position) + 1 + line_shift,
                    old_text=_expand_whole_lines(
                        original, position, position + len(anchor.old_string)
                    ),
                    new_text=(
                        ""
                        if not anchor.new_string
                        else _expand_whole_lines(
                            modified,
                            post_position,
                            post_position + len(anchor.new_string),
                        )
                    ),
                )
            )
            line_shift += anchor.new_string.count("\n") - anchor.old_string.count("\n")
            char_shift += len(anchor.new_string) - len(anchor.old_string)
        return modified, sites

    @staticmethod
    async def _multi_anchor_evidence(
        *, file_path: Path, modified: str, sites: list[AppliedSite], host_written: bool
    ) -> EditEvidence:
        """Render a multi-anchor edit through the shared bounded echo helpers.

        ``build_edit_evidence`` models one anchor pair, so the assembled site
        list is rendered with the same renderers and the same post-edit
        revision read it uses.
        """
        revision = await _read_post_edit_revision(file_path) if host_written else None
        return EditEvidence(
            applied_diff=render_applied_diff(sites),
            context=render_context_window(
                modified, first_site=sites[0] if sites else None
            ),
            revision=revision,
            replaced_count=len(sites),
        )

    @staticmethod
    def _count_mismatch_error(expected: int, found: int, lines: list[int]) -> ToolError:
        location = (
            f" First {min(len(lines), EDIT_DIFF_MAX_SITES)} site(s) at lines "
            f"{_bounded_line_numbers(lines)}."
            if lines
            else ""
        )
        return ToolError(
            f"expected_site_count is {expected} but the anchor matched {found} "
            f"site(s), so no changes were made.{location} Re-read the file and pass "
            f"the exact count, or narrow the anchor."
        )

    @staticmethod
    def _anchor_count_error(index: int, found: int, lines: list[int]) -> ToolError:
        location = (
            f" First {min(len(lines), EDIT_DIFF_MAX_SITES)} site(s) at lines "
            f"{_bounded_line_numbers(lines)}."
            if lines
            else ""
        )
        return ToolError(
            f"multi_anchor[{index}] must match exactly once but matched {found} "
            f"site(s), so no changes were made.{location} No anchor is ever "
            f"applied partially."
        )

    @staticmethod
    def _apply_edit(
        content: str, old_string: str, new_string: str, replace_all: bool
    ) -> str:
        if replace_all:
            return content.replace(old_string, new_string)
        return content.replace(old_string, new_string, 1)
