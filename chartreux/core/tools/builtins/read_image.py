from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
import stat
from typing import TYPE_CHECKING, Any, final

from pydantic import BaseModel, Field

from chartreux.core.events import ToolStreamEvent
from chartreux.core.llm_models import ImageAttachment
from chartreux.core.scratchpad import is_scratchpad_display_path
from chartreux.core.session.image_snapshot import (
    ImageSnapshotError,
    snapshot_image_bytes,
)
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from chartreux.core.tools.utils import (
    DEFAULT_SENSITIVE_PATTERNS,
    ToolPath,
    display_file_path,
    resolve_file_tool_permission,
    resolve_tool_path,
)
from chartreux.utils.images import IMAGE_EXTENSIONS, MAX_IMAGE_BYTES
from chartreux.utils.tool_presentation import ToolEffectKind

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema
    from chartreux.core.events import ToolResultEvent


_MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_WEBP_HEADER_LENGTH = 12
_PNG_IHDR_LENGTH = 13
_JPEG_MARKER_PREFIX = 0xFF
_JPEG_END_OF_IMAGE_MARKER = 0xD9
_JPEG_START_OF_SCAN_MARKER = 0xDA
_JPEG_STANDALONE_MARKERS = {0x00, 0xD8, _JPEG_END_OF_IMAGE_MARKER}
_JPEG_RESTART_MARKER_FIRST = 0xD0
_JPEG_RESTART_MARKER_LAST = 0xD7
_JPEG_LENGTH_FIELD_BYTES = 2
_JPEG_MIN_SOF_SEGMENT_LENGTH = 8
_JPEG_MIN_SOS_SEGMENT_LENGTH = 6
_GIF_LOGICAL_SCREEN_DESCRIPTOR_LENGTH = 13
_WEBP_VP8_MIN_CHUNK_LENGTH = 10
_WEBP_VP8L_MIN_CHUNK_LENGTH = 5
_WEBP_VP8L_SIGNATURE = 0x2F
_MAX_IMAGE_DIMENSION = 100_000


def _has_plausible_dimensions(width: int, height: int) -> bool:
    return 0 < width <= _MAX_IMAGE_DIMENSION and 0 < height <= _MAX_IMAGE_DIMENSION


def _has_valid_png_structure(data: bytes) -> bool:
    position = 8
    saw_ihdr = saw_idat = False
    while position + 12 <= len(data):
        length = int.from_bytes(data[position : position + 4], "big")
        chunk_type = data[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(data):
            return False
        chunk_data = data[position + 8 : position + 8 + length]
        if chunk_type == b"IHDR":
            if saw_ihdr or length != _PNG_IHDR_LENGTH:
                return False
            width = int.from_bytes(chunk_data[:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            if not _has_plausible_dimensions(width, height):
                return False
            saw_ihdr = True
        elif chunk_type == b"IDAT":
            saw_idat = True
        elif chunk_type == b"IEND":
            return saw_ihdr and saw_idat and length == 0
        position = end
    return False


def _has_valid_jpeg_structure(data: bytes) -> bool:  # noqa: PLR0911, PLR0912
    if not data.startswith(b"\xff\xd8"):
        return False

    position = 2
    marker: int | None = None
    saw_sof = saw_sos = False
    while position < len(data) or marker is not None:
        if marker is None:
            if data[position] != _JPEG_MARKER_PREFIX:
                return False
            while position < len(data) and data[position] == _JPEG_MARKER_PREFIX:
                position += 1
            if position >= len(data):
                return False
            marker = data[position]
            position += 1

        if marker == _JPEG_END_OF_IMAGE_MARKER:
            return saw_sof and saw_sos and position == len(data)
        if (
            marker in _JPEG_STANDALONE_MARKERS
            or _JPEG_RESTART_MARKER_FIRST <= marker <= _JPEG_RESTART_MARKER_LAST
        ):
            return False
        if position + _JPEG_LENGTH_FIELD_BYTES > len(data):
            return False
        segment_length = int.from_bytes(
            data[position : position + _JPEG_LENGTH_FIELD_BYTES], "big"
        )
        if segment_length < _JPEG_LENGTH_FIELD_BYTES or position + segment_length > len(
            data
        ):
            return False
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if segment_length < _JPEG_MIN_SOF_SEGMENT_LENGTH:
                return False
            height = int.from_bytes(data[position + 3 : position + 5], "big")
            width = int.from_bytes(data[position + 5 : position + 7], "big")
            if not _has_plausible_dimensions(width, height):
                return False
            saw_sof = True
        elif marker == _JPEG_START_OF_SCAN_MARKER:
            if not saw_sof or segment_length < _JPEG_MIN_SOS_SEGMENT_LENGTH:
                return False
            saw_sos = True
            position += segment_length
            while position < len(data):
                if data[position] != _JPEG_MARKER_PREFIX:
                    position += 1
                    continue
                position += 1
                while position < len(data) and data[position] == _JPEG_MARKER_PREFIX:
                    position += 1
                if position >= len(data):
                    return False
                marker = data[position]
                position += 1
                if marker == 0x00 or (
                    _JPEG_RESTART_MARKER_FIRST <= marker <= _JPEG_RESTART_MARKER_LAST
                ):
                    continue
                break
            else:
                return False
            continue
        position += segment_length
        marker = None
    return False


def _has_valid_gif_structure(data: bytes) -> bool:
    if len(data) < _GIF_LOGICAL_SCREEN_DESCRIPTOR_LENGTH:
        return False
    width = int.from_bytes(data[6:8], "little")
    height = int.from_bytes(data[8:10], "little")
    if not _has_plausible_dimensions(width, height):
        return False
    position = 13
    packed_fields = data[10]
    if packed_fields & 0x80:
        position += 3 * (2 ** ((packed_fields & 0x07) + 1))
    return position <= len(data) and b"\x2c" in data[position:] and data.endswith(b";")


def _has_valid_webp_structure(data: bytes) -> bool:  # noqa: PLR0911
    if len(data) < _WEBP_HEADER_LENGTH or int.from_bytes(
        data[4:8], "little"
    ) + 8 != len(data):
        return False
    position = _WEBP_HEADER_LENGTH
    while position + 8 <= len(data):
        chunk_type = data[position : position + 4]
        chunk_size = int.from_bytes(data[position + 4 : position + 8], "little")
        chunk_data_start = position + 8
        chunk_data_end = chunk_data_start + chunk_size
        if chunk_data_end > len(data):
            return False
        chunk_data = data[chunk_data_start:chunk_data_end]
        if chunk_type == b"VP8 " and chunk_size >= _WEBP_VP8_MIN_CHUNK_LENGTH:
            if chunk_data[3:6] != b"\x9d\x01\x2a":
                return False
            width = int.from_bytes(chunk_data[6:8], "little") & 0x3FFF
            height = int.from_bytes(chunk_data[8:10], "little") & 0x3FFF
            return _has_plausible_dimensions(width, height)
        if (
            chunk_type == b"VP8L"
            and chunk_size >= _WEBP_VP8L_MIN_CHUNK_LENGTH
            and chunk_data[0] == _WEBP_VP8L_SIGNATURE
        ):
            dimensions = int.from_bytes(chunk_data[1:5], "little")
            width = (dimensions & 0x3FFF) + 1
            height = ((dimensions >> 14) & 0x3FFF) + 1
            return _has_plausible_dimensions(width, height)
        if chunk_type == b"VP8X" and chunk_size >= _WEBP_VP8_MIN_CHUNK_LENGTH:
            width = int.from_bytes(chunk_data[4:7], "little") + 1
            height = int.from_bytes(chunk_data[7:10], "little") + 1
            return _has_plausible_dimensions(width, height)
        position = chunk_data_end + (chunk_size % 2)
    return False


def _has_valid_image_structure(data: bytes, mime_type: str) -> bool:
    if mime_type == "image/png":
        return _has_valid_png_structure(data)
    if mime_type == "image/jpeg":
        return _has_valid_jpeg_structure(data)
    if mime_type == "image/gif":
        return _has_valid_gif_structure(data)
    if mime_type == "image/webp":
        return _has_valid_webp_structure(data)
    return False


def _mime_from_signature(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if (
        len(data) >= _WEBP_HEADER_LENGTH
        and data.startswith(b"RIFF")
        and data[8:_WEBP_HEADER_LENGTH] == b"WEBP"
    ):
        return "image/webp"
    return None


def _read_validated_image(path: Path) -> tuple[bytes, str]:
    try:
        mode = path.stat().st_mode
    except FileNotFoundError as exc:
        raise ToolError(f"File not found at: {path}") from exc
    except OSError as exc:
        raise ToolError(f"Unable to inspect image {path}: {exc}") from exc

    if not stat.S_ISREG(mode):
        raise ToolError(f"Path is not a regular file: {path}")

    extension = path.suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        raise ToolError(
            f"Unsupported image extension '{extension}'. "
            f"Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    try:
        with path.open("rb") as file:
            data = file.read(MAX_IMAGE_BYTES + 1)
    except OSError as exc:
        raise ToolError(f"Unable to read image {path}: {exc}") from exc

    if not data:
        raise ToolError(f"Image file is empty: {path}")
    if len(data) > MAX_IMAGE_BYTES:
        raise ToolError(f"Image exceeds the {MAX_IMAGE_BYTES} byte size limit: {path}")

    mime_type = _mime_from_signature(data)
    if mime_type is None:
        raise ToolError(f"Image has an invalid or unsupported signature: {path}")
    if mime_type != _MIME_BY_EXTENSION[extension]:
        raise ToolError(
            f"Image extension '{extension}' does not match its {mime_type} signature: {path}"
        )
    if not _has_valid_image_structure(data, mime_type):
        raise ToolError(f"Image is corrupt or truncated: {path}")
    return data, mime_type


class ReadImageArgs(BaseModel):
    file_path: ToolPath = Field(
        description="The image file path, relative to the tool working directory or absolute"
    )


class ReadImageResult(BaseModel):
    file_path: str
    alias: str
    mime_type: str
    size_bytes: int
    attachment: ImageAttachment | None = Field(default=None, exclude=True)


class ReadImageConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    sensitive_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SENSITIVE_PATTERNS),
        description="File patterns that are never readable.",
    )


class ReadImageState(BaseToolState):
    pass


class ReadImage(
    BaseTool[ReadImageArgs, ReadImageResult, ReadImageConfig, ReadImageState],
    ToolUIData[ReadImageArgs, ReadImageResult],
):
    effect_kind = ToolEffectKind.FILE_READ

    def __init__(
        self,
        config_getter: Callable[[], ReadImageConfig],
        state: ReadImageState,
        *,
        runtime_config_getter: Callable[[], ChartreuxConfigSchema] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config_getter, state, **kwargs)
        self._runtime_config_getter = runtime_config_getter

    def _set_runtime_config_getter(
        self, getter: Callable[[], ChartreuxConfigSchema]
    ) -> None:
        self._runtime_config_getter = getter

    @staticmethod
    def _supports_images(config: ChartreuxConfigSchema | None) -> bool:
        if config is None:
            return False
        try:
            return config.get_active_model().supports_images
        except (AttributeError, KeyError, ValueError):
            return False

    @classmethod
    def is_available(cls, config: ChartreuxConfigSchema | None = None) -> bool:
        return cls._supports_images(config)

    def _runtime_supports_images(self) -> bool:
        if self._runtime_config_getter is None:
            return False
        try:
            return self._supports_images(self._runtime_config_getter())
        except Exception:
            return False

    def resolve_permission(self, args: ReadImageArgs) -> PermissionContext | None:
        return resolve_file_tool_permission(
            args.file_path,
            tool_name=self.get_name(),
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
            config_permission=self.config.permission,
            sensitive_patterns=self.config.sensitive_patterns,
            workspace=self.workspace,
            scratchpad_dir=self.scratchpad_dir,
        )

    def get_result_images(
        self, result: ReadImageResult
    ) -> list[ImageAttachment] | None:
        return [result.attachment] if result.attachment is not None else None

    @final
    async def run(
        self, args: ReadImageArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | ReadImageResult, None]:
        if not self._runtime_supports_images():
            raise ToolError("read_image requires an active vision-capable model")
        if not args.file_path.strip():
            raise ToolError("file_path cannot be empty")

        try:
            file_path = resolve_tool_path(args.file_path, self.cwd)
        except (OSError, ValueError) as exc:
            raise ToolError(
                f"Unable to resolve image path {args.file_path!r}: {exc}"
            ) from exc
        data, mime_type = await asyncio.to_thread(_read_validated_image, file_path)
        try:
            attachment = await asyncio.to_thread(
                snapshot_image_bytes,
                data,
                alias=file_path.name,
                mime_type=mime_type,
                session_dir=ctx.session_dir if ctx is not None else None,
            )
        except (ImageSnapshotError, OSError) as exc:
            raise ToolError(f"Unable to snapshot image {file_path}: {exc}") from exc

        if ctx is not None:
            yield ToolStreamEvent(
                tool_name=self.get_name(),
                tool_call_id=ctx.tool_call_id,
                message="Reading image...",
            )
        yield ReadImageResult(
            file_path=str(file_path),
            alias=file_path.name,
            mime_type=mime_type,
            size_bytes=len(data),
            attachment=attachment,
        )

    @classmethod
    def format_call_display(cls, args: ReadImageArgs) -> ToolCallDisplay:
        suffix = "(scratchpad)" if is_scratchpad_display_path(args.file_path) else ""
        message = display_file_path(args.file_path)
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
        if not isinstance(event.result, ReadImageResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        suffix = (
            "(scratchpad)" if is_scratchpad_display_path(event.result.file_path) else ""
        )
        return ToolResultDisplay(
            success=True,
            verb="Read",
            message=f"image {display_file_path(event.result.file_path)}",
            suffix=suffix,
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Reading image"
