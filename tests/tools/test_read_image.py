from __future__ import annotations

import base64
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.llm_models import FileImageSource, InlineImageSource
from chartreux.core.tools.base import InvokeContext, ToolError, ToolPermission
from chartreux.core.tools.builtins.read_image import (
    MAX_IMAGE_BYTES,
    ReadImage,
    ReadImageArgs,
    ReadImageConfig,
    ReadImageState,
    _has_valid_image_structure,
)
from chartreux.core.tools.permissions import PermissionContext
from tests.mock.utils import collect_result

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQAAAAA3bvkkAAAACklEQVQI12NoAAAAggCB3UNq9AAAAABJRU5ErkJggg=="
)
JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=="
)
JPEG_TRUNCATED_AFTER_SOF = JPEG[:102]
JPEG_MISSING_EOI = JPEG[:-2]
JPEG_TRUNCATED_MID_SCAN = JPEG[:-3]
GIF = base64.b64decode("R0lGODlhAQABAPAAAP///wAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==")
WEBP = base64.b64decode("UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAgA0JaQAA3AA/vuUAAA=")


@pytest.mark.parametrize(
    ("data", "mime"),
    [
        (PNG, "image/png"),
        (JPEG, "image/jpeg"),
        (GIF, "image/gif"),
        (WEBP, "image/webp"),
    ],
)
def test_minimal_images_have_valid_structure(data: bytes, mime: str) -> None:
    assert _has_valid_image_structure(data, mime)


def _runtime(vision: bool = True) -> ChartreuxConfigSchema:
    return cast(
        ChartreuxConfigSchema,
        SimpleNamespace(
            get_active_model=lambda: SimpleNamespace(supports_images=vision)
        ),
    )


def _tool(tmp_path: Path, *, vision: bool = True, **kwargs: object) -> ReadImage:
    return ReadImage(
        lambda: ReadImageConfig(),
        ReadImageState(),
        cwd=tmp_path,
        runtime_config_getter=lambda: _runtime(vision),
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "data", "mime"),
    [
        ("image.PNG", PNG, "image/png"),
        ("image.JPG", JPEG, "image/jpeg"),
        ("image.jpeg", JPEG, "image/jpeg"),
        ("image.gif", GIF, "image/gif"),
        ("image.webp", WEBP, "image/webp"),
    ],
)
async def test_reads_supported_images(
    tmp_path: Path, name: str, data: bytes, mime: str
) -> None:
    path = tmp_path / name
    path.write_bytes(data)

    result = await collect_result(_tool(tmp_path).run(ReadImageArgs(file_path=name)))

    assert result.file_path == str(path.resolve())
    assert result.mime_type == mime
    assert result.size_bytes == len(data)
    assert result.attachment is not None
    assert isinstance(result.attachment.source, InlineImageSource)
    assert "attachment" not in result.model_dump()
    assert _tool(tmp_path).get_result_images(result) == [result.attachment]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("signature-only.png", b"\x89PNG\r\n\x1a\n"),
        ("truncated.png", PNG[:33]),
        ("truncated.jpg", JPEG[:20]),
        ("truncated-after-sof.jpg", JPEG_TRUNCATED_AFTER_SOF),
        ("missing-eoi.jpg", JPEG_MISSING_EOI),
        ("truncated-mid-scan.jpg", JPEG_TRUNCATED_MID_SCAN),
        ("truncated.gif", GIF[:13]),
        ("truncated.webp", WEBP[:-1]),
    ],
)
async def test_rejects_corrupt_or_truncated_images(
    tmp_path: Path, name: str, data: bytes
) -> None:
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises(ToolError, match="corrupt or truncated"):
        await collect_result(_tool(tmp_path).run(ReadImageArgs(file_path=str(path))))


@pytest.mark.asyncio
async def test_rejects_empty_image(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"")
    with pytest.raises(ToolError, match="empty"):
        await collect_result(_tool(tmp_path).run(ReadImageArgs(file_path=str(path))))


@pytest.mark.asyncio
async def test_rejects_mime_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(JPEG)
    with pytest.raises(ToolError, match="does not match"):
        await collect_result(_tool(tmp_path).run(ReadImageArgs(file_path=str(path))))


@pytest.mark.asyncio
async def test_size_limit_is_inclusive(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted.png"
    accepted.write_bytes(PNG + b"x" * (MAX_IMAGE_BYTES - len(PNG)))
    assert (
        await collect_result(
            _tool(tmp_path).run(ReadImageArgs(file_path=str(accepted)))
        )
    ).size_bytes == MAX_IMAGE_BYTES

    rejected = tmp_path / "rejected.png"
    rejected.write_bytes(PNG + b"x" * (MAX_IMAGE_BYTES + 1 - len(PNG)))
    with pytest.raises(ToolError, match="size limit"):
        await collect_result(
            _tool(tmp_path).run(ReadImageArgs(file_path=str(rejected)))
        )


@pytest.mark.asyncio
async def test_rejects_missing_directory_and_fifo(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    with pytest.raises(ToolError, match="not found"):
        await collect_result(tool.run(ReadImageArgs(file_path="missing.png")))
    directory = tmp_path / "directory.png"
    directory.mkdir()
    with pytest.raises(ToolError, match="regular file"):
        await collect_result(tool.run(ReadImageArgs(file_path=str(directory))))
    fifo = tmp_path / "pipe.png"
    os.mkfifo(fifo)
    with pytest.raises(ToolError, match="regular file"):
        await collect_result(tool.run(ReadImageArgs(file_path=str(fifo))))


@pytest.mark.asyncio
async def test_relative_path_uses_tool_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    (tmp_path / "image.png").write_bytes(PNG)
    result = await collect_result(
        _tool(tmp_path).run(ReadImageArgs(file_path="image.png"))
    )
    assert result.file_path == str((tmp_path / "image.png").resolve())


def test_permission_uses_read_file_policy(tmp_path: Path) -> None:
    sensitive = _tool(tmp_path).resolve_permission(ReadImageArgs(file_path=".env"))
    assert isinstance(sensitive, PermissionContext)
    assert sensitive.permission is ToolPermission.NEVER

    denied_tool = ReadImage(
        lambda: ReadImageConfig(denylist=[str(tmp_path / "denied.png")]),
        ReadImageState(),
        cwd=tmp_path,
        runtime_config_getter=lambda: _runtime(),
    )
    denied = denied_tool.resolve_permission(ReadImageArgs(file_path="denied.png"))
    assert isinstance(denied, PermissionContext)
    assert denied.permission is ToolPermission.NEVER

    outside = _tool(tmp_path).resolve_permission(
        ReadImageArgs(file_path="/tmp/outside.png")
    )
    assert isinstance(outside, PermissionContext)
    assert outside.permission is ToolPermission.NEVER


@pytest.mark.asyncio
async def test_snapshots_dedupe_and_inline_when_logging_disabled(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(PNG)
    session_dir = tmp_path / "session"
    tool = _tool(tmp_path)
    first = await collect_result(
        tool.run(
            ReadImageArgs(file_path=str(image)),
            InvokeContext("one", session_dir=session_dir),
        )
    )
    second = await collect_result(
        tool.run(
            ReadImageArgs(file_path=str(image)),
            InvokeContext("two", session_dir=session_dir),
        )
    )
    assert first.attachment is not None
    assert second.attachment is not None
    assert isinstance(first.attachment.source, FileImageSource)
    assert isinstance(second.attachment.source, FileImageSource)
    assert first.attachment.source.path == second.attachment.source.path
    assert len(list((session_dir / "attachments").iterdir())) == 1

    inline = await collect_result(tool.run(ReadImageArgs(file_path=str(image))))
    assert inline.attachment is not None
    assert isinstance(inline.attachment.source, InlineImageSource)


@pytest.mark.asyncio
async def test_non_vision_model_is_unavailable_and_fails_before_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not ReadImage.is_available(_runtime(False))
    path = tmp_path / "image.png"
    path.write_bytes(PNG)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: pytest.fail("opened"))
    with pytest.raises(ToolError, match="vision-capable"):
        await collect_result(
            _tool(tmp_path, vision=False).run(ReadImageArgs(file_path=str(path)))
        )
