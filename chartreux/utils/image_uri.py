from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from chartreux.app_server.models import (
    FileImageSource,
    ImageAttachment,
    InlineImageSource,
    SessionImageContentBlock,
)
from chartreux.utils.paths import file_uri_to_path


def image_attachment_from_session_block(
    block: SessionImageContentBlock,
) -> ImageAttachment:
    if block.uri.startswith("data:"):
        header, separator, data = block.uri.partition(",")
        if not separator or not header.endswith(";base64"):
            raise ValueError("Queued image URI is not base64 data")
        media_type = block.media_type or header[5:-7]
        if not media_type:
            raise ValueError("Queued image URI has no media type")
        return ImageAttachment(
            source=InlineImageSource(data=data),
            alias=block.alt_text or "image",
            mime_type=media_type,
        )

    parsed = urlparse(block.uri)
    if parsed.scheme not in {"", "file"}:
        raise ValueError(f"Queued image URI is not local: {block.uri!r}")
    path = file_uri_to_path(block.uri) if parsed.scheme == "file" else block.uri
    if block.media_type is None:
        raise ValueError("Queued image URI has no media type")
    return ImageAttachment(
        source=FileImageSource(path=path),
        alias=block.alt_text or Path(path).name or "image",
        mime_type=block.media_type,
    )


__all__ = ["image_attachment_from_session_block"]
