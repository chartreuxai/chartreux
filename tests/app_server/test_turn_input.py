from __future__ import annotations

from chartreux.app_server._turn_input import vibe_content_blocks
from chartreux.app_server.models import FileImageSource, ImageContentBlock
from chartreux.app_server.protocol import SessionImageContentBlock


def test_vibe_content_blocks_decode_posix_file_uri() -> None:
    block = SessionImageContentBlock(
        uri="file:///tmp/image%20one.png", media_type="image/png", alt_text="image one"
    )

    content = vibe_content_blocks([block])

    image = content[0]
    assert isinstance(image, ImageContentBlock)
    assert isinstance(image.attachment.source, FileImageSource)
    assert image.attachment.source.path == "/tmp/image one.png"
