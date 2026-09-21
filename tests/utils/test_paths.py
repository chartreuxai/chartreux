from __future__ import annotations

import pytest

from chartreux.utils import paths
from chartreux.utils.paths import is_foreign_windows_path


def test_file_uri_to_path_decodes_posix_path() -> None:
    assert paths.file_uri_to_path("file:///tmp/image%20one.png") == "/tmp/image one.png"


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("file://server/share/image.png", "//server/share/image.png"),
        ("file://server/share/image%20one.png", "//server/share/image one.png"),
        ("file://localhost/tmp/image.png", "/tmp/image.png"),
    ],
)
def test_file_uri_to_path_preserves_posix_authorities(uri: str, expected: str) -> None:
    assert paths.file_uri_to_path(uri) == expected


def test_file_uri_to_path_rejects_other_schemes() -> None:
    with pytest.raises(ValueError, match="Expected a file URI"):
        paths.file_uri_to_path("https://server/share/image.png")


def test_foreign_windows_path_recognizes_drive_absolute_and_unc_paths() -> None:
    assert is_foreign_windows_path("C:\\Users\\acmedev\\notes.md")
    assert is_foreign_windows_path("\\\\server\\share\\notes.md")


def test_foreign_windows_path_does_not_classify_posix_or_drive_relative_paths() -> None:
    assert not is_foreign_windows_path("/tmp/notes.md")
    assert not is_foreign_windows_path("C:notes.md")
