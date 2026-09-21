from __future__ import annotations

import pytest

from chartreux import __version__
from chartreux.utils.http import get_server_url_from_api_base, get_user_agent


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        ("https://api.mistral.ai/v1", "https://api.mistral.ai"),
        ("https://on-prem.example.com/v1", "https://on-prem.example.com"),
        ("http://localhost:8080/v2", "http://localhost:8080"),
        ("not-a-url", None),
        ("ftp://example.com/v1", None),
    ],
)
def test_get_server_url_from_api_base(api_base: str, expected: str | None) -> None:
    assert get_server_url_from_api_base(api_base) == expected


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("mistral", f"mistral-client-python/Chartreux/{__version__}"),
        ("generic", f"Chartreux/{__version__}"),
        (None, f"Chartreux/{__version__}"),
    ],
)
def test_get_user_agent(backend: str | None, expected: str) -> None:
    assert get_user_agent(backend) == expected
