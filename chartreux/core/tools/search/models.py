from __future__ import annotations

from pydantic import BaseModel

MAX_QUERY_LENGTH = 2_000
MAX_RESULTS = 20
MAX_UPSTREAM_JSON_BYTES = 1_024 * 1_024
MAX_SNIPPET_LENGTH = 2_000
MAX_SERIALIZED_RESULT_BYTES = 120_000


class SearchSource(BaseModel):
    title: str
    url: str
    snippet: str | None


class SearchResponse(BaseModel):
    query: str
    provider: str
    answer: str | None
    sources: list[SearchSource]
    was_truncated: bool
