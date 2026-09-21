from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.core.llm_models import (
    FileImageSource,
    ImageAttachment,
    InlineImageSource,
)


@pytest.mark.parametrize("flat_source", [{"path": "/tmp/a.png"}, {"data": "Zm9v"}])
def test_rejects_flat_source_shape(flat_source: dict[str, str]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ImageAttachment.model_validate({
            **flat_source,
            "alias": "a.png",
            "mime_type": "image/png",
        })

    assert [(error["loc"], error["type"]) for error in exc_info.value.errors()] == [
        (("source",), "missing")
    ]


@pytest.mark.parametrize(
    "source",
    [{"kind": "file", "path": "/tmp/a.png"}, {"kind": "inline", "data": "Zm9v"}],
)
def test_nested_source_round_trips(source: dict[str, str]) -> None:
    payload = {"source": source, "alias": "a.png", "mime_type": "image/png"}
    att = ImageAttachment.model_validate(payload)

    assert att.model_dump(mode="json") == payload
    assert ImageAttachment.model_validate_json(att.model_dump_json()) == att


def test_nested_source_is_not_overridden_by_flat_fields() -> None:
    att = ImageAttachment.model_validate({
        "source": {"kind": "inline", "data": "Zm9v"},
        "path": "/tmp/ignored.png",
        "data": "ignored",
        "alias": "a.png",
        "mime_type": "image/png",
    })

    assert att.source == InlineImageSource(data="Zm9v")


def test_source_path_construction() -> None:
    att = ImageAttachment(
        source=FileImageSource(path=Path("/tmp/a.png")),
        alias="a.png",
        mime_type="image/png",
    )

    assert isinstance(att.source, FileImageSource)
    assert att.source.path == Path("/tmp/a.png")


def test_file_source_round_trips_through_json() -> None:
    att = ImageAttachment(
        source=FileImageSource(path=Path("/tmp/a.png")),
        alias="a.png",
        mime_type="image/png",
    )

    dumped = att.model_dump(exclude_none=True, mode="json")
    assert dumped["source"] == {"kind": "file", "path": "/tmp/a.png"}
    assert ImageAttachment.model_validate(dumped) == att


def test_rejects_attachment_without_source() -> None:
    with pytest.raises(ValidationError):
        ImageAttachment.model_validate({"alias": "a.png", "mime_type": "image/png"})
