from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.policy.models import ResourceKind
from chartreux.core.policy.resources import ResourceLoader, ResourceResolutionError


@pytest.fixture
def loader(tmp_path: Path) -> tuple[ResourceLoader, Path]:
    user = tmp_path / "user"
    user.mkdir()
    builtin = tmp_path / "shipped.md"
    builtin.write_text("shipped body", encoding="utf-8")
    return ResourceLoader(
        {ResourceKind.IDENTITY: user}, {ResourceKind.IDENTITY: {"coder": builtin}}
    ), user


def test_bare_builtin_and_qualification_are_deterministic(loader) -> None:
    resource_loader, user = loader
    (user / "custom.md").write_text("custom body", encoding="utf-8")
    assert resource_loader.resolve(ResourceKind.IDENTITY, "coder").source == "builtin"
    assert (
        resource_loader.resolve(ResourceKind.IDENTITY, "builtin:coder").source
        == "builtin"
    )
    assert (
        resource_loader.resolve(ResourceKind.IDENTITY, "user:custom").source == "user"
    )
    assert resource_loader.resolve(ResourceKind.IDENTITY, "custom").source == "user"


def test_shipped_names_are_reserved_from_user_namespace(loader) -> None:
    resource_loader, user = loader
    (user / "coder.md").write_text("shadow", encoding="utf-8")
    with pytest.raises(ResourceResolutionError, match="reserved built-in"):
        resource_loader.resolve(ResourceKind.IDENTITY, "user:coder")


@pytest.mark.parametrize(
    "reference", ["../escape", "a/b", "builtin:", "user:a:b", "other:name", "bad name"]
)
def test_rejects_traversal_and_malformed_ids(loader, reference: str) -> None:
    with pytest.raises(ResourceResolutionError):
        loader[0].resolve(ResourceKind.IDENTITY, reference)


def test_rejects_unreadable_empty_bad_encoding_and_oversized(
    loader, tmp_path: Path
) -> None:
    resource_loader, user = loader
    with pytest.raises(ResourceResolutionError, match="cannot read"):
        resource_loader.resolve(ResourceKind.IDENTITY, "missing")
    (user / "empty.md").write_bytes(b" \n")
    with pytest.raises(ResourceResolutionError, match="empty"):
        resource_loader.resolve(ResourceKind.IDENTITY, "empty")
    (user / "encoding.md").write_bytes(b"\xff")
    with pytest.raises(ResourceResolutionError, match="cannot read"):
        resource_loader.resolve(ResourceKind.IDENTITY, "encoding")
    (user / "large.md").write_bytes(b"x" * (ResourceLoader.MAX_BYTES + 1))
    with pytest.raises(ResourceResolutionError, match="exceeds 65536"):
        resource_loader.resolve(ResourceKind.IDENTITY, "large")


def test_custom_identity_does_not_inherit_builtin(loader) -> None:
    resource_loader, user = loader
    (user / "custom.md").write_text("standalone", encoding="utf-8")
    resolved = resource_loader.resolve(ResourceKind.IDENTITY, "user:custom")
    assert resolved.source == "user"
    assert resolved.content == "standalone"


def test_project_prompt_directory_is_not_a_resolution_source(
    loader, tmp_path: Path
) -> None:
    resource_loader, user = loader
    project = tmp_path / "project-prompts"
    project.mkdir()
    (project / "only-project.md").write_text("project", encoding="utf-8")
    with pytest.raises(ResourceResolutionError, match="cannot read"):
        resource_loader.resolve(ResourceKind.IDENTITY, "only-project")
