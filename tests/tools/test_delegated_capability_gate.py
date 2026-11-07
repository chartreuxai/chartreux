"""Delegation capability gate: the child's committed model decides capability.

Pins the v0.7.4 read_image fix (design: read-image-delegation-fix-design.md).
The ancestor chain contributes AUTHORITY only (granted configuration,
delegation posture, every ancestor's delegation ceiling); a capability gate
(``is_available``) is evaluated against the CHILD's own committed active
model. Before the fix, a non-vision parent re-ran ``is_available`` with its
own model inside the ceiling walk and stripped ``read_image`` from every
child catalog, including vision-capable children.

Enforcement of the real constraint stays elsewhere and is pinned here too:
the send boundary (``ImagesNotSupportedError``), the child launch guard
(``InvalidLaunchModelError``), and ``ReadImage.run``'s own re-check.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from chartreux.core.agent_loop.errors import ImagesNotSupportedError
from chartreux.core.agent_loop.llm_gateway import messages_for_backend
from chartreux.core.agents.launch import _validate_history
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig, ProviderConfig
from chartreux.core.llm_models import (
    Backend,
    ImageAttachment,
    InlineImageSource,
    LLMMessage,
    Role,
)
from chartreux.core.model_catalog.resolver import resolver_for
from chartreux.core.subagents import InvalidLaunchModelError
from chartreux.core.tools.base import ToolError
from chartreux.core.tools.builtins.read_image import (
    ReadImage,
    ReadImageArgs,
    ReadImageConfig,
    ReadImageState,
)
from chartreux.core.tools.manager import NoSuchToolError, ToolManager
from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQAAAAA3bvkkAAAACklEQVQI12NoAAAAggCB3UNq9AAAAABJRU5ErkJggg=="
)

MUTATING_TOOLS = ("bash", "bash_start", "edit", "write_file")
NON_MUTATING_TOOLS = ("read_file", "grep", "read_image")


def _config(active_model: str) -> ChartreuxConfigSchema:
    models = [
        ModelConfig(
            name="vision", provider="mistral", alias="vision", supports_images=True
        ),
        ModelConfig(
            name="text", provider="mistral", alias="text", supports_images=False
        ),
    ]
    config = build_test_vibe_config(
        active_model=active_model,
        models=models,
        providers=[
            ProviderConfig(
                name="mistral",
                api_base="https://api.mistral.ai/v1",
                api_key_env_var="MISTRAL_API_KEY",
                backend=Backend.MISTRAL,
            )
        ],
    )
    # A catalog is gated on the COMMITTED active model, so bind the
    # resolution the way a launch does rather than leaving it unpinned.
    resolved = resolver_for(config).resolve(
        active_model, allowed_models=config.allowed_models
    )
    return config.attach_committed_model(resolved.identity)


def _manager(
    tmp_path: Path,
    config: ChartreuxConfigSchema,
    identity: str,
    parent: ToolManager | None = None,
) -> ToolManager:
    return ToolManager(
        lambda: config,
        cwd=tmp_path,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
        bound_identity=identity,
        parent_authority_getter=(lambda: parent) if parent is not None else None,
        parent_authority_revision_getter=((lambda: 0) if parent is not None else None),
    )


def _spec_names(manager: ToolManager) -> set[str]:
    return {spec.name for spec in manager.available_tool_specs()}


def test_vision_child_of_non_vision_parent_keeps_read_image(tmp_path: Path) -> None:
    """Non-vision parent + vision child: the child catalog stays truthful."""
    parent = _manager(tmp_path, _config("text"), "coder")
    child = _manager(tmp_path, _config("vision"), "researcher", parent)

    assert "read_image" not in _spec_names(parent)
    assert ReadImage.is_available(child._config)  # type: ignore[attr-defined]
    assert "read_image" in _spec_names(child)
    assert "read_image" in child.available_tools
    assert type(child.get("read_image")) is ReadImage


def test_non_vision_child_of_vision_parent_excludes_read_image(tmp_path: Path) -> None:
    """Vision parent + non-vision child: the child-side gate still applies."""
    parent = _manager(tmp_path, _config("vision"), "coder")
    child = _manager(tmp_path, _config("text"), "researcher", parent)

    assert "read_image" in _spec_names(parent)
    assert not ReadImage.is_available(child._config)  # type: ignore[attr-defined]
    assert "read_image" not in _spec_names(child)
    assert "read_image" not in child.available_tools
    with pytest.raises(NoSuchToolError):
        child.get("read_image")


def test_parent_catalog_unaffected_by_child_wiring(tmp_path: Path) -> None:
    """The parent's own catalog keeps gating on the parent's own model."""
    parent = _manager(tmp_path, _config("text"), "coder")
    before_specs = _spec_names(parent)
    before_delegable = set(parent.delegable_tools)

    child = _manager(tmp_path, _config("vision"), "researcher", parent)
    assert "read_image" in _spec_names(child)

    assert _spec_names(parent) == before_specs
    assert set(parent.delegable_tools) == before_delegable
    assert "read_image" not in before_specs
    assert "read_image" not in before_delegable
    with pytest.raises(NoSuchToolError):
        parent.get("read_image")

    # A rootless vision manager keeps its own capability, parentless.
    root = _manager(tmp_path, _config("vision"), "researcher")
    assert "read_image" in _spec_names(root)


def test_authority_filtering_unchanged_under_the_capability_fix(tmp_path: Path) -> None:
    """Posture and ceiling carve-outs keep filtering on authority alone.

    A read-only parent's delegation posture still denies the mutating set,
    and the non-mutating set (including the capability-gated read_image)
    rides through — the capability fix widens no authority axis.
    """
    parent = _manager(tmp_path, _config("text"), "researcher")
    child = _manager(tmp_path, _config("vision"), "coder", parent)

    specs = _spec_names(child)
    for name in MUTATING_TOOLS:
        assert name not in specs, name
    for name in NON_MUTATING_TOOLS:
        assert name in specs, name

    # The parent's PERSONAL axis keeps the v0.5.9 execution carve-out
    # (identities.py:51): read-only posture does not hide its own writes.
    assert set(MUTATING_TOOLS) <= set(parent.available_tools)
    # ... while its delegation ceiling stays non-mutating.
    assert set(MUTATING_TOOLS).isdisjoint(set(parent.delegable_tools))


@pytest.mark.asyncio
async def test_non_vision_child_still_fails_at_launch_and_send_boundary(
    tmp_path: Path,
) -> None:
    """Enforcement stays where it already worked: launch, send, run()."""
    config = _config("text")
    model = config.get_active_model()
    message = LLMMessage(
        role=Role.user,
        content="look",
        images=[
            ImageAttachment(
                source=InlineImageSource(data=base64.b64encode(PNG).decode("ascii")),
                alias="text",
                mime_type="image/png",
            )
        ],
    )

    with pytest.raises(ImagesNotSupportedError):
        messages_for_backend([message], model)

    with pytest.raises(InvalidLaunchModelError, match="images"):
        _validate_history(config, model, [message])

    path = tmp_path / "image.png"
    path.write_bytes(PNG)
    tool = ReadImage(
        lambda: ReadImageConfig(),
        ReadImageState(),
        cwd=tmp_path,
        runtime_config_getter=lambda: config,
    )
    with pytest.raises(ToolError, match="vision-capable"):
        await collect_result(tool.run(ReadImageArgs(file_path=str(path))))
