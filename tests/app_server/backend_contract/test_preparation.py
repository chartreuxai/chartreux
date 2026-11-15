from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
import tomllib

import pytest
import pytest_asyncio
import respx
import tomli_w

from chartreux.app_server.models import FileImageSource, TextContentBlock
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ProtocolErrorCode,
    SessionOptions,
    WorkspacePromptPrepareParams,
    WorkspacePromptPrepareResponse,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import STDIO_TRANSPORT
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.models import (
    ModelConfig,
    ProviderConfig,
    SessionLoggingConfig,
)
from chartreux.core.llm_models import Backend
from chartreux.utils.images import MAX_IMAGES_PER_MESSAGE
from tests.app_server.backend_contract.conftest import (
    BackendContractConnection,
    connect_backend_contract_host,
    connect_backend_contract_stdio_host,
    install_memory_root_agent,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

_VISION_ALIAS = "vision"


def _vision_config(*, save_dir: str) -> object:
    models = [
        ModelConfig(
            name="vision-model",
            provider="mistral",
            alias=_VISION_ALIAS,
            supports_images=True,
        )
    ]
    providers = [
        ProviderConfig(
            name="mistral",
            api_base="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            backend=Backend.MISTRAL,
        )
    ]
    return build_test_vibe_config(
        active_model=_VISION_ALIAS,
        models=models,
        providers=providers,
        session_logging=SessionLoggingConfig(enabled=True, save_dir=save_dir),
    )


def _prepare_params(
    session_id: str, message: str, **kwargs: object
) -> WorkspacePromptPrepareParams:
    return WorkspacePromptPrepareParams(
        session_id=session_id,
        message=message,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_prompt_prepare_returns_the_prepared_prompt_on_the_selected_transport(
    backend_contract_transport_connection: BackendContractConnection,
    backend_contract_transport_session: AppServerSession,
    backend_contract_mistral_api: respx.Route,
) -> None:
    message = "plain prompt with no mentions"
    result = await backend_contract_transport_connection.client.request(
        "workspace/prompt/prepare",
        _prepare_params(
            backend_contract_transport_session.session_id,
            message,
            title_content=[TextContentBlock(text="a title candidate")],
        ),
    )

    response = WorkspacePromptPrepareResponse.model_validate(result)
    assert response.prompt.display_text == message
    assert response.prompt.prompt_text == message
    assert response.prompt.images == []
    assert response.prompt.auto_title is None
    assert response.prompt.mentions.count == 0


@pytest.mark.asyncio
async def test_prompt_prepare_resolves_mentions_against_the_session_cwd(
    backend_contract_transport_session: AppServerSession, tmp_working_directory: Path
) -> None:
    marker = "mentioned file contents never cross into the prepared prompt"
    (tmp_working_directory / "notes.txt").write_text(marker, encoding="utf-8")
    message = "read @notes.txt please"

    prompt = (
        await backend_contract_transport_session.resources.workspace.prepare_prompt(
            message
        )
    )

    assert prompt.display_text == message
    assert prompt.prompt_text == message
    assert marker not in prompt.model_dump_json()
    assert prompt.mentions.count == 1
    assert prompt.mentions.context_types == {"file": 1}
    assert prompt.mentions.file_extensions == {".txt": 1}


@pytest_asyncio.fixture
async def preparation_persistent_connection(
    monkeypatch: pytest.MonkeyPatch,
    config_dir: Path,
    tmp_path: Path,
    stdio_after_autouse: str,
    backend_contract_session_options: SessionOptions,
    backend_contract_capabilities: ClientCapabilities,
) -> AsyncIterator[BackendContractConnection]:
    """One transport-parameterized connection with session logging enabled.

    Session logging gives the session a real directory, so an attachment
    snapshot is a filesystem write into the agent-owned session dir -- the
    leg that crosses the process boundary on stdio.
    """
    save_dir = tmp_path / "sessions"
    config_file = config_dir / "config.toml"
    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    config["session_logging"] = {"enabled": True, "save_dir": str(save_dir)}
    config_file.write_text(tomli_w.dumps(config), encoding="utf-8")
    # An image-capable main model on the harness (in-process) leg; the stdio
    # leg's memory root gets the same model through its own config.
    (config_dir / "models.toml").write_text(
        tomli_w.dumps({
            "models": {
                _VISION_ALIAS: {
                    "deployments": [
                        {
                            "provider": "mistral",
                            "name": "vision-model",
                            "supports_images": True,
                        }
                    ]
                }
            },
            "roles": {"main": {"model": _VISION_ALIAS, "thinking": "medium"}},
        }),
        encoding="utf-8",
    )

    loop: AgentLoop | None = None
    if stdio_after_autouse == STDIO_TRANSPORT:
        loop = build_test_agent_loop(
            config=_vision_config(save_dir=str(save_dir))  # type: ignore[arg-type]
        )
        install_memory_root_agent(monkeypatch, loop)
        connection = await connect_backend_contract_stdio_host(
            session_options=backend_contract_session_options,
            capabilities=backend_contract_capabilities,
        )
    else:
        connection = await connect_backend_contract_host(
            session_options=backend_contract_session_options,
            capabilities=backend_contract_capabilities,
        )
    try:
        yield connection
    finally:
        await connection.host.close()
        if loop is not None:
            await loop.aclose()


@pytest_asyncio.fixture
async def preparation_persistent_session(
    preparation_persistent_connection: BackendContractConnection,
    backend_contract_mistral_api: respx.Route,
) -> AsyncIterator[AppServerSession]:
    session = await preparation_persistent_connection.host.open_session()
    try:
        yield session
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_prompt_prepare_writes_the_attachment_snapshot_into_the_session_dir(
    preparation_persistent_session: AppServerSession,
    tmp_working_directory: Path,
    tmp_path: Path,
) -> None:
    (tmp_working_directory / "shot.png").write_bytes(PNG_BYTES)
    original = (tmp_working_directory / "shot.png").resolve()

    prompt = await preparation_persistent_session.resources.workspace.prepare_prompt(
        "look at @shot.png"
    )

    assert len(prompt.images) == 1
    image = prompt.images[0]
    assert image.alias == "shot.png"
    assert image.mime_type == "image/png"
    assert isinstance(image.source, FileImageSource)
    snapshot = Path(image.source.path)
    assert snapshot.is_file()
    assert snapshot != original
    assert snapshot.read_bytes() == PNG_BYTES
    # The snapshot landed under the session directory the agent owns, not
    # next to the mentioned file: the attachment write crossed the boundary.
    assert snapshot.parent.name == "attachments"
    assert tmp_path / "sessions" in snapshot.parents
    assert prompt.mentions.count == 1
    assert prompt.mentions.context_types == {"image": 1}


@pytest.mark.asyncio
async def test_prompt_prepare_rejects_too_many_attachments_typed_on_both_transports(
    backend_contract_transport_connection: BackendContractConnection,
    backend_contract_transport_session: AppServerSession,
    tmp_working_directory: Path,
) -> None:
    mentions = []
    for index in range(MAX_IMAGES_PER_MESSAGE + 1):
        name = f"img{index}.png"
        (tmp_working_directory / name).write_bytes(PNG_BYTES)
        mentions.append(f"@{name}")

    with pytest.raises(AppServerResponseError) as exc_info:
        await backend_contract_transport_connection.client.request(
            "workspace/prompt/prepare",
            _prepare_params(
                backend_contract_transport_session.session_id, " ".join(mentions)
            ),
        )

    assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
    assert "Too many image attachments" in exc_info.value.error.message


@pytest.mark.asyncio
async def test_prompt_prepare_rejects_an_oversize_attachment_typed_on_both_transports(
    monkeypatch: pytest.MonkeyPatch,
    backend_contract_transport_connection: BackendContractConnection,
    backend_contract_transport_session: AppServerSession,
    tmp_working_directory: Path,
) -> None:
    monkeypatch.setattr("chartreux.core.session.image_snapshot.MAX_IMAGE_BYTES", 32)
    (tmp_working_directory / "shot.png").write_bytes(PNG_BYTES + b"\x00" * 64)

    with pytest.raises(AppServerResponseError) as exc_info:
        await backend_contract_transport_connection.client.request(
            "workspace/prompt/prepare",
            _prepare_params(
                backend_contract_transport_session.session_id, "look at @shot.png"
            ),
        )

    assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
    assert "Image is too large" in exc_info.value.error.message


@pytest.mark.asyncio
async def test_prompt_prepare_rejects_an_unknown_session_on_both_transports(
    backend_contract_transport_connection: BackendContractConnection,
    backend_contract_transport_session: AppServerSession,
) -> None:
    with pytest.raises(AppServerResponseError) as exc_info:
        await backend_contract_transport_connection.client.request(
            "workspace/prompt/prepare", _prepare_params("no-such-session", "hello")
        )

    assert exc_info.value.error.code is ProtocolErrorCode.NOT_FOUND
    assert "no-such-session" in exc_info.value.error.message


@pytest.mark.asyncio
async def test_prompt_prepare_rejects_malformed_params_on_both_transports(
    backend_contract_transport_connection: BackendContractConnection,
    backend_contract_transport_session: AppServerSession,
) -> None:
    with pytest.raises(AppServerResponseError) as exc_info:
        await backend_contract_transport_connection.client.request(
            "workspace/prompt/prepare",
            {"sessionId": backend_contract_transport_session.session_id},
        )

    assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS


@pytest.mark.asyncio
async def test_prompt_prepare_requires_an_attached_session_on_both_transports(
    backend_contract_transport_connection: BackendContractConnection,
) -> None:
    with pytest.raises(AppServerResponseError) as exc_info:
        await backend_contract_transport_connection.client.request(
            "workspace/prompt/prepare", _prepare_params("any-session", "hello")
        )

    assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT


@pytest.mark.asyncio
async def test_prompt_prepare_rejects_images_for_a_text_only_model_on_both_transports(
    backend_contract_transport_connection: BackendContractConnection,
    backend_contract_transport_session: AppServerSession,
    tmp_working_directory: Path,
) -> None:
    """The default test model takes no images; the agent-side model check must
    answer the same typed refusal on both transports.
    """
    (tmp_working_directory / "shot.png").write_bytes(PNG_BYTES)

    with pytest.raises(AppServerResponseError) as exc_info:
        await backend_contract_transport_connection.client.request(
            "workspace/prompt/prepare",
            _prepare_params(
                backend_contract_transport_session.session_id, "look at @shot.png"
            ),
        )

    assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
    assert "does not support images" in exc_info.value.error.message
