from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.app_server._workspace import (
    PromptPreparationError,
    WorkspaceTrustError,
    decide_workspace_trust,
    prepare_prompt,
    read_untrusted_config_dirs,
)
from chartreux.app_server.models import ImageAttachment
from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.llm_models import Backend
from chartreux.core.paths import TRUSTED_FOLDERS_FILE
from chartreux.core.trusted_folders import TrustedFoldersManager
from chartreux.utils.images import MAX_IMAGE_BYTES, MAX_IMAGES_PER_MESSAGE
from tests.conftest import build_test_agent_loop, build_test_vibe_config

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _vision_config(*, supports_images: bool = True, display_name: str | None = None):
    models = [
        ModelConfig(
            name="mistral-vibe-cli-latest",
            provider="mistral",
            alias="devstral-latest",
            display_name=display_name,
            supports_images=supports_images,
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
        active_model="devstral-latest", models=models, providers=providers
    )


def test_prepare_prompt_snapshots_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shot.png").write_bytes(PNG_BYTES)
    agent_loop = build_test_agent_loop(config=_vision_config())

    prompt = prepare_prompt(agent_loop, "look at @shot.png")

    assert len(prompt.images) == 1
    assert isinstance(prompt.images[0], ImageAttachment)
    assert prompt.images[0].alias == "shot.png"
    assert prompt.images[0].mime_type == "image/png"
    assert prompt.mentions.count == 1
    assert prompt.mentions.context_types == {"image": 1}


def test_prepare_prompt_never_embeds_mentioned_file_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    marker = "private contents must cross the read_file permission boundary"
    (tmp_path / "secret.txt").write_text(marker, encoding="utf-8")
    agent_loop = build_test_agent_loop()
    message = "read @secret.txt"

    prompt = prepare_prompt(agent_loop, message)

    assert prompt.display_text == message
    assert prompt.prompt_text == message
    assert marker not in prompt.model_dump_json()


def test_prepare_prompt_rejects_too_many_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    mentions = []
    for index in range(MAX_IMAGES_PER_MESSAGE + 1):
        name = f"img{index}.png"
        (tmp_path / name).write_bytes(PNG_BYTES)
        mentions.append(f"@{name}")
    agent_loop = build_test_agent_loop(config=_vision_config())

    with pytest.raises(PromptPreparationError, match="Too many image attachments"):
        prepare_prompt(agent_loop, " ".join(mentions))


def test_prepare_prompt_rejects_images_for_text_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shot.png").write_bytes(PNG_BYTES)
    agent_loop = build_test_agent_loop(config=_vision_config(supports_images=False))

    with pytest.raises(
        PromptPreparationError, match="`mistral/default/mistral-vibe-cli-latest`"
    ):
        prepare_prompt(agent_loop, "look at @shot.png")


def test_prepare_prompt_rejection_names_the_display_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shot.png").write_bytes(PNG_BYTES)
    agent_loop = build_test_agent_loop(
        config=_vision_config(
            supports_images=False, display_name="glm-5.2 (Mistral Hosted)"
        )
    )

    with pytest.raises(
        PromptPreparationError,
        match=r"`mistral/default/mistral-vibe-cli-latest` does not support",
    ):
        prepare_prompt(agent_loop, "look at @shot.png")


def test_prepare_prompt_rejects_oversize_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("chartreux.core.session.image_snapshot.MAX_IMAGE_BYTES", 32)
    (tmp_path / "shot.png").write_bytes(PNG_BYTES + b"\x00" * 64)
    agent_loop = build_test_agent_loop(config=_vision_config())

    with pytest.raises(PromptPreparationError, match="Image is too large"):
        prepare_prompt(agent_loop, "look at @shot.png")


def test_image_limits_match_public_cli_contract() -> None:
    assert MAX_IMAGE_BYTES == 10 * 1024 * 1024
    assert MAX_IMAGES_PER_MESSAGE == 8


def test_decide_workspace_trust_preserves_validated_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("instructions\n", encoding="utf-8")
    trust_store = TrustedFoldersManager()
    validated_identity = (1, 1)
    replacement_identity = (2, 2)
    monkeypatch.setattr(
        "chartreux.app_server._workspace.filesystem_identity",
        lambda _path: validated_identity,
    )
    monkeypatch.setattr(
        "chartreux.core.trusted_folders.filesystem_identity",
        lambda _path: replacement_identity,
    )

    with pytest.raises(WorkspaceTrustError, match="Workspace changed"):
        decide_workspace_trust(
            tmp_path,
            "trust_cwd",
            trust_store,
            expected_cwd=tmp_path,
            expected_cwd_identity=validated_identity,
        )

    assert trust_store.is_trusted(tmp_path) is None


def test_decide_workspace_decline_preserves_validated_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("instructions\n", encoding="utf-8")
    trust_store = TrustedFoldersManager()
    validated_identity = (1, 1)
    replacement_identity = (2, 2)
    monkeypatch.setattr(
        "chartreux.app_server._workspace.filesystem_identity",
        lambda _path: validated_identity,
    )
    monkeypatch.setattr(
        "chartreux.core.trusted_folders.filesystem_identity",
        lambda _path: replacement_identity,
    )

    with pytest.raises(WorkspaceTrustError, match="Workspace changed"):
        decide_workspace_trust(
            tmp_path,
            "decline",
            trust_store,
            expected_cwd=tmp_path,
            expected_cwd_identity=validated_identity,
        )

    assert trust_store.is_explicitly_untrusted(tmp_path) is False


def test_decide_workspace_repo_trust_preserves_validated_cwd_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    cwd = repo_root / "workspace"
    (repo_root / ".git").mkdir(parents=True)
    (repo_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("instructions\n", encoding="utf-8")
    trust_store = TrustedFoldersManager()
    validated_identity = (1, 1)
    replacement_identity = (2, 2)
    monkeypatch.setattr(
        "chartreux.app_server._workspace.filesystem_identity",
        lambda _path: validated_identity,
    )
    monkeypatch.setattr(
        "chartreux.core.trusted_folders.filesystem_identity",
        lambda _path: replacement_identity,
    )

    with pytest.raises(WorkspaceTrustError, match="Workspace changed"):
        decide_workspace_trust(
            cwd,
            "trust_repo",
            trust_store,
            expected_cwd=cwd,
            expected_cwd_identity=validated_identity,
        )

    assert trust_store.is_trusted(repo_root) is None


def test_read_untrusted_config_dirs_reports_dirs_and_settings_path(
    tmp_path: Path,
) -> None:
    vibe_dir = tmp_path / ".chartreux"
    (vibe_dir / "tools").mkdir(parents=True)
    trust_store = TrustedFoldersManager()
    trust_store.add_trusted(tmp_path)
    trust_store.add_untrusted(vibe_dir)

    response = read_untrusted_config_dirs(tmp_path, trust_store)

    assert response.dirs == [str(vibe_dir.resolve())]
    assert response.settings_path == str(TRUSTED_FOLDERS_FILE.path)


def test_read_untrusted_config_dirs_empty_when_nothing_broken(tmp_path: Path) -> None:
    (tmp_path / ".chartreux" / "tools").mkdir(parents=True)
    trust_store = TrustedFoldersManager()
    trust_store.add_trusted(tmp_path)

    response = read_untrusted_config_dirs(tmp_path, trust_store)

    assert response.dirs == []
