from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from git import Repo
import pytest

from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server._handler import CoreRequestHandler
from chartreux.app_server._host import HostRequestHandler
from chartreux.app_server._workspace import (
    PromptPreparationError,
    WorkspaceTrustError,
    decide_workspace_trust,
    prepare_prompt,
    read_untrusted_config_dirs,
    read_workspace_trust,
)
from chartreux.app_server.models import ImageAttachment
from chartreux.app_server.protocol import ProtocolErrorCode
from chartreux.core.agent_loop.errors import AgentLoopStateError
from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.git.worktree import WorktreeRepository
from chartreux.core.llm_models import Backend
from chartreux.core.paths import TRUSTED_FOLDERS_FILE
from chartreux.core.trusted_folders import (
    TrustedFoldersManager,
    TrustStoreState,
    WorkspaceTrustStatus,
)
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
        PromptPreparationError, match="`mistral/mistral-vibe-cli-latest`"
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
        match=r"`mistral/mistral-vibe-cli-latest` does not support",
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


def _corrupt_trust_store(config_dir: Path) -> TrustedFoldersManager:
    """A store that exists but cannot be parsed, loaded from that state."""
    (config_dir / "trusted_folders.toml").write_text("[malformed")
    store = TrustedFoldersManager()
    assert store.store_state is TrustStoreState.MALFORMED
    return store


def _repo_and_worktree(root: Path) -> tuple[Path, Path]:
    """A repository and a real linked worktree of it.

    A move only accepts a destination derived from the session's own
    repository, so the refusal has to be reached from a directory the move
    would otherwise accept — a bare pair of directories would refuse for the
    wrong reason.
    """
    repo = root / "repo"
    repo.mkdir()
    git_repo = Repo.init(repo, initial_branch="main")
    git_repo.config_writer().set_value("user", "name", "Tester").release()
    git_repo.config_writer().set_value("user", "email", "t@example.com").release()
    (repo / "file.txt").write_text("hello\n")
    git_repo.index.add(["file.txt"])
    git_repo.index.commit("initial")
    with WorktreeRepository.open(repo) as repository:
        prepared = repository.prepare("feature")
    return repo, prepared.path.resolve()


def _session_decision_handler(
    workspace: Path, trust_store: TrustedFoldersManager
) -> CoreRequestHandler:
    """The session-scope decision path with only its own seam populated.

    That path reads the loop's cwd, the loop's harness files, the session
    holder and the idle gate; building the rest of the handler would add
    nothing this node can observe.
    """
    handler = CoreRequestHandler.__new__(CoreRequestHandler)
    handler._agent_loop = SimpleNamespace(  # type: ignore[assignment]
        cwd=workspace,
        harness_files=HarnessFilesManager(sources=(), trust_store=trust_store),
    )
    handler._execution = SimpleNamespace(  # type: ignore[assignment]
        active=None, require_idle=lambda: None
    )
    handler._require_session = lambda _session_id: None  # type: ignore[assignment]
    return handler


def test_workspace_decision_translates_a_refused_persist(
    tmp_path: Path, config_dir: Path
) -> None:
    """A write the store refuses is an error, never a decision that was made.

    The store loads as missing (the genuine first-run default), so the
    usability guard passes; a cooperating writer then makes it unusable before
    the decision is persisted. The checked write refuses and reports that only
    to the core apply path, which the workspace API would otherwise hand back
    as an unchanged status — as if the decision had been accepted and simply
    had no effect. It is translated instead, and the store is left alone.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("instructions\n", encoding="utf-8")
    trust_store = TrustedFoldersManager()
    assert trust_store.store_state is TrustStoreState.MISSING
    store_path = config_dir / "trusted_folders.toml"
    store_path.write_text("[malformed")

    with pytest.raises(WorkspaceTrustError, match="was not saved"):
        decide_workspace_trust(workspace, "trust_cwd", trust_store)

    assert trust_store.is_trusted(workspace) is None
    assert store_path.read_text() == "[malformed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "consumer", ["workspace-api", "host-dispatch", "session-decision", "relocation"]
)
async def test_group_b_consumer_never_reswallows_an_unusable_trust_store(
    consumer: str, tmp_path: Path, config_dir: Path
) -> None:
    """OBSERVABLE: no group-B consumer answers a trust question from a broken store.

    A malformed store reads as "nothing is trusted" in the store's in-memory
    lookups, so every consumer below would otherwise report a definite answer
    it cannot support: the workspace API would return a status with trust
    details (and accept a decision it cannot persist), the host and the session
    handler would hand the caller a normal response, and a workspace move would
    grant session trust and bind a destination whose project configuration is
    unknown. Each surfaces the typed store state instead — ``WorkspaceTrustError``
    from the workspace API, ``RequestFailure(INVALID_PARAMS)`` from both server
    paths, ``AgentLoopStateError`` from the move — and re-swallowing the error
    in any of them fails this node.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("instructions\n", encoding="utf-8")
    trust_store = _corrupt_trust_store(config_dir)

    if consumer == "workspace-api":
        calls = (
            lambda: read_workspace_trust(workspace, trust_store),
            lambda: read_untrusted_config_dirs(workspace, trust_store),
            lambda: decide_workspace_trust(workspace, "trust_cwd", trust_store),
        )
        for call in calls:
            with pytest.raises(WorkspaceTrustError, match="malformed"):
                call()
    elif consumer == "host-dispatch":
        host = HostRequestHandler(
            HarnessFilesManager(sources=(), trust_store=trust_store)
        )
        with pytest.raises(RequestFailure) as failure:
            await host.dispatch("workspace/trust/status", {"cwd": str(workspace)})
        assert failure.value.code is ProtocolErrorCode.INVALID_PARAMS
        assert "malformed" in str(failure.value)
    elif consumer == "session-decision":
        handler = _session_decision_handler(workspace, trust_store)
        with pytest.raises(RequestFailure) as failure:
            await handler.dispatch(
                "workspace/trust/decision",
                {"decision": "trust_cwd", "cwd": str(workspace), "sessionId": "s1"},
            )
        assert failure.value.code is ProtocolErrorCode.INVALID_PARAMS
        assert "malformed" in str(failure.value)
    else:
        repo, worktree = _repo_and_worktree(tmp_path)
        agent_loop = build_test_agent_loop(
            cwd=repo,
            harness_files=HarnessFilesManager(
                sources=("project",), trust_store=trust_store
            ).for_session(repo, workspace_roots=[repo]),
        )
        try:
            with pytest.raises(AgentLoopStateError, match="trust store is malformed"):
                await agent_loop.relocate(worktree)

            assert agent_loop.cwd == repo
            assert agent_loop.harness_files.cwd == repo
            assert (
                trust_store.trust_status(worktree) is not WorkspaceTrustStatus.SESSION
            )
        finally:
            await agent_loop.aclose()
