from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from chartreux.core import env
from chartreux.core.config.fingerprint import create_file_fingerprint
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.config.layer import (
    ConfigLayer,
    RawConfig,
    TrustStoreStateError,
    UntrustedLayerError,
    require_answerable_trust_store,
)
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.patch import AddOperationPatch, ConfigPatch
from chartreux.core.config.types import (
    MISSING_BACKING_STORE_DATA_FINGERPRINT,
    LayerConfigSnapshot,
)
from chartreux.core.paths._vibe_home import GlobalPath
from chartreux.core.trusted_folders import (
    TrustedFoldersManager,
    TrustStoreState,
    trusted_folders_manager,
)


@pytest.mark.asyncio
async def test_reads_toml_when_trusted(tmp_working_directory: Path) -> None:
    trusted_folders_manager.add_trusted(tmp_working_directory)
    config_path = tmp_working_directory / ".chartreux" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('theme = "dark"\n')

    layer = ProjectConfigLayer(path=tmp_working_directory)
    data = await layer.load()
    assert data.model_extra == {"theme": "dark"}
    fp1 = layer.fingerprint
    assert isinstance(fp1, str)
    assert fp1

    config_path.unlink()
    data = await layer.load(force=True)
    assert data.model_extra == {}
    assert layer.fingerprint == MISSING_BACKING_STORE_DATA_FINGERPRINT


@pytest.mark.asyncio
async def test_duplicate_model_aliases_dedup_last_wins(
    tmp_working_directory: Path,
) -> None:
    trusted_folders_manager.add_trusted(tmp_working_directory)
    config_path = tmp_working_directory / ".chartreux" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "[[models]]\n"
        'name = "devstral-small-latest"\n'
        'provider = "mistral"\n'
        'alias = "devstral-small"\n'
        "\n"
        "[[models]]\n"
        'name = "devstral-small-next"\n'
        'provider = "mistral"\n'
        'alias = "devstral-small"\n'
    )

    layer = ProjectConfigLayer(path=tmp_working_directory)
    data = await layer.load()

    assert data.model_extra is not None
    models = data.model_extra["models"]
    assert list(models) == ["devstral-small"]
    assert models["devstral-small"]["name"] == "devstral-small-next"


@pytest.mark.asyncio
async def test_untrusted_raises(tmp_working_directory: Path) -> None:
    trusted_folders_manager.add_untrusted(tmp_working_directory)
    config_path = tmp_working_directory / ".chartreux" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('theme = "dark"\n')

    layer = ProjectConfigLayer(path=tmp_working_directory)
    with pytest.raises(UntrustedLayerError):
        await layer.load()


@pytest.mark.asyncio
async def test_unresolved_trust_defaults_to_untrusted(
    tmp_working_directory: Path,
) -> None:
    config_path = tmp_working_directory / ".chartreux" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('key = "value"\n')

    layer = ProjectConfigLayer(path=tmp_working_directory)
    with pytest.raises(UntrustedLayerError):
        await layer.load()


@pytest.mark.asyncio
async def test_resolve_and_load_never_mutate_trust_store_when_untrusted(
    tmp_working_directory: Path,
) -> None:
    """Resolving/loading must be a pure read of the trust store.

    Trust resolution used to write back into the store, marking the layer's
    ``.chartreux`` directory untrusted while the parent stayed trusted. That silently
    disabled project config on the next launch. Neither ``resolve_trust`` nor a
    failed ``load`` may touch the store.
    """
    config_path = tmp_working_directory / ".chartreux" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('theme = "dark"\n')

    trusted_before = list(trusted_folders_manager._trusted)
    untrusted_before = list(trusted_folders_manager._untrusted)

    layer = ProjectConfigLayer(path=tmp_working_directory)
    assert await layer.resolve_trust() is False
    with pytest.raises(UntrustedLayerError):
        await layer.load()

    assert trusted_folders_manager.is_explicitly_untrusted(config_path.parent) is False
    assert trusted_folders_manager._trusted == trusted_before
    assert trusted_folders_manager._untrusted == untrusted_before


@pytest.mark.asyncio
async def test_resolve_and_load_do_not_untrust_vibe_dir_under_untrusted_parent(
    tmp_working_directory: Path,
) -> None:
    """Trust resolution is a pure read even with an explicitly untrusted parent.

    Distrusting the workspace must not cascade a fresh untrusted entry onto its
    ``.chartreux`` directory; only the entry the user set may exist afterwards.
    """
    trusted_folders_manager.add_untrusted(tmp_working_directory)
    config_path = tmp_working_directory / ".chartreux" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('theme = "dark"\n')

    untrusted_before = list(trusted_folders_manager._untrusted)

    layer = ProjectConfigLayer(path=tmp_working_directory)
    assert await layer.resolve_trust() is False
    with pytest.raises(UntrustedLayerError):
        await layer.load()

    assert trusted_folders_manager.is_explicitly_untrusted(config_path.parent) is False
    assert trusted_folders_manager._untrusted == untrusted_before


@pytest.mark.asyncio
async def test_missing_file_returns_empty(tmp_working_directory: Path) -> None:
    trusted_folders_manager.add_trusted(tmp_working_directory)

    layer = ProjectConfigLayer(path=tmp_working_directory)
    data = await layer.load()
    assert data.model_extra == {}
    assert layer.fingerprint == MISSING_BACKING_STORE_DATA_FINGERPRINT


@pytest.mark.asyncio
async def test_default_name(tmp_working_directory: Path) -> None:
    layer = ProjectConfigLayer(path=tmp_working_directory)
    assert layer.name == "project-toml"


@pytest.mark.asyncio
async def test_trust_uses_path_parent_for_resolution(
    tmp_working_directory: Path,
) -> None:
    project_dir = tmp_working_directory / "my-project"
    project_dir.mkdir()

    config_path = project_dir / ".chartreux" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('value = "trusted"\n')
    trusted_folders_manager.add_trusted(project_dir / ".chartreux")

    layer = ProjectConfigLayer(path=project_dir)
    data = await layer.load()
    assert data.model_extra == {"value": "trusted"}


@pytest.mark.asyncio
async def test_find_file_prefers_closest_config(tmp_working_directory: Path) -> None:
    parent_config = tmp_working_directory / ".chartreux" / "config.toml"
    parent_config.parent.mkdir(parents=True, exist_ok=True)
    parent_config.write_text('level = "parent"\n')

    child = tmp_working_directory / "child"
    child.mkdir()
    child_config = child / ".chartreux" / "config.toml"
    child_config.parent.mkdir(parents=True, exist_ok=True)
    child_config.write_text('level = "child"\n')

    trusted_folders_manager.add_trusted(child)
    layer = ProjectConfigLayer(path=child)
    data = await layer.load()
    assert data.model_extra == {"level": "child"}


@pytest.mark.asyncio
async def test_find_file_result_is_cached(tmp_working_directory: Path) -> None:
    config_path = tmp_working_directory / ".chartreux" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("x = 1\n")
    trusted_folders_manager.add_trusted(tmp_working_directory)

    layer = ProjectConfigLayer(path=tmp_working_directory)
    await layer.load()

    assert layer._is_set is True
    cached_path = layer._config_file_path
    await layer.resolve_trust()
    assert layer._config_file_path is cached_path


@pytest.mark.asyncio
async def test_is_file_discovered_reflects_cached_discovery_state(
    tmp_working_directory: Path,
) -> None:
    trusted_folders_manager.add_trusted(tmp_working_directory)
    config_path = tmp_working_directory / ".chartreux" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('theme = "dark"\n')

    layer = ProjectConfigLayer(path=tmp_working_directory)

    assert layer.is_file_discovered is False

    await layer.load()

    assert layer.is_file_discovered is True


@pytest.mark.asyncio
async def test_config_file_added_after_first_search_is_not_detected(
    tmp_working_directory: Path,
) -> None:
    layer = ProjectConfigLayer(path=tmp_working_directory)
    await layer._find_config_file()

    assert layer._is_set is True
    assert layer._config_file_path is None

    config_path = tmp_working_directory / ".chartreux" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('theme = "dark"\n')

    await layer._find_config_file()

    assert layer._config_file_path is None


@pytest.mark.asyncio
async def test_finds_config_in_parent_directory(tmp_working_directory: Path) -> None:
    trusted_folders_manager.add_trusted(tmp_working_directory)
    config_path = tmp_working_directory / ".chartreux" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('theme = "dark"\n')

    subdir = tmp_working_directory / "sub" / "project"
    subdir.mkdir(parents=True)

    layer = ProjectConfigLayer(path=subdir)
    data = await layer.load()
    assert data.model_extra == {"theme": "dark"}


@pytest.mark.asyncio
async def test_trusted_ancestor_satisfies_trust_check(
    tmp_working_directory: Path,
) -> None:
    parent = tmp_working_directory / "workspace"
    parent.mkdir()
    child = parent / "project"
    child.mkdir()

    trusted_folders_manager.add_trusted(parent)

    config_path = child / ".chartreux" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('theme = "dark"\n')

    layer = ProjectConfigLayer(path=child)
    data = await layer.load()
    assert data.model_extra == {"theme": "dark"}


@pytest.mark.asyncio
async def test_walk_stops_at_vibe_home_parent(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create an isolated tmp root and point CHARTREUX_HOME into it
    tmp_root = tmp_path_factory.mktemp("walk_stop")
    fake_vibe_home = tmp_root / ".chartreux"
    fake_vibe_home.mkdir()
    monkeypatch.setattr(
        "chartreux.core.config.layers.project.CHARTREUX_HOME",
        GlobalPath(lambda: fake_vibe_home),
    )

    # Place a config at tmp_root/.chartreux/config.toml — should NOT be picked up
    home_config = fake_vibe_home / "config.toml"
    home_config.write_text('theme = "dark"\n')

    # subdir lives inside tmp_root so the walk would reach the config without the stop guard
    subdir = tmp_root / "vibe-test-project"
    subdir.mkdir()
    trusted_folders_manager.add_trusted(subdir)

    layer = ProjectConfigLayer(path=subdir)
    data = await layer.load()
    assert data.model_extra == {}


@pytest.mark.asyncio
async def test_apply_creates_file_when_none_discovered(
    tmp_working_directory: Path,
) -> None:
    # No .chartreux/config.toml exists; the layer is still trusted (nothing to distrust).
    layer = ProjectConfigLayer(path=tmp_working_directory)
    await layer.load()
    assert layer.fingerprint == MISSING_BACKING_STORE_DATA_FINGERPRINT

    await layer.apply(
        ConfigPatch(
            AddOperationPatch(path="/theme", value="dark"),
            fingerprint=MISSING_BACKING_STORE_DATA_FINGERPRINT,
        )
    )

    created_path = tmp_working_directory / ".chartreux" / "config.toml"
    with created_path.open("rb") as file:
        assert tomllib.load(file) == {"theme": "dark"}
        assert layer.fingerprint == create_file_fingerprint(file)

    assert layer.is_file_discovered
    assert layer.config_file_path == created_path


def _malformed_trust_store(config_dir: Path) -> TrustedFoldersManager:
    """A store that exists but cannot be parsed, loaded from that state."""
    (config_dir / "trusted_folders.toml").write_text("[malformed")
    store = TrustedFoldersManager()
    assert store.store_state is TrustStoreState.MALFORMED
    return store


class _StoreBackedLayer(ConfigLayer[RawConfig]):
    """A minimal layer whose trust check goes through the shared guard."""

    def __init__(self, trust_store: TrustedFoldersManager) -> None:
        super().__init__(name="store-backed")
        self._trust_store = trust_store

    async def _build_config_snapshot(self) -> LayerConfigSnapshot:
        raise AssertionError("no snapshot is read from an unusable store")

    async def _save_to_store(self, _next_config: RawConfig) -> str:
        raise AssertionError("nothing is written to an unusable store")

    async def _check_trust(self) -> bool:
        require_answerable_trust_store(self._trust_store, self.name)
        return True


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["env", "layer", "project", "harness-files"])
async def test_group_c_consumer_never_reswallows_an_unusable_trust_store(
    consumer: str, tmp_path: Path, config_dir: Path
) -> None:
    """OBSERVABLE: no group-C consumer answers a trust question from a broken store.

    A malformed store answers ``is_trusted``/``find_trust_root`` with "nothing
    is trusted", so every consumer below would otherwise proceed with a
    definite answer it cannot support: the project-env API would skip the
    trusted folder's ``.env``, the project config layer would read the folder as
    explicitly untrusted (or, with a stale cached verdict, as trusted), and the
    harness-files manager would load a folder's instructions from an unknown
    state (the directory listing stays fail-closed, contributing no root). Each
    surfaces the typed store state instead — ``TrustStoreStateError`` — and the
    failure path never grants trust. The layer case additionally pins that the
    base-class trust dispatch re-raises the typed error instead of collapsing it
    into ``LayerImplementationError``. Re-swallowing the error in any consumer
    fails this node.
    """
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        "PROJECT_KEY=sk-project-0123456789\n", encoding="utf-8"
    )
    config_file = project / ".chartreux" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('theme = "dark"\n', encoding="utf-8")
    store = _malformed_trust_store(config_dir)

    if consumer == "env":
        with pytest.raises(TrustStoreStateError, match="trust store malformed"):
            env.project_env_for_cwd(project, manager=store)
        with pytest.raises(TrustStoreStateError, match="trust store malformed"):
            env.project_env_file(project, manager=store)
        assert store.is_trusted(project) is not True
    elif consumer == "layer":
        layer = _StoreBackedLayer(store)
        with pytest.raises(TrustStoreStateError, match="trust store malformed"):
            await layer.resolve_trust()
        with pytest.raises(TrustStoreStateError, match="trust store malformed"):
            await layer.load()
        assert store.is_trusted(project) is not True
    elif consumer == "project":
        layer = ProjectConfigLayer(path=project, trust_store=store)
        with pytest.raises(TrustStoreStateError, match="trust store malformed"):
            await layer.load()
        with pytest.raises(TrustStoreStateError, match="trust store malformed"):
            await layer.resolve_trust()
        assert store.is_trusted(config_file.parent) is not True
    else:
        manager = HarnessFilesManager(
            sources=("project",), cwd=project, trust_store=store
        )
        with pytest.raises(TrustStoreStateError, match="trust store malformed"):
            manager.load_instruction_documents()
        with pytest.raises(TrustStoreStateError, match="trust store malformed"):
            manager.load_project_docs()
        # The listing stays fail-closed: an unknown state contributes no root,
        # so the project surface is never enabled from an unanswered question.
        assert manager.project_roots == []
        assert store.is_trusted(project) is not True


@pytest.mark.asyncio
async def test_group_c_missing_store_stays_the_first_run_default(
    tmp_path: Path,
) -> None:
    """OBSERVABLE: ``MISSING`` is not an error — nothing persisted, nothing trusted.

    The first launch of a fresh install has no store at all, and every group-C
    consumer must keep its pre-existing untrusted answer rather than raise.
    Only unreadable, malformed, and invalid stores are errors.
    """
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        "PROJECT_KEY=sk-project-0123456789\n", encoding="utf-8"
    )
    config_file = project / ".chartreux" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('theme = "dark"\n', encoding="utf-8")
    store = TrustedFoldersManager()
    assert store.store_state is TrustStoreState.MISSING

    assert env.project_env_for_cwd(project, manager=store) == {}
    layer = ProjectConfigLayer(path=project, trust_store=store)
    with pytest.raises(UntrustedLayerError):
        await layer.load()
    manager = HarnessFilesManager(sources=("project",), cwd=project, trust_store=store)
    assert manager.project_roots == []
    assert store.is_trusted(project) is None
