from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
import stat
import threading
import tomllib
from typing import IO

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._dangerous_command_authority import GRANTS_FIELD
from chartreux.core.config._dangerous_command_persistence import (
    SavedGrantsRead,
    read_saved_dangerous_command_grants,
)
from chartreux.core.config.fingerprint import create_file_fingerprint
from chartreux.core.config.layers import _base
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import (
    ConfigOrchestrator,
    ConfigPatchValidationError,
)
from chartreux.core.config.patch import AddOperationPatch
from chartreux.core.config.types import MISSING_BACKING_STORE_DATA_FINGERPRINT
from chartreux.core.trusted_folders import trusted_folders_manager

PUSH = "git.push"


async def make_orchestrator(path: Path) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    user = UserConfigLayer(path=path)
    session = OverridesLayer(data={})
    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, session],
        default_layer_resolver=lambda: session,
    )


def live_state(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> tuple[object, object, object, object]:
    layer = orchestrator.get_layer("user-toml")
    return (
        orchestrator.config,
        orchestrator.restrictions,
        orchestrator.accepted_token,
        (layer.cached_data, layer.fingerprint),
    )


def _external_write_between_revision_reads(
    path: Path, external: str
) -> Callable[[IO[bytes]], str]:
    """Build a fingerprint hook that rewrites *path* on the second target read.

    The checked write fingerprints the target twice: once before the temp
    write and once under the lock immediately before the replace. The second
    read observes *external*, exactly as if a non-cooperating writer replaced
    the store during the temp write.
    """
    original = _base.create_file_fingerprint
    target_reads = 0

    def intervening(file: IO[bytes]) -> str:
        nonlocal target_reads
        if Path(file.name) == path:
            target_reads += 1
            if target_reads == 2:
                path.write_text(external)
        return original(file)

    return intervening


@pytest.mark.asyncio
async def test_saves_grant_preserving_disk_content_without_publishing(
    tmp_path: Path,
) -> None:
    other = tmp_path / "other-project"
    path = tmp_path / "config.toml"
    path.write_text(
        'theme = "dark"\n'
        '\n[tools.bash]\ndenylist = ["user-denial"]\n'
        "\n"
        f"[{GRANTS_FIELD}]\n"
        f'"{other}" = ["{PUSH}"]\n'
    )
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)
    grants_before = orchestrator.config.dangerous_command_grants_by_project

    result = await orchestrator.save_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )

    assert result.error is None
    assert result.persistence == "saved"
    assert result.application == "unchanged"
    with path.open("rb") as file:
        assert result.revision == create_file_fingerprint(file)
    data = tomllib.loads(path.read_text())
    assert data["theme"] == "dark"
    assert data["tools"] == {"bash": {"denylist": ["user-denial"]}}
    assert data[GRANTS_FIELD] == {str(other): [PUSH], str(tmp_path): [PUSH]}
    # Runtime publication never happened: the accepted snapshot, restrictions,
    # token, live layer cache, and the effective projection all still describe
    # the pre-save state.
    assert live_state(orchestrator) == before
    assert orchestrator.config.dangerous_command_grants_by_project == grants_before
    assert orchestrator.restrictions[-1].dangerous_command_grants == ()


@pytest.mark.asyncio
async def test_absent_user_config_file_is_created(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    assert layer.fingerprint == MISSING_BACKING_STORE_DATA_FINGERPRINT
    before = live_state(orchestrator)

    result = await orchestrator.save_dangerous_command_grant(
        project=tmp_path,
        operation_id=PUSH,
        expected_revision=MISSING_BACKING_STORE_DATA_FINGERPRINT,
    )

    assert result.error is None
    assert result.persistence == "saved"
    assert result.application == "unchanged"
    assert path.exists()
    assert tomllib.loads(path.read_text()) == {GRANTS_FIELD: {str(tmp_path): [PUSH]}}
    # The live cache still describes the absent file: nothing was published.
    assert layer.fingerprint == MISSING_BACKING_STORE_DATA_FINGERPRINT
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_stale_expected_revision_fails_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    stale = layer.fingerprint
    assert stale is not None
    before = live_state(orchestrator)
    external = 'theme = "light"\n'
    path.write_text(external)

    for revision in (stale, "vibe:bogus"):
        result = await orchestrator.save_dangerous_command_grant(
            project=tmp_path, operation_id=PUSH, expected_revision=revision
        )
        assert result.persistence == "not_saved"
        assert result.application == "unchanged"
        assert result.error == "conflict"
        assert result.revision is None

    assert path.read_text() == external
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_intervening_writer_conflicts_at_write_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)
    external = 'theme = "light"\n'
    original = UserConfigLayer.save_checked

    async def intervening(
        self: UserConfigLayer, next_config: object, *, expected_revision: str
    ) -> str:
        # A second writer replaces the file after the optimistic pre-check.
        path.write_text(external)
        return await original(self, next_config, expected_revision=expected_revision)  # type: ignore[arg-type]

    monkeypatch.setattr(UserConfigLayer, "save_checked", intervening)
    result = await orchestrator.save_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "conflict"
    assert result.revision is None
    assert path.read_text() == external
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_intervening_write_between_revision_read_and_replace_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)
    external = 'theme = "light"\n# written between the revision reads\n'
    monkeypatch.setattr(
        _base,
        "create_file_fingerprint",
        _external_write_between_revision_reads(path, external),
    )

    result = await orchestrator.save_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "conflict"
    assert result.revision is None
    # The intervening writer's content survives byte-for-byte.
    assert path.read_text() == external
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_remove_intervening_write_between_revision_read_and_replace_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'[{GRANTS_FIELD}]\n"{tmp_path}" = ["{PUSH}"]\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)
    external = 'theme = "light"\n# written between the revision reads\n'
    monkeypatch.setattr(
        _base,
        "create_file_fingerprint",
        _external_write_between_revision_reads(path, external),
    )

    result = await orchestrator.remove_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "conflict"
    assert result.revision is None
    # The intervening writer's content survives byte-for-byte.
    assert path.read_text() == external
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disk_content",
    [
        f'{GRANTS_FIELD} = "nope"\n',
        f'[{GRANTS_FIELD}]\n"relative-project" = ["{PUSH}"]\n',
        f'[{GRANTS_FIELD}]\n"/project" = ["git.push.force"]\n',
        "this is not valid = = = toml [[[\n",
    ],
)
async def test_malformed_disk_source_is_typed_error(
    tmp_path: Path, disk_content: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)
    path.write_text(disk_content)
    with path.open("rb") as file:
        fresh = create_file_fingerprint(file)

    result = await orchestrator.save_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=fresh
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert result.revision is None
    assert path.read_text() == disk_content
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation_id", ["git.push.force", "git.push.delete", "git.checkout", "nope"]
)
async def test_non_persistable_operation_argument_is_typed_error(
    tmp_path: Path, operation_id: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)

    result = await orchestrator.save_dangerous_command_grant(
        project=tmp_path, operation_id=operation_id, expected_revision=expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert result.revision is None
    assert path.read_text() == 'theme = "dark"\n'
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_relative_project_argument_is_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)

    result = await orchestrator.save_dangerous_command_grant(
        project=Path("relative-project"), operation_id=PUSH, expected_revision=expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert result.revision is None
    assert path.read_text() == 'theme = "dark"\n'
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_project_layer_is_not_a_user_target(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    config_file = project_dir / ".chartreux" / "config.toml"
    config_file.parent.mkdir(parents=True)
    trusted_folders_manager.trust_for_session(config_file.parent)
    project = ProjectConfigLayer(path=project_dir)
    session = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), project, session],
        default_layer_resolver=lambda: session,
    )

    result = await orchestrator.save_dangerous_command_grant(
        project=project_dir, operation_id=PUSH, expected_revision="any"
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert result.revision is None
    assert not config_file.exists()


@pytest.mark.asyncio
async def test_subclass_cannot_persist_grants(tmp_path: Path) -> None:
    class FakeUserLayer(UserConfigLayer):
        pass

    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    fake = FakeUserLayer(path=path)
    session = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), fake, session],
        default_layer_resolver=lambda: session,
    )
    expected = fake.fingerprint
    assert expected is not None

    result = await orchestrator.save_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert path.read_text() == 'theme = "dark"\n'


@pytest.mark.asyncio
async def test_child_orchestrator_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before_bytes = path.read_bytes()
    children = [orchestrator._copy_for_child(), orchestrator._copy_for_child().copy()]

    for child in children:
        result = await child.save_dangerous_command_grant(
            project=tmp_path, operation_id=PUSH, expected_revision="any"
        )
        assert result.persistence == "not_saved"
        assert result.application == "unchanged"
        assert result.error == "validation"
        assert result.revision is None
        removed = await child.remove_dangerous_command_grant(
            project=tmp_path, operation_id=PUSH, expected_revision="any"
        )
        assert removed.persistence == "not_saved"
        assert removed.application == "unchanged"
        assert removed.error == "validation"
        assert removed.revision is None

    assert path.read_bytes() == before_bytes
    # The root orchestrator is unaffected and its mutation lock is free.
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    result = await orchestrator.save_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )
    assert result.persistence == "saved"
    assert not orchestrator._mutation_lock.locked()


@pytest.mark.asyncio
async def test_concurrent_saves_serialize_under_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)

    first = orchestrator.save_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )
    second = orchestrator.save_dangerous_command_grant(
        project=tmp_path / "other", operation_id=PUSH, expected_revision=expected
    )
    results = await asyncio.gather(first, second)

    # One save wins the fresh disk revision; the other conflicts cleanly
    # instead of interleaving a second write or a partial merge.
    assert sorted((r.persistence, r.error or "") for r in results) == [
        ("not_saved", "conflict"),
        ("saved", ""),
    ]
    data = tomllib.loads(path.read_text())
    assert len(data[GRANTS_FIELD]) == 1
    assert data["theme"] == "dark"
    assert not orchestrator._mutation_lock.locked()
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_save_and_remove_serialize_under_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'[{GRANTS_FIELD}]\n"{tmp_path}" = ["{PUSH}"]\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None

    saved = orchestrator.save_dangerous_command_grant(
        project=tmp_path / "other", operation_id=PUSH, expected_revision=expected
    )
    removed = orchestrator.remove_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )
    saved_result, removed_result = await asyncio.gather(saved, removed)

    # Exactly one edit wins the fresh disk revision; the other conflicts
    # cleanly instead of interleaving a second write or a partial merge.
    assert sorted(
        (r.persistence, r.error or "") for r in (saved_result, removed_result)
    ) == [("not_saved", "conflict"), ("saved", "")]
    data = tomllib.loads(path.read_text())
    if saved_result.persistence == "saved":
        assert removed_result.error == "conflict"
        assert data[GRANTS_FIELD] == {
            str(tmp_path): [PUSH],
            str(tmp_path / "other"): [PUSH],
        }
    else:
        assert saved_result.error == "conflict"
        assert GRANTS_FIELD not in data
    assert not orchestrator._mutation_lock.locked()


@pytest.mark.asyncio
async def test_saves_accumulate_and_deduplicate_canonical_grants(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)

    first = await orchestrator.save_dangerous_command_grant(
        project=tmp_path,
        operation_id=PUSH,
        expected_revision=MISSING_BACKING_STORE_DATA_FINGERPRINT,
    )
    assert first.persistence == "saved"
    assert first.revision is not None
    # A non-canonical spelling of the project key canonicalizes on save.
    second = await orchestrator.save_dangerous_command_grant(
        project=tmp_path / "sub" / "..",
        operation_id=PUSH,
        expected_revision=first.revision,
    )
    assert second.persistence == "saved"
    assert second.revision is not None
    # A duplicate canonical grant merges as a no-op.
    third = await orchestrator.save_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=second.revision
    )
    assert third.persistence == "saved"

    assert tomllib.loads(path.read_text())[GRANTS_FIELD] == {str(tmp_path): [PUSH]}


@pytest.mark.asyncio
async def test_saved_grants_reload_through_user_layer(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)
    saved = await orchestrator.save_dangerous_command_grant(
        project=tmp_path,
        operation_id=PUSH,
        expected_revision=MISSING_BACKING_STORE_DATA_FINGERPRINT,
    )
    assert saved.persistence == "saved"

    reloaded = await make_orchestrator(path)
    assert reloaded.config.dangerous_command_grants_by_project == {
        str(tmp_path): [PUSH]
    }
    authority = next(r for r in reloaded.restrictions if r.layer_name == "user-toml")
    assert authority.dangerous_command_grants[0].project == tmp_path
    assert authority.dangerous_command_grants[0].operations == (PUSH,)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_write", [False, True])
async def test_cancelled_save_reports_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_write: bool
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = _base._write_toml_snapshot

    def blocked(
        path: Path, data: object, *, expected_revision: str | None = None
    ) -> str:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        if fail_write:
            raise OSError("secret-sentinel")
        return original(path, data, expected_revision=expected_revision)  # type: ignore[arg-type]

    monkeypatch.setattr(_base, "_write_toml_snapshot", blocked)
    task = asyncio.create_task(
        orchestrator.save_dangerous_command_grant(
            project=tmp_path, operation_id=PUSH, expected_revision=expected
        )
    )
    await asyncio.wait_for(started.wait(), 5)
    try:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert orchestrator._mutation_lock.locked()
    finally:
        release.set()
    result = await asyncio.wait_for(task, 5)

    if fail_write:
        assert result.persistence == "not_saved"
        assert result.error == "write"
        assert result.revision is None
        assert path.read_text() == 'theme = "dark"\n'
    else:
        assert result.persistence == "saved"
        assert result.application == "unchanged"
        assert result.error == "cancelled"
        assert result.revision is not None
        assert tomllib.loads(path.read_text())[GRANTS_FIELD] == {str(tmp_path): [PUSH]}
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_durability_uncertainty_reported_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    original_fsync = _base.os.fsync

    def fsync(fd: int) -> None:
        if stat.S_ISDIR(_base.os.fstat(fd).st_mode):
            raise OSError("secret-sentinel")
        original_fsync(fd)

    monkeypatch.setattr(_base.os, "fsync", fsync)
    result = await orchestrator.save_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )

    assert "secret-sentinel" not in repr(result)
    assert result.persistence == "durability_uncertain"
    assert result.application == "unchanged"
    assert result.error is None
    assert result.revision is not None
    # The atomic replacement completed; only directory durability is uncertain.
    with path.open("rb") as file:
        assert result.revision == create_file_fingerprint(file)
    data = tomllib.loads(path.read_text())
    assert data["theme"] == "dark"
    assert data[GRANTS_FIELD] == {str(tmp_path): [PUSH]}
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_write_failure_reports_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None

    def deny_write(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(_base.tempfile, "NamedTemporaryFile", deny_write)
    result = await orchestrator.save_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "write"
    assert result.revision is None
    assert path.read_text() == 'theme = "dark"\n'
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_generic_grants_field_rejection_preserved(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    saved = await orchestrator.save_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )
    assert saved.persistence == "saved"
    assert saved.revision is not None
    before_bytes = path.read_bytes()
    injected = {str(tmp_path): [PUSH], str(tmp_path / "other"): [PUSH]}

    with pytest.raises(ConfigPatchValidationError, match=GRANTS_FIELD):
        await orchestrator.apply_patch(
            [AddOperationPatch(path=f"/{GRANTS_FIELD}", value=injected)],
            reason="generic",
        )
    result = await orchestrator.save(
        [AddOperationPatch(path=f"/{GRANTS_FIELD}", value=injected)],
        target="user",
        expected_revision=saved.revision,
        reason="generic",
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert path.read_bytes() == before_bytes
    assert tomllib.loads(path.read_text())[GRANTS_FIELD] == {str(tmp_path): [PUSH]}


@pytest.mark.asyncio
async def test_removes_grant_preserving_disk_content_without_publishing(
    tmp_path: Path,
) -> None:
    other = tmp_path / "other-project"
    path = tmp_path / "config.toml"
    path.write_text(
        'theme = "dark"\n'
        '\n[tools.bash]\ndenylist = ["user-denial"]\n'
        "\n"
        f"[{GRANTS_FIELD}]\n"
        f'"{tmp_path}" = ["{PUSH}"]\n'
        f'"{other}" = ["{PUSH}"]\n'
    )
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)
    grants_before = orchestrator.config.dangerous_command_grants_by_project

    result = await orchestrator.remove_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )

    assert result.error is None
    assert result.persistence == "saved"
    assert result.application == "unchanged"
    with path.open("rb") as file:
        assert result.revision == create_file_fingerprint(file)
    data = tomllib.loads(path.read_text())
    assert data["theme"] == "dark"
    assert data["tools"] == {"bash": {"denylist": ["user-denial"]}}
    # The other project's entry is preserved; the emptied entry is dropped.
    assert data[GRANTS_FIELD] == {str(other): [PUSH]}
    # The removal outcome is its own: nothing was published to live authority.
    assert live_state(orchestrator) == before
    assert orchestrator.config.dangerous_command_grants_by_project == grants_before


@pytest.mark.asyncio
async def test_removes_last_grant_drops_the_field(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'theme = "dark"\n\n[{GRANTS_FIELD}]\n"{tmp_path}" = ["{PUSH}"]\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)

    result = await orchestrator.remove_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )

    assert result.persistence == "saved"
    assert result.application == "unchanged"
    assert result.error is None
    data = tomllib.loads(path.read_text())
    assert data == {"theme": "dark"}
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_remove_absent_grant_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)

    result = await orchestrator.remove_dangerous_command_grant(
        project=tmp_path,
        operation_id=PUSH,
        expected_revision=MISSING_BACKING_STORE_DATA_FINGERPRINT,
    )

    assert result.error is None
    assert result.persistence == "saved"
    assert result.application == "unchanged"
    assert result.revision is not None
    # The desired end state (no saved grant) holds on disk.
    read = await read_saved_dangerous_command_grants(orchestrator, project=tmp_path)
    assert read == SavedGrantsRead(operations=(), user_revision=result.revision)
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_remove_stale_revision_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'[{GRANTS_FIELD}]\n"{tmp_path}" = ["{PUSH}"]\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    stale = layer.fingerprint
    assert stale is not None
    before = live_state(orchestrator)
    path.write_text('theme = "light"\n')

    result = await orchestrator.remove_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=stale
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "conflict"
    assert result.revision is None
    assert path.read_text() == 'theme = "light"\n'
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disk_content",
    [
        f'[{GRANTS_FIELD}]\n"/project" = ["git.push.force"]\n',
        f'[{GRANTS_FIELD}]\n"relative-project" = ["{PUSH}"]\n',
        "this is not valid = = = toml [[[\n",
    ],
)
async def test_remove_malformed_disk_source_is_typed_error(
    tmp_path: Path, disk_content: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)
    path.write_text(disk_content)
    with path.open("rb") as file:
        fresh = create_file_fingerprint(file)

    result = await orchestrator.remove_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=fresh
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert result.revision is None
    assert path.read_text() == disk_content
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_remove_non_persistable_operation_is_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'[{GRANTS_FIELD}]\n"{tmp_path}" = ["{PUSH}"]\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)

    result = await orchestrator.remove_dangerous_command_grant(
        project=tmp_path, operation_id="git.push.force", expected_revision=expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert result.revision is None
    assert tomllib.loads(path.read_text())[GRANTS_FIELD] == {str(tmp_path): [PUSH]}
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_remove_without_actual_user_layer_fails_honestly(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    config_file = project_dir / ".chartreux" / "config.toml"
    config_file.parent.mkdir(parents=True)
    trusted_folders_manager.trust_for_session(config_file.parent)
    project = ProjectConfigLayer(path=project_dir)
    session = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), project, session],
        default_layer_resolver=lambda: session,
    )

    result = await orchestrator.remove_dangerous_command_grant(
        project=project_dir, operation_id=PUSH, expected_revision="any"
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert result.revision is None
    assert not config_file.exists()


@pytest.mark.asyncio
async def test_remove_write_failure_reports_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'[{GRANTS_FIELD}]\n"{tmp_path}" = ["{PUSH}"]\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    disk_before = path.read_bytes()

    def deny_write(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(_base.tempfile, "NamedTemporaryFile", deny_write)
    result = await orchestrator.remove_dangerous_command_grant(
        project=tmp_path, operation_id=PUSH, expected_revision=expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "write"
    assert result.revision is None
    assert path.read_bytes() == disk_before
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_read_reflects_a_freshly_saved_grant(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)
    saved = await orchestrator.save_dangerous_command_grant(
        project=tmp_path,
        operation_id=PUSH,
        expected_revision=MISSING_BACKING_STORE_DATA_FINGERPRINT,
    )
    assert saved.persistence == "saved"
    layer = orchestrator.get_layer("user-toml")
    before = live_state(orchestrator)

    read = await read_saved_dangerous_command_grants(orchestrator, project=tmp_path)

    assert read == SavedGrantsRead(operations=(PUSH,), user_revision=saved.revision)
    assert read.unavailable is None
    with path.open("rb") as file:
        assert read.user_revision == create_file_fingerprint(file)
    # The read published nothing and did not adopt the saved file into the
    # live cache: the layer still describes the absent file.
    assert live_state(orchestrator) == before
    assert layer.fingerprint == MISSING_BACKING_STORE_DATA_FINGERPRINT


@pytest.mark.asyncio
async def test_read_reflects_disk_after_an_external_write(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'[{GRANTS_FIELD}]\n"{tmp_path}" = ["{PUSH}"]\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    cached_revision = layer.fingerprint
    assert cached_revision is not None
    before = live_state(orchestrator)
    path.write_text(
        f"[{GRANTS_FIELD}]\n"
        f'"{tmp_path}" = ["{PUSH}"]\n'
        f'"{tmp_path / "other"}" = ["{PUSH}"]\n'
    )

    read = await read_saved_dangerous_command_grants(orchestrator, project=tmp_path)

    assert read.unavailable is None
    assert read.operations == (PUSH,)
    with path.open("rb") as file:
        assert read.user_revision == create_file_fingerprint(file)
    assert read.user_revision != cached_revision
    # The live cache is stale and untouched: the read saw disk, not memory.
    assert layer.fingerprint == cached_revision
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_read_missing_file_is_empty_not_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)

    read = await read_saved_dangerous_command_grants(orchestrator, project=tmp_path)

    assert read == SavedGrantsRead(
        operations=(), user_revision=MISSING_BACKING_STORE_DATA_FINGERPRINT
    )
    assert read.unavailable is None
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disk_content",
    [
        "this is not valid = = = toml [[[\n",
        f'[{GRANTS_FIELD}]\n"relative-project" = ["{PUSH}"]\n',
        f'[{GRANTS_FIELD}]\n"/project" = ["git.push.force"]\n',
    ],
)
async def test_read_unreadable_source_is_unavailable_not_empty(
    tmp_path: Path, disk_content: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)
    path.write_text(disk_content)

    read = await read_saved_dangerous_command_grants(orchestrator, project=tmp_path)

    assert read == SavedGrantsRead(unavailable="read")
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_read_does_not_leak_other_projects(tmp_path: Path) -> None:
    other = tmp_path / "other-project"
    path = tmp_path / "config.toml"
    path.write_text(
        f'[{GRANTS_FIELD}]\n"{tmp_path}" = ["{PUSH}"]\n"{other}" = ["{PUSH}"]\n'
    )
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")

    mine = await read_saved_dangerous_command_grants(orchestrator, project=tmp_path)
    theirs = await read_saved_dangerous_command_grants(orchestrator, project=other)
    absent = await read_saved_dangerous_command_grants(
        orchestrator, project=tmp_path / "absent"
    )

    assert mine.unavailable is None
    assert mine.operations == (PUSH,)
    assert theirs.unavailable is None
    assert theirs.operations == (PUSH,)
    assert absent.unavailable is None
    assert absent.operations == ()
    # One force-loaded source: the same user-layer revision for every project.
    assert mine.user_revision == theirs.user_revision == absent.user_revision
    assert mine.user_revision == layer.fingerprint


@pytest.mark.asyncio
async def test_read_without_actual_user_layer_is_unavailable(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    config_file = project_dir / ".chartreux" / "config.toml"
    config_file.parent.mkdir(parents=True)
    trusted_folders_manager.trust_for_session(config_file.parent)
    project = ProjectConfigLayer(path=project_dir)
    session = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), project, session],
        default_layer_resolver=lambda: session,
    )

    read = await read_saved_dangerous_command_grants(orchestrator, project=project_dir)

    assert read == SavedGrantsRead(unavailable="no_user_source")
    assert not config_file.exists()


@pytest.mark.asyncio
async def test_read_needs_no_lock_and_publishes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)

    async with orchestrator._mutation_lock:
        read = await read_saved_dangerous_command_grants(orchestrator, project=tmp_path)

    assert read.unavailable is None
    assert read.operations == ()
    assert not orchestrator._mutation_lock.locked()
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
async def test_read_relative_project_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)

    with pytest.raises(ValueError, match=GRANTS_FIELD):
        await read_saved_dangerous_command_grants(
            orchestrator, project=Path("relative-project")
        )
