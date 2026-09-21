from __future__ import annotations

import asyncio
from pathlib import Path
import stat
import threading
import tomllib

import pytest

from chartreux.core.config._restrictions import ConfigCandidate
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layer import RawConfig
from chartreux.core.config.layers import _base
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.patch import AddOperationPatch
from chartreux.core.config.types import (
    MISSING_BACKING_STORE_DATA_FINGERPRINT,
    ConfigChangeEvent,
)
from chartreux.core.trusted_folders import trusted_folders_manager


async def make_config(path: Path) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    user = UserConfigLayer(path=path)
    session = OverridesLayer(data={"auto_compact_threshold": 70})
    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[user, session],
        default_layer_resolver=lambda: session,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "exception", "cancellation"])
async def test_session_edit_notifies_original_bus_after_acceptance(
    tmp_path: Path, failure: str | None
) -> None:
    path = tmp_path / "config.toml"
    config = await make_config(path)
    seen: list[ConfigChangeEvent] = []
    applied: list[int] = []

    def observe(event: ConfigChangeEvent) -> None:
        assert config.config.auto_compact_threshold == 50
        assert applied == [50]
        seen.append(event)
        if failure == "exception":
            raise RuntimeError("subscriber-sentinel")
        if failure == "cancellation":
            raise asyncio.CancelledError

    config.subscribe(observe)

    async def preflight(candidate: ChartreuxConfigSchema) -> None:
        assert candidate.auto_compact_threshold == 50
        assert config.config.auto_compact_threshold == 70
        assert not seen

    assert not await config.apply_session_patch(
        [AddOperationPatch(path="/auto_compact_threshold", value=50)],
        reason="session notification",
        preflight=preflight,
        apply=lambda candidate: applied.append(candidate.auto_compact_threshold),
    )
    assert len(seen) == 1
    assert seen[0].before["auto_compact_threshold"] == 70
    assert seen[0].after["auto_compact_threshold"] == 50
    assert seen[0].reason == "session notification"
    assert "auto_compact_threshold" in seen[0].changed_keys
    assert all(key.endswith("auto_compact_threshold") for key in seen[0].changed_keys)
    assert config.config.auto_compact_threshold == 50
    assert not path.exists()


@pytest.mark.asyncio
async def test_session_preflight_cancellation_does_not_publish(tmp_path: Path) -> None:
    config = await make_config(tmp_path / "config.toml")
    before = config.config
    token = config.accepted_token
    seen: list[ConfigChangeEvent] = []
    applied: list[ChartreuxConfigSchema] = []
    config.subscribe(seen.append)

    async def preflight(_candidate: ChartreuxConfigSchema) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await config.apply_session_patch(
            [AddOperationPatch(path="/auto_compact_threshold", value=50)],
            reason="cancelled session edit",
            preflight=preflight,
            apply=applied.append,
        )
    assert config.config is before
    assert config.accepted_token is token
    assert not seen
    assert not applied


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "preflight", "apply"])
async def test_session_preparation_acceptance_and_reload(
    tmp_path: Path, failure: str | None
) -> None:
    path = tmp_path / "config.toml"
    config = await make_config(path)
    before = config.config
    restrictions = config.restrictions
    token = config.accepted_token

    async def preflight(candidate: ChartreuxConfigSchema) -> None:
        assert candidate.auto_compact_threshold == 50
        assert config.config is before
        if failure == "preflight":
            raise ValueError("synthetic preparation failure")

    def apply(candidate: ChartreuxConfigSchema) -> None:
        assert candidate.auto_compact_threshold == 50
        assert config.config is before
        if failure == "apply":
            raise ValueError("synthetic application failure")

    async def update() -> None:
        assert not await config.apply_session_patch(
            [AddOperationPatch(path="/auto_compact_threshold", value=50)],
            reason="test session acceptance",
            preflight=preflight,
            apply=apply,
        )

    if failure is not None:
        with pytest.raises(ValueError, match="synthetic"):
            await update()
        assert config.config is before
        assert config.restrictions is restrictions
        assert config.accepted_token is token
    else:
        await update()
        assert config.config.auto_compact_threshold == 50
    assert not path.exists()
    await config.reload()
    assert config.config.auto_compact_threshold == (70 if failure else 50)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["none", "validation", "conflict", "race", "write", "durability", "application"],
)
async def test_single_target_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "config.toml"
    config = await make_config(path)
    before = config.config
    restrictions = config.restrictions
    token = config.accepted_token
    operation = AddOperationPatch(path="/auto_compact_threshold", value=50)
    if failure == "validation":
        operation = AddOperationPatch(path="/unknown_setting", value="secret-sentinel")
    if failure == "conflict":
        path.write_text("auto_compact_threshold = 60\n")

    async def preflight(_candidate: ConfigCandidate[ChartreuxConfigSchema]) -> None:
        if failure == "race":
            path.write_text("auto_compact_threshold = 60\n")

    def application(_candidate: ConfigCandidate[ChartreuxConfigSchema]) -> None:
        if failure == "application":
            raise RuntimeError("secret-sentinel")

    original_fsync = _base.os.fsync

    def fsync(fd: int) -> None:
        directory = stat.S_ISDIR(_base.os.fstat(fd).st_mode)
        if (failure == "write" and not directory) or (
            failure == "durability" and directory
        ):
            raise OSError("secret-sentinel")
        original_fsync(fd)

    monkeypatch.setattr(_base.os, "fsync", fsync)
    result = await config.save(
        [operation],
        target="user",
        expected_revision=MISSING_BACKING_STORE_DATA_FINGERPRINT,
        reason="test",
        preflight=preflight,
        apply=application,
    )
    assert "secret-sentinel" not in repr(result)
    if failure in {"validation", "conflict", "race", "write"}:
        assert result.persistence == "not_saved"
        assert result.application == "unchanged"
        assert config.config is before
        assert config.restrictions is restrictions
        assert config.accepted_token is token
        if failure in {"conflict", "race"}:
            assert result.error == "conflict"
            assert path.read_text() == "auto_compact_threshold = 60\n"
        else:
            assert not path.exists()
    else:
        assert tomllib.loads(path.read_text())["auto_compact_threshold"] == 50
        assert result.persistence == (
            "durability_uncertain" if failure == "durability" else "saved"
        )
        if failure == "application":
            assert result.application == "failed"
            assert config.config is before
            assert config.restrictions is restrictions
            assert config.accepted_token is token
            await config.reload()
        else:
            assert result.application == "applied"
            assert config.get_layer("user-toml").fingerprint == result.revision
        # The saved user value remains visibly shadowed, never session-mirrored.
        assert config.config.auto_compact_threshold == 70
        assert config.get_layer("overrides").cached_data == RawConfig.model_validate({
            "auto_compact_threshold": 70
        })


@pytest.mark.asyncio
async def test_mixed_targets_and_child_copies_cannot_save(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    config = await make_config(path)
    operation = AddOperationPatch(path="/auto_compact_threshold", value=50)
    for candidate, operations in [
        (
            config,
            [
                operation,
                operation.model_copy(update={"target_layer_name": "overrides"}),
            ],
        ),
        (config._copy_for_child(), [operation]),
        (config._copy_for_child().copy(), [operation]),
    ]:
        result = await candidate.save(
            [*operations],
            target="user",
            expected_revision=MISSING_BACKING_STORE_DATA_FINGERPRINT,
            reason="test",
        )
        assert result.persistence == "not_saved"
        assert result.application == "unchanged"
        assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_write", [False, True])
async def test_cancel_waits_for_writer_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_write: bool
) -> None:
    path = tmp_path / "config.toml"
    config = await make_config(path)
    before = config.config
    token = config.accepted_token
    restrictions = config.restrictions
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = _base._write_toml_snapshot

    def blocked(
        path: Path, data: RawConfig, *, expected_revision: str | None = None
    ) -> str:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        if fail_write:
            raise OSError("secret-sentinel")
        return original(path, data, expected_revision=expected_revision)

    monkeypatch.setattr(_base, "_write_toml_snapshot", blocked)
    task = asyncio.create_task(
        config.save(
            [AddOperationPatch(path="/auto_compact_threshold", value=50)],
            target="user",
            expected_revision=MISSING_BACKING_STORE_DATA_FINGERPRINT,
            reason="test",
        )
    )
    await asyncio.wait_for(started.wait(), 5)
    try:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert config._mutation_lock.locked()
    finally:
        release.set()
    result = await asyncio.wait_for(task, 5)
    assert result.persistence == ("not_saved" if fail_write else "saved")
    assert result.application == "unchanged"
    assert config.config is before
    assert config.accepted_token is token
    assert config.restrictions is restrictions
    if fail_write:
        assert not path.exists()
    else:
        assert result.error == "cancelled"
        assert tomllib.loads(path.read_text())["auto_compact_threshold"] == 50


@pytest.mark.asyncio
async def test_subscriber_failure_reports_committed_state(tmp_path: Path) -> None:
    config = await make_config(tmp_path / "config.toml")

    def fail(_event: object) -> None:
        raise RuntimeError("secret-sentinel")

    config.subscribe(fail)
    result = await config.save(
        [AddOperationPatch(path="/auto_compact_threshold", value=50)],
        target="user",
        expected_revision=MISSING_BACKING_STORE_DATA_FINGERPRINT,
        reason="test",
    )
    assert result.persistence == "saved"
    assert result.application == "applied"
    assert result.error == "notification"
    assert config.get_layer("user-toml").fingerprint == result.revision
    assert "secret-sentinel" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("replace_failure", [False, True])
async def test_project_save_and_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    replace_failure: bool,
) -> None:
    path = tmp_path / ".chartreux" / "config.toml"
    path.parent.mkdir()
    trusted_folders_manager.trust_for_session(path.parent)
    before = b"# preserve until replace\nauto_compact_threshold = 60\n"
    if existing:
        path.write_bytes(before)
    project = ProjectConfigLayer(path=tmp_path)
    session = OverridesLayer(data={})
    config = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[project, session],
        default_layer_resolver=lambda: session,
    )
    snapshot = config.config
    revision = project.fingerprint
    assert revision is not None

    def fail_replace(_self: Path, _target: Path) -> Path:
        raise OSError("secret-sentinel")

    if replace_failure:
        monkeypatch.setattr(Path, "replace", fail_replace)
    result = await config.save(
        [AddOperationPatch(path="/auto_compact_threshold", value=50)],
        target="project",
        expected_revision=revision,
        reason="test",
    )
    if replace_failure:
        assert result.persistence == "not_saved"
        assert config.config is snapshot
        assert path.read_bytes() == before if existing else not path.exists()
    else:
        assert result.persistence == "saved"
        assert result.application == "applied"
        assert config.config.auto_compact_threshold == 50
        assert project.config_file_path == path
        assert project.is_file_discovered
        assert project._is_set
        assert project._config_file_path == path
        assert tomllib.loads(path.read_text())["auto_compact_threshold"] == 50
        conflict = await config.save(
            [AddOperationPatch(path="/auto_compact_threshold", value=40)],
            target="project",
            expected_revision=revision,
            reason="test",
        )
        assert conflict.error == "conflict"
        assert config.config.auto_compact_threshold == 50


@pytest.mark.asyncio
async def test_child_legacy_mixed_patch_cannot_write_any_layer(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    root = await make_config(path)
    child = root._copy_for_child().copy()
    before = child.config
    failures = await child.apply_patch(
        [
            AddOperationPatch(path="/auto_compact_threshold", value=50),
            AddOperationPatch(
                path="/auto_compact_threshold", value=40, target_layer_name="user-toml"
            ),
        ],
        reason="test",
    )
    assert failures
    assert child.config is before
    assert not path.exists()
    assert not await child.set_field("/auto_compact_threshold", 30)
    assert child.config.auto_compact_threshold == 30
    assert root.config.auto_compact_threshold == 70
