from __future__ import annotations

import asyncio
import fcntl
from pathlib import Path
import tomllib
from typing import IO

import pytest

from chartreux.core.config.fingerprint import create_file_fingerprint
from chartreux.core.config.layer import RawConfig
from chartreux.core.config.layers import _base
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.patch import AddOperationPatch, ConfigPatch
from chartreux.core.config.types import ConcurrencyConflictError

EXTERNAL = 'theme = "light"\n# written by a competing writer\n'


@pytest.mark.asyncio
async def test_checked_save_succeeds_without_a_competitor(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    layer = UserConfigLayer(path=path)
    await layer.load()
    revision = layer.fingerprint
    assert revision is not None

    next_revision = await layer.save_checked(
        RawConfig.model_validate({"theme": "light"}), expected_revision=revision
    )

    with path.open("rb") as file:
        assert next_revision == create_file_fingerprint(file)
    assert tomllib.loads(path.read_text()) == {"theme": "light"}
    # The cross-process lock is a persistent dot-prefixed sibling, and the
    # temp replacement leaves no artifacts behind.
    assert (tmp_path / f".{path.name}.lock").is_file()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.asyncio
async def test_intervening_write_between_revision_read_and_replace_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    layer = UserConfigLayer(path=path)
    snapshot = await layer.load()
    revision = layer.fingerprint
    assert revision is not None
    original = _base.create_file_fingerprint
    target_reads = 0

    def intervening(file: IO[bytes]) -> str:
        nonlocal target_reads
        if Path(file.name) == path:
            target_reads += 1
            if target_reads == 2:
                # A non-cooperating writer replaces the store after the first
                # under-lock revision read and before the replacement.
                path.write_text(EXTERNAL)
        return original(file)

    monkeypatch.setattr(_base, "create_file_fingerprint", intervening)

    with pytest.raises(ConcurrencyConflictError) as exc_info:
        await layer.save_checked(snapshot, expected_revision=revision)

    assert exc_info.value.expected_fp == revision
    with path.open("rb") as file:
        assert exc_info.value.actual_fp == create_file_fingerprint(file)
    # The intervening writer's content survives byte-for-byte.
    assert path.read_text() == EXTERNAL
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.asyncio
async def test_checked_save_waits_for_the_lock_holder_and_then_conflicts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    layer = UserConfigLayer(path=path)
    snapshot = await layer.load()
    revision = layer.fingerprint
    assert revision is not None

    with _base._write_lock_path(path).open("a") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        task = asyncio.create_task(
            layer.save_checked(snapshot, expected_revision=revision)
        )
        await asyncio.sleep(0.1)
        # The save waits for the cooperating holder instead of interleaving.
        assert not task.done()
        path.write_text(EXTERNAL)
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)

    with pytest.raises(ConcurrencyConflictError):
        await task
    assert path.read_text() == EXTERNAL
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.asyncio
async def test_held_lock_times_out_instead_of_blocking_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    layer = UserConfigLayer(path=path)
    snapshot = await layer.load()
    revision = layer.fingerprint
    assert revision is not None
    monkeypatch.setattr(_base, "_WRITE_LOCK_ACQUIRE_TIMEOUT", 0.05)

    with _base._write_lock_path(path).open("a") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        with pytest.raises(TimeoutError):
            await layer.save_checked(snapshot, expected_revision=revision)

    assert path.read_text() == 'theme = "dark"\n'
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
    # The bounded wait released the attempt; a later save still succeeds.
    next_revision = await layer.save_checked(
        RawConfig.model_validate({"theme": "light"}), expected_revision=revision
    )
    with path.open("rb") as file:
        assert next_revision == create_file_fingerprint(file)
    assert tomllib.loads(path.read_text()) == {"theme": "light"}


@pytest.mark.asyncio
async def test_read_only_lock_file_still_saves_and_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    layer = UserConfigLayer(path=path)
    snapshot = await layer.load()
    revision = layer.fingerprint
    assert revision is not None
    lock_path = _base._write_lock_path(path)
    lock_path.touch()
    lock_path.chmod(0o444)

    # An unwritable existing lock file must not block the save: the lock
    # falls back to a read-only descriptor.
    next_revision = await layer.save_checked(snapshot, expected_revision=revision)
    with path.open("rb") as file:
        assert next_revision == create_file_fingerprint(file)

    # The read-only descriptor still locks: a held lock stalls the next save.
    monkeypatch.setattr(_base, "_WRITE_LOCK_ACQUIRE_TIMEOUT", 0.05)
    with lock_path.open("rb") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        with pytest.raises(TimeoutError):
            await layer.save_checked(snapshot, expected_revision=revision)

    assert tomllib.loads(path.read_text()) == {"theme": "dark"}
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.asyncio
async def test_unchecked_patch_save_also_waits_for_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    layer = UserConfigLayer(path=path)
    await layer.load()
    fingerprint = layer.fingerprint
    assert fingerprint is not None

    with _base._write_lock_path(path).open("a") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        task = asyncio.create_task(
            layer.apply(
                ConfigPatch(
                    AddOperationPatch(path="/theme", value="light"),
                    fingerprint=fingerprint,
                )
            )
        )
        await asyncio.sleep(0.1)
        # The unchecked patch path takes the same exclusive write lock.
        assert not task.done()
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)

    await task
    assert tomllib.loads(path.read_text()) == {"theme": "light"}
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
