from __future__ import annotations

import os
from pathlib import Path
import tomllib
from uuid import uuid4

import pytest

from chartreux.core.config.fingerprint import create_file_fingerprint
from chartreux.core.config.layer import (
    ConfigStorageError,
    LayerImplementationError,
    LayerNotLoadedError,
)
from chartreux.core.config.layers import _base
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.patch import (
    AddOperationPatch,
    ConfigPatch,
    RemoveOperationPatch,
    ReplaceOperationPatch,
)
from chartreux.core.config.types import (
    MISSING_BACKING_STORE_DATA_FINGERPRINT,
    ConcurrencyConflictError,
)


def random_config_file_name() -> str:
    return f"config-{uuid4().hex}.toml"


@pytest.mark.asyncio
async def test_reads_toml_file(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text('active_model = "mistral-large"\ncount = 42\n')

    layer = UserConfigLayer(path=path)
    data = await layer.load()
    assert data.model_extra == {"active_model": "mistral-large", "count": 42}
    fingerprint = layer.fingerprint
    assert isinstance(fingerprint, str)
    assert fingerprint


@pytest.mark.asyncio
async def test_always_trusted(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text('key = "value"\n')

    layer = UserConfigLayer(path=path)
    assert layer.is_trusted is None
    data = await layer.load()
    assert layer.is_trusted is True
    assert data.model_extra == {"key": "value"}


@pytest.mark.asyncio
async def test_missing_file_returns_empty(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    layer = UserConfigLayer(path=path)
    data = await layer.load()
    assert data.model_extra == {}
    assert layer.fingerprint == MISSING_BACKING_STORE_DATA_FINGERPRINT


@pytest.mark.asyncio
async def test_apply_creates_file_when_it_does_not_exist(
    tmp_working_directory: Path,
) -> None:
    path = tmp_working_directory / random_config_file_name()
    layer = UserConfigLayer(path=path)

    await layer.load()
    assert layer.fingerprint == MISSING_BACKING_STORE_DATA_FINGERPRINT

    await layer.apply(
        ConfigPatch(
            AddOperationPatch(path="/active_model", value="mistral-large"),
            fingerprint=MISSING_BACKING_STORE_DATA_FINGERPRINT,
        )
    )

    with path.open("rb") as file:
        assert tomllib.load(file) == {"active_model": "mistral-large"}
        assert layer.fingerprint == create_file_fingerprint(file)


@pytest.mark.asyncio
async def test_apply_sets_field_and_refreshes_cache(
    tmp_working_directory: Path,
) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text("""\
active_model = "old"

[tools]
disabled_tools = ["bash", "python"]
deprecated_setting = true
""")
    layer = UserConfigLayer(path=path)

    await layer.load()
    fingerprint = layer.fingerprint
    assert isinstance(fingerprint, str)

    await layer.apply(
        ConfigPatch(
            ReplaceOperationPatch(path="/active_model", value="new"),
            AddOperationPatch(path="/tools/enabled_tools", value=["read"]),
            AddOperationPatch(path="/tools/disabled_tools/-", value="node"),
            RemoveOperationPatch(path="/tools/disabled_tools/0"),
            RemoveOperationPatch(path="/tools/deprecated_setting"),
            fingerprint=fingerprint,
        )
    )

    expected_data = {
        "active_model": "new",
        "tools": {"disabled_tools": ["python", "node"], "enabled_tools": ["read"]},
    }
    with path.open("rb") as file:
        assert tomllib.load(file) == expected_data

    cached_data = layer._state.data
    assert cached_data is not None
    assert cached_data.model_extra == expected_data
    assert layer.fingerprint != fingerprint


@pytest.mark.asyncio
async def test_apply_raises_config_storage_error_when_write_fails(
    tmp_working_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A read-only config.toml (e.g. symlinked read-only from the Nix store)
    # makes the atomic write raise OSError. It must surface as a typed
    # ConfigStorageError, not an uncaught traceback. chmod is unreliable under
    # root, so raise the OSError from the write path directly.
    path = tmp_working_directory / random_config_file_name()
    path.write_text('active_model = "old"\n')
    layer = UserConfigLayer(path=path)
    await layer.load()
    fingerprint = layer.fingerprint
    assert isinstance(fingerprint, str)

    def deny_write(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(_base.tempfile, "NamedTemporaryFile", deny_write)

    with pytest.raises(ConfigStorageError) as excinfo:
        await layer.apply(
            ConfigPatch(
                ReplaceOperationPatch(path="/active_model", value="new"),
                fingerprint=fingerprint,
            )
        )

    assert excinfo.value.path == path
    assert excinfo.value.operation == "write"
    assert isinstance(excinfo.value.__cause__, PermissionError)


@pytest.mark.asyncio
async def test_load_raises_config_storage_error_when_read_fails(
    tmp_working_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text('active_model = "old"\n')
    layer = UserConfigLayer(path=path)

    original_open = Path.open

    def deny_open(self: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if self == path:
            raise PermissionError(13, "Permission denied", str(path))
        return original_open(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", deny_open)

    with pytest.raises(ConfigStorageError) as excinfo:
        await layer.load()

    assert excinfo.value.path == path
    assert excinfo.value.operation == "read"
    assert isinstance(excinfo.value.__cause__, PermissionError)


@pytest.mark.asyncio
async def test_apply_cache_fingerprint_matches_written_file(
    tmp_working_directory: Path,
) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text("")
    layer = UserConfigLayer(path=path)

    await layer.load()
    fingerprint = layer.fingerprint
    assert isinstance(fingerprint, str)

    await layer.apply(
        ConfigPatch(
            AddOperationPatch(path="/active_model", value="mistral-large"),
            fingerprint=fingerprint,
        )
    )

    with path.open("rb") as file:
        assert layer.fingerprint == create_file_fingerprint(file)


@pytest.mark.asyncio
async def test_apply_uses_unique_temp_file(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    fixed_tmp_path = tmp_working_directory / f".{path.name}.tmp"
    path.write_text("")
    fixed_tmp_path.write_text("stale")
    layer = UserConfigLayer(path=path)

    await layer.load()
    fingerprint = layer.fingerprint
    assert isinstance(fingerprint, str)

    await layer.apply(
        ConfigPatch(
            AddOperationPatch(path="/active_model", value="mistral-large"),
            fingerprint=fingerprint,
        )
    )

    assert fixed_tmp_path.read_text() == "stale"
    assert list(tmp_working_directory.glob(f".{path.name}.*.tmp")) == []
    with path.open("rb") as file:
        assert tomllib.load(file) == {"active_model": "mistral-large"}


@pytest.mark.asyncio
async def test_checked_write_rejects_replaced_snapshot_revision(
    tmp_working_directory: Path,
) -> None:
    path = tmp_working_directory / random_config_file_name()
    replacement = tmp_working_directory / f"replacement-{path.name}"
    path.write_text('active_model = "opened"\n')
    replacement.write_text('active_model = "replacement"\n')
    layer = UserConfigLayer(path=path)

    snapshot = await layer.load()
    revision = layer.fingerprint
    assert revision is not None
    os.replace(replacement, path)

    with pytest.raises(ConcurrencyConflictError) as exc_info:
        await layer.save_checked(snapshot, expected_revision=revision)

    assert exc_info.value.expected_fp == revision
    with path.open("rb") as file:
        assert exc_info.value.actual_fp == create_file_fingerprint(file)
    assert tomllib.loads(path.read_text()) == {"active_model": "replacement"}


def test_atomic_replace_preserves_replacement_fingerprint(
    tmp_working_directory: Path,
) -> None:
    path = tmp_working_directory / random_config_file_name()
    replacement = tmp_working_directory / f".{path.name}.tmp"
    path.write_text("key = 1")
    replacement.write_text("key = 2")

    with replacement.open("rb") as file:
        replacement_fingerprint = create_file_fingerprint(file)

    os.replace(replacement, path)

    with path.open("rb") as file:
        assert create_file_fingerprint(file) == replacement_fingerprint


@pytest.mark.asyncio
async def test_apply_raises_when_layer_is_not_loaded(
    tmp_working_directory: Path,
) -> None:
    path = tmp_working_directory / random_config_file_name()
    layer = UserConfigLayer(path=path)

    with pytest.raises(LayerNotLoadedError, match="loaded before applying patches"):
        await layer.apply(
            ConfigPatch(
                AddOperationPatch(path="/active_model", value="mistral-large"),
                fingerprint=MISSING_BACKING_STORE_DATA_FINGERPRINT,
            )
        )


@pytest.mark.asyncio
async def test_apply_raises_when_cache_is_invalidated(
    tmp_working_directory: Path,
) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text('active_model = "old"\n')
    layer = UserConfigLayer(path=path)

    await layer.load()
    fingerprint = layer.fingerprint
    assert isinstance(fingerprint, str)
    await layer.invalidate_cache()

    with pytest.raises(LayerNotLoadedError, match="loaded before applying patches"):
        await layer.apply(
            ConfigPatch(
                ReplaceOperationPatch(path="/active_model", value="new"),
                fingerprint=fingerprint,
            )
        )


@pytest.mark.asyncio
async def test_apply_creates_parent_directory_when_it_does_not_exist(
    tmp_working_directory: Path,
) -> None:
    path = tmp_working_directory / "nested" / random_config_file_name()
    layer = UserConfigLayer(path=path)

    await layer.load()

    await layer.apply(
        ConfigPatch(
            AddOperationPatch(path="/active_model", value="mistral-large"),
            fingerprint=MISSING_BACKING_STORE_DATA_FINGERPRINT,
        )
    )

    with path.open("rb") as file:
        assert tomllib.load(file) == {"active_model": "mistral-large"}


@pytest.mark.asyncio
async def test_commit_sets_missing_nested_field(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text("[models]\n")
    layer = UserConfigLayer(path=path)

    await layer.load()
    fingerprint = layer.fingerprint
    assert isinstance(fingerprint, str)

    await layer.apply(
        ConfigPatch(
            AddOperationPatch(path="/models/active_model", value="mistral-large"),
            fingerprint=fingerprint,
        )
    )

    with path.open("rb") as file:
        assert tomllib.load(file) == {"models": {"active_model": "mistral-large"}}


@pytest.mark.asyncio
async def test_apply_overwrites_external_file_changes(
    tmp_working_directory: Path,
) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text('active_model = "old"\n')
    layer = UserConfigLayer(path=path)

    await layer.load()
    fingerprint = layer.fingerprint
    assert isinstance(fingerprint, str)
    path.write_text('active_model = "external"\n')

    await layer.apply(
        ConfigPatch(
            ReplaceOperationPatch(path="/active_model", value="new"),
            fingerprint=fingerprint,
        )
    )

    with path.open("rb") as file:
        assert tomllib.load(file) == {"active_model": "new"}
    data = await layer.load()
    assert data.model_extra == {"active_model": "new"}


@pytest.mark.asyncio
async def test_nested_toml_structure(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text("""\
[models]
active_model = "test"

[[models.items]]
alias = "a"
provider = "p"
""")
    layer = UserConfigLayer(path=path)
    data = await layer.load()
    assert data.model_extra == {
        "models": {"active_model": "test", "items": [{"alias": "a", "provider": "p"}]}
    }


@pytest.mark.asyncio
async def test_invalid_toml_raises(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text("this is not valid = = = toml [[[")
    layer = UserConfigLayer(path=path)
    with pytest.raises(LayerImplementationError, match="_build_config_snapshot"):
        await layer.load()


@pytest.mark.asyncio
async def test_force_reload_reads_fresh_data(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text('value = "first"\n')
    layer = UserConfigLayer(path=path)

    data1 = await layer.load()
    fp1 = layer.fingerprint
    assert data1.model_extra == {"value": "first"}
    assert isinstance(fp1, str)
    assert fp1

    path.write_text('value = "second"\n')
    data2 = await layer.load(force=True)
    fp2 = layer.fingerprint
    assert data2.model_extra == {"value": "second"}
    assert isinstance(fp2, str)
    assert fp2
    assert fp1 != fp2

    path.unlink()
    data3 = await layer.load(force=True)
    assert data3.model_extra == {}
    assert layer.fingerprint == MISSING_BACKING_STORE_DATA_FINGERPRINT


@pytest.mark.asyncio
async def test_empty_toml_file(tmp_working_directory: Path) -> None:
    path = tmp_working_directory / random_config_file_name()
    path.write_text("")
    layer = UserConfigLayer(path=path)
    data = await layer.load()
    assert data.model_extra == {}
