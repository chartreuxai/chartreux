from __future__ import annotations

import multiprocessing
from pathlib import Path
import tempfile
import threading
import time
import tomllib

from chartreux.core.model_catalog.loader import CatalogStore
from chartreux.ui.providers.contracts import (
    CatalogChanges,
    CatalogValidationError,
    CatalogWriteResult,
)


def _apply_changes_in_process(
    path: Path,
    provider_id: str,
    read_complete: multiprocessing.Queue[None] | None,
    release: multiprocessing.Queue[None] | None,
    result: multiprocessing.Queue[str],
) -> None:
    store = CatalogStore(path)
    if read_complete is not None:
        read_overlay = store._read_overlay

        def delayed_read() -> dict[str, object]:
            overlay = read_overlay()
            read_complete.put(None)
            assert release is not None
            release.get(timeout=5)
            return overlay

        store._read_overlay = delayed_read  # type: ignore[method-assign]
    try:
        outcome = store.apply_changes(
            CatalogChanges(
                provider_id, {"api_base": f"https://{provider_id}.example/v1"}
            )
        )
        result.put("success" if isinstance(outcome, CatalogWriteResult) else "invalid")
    except Exception as exc:
        result.put(f"error:{exc}")


def _changes(**kwargs: object) -> CatalogChanges:
    provider = kwargs.pop("provider", {"api_base": "https://test.example/v1"})
    return CatalogChanges(
        provider_id="test/default",
        provider=provider,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _written(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text())


def test_apply_changes_writes_sparse_atomic_provider_model_and_tag_batch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.toml"
    result = CatalogStore(path).apply_changes(
        _changes(
            models={
                "new-model": {
                    "aliases": ["new-alias"],
                    "deployments": [{"provider": "test/default", "name": "new-wire"}],
                }
            },
            tags={"new": ("new-model",)},
        )
    )

    assert isinstance(result, CatalogWriteResult) and result.changed
    raw = _written(path)
    assert set(raw) == {"providers", "models", "tags"}
    assert raw["providers"] == {"test/default": {"api_base": "https://test.example/v1"}}
    assert raw["models"] == {
        "new-model": {
            "aliases": ["new-alias"],
            "deployments": [{"provider": "test/default", "name": "new-wire"}],
        }
    }
    assert raw["tags"] == {"new": ["new-model"]}
    assert result.snapshot.catalog.models["new-model"].deployments[0].name == "new-wire"


def test_unchanged_existing_selection_is_a_noop(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    store = CatalogStore(path)
    result = store.apply_changes(
        _changes(
            provider={},
            models={
                "glm-5-2": {
                    "deployments": [{"provider": "mistral/default", "name": "glm-5-2"}]
                }
            },
        )
    )

    assert isinstance(result, CatalogWriteResult) and not result.changed
    assert not path.exists()


def test_append_preserves_effective_order_and_raw_user_overrides(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.toml"
    path.write_text("""
[providers."test/default"]
api_base = "https://test.example"

[models.glm-5-2]
deployments = [
  { provider = "mistral/default", name = "user-wire", supports_images = true },
]
""")

    result = CatalogStore(path).apply_changes(
        _changes(
            models={
                "glm-5-2": {
                    "deployments": [{"provider": "test/default", "name": "test-wire"}]
                }
            }
        )
    )

    assert isinstance(result, CatalogWriteResult) and result.changed
    deployments = _written(path)["models"]["glm-5-2"]["deployments"]  # type: ignore[index]
    assert deployments == [
        {"provider": "mistral/default", "name": "user-wire", "supports_images": True},
        {"provider": "test/default", "name": "test-wire"},
    ]


def test_append_uses_shipped_stubs_without_losing_inherited_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.toml"
    result = CatalogStore(path).apply_changes(
        _changes(
            models={
                "glm-5-2": {
                    "deployments": [
                        {"provider": "test/default", "name": "test-glm-5-2"}
                    ]
                }
            }
        )
    )

    assert isinstance(result, CatalogWriteResult) and result.changed
    deployments = _written(path)["models"]["glm-5-2"]["deployments"]  # type: ignore[index]
    assert deployments[0] == {"provider": "mistral/default"}
    model = result.snapshot.catalog.models["glm-5-2"]
    assert model.thinking == "high"
    assert model.temperature == 0.2
    assert model.deployments[0].auto_compact_threshold == 400000.0


def test_tags_and_aliases_are_preserved_and_empty_tag_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.toml"
    store = CatalogStore(path)
    first = store.apply_changes(
        _changes(
            models={
                "one": {
                    "aliases": ["one-alias"],
                    "deployments": [{"provider": "test/default", "name": "one"}],
                },
                "two": {"deployments": [{"provider": "test/default", "name": "two"}]},
            },
            tags={"first": ("one",), "second": ("two",)},
        )
    )
    assert isinstance(first, CatalogWriteResult)
    repeated = store.apply_changes(_changes(tags={"first": ("one",)}))
    assert isinstance(repeated, CatalogWriteResult) and not repeated.changed
    before = path.read_bytes()

    result = store.apply_changes(_changes(tags={"first": ()}))

    assert isinstance(result, CatalogValidationError)
    assert "cannot be empty" in result.message
    assert path.read_bytes() == before
    raw = _written(path)
    assert raw["models"]["one"]["aliases"] == ["one-alias"]  # type: ignore[index]
    assert raw["tags"] == {"first": ["one"], "second": ["two"]}


def test_zero_price_is_written_while_unknown_price_fields_are_omitted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.toml"
    result = CatalogStore(path).apply_changes(
        _changes(
            models={
                "free": {
                    "deployments": [
                        {
                            "provider": "test/default",
                            "name": "free",
                            "prices": {"input": 0.0},
                        }
                    ]
                }
            }
        )
    )

    assert isinstance(result, CatalogWriteResult)
    prices = _written(path)["models"]["free"]["deployments"][0]["prices"]  # type: ignore[index]
    assert prices == {"input": 0.0}
    deployment = result.snapshot.catalog.models["free"].deployments[0]
    assert deployment.prices.input == 0.0
    assert deployment.prices.output is None


def test_failed_validation_leaves_original_bytes_and_repeated_save_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.toml"
    store = CatalogStore(path)
    changes = _changes(
        models={
            "saved": {"deployments": [{"provider": "test/default", "name": "saved"}]}
        }
    )
    first = store.apply_changes(changes)
    assert isinstance(first, CatalogWriteResult) and first.changed
    initial = path.read_bytes()

    repeated = store.apply_changes(changes)
    invalid = store.apply_changes(
        _changes(
            models={
                "invalid": {
                    "deployments": [{"provider": "test/default", "name": "bad@wire"}]
                }
            }
        )
    )

    assert isinstance(repeated, CatalogWriteResult) and not repeated.changed
    assert isinstance(invalid, CatalogValidationError)
    assert path.read_bytes() == initial


def test_interleaved_writes_preserve_both_changes(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "models.toml"
    store = CatalogStore(path)
    first_write_started = threading.Event()
    release_first_write = threading.Event()
    second_read_started = threading.Event()
    read_lock = threading.Lock()
    reads = 0
    first_call = True
    original = store._atomic_write
    read_overlay = store._read_overlay

    def interleaved_read() -> dict[str, object]:
        nonlocal reads
        with read_lock:
            reads += 1
            if reads == 2:
                second_read_started.set()
        return read_overlay()

    def interleave(overlay: object) -> None:
        nonlocal first_call
        if first_call:
            first_call = False
            first_write_started.set()
            assert release_first_write.wait(timeout=5)
        original(overlay)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "_read_overlay", interleaved_read)
    monkeypatch.setattr(store, "_atomic_write", interleave)
    first = threading.Thread(target=lambda: store.apply_changes(_changes()))
    second = threading.Thread(
        target=lambda: store.apply_changes(
            CatalogChanges("second/default", {"api_base": "https://second.example/v1"})
        )
    )

    first.start()
    assert first_write_started.wait(timeout=5)
    second.start()
    assert not second_read_started.wait(timeout=0.1)
    release_first_write.set()
    assert second_read_started.wait(timeout=5)
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert _written(path)["providers"] == {
        "test/default": {"api_base": "https://test.example/v1"},
        "second/default": {"api_base": "https://second.example/v1"},
    }


def test_process_independent_stores_preserve_both_writes(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    context = multiprocessing.get_context("fork")
    first_read = context.Queue()
    release_first = context.Queue()
    first_result = context.Queue()
    second_result = context.Queue()
    first = context.Process(
        target=_apply_changes_in_process,
        args=(path, "first/default", first_read, release_first, first_result),
    )
    first.start()
    first_read.get(timeout=5)
    second = context.Process(
        target=_apply_changes_in_process,
        args=(path, "second/default", None, None, second_result),
    )
    second.start()
    time.sleep(0.1)
    release_first.put(None)
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert first_result.get(timeout=1) == "success"
    assert second_result.get(timeout=1) == "success"
    providers = _written(path)["providers"]
    assert isinstance(providers, dict)
    assert set(providers) == {"first/default", "second/default"}


def test_atomic_writes_use_unique_temp_files(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "models.toml"
    names: list[str] = []
    original = tempfile.mkstemp

    def capture(*args: object, **kwargs: object) -> tuple[int, str]:
        descriptor, name = original(*args, **kwargs)  # type: ignore[arg-type]
        names.append(name)
        return descriptor, name

    monkeypatch.setattr(tempfile, "mkstemp", capture)
    store = CatalogStore(path)
    assert isinstance(store.apply_changes(_changes()), CatalogWriteResult)
    assert isinstance(
        store.apply_changes(_changes(provider={"api_base": "https://other.example"})),
        CatalogWriteResult,
    )

    assert len(names) == 2
    assert names[0] != names[1]
    assert all(Path(name).parent == tmp_path for name in names)
    assert not any(Path(name).exists() for name in names)
