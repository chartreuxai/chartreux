from __future__ import annotations

import asyncio
from collections.abc import Sequence
import copy
from pathlib import Path
import tomllib
from typing import Annotated, Any

from pydantic import Field, ValidationError
import pytest

from chartreux.core.config.event_bus import EventBus
from chartreux.core.config.layer import (
    ConfigLayer,
    ConfigPatchApplicationError,
    LayerImplementationError,
    LayerNotLoadedError,
    RawConfig,
)
from chartreux.core.config.layers.environment import EnvironmentLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import (
    ConfigOrchestrator,
    ConfigPatchValidationError,
    DefaultLayerResolutionError,
)
from chartreux.core.config.patch import (
    AddOperationPatch,
    RemoveOperationPatch,
    ReplaceOperationPatch,
)
from chartreux.core.config.schema import (
    ConfigFragment,
    ConfigSchema,
    WithConcatMerge,
    WithReplaceMerge,
)
from chartreux.core.config.types import (
    ConcurrencyConflictError,
    ConfigChangeEvent,
    LayerConfigSnapshot,
)


class FakeLayer(ConfigLayer[RawConfig]):
    def __init__(self, *, name: str, data: dict[str, Any]) -> None:
        super().__init__(name=name)
        self._data = data

    async def _check_trust(self) -> bool:
        return True

    async def _build_config_snapshot(self) -> LayerConfigSnapshot:
        return LayerConfigSnapshot(data=dict(self._data), fingerprint="fp")

    async def _save_to_store(self, _next_config: RawConfig) -> str:
        raise NotImplementedError


class NormalizingWritableLayer(FakeLayer):
    async def _save_to_store(self, next_config: RawConfig) -> str:
        value = next_config.model_dump()["value"]
        self._data = {"value": f"{value}-normalized"}
        return "write-fp"


class FieldWritableLayer(FakeLayer):
    def __init__(
        self,
        *,
        name: str,
        field_name: str,
        data: dict[str, Any],
        barrier: ParallelSaveBarrier | None = None,
    ) -> None:
        super().__init__(name=name, data=data)
        self._field_name = field_name
        self._barrier = barrier

    async def _save_to_store(self, next_config: RawConfig) -> str:
        if self._barrier is not None:
            await self._barrier.wait(self.name)

        value = next_config.model_dump()[self._field_name]
        self._data = {self._field_name: value}
        return "write-fp"


class RawWritableLayer(FakeLayer):
    async def _save_to_store(self, next_config: RawConfig) -> str:
        self._data = next_config.model_dump()
        return "write-fp"


class CoordinatedLoadLayer(RawWritableLayer):
    def __init__(self, *, name: str, data: dict[str, Any]) -> None:
        super().__init__(name=name, data=data)
        self._block_loads = False
        self._blocked_loads = 0
        self.first_load_entered = asyncio.Event()
        self.second_load_entered = asyncio.Event()
        self.release_loads = asyncio.Event()

    def __deepcopy__(self, memo: dict[int, Any]) -> CoordinatedLoadLayer:
        clone = type(self)(name=self.name, data=copy.deepcopy(self._data, memo))
        clone._state = copy.deepcopy(self._state, memo)
        clone._block_loads = self._block_loads
        clone._blocked_loads = self._blocked_loads
        clone.first_load_entered = self.first_load_entered
        clone.second_load_entered = self.second_load_entered
        clone.release_loads = self.release_loads
        return clone

    async def load(self, *, force: bool = False) -> RawConfig:
        data = await super().load(force=force)
        if self._block_loads:
            self._blocked_loads += 1
            if self._blocked_loads == 1:
                self.first_load_entered.set()
            elif self._blocked_loads == 2:
                self.second_load_entered.set()
            await self.release_loads.wait()
        return data


class FailingSaveLayer(FakeLayer):
    async def _save_to_store(self, _next_config: RawConfig) -> str:
        raise RuntimeError("boom")


class ApplyErrorLayer(FakeLayer):
    def __init__(self, *, name: str, data: dict[str, Any], error: Exception) -> None:
        super().__init__(name=name, data=data)
        self._error = error

    def __deepcopy__(self, memo: dict[int, Any]) -> ApplyErrorLayer:
        # The injected exception is an external failure fixture, not config state.
        clone = type(self)(
            name=self.name, data=copy.deepcopy(self._data, memo), error=self._error
        )
        clone._state = copy.deepcopy(self._state, memo)
        return clone

    async def apply(self, *args: Any, **kwargs: Any) -> None:
        raise self._error


class ParallelSaveBarrier:
    def __init__(self, expected_starts: int) -> None:
        self.expected_starts = expected_starts
        self.started_layers: list[str] = []
        self.all_started = asyncio.Event()

    def __deepcopy__(self, memo: dict[int, Any]) -> ParallelSaveBarrier:
        # Like a backing store, the save barrier is shared across staged reads.
        return self

    async def wait(self, layer_name: str) -> None:
        self.started_layers.append(layer_name)
        if len(self.started_layers) == self.expected_starts:
            self.all_started.set()

        await asyncio.wait_for(self.all_started.wait(), timeout=0.1)


class SimpleSchema(ConfigSchema):
    value: Annotated[str, WithReplaceMerge()] = "default"


class MultiValueSchema(ConfigSchema):
    first: Annotated[str, WithReplaceMerge()] = "default-first"
    second: Annotated[str, WithReplaceMerge()] = "default-second"


class ListSchema(ConfigSchema):
    values: Annotated[list[str], WithReplaceMerge()] = Field(default_factory=list)


class EntriesSchema(ConfigSchema):
    entries: Annotated[list[dict[str, str]], WithReplaceMerge()] = Field(
        default_factory=list
    )


class ToolsFragment(ConfigFragment):
    enabled_tools: Annotated[list[str], WithConcatMerge()] = Field(default_factory=list)
    disabled_tools: Annotated[list[str], WithConcatMerge()] = Field(
        default_factory=list
    )
    deprecated_setting: Annotated[bool, WithReplaceMerge()] = False


class ToolSchema(ConfigSchema):
    active_model: Annotated[str, WithReplaceMerge()] = "default-model"
    tools: ToolsFragment = Field(default_factory=ToolsFragment)


class RoutingSchema(ConfigSchema):
    active_model: Annotated[str, WithReplaceMerge()] = "default-model"
    selected_agent: Annotated[str, WithReplaceMerge()] = "default-agent"


class CliRoutingSchema(ConfigSchema):
    active_model: Annotated[str, WithReplaceMerge()] = "default-model"
    selected_agent: Annotated[str, WithReplaceMerge()] = "default-agent"
    enabled_tools: Annotated[list[str], WithConcatMerge()] = Field(default_factory=list)


class RequiredPairSchema(ConfigSchema):
    first: Annotated[str, WithReplaceMerge()]
    second: Annotated[str, WithReplaceMerge()]


def assert_single_failure[E: BaseException](
    result: Sequence[BaseException], expected_type: type[E]
) -> E:
    assert len(result) == 1
    failure = result[0]
    assert isinstance(failure, expected_type)
    return failure


def unused_default_layer() -> ConfigLayer[RawConfig]:
    return FakeLayer(name="unused-default", data={})


@pytest.mark.asyncio
async def test_create_builds_config() -> None:
    layer = FakeLayer(name="test", data={"value": "hello"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )
    assert orch.config.model_dump() == {"value": "hello"}
    assert orch.config.origin_of("value") == "test"


@pytest.mark.asyncio
async def test_get_layer_returns_named_layer() -> None:
    layer = FakeLayer(name="my-layer", data={})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )
    assert orch.get_layer("my-layer") is layer


@pytest.mark.asyncio
async def test_insert_layer_takes_effect_after_reload() -> None:
    base = FakeLayer(name="base", data={"value": "base"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[base], default_layer_resolver=lambda: base
    )
    assert orch.config.value == "base"

    orch.insert_layer(FakeLayer(name="top", data={"value": "top"}), 1)
    await orch.reload()

    assert orch.config.value == "top"
    assert [layer.name for layer in orch.layers] == ["base", "top"]


@pytest.mark.asyncio
async def test_remove_layer_takes_effect_after_reload() -> None:
    base = FakeLayer(name="base", data={"value": "base"})
    top = FakeLayer(name="top", data={"value": "top"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[base, top], default_layer_resolver=lambda: base
    )
    assert orch.config.value == "top"

    removed = orch.remove_layer(1)
    await orch.reload()

    assert removed is top
    assert orch.config.value == "base"
    assert [layer.name for layer in orch.layers] == ["base"]


@pytest.mark.asyncio
async def test_get_layer_unknown_raises() -> None:
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[], default_layer_resolver=unused_default_layer
    )
    with pytest.raises(KeyError, match="No layer named 'unknown'"):
        orch.get_layer("unknown")


@pytest.mark.asyncio
async def test_reload_picks_up_changes() -> None:
    layer = FakeLayer(name="mutable", data={"value": "original"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )
    assert orch.config.value == "original"

    layer._data = {"value": "updated"}
    await orch.reload()
    assert orch.config.value == "updated"


@pytest.mark.asyncio
async def test_config_is_immutable() -> None:
    layer = FakeLayer(name="test", data={"value": "hello"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )
    with pytest.raises(ValidationError, match="frozen"):
        orch.config.value = "changed"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_origin_of_missing_key_returns_none() -> None:
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[], default_layer_resolver=unused_default_layer
    )
    assert orch.config.origin_of("nonexistent") is None


@pytest.mark.asyncio
async def test_apply_patch_empty_operations_is_noop() -> None:
    layer = FakeLayer(name="test", data={"value": "hello"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )

    result = await orch.apply_patch([], reason="no-op")

    assert result == []
    assert orch.config.value == "hello"


@pytest.mark.asyncio
async def test_apply_patch_rejects_invalid_schema_result_before_routing() -> None:
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[], default_layer_resolver=unused_default_layer
    )
    with pytest.raises(ConfigPatchValidationError) as exc_info:
        await orch.apply_patch(
            [ReplaceOperationPatch(path="/value", value={"invalid": "shape"})],
            reason="test invalid patch",
        )

    assert exc_info.value.args == (
        "Config patch failed preflight validation against the merged config; fix the patch payload and retry",
    )
    assert isinstance(exc_info.value.__cause__, ValidationError)


@pytest.mark.asyncio
async def test_apply_patch_unknown_explicit_target_returns_failure() -> None:
    layer = NormalizingWritableLayer(name="user-toml", data={})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )

    result = await orch.apply_patch(
        [
            AddOperationPatch(
                path="/value", value="updated", target_layer_name="missing-layer"
            )
        ],
        reason="test update",
    )

    failure = assert_single_failure(result, KeyError)
    assert str(failure) == "\"No layer named 'missing-layer'\""
    assert orch.config.value == "default"
    assert layer._data == {}


@pytest.mark.asyncio
async def test_apply_patch_uses_default_layer_resolver_and_reloads_config() -> None:
    layer = NormalizingWritableLayer(name="user-toml", data={})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )

    result = await orch.apply_patch(
        [AddOperationPatch(path="/value", value="updated")], reason="test update"
    )

    assert result == []
    assert orch.config.value == "updated"
    await orch.reload()
    assert orch.config.value == "updated-normalized"
    assert layer._data == {"value": "updated-normalized"}


@pytest.mark.asyncio
async def test_set_field_uses_default_layer_resolver() -> None:
    layer = NormalizingWritableLayer(name="user-toml", data={})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )

    result = await orch.set_field("/value", "updated", reason="set field")

    assert result == []
    assert orch.config.value == "updated"
    await orch.reload()
    assert orch.config.value == "updated-normalized"


@pytest.mark.asyncio
async def test_apply_patch_preflight_sees_merged_candidate_before_write() -> None:
    writable = RawWritableLayer(name="user-toml", data={})
    higher_priority = FakeLayer(name="environment", data={"value": "masked"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema,
        layers=[writable, higher_priority],
        default_layer_resolver=lambda: writable,
    )
    seen: list[str] = []

    async def preflight(candidate: SimpleSchema) -> None:
        seen.append(candidate.value)
        assert writable._data == {}

    result = await orch.set_field(
        "/value", "persisted", reason="preflight candidate", preflight=preflight
    )

    assert result == []
    assert seen == ["masked"]
    assert writable._data == {"value": "persisted"}
    assert orch.config.value == "masked"


@pytest.mark.asyncio
async def test_apply_patch_preflight_rejection_prevents_persistence() -> None:
    class RejectedCandidate(Exception):
        pass

    layer = RawWritableLayer(name="user-toml", data={"value": "original"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )

    async def reject(candidate: SimpleSchema) -> None:
        assert candidate.value == "updated"
        raise RejectedCandidate

    with pytest.raises(RejectedCandidate):
        await orch.set_field("/value", "updated", preflight=reject)

    assert layer._data == {"value": "original"}
    assert orch.config.value == "original"


@pytest.mark.asyncio
async def test_reload_preflight_rejection_keeps_the_previous_snapshot() -> None:
    """*Prepare*: A loaded config whose backing layer changes to a rejected value.
    *Do*: Reload with a preflight that rejects the rebuilt candidate.
    *Assert*: The orchestrator keeps serving the previously accepted snapshot.
    """

    class RejectedCandidate(Exception):
        pass

    # Prepare
    layer = RawWritableLayer(name="user-toml", data={"value": "original"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )
    layer._data = {"value": "rejected"}

    async def reject(candidate: SimpleSchema) -> None:
        assert candidate.value == "rejected"
        raise RejectedCandidate

    # Do / Assert
    with pytest.raises(RejectedCandidate):
        await orch.reload(preflight=reject)

    assert orch.config.value == "original"


@pytest.mark.asyncio
async def test_apply_patch_serializes_preflight_through_persistence() -> None:
    layer = RawWritableLayer(name="user-toml", data={"value": "original"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first_preflight(candidate: SimpleSchema) -> None:
        assert candidate.value == "first"
        first_entered.set()
        await release_first.wait()

    async def second_preflight(candidate: SimpleSchema) -> None:
        assert candidate.value == "second"
        second_entered.set()

    first = asyncio.create_task(
        orch.set_field("/value", "first", preflight=first_preflight)
    )
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    second = asyncio.create_task(
        orch.set_field("/value", "second", preflight=second_preflight)
    )
    await asyncio.sleep(0)
    assert not second_entered.is_set()

    release_first.set()
    assert await first == []
    assert await second == []
    assert second_entered.is_set()
    assert layer._data == {"value": "second"}
    assert orch.config.value == "second"


@pytest.mark.asyncio
async def test_mutate_field_serializes_read_modify_write_with_preflight() -> None:
    layer = RawWritableLayer(name="user-toml", data={"values": []})
    orch = await ConfigOrchestrator.create(
        schema=ListSchema, layers=[layer], default_layer_resolver=lambda: layer
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first_preflight(candidate: ListSchema) -> None:
        assert candidate.values == ["first"]
        first_entered.set()
        await release_first.wait()

    async def second_preflight(candidate: ListSchema) -> None:
        assert candidate.values == ["first", "second"]
        second_entered.set()

    first = asyncio.create_task(
        orch.mutate_field(
            "/values",
            lambda values: [*values, "first"],
            default=[],
            preflight=first_preflight,
        )
    )
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    second = asyncio.create_task(
        orch.mutate_field(
            "/values",
            lambda values: [*values, "second"],
            default=[],
            preflight=second_preflight,
        )
    )
    await asyncio.sleep(0)
    assert not second_entered.is_set()

    release_first.set()
    assert await first == []
    assert await second == []
    assert layer._data == {"values": ["first", "second"]}
    assert orch.config.values == ["first", "second"]


@pytest.mark.asyncio
async def test_upsert_field_serializes_read_select_apply() -> None:
    layer = CoordinatedLoadLayer(name="user-toml", data={"entries": []})
    orch = await ConfigOrchestrator.create(
        schema=EntriesSchema, layers=[layer], default_layer_resolver=lambda: layer
    )
    layer._block_loads = True
    first = asyncio.create_task(
        orch.upsert_field(
            "/entries", key_field="name", value={"name": "provider", "value": "one"}
        )
    )
    await asyncio.wait_for(layer.first_load_entered.wait(), timeout=1)
    second = asyncio.create_task(
        orch.upsert_field(
            "/entries", key_field="name", value={"name": "provider", "value": "two"}
        )
    )
    try:
        await asyncio.wait_for(layer.second_load_entered.wait(), timeout=0.1)
    except TimeoutError:
        pass
    finally:
        layer.release_loads.set()

    assert await first == []
    assert await second == []
    assert len(layer._data["entries"]) == 1
    assert layer._data["entries"][0]["name"] == "provider"
    assert len(orch.config.entries) == 1


@pytest.mark.asyncio
async def test_apply_patch_publishes_change_event_after_reload() -> None:
    layer = RawWritableLayer(name="user-toml", data={"value": "hello"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )
    received: list[ConfigChangeEvent] = []
    orch.subscribe(received.append)

    result = await orch.apply_patch(
        [ReplaceOperationPatch(path="/value", value="updated")], reason="test update"
    )

    assert result == []
    assert len(received) == 1
    event = received[0]
    assert event.changed_keys == frozenset({"value"})
    assert event.before == {"value": "hello"}
    assert event.after == {"value": "updated"}
    assert event.reason == "test update"
    assert orch.config.value == "updated"


@pytest.mark.asyncio
async def test_apply_patch_does_not_publish_when_merged_value_is_unchanged() -> None:
    layer = RawWritableLayer(name="user-toml", data={"value": "hello"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )
    received: list[ConfigChangeEvent] = []
    orch.subscribe(received.append)

    result = await orch.apply_patch(
        [ReplaceOperationPatch(path="/value", value="hello")], reason="no-op update"
    )

    assert result == []
    assert received == []
    assert orch.config.value == "hello"


@pytest.mark.asyncio
async def test_apply_patch_does_not_publish_when_higher_priority_layer_masks_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHARTREUX_ACTIVE_MODEL", "env-model")
    user_layer = RawWritableLayer(name="user-toml", data={})
    orch = await ConfigOrchestrator.create(
        schema=RoutingSchema,
        layers=[user_layer, EnvironmentLayer(schema=RoutingSchema)],
        default_layer_resolver=lambda: user_layer,
    )
    received: list[ConfigChangeEvent] = []
    orch.subscribe(received.append)

    result = await orch.apply_patch(
        [AddOperationPatch(path="/active_model", value="persisted-in-user-file")],
        reason="masked update",
    )

    assert result == []
    assert received == []
    assert user_layer._data == {"active_model": "persisted-in-user-file"}
    assert orch.config.active_model == "env-model"


@pytest.mark.asyncio
async def test_apply_patch_publishes_only_successful_layer_operations() -> None:
    first_layer = FieldWritableLayer(
        name="first-layer", field_name="first", data={"first": "one"}
    )
    second_layer = FailingSaveLayer(name="second-layer", data={"second": "two"})
    orch = await ConfigOrchestrator.create(
        schema=MultiValueSchema,
        layers=[first_layer, second_layer],
        default_layer_resolver=lambda: first_layer,
    )
    wildcard_events: list[ConfigChangeEvent] = []
    second_events: list[ConfigChangeEvent] = []
    orch.subscribe(wildcard_events.append)
    orch.subscribe(second_events.append, keys={"second"})

    result = await orch.apply_patch(
        [
            ReplaceOperationPatch(
                path="/first", value="updated-one", target_layer_name="first-layer"
            ),
            ReplaceOperationPatch(
                path="/second", value="updated-two", target_layer_name="second-layer"
            ),
        ],
        reason="test partial apply",
    )

    assert_single_failure(result, LayerImplementationError)
    assert len(wildcard_events) == 1
    assert second_events == []
    event = wildcard_events[0]
    assert event.changed_keys == frozenset({"first"})
    assert event.before == {"first": "one", "second": "two"}
    assert event.after == {"first": "updated-one", "second": "two"}
    assert event.reason == "test partial apply"


@pytest.mark.asyncio
async def test_apply_patch_publishes_changed_list_field_as_subscribable_key() -> None:
    layer = RawWritableLayer(
        name="user-toml", data={"tools": {"enabled_tools": ["read"]}}
    )
    orch = await ConfigOrchestrator.create(
        schema=ToolSchema, layers=[layer], default_layer_resolver=lambda: layer
    )
    received: list[ConfigChangeEvent] = []
    orch.subscribe(received.append, keys={"tools/enabled_tools"})

    result = await orch.apply_patch(
        [AddOperationPatch(path="/tools/enabled_tools/-", value="grep")],
        reason="append tool",
    )

    assert result == []
    assert len(received) == 1
    assert received[0].changed_keys == frozenset({"tools/enabled_tools"})
    assert orch.config.tools.enabled_tools == ["read", "grep"]


@pytest.mark.asyncio
async def test_apply_patch_does_not_publish_when_all_layers_fail() -> None:
    layer = ApplyErrorLayer(name="test", data={"value": "hello"}, error=RuntimeError())
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )
    received: list[ConfigChangeEvent] = []
    orch.subscribe(received.append)

    result = await orch.apply_patch(
        [ReplaceOperationPatch(path="/value", value="updated")], reason="failed update"
    )

    assert_single_failure(result, RuntimeError)
    assert received == []


@pytest.mark.asyncio
async def test_copy_preserves_default_layer_resolution() -> None:
    layer = RawWritableLayer(name="user-toml", data={})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )

    forked = orch.copy()
    result = await forked.set_field("/value", "forked", reason="set field")

    assert result == []
    assert forked.config.value == "forked"
    assert orch.config.value == "default"


@pytest.mark.asyncio
async def test_apply_patch_returns_layer_save_error_after_other_layer_commits() -> None:
    first_layer = FieldWritableLayer(
        name="first-layer", field_name="first", data={"first": "one"}
    )
    second_layer = FailingSaveLayer(name="second-layer", data={"second": "two"})
    orch = await ConfigOrchestrator.create(
        schema=MultiValueSchema,
        layers=[first_layer, second_layer],
        default_layer_resolver=lambda: first_layer,
    )

    result = await orch.apply_patch(
        [
            ReplaceOperationPatch(
                path="/first", value="updated-one", target_layer_name="first-layer"
            ),
            ReplaceOperationPatch(
                path="/second", value="updated-two", target_layer_name="second-layer"
            ),
        ],
        reason="test partial apply",
    )

    failure = assert_single_failure(result, LayerImplementationError)
    assert str(failure) == "Layer 'second-layer': _save_to_store() failed"
    assert isinstance(failure.__cause__, RuntimeError)
    assert first_layer._data == {"first": "updated-one"}
    assert second_layer._data == {"second": "two"}


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            ConcurrencyConflictError(expected_fp="before", actual_fp="after"),
            id="concurrency-conflict",
        ),
        pytest.param(ConfigPatchApplicationError("test"), id="patch-application-error"),
        pytest.param(RuntimeError("unexpected bug"), id="unexpected-runtime-error"),
    ],
)
@pytest.mark.asyncio
async def test_apply_patch_returns_layer_apply_error_in_failures(
    error: Exception,
) -> None:
    layer = ApplyErrorLayer(name="test", data={"value": "hello"}, error=error)
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )

    result = await orch.apply_patch(
        [
            ReplaceOperationPatch(
                path="/value", value="updated", target_layer_name="test"
            )
        ],
        reason="test update",
    )

    assert assert_single_failure(result, type(error)) is error


@pytest.mark.asyncio
async def test_apply_patch_returns_unloaded_layer_error_in_failures() -> None:
    loaded_layer = FakeLayer(name="loaded", data={"value": "hello"})
    target_layer = FakeLayer(name="target", data={"value": "hello"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema,
        layers=[loaded_layer, target_layer],
        default_layer_resolver=lambda: loaded_layer,
    )
    await target_layer.invalidate_cache()

    result = await orch.apply_patch(
        [
            ReplaceOperationPatch(
                path="/value", value="updated", target_layer_name="target"
            )
        ],
        reason="test update",
    )

    failure = assert_single_failure(result, LayerNotLoadedError)
    assert str(failure) == "Layer 'target' must be loaded before applying patches"


@pytest.mark.asyncio
async def test_apply_patch_applies_layers_in_parallel() -> None:
    barrier = ParallelSaveBarrier(expected_starts=2)
    first_layer = FieldWritableLayer(
        name="first-layer", field_name="first", data={"first": "one"}, barrier=barrier
    )
    second_layer = FieldWritableLayer(
        name="second-layer",
        field_name="second",
        data={"second": "two"},
        barrier=barrier,
    )
    orch = await ConfigOrchestrator.create(
        schema=MultiValueSchema,
        layers=[first_layer, second_layer],
        default_layer_resolver=lambda: first_layer,
    )

    result = await orch.apply_patch(
        [
            ReplaceOperationPatch(
                path="/first", value="updated-one", target_layer_name="first-layer"
            ),
            ReplaceOperationPatch(
                path="/second", value="updated-two", target_layer_name="second-layer"
            ),
        ],
        reason="test parallel apply",
    )

    assert result == []
    assert barrier.started_layers == ["first-layer", "second-layer"]
    assert first_layer._data == {"first": "updated-one"}
    assert second_layer._data == {"second": "updated-two"}


@pytest.mark.asyncio
async def test_apply_patch_end_to_end_updates_real_user_config_file(
    tmp_working_directory: Path,
) -> None:
    toml_path = tmp_working_directory / "config.toml"
    toml_path.write_text(
        """\
active_model = "old"

[tools]
disabled_tools = ["bash", "python"]
deprecated_setting = true
""",
        encoding="utf-8",
    )

    user_layer = UserConfigLayer(path=toml_path)
    orch = await ConfigOrchestrator.create(
        schema=ToolSchema,
        layers=[user_layer],
        default_layer_resolver=lambda: user_layer,
    )

    result = await orch.apply_patch(
        [
            ReplaceOperationPatch(path="/active_model", value="new"),
            AddOperationPatch(path="/tools/enabled_tools", value=["read"]),
            AddOperationPatch(path="/tools/disabled_tools/-", value="node"),
            RemoveOperationPatch(path="/tools/disabled_tools/0"),
            RemoveOperationPatch(path="/tools/deprecated_setting"),
        ],
        reason="update user defaults",
    )

    assert result == []
    with toml_path.open("rb") as file:
        assert tomllib.load(file) == {
            "active_model": "new",
            "tools": {"disabled_tools": ["python", "node"], "enabled_tools": ["read"]},
        }
    assert orch.config.active_model == "new"
    assert orch.config.tools.disabled_tools == ["python", "node"]
    assert orch.config.tools.enabled_tools == ["read"]


@pytest.mark.asyncio
async def test_persisted_active_model_reads_the_pinned_value(
    tmp_working_directory: Path,
) -> None:
    toml_path = tmp_working_directory / "config.toml"
    toml_path.write_text(
        'active_model = "target-testing-model-alias"\n', encoding="utf-8"
    )
    user_layer = UserConfigLayer(path=toml_path)
    orch = await ConfigOrchestrator.create(
        schema=ToolSchema,
        layers=[user_layer],
        default_layer_resolver=lambda: user_layer,
    )

    assert orch.persisted_active_model() == "target-testing-model-alias"


@pytest.mark.asyncio
async def test_persisted_active_model_empty_when_not_pinned(
    tmp_working_directory: Path,
) -> None:
    toml_path = tmp_working_directory / "config.toml"
    # No active_model key persisted: the user is on the unpinned default.
    toml_path.write_text("tools = {}\n", encoding="utf-8")
    user_layer = UserConfigLayer(path=toml_path)
    orch = await ConfigOrchestrator.create(
        schema=ToolSchema,
        layers=[user_layer],
        default_layer_resolver=lambda: user_layer,
    )

    assert orch.persisted_active_model() == ""


@pytest.mark.asyncio
async def test_apply_patch_creates_user_file_when_it_is_missing(
    tmp_working_directory: Path,
) -> None:
    toml_path = tmp_working_directory / "config.toml"
    user_layer = UserConfigLayer(path=toml_path)
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema,
        layers=[user_layer],
        default_layer_resolver=lambda: user_layer,
    )

    result = await orch.apply_patch(
        [AddOperationPatch(path="/value", value="created-later")],
        reason="fallback write without user file",
    )

    assert result == []
    with toml_path.open("rb") as file:
        assert tomllib.load(file) == {"value": "created-later"}
    assert orch.config.value == "created-later"


@pytest.mark.asyncio
async def test_set_field_appends_to_concat_list_without_rebuilding_from_merged(
    tmp_working_directory: Path,
) -> None:
    toml_path = tmp_working_directory / "config.toml"
    toml_path.write_text(
        'active_model = "old"\n\n[tools]\nenabled_tools = ["read"]\n', encoding="utf-8"
    )

    user_layer = UserConfigLayer(path=toml_path)
    orch = await ConfigOrchestrator.create(
        schema=ToolSchema,
        layers=[user_layer],
        default_layer_resolver=lambda: user_layer,
    )

    assert await orch.set_field("/tools/enabled_tools/-", "grep") == []

    with toml_path.open("rb") as file:
        assert tomllib.load(file) == {
            "active_model": "old",
            "tools": {"enabled_tools": ["read", "grep"]},
        }
    assert orch.config.tools.enabled_tools == ["read", "grep"]


@pytest.mark.asyncio
async def test_apply_patch_returns_failure_when_resolver_returns_unknown_layer() -> (
    None
):
    loaded_layer = RawWritableLayer(name="loaded", data={})
    unknown_layer = RawWritableLayer(name="unknown", data={})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema,
        layers=[loaded_layer],
        default_layer_resolver=lambda: unknown_layer,
    )

    with pytest.raises(DefaultLayerResolutionError, match="unknown layer 'unknown'"):
        await orch.apply_patch(
            [AddOperationPatch(path="/value", value="updated")],
            reason="unknown resolver target",
        )

    assert orch.config.value == "default"


@pytest.mark.asyncio
async def test_apply_patch_explicit_target_does_not_resolve_default_layer() -> None:
    layer = RawWritableLayer(name="target", data={"value": "original"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema,
        layers=[layer],
        default_layer_resolver=lambda: (_ for _ in ()).throw(
            AssertionError("resolver should not be called")
        ),
    )

    result = await orch.apply_patch(
        [
            ReplaceOperationPatch(
                path="/value", value="updated", target_layer_name="target"
            )
        ],
        reason="explicit target only",
    )

    assert result == []
    assert orch.config.value == "updated"


@pytest.mark.asyncio
async def test_subscribe_registers_on_the_bus() -> None:
    bus = EventBus()
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema,
        layers=[],
        default_layer_resolver=unused_default_layer,
        bus=bus,
    )
    received: list[ConfigChangeEvent] = []
    orch.subscribe(received.append)

    event = ConfigChangeEvent(
        changed_keys=frozenset({"value"}), before={}, after={}, reason=""
    )
    bus.publish(event)

    assert received == [event]


@pytest.mark.asyncio
async def test_orchestrator_deepcopies_and_stays_functional() -> None:
    """IMPORTANT: If this test fails, care about the deep copy made in fork() in agent loop.
    Either fix the deepcopy or remove it and implement a proper clone() method on the orchestrator.
    """
    layer = FakeLayer(name="test", data={"value": "hello"})
    orch = await ConfigOrchestrator.create(
        schema=SimpleSchema, layers=[layer], default_layer_resolver=lambda: layer
    )

    forked = copy.deepcopy(orch)

    assert forked.config.value == "hello"
    forked_layer = forked.get_layer("test")
    assert forked_layer is not layer
    assert isinstance(forked_layer, FakeLayer)

    forked_layer._data = {"value": "forked"}
    await forked.reload()

    assert forked.config.value == "forked"
    assert orch.config.value == "hello"
