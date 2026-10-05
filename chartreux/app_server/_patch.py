from __future__ import annotations

from collections.abc import Collection
from copy import deepcopy
from typing import Any, cast

from jsonpatch import apply_patch, make_patch
from jsonpointer import resolve_pointer
from pydantic import JsonValue

from chartreux.app_server.models import (
    CancelledEffectState,
    CompletedEffectState,
    FailedEffectState,
    JsonPatchOperation,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
)


def _json_equal(left: JsonValue, right: JsonValue) -> bool:
    """Compare JSON recursively without Python's bool/int equivalence."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def is_completed_effect_replay(
    entry: PublicHistoryEntry, operations: list[JsonPatchOperation]
) -> bool:
    """Recognize an already-reflected completion, ignoring its delivery timestamp."""
    if not (
        isinstance(entry, PublicEffectEntry)
        and isinstance(entry.state, CompletedEffectState)
        and entry.generation_status is PublicEntryGenerationStatus.COMPLETED
        and any(operation.path == "/state" for operation in operations)
    ):
        return False
    raw = entry.model_dump(mode="json", by_alias=True)
    for operation in operations:
        if operation.op != "replace" or operation.path not in {
            "/state",
            "/generationStatus",
            "/updatedAt",
        }:
            return False
        if operation.path != "/updatedAt" and not _json_equal(
            raw[operation.path[1:]], operation.value
        ):
            return False
    return True


def is_timing_only_patch(
    entry: PublicHistoryEntry, operations: list[JsonPatchOperation]
) -> bool:
    """Allow duration settlement and content-preserving terminal recovery replays."""
    raw = entry.model_dump(mode="json", by_alias=True)
    if isinstance(entry, PublicMessageEntry) and entry.role == "assistant":
        path = "/turnDurationMs"
    elif isinstance(entry, PublicEffectEntry) and isinstance(
        entry.state, CompletedEffectState | FailedEffectState | CancelledEffectState
    ):
        path = "/state/durationMs"
    else:
        path = None
    if not operations:
        return False
    for operation in operations:
        if operation.op in {"add", "replace"} and operation.path == path:
            continue
        if operation.op != "replace" or operation.path not in {
            "/state",
            "/generationStatus",
        }:
            return False
        field = operation.path[1:]
        if field not in raw:
            return False
        previous = raw[field]
        current = operation.value
        if path == "/state/durationMs" and field == "state":
            if not isinstance(current, dict) or not isinstance(previous, dict):
                return False
            previous = {
                key: value for key, value in previous.items() if key != "durationMs"
            }
            current = {
                key: value for key, value in current.items() if key != "durationMs"
            }
        if not _json_equal(previous, current):
            return False
    return True


def apply_json_patch(
    value: JsonValue, operations: list[JsonPatchOperation]
) -> JsonValue:
    document: JsonValue = deepcopy(value)
    for operation in operations:
        document = cast(
            JsonValue,
            apply_patch(
                document, [_standard_operation(document, operation)], in_place=True
            ),
        )
    return document


def _model_operations(
    source: dict[str, Any], raw_operation: dict[str, JsonValue]
) -> list[JsonPatchOperation]:
    """Validate a raw ``jsonpatch`` operation, expanding unsupported ops.

    ``jsonpatch``'s diff builder can emit ``move`` and ``copy`` operations, which
    carry a ``from`` pointer and are not part of the wire ``JsonPatchOperation``
    op set. Expand them into the modeled ``add``/``remove`` equivalents before
    validation so a relocated value never crashes the event stream.
    """
    match raw_operation.get("op"):
        case "move":
            from_path = str(raw_operation["from"])
            value = cast(JsonValue, resolve_pointer(source, from_path))
            return [
                JsonPatchOperation(op="remove", path=from_path),
                JsonPatchOperation(
                    op="add", path=str(raw_operation["path"]), value=value
                ),
            ]
        case "copy":
            value = cast(JsonValue, resolve_pointer(source, str(raw_operation["from"])))
            return [
                JsonPatchOperation(
                    op="add", path=str(raw_operation["path"]), value=value
                )
            ]
    return [JsonPatchOperation.model_validate(raw_operation)]


def make_json_patch(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    append_paths: Collection[str] = (),
) -> list[JsonPatchOperation]:
    operations: list[JsonPatchOperation] = []
    for raw_operation in make_patch(source, target).patch:
        modeled = _model_operations(source, raw_operation)
        if not (
            len(modeled) == 1
            and modeled[0].op == "replace"
            and modeled[0].path in append_paths
        ):
            operations.extend(modeled)
            continue
        previous = resolve_pointer(source, modeled[0].path)
        current = modeled[0].value
        if not (
            isinstance(previous, str)
            and isinstance(current, str)
            and current.startswith(previous)
        ):
            operations.extend(modeled)
            continue
        operations.append(
            JsonPatchOperation(
                op="append", path=modeled[0].path, value=current[len(previous) :]
            )
        )
    return operations


def _standard_operation(
    document: JsonValue, operation: JsonPatchOperation
) -> dict[str, JsonValue]:
    match operation.op:
        case "append":
            current = resolve_pointer(document, operation.path)
            if not isinstance(current, str) or not isinstance(operation.value, str):
                raise ValueError("Append patches require string values")
            return {
                "op": "replace",
                "path": operation.path,
                "value": current + operation.value,
            }
        case "remove":
            return {"op": operation.op, "path": operation.path}
        case "add" | "replace" | "test":
            return {
                "op": operation.op,
                "path": operation.path,
                "value": operation.value,
            }
