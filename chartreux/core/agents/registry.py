"""Child configuration assembly for launches (package 16a keeps the launch path).

The profile registry is retired with the profile axis (package 16a):
agents/*.toml discovery and the profile-override layer are gone, and the
child's resolved identity selection rides the launch-override inputs instead
(see :func:`build_child_orchestrator`). What remains is the pure layer
machinery the resource-free launch resolution needs.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import TYPE_CHECKING, cast

from chartreux.core.config.layers.launch_overrides import LaunchOverridesLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema


def _mutable_copy(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _mutable_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_copy(item) for item in value]
    return copy.deepcopy(value)


def apply_launch_overrides(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], data: dict[str, object]
) -> None:
    """Install one child-only launch layer after runtime overrides."""
    layers = orchestrator.layers
    current = next(
        (
            (index, layer)
            for index, layer in enumerate(layers)
            if isinstance(layer, LaunchOverridesLayer)
        ),
        None,
    )
    launch_layer = LaunchOverridesLayer(
        data=cast(dict[str, object], _mutable_copy(data)),
        name=current[1].name if current is not None else LaunchOverridesLayer.NAME,
    )
    if current is not None:
        orchestrator.remove_layer(current[0])
    insertion_index = max(
        (
            index + 1
            for index, layer in enumerate(orchestrator.layers)
            if isinstance(layer, OverridesLayer)
        ),
        default=len(orchestrator.layers),
    )
    orchestrator.insert_layer(launch_layer, insertion_index)
    orchestrator.rebuild()


def _strip_inherited_identity_selection(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> None:
    """Drop inherited runtime ``agent.identity`` overrides from a child chain.

    The child's identity is bound by the launch resolution: admission
    validates that binding and the runtime enforces it. An inherited runtime
    override (the parent session's own identity selection) sits in the merge
    order and would otherwise silently disagree with the validated admission
    (S2 identity disagreement): a parent running ``coder`` would launch a
    ``reviewer`` child that executes as ``coder``. Children never inherit the
    parent's identity selection; the launch inputs install the child's own
    selection afterwards.
    """
    for index, layer in enumerate(orchestrator.layers):
        if not isinstance(layer, OverridesLayer):
            continue
        data = dict(layer._data)
        agent = data.get("agent")
        if not (isinstance(agent, Mapping) and "identity" in agent):
            continue
        agent = {key: value for key, value in agent.items() if key != "identity"}
        if agent:
            data["agent"] = agent
        else:
            data.pop("agent", None)
        orchestrator.remove_layer(index)
        orchestrator.insert_layer(OverridesLayer(data=data, name=layer.name), index)


def _strip_runtime_thinking_overrides(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> None:
    """Drop inherited runtime thinking overrides for an explicitly modeled launch.

    Main-session thinking choices are local to the main assistant: a launch
    that carries its own model selection starts from that model's own preset
    even when it shares the selected model.
    """
    for index, layer in enumerate(orchestrator.layers):
        if not isinstance(layer, OverridesLayer):
            continue
        data = (
            dict(layer.cached_data.model_dump(exclude_unset=True))
            if layer.cached_data
            else {}
        )
        if "thinking_overrides" in data:
            data.pop("thinking_overrides")
            orchestrator.remove_layer(index)
            orchestrator.insert_layer(OverridesLayer(data=data, name=layer.name), index)
        break


def build_child_orchestrator(
    source: ConfigOrchestrator[ChartreuxConfigSchema],
    launch_overrides: Mapping[str, object],
    *,
    strip_runtime_thinking: bool = False,
) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    """Build the authoritative resource-free child configuration assembly.

    The launch overrides carry the child's resolved identity selection
    (``agent.identity``) and its model/thinking inputs; the child never
    inherits the parent's identity selection (S2 identity disagreement).
    ``strip_runtime_thinking`` drops the parent's runtime thinking overrides
    for a launch that carries its own model selection.
    """
    candidate = source._copy_for_child()
    _strip_inherited_identity_selection(candidate)
    if strip_runtime_thinking:
        _strip_runtime_thinking_overrides(candidate)
    apply_launch_overrides(candidate, dict(launch_overrides))
    return candidate
