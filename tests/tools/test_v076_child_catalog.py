"""P1 slice 1 — per-identity withheld sets and their recomputation.

The v0.7.6 bounded-child catalog gate replaces the single-name child
withholding (``ask_user_question`` only) with ONE rule, evaluated per child
identity: a tool definition is withheld from a bounded child's catalog IFF
the child cannot invoke the tool under the default deployment. Invocation
needs all three gates, so a structural denial by any one of them withholds:

1. the LOOP-LEVEL child restriction (``CHILD_CONTEXT_UNAVAILABLE_TOOLS``):
   children never prompt the user, they cannot spawn (``Task.run`` refuses at
   the depth limit of 1), and they hold no owner-side run inventory — the
   loop hands a child ``subagent_manager=None``, so check/wait/get/cancel/
   release/resume/reconcile are owner-only;
2. the IDENTITY POSTURE, including the identity's execution carve-out;
3. the LAUNCHING PARENT'S DELEGATION CEILING.

ENUMERATION COMPLETENESS TABLE. Default deployment = a ``coder`` root (the
shipped ``[agent].identity`` default, delegation ALL_GRANTED), so gate 3
withholds nothing by default. The universe is the 23 reviewed builtin names
in ``TOOL_CLASSIFICATIONS``:

| identity     | personal posture (classes + carve-out)            | withheld |
|--------------|---------------------------------------------------|----------|
| coder        | ALL_GRANTED                                       | 9        |
| orchestrator | NON_MUTATING + {bash, bash_start, edit, write_file} | 9      |
| researcher   | NON_MUTATING + {bash, bash_start, edit, write_file} | 9      |
| advisor      | NON_MUTATING + {bash, bash_start, edit, write_file} | 9      |
| reviewer     | NON_MUTATING + {bash}                             | 12       |
| custom       | none (no harness-known posture on either axis)    | 9        |

The 9-name loop-level core is ask_user_question, task, check_agents,
wait_for_agent, get_agent_result, cancel_agent, release_agent, resume_agent,
reconcile_result; the reviewer adds the mutating builtins its carve-out does
not reach (bash_start, edit, write_file).

AFFIRMATION: every name NOT in a withheld set is invocation-possible under
that deployment — it is on the manager's personal axis, which applies both
ceiling gates live, and resolves through ``get()``. The advertised high-mass
definitions are exactly those (bash and its job readers/writers, read_file,
grep, read_image, web_search, web_fetch, edit, write_file, skill, todo) — the
~24 KB ``task`` definition is withheld, which is the release's pinned 6k
saving gate (slice 3).

Withholding is advertisement-only: a forced or stale call for a withheld
name still resolves and dies on the tool's own explicit blocker error, a
single-turn execution error surfaced to the model — never an opaque
unknown-tool failure.

Slices 2 (MCP per-server gating, fragment/listing compression, the
prompt-text-vs-definition consistency check) and 3 (fixture-class behavioral
gates, the pinned task-definition saving gate, the per-tool savings table)
append their own classes below.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, ClassVar, NamedTuple, cast
from unittest.mock import AsyncMock, patch

import pytest

from chartreux.core.agent_loop._completion_coordinator import (
    _serialized_tool_definitions,
)
from chartreux.core.agents import AgentManager
from chartreux.core.agents.identities import (
    AgentIdentity,
    builtin_identity_names,
    get_builtin_identity,
)
from chartreux.core.agents.identity_launch import resolve_identity_launch
from chartreux.core.agents.launch import _validate_tools
from chartreux.core.config import (
    ChartreuxConfigSchema,
    MCPHttp,
    ModelConfig,
    ProviderConfig,
)
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.launch_types import LaunchConfig, LaunchToolOverride
from chartreux.core.llm.format import (
    APIToolFormatHandler,
    ParsedMessage,
    ParsedToolCall,
)
from chartreux.core.llm_models import AvailableTool, Backend
from chartreux.core.model_catalog.resolver import resolver_for
from chartreux.core.policy.composition import (
    FRAGMENTS_DIR,
    ExecutionContext,
    taught_tool_names,
)
from chartreux.core.skills.manager import SkillManager
from chartreux.core.subagents import InvalidLaunchToolError
from chartreux.core.system_prompt import (
    _get_available_skills_section,
    get_universal_system_prompt,
)
from chartreux.core.tools.base import InvokeContext, ToolError
from chartreux.core.tools.builtins import edit as edit_module
from chartreux.core.tools.builtins.check_agents import CheckAgents, CheckAgentsArgs
from chartreux.core.tools.builtins.edit import EditArgs
from chartreux.core.tools.builtins.task import Task, TaskArgs
from chartreux.core.tools.classification import TOOL_CLASSIFICATIONS
from chartreux.core.tools.manager import (
    CHILD_CONTEXT_UNAVAILABLE_TOOLS,
    NoSuchToolError,
    ToolManager,
    child_withheld_tools,
)
from chartreux.core.tools.mcp import MCPRegistry, MCPToolResult, RemoteTool
from chartreux.core.tools.mcp.registry import MCPServerGate
from chartreux.core.tools.mcp.tools import _OpenArgs
from chartreux.core.tools.models import ToolPermission
from chartreux.core.tools.remote import MCPTool
from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

# The loop-level restriction: children cannot prompt, cannot spawn, and own no
# run inventory (recovery is owner-only).
LOOP_LEVEL_NAMES = frozenset({
    "ask_user_question",
    "cancel_agent",
    "check_agents",
    "get_agent_result",
    "reconcile_result",
    "release_agent",
    "resume_agent",
    "task",
    "wait_for_agent",
})

# The reviewer's personal posture is NON_MUTATING with a bash-only carve-out,
# so these mutating builtins are outside its reach (they stay invocable for
# the coder/orchestrator/researcher/advisor carve-out set).
REVIEWER_EXTRA_WITHHELD = frozenset({"bash_start", "edit", "write_file"})

# Per-identity default withheld sets (the enumeration table above).
EXPECTED_DEFAULT_WITHHELD: dict[str, frozenset[str]] = {
    "coder": LOOP_LEVEL_NAMES,
    "orchestrator": LOOP_LEVEL_NAMES,
    "researcher": LOOP_LEVEL_NAMES,
    "advisor": LOOP_LEVEL_NAMES,
    "reviewer": LOOP_LEVEL_NAMES | REVIEWER_EXTRA_WITHHELD,
}


def _config(**overrides: Any) -> ChartreuxConfigSchema:
    """A one-model vision config, so the full shipped catalog is available."""
    models = [
        ModelConfig(
            name="vision", provider="mistral", alias="vision", supports_images=True
        )
    ]
    config = build_test_vibe_config(
        active_model="vision",
        models=models,
        providers=[
            ProviderConfig(
                name="mistral",
                api_base="https://api.mistral.ai/v1",
                api_key_env_var="MISTRAL_API_KEY",
                backend=Backend.MISTRAL,
            )
        ],
        **overrides,
    )
    resolved = resolver_for(config).resolve(
        "vision", allowed_models=config.allowed_models
    )
    return config.attach_committed_model(resolved.identity)


def _manager(
    tmp_path: Path,
    config_getter: Callable[[], ChartreuxConfigSchema],
    identity: str,
    parent: ToolManager | None = None,
) -> ToolManager:
    return ToolManager(
        config_getter,
        cwd=tmp_path,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
        bound_identity=identity,
        scratchpad_dir=tmp_path,
        # Ship the reviewed builtins only: an isolated harness keeps the
        # environment's own user tools out of the assertions.
        harness_files=HarnessFilesManager(sources=(), cwd=tmp_path),
        parent_authority_getter=(lambda: parent) if parent is not None else None,
        parent_authority_revision_getter=((lambda: 0) if parent is not None else None),
    )


def _spec_names(manager: ToolManager) -> set[str]:
    return {spec.name for spec in manager.available_tool_specs()}


def _identity(name: str) -> AgentIdentity:
    identity = get_builtin_identity(name)
    assert identity is not None
    return identity


class TestLoopLevelChildRestriction:
    """The structural, authority-independent part of the withheld rule."""

    def test_the_set_is_exactly_the_structurally_denied_tools(self) -> None:
        assert CHILD_CONTEXT_UNAVAILABLE_TOOLS == LOOP_LEVEL_NAMES
        # Every name is a reviewed builtin, so the set can never drift into
        # unnamed territory without a classification-table change.
        assert LOOP_LEVEL_NAMES <= set(TOOL_CLASSIFICATIONS)

    @pytest.mark.asyncio
    @pytest.mark.timeout(24)
    async def test_a_child_cannot_spawn_or_reach_the_run_inventory(
        self, tmp_path: Path
    ) -> None:
        child = _manager(tmp_path, lambda: _config(), "coder")
        task = cast(Task, child.get("task"))
        nested = InvokeContext(
            tool_call_id="nested", is_subagent=True, agent_manager=cast(Any, object())
        )
        with pytest.raises(ToolError, match="depth limit|cannot be spawned"):
            await collect_result(task.run(TaskArgs(task="nested"), nested))

        # The owner-side inventory needs a subagent manager, and a child's
        # invoke context carries None — recovery never descends to a child.
        inventory = cast(CheckAgents, child.get("check_agents"))
        with pytest.raises(ToolError, match="subagent manager"):
            await collect_result(
                inventory.run(CheckAgentsArgs(), InvokeContext(tool_call_id="check"))
            )


class TestDefaultWithheldSets:
    """The per-identity default computation (default deployment)."""

    def test_the_table_covers_every_shipped_identity(self) -> None:
        assert set(EXPECTED_DEFAULT_WITHHELD) == set(builtin_identity_names())

    @pytest.mark.parametrize("identity_name", sorted(EXPECTED_DEFAULT_WITHHELD))
    def test_per_identity_default_withheld_set(self, identity_name: str) -> None:
        withheld = child_withheld_tools(_identity(identity_name))
        assert withheld == EXPECTED_DEFAULT_WITHHELD[identity_name]

    def test_custom_identity_withholds_only_the_loop_level_set(self) -> None:
        # Custom identities carry no harness-known posture (both axes
        # unrestricted) and are defined outside _IDENTITIES; only the
        # loop-level restriction can apply to them.
        assert child_withheld_tools(None) == LOOP_LEVEL_NAMES

    def test_ordinary_coder_child_withholds_task_and_the_recovery_set(
        self, tmp_path: Path
    ) -> None:
        parent = _manager(tmp_path, lambda: _config(), "coder")
        child = _manager(tmp_path, lambda: _config(), "coder", parent)

        assert child.child_catalog_withholdings() == LOOP_LEVEL_NAMES
        advertised = _spec_names(child)
        assert advertised == set(TOOL_CLASSIFICATIONS) - LOOP_LEVEL_NAMES
        # The gate's names, explicitly: the spawn tool and every recovery
        # tool are gone from the child's advertised catalog.
        assert {"task"} | (LOOP_LEVEL_NAMES - {"ask_user_question"}) <= (
            set(TOOL_CLASSIFICATIONS) - advertised
        )

    def test_reviewer_child_also_withholds_its_unreachable_mutating_set(
        self, tmp_path: Path
    ) -> None:
        parent = _manager(tmp_path, lambda: _config(), "coder")
        child = _manager(tmp_path, lambda: _config(), "reviewer", parent)

        expected = LOOP_LEVEL_NAMES | REVIEWER_EXTRA_WITHHELD
        assert child.child_catalog_withholdings() == expected
        advertised = _spec_names(child)
        assert advertised.isdisjoint(REVIEWER_EXTRA_WITHHELD)
        assert "bash" in advertised  # the reviewer's own carve-out survives

    def test_root_catalog_withholds_nothing(self, tmp_path: Path) -> None:
        root = _manager(tmp_path, lambda: _config(), "coder")
        assert root.child_catalog_withholdings() == frozenset()
        assert _spec_names(root) == set(root.registered_tools)

    @pytest.mark.timeout(24)
    def test_every_unlisted_tool_is_invocation_possible(self, tmp_path: Path) -> None:
        """AFFIRMATION: not-withheld ⇒ invocable under the default deployment."""
        parent = _manager(tmp_path, lambda: _config(), "coder")
        child = _manager(tmp_path, lambda: _config(), "coder", parent)
        withheld = child.child_catalog_withholdings()

        for name in set(TOOL_CLASSIFICATIONS) - withheld:
            assert name in child.available_tools, name
            assert child.get(name) is not None, name

    def test_advertised_and_withheld_partition_the_registry(
        self, tmp_path: Path
    ) -> None:
        parent = _manager(tmp_path, lambda: _config(), "coder")
        child = _manager(tmp_path, lambda: _config(), "reviewer", parent)
        withheld = child.child_catalog_withholdings()
        advertised = _spec_names(child)

        assert advertised.isdisjoint(withheld)
        assert advertised | withheld == set(child.registered_tools)


class TestForcedCallOnAWithheldDefinition:
    """A withheld definition is an advertisement cut, never a resolution cut."""

    @pytest.mark.timeout(24)
    def test_a_withheld_name_still_resolves_to_its_tool(self, tmp_path: Path) -> None:
        parent = _manager(tmp_path, lambda: _config(), "coder")
        child = _manager(tmp_path, lambda: _config(), "coder", parent)

        for name in LOOP_LEVEL_NAMES:
            assert name not in _spec_names(child)
            assert child.get(name) is not None, name

    def test_a_ceiling_gated_name_fails_at_resolution(self, tmp_path: Path) -> None:
        # Outside both the reviewer's personal axis and its posture's
        # carve-out: the invocation path denies before any tool body runs,
        # surfacing as the executor's single-turn "error getting tool" event.
        parent = _manager(tmp_path, lambda: _config(), "coder")
        child = _manager(tmp_path, lambda: _config(), "reviewer", parent)

        assert "edit" not in _spec_names(child)
        with pytest.raises(NoSuchToolError):
            child.get("edit")


class TestDynamicAuthorityRecomputation:
    """Withholding is recomputed at EVERY spec assembly, from current authority."""

    def test_delegation_grant_unwithholds_at_the_next_spec_assembly(
        self, tmp_path: Path
    ) -> None:
        config = _config()
        parent = _manager(tmp_path, lambda: config, "researcher")
        child = _manager(tmp_path, lambda: config, "coder", parent)

        assert "edit" not in _spec_names(child)
        # The launching parent's delegation set broadens in place (the loop's
        # own identity rebind path); the child needs no rebuild.
        parent._bind_identity("coder")
        assert "edit" in _spec_names(child)
        assert "edit" not in child.child_catalog_withholdings()

    def test_parent_revocation_drops_the_tool_at_the_next_assembly(
        self, tmp_path: Path
    ) -> None:
        revoked = _config(disabled_tools=["grep"])
        holder: dict[str, ChartreuxConfigSchema] = {"config": revoked}
        parent = _manager(tmp_path, lambda: holder["config"], "coder")
        child = _manager(tmp_path, lambda: holder["config"], "coder", parent)

        assert "grep" not in _spec_names(child)
        assert "grep" in child.child_catalog_withholdings()

        holder["config"] = _config()
        assert "grep" in _spec_names(child)
        assert "grep" not in child.child_catalog_withholdings()

        holder["config"] = revoked
        assert "grep" not in _spec_names(child)


class TestLaunchAdmissionExecutionCeiling:
    """Admission rejects overrides that would re-enable a child-denied tool."""

    def test_enabling_a_child_context_unavailable_tool_is_rejected(self) -> None:
        with pytest.raises(
            InvalidLaunchToolError, match="unavailable to bounded children"
        ):
            _validate_tools(
                LaunchConfig(enabled_tools=["task"]),
                frozenset({"task"}),
                frozenset({"task"}),
                _identity("coder"),
            )

    def test_granting_permission_on_a_child_context_unavailable_tool_is_rejected(
        self,
    ) -> None:
        with pytest.raises(
            InvalidLaunchToolError, match="unavailable to bounded children"
        ):
            _validate_tools(
                LaunchConfig(
                    tools={
                        "check_agents": LaunchToolOverride(
                            permission=ToolPermission.ALWAYS
                        )
                    }
                ),
                frozenset({"check_agents"}),
                frozenset({"check_agents"}),
                _identity("coder"),
            )

    def test_narrowing_overrides_stay_allowed_on_the_same_tools(self) -> None:
        # Monitoring-only tools deny at the decision, and a deny-list entry
        # only ever narrows — both remain legal below the ceiling.
        _validate_tools(
            LaunchConfig(
                tools={"task": LaunchToolOverride(permission=ToolPermission.NEVER)}
            ),
            frozenset({"task"}),
            frozenset({"task"}),
            _identity("coder"),
        )
        _validate_tools(
            LaunchConfig(disabled_tools=["task", "wait_for_agent"]),
            frozenset({"task", "wait_for_agent"}),
            frozenset({"task", "wait_for_agent"}),
            _identity("coder"),
        )

    def test_an_admitted_tool_still_passes_admission(self) -> None:
        _validate_tools(
            LaunchConfig(enabled_tools=["grep", "read_file"]),
            frozenset({"grep", "read_file"}),
            frozenset({"grep", "read_file"}),
            _identity("coder"),
        )


# --- Slice 2: MCP per-server gating, compression, consistency (P1b) --------

MCP_SERVER = "playwright"
MCP_REMOTE = "navigate"
MCP_TOOL = f"{MCP_SERVER}_{MCP_REMOTE}"


def _mcp_server(name: str = MCP_SERVER) -> MCPHttp:
    # Loopback only: discovery and calls are mocked at the transport boundary,
    # so no test here can reach a live MCP provider.
    return MCPHttp(
        name=name, transport="streamable-http", url=f"http://127.0.0.1:9/{name}"
    )


def _discovery(remote_name: str = MCP_REMOTE) -> Any:
    return patch(
        "chartreux.core.tools.mcp.registry.list_tools_http",
        return_value=[RemoteTool(name=remote_name, description="Open a page")],
    )


def _mcp_config(*server_names: str) -> ChartreuxConfigSchema:
    return _config(mcp_servers=[_mcp_server(name) for name in server_names])


def _mcp_manager(
    tmp_path: Path,
    config_getter: Callable[[], ChartreuxConfigSchema],
    identity: str,
    *,
    registry: MCPRegistry,
    parent: ToolManager | None = None,
) -> ToolManager:
    """A manager that integrates MCP through ``registry`` at construction."""
    return ToolManager(
        config_getter,
        mcp_registry=registry,
        cwd=tmp_path,
        defer_mcp=False,
        accepted_token_getter=lambda: "accepted",
        bound_identity=identity,
        scratchpad_dir=tmp_path,
        harness_files=HarnessFilesManager(sources=(), cwd=tmp_path),
        parent_authority_getter=(lambda: parent) if parent is not None else None,
        parent_authority_revision_getter=((lambda: 0) if parent is not None else None),
    )


class TestMCPServerGatingPrimitive:
    """The per-SERVER admission primitive (MCP registers flat; no group)."""

    def test_a_root_registry_admits_every_configured_server(self) -> None:
        root = MCPRegistry()

        assert root.server_gate is None
        assert root.admits_server(MCP_SERVER)
        assert root.admits_server("any-other-server")

    def test_a_bounded_child_view_admits_no_server_by_default(self) -> None:
        child = MCPRegistry().for_bounded_child()

        assert child.server_gate == MCPServerGate()
        assert not child.admits_server(MCP_SERVER)
        with _discovery() as discover:
            assert child.get_tools([_mcp_server()]) == {}
        # A gated-out server is never discovered: its proxy classes cannot
        # register anywhere, so "no definition" and "no usability" coincide.
        discover.assert_not_called()

    def test_an_opt_in_admits_exactly_the_named_server(self) -> None:
        child = MCPRegistry().for_bounded_child({MCP_SERVER})

        assert child.admits_server(MCP_SERVER)
        assert not child.admits_server("files")
        with _discovery():
            assert set(child.get_tools([_mcp_server()])) == {MCP_TOOL}

    def test_an_opt_in_cannot_exceed_the_parent_gate(self) -> None:
        parent = MCPRegistry()
        parent.configure_server_gate(MCPServerGate(frozenset({MCP_SERVER})))

        child = parent.for_bounded_child({MCP_SERVER, "files"})

        assert child.server_gate is not None
        assert child.server_gate.opted_in == frozenset({MCP_SERVER})
        # Clearing a child's gate falls back to the parent's admission, never
        # to "everything configured".
        child.configure_server_gate(None)
        assert child.server_gate == MCPServerGate(frozenset({MCP_SERVER}))

    def test_a_gate_change_applies_at_the_next_discovery(self) -> None:
        child = MCPRegistry().for_bounded_child({MCP_SERVER})
        server = _mcp_server()
        with _discovery():
            handed_out = child.get_tools([server])
        assert set(handed_out) == {MCP_TOOL}

        child.configure_server_gate(MCPServerGate())

        with _discovery() as discover:
            assert child.get_tools([server]) == {}
        discover.assert_not_called()
        # The class already handed to a manager keeps working: gating is an
        # admission decision at the next catalog assembly, never a revocation
        # of a legitimate in-flight call.
        assert cast(type[MCPTool], handed_out[MCP_TOOL]).get_server_name() == MCP_SERVER

    def test_the_server_name_source_is_the_proxy_class_alias(self) -> None:
        # The per-server key is the alias the proxy class carries; the frozen
        # definition site is MCPTool.get_server_name (core/tools/remote.py).
        with _discovery():
            tools = MCPRegistry().get_tools([_mcp_server()])

        assert set(tools) == {MCP_TOOL}
        assert cast(type[MCPTool], tools[MCP_TOOL]).get_server_name() == MCP_SERVER


class TestMCPServerGatingBehavioural:
    """Browser-opt-in child vs non-opted child, end to end through a manager."""

    @pytest.mark.timeout(24)
    def test_an_opted_in_child_keeps_the_group_definition(self, tmp_path: Path) -> None:
        config = _mcp_config(MCP_SERVER)
        root = MCPRegistry()
        with _discovery():
            parent = _mcp_manager(tmp_path, lambda: config, "coder", registry=root)
            child = _mcp_manager(
                tmp_path,
                lambda: config,
                "coder",
                registry=root.for_bounded_child({MCP_SERVER}),
                parent=parent,
            )

        # The parent keeps its MCP catalog and delegates it downward.
        assert MCP_TOOL in _spec_names(parent)
        # The opt-in child keeps the definition AND the resolution path.
        assert MCP_TOOL in _spec_names(child)
        assert MCP_TOOL not in child.child_catalog_withholdings()
        tool = child.get(MCP_TOOL)
        assert tool is not None
        assert cast(MCPTool, tool).get_server_name() == MCP_SERVER

    @pytest.mark.asyncio
    @pytest.mark.timeout(24)
    async def test_an_opted_in_child_can_call_the_group_tool(
        self, tmp_path: Path
    ) -> None:
        config = _mcp_config(MCP_SERVER)
        root = MCPRegistry()
        with _discovery():
            parent = _mcp_manager(tmp_path, lambda: config, "coder", registry=root)
            child = _mcp_manager(
                tmp_path,
                lambda: config,
                "coder",
                registry=root.for_bounded_child({MCP_SERVER}),
                parent=parent,
            )
        tool = child.get(MCP_TOOL)
        assert tool is not None

        answered = MCPToolResult(
            ok=True, server=MCP_SERVER, tool=MCP_REMOTE, text="opened"
        )
        with patch(
            "chartreux.core.tools.mcp.tools.call_tool_http",
            new=AsyncMock(return_value=answered),
        ):
            result = await collect_result(
                tool.run(_OpenArgs(), InvokeContext(tool_call_id="mcp-call"))
            )

        assert result.ok is True
        assert result.text == "opened"

    @pytest.mark.timeout(24)
    def test_a_non_opted_child_has_neither_definition_nor_usability(
        self, tmp_path: Path
    ) -> None:
        config = _mcp_config(MCP_SERVER)
        root = MCPRegistry()
        with _discovery():
            parent = _mcp_manager(tmp_path, lambda: config, "coder", registry=root)
            child = _mcp_manager(
                tmp_path,
                lambda: config,
                "coder",
                registry=root.for_bounded_child(),
                parent=parent,
            )

        # Discovery succeeded for the parent: the server and its tool exist, so
        # the child's absence is the gate's doing, not a failed connection.
        assert MCP_TOOL in _spec_names(parent)
        assert MCP_TOOL not in _spec_names(child)
        assert MCP_TOOL not in child.available_tools
        with pytest.raises(NoSuchToolError):
            child.get(MCP_TOOL)


class TestSkillDiscoveryPreserved:
    """Compression never costs the child its skill surface."""

    @pytest.mark.timeout(24)
    def test_a_child_catalog_still_advertises_the_skill_tool(
        self, tmp_path: Path
    ) -> None:
        parent = _manager(tmp_path, lambda: _config(), "coder")
        child = _manager(tmp_path, lambda: _config(), "coder", parent)

        assert "skill" in _spec_names(child)
        assert "skill" not in child.child_catalog_withholdings()
        assert child.get("skill") is not None

    @pytest.mark.timeout(24)
    def test_a_child_prompt_still_lists_the_available_skills(self) -> None:
        config = _config(
            agent={"identity": "coder"},
            include_prompt_detail=True,
            include_project_context=False,
        )
        prompt = _child_prompt(config)

        assert "# Available Skills" in prompt
        assert "use the `skill` tool to load the full skill instructions." in prompt
        assert "`/skill-name`" in prompt
        assert "you do not need to call the `skill` tool yourself." in prompt


def _child_prompt(config: ChartreuxConfigSchema) -> str:
    return str(
        get_universal_system_prompt(
            config,
            SkillManager(lambda: config),
            AgentManager(FakeConfigOrchestrator(config)),
            is_subagent=True,
        )
    )


def _main_prompt(config: ChartreuxConfigSchema) -> str:
    return str(
        get_universal_system_prompt(
            config,
            SkillManager(lambda: config),
            AgentManager(FakeConfigOrchestrator(config)),
        )
    )


_EDIT_PROMPT_PATH = Path(edit_module.__file__).parent / "prompts" / "edit.md"

# What the edit prompt's backticked identifiers name BESIDES its arguments: the
# tool it tells the caller to read with, the result fields it reports, and the
# literal boolean of a mode clause. Naming anything else in backticks would be
# a drift between the description and the schema.
_EDIT_PROMPT_NON_ARGUMENT_IDENTIFIERS = frozenset({
    "applied_diff",
    "context",
    "read_file",
    "replaced_count",
    "revision",
    "true",
})


def _edit_prompt_text() -> str:
    """The landed edit description, read from prompts/edit.md."""
    return _EDIT_PROMPT_PATH.read_text(encoding="utf-8")


def _backticked_identifiers(text: str) -> frozenset[str]:
    """Every identifier spelled inside backticks in one passage."""
    found: set[str] = set()
    for span in re.findall(r"`([^`]+)`", text):
        found.update(re.findall(r"[a-z_][a-z0-9_]*", span))
    return frozenset(found)


class TestPromptTextVsDefinitionConsistency:
    """Binary gate: no child passage teaches a tool its catalog withholds."""

    @pytest.mark.parametrize("identity", ["coder", "researcher", "reviewer", "advisor"])
    @pytest.mark.timeout(24)
    def test_no_child_prompt_passage_teaches_a_withheld_tool(
        self, identity: str
    ) -> None:
        config = _config(
            agent={"identity": identity},
            include_prompt_detail=True,
            include_project_context=False,
        )
        prompt = _child_prompt(config)
        withheld = child_withheld_tools(_identity(identity))

        assert taught_tool_names(prompt) & withheld == frozenset()
        # The shared reference is filtered at COMPOSITION, not by editing the
        # source document: a main context still reads the passage it can act
        # on, and the fragment that teaches the child holds the replacement.
        assert "**Asking the user.**" not in prompt
        assert "**Asking the user.**" in _main_prompt(
            _config(
                agent={"identity": identity},
                include_prompt_detail=True,
                include_project_context=False,
            )
        )
        assert "Do not ask the user questions" in prompt
        # The tools the child CAN invoke stay taught: the filter is exact.
        assert "use the `skill` tool" in prompt

    @pytest.mark.timeout(24)
    def test_the_coverage_holds_for_the_composition_read_site(self) -> None:
        # system_prompt.py's execution-context branch renders the bounded-child
        # fragment, and the consistency rule holds for the text it produces.
        config = _config(
            agent={"identity": "coder"},
            include_prompt_detail=True,
            include_project_context=False,
        )
        prompt = _child_prompt(config)

        assert "## Execution context — bounded child" in prompt
        assert "# Workflow:" not in prompt
        assert (
            taught_tool_names(prompt) & child_withheld_tools(_identity("coder"))
            == frozenset()
        )

    @pytest.mark.timeout(24)
    def test_the_coverage_holds_for_the_launch_assembly_read_site(self) -> None:
        # identity_launch.py's bounded-child assembly fixes the launch's
        # execution context and identity facts; both feed the check below.
        config = _config(
            agent={"identity": "coder"},
            include_prompt_detail=True,
            include_project_context=False,
        )
        resolved = resolve_identity_launch(
            {"task": "probe", "identity": "coder", "model": "vision"},
            config=config,
            authorized_tool_names=frozenset({"bash", "read_file"}),
        )

        assert resolved.execution_context is ExecutionContext.BOUNDED_CHILD
        withheld = child_withheld_tools(resolved.identity_facts)
        assert withheld == child_withheld_tools(_identity("coder"))
        prompt = _child_prompt(config)
        assert taught_tool_names(prompt) & withheld == frozenset()

    @pytest.mark.timeout(24)
    def test_the_edit_description_matches_its_landed_schema(self) -> None:
        """P6c drift gate: the description teaches exactly the landed modes.

        The edit tool's definition IS its ``prompts/edit.md`` description, so a
        description that names a parameter the schema dropped (or omits one it
        grew) advertises a mode the executor rejects. Both directions are
        checked against the landed schema, so the P6c rewrite and the argument
        model can never drift apart.
        """
        prompt = _edit_prompt_text()
        schema_names = frozenset(EditArgs.model_fields)

        assert _EDIT_PROMPT_PATH.name == "edit.md"
        # The definition served to the model is this file, byte for byte.
        assert edit_module.Edit.get_tool_prompt() == prompt

        taught_arguments = (
            _backticked_identifiers(prompt) - _EDIT_PROMPT_NON_ARGUMENT_IDENTIFIERS
        )
        assert taught_arguments == schema_names
        # The P6c modes are the ones the description must keep teaching.
        assert {"multi_anchor", "expected_site_count"} <= taught_arguments


# The pre-compression fragment shipped at the P1a checkpoint: 11,087 bytes
# (`git show 041e6b6e:chartreux/core/prompts/fragments/bounded_child.md`).
_FRAGMENT_BASELINE_BYTES = 11_087

# The pure-duplication clauses the compression removed, each paired with the
# statement that already carries its obligation in the compressed fragment.
_REMOVED_DUPLICATES: tuple[tuple[str, str], ...] = (
    (
        "Your parent owns the workflow; you execute one bounded assignment inside it.",
        "**The parent owns the workflow.**",
    ),
    (
        "Before changing a file, re-read it when its current state matters; "
        "retained history is context, not proof that a file is unchanged.",
        "Re-read a file when its current state matters;",
    ),
    (
        "Do not claim checks that you did not perform.",
        "never claim a check you did not run, and never conceal a failed one",
    ),
    (
        "do not pretend success or conceal failed checks",
        "never claim a check you did not run, and never conceal a failed one",
    ),
    (
        "Wrapping the action to evade validation is not adaptation.",
        "producing the same refused effect through a different tool, script, or "
        "mechanism is still a retry",
    ),
)


class TestBoundedChildFragmentCompression:
    """10-25% redundancy removed, discovery and compliance text preserved."""

    @pytest.mark.timeout(24)
    def test_the_compression_lands_in_band_and_reports_its_savings(
        self, record_property: Callable[..., None]
    ) -> None:
        current = (FRAGMENTS_DIR / "bounded_child.md").read_bytes()
        savings = 1 - len(current) / _FRAGMENT_BASELINE_BYTES
        record_property("bounded_child_fragment_savings", round(savings, 4))
        print(
            f"bounded_child fragment compression: {_FRAGMENT_BASELINE_BYTES} -> "
            f"{len(current)} bytes, {savings:.2%} removed"
        )

        assert 0.10 <= savings <= 0.25

    def test_the_removed_clauses_are_pure_duplicates(self) -> None:
        text = (FRAGMENTS_DIR / "bounded_child.md").read_text(encoding="utf-8")

        for removed, carrier in _REMOVED_DUPLICATES:
            assert removed not in text
            # The obligation survives; only its second spelling is gone.
            assert carrier in text

    def test_every_compliance_statement_survives(self) -> None:
        text = (FRAGMENTS_DIR / "bounded_child.md").read_text(encoding="utf-8")

        for clause in (
            "1. Denials:",
            "2. Strategy switch:",
            "3. Destinations:",
            "4. Provider isolation:",
            "5. Edited paths:",
            "6. Scope reconciliation:",
            "7. Evidence-linked work loop:",
            "8. Receipts and tool-result trust:",
            "9. No-edit proof obligation:",
            "10. File-state metadata is not authority:",
            "11. Completion citations:",
            "12. Completion envelope emission:",
            "never create an unconditional commit mandate for exempt or "
            "read-only assignments",
            "completion_envelope_v1",
            "the disclosure requirement applies to every output form.",
            '"files_touched": ["the cumulative record of every repository path',
        ):
            assert clause in text

    def test_every_discovery_statement_survives(self) -> None:
        text = (FRAGMENTS_DIR / "bounded_child.md").read_text(encoding="utf-8")

        for clause in (
            "You are a bounded child agent",
            "non-interactive",
            "return a structured blocker",
            "Never restart the parent's workflow",
            "re-invoke its ceremony",
            "Do not spawn, delegate to, or coordinate nested subagents.",
            "## Child operating rules",
            "## Output format",
        ):
            assert clause in text


# The skills-listing preamble as it shipped before this chunk (P1b baseline).
_SKILLS_PREAMBLE_BASELINE = "\n".join([
    "You have access to the following skills. When a task matches a skill's description,",
    "use the `skill` tool to load the full skill instructions.",
    "",
    "When a user message is exactly `/skill-name` (optionally followed by extra",
    "instructions), the user has explicitly invoked that skill. Its instructions are",
    "loaded for you automatically: you will see a `skill` tool call and result",
    "immediately after that message. Treat the loaded content as the active",
    "instructions and act on it — you do not need to call the `skill` tool yourself.",
])


class _StubSkills:
    available_skills: ClassVar[dict[str, Any]] = {
        "demo": SimpleNamespace(
            name="demo", description="A demo skill", skill_path=None
        )
    }


def _skills_section() -> str:
    return _get_available_skills_section(cast(Any, _StubSkills()))


def _skills_preamble(section: str) -> str:
    body = section.split("\n\n", 1)[1]
    return body.split("\n\n<available_skills>", 1)[0]


class TestSkillsListingCompression:
    """Same band, same constraint: discovery + compliance text preserved."""

    @pytest.mark.timeout(24)
    def test_the_compression_lands_in_band_and_reports_its_savings(
        self, record_property: Callable[..., None]
    ) -> None:
        baseline = _SKILLS_PREAMBLE_BASELINE.encode()
        current = _skills_preamble(_skills_section()).encode()
        savings = 1 - len(current) / len(baseline)
        record_property("skills_preamble_savings", round(savings, 4))
        print(
            f"skills-listing preamble compression: {len(baseline)} -> "
            f"{len(current)} bytes, {savings:.2%} removed"
        )

        assert 0.10 <= savings <= 0.25

    def test_the_discovery_and_compliance_text_survives(self) -> None:
        preamble = _skills_preamble(_skills_section())

        assert "use the `skill` tool to load the full skill instructions." in preamble
        assert "`/skill-name`" in preamble
        assert "explicitly" in preamble
        assert "you do not need to call the `skill` tool yourself." in preamble
        # The retired manual-read path stays retired.
        assert "read the files manually" not in preamble

    def test_the_rendered_entry_loop_is_untouched(self) -> None:
        # P9's rendered-entry budget serializes through this landed code, so
        # the per-skill rendering keeps its exact shape.
        section = _skills_section()

        assert "<available_skills>" in section
        assert "  <skill>" in section
        assert "    <name>demo</name>" in section
        assert "    <description>A demo skill</description>" in section
        assert "<path>" not in section


# --- Slice 3: fixture-class gates + the pinned saving gate (P1c) -----------

# THE PINNED SAVING GATE (plan 1c, R2-GLM-4, re-based with user authorization
# 2026-11-08): the task tool's serialized definition delta between the
# ordinary-coder child's withheld assembly and its default-authority
# un-withheld assembly, both computed in the landed build. The landed build
# measures 23,653 bytes (5,913 tokens at the release's bytes/4 rate); the
# original 24,000-byte (6k) pin was re-based to 23,600 bytes because the user
# accepted the 347-byte shortfall as immaterial. The measured value is still
# reported honestly by the test output.
TASK_TOOL_DEF_DELTA_GATE_BYTES = 23_600

# The enumerated recovery set: the owner-only run inventory. The loop-level
# restriction minus the user-question tool (children answer with a blocker
# instead) and the spawn tool (covered by the task-definition gate).
RECOVERY_SET = LOOP_LEVEL_NAMES - {"ask_user_question", "task"}


def _wire_tools(manager: ToolManager) -> list[AvailableTool]:
    """The landed request assembly: the definitions the provider is offered."""
    return APIToolFormatHandler().get_available_tools(manager)


def _wire_blob(manager: ToolManager) -> str:
    return _serialized_tool_definitions(_wire_tools(manager))


def _definition_bytes(tool: AvailableTool) -> int:
    """One definition's bytes, in the same serialized form as the telemetry."""
    return len(
        json.dumps(
            tool.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _literal_definition(blob: str, name: str) -> bool:
    """Whether the serialized request literally carries this definition."""
    return f'"name":"{name}"' in blob


@contextmanager
def _definitions_unwithheld(manager: ToolManager) -> Iterator[ToolManager]:
    """The same fixture assembled with the withholding decision disabled.

    This is the DEFAULT-AUTHORITY un-withheld assembly: the identical catalog
    the same authority admits, minus the advertisement cut. Neither side is a
    different fixture, so the byte delta is the withholding decision alone.
    """
    with patch.object(ToolManager, "_child_catalog_withholds", return_value=False):
        yield manager


class _Savings(NamedTuple):
    """One fixture class's catalog, its authority assembly, and their bytes."""

    catalog: dict[str, int]
    authority: dict[str, int]
    catalog_blob_bytes: int
    authority_blob_bytes: int

    @property
    def withheld_names(self) -> frozenset[str]:
        return frozenset(self.authority) - frozenset(self.catalog)

    @property
    def delta_bytes(self) -> int:
        return self.authority_blob_bytes - self.catalog_blob_bytes

    def table(self) -> dict[str, int]:
        """Per-tool saving: the bytes each withheld definition no longer costs."""
        return {name: self.authority[name] for name in sorted(self.withheld_names)}

    def total(self) -> int:
        """The table's total: the whole-set delta, tool by tool."""
        return sum(self.table().values())


def _savings(manager: ToolManager) -> _Savings:
    catalog_tools = _wire_tools(manager)
    with _definitions_unwithheld(manager):
        authority_tools = _wire_tools(manager)
    return _Savings(
        catalog={tool.function.name: _definition_bytes(tool) for tool in catalog_tools},
        authority={
            tool.function.name: _definition_bytes(tool) for tool in authority_tools
        },
        catalog_blob_bytes=len(
            _serialized_tool_definitions(catalog_tools).encode("utf-8")
        ),
        authority_blob_bytes=len(
            _serialized_tool_definitions(authority_tools).encode("utf-8")
        ),
    )


def _ordinary_coder_child(tmp_path: Path) -> ToolManager:
    """Fixture class 1: the ordinary coder child under default authority."""
    config = _config()
    parent = _manager(tmp_path, lambda: config, "coder")
    return _manager(tmp_path, lambda: config, "coder", parent)


def _browser_opt_in_child(tmp_path: Path) -> ToolManager:
    """Fixture class 2: the child that opted into one MCP server."""
    config = _mcp_config(MCP_SERVER)
    registry = MCPRegistry()
    with _discovery():
        parent = _mcp_manager(tmp_path, lambda: config, "coder", registry=registry)
        return _mcp_manager(
            tmp_path,
            lambda: config,
            "coder",
            registry=registry.for_bounded_child({MCP_SERVER}),
            parent=parent,
        )


def _small_catalog_child(tmp_path: Path) -> ToolManager:
    """Fixture class 3: a child whose parent delegation admits few tools."""
    config = _config()
    parent = _manager(tmp_path, lambda: config, "reviewer")
    return _manager(tmp_path, lambda: config, "coder", parent)


class TestFixtureClassDefinitionPresence:
    """Literal request-definition presence, one assertion set per class."""

    @pytest.mark.timeout(24)
    def test_ordinary_coder_child_carries_no_task_or_recovery_definition(
        self, tmp_path: Path
    ) -> None:
        child = _ordinary_coder_child(tmp_path)
        blob = _wire_blob(child)

        assert not _literal_definition(blob, "task")
        for name in sorted(RECOVERY_SET):
            assert not _literal_definition(blob, name), name
        # The definitions the child can act on stay there, literally.
        for name in ("bash", "read_file", "grep", "edit", "write_file", "skill"):
            assert _literal_definition(blob, name), name
        assert {tool.function.name for tool in _wire_tools(child)} == set(
            TOOL_CLASSIFICATIONS
        ) - LOOP_LEVEL_NAMES

    @pytest.mark.timeout(24)
    def test_browser_opt_in_child_carries_the_mcp_definition_and_no_task(
        self, tmp_path: Path
    ) -> None:
        child = _browser_opt_in_child(tmp_path)
        blob = _wire_blob(child)

        assert _literal_definition(blob, MCP_TOOL)
        assert not _literal_definition(blob, "task")
        for name in sorted(RECOVERY_SET):
            assert not _literal_definition(blob, name), name

    @pytest.mark.timeout(24)
    def test_small_catalog_child_carries_only_its_small_catalog(
        self, tmp_path: Path
    ) -> None:
        child = _small_catalog_child(tmp_path)
        savings = _savings(child)
        blob = _wire_blob(child)

        assert set(savings.catalog) == _spec_names(child)
        # Fewer than half the reviewed universe: this class really is small,
        # and it still withholds the whole loop-level core.
        assert len(savings.catalog) * 2 <= len(TOOL_CLASSIFICATIONS)
        assert savings.withheld_names == LOOP_LEVEL_NAMES
        for name in savings.catalog:
            assert _literal_definition(blob, name), name
        assert not _literal_definition(blob, "task")


class TestPerFixtureClassSavings:
    """The savings are reported per fixture class, not from one fixture."""

    @pytest.mark.timeout(60)
    def test_the_savings_table_covers_every_fixture_class(
        self, tmp_path: Path, record_property: Callable[..., None]
    ) -> None:
        rows: list[tuple[str, int, int, int]] = []
        for label, build in (
            ("ordinary-coder", _ordinary_coder_child),
            ("browser-opt-in", _browser_opt_in_child),
            ("small-catalog", _small_catalog_child),
        ):
            savings = _savings(build(tmp_path))
            # Every class withholds exactly the loop-level core.
            assert savings.withheld_names == LOOP_LEVEL_NAMES, label
            assert savings.delta_bytes > 0, label
            rows.append((
                label,
                savings.catalog_blob_bytes,
                savings.authority_blob_bytes,
                savings.delta_bytes,
            ))

        print("\nfixture class        catalog B   authority B   delta B   saved")
        for label, catalog_bytes, authority_bytes, delta_bytes in rows:
            print(
                f"{label:20s} {catalog_bytes:10d} {authority_bytes:12d} "
                f"{delta_bytes:8d} {delta_bytes / authority_bytes:7.1%}"
            )
        record_property(
            "fixture_class_savings_bytes", {label: delta for label, _, _, delta in rows}
        )


class TestPerToolSavingsTable:
    """Per-tool reporting, and the recovery set's own positive-saving gate."""

    @pytest.mark.timeout(24)
    def test_every_withheld_definition_reports_a_positive_saving(
        self, tmp_path: Path, record_property: Callable[..., None]
    ) -> None:
        savings = _savings(_ordinary_coder_child(tmp_path))
        table = savings.table()

        assert set(table) == LOOP_LEVEL_NAMES
        for name, saved in table.items():
            assert saved > 0, name

        print("\nper-tool savings (bytes no longer serialized)")
        for name, saved in table.items():
            print(f"  {name:20s} {saved:7d}")
        print(f"  {'whole set':20s} {savings.total():7d}")
        record_property("per_tool_savings_bytes", table)

    @pytest.mark.timeout(24)
    def test_each_enumerated_recovery_tool_saves_positively(
        self, tmp_path: Path
    ) -> None:
        table = _savings(_ordinary_coder_child(tmp_path)).table()

        assert RECOVERY_SET == LOOP_LEVEL_NAMES - {"ask_user_question", "task"}
        for name in sorted(RECOVERY_SET):
            assert table[name] > 0, name


class TestPinnedTaskDefinitionSavingGate:
    """The design's 6k gate, carried by the task definition alone."""

    @pytest.mark.timeout(24)
    def test_ordinary_coder_task_tooldef_delta(
        self, tmp_path: Path, record_property: Callable[..., None]
    ) -> None:
        savings = _savings(_ordinary_coder_child(tmp_path))
        delta = savings.table()["task"]
        record_property("task_tooldef_delta_bytes", delta)
        print(
            f"\ntask definition delta: {delta} bytes "
            f"({delta / 4:.0f} tokens at bytes/4), gate "
            f"{TASK_TOOL_DEF_DELTA_GATE_BYTES} bytes"
        )

        assert delta >= TASK_TOOL_DEF_DELTA_GATE_BYTES, (
            f"the task definition's serialized delta is {delta} bytes "
            f"({delta / 4:.0f} tokens at the release's bytes/4 rate): the pinned "
            f"{TASK_TOOL_DEF_DELTA_GATE_BYTES}-byte gate is not met by the landed "
            "build (task description + parameter schema, serialized with the "
            "release's compact request-shape form)"
        )

    @pytest.mark.timeout(24)
    def test_the_delta_is_the_task_definition_alone(self, tmp_path: Path) -> None:
        savings = _savings(_ordinary_coder_child(tmp_path))

        assert "task" not in savings.catalog
        assert savings.table()["task"] == savings.authority["task"]
        # The recovery set is real but small: it cannot carry this gate.
        assert (
            sum(savings.authority[name] for name in RECOVERY_SET)
            < savings.authority["task"]
        )


class TestForcedCallErrorPath:
    """A forced call on a withheld definition is a single-turn error."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(24)
    async def test_a_forced_task_call_resolves_and_hits_its_blocker(
        self, tmp_path: Path
    ) -> None:
        child = _ordinary_coder_child(tmp_path)
        parsed = ParsedMessage(
            tool_calls=[
                ParsedToolCall(
                    tool_name="task", raw_args={"task": "probe"}, call_id="forced"
                )
            ]
        )

        resolved = APIToolFormatHandler().resolve_tool_calls(parsed, child)

        # Not an unknown-tool failure: the withheld definition still resolves,
        # and the tool's own blocker ends the turn with an explicit error.
        assert resolved.failed_calls == []
        assert [call.tool_name for call in resolved.tool_calls] == ["task"]
        task = cast(Task, child.get("task"))
        with pytest.raises(ToolError, match="depth limit|cannot be spawned"):
            await collect_result(
                task.run(
                    TaskArgs(task="probe"),
                    InvokeContext(tool_call_id="forced", is_subagent=True),
                )
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(24)
    async def test_a_forced_recovery_call_resolves_and_hits_its_blocker(
        self, tmp_path: Path
    ) -> None:
        child = _ordinary_coder_child(tmp_path)
        parsed = ParsedMessage(
            tool_calls=[
                ParsedToolCall(tool_name="check_agents", raw_args={}, call_id="forced")
            ]
        )

        resolved = APIToolFormatHandler().resolve_tool_calls(parsed, child)

        assert resolved.failed_calls == []
        assert [call.tool_name for call in resolved.tool_calls] == ["check_agents"]
        inventory = cast(CheckAgents, child.get("check_agents"))
        with pytest.raises(ToolError, match="subagent manager"):
            await collect_result(
                inventory.run(CheckAgentsArgs(), InvokeContext(tool_call_id="forced"))
            )


class TestMidLaunchAuthorityRegrant:
    """Mid-launch authority moves the definitions at the next assembly."""

    @pytest.mark.timeout(24)
    def test_a_mid_launch_regrant_returns_the_definition(self, tmp_path: Path) -> None:
        config = _config()
        parent = _manager(tmp_path, lambda: config, "reviewer")
        child = _manager(tmp_path, lambda: config, "coder", parent)

        assert not _literal_definition(_wire_blob(child), "edit")
        # The parent's own identity rebinds in place; the child is untouched.
        parent._bind_identity("coder")
        assert _literal_definition(_wire_blob(child), "edit")
        assert "edit" not in child.child_catalog_withholdings()

    @pytest.mark.timeout(24)
    def test_parent_revocation_drops_the_definition(self, tmp_path: Path) -> None:
        revoked = _config(disabled_tools=["grep"])
        holder: dict[str, ChartreuxConfigSchema] = {"config": revoked}
        parent = _manager(tmp_path, lambda: holder["config"], "coder")
        child = _manager(tmp_path, lambda: holder["config"], "coder", parent)

        assert not _literal_definition(_wire_blob(child), "grep")
        holder["config"] = _config()
        assert _literal_definition(_wire_blob(child), "grep")
        holder["config"] = revoked
        assert not _literal_definition(_wire_blob(child), "grep")

    @pytest.mark.timeout(24)
    def test_the_loop_level_core_never_returns_under_a_full_grant(
        self, tmp_path: Path
    ) -> None:
        config = _config()
        parent = _manager(tmp_path, lambda: config, "reviewer")
        child = _manager(tmp_path, lambda: config, "coder", parent)

        parent._bind_identity("coder")
        blob = _wire_blob(child)

        assert _literal_definition(blob, "bash")
        for name in sorted(LOOP_LEVEL_NAMES):
            assert not _literal_definition(blob, name), name


class TestSkillDiscoveryPerFixtureClass:
    """Every fixture class keeps the skill surface."""

    @pytest.mark.timeout(60)
    def test_every_class_keeps_the_skill_definition(self, tmp_path: Path) -> None:
        for build in (
            _ordinary_coder_child,
            _browser_opt_in_child,
            _small_catalog_child,
        ):
            child = build(tmp_path)
            assert _literal_definition(_wire_blob(child), "skill"), build.__name__
            assert child.get("skill") is not None, build.__name__
