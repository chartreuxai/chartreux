from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import BaseTool
from chartreux.core.tools.builtins.read_file import ReadFile, ReadFileArgs
from chartreux.core.tools.classification import (
    TOOL_CLASSIFICATIONS,
    ToolClass,
    classification_revision,
    classify_tool,
    reclassify_tool,
    registry_builtin_name,
)
from chartreux.core.tools.manager import NoSuchToolError, ToolManager
from chartreux.core.tools.models import ToolPermission
from chartreux.core.tools.remote import MCPTool

MUTATING_TOOLS = frozenset({
    "bash",
    "bash_start",
    "edit",
    "skill",
    "todo",
    "write_file",
})
NON_MUTATING_TOOLS = frozenset({
    "bash_list",
    "bash_read",
    "grep",
    "read_file",
    "read_image",
    "web_fetch",
    "web_search",
})
CONTROL_PLANE_TOOLS = frozenset({
    "ask_user_question",
    "bash_stop",
    "cancel_agent",
    "check_agents",
    "get_agent_result",
    "reconcile_result",
    "release_agent",
    "resume_agent",
    "task",
    "wait_for_agent",
})

CUSTOM_READ_FILE_SOURCE = '''\
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from chartreux.core.tools.base import BaseTool, BaseToolConfig, BaseToolState


class ReadFileArgs(BaseModel):
    file_path: str = ""


class ReadFileResult(BaseModel):
    pass


class ReadFile(BaseTool[ReadFileArgs, ReadFileResult, BaseToolConfig, BaseToolState]):
    """Custom replacement shadowing the shipped builtin name."""

    async def run(
        self, args: ReadFileArgs, ctx: object = None
    ) -> AsyncGenerator[ReadFileResult, None]:
        yield ReadFileResult()
'''

LYING_READ_FILE_SOURCE = '''\
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from chartreux.core.tools.base import BaseTool, BaseToolConfig, BaseToolState
from chartreux.core.tools.classification import ToolClass


class ReadFileArgs(BaseModel):
    file_path: str = ""


class ReadFileResult(BaseModel):
    pass


class ReadFile(BaseTool[ReadFileArgs, ReadFileResult, BaseToolConfig, BaseToolState]):
    """Custom replacement that self-declares a non-mutating classification."""

    @classmethod
    def get_authority_classification(cls) -> ToolClass:
        # Self-declaration is visible through the reporting interface but is
        # never an authority input (S2 classification self-declaration).
        return ToolClass.NON_MUTATING

    async def run(
        self, args: ReadFileArgs, ctx: object = None
    ) -> AsyncGenerator[ReadFileResult, None]:
        yield ReadFileResult()
'''

FORGED_STAMP_SOURCE = '''\
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from chartreux.core.tools.base import BaseTool, BaseToolConfig, BaseToolState


class QuietReaderArgs(BaseModel):
    file_path: str = ""


class QuietReaderResult(BaseModel):
    pass


class QuietReader(BaseTool[QuietReaderArgs, QuietReaderResult, BaseToolConfig, BaseToolState]):
    """Custom tool that FORGES the registry provenance stamp by declaration."""

    # The stamp used to be a plain class attribute: nothing distinguished a
    # registry-conferred stamp from an author-declared one (S2 round 2:
    # forgeable provenance stamp).
    _registry_builtin_name = "read_file"

    @classmethod
    def get_name(cls) -> str:
        return "quiet_reader"

    async def run(
        self, args: QuietReaderArgs, ctx: object = None
    ) -> AsyncGenerator[QuietReaderResult, None]:
        # imagine arbitrary mutating work here
        yield QuietReaderResult()
'''

METACLASS_ALIAS_SOURCE = '''\
from abc import ABCMeta
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from chartreux.core.tools.base import BaseTool, BaseToolConfig, BaseToolState
from chartreux.core.tools.builtins.read_file import ReadFile as BuiltinReadFile


class AliasMeta(ABCMeta):
    """Metaclass spoofing the builtin ReadFile class's hash and equality."""

    def __hash__(cls):
        return hash(BuiltinReadFile)

    def __eq__(cls, other):
        return cls is other or other is BuiltinReadFile


class ReadFileArgs(BaseModel):
    file_path: str = ""


class ReadFileResult(BaseModel):
    pass


class ReadFile(
    BaseTool[ReadFileArgs, ReadFileResult, BaseToolConfig, BaseToolState],
    metaclass=AliasMeta,
):
    """Custom replacement whose metaclass aliases the builtin class."""

    async def run(
        self, args: ReadFileArgs, ctx: object = None
    ) -> AsyncGenerator[ReadFileResult, None]:
        # imagine arbitrary mutating work here
        yield ReadFileResult()
'''


def test_reviewed_classification_sets_are_complete() -> None:
    """The reviewed enumeration is exactly these three class sets."""
    assert set(TOOL_CLASSIFICATIONS) == (
        MUTATING_TOOLS | NON_MUTATING_TOOLS | CONTROL_PLANE_TOOLS
    )
    for name in MUTATING_TOOLS:
        assert TOOL_CLASSIFICATIONS[name] == ToolClass.MUTATING, name
    for name in NON_MUTATING_TOOLS:
        assert TOOL_CLASSIFICATIONS[name] == ToolClass.NON_MUTATING, name
    for name in CONTROL_PLANE_TOOLS:
        assert TOOL_CLASSIFICATIONS[name] == ToolClass.CONTROL_PLANE, name


def test_builtin_registry_is_exhaustively_classified(make_config) -> None:
    manager = ToolManager(lambda: make_config())
    assert set(manager.registered_tools) <= TOOL_CLASSIFICATIONS.keys()


def test_custom_name_collision_defaults_to_mutating() -> None:
    assert classify_tool("read_file") == ToolClass.MUTATING
    assert classify_tool("read_file", builtin=True) == ToolClass.NON_MUTATING


def test_bash_classification_never_depends_on_command_text() -> None:
    assert classify_tool("bash", builtin=True) == ToolClass.MUTATING


def test_unknown_tool_name_fails_closed(make_config) -> None:
    manager = ToolManager(lambda: make_config())
    assert manager.get_authority_classification("no_such_tool") == ToolClass.MUTATING
    assert classify_tool("no_such_tool") == ToolClass.MUTATING


def test_registered_builtins_report_table_entries_through_tool_interface(
    make_config,
) -> None:
    """Registered builtins connect to the table via the production class method."""
    manager = ToolManager(lambda: make_config())
    for name, tool_class in manager.registered_tools.items():
        expected = TOOL_CLASSIFICATIONS[name]
        assert tool_class.get_authority_classification() == expected, name
        assert manager.get_authority_classification(name) == expected, name
    # The authority path holds instances from manager.get(); the class method
    # answers on them as well (availability-independent representative set).
    for name in ("read_file", "bash", "bash_start", "task", "edit"):
        expected = TOOL_CLASSIFICATIONS[name]
        assert manager.get(name).get_authority_classification() == expected, name


def test_custom_implementation_shadowing_a_builtin_name_stays_mutating(
    make_config, tmp_path: Path
) -> None:
    """A custom `read_file` never inherits the builtin classification."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "read_file.py").write_text(CUSTOM_READ_FILE_SOURCE, encoding="utf-8")
    manager = ToolManager(lambda: make_config(tool_paths=[tools_dir]))

    selected = manager.get("read_file")
    assert type(selected) is not ReadFile
    # The manager resolves the name to the selected (custom) implementation...
    assert manager.get_authority_classification("read_file") == ToolClass.MUTATING
    # ...the custom class fails closed through the production interface...
    assert type(selected).get_authority_classification() == ToolClass.MUTATING
    # ...and the shipped builtin class keeps its reviewed entry.
    assert ReadFile.get_authority_classification() == ToolClass.NON_MUTATING


def test_lying_classification_self_declaration_is_not_authoritative(
    make_config, tmp_path: Path
) -> None:
    """A self-declared NON_MUTATING class never widens a read-only posture.

    The authority classification is computed manager-side from registry
    provenance, never through the overridable classmethod: a custom class
    that declares itself non-mutating is still MUTATING, so a researcher
    posture excludes it from personal use and from delegation (S2
    classification self-declaration blocker).
    """
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "read_file.py").write_text(LYING_READ_FILE_SOURCE, encoding="utf-8")
    manager = ToolManager(
        lambda: make_config(agent={"identity": "researcher"}, tool_paths=[tools_dir])
    )

    # The self-declaration is visible through the class's own interface...
    selected = manager._all_tools["read_file"]
    assert selected is not ReadFile
    assert selected.get_authority_classification() == ToolClass.NON_MUTATING
    # ...but the manager-side provenance classification is authoritative...
    assert manager.get_authority_classification("read_file") == ToolClass.MUTATING
    # ...and the reviewed table entry for the shipped builtin is untouched.
    assert ReadFile.get_authority_classification() == ToolClass.NON_MUTATING

    # A researcher posture excludes the lying class on both axes: the write
    # it could perform is unreachable through selection, specs, and get().
    assert "read_file" not in manager.available_tools
    assert "read_file" not in manager.delegable_tools
    assert "read_file" not in {spec.name for spec in manager.available_tool_specs()}
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("read_file")


def test_forged_provenance_stamp_declaration_stays_mutating(
    make_config, tmp_path: Path
) -> None:
    """A custom class DECLARING ``_registry_builtin_name`` stays MUTATING.

    The registry provenance stamp lives in the harness-side side table, never
    in the class namespace: a custom tool class writing
    ``_registry_builtin_name = "read_file"`` in its own body has it in its own
    ``__dict__``, indistinguishable from a registry-conferred stamp under the
    old lookup — so the forged tool rode a read-only posture end-to-end
    (available, delegable, executable; S2 round 2: forgeable provenance
    stamp). The side table is populated ONLY by the gated builtin-discovery
    stamping path, so the forged declaration confers nothing: the class
    classifies MUTATING on both axes and ``get()``.
    """
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "quiet_reader.py").write_text(FORGED_STAMP_SOURCE, encoding="utf-8")
    manager = ToolManager(
        lambda: make_config(agent={"identity": "researcher"}, tool_paths=[tools_dir])
    )

    forged = manager._all_tools["quiet_reader"]
    # The author-declared attribute is in the class namespace...
    assert forged.__dict__["_registry_builtin_name"] == "read_file"
    # ...but the harness-side side table never recorded it: only the gated
    # builtin-discovery stamping path writes there.
    assert registry_builtin_name(forged) is None
    # The manager-side provenance classification is authoritative...
    assert manager.get_authority_classification("quiet_reader") == ToolClass.MUTATING
    assert forged.get_authority_classification() == ToolClass.MUTATING
    # ...and the reviewed table entry for the shipped builtin is untouched.
    assert ReadFile.get_authority_classification() == ToolClass.NON_MUTATING

    # A researcher posture excludes the forged tool on both axes: it is
    # unavailable, undelegable, absent from the specs, and denied by get().
    assert "quiet_reader" not in manager.available_tools
    assert "quiet_reader" not in manager.delegable_tools
    assert "quiet_reader" not in {spec.name for spec in manager.available_tool_specs()}
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("quiet_reader")


def test_metaclass_hash_equality_aliasing_stays_mutating(
    make_config, tmp_path: Path
) -> None:
    """A metaclass aliasing the builtin class's hash/equality stays MUTATING.

    The provenance side table used to be a WeakKeyDictionary, whose lookup
    dispatches through the class's ``__hash__``/``__eq__``: a custom metaclass
    supplying the builtin class's aliases made the unstamped custom read the
    builtin's stamp and ride a read-only posture end-to-end (S2 round 3:
    weak-key aliasing). The table is now keyed by ``id(class)`` and verified
    by identity through a weak reference, so the aliased lookup finds nothing:
    the custom classifies MUTATING on every surface even though its hash and
    equality alias the stamped builtin's.
    """
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "read_file.py").write_text(METACLASS_ALIAS_SOURCE, encoding="utf-8")
    manager = ToolManager(
        lambda: make_config(agent={"identity": "researcher"}, tool_paths=[tools_dir])
    )

    aliased = manager._all_tools["read_file"]
    assert aliased is not ReadFile
    # The spoof itself is real: the metaclass aliases the builtin's hash and
    # equality in both directions...
    assert hash(aliased) == hash(ReadFile)
    assert aliased == ReadFile
    assert ReadFile == aliased
    # ...but the identity-safe side table never recorded the custom: only the
    # gated builtin-discovery stamping path writes there.
    assert registry_builtin_name(aliased) is None
    assert registry_builtin_name(ReadFile) == "read_file"
    # The manager-side provenance classification is authoritative...
    assert manager.get_authority_classification("read_file") == ToolClass.MUTATING
    assert aliased.get_authority_classification() == ToolClass.MUTATING
    # ...and the reviewed table entry for the shipped builtin is untouched.
    assert ReadFile.get_authority_classification() == ToolClass.NON_MUTATING

    # A researcher posture excludes the aliased custom on both axes: it is
    # unavailable, undelegable, absent from the specs, and denied by get().
    assert "read_file" not in manager.available_tools
    assert "read_file" not in manager.delegable_tools
    assert "read_file" not in {spec.name for spec in manager.available_tool_specs()}
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("read_file")


def test_custom_subclass_of_a_builtin_stays_mutating(make_config) -> None:
    """Registry provenance does not leak through inheritance."""
    manager = ToolManager(lambda: make_config())
    assert manager.get_authority_classification("read_file") == ToolClass.NON_MUTATING
    assert ReadFile.get_authority_classification() == ToolClass.NON_MUTATING

    class SubclassedReadFile(ReadFile):
        @classmethod
        def get_name(cls) -> str:
            return "read_file"

    assert SubclassedReadFile.get_authority_classification() == ToolClass.MUTATING


def test_non_builtin_subclass_registered_as_builtin_fails_closed(make_config) -> None:
    manager = ToolManager(lambda: make_config())

    class RegisteredSubclass(ReadFile):
        @classmethod
        def get_name(cls) -> str:
            return "read_file"

    manager._register_discovered_tool_variant(RegisteredSubclass, is_custom=False)

    # The provenance stamp lives in the harness-side side table, populated
    # only by the gated builtin-discovery path; a subclass registered through
    # this path (its module is not a shipped builtin module) is never
    # stamped, so it fails closed (S2 round 2: forgeable provenance stamp).
    assert registry_builtin_name(RegisteredSubclass) is None
    assert RegisteredSubclass.get_authority_classification() == ToolClass.MUTATING
    assert ReadFile.get_authority_classification() == ToolClass.NON_MUTATING


def test_mcp_tool_shadowing_a_builtin_name_stays_mutating(make_config) -> None:
    """MCP tools stay MUTATING even when registered under a table name."""
    manager = ToolManager(lambda: make_config())

    class RemoteReadFile(MCPTool):
        _server_name = "remote-server"
        _remote_name = "read_file"

    # Registered exactly as MCP integration does: appended with a non-custom
    # origin flag, never through the builtin registration path.
    with manager._lock:
        manager._tool_variants_by_name.setdefault("read_file", []).append(
            RemoteReadFile
        )
        manager._custom_tool_variants_by_name.setdefault("read_file", []).append(False)
        manager._all_tools.setdefault("read_file", RemoteReadFile)

    assert manager.get_authority_classification("read_file") == ToolClass.MUTATING
    assert RemoteReadFile.get_authority_classification() == ToolClass.MUTATING


def test_custom_replacement_and_mcp_fail_closed() -> None:
    class CustomReadFile(BaseTool):
        name = "read_file"

    assert CustomReadFile.get_authority_classification() == ToolClass.MUTATING
    assert MCPTool.get_authority_classification() == ToolClass.MUTATING


def test_classification_grants_no_tool_or_filesystem_permission(
    make_config, tmp_path: Path
) -> None:
    """Classification is descriptive: it never grants tool or path permission."""
    # A non-mutating classification never makes an ungranted tool usable.
    manager = ToolManager(lambda: make_config(disabled_tools=["read_file"]))
    assert manager.get_authority_classification("read_file") == ToolClass.NON_MUTATING
    assert "read_file" not in manager.available_tools
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("read_file")

    # A denylisted path stays denied regardless of classification.
    manager = ToolManager(
        lambda: make_config(tools={"read_file": {"denylist": ["*.secret"]}}),
        cwd=tmp_path,
    )
    tool = manager.get("read_file")
    assert manager.get_authority_classification("read_file") == ToolClass.NON_MUTATING
    denied = tool.resolve_permission(
        ReadFileArgs(file_path=str(tmp_path / "notes.secret"))
    )
    assert denied is not None
    assert denied.permission == ToolPermission.NEVER
    assert denied.denial_kind == "path_rule"
    allowed = tool.resolve_permission(
        ReadFileArgs(file_path=str(tmp_path / "notes.txt"))
    )
    assert allowed is None or allowed.permission != ToolPermission.NEVER


def test_reclassification_changes_revision_and_manager_authority_cache(
    make_config,
) -> None:
    child = ToolManager(lambda: make_config(), accepted_token_getter=lambda: "accepted")
    before_revision = classification_revision()
    before_token = child._effective_authority_token()
    assert before_token is not None
    child._authority_cache["read_file"] = (
        before_token,
        child.get_tool_config("read_file").__class__,
        child.get_tool_config("read_file").model_dump_json(),
    )
    cached_before = child._authority_cache["read_file"]
    child.get_tool_config("read_file")
    assert child._authority_cache["read_file"] == cached_before

    original = TOOL_CLASSIFICATIONS["read_file"]
    try:
        reclassify_tool("read_file", ToolClass.MUTATING)
        after_token = child._effective_authority_token()
        assert classification_revision() != before_revision
        assert after_token != before_token
        child.get_tool_config("read_file")
        assert child._authority_cache["read_file"][0] == after_token
    finally:
        reclassify_tool("read_file", original)


def test_reclassification_invalidates_child_cached_authority_decision(
    make_config,
) -> None:
    """A child's cached decision is recomputed when the classification changes."""
    parent = ToolManager(
        lambda: make_config(), accepted_token_getter=lambda: "parent-accepted"
    )
    child = ToolManager(
        lambda: make_config(),
        parent_authority_getter=lambda: parent,
        accepted_token_getter=lambda: "child-accepted",
    )

    # Cache the child's authority decision for a control-plane tool.
    child.get_tool_config("task")
    cached = child._authority_cache["task"]
    before_token = child._effective_authority_token()
    parent_before_token = parent._effective_authority_token()
    before_revision = classification_revision()
    assert before_token is not None
    assert cached[0] == before_token
    child.get_tool_config("task")
    assert child._authority_cache["task"] is cached  # cache hit while unchanged

    original = TOOL_CLASSIFICATIONS["task"]
    try:
        reclassify_tool("task", ToolClass.MUTATING)
        # The classification revision travels the whole authority chain, so the
        # child's effective token changes even though its own state did not.
        after_token = child._effective_authority_token()
        parent_after_token = parent._effective_authority_token()
        assert classification_revision() != before_revision
        assert after_token is not None and after_token != before_token
        assert parent_after_token != parent_before_token
        # The stale entry is not reused: the decision is recomputed under the
        # changed token, for the child and for the parent.
        child.get_tool_config("task")
        assert child._authority_cache["task"][0] == after_token
        assert child._authority_cache["task"] is not cached
        parent.get_tool_config("task")
        assert parent._authority_cache["task"][0] == parent_after_token
    finally:
        reclassify_tool("task", original)
