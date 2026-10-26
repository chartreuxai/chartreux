"""Per-axis identity authority: granted, personal, and delegation postures.

Pins the v0.5.7 authority model (design §8): three independent axes.

- GRANTED — config-level grants (allow/deny lists, per-tool permission,
  source restrictions). The granted ceiling never widens down the tree.
- PERSONAL — the identity posture: which classified tool classes the
  identity may execute itself. Enforced INSIDE ToolManager, below the
  config layer, so launch-level and config-level tool overrides can never
  loosen it (the documented v0.5.7 policy reversal).
- DELEGATION — the ceiling on what the identity may pass to children.
  Enforced by the ``_parent_allows_tool`` chain-walk: a child's tool is
  admitted only when every ancestor's delegation set allows it.

Chain rule: a child's effective tools are the granted tree ceiling ∩ every
ancestor's delegation set ∩ the child's personal posture. Enforcement uses
the S1-approved classification classes (mutating / non-mutating /
control-plane), never tool-name lists: control-plane tools (task, agent
lifecycle, user questions) ride with a read-only posture, so an
orchestrator can delegate without gaining direct write authority.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.core.agents.identities import UnknownIdentityError
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.tools.base import BaseToolConfig
from chartreux.core.tools.builtins.bash import BashArgs
from chartreux.core.tools.builtins.edit import EditArgs
from chartreux.core.tools.classification import ToolClass
from chartreux.core.tools.manager import NoSuchToolError, ToolManager
from chartreux.core.tools.models import ToolPermission
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import collect_result

MUTATING_TOOLS = frozenset({"bash", "bash_start", "edit", "write_file"})
NON_MUTATING_TOOLS = frozenset({"read_file", "skill"})
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
    "todo",
    "wait_for_agent",
})

# A custom read_file variant whose availability alternates on every predicate
# call: an availability decision and a name-keyed re-resolution inside the
# SAME admission decision see different answers (S2 round 2: posture reselects
# by name).
ALTERNATING_READ_FILE_SOURCE = '''\
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from chartreux.core.tools.base import BaseTool, BaseToolConfig, BaseToolState


class ReadFileArgs(BaseModel):
    file_path: str = ""


class ReadFileResult(BaseModel):
    pass


class ReadFile(BaseTool[ReadFileArgs, ReadFileResult, BaseToolConfig, BaseToolState]):
    """Custom variant whose availability alternates on every predicate call."""

    _available = True

    @classmethod
    def is_available(cls, config=None) -> bool:
        cls._available = not cls._available
        return not cls._available

    async def run(
        self, args: ReadFileArgs, ctx: object = None
    ) -> AsyncGenerator[ReadFileResult, None]:
        yield ReadFileResult()
'''

# A custom read_file variant whose availability depends on the identity: it is
# available only under a coder child, never under the researcher parent (S2
# round 3: the ancestor chain re-resolves by name).
CONFIG_DEPENDENT_READ_FILE_SOURCE = '''\
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from chartreux.core.tools.base import BaseTool, BaseToolConfig, BaseToolState


class ReadFileArgs(BaseModel):
    file_path: str = ""


class ReadFileResult(BaseModel):
    pass


class ReadFile(BaseTool[ReadFileArgs, ReadFileResult, BaseToolConfig, BaseToolState]):
    """Custom variant available only under a coder identity."""

    @classmethod
    def is_available(cls, config=None) -> bool:
        return bool(config is not None and config.agent.identity == "coder")

    async def run(
        self, args: ReadFileArgs, ctx: object = None
    ) -> AsyncGenerator[ReadFileResult, None]:
        # imagine arbitrary mutating work here
        yield ReadFileResult()
'''

# The mirror image of the above: a custom read_file variant available only
# under the RESEARCHER identity, so the researcher PARENT selects it while a
# coder child cannot see it and selects the genuine builtin (S2 round 4:
# invocation/availability implementation consistency, reverse-variant case).
RESEARCHER_SELECTED_READ_FILE_SOURCE = '''\
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from chartreux.core.tools.base import BaseTool, BaseToolConfig, BaseToolState


class ReadFileArgs(BaseModel):
    file_path: str = ""


class ReadFileResult(BaseModel):
    pass


class ReadFile(BaseTool[ReadFileArgs, ReadFileResult, BaseToolConfig, BaseToolState]):
    """Custom variant available only under a researcher identity."""

    selection_priority = 1

    @classmethod
    def is_available(cls, config=None) -> bool:
        return bool(config is not None and config.agent.identity == "researcher")

    async def run(
        self, args: ReadFileArgs, ctx: object = None
    ) -> AsyncGenerator[ReadFileResult, None]:
        # imagine arbitrary mutating work here
        yield ReadFileResult()
'''

# An unstamped custom variant of the carved-out ``bash`` name: it wins
# selection under its own name, but the registry never stamped it, so the
# reviewer's execution carve-out must not admit it (P1-h fail-closed pin).
UNSTAMPED_CUSTOM_BASH_SOURCE = '''\
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from chartreux.core.tools.base import BaseTool, BaseToolConfig, BaseToolState


class BashArgs(BaseModel):
    command: str = ""


class BashResult(BaseModel):
    pass


class Bash(BaseTool[BashArgs, BashResult, BaseToolConfig, BaseToolState]):
    """Unstamped custom variant of the carved-out bash name."""

    async def run(
        self, args: BashArgs, ctx: object = None
    ) -> AsyncGenerator[BashResult, None]:
        # imagine arbitrary mutating work here
        yield BashResult()
'''


def _manager(
    tmp_path: Path,
    *,
    identity: str | None = None,
    parent: ToolManager | None = None,
    **config: object,
) -> ToolManager:
    kwargs: dict[str, object] = dict(config)
    if identity is not None:
        kwargs["agent"] = {"identity": identity}
    resolved = build_test_vibe_config(**kwargs)  # type: ignore[arg-type]
    return ToolManager(
        lambda: resolved,
        cwd=tmp_path,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
        parent_authority_getter=(lambda: parent) if parent is not None else None,
    )


def test_coder_identity_is_unrestricted_on_both_axes(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    assert MUTATING_TOOLS <= set(manager.available_tools)
    assert MUTATING_TOOLS <= set(manager.delegable_tools)
    assert manager.get("edit") is not None


def test_root_researcher_cannot_use_mutating_tools_personally(tmp_path: Path) -> None:
    # The write tools are GRANTED at the config level; the researcher's
    # personal posture still excludes them below the config layer.
    manager = _manager(tmp_path, identity="researcher")
    assert not (MUTATING_TOOLS & set(manager.available_tools))
    assert not (MUTATING_TOOLS & {spec.name for spec in manager.available_tool_specs()})
    for name in MUTATING_TOOLS:
        with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
            manager.get(name)
    assert "read_file" in manager.available_tools
    assert "grep" in manager.available_tools


def test_root_researcher_cannot_delegate_mutating_tools(tmp_path: Path) -> None:
    manager = _manager(tmp_path, identity="researcher")
    assert not (MUTATING_TOOLS & set(manager.delegable_tools))
    assert "read_file" in manager.delegable_tools
    assert "task" in manager.delegable_tools


def test_root_orchestrator_cannot_write_personally_but_delegates_writes(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path, identity="orchestrator")
    # Personal axis: no mutating tool, despite the granted config.
    assert not (MUTATING_TOOLS & set(manager.available_tools))
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("edit")
    # Delegation axis: the full granted ceiling, writes included.
    assert MUTATING_TOOLS <= set(manager.delegable_tools)
    # Control-plane stays personal: the orchestrator can delegate at all.
    assert "task" in manager.available_tools


def test_control_plane_tools_survive_a_read_only_posture(tmp_path: Path) -> None:
    for identity in ("researcher", "reviewer", "advisor", "orchestrator"):
        manager = _manager(tmp_path, identity=identity)
        assert CONTROL_PLANE_TOOLS <= set(manager.available_tools), identity
        assert "task" in manager.available_tools


def test_read_only_identities_resolve_skill_and_orchestrator_resolves_todo(
    tmp_path: Path,
) -> None:
    # skill is session-scoped instruction loading, not a mutation: every
    # read-only identity admits it. todo writes session-local tracking state
    # only, so it rides with the orchestrator's control-plane set.
    for identity in ("researcher", "reviewer", "advisor"):
        manager = _manager(tmp_path, identity=identity)
        assert "skill" in manager.available_tools, identity
        assert manager.get("skill") is not None, identity
    orchestrator = _manager(tmp_path, identity="orchestrator")
    assert "todo" in orchestrator.available_tools
    assert orchestrator.get("todo") is not None


def test_child_of_orchestrator_writes_within_the_granted_ceiling(
    tmp_path: Path,
) -> None:
    parent = _manager(tmp_path, identity="orchestrator")
    child = _manager(tmp_path, identity="coder", parent=parent)
    # The child's personal posture (coder) plus the parent's delegation set
    # (all granted) admit the mutating tools...
    assert MUTATING_TOOLS <= set(child.available_tools)
    # ...and delegated execution actually resolves permission through the
    # parent's policy (the parent's personal posture excludes the tool).
    target = tmp_path / "notes.txt"
    target.write_text("original\n", encoding="utf-8")
    edit = child.get("edit")
    context = edit.resolve_permission(
        EditArgs(file_path=str(target), old_string="original", new_string="changed")
    )
    assert context is None or context.permission != ToolPermission.NEVER


def test_child_of_researcher_cannot_write_despite_coder_identity(
    tmp_path: Path,
) -> None:
    parent = _manager(tmp_path, identity="researcher")
    child = _manager(tmp_path, identity="coder", parent=parent)
    # The child's own posture is unrestricted, but the ancestor's delegation
    # set (non-mutating) caps the chain: the coder child cannot write.
    assert not (MUTATING_TOOLS & set(child.available_tools))
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        child.get("edit")
    assert "read_file" in child.available_tools


def test_child_tool_overrides_cannot_loosen_the_child_identity_posture(
    tmp_path: Path,
) -> None:
    # Documented v0.5.7 policy reversal, pinned below the config layer: the
    # reviewer identity's personal posture is non-mutating, and neither a
    # config allowlist nor a per-tool "always" override can loosen it.
    manager = _manager(
        tmp_path,
        identity="reviewer",
        enabled_tools=["edit", "read_file"],
        tools={"edit": {"permission": "always"}},
    )
    assert "edit" not in manager.available_tools
    assert "edit" not in {spec.name for spec in manager.available_tool_specs()}
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("edit")
    assert "read_file" in manager.available_tools


def test_narrowing_remains_allowed_below_the_posture(tmp_path: Path) -> None:
    # Narrowing is the one direction overrides may move: disabling an
    # otherwise-allowed tool still takes effect under a restricted posture.
    manager = _manager(tmp_path, identity="reviewer")
    assert "read_file" in manager.available_tools
    narrowed = _manager(tmp_path, identity="reviewer", disabled_tools=["read_file"])
    assert "read_file" not in narrowed.available_tools
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        narrowed.get("read_file")
    assert (
        _manager(
            tmp_path, identity="reviewer", tools={"read_file": {"permission": "never"}}
        )
        .get_tool_config("read_file")
        .permission
        is ToolPermission.NEVER
    )


def test_explicit_root_denial_prevents_delegated_writes(tmp_path: Path) -> None:
    # Availability-level denial: the granted ceiling never widens.
    parent = _manager(tmp_path, identity="orchestrator", disabled_tools=["edit"])
    child = _manager(tmp_path, identity="coder", parent=parent)
    assert "edit" not in child.available_tools
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        child.get("edit")
    # Permission-level denial: the child's config inherits the NEVER ceiling.
    parent = _manager(
        tmp_path, identity="orchestrator", tools={"edit": {"permission": "never"}}
    )
    child = _manager(tmp_path, identity="coder", parent=parent)
    assert child.get_tool_config("edit").permission is ToolPermission.NEVER


def test_custom_identity_delegation_defaults_to_the_full_granted_ceiling(
    tmp_path: Path,
) -> None:
    # Design decision 8 (named in the S2 review scope): a custom identity
    # carries no harness-known posture — unrestricted personal use, and a
    # delegation set defaulting to the FULL granted ceiling.
    for reference in ("user:my-custom", "my-custom"):
        manager = _manager(tmp_path, identity=reference)
        assert MUTATING_TOOLS <= set(manager.available_tools), reference
        assert MUTATING_TOOLS <= set(manager.delegable_tools), reference
        assert manager.get("edit") is not None
    # The granted ceiling still caps the custom identity's delegation.
    manager = _manager(tmp_path, identity="user:my-custom", disabled_tools=["edit"])
    assert "edit" not in manager.delegable_tools


def test_builtin_qualified_identity_reference_keeps_its_posture(tmp_path: Path) -> None:
    manager = _manager(tmp_path, identity="builtin:researcher")
    assert not (MUTATING_TOOLS & set(manager.available_tools))
    assert not (MUTATING_TOOLS & set(manager.delegable_tools))


def test_delegation_and_personal_axes_are_independently_monotonic(
    tmp_path: Path,
) -> None:
    # A coder parent grants writes; a researcher CHILD still cannot use them
    # personally (personal axis), while its delegation set stays read-only.
    parent = _manager(tmp_path, identity="coder")
    child = _manager(tmp_path, identity="researcher", parent=parent)
    assert MUTATING_TOOLS <= set(parent.delegable_tools)
    assert not (MUTATING_TOOLS & set(child.available_tools))
    assert not (MUTATING_TOOLS & set(child.delegable_tools))
    # The grandchild chain keeps narrowing: a coder grandchild of the
    # researcher is still capped by the researcher's delegation set.
    grandchild = _manager(tmp_path, identity="coder", parent=child)
    assert not (MUTATING_TOOLS & set(grandchild.available_tools))


def test_unavailable_parent_authority_fails_closed_on_both_axes(tmp_path: Path) -> None:
    """A parent chain gap fails closed, never open (3b lifecycle hardening).

    An ancestor whose authority cannot be resolved admits no personal tool
    and no delegation: both axes consult the live chain, so a gap yields the
    empty ceiling on each. The authority token also disables caching while
    the chain is broken, so no entry obtained while the chain was healthy
    is ever reused across the gap.
    """
    resolved = build_test_vibe_config(agent={"identity": "coder"})

    def unavailable() -> ToolManager:
        raise RuntimeError("parent authority unavailable")

    manager = ToolManager(
        lambda: resolved,
        cwd=tmp_path,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
        parent_authority_getter=unavailable,
    )
    # No cached authority decision can be served: the token is unavailable.
    assert manager._effective_authority_token() is None
    # Both axes fail closed through the gap. File tools fail on the
    # chain-walked scratchpad authority; other tools on availability.
    assert manager.available_tools == {}
    assert manager.delegable_tools == {}
    for name in ("edit", "read_file", "task"):
        with pytest.raises(
            NoSuchToolError, match="unavailable|Unknown or disabled tool"
        ):
            manager.get(name)


def test_bound_identity_freezes_posture_against_in_place_flips(tmp_path: Path) -> None:
    """The conversation's identity binding wins over a live config flip.

    A manager constructed with a bound identity derives its posture from the
    BOUND identity (design §3, decision log item 3): an in-place
    ``config.agent.identity`` edit is a pending selection for new
    conversations and never re-postures an existing manager.
    """
    resolved = build_test_vibe_config(agent={"identity": "coder"})
    manager = ToolManager(
        lambda: resolved,
        cwd=tmp_path,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
        bound_identity="researcher",
    )
    assert not (MUTATING_TOOLS & set(manager.available_tools))
    assert "read_file" in manager.available_tools
    # The bound identity holds even against a live in-place flip to coder.
    resolved.agent.identity = "coder"
    assert not (MUTATING_TOOLS & set(manager.available_tools))
    assert not (MUTATING_TOOLS & set(manager.delegable_tools))
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("edit")


def test_unknown_builtin_qualified_identity_fails_at_config_load() -> None:
    """A typo'd ``builtin:`` selection is a config-load error (criterion 2)."""
    with pytest.raises(ValidationError, match="unknown built-in identity"):
        build_test_vibe_config(agent={"identity": "builtin:reviever"})


def test_unknown_bare_identity_fails_closed_at_loop_construction(
    tmp_path: Path,
) -> None:
    """An unknown bare identity is an actionable loop-construction error.

    The schema accepts any well-formed id, so the fail-closed check lives at
    loop construction: an identity that is neither a builtin nor a resolvable
    user resource raises instead of defaulting to an unrestricted posture
    (S2 unknown-identity fail-open).
    """
    config = build_test_vibe_config(agent={"identity": "reviever"})
    with pytest.raises(UnknownIdentityError, match="reviever"):
        build_test_agent_loop(config=config, cwd=tmp_path)


@pytest.mark.asyncio
async def test_valid_custom_identity_loop_keeps_decision_eight_behavior(
    tmp_path: Path, config_dir: Path
) -> None:
    """A valid custom identity keeps decision-8 behavior at the loop level.

    The ``user:``-qualified selection is validated at loop construction: the
    prompt resource must exist in a user prompts directory (S2 round 2:
    unvalidated user:-qualified identities), and a valid custom keeps its
    decision-8 posture (unrestricted personal + full granted delegation).
    """
    prompts = config_dir / "prompts"
    prompts.mkdir()
    (prompts / "my-custom.md").write_text("# My custom identity\n", encoding="utf-8")
    config = build_test_vibe_config(agent={"identity": "user:my-custom"})
    loop = build_test_agent_loop(config=config, cwd=tmp_path)
    try:
        assert loop._bound_identity == "user:my-custom"
        assert MUTATING_TOOLS <= set(loop.tool_manager.available_tools)
        assert MUTATING_TOOLS <= set(loop.tool_manager.delegable_tools)
        assert loop.tool_manager.get("edit") is not None
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_loop_binds_a_bare_custom_identity_with_a_prompt_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare custom identity binds when a user prompt resource exists.

    ``user:``-qualified selections are explicit custom qualifications; a bare
    non-builtin name is valid only when ``<name>.md`` exists in a user prompts
    directory — otherwise construction fails closed.
    """
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "my-custom.md").write_text("# My custom identity\n", encoding="utf-8")

    class _PromptDirs(HarnessFilesManager):
        @property
        def user_prompts_dirs(self) -> list[Path]:
            return [prompts]

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.get_harness_files_manager",
        lambda: _PromptDirs(),
    )
    config = build_test_vibe_config(agent={"identity": "my-custom"})
    loop = build_test_agent_loop(config=config, cwd=tmp_path)
    try:
        assert loop._bound_identity == "my-custom"
        assert MUTATING_TOOLS <= set(loop.tool_manager.available_tools)
    finally:
        await loop.aclose()
    # Without the resource the same bare name fails closed.
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.get_harness_files_manager",
        lambda: _PromptDirs(),
    )
    (prompts / "my-custom.md").unlink()
    with pytest.raises(UnknownIdentityError, match="my-custom"):
        build_test_agent_loop(config=config, cwd=tmp_path)


def test_bare_malformed_identity_fails_at_loop_construction_with_a_matching_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare traversal identity fails at loop construction — matching file.

    The schema rejects traversal/nested/multi-segment ids at config load,
    but an unsanctioned in-place ``config.agent.identity`` edit bypasses the
    schema validators; loop construction must still fail closed — even when
    a matching prompt file exists (S2 round 3 W4: the bare branch accepted
    any reference whose file existed, deferring the rejection to the
    ResourceLoader at prompt assembly).
    """
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    # The file the OLD existence-only check would have matched: the bare
    # reference '../x' resolves to <prompts>/../x.md.
    (tmp_path / "x.md").write_text("# traversal target\n", encoding="utf-8")

    class _PromptDirs(HarnessFilesManager):
        @property
        def user_prompts_dirs(self) -> list[Path]:
            return [prompts]

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.get_harness_files_manager",
        lambda: _PromptDirs(),
    )
    config = build_test_vibe_config(agent={"identity": "coder"})
    # In-place edit: bypasses the schema validator, reaches loop construction.
    config.agent.identity = "../x"
    with pytest.raises(UnknownIdentityError, match="Malformed custom identity"):
        build_test_agent_loop(config=config, cwd=tmp_path)


def test_cache_hit_rechecks_the_live_identity_posture(tmp_path: Path) -> None:
    """A warm selection cannot serve a tool the live posture excludes (S2 W2).

    The live posture and filter checks re-run on EVERY availability decision,
    cache hits included: an unsanctioned in-place identity flip must not
    leave a warm selection returning the tool through ``get()``.
    """
    resolved = build_test_vibe_config(agent={"identity": "coder"})
    manager = ToolManager(
        lambda: resolved,
        cwd=tmp_path,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
    )
    assert manager.get("edit") is not None  # warm the selection and instance
    resolved.agent.identity = "orchestrator"
    assert "edit" not in manager.available_tools
    assert "edit" not in {spec.name for spec in manager.available_tool_specs()}
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("edit")


def test_cache_hit_rejects_a_malformed_in_place_tool_entry(tmp_path: Path) -> None:
    """A malformed in-place ``tools`` entry fails closed on the cache-hit path.

    The schema types ``tools`` entries as config mappings; an in-place edit
    that stores a model instance instead violates that type, and broken
    config never serves a warm selection (S2 W2). Well-formed dict entries
    keep the pinned execution-time denial semantics.
    """
    resolved = build_test_vibe_config(agent={"identity": "coder"})
    manager = ToolManager(
        lambda: resolved,
        cwd=tmp_path,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
    )
    assert manager.get("edit") is not None
    resolved.tools["edit"] = BaseToolConfig(permission=ToolPermission.NEVER)  # type: ignore[dict-item]
    assert "edit" not in manager.available_tools
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("edit")
    # A well-formed dict entry keeps the execution-time denial: the tool stays
    # selectable, and the patched NEVER reaches the execution decision.
    resolved.tools["edit"] = {"permission": "never"}
    assert "edit" in manager.available_tools
    assert manager.get_tool_config("edit").permission is ToolPermission.NEVER


def test_posture_classifies_the_exact_selected_variant(tmp_path: Path) -> None:
    """The posture check classifies the SELECTED class, never a name re-resolution.

    With an unstamped custom ``read_file`` variant alternating availability, a
    name-keyed posture lookup re-resolves the name inside the SAME admission
    decision and can classify the stamped BUILTIN while the CUSTOM class is
    the one selected — letting the unstamped custom ride a read-only posture
    (S2 round 2: posture reselects by name). The posture check must classify
    the exact selected class: it fails closed as MUTATING even though the
    stamped builtin of the same name is available.
    """
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "read_file.py").write_text(
        ALTERNATING_READ_FILE_SOURCE, encoding="utf-8"
    )
    manager = _manager(tmp_path, identity="researcher", tool_paths=[tools_dir])

    custom = manager._all_tools["read_file"]
    assert custom is not None

    def reset() -> None:
        # Each availability decision consumes one predicate call; reset the
        # alternating flag so every surface starts with the custom available
        # (and therefore selected — the later discovery index wins ties).
        custom._available = True  # type: ignore[attr-defined]

    reset()
    # The custom variant wins selection; the selected class is unstamped, so
    # the manager-side classification of the name is MUTATING even though the
    # stamped builtin of the same name is available.
    assert manager.get_authority_classification("read_file") is ToolClass.MUTATING
    # The researcher's read-only posture excludes the SELECTED custom on both
    # axes: it is denied on every surface even though the builtin of the same
    # name is stamped and available.
    reset()
    assert "read_file" not in manager.available_tools
    reset()
    assert "read_file" not in manager.delegable_tools
    reset()
    assert "read_file" not in {spec.name for spec in manager.available_tool_specs()}
    reset()
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("read_file")
    # The shipped builtin keeps its reviewed entry, and a manager without the
    # custom variant still serves the builtin under the same posture.
    from chartreux.core.tools.builtins.read_file import ReadFile

    assert ReadFile.get_authority_classification() is ToolClass.NON_MUTATING
    plain = _manager(tmp_path, identity="researcher")
    assert "read_file" in plain.available_tools


def test_posture_denies_a_selected_custom_when_a_builtin_variant_is_available(
    tmp_path: Path,
) -> None:
    """A selected unstamped custom variant is denied even when the stamped
    builtin of the same name is available and selected elsewhere.

    The availability decision selects one concrete class per name; the
    posture check must classify THAT class. Here the custom variant is
    statically available and wins selection, so the researcher posture
    excludes the name on both axes — while the builtin class itself keeps its
    reviewed entry (S2 round 2: posture reselects by name).
    """
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "read_file.py").write_text(
        ALTERNATING_READ_FILE_SOURCE.replace(
            "        cls._available = not cls._available\n"
            "        return not cls._available\n",
            "        return True\n",
        ),
        encoding="utf-8",
    )
    manager = _manager(tmp_path, identity="researcher", tool_paths=[tools_dir])
    assert manager._all_tools["read_file"] is not None
    assert manager.get_authority_classification("read_file") is ToolClass.MUTATING
    assert "read_file" not in manager.available_tools
    assert "read_file" not in manager.delegable_tools
    assert "read_file" not in {spec.name for spec in manager.available_tool_specs()}
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("read_file")


def test_ancestor_chain_classifies_the_children_selected_implementation(
    tmp_path: Path,
) -> None:
    """The ancestor chain admits the child's EXACT selected implementation.

    A custom ``read_file`` variant is available only under a coder identity:
    the researcher parent selects (and delegates) the non-mutating BUILTIN by
    name, while the coder child selects its own mutating custom variant. The
    chain-walk used to validate the tool NAME against each ancestor's
    delegable set — derived for the ANCESTOR's selected implementation — so
    the child's mutating custom was admitted through availability, get(), and
    permission resolution (S2 round 3: the ancestor chain re-resolves by
    name). Every ancestor's delegation posture now classifies the
    implementation the child will RUN: the custom is denied on every surface,
    while the genuine builtin under the same name remains delegable.
    """
    from chartreux.core.tools.builtins.read_file import ReadFile

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "read_file.py").write_text(
        CONFIG_DEPENDENT_READ_FILE_SOURCE, encoding="utf-8"
    )
    parent = _manager(tmp_path, identity="researcher", tool_paths=[tools_dir])
    child = _manager(tmp_path, identity="coder", parent=parent, tool_paths=[tools_dir])

    # The researcher parent selects the builtin; the coder child selects its
    # own custom variant of the same name.
    parent_selected = parent._select_available_variant(
        "read_file", parent._all_tools["read_file"]
    )
    custom = child._select_available_variant("read_file", child._all_tools["read_file"])
    assert parent_selected is ReadFile
    assert custom is not None and custom is not ReadFile
    # The custom variant is unstamped, so the child's manager-side
    # classification of the name is MUTATING.
    assert child.get_authority_classification("read_file") is ToolClass.MUTATING

    # The child's custom implementation is denied on every surface: personal
    # availability, delegation availability, model-facing specs, get(), path
    # authority, and the execution-time authority check.
    assert "read_file" not in child.available_tools
    assert "read_file" not in child.delegable_tools
    assert "read_file" not in {spec.name for spec in child.available_tool_specs()}
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        child.get("read_file")
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        child.path_authority("read_file")
    assert not child._tool_authority_is_current("read_file", custom)

    # The genuine builtin under the same name remains delegable: the
    # researcher parent still passes it, and a coder child without the custom
    # variant runs it through the same chain.
    assert parent.delegable_tools["read_file"] is ReadFile
    plain_child = _manager(tmp_path, identity="coder", parent=parent)
    assert "read_file" in plain_child.available_tools
    assert "read_file" in plain_child.delegable_tools
    assert type(plain_child.get("read_file")) is ReadFile

    # The denial holds at depth two: a coder grandchild selecting the same
    # custom variant stays capped by the researcher ancestor through the
    # child's delegation ceiling.
    grandchild = _manager(
        tmp_path, identity="coder", parent=child, tool_paths=[tools_dir]
    )
    assert "read_file" not in grandchild.available_tools
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        grandchild.get("read_file")


def test_invocation_classifies_the_children_execution_implementation(
    tmp_path: Path,
) -> None:
    """Availability and invocation classify the SAME implementation (S2 round 4).

    Reverse-variant case: a custom ``read_file`` available only under the
    researcher identity is SELECTED by the researcher parent, while the coder
    child selects the genuine builtin. Availability admits the child's builtin
    (the ancestor chain classifies the carried class), so the invocation guard
    must agree: the parent's delegation posture classifies the child's
    execution implementation, and an in-workspace read resolves ALLOWED —
    never the pre-fix over-narrowing NEVER ("Tool is unavailable under parent
    authority") that classified the PARENT's own variant through a name-keyed
    re-resolution.

    The policy side stays the parent's own (peer-seat safety boundary): the
    delegated policy instance is built from the PARENT's selected variant
    under the parent's config — never the child's execution implementation —
    and the parent's per-tool policy remains authoritative for permission
    values, so a child-forged ``resolve_permission`` cannot weaken the
    parent's ceiling.
    """
    from chartreux.core.tools.builtins.read_file import ReadFile, ReadFileArgs

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "read_file.py").write_text(
        RESEARCHER_SELECTED_READ_FILE_SOURCE, encoding="utf-8"
    )
    parent = _manager(tmp_path, identity="researcher", tool_paths=[tools_dir])
    child = _manager(tmp_path, identity="coder", parent=parent, tool_paths=[tools_dir])

    # The researcher parent selects the unstamped custom; the coder child
    # selects the genuine builtin.
    parent_selected = parent._select_available_variant(
        "read_file", parent._all_tools["read_file"]
    )
    child_selected = child._select_available_variant(
        "read_file", child._all_tools["read_file"]
    )
    assert parent_selected is not None and parent_selected is not ReadFile
    assert child_selected is ReadFile

    # Availability admits the child's builtin...
    assert "read_file" in child.available_tools
    served = child.get("read_file")
    assert type(served) is ReadFile
    # ...and invocation agrees: the in-workspace read resolves ALLOWED.
    target = tmp_path / "notes.txt"
    target.write_text("original\n", encoding="utf-8")
    decision = served.resolve_permission(ReadFileArgs(file_path=str(target)))
    assert decision is None or decision.permission != ToolPermission.NEVER

    # The parent's delegated policy instance is the PARENT's own variant under
    # the parent's config — never the child's execution implementation.
    policy_instance = parent._delegated_tool("read_file", ReadFile)
    assert type(policy_instance) is parent_selected
    # The parent's own per-tool policy stays authoritative for values: a
    # parent-level NEVER still denies the child's invocation.
    strict_parent = _manager(
        tmp_path,
        identity="researcher",
        tool_paths=[tools_dir],
        tools={"read_file": {"permission": "never"}},
    )
    strict_child = _manager(
        tmp_path, identity="coder", parent=strict_parent, tool_paths=[tools_dir]
    )
    strict_decision = strict_child.get("read_file").resolve_permission(
        ReadFileArgs(file_path=str(target))
    )
    assert strict_decision is not None
    assert strict_decision.permission is ToolPermission.NEVER
    assert strict_decision.reason == "Tool is disabled by parent policy"


def test_invocation_guard_denies_the_children_custom_under_a_read_only_chain(
    tmp_path: Path,
) -> None:
    """The mirror case stays NEVER at the invocation layer (S2 round 4).

    A custom ``read_file`` available only under the coder identity is selected
    by the coder child, while the researcher parent selects the builtin.
    Availability denies the child's mutating custom on every surface (pinned
    above, S2 round 3); the invocation guard must classify the SAME child
    implementation — the pre-fix name-only guard classified the parent's
    builtin and admitted what availability denied. Called the way the child's
    bound instance would call it (with the child's selected class), the guard
    stays NEVER: threading the execution implementation narrows the guard to
    the child's implementation without widening anything.
    """
    from chartreux.core.tools.builtins.read_file import ReadFile, ReadFileArgs

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "read_file.py").write_text(
        CONFIG_DEPENDENT_READ_FILE_SOURCE, encoding="utf-8"
    )
    parent = _manager(tmp_path, identity="researcher", tool_paths=[tools_dir])
    child = _manager(tmp_path, identity="coder", parent=parent, tool_paths=[tools_dir])

    custom = child._select_available_variant("read_file", child._all_tools["read_file"])
    assert custom is not None and custom is not ReadFile
    # The availability surfaces stay denied (S2 round 3 pin).
    assert "read_file" not in child.available_tools
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        child.get("read_file")
    # The invocation guard classifies the child's selected custom and stays
    # NEVER — the mirror of the reverse-variant case must not widen.
    target = tmp_path / "notes.txt"
    target.write_text("original\n", encoding="utf-8")
    denial = child._parent_permission(
        "read_file", ReadFileArgs(file_path=str(target)), custom
    )
    assert denial is not None
    assert denial.permission is ToolPermission.NEVER
    assert denial.reason == "Tool is unavailable under parent authority"


def test_explicit_none_tool_entry_fails_closed_on_every_surface(tmp_path: Path) -> None:
    """An explicit null ``tools`` entry denies the tool everywhere (S2 round 2).

    The schema types ``tools`` entries as config mappings, so an explicit
    ``None`` is malformed; an in-place edit storing it must fail closed on
    every surface — inventories, specs, ``get()``, and the effective
    ``get_tool_config`` permission — never a silent reset to defaults (which
    would flip a warm ``"never"`` entry to ``always``).
    """
    resolved = build_test_vibe_config(agent={"identity": "coder"})
    manager = ToolManager(
        lambda: resolved,
        cwd=tmp_path,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
    )
    warm = manager.get("edit")
    assert warm is not None
    assert manager.get_tool_config("edit").permission is ToolPermission.ALWAYS
    # Replace the warm entry with an explicit null (malformed per the schema
    # type) without an accepted-token bump.
    resolved.tools["edit"] = None  # type: ignore[dict-item]
    assert "edit" not in manager.available_tools
    assert "edit" not in manager.delegable_tools
    assert "edit" not in {spec.name for spec in manager.available_tool_specs()}
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("edit")
    # The effective permission is NEVER, not a reset to the default ALWAYS.
    assert manager.get_tool_config("edit").permission is ToolPermission.NEVER
    # The warm instance is denied at the execution decision as well.
    from chartreux.core.tools.builtins.edit import EditArgs

    decision = warm.resolve_permission(
        EditArgs(file_path=str(tmp_path / "x.txt"), old_string="a", new_string="b")
    )
    assert decision is not None and decision.permission is ToolPermission.NEVER
    # An absent entry still means "no per-tool policy": the tool is served.
    del resolved.tools["edit"]
    assert "edit" in manager.available_tools
    assert manager.get_tool_config("edit").permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "reference",
    [
        "user:reviewer",  # reserved shipped name
        "user:coder",  # reserved shipped name
        "user:orchestrator",  # reserved shipped name
        "user:does-not-exist",  # no prompt resource
    ],
)
def test_user_qualified_identity_references_fail_closed_at_loop_construction(
    tmp_path: Path, config_dir: Path, reference: str
) -> None:
    """``user:``-qualified selections are validated at loop construction.

    Reserved shipped names and references whose ``<name>.md`` prompt resource
    does not exist in the user prompts directories are actionable errors —
    never a deferred rejection at prompt assembly (S2 round 2: unvalidated
    user:-qualified identities; design criterion 2; package-1
    reserved-names acceptance). Malformed syntax forms are rejected even
    earlier, at config load (see tests/core/agents/test_identities.py for the
    loop-level syntax check).
    """
    (config_dir / "prompts").mkdir()
    config = build_test_vibe_config(agent={"identity": reference})
    with pytest.raises(UnknownIdentityError) as raised:
        build_test_agent_loop(config=config, cwd=tmp_path)
    message = str(raised.value)
    assert reference in message
    if reference == "user:does-not-exist":
        assert "does-not-exist.md" in message
        assert str(config_dir / "prompts") in message
    else:
        assert "Reserved identity name" in message


@pytest.mark.asyncio
async def test_user_qualified_identity_with_a_resource_binds_at_loop_construction(
    tmp_path: Path, config_dir: Path
) -> None:
    """A ``user:`` selection whose prompt resource exists binds normally."""
    prompts = config_dir / "prompts"
    prompts.mkdir()
    (prompts / "my-custom.md").write_text("# My custom identity\n", encoding="utf-8")
    config = build_test_vibe_config(agent={"identity": "user:my-custom"})
    loop = build_test_agent_loop(config=config, cwd=tmp_path)
    try:
        assert loop._bound_identity == "user:my-custom"
    finally:
        await loop.aclose()


# --- Reviewer execution carve-out (settlement decision 3, P1-h) --------------


@pytest.mark.asyncio
async def test_reviewer_resolves_bash_and_runs_a_verification_command(
    tmp_path: Path,
) -> None:
    """The exit-gate reviewer executes verification itself (decision 3, P1-h).

    The reviewer's personal posture stays non-mutating, but its identity-level
    execution carve-out admits the bash tool: the reviewer resolves it through
    the manager's personal axis and runs a verification-style command.
    """
    manager = _manager(tmp_path, identity="reviewer")
    assert "bash" in manager.available_tools
    assert "bash" in {spec.name for spec in manager.available_tool_specs()}
    bash_tool = manager.get("bash")
    result = await collect_result(
        bash_tool.run(BashArgs(command="echo verification-ok"))
    )
    assert result.exit_code == 0
    assert result.stdout == "verification-ok\n"


def test_reviewer_carve_out_is_bash_only(tmp_path: Path) -> None:
    """The non-mutating contract holds except the execution carve-out.

    The carve-out admits exactly the bash surface; the write tools and
    bash_start stay excluded, and an override cannot loosen them.
    """
    manager = _manager(tmp_path, identity="reviewer")
    for name in ("edit", "write_file", "bash_start"):
        assert name not in manager.available_tools, name
        with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
            manager.get(name)
    loosened = _manager(
        tmp_path,
        identity="reviewer",
        enabled_tools=["edit", "write_file", "bash_start"],
    )
    for name in ("edit", "write_file", "bash_start"):
        assert name not in loosened.available_tools, name


def test_researcher_and_advisor_remain_without_the_bash_tool(tmp_path: Path) -> None:
    """The carve-out is reviewer-scoped: researcher and advisor keep no shell."""
    for identity in ("researcher", "advisor"):
        manager = _manager(tmp_path, identity=identity)
        assert "bash" not in manager.available_tools, identity
        assert "bash" not in manager.delegable_tools, identity
        with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
            manager.get("bash")


def test_reviewer_carve_out_never_reaches_the_delegation_axis(tmp_path: Path) -> None:
    """The reviewer runs bash itself but cannot hand the shell to a child.

    The carve-out is personal-axis only: the reviewer's delegation set stays
    non-mutating, so a coder child under a reviewer is still capped by the
    ancestor chain.
    """
    manager = _manager(tmp_path, identity="reviewer")
    assert "bash" in manager.available_tools
    assert "bash" not in manager.delegable_tools
    child = _manager(tmp_path, identity="coder", parent=manager)
    assert "bash" not in child.available_tools
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        child.get("bash")


def test_unstamped_custom_bash_variant_fails_closed_under_the_reviewer(
    tmp_path: Path,
) -> None:
    """The carve-out admits the reviewed builtin, never a custom namesake.

    Launch admission is name-keyed, but the runtime boundary is the registry-
    provenance stamp of the exact selected class: an unstamped custom variant
    named ``bash`` classifies MUTATING and stays excluded even under the
    reviewer's carve-out.
    """
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "bash.py").write_text(UNSTAMPED_CUSTOM_BASH_SOURCE, encoding="utf-8")
    manager = _manager(tmp_path, identity="reviewer", tool_paths=[tools_dir])
    assert manager._all_tools["bash"] is not None
    assert "bash" not in manager.available_tools
    with pytest.raises(NoSuchToolError, match="Unknown or disabled tool"):
        manager.get("bash")
