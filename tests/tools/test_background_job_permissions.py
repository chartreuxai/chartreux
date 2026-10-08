from __future__ import annotations

from collections.abc import AsyncGenerator
import json

from pydantic import BaseModel
import pytest

from chartreux.core.background_jobs import BashStartArgs
from chartreux.core.tools import secret_redaction as sr
from chartreux.core.tools.base import BaseTool, BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_permission_resolver import (
    ShellPermissionResolver,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.manager import NoSuchToolError, ToolManager
from chartreux.core.tools.permissions import PermissionContext
from tests.conftest import build_test_vibe_config


class StartStub(BaseTool[BashStartArgs, BaseModel, BashToolConfig, BaseToolState]):
    @classmethod
    def get_name(cls) -> str:
        return "bash_start"

    def resolve_permission(self, args: BashStartArgs) -> PermissionContext:
        return PermissionContext(permission=ToolPermission.ALWAYS)

    async def run(
        self, args: BashStartArgs, ctx=None
    ) -> AsyncGenerator[BaseModel, None]:
        yield BaseModel()


def manager_for(tmp_path, *, parent=None, **config):
    settings = build_test_vibe_config(**config)
    manager = ToolManager(
        lambda: settings,
        cwd=tmp_path,
        defer_mcp=True,
        accepted_token_getter=lambda: "unchanged",
        parent_authority_getter=(lambda: parent) if parent else None,
    )
    manager._register_discovered_tool_variant(StartStub, is_custom=False)
    return manager


@pytest.mark.parametrize(
    "command",
    [
        "echo hello",
        "curl example.org",
        "bash -i",
        "python",
        "eval true",
        "cat /outside/file",
        "echo x > .git/config",
        # Shipped default denylist entries, including the normalized git form.
        "git push",
        "git -C . push",
        "git checkout main",
        "git stash drop",
        "git stash clear",
        "git restore file.txt",
        "git switch --discard-changes",
        "git switch -f main",
        "git reflog expire --all",
        "git reflog delete main",
    ],
)
@pytest.mark.parametrize("child", [False, True])
def test_foreground_start_permission_parity(tmp_path, command, child):
    parent = manager_for(tmp_path)
    manager = manager_for(tmp_path, parent=parent) if child else parent
    foreground = manager.get("bash").resolve_permission(BashArgs(command=command))
    launch = manager.get("bash_start").resolve_permission(
        BashStartArgs(command=command)
    )
    assert foreground is not None and launch is not None
    assert foreground.permission == launch.permission
    shell = manager.get("bash")
    shared = ShellPermissionResolver(
        shell.config, shell.cwd, shell.workspace, shell.path_authority
    )
    resolved = shared.resolve_permission(BashStartArgs(command=command))
    assert resolved is not None
    assert resolved.permission == foreground.permission


@pytest.mark.parametrize("child", [False, True])
def test_cleared_denylist_parity_for_git_forms(tmp_path, child):
    # A configured empty denylist removes the shipped git defaults; both
    # adapters agree that the forms are then permitted.
    parent = manager_for(tmp_path, tools={"bash": {"denylist": []}})
    manager = (
        manager_for(tmp_path, parent=parent, tools={"bash": {"denylist": []}})
        if child
        else parent
    )
    for command in (
        "git push",
        "git -C . push",
        "git checkout main",
        "git stash drop",
        "git stash clear",
        "git restore file.txt",
        "git switch --discard-changes",
        "git switch -f main",
        "git reflog expire --all",
        "git reflog delete main",
    ):
        foreground = manager.get("bash").resolve_permission(BashArgs(command=command))
        launch = manager.get("bash_start").resolve_permission(
            BashStartArgs(command=command)
        )
        assert foreground is not None and launch is not None
        assert foreground.permission == launch.permission
        assert foreground.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize("child", [False, True])
def test_replaced_denylist_parity_for_git_forms(tmp_path, child):
    # A replacement list omitting the git entries permits those forms while
    # preserving unrelated defaults; both adapters agree.
    parent = manager_for(tmp_path, tools={"bash": {"denylist": ["curl"]}})
    manager = (
        manager_for(tmp_path, parent=parent, tools={"bash": {"denylist": ["curl"]}})
        if child
        else parent
    )
    for command, expected in (
        ("git push", ToolPermission.ALWAYS),
        ("git -C . push", ToolPermission.ALWAYS),
        ("git checkout main", ToolPermission.ALWAYS),
        ("git stash drop", ToolPermission.ALWAYS),
        ("git stash clear", ToolPermission.ALWAYS),
        ("git restore file.txt", ToolPermission.ALWAYS),
        ("git switch --discard-changes", ToolPermission.ALWAYS),
        ("git switch -f main", ToolPermission.ALWAYS),
        ("git reflog expire --all", ToolPermission.ALWAYS),
        ("git reflog delete main", ToolPermission.ALWAYS),
        ("curl example.org", ToolPermission.NEVER),
    ):
        foreground = manager.get("bash").resolve_permission(BashArgs(command=command))
        launch = manager.get("bash_start").resolve_permission(
            BashStartArgs(command=command)
        )
        assert foreground is not None and launch is not None
        assert foreground.permission == expected
        assert launch.permission == expected


@pytest.mark.parametrize("child", [False, True])
def test_canonical_never_blocks_start(tmp_path, child):
    parent = manager_for(tmp_path, tools={"bash": {"permission": "never"}})
    manager = manager_for(tmp_path, parent=parent) if child else parent
    assert manager.get_tool_config("bash_start").permission is ToolPermission.NEVER
    result = manager.get("bash_start").resolve_permission(
        BashStartArgs(command="echo ok")
    )
    assert result is not None
    assert result.permission is ToolPermission.NEVER


def test_parent_custom_resolver_receives_foreground_arguments(tmp_path):
    received = []

    class CustomBash(Bash):
        @classmethod
        def get_name(cls):
            return "bash"

        def resolve_permission(self, args: BashArgs):
            assert type(args) is BashArgs
            received.append(args)
            return PermissionContext(
                permission=ToolPermission.NEVER, reason="custom denial"
            )

    parent = manager_for(tmp_path)
    parent._register_discovered_tool_variant(CustomBash, is_custom=True)
    child = manager_for(tmp_path, parent=parent)
    result = child.get("bash_start").resolve_permission(
        BashStartArgs(command="echo ok", label="job")
    )
    assert result is not None
    assert result.reason == "custom denial"
    assert received == [BashArgs(command="echo ok")]


def test_custom_bash_availability_invalidates_launch_selection(tmp_path):
    available = [True]

    class CustomBash(Bash):
        @classmethod
        def get_name(cls):
            return "bash"

        @classmethod
        def is_available(cls, config=None):
            return available[0]

    parent = manager_for(tmp_path)
    # Isolate canonical authority to a custom implementation, with no fallback.
    parent._tool_variants_by_name["bash"] = []
    parent._register_discovered_tool_variant(CustomBash, is_custom=True)
    child = manager_for(tmp_path, parent=parent)
    for state in (True, False, True, False):
        available[0] = state
        for manager in (parent, child):
            assert not manager._name_versionable("bash_start")
            assert ("bash_start" in manager.available_tools) is state
            assert (
                "bash_start" in {spec.name for spec in manager.available_tool_specs()}
            ) is state
            if state:
                assert isinstance(manager.get("bash_start"), StartStub)
            else:
                with pytest.raises(NoSuchToolError):
                    manager.get("bash_start")


def test_disabling_shell_does_not_alias_recovery_names(tmp_path):
    manager = manager_for(tmp_path, disabled_tools=["bash"])
    with pytest.raises(NoSuchToolError):
        manager.get("bash_start")
    for name in ("bash_read", "bash_stop", "bash_list", "bash_other"):
        assert manager.get_tool_config(name).permission is not ToolPermission.NEVER


@pytest.mark.parametrize(
    "command",
    [
        r"printf synth\etic-credential",
        "printf 'synthetic-'credential",
        r"printf $'synthetic\x2dcredential'",
    ],
)
def test_recorded_start_redacts_shell_escaped_credentials(command):
    from chartreux.core.llm_models import FunctionCall, ToolCall

    call = ToolCall(
        function=FunctionCall(
            name="bash_start",
            arguments=json.dumps({"command": command, "label": "job"}),
        )
    )
    with sr.bind_policy(
        sr.ScrubPolicy(redaction_credentials=(("TEST_TOKEN", "synthetic-credential"),))
    ):
        cleaned = sr.sanitize_recorded_tool_call(call)
        recorded = json.loads(cleaned.function.arguments)
        assert "[REDACTED]" in recorded["command"]
        assert recorded["label"] == "job"


@pytest.mark.parametrize("key", ["permission", "denylist", "cwd", "unknown"])
def test_launch_has_no_separate_permission_config(tmp_path, key):
    with pytest.raises(ValueError, match=r"\[tools.bash\]"):
        manager_for(tmp_path, tools={"bash_start": {key: "always"}})
    manager = manager_for(tmp_path, tools={"bash": {"denylist": ["echo"]}})
    for name in ("bash", "bash_start"):
        assert manager.get_tool_config(name).denylist == ["echo"]
    assert manager.path_authority("bash_start").tool_name == "bash"


def test_parent_launch_specific_denial_is_preserved(tmp_path):
    class DeniedStart(StartStub):
        def resolve_permission(self, args: BashStartArgs) -> PermissionContext:
            return PermissionContext(
                permission=ToolPermission.NEVER, reason="parent launch denial"
            )

    parent = manager_for(tmp_path)
    parent._register_discovered_tool_variant(DeniedStart, is_custom=True)
    child = manager_for(tmp_path, parent=parent)
    result = child.get("bash_start").resolve_permission(
        BashStartArgs(command="echo ok")
    )
    assert result is not None and result.reason == "parent launch denial"


def test_parent_denylist_cannot_be_removed_by_child(tmp_path):
    parent = manager_for(tmp_path, tools={"bash": {"denylist": ["echo"]}})
    child = manager_for(tmp_path, parent=parent, tools={"bash": {"denylist": []}})
    result = child.get("bash_start").resolve_permission(
        BashStartArgs(command="echo ok")
    )
    assert result is not None and result.permission is ToolPermission.NEVER
    assert "denylist" in (result.reason or "")


def test_parent_git_denial_cannot_be_removed_by_child(tmp_path):
    # The shipped git defaults are removable by configuration, but a parent's
    # configured denial is inherited authority the child cannot drop.
    parent = manager_for(tmp_path, tools={"bash": {"denylist": ["git push"]}})
    child = manager_for(tmp_path, parent=parent, tools={"bash": {"denylist": []}})
    for tool_name, args in (
        ("bash", BashArgs(command="git push")),
        ("bash_start", BashStartArgs(command="git push")),
    ):
        result = child.get(tool_name).resolve_permission(args)
        assert result is not None and result.permission is ToolPermission.NEVER
        assert "denylist" in (result.reason or "")


def test_unavailable_parent_authority_fails_closed(tmp_path):
    parent = manager_for(tmp_path)
    child = manager_for(tmp_path, parent=parent)
    retained = child.get("bash_start")

    def unavailable():
        raise RuntimeError("parent retired")

    child._parent_authority_getter = unavailable
    result = retained.resolve_permission(BashStartArgs(command="echo ok"))
    assert result is not None and result.permission is ToolPermission.NEVER


def test_launch_instruction_reads_are_exact_and_read_only(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("instruction")
    other = tmp_path / "other.txt"
    other.write_text("other")
    parent = manager_for(workspace)
    parent.set_instruction_read_files(frozenset({instruction}))
    child = manager_for(workspace, parent=parent)
    for manager in (parent, child):
        for command, expected in (
            (f"cat {instruction}", ToolPermission.ALWAYS),
            (f"cat {other}", ToolPermission.NEVER),
            (f"echo x > {instruction}", ToolPermission.NEVER),
        ):
            result = manager.get("bash_start").resolve_permission(
                BashStartArgs(command=command)
            )
            assert result is not None and result.permission is expected


@pytest.mark.parametrize("name", ["bash_read", "bash_stop", "bash_list", "bash_other"])
def test_recovery_tools_do_not_resolve_shell_source(tmp_path, name):
    class RecoveryStub(StartStub):
        @classmethod
        def get_name(cls):
            return name

    manager = manager_for(tmp_path, disabled_tools=["bash"])
    manager._register_discovered_tool_variant(RecoveryStub, is_custom=False)
    result = manager.get(name).resolve_permission(BashStartArgs(command="curl denied"))
    assert result is not None and result.permission is ToolPermission.ALWAYS
    assert name in {spec.name for spec in manager.available_tool_specs()}
