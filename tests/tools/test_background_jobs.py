from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError
import pytest

from chartreux.acp.session_updates import _TOOL_KINDS
from chartreux.app_server._tool_projection import (
    project_effect_output,
    project_effect_output_value,
)
from chartreux.core.background_jobs import (
    BackgroundJobRegistry,
    BashListArgs,
    BashReadArgs,
    BashStartArgs,
    BashStopArgs,
)
from chartreux.core.events import ToolResultEvent
from chartreux.core.llm.format import ResolvedToolCall
from chartreux.core.llm_models import Role
from chartreux.core.tools.base import InvokeContext, ToolError, ToolPermission
from chartreux.core.tools.builtins.bash import BashArgs
from chartreux.core.tools.builtins.bash_list import BashList
from chartreux.core.tools.builtins.bash_read import BashRead
from chartreux.core.tools.builtins.bash_start import BashStart
from chartreux.core.tools.builtins.bash_stop import BashStop
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.secret_redaction import ScrubPolicy, bind_policy
from chartreux.core.tools.ui import ToolUIDataAdapter
from chartreux.utils.tool_presentation import ToolEffectKind
from tests.conftest import build_test_agent_loop, build_test_vibe_config


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.asyncio
async def test_launch_uses_authorized_cwd_and_new_runtime_after_relocation(
    tmp_path, child
):
    registry = BackgroundJobRegistry()
    root = registry.root_port()
    port = root.borrow() if child else root
    ctx = InvokeContext(tool_call_id="cwd", background_jobs=port)
    target = tmp_path / "relocated"
    target.mkdir()
    try:
        original = manager(tmp_path).get("bash_start")
        assert (
            original.resolve_permission(
                BashStartArgs(command="pwd; printf 'ready\\n'; sleep 30")
            )
            is not None
        )
        old = await run(
            original, BashStartArgs(command="pwd; printf 'ready\\n'; sleep 30"), ctx
        )
        old_page = await port.read(BashReadArgs(job_id=old.job.job_id, wait_seconds=2))
        assert "".join(r.text for r in old_page.records).splitlines()[0] == str(
            tmp_path
        )
        # Runtime rebinding installs tools with the new trusted cwd; the old OS
        # process must remain bound to its launch directory.
        relocated = manager(target).get("bash_start")
        new = await run(
            relocated, BashStartArgs(command="pwd; printf 'ready\\n'; sleep 30"), ctx
        )
        new_page = await port.read(BashReadArgs(job_id=new.job.job_id, wait_seconds=2))
        assert "".join(r.text for r in new_page.records).splitlines()[0] == str(target)
        assert "".join(
            r.text
            for r in (await port.read(BashReadArgs(job_id=old.job.job_id))).records
        ).splitlines()[0] == str(tmp_path)
    finally:
        await registry.aclose()


@pytest.mark.parametrize("child", [False, True])
def test_launch_analyzes_once_and_rechecks_each_request(tmp_path, monkeypatch, child):
    from chartreux.core.tools.builtins import _shell_permission_resolver as resolver

    parent = manager(tmp_path)
    tools = manager(tmp_path) if child else parent
    if child:
        tools._parent_authority_getter = lambda: parent
    original = resolver._analyze_guardrail_source_uncached
    calls = []

    def analyze(command, **kwargs):
        calls.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(resolver, "_analyze_guardrail_source_uncached", analyze)
    tool = tools.get("bash_start")
    args = BashStartArgs(command="echo ok")
    context = tool.resolve_permission(args)
    assert context is not None and context.permission is ToolPermission.ALWAYS
    assert calls == ["echo ok"]
    context = tool.resolve_permission(args)
    assert context is not None and context.permission is ToolPermission.ALWAYS
    assert calls == ["echo ok", "echo ok"]


TOOLS = (BashStart, BashRead, BashStop, BashList)


def manager(tmp_path: Path, **kwargs: Any) -> ToolManager:
    config = build_test_vibe_config(**kwargs)
    return ToolManager(lambda: config, cwd=tmp_path, defer_mcp=True)


async def run(tool: Any, args: Any, ctx: InvokeContext | None = None) -> Any:
    return [result async for result in tool.run(args, ctx)][0]


@pytest.mark.parametrize(
    ("schema", "values"),
    [
        (BashStartArgs, {"command": " "}),
        (BashStartArgs, {"command": "echo ok", "label": "x" * 65}),
        (BashStartArgs, {"command": "echo ok", "label": "two\nlines"}),
        *[
            (BashStartArgs, {"command": "echo ok", key: "x"})
            for key in ("cwd", "env", "launch_cap", "timeout")
        ],
        (BashReadArgs, {"job_id": ""}),
        *[
            (BashReadArgs, {"job_id": "j", key: value})
            for key, value in (
                ("cursor", -1),
                ("cursor", True),
                ("max_bytes", 4095),
                ("max_bytes", 64001),
                ("wait_seconds", 31),
                ("wait_seconds", float("nan")),
            )
        ],
        (BashStopArgs, {"job_id": ""}),
        (BashStopArgs, {"job_id": "j", "force": True}),
        (BashListArgs, {"include_finished": "true"}),
        (BashListArgs, {"creator": "guess"}),
    ],
)
def test_schema_rejects_invalid_controls(schema, values):
    with pytest.raises(ValidationError):
        schema.model_validate(values)


@pytest.mark.parametrize("tool_class", TOOLS)
def test_discovery_schema_prompt_defaults_and_presentation(tmp_path, tool_class):
    tools = manager(tmp_path)
    name = tool_class.get_name()
    assert tools.registered_tools[name] is tool_class
    assert type(tools.get(name)) is tool_class
    spec = next(spec for spec in tools.available_tool_specs() if spec.name == name)
    assert spec.parameters == tool_class.get_parameters()
    assert spec.parameters["additionalProperties"] is False
    assert spec.description == tool_class.get_tool_prompt()
    assert "5-15" in spec.description and "compaction" in spec.description
    assert tools.get_tool_config(name).permission is ToolPermission.ALWAYS
    assert tool_class.effect_kind is ToolEffectKind.TOOL
    assert _TOOL_KINDS[tool_class.effect_kind] == "other"


def test_recovery_overrides_and_description_overrides(tmp_path):
    tools = manager(tmp_path, tools={"bash_read": {"permission": "never"}})
    assert tools.get_tool_config("bash_read").permission is ToolPermission.NEVER
    tools._tool_descriptions["bash_list"] = "Custom recovery guidance"
    spec = next(
        spec for spec in tools.available_tool_specs() if spec.name == "bash_list"
    )
    assert spec.description == "Custom recovery guidance"


@pytest.mark.parametrize(
    "tool_class,args",
    [
        (BashStart, BashStartArgs(command="echo ok")),
        (BashRead, BashReadArgs(job_id="j")),
        (BashStop, BashStopArgs(job_id="j")),
        (BashList, BashListArgs()),
    ],
)
@pytest.mark.parametrize("ctx", [None, InvokeContext(tool_call_id="missing")])
@pytest.mark.asyncio
async def test_missing_registry_fails_safely(tmp_path, tool_class, args, ctx):
    with pytest.raises(ToolError, match="unavailable"):
        await run(manager(tmp_path).get(tool_class.get_name()), args, ctx)


@pytest.mark.parametrize(
    "command",
    ["echo ok", "curl example.org", "eval true", "echo x > .git/config", "sleep 30"],
)
@pytest.mark.parametrize("child", [False, True])
def test_real_adapters_permission_parity_and_parent_denial(tmp_path, command, child):
    parent = manager(tmp_path, tools={"bash": {"denylist": ["echo"]}})
    tools = manager(tmp_path) if child else parent
    if child:
        tools._parent_authority_getter = lambda: parent
    foreground = tools.get("bash").resolve_permission(BashArgs(command=command))
    launch = tools.get("bash_start").resolve_permission(BashStartArgs(command=command))
    assert foreground is not None and launch is not None
    assert foreground.permission == launch.permission


def test_never_launch_policy_leaves_recovery_available(tmp_path):
    tools = manager(tmp_path, tools={"bash": {"permission": "never"}})
    assert tools.get_tool_config("bash_start").permission is ToolPermission.NEVER
    result = tools.get("bash_start").resolve_permission(
        BashStartArgs(command="echo ok")
    )
    assert result is not None and result.permission is ToolPermission.NEVER
    for name in ("bash_read", "bash_stop", "bash_list"):
        assert tools.get_tool_config(name).permission is ToolPermission.ALWAYS


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.asyncio
async def test_start_read_later_context_stop_list_and_persisted_results(
    tmp_path, child
):
    tools = manager(tmp_path)
    registry = BackgroundJobRegistry()
    root = registry.root_port()
    port = root.borrow() if child else root

    # A terminal port must never be used for managed jobs.
    class NoTerminal:
        def __getattr__(self, name):
            raise AssertionError(f"Terminal delegation attempted: {name}")

    ctx = InvokeContext(
        tool_call_id="first", background_jobs=port, tool_io=cast(Any, NoTerminal())
    )
    try:
        started = await run(
            tools.get("bash_start"),
            BashStartArgs(command="printf 'ready\\n'; sleep 30"),
            ctx,
        )
        assert started.next_cursor == 0
        job_id = started.job.job_id
        page = await run(
            tools.get("bash_read"), BashReadArgs(job_id=job_id, wait_seconds=2), ctx
        )
        assert page.next_cursor > started.next_cursor
        assert "ready\n" in "".join(record.text for record in page.records)
        assert page.job.state == "running" and page.job.exit_code is None
        assert sum(len(record.text.encode()) for record in page.records) <= 4096
        later = InvokeContext(tool_call_id="later", background_jobs=port)
        listed = await run(tools.get("bash_list"), BashListArgs(), later)
        assert [job.job_id for job in listed.jobs] == [job_id]
        assert set(listed.model_dump()) == {"jobs"}
        stopped = await run(tools.get("bash_stop"), BashStopArgs(job_id=job_id), later)
        assert not stopped.already_finished
        assert stopped.job.output_complete and stopped.job.state == "stopped"
        again = await run(tools.get("bash_stop"), BashStopArgs(job_id=job_id), later)
        assert again.already_finished
        final = await run(
            tools.get("bash_read"),
            BashReadArgs(job_id=job_id, cursor=page.next_cursor),
            later,
        )
        assert "ready" in "".join(
            record.text for record in (*page.records, *final.records)
        )
        listed = await run(
            tools.get("bash_list"), BashListArgs(include_finished=True), later
        )
        for tool_class, result in zip(
            TOOLS, (started, final, stopped, listed), strict=True
        ):
            # model_dump is also the post_tool hook payload; JSON is the persisted result.
            payload = result.model_dump(mode="json")
            assert type(result).model_validate_json(result.model_dump_json()) == result
            assert tool_class.project_result(result) == payload
        assert root.list(BashListArgs(include_finished=True)).jobs[0].job_id == job_id
        if child:
            sibling = InvokeContext(
                tool_call_id="sibling", background_jobs=root.borrow()
            )
            with pytest.raises(ToolError):
                await run(tools.get("bash_read"), BashReadArgs(job_id=job_id), sibling)
    finally:
        await registry.aclose()


@pytest.mark.parametrize(
    "command",
    [
        r"printf synth\etic-credential",
        "printf 'synthetic-'credential",
        r"printf $'synthetic\x2dcredential'",
    ],
)
@pytest.mark.asyncio
async def test_shell_source_sanitized_in_all_tool_summaries(tmp_path, command):
    tools = manager(tmp_path)
    registry = BackgroundJobRegistry()
    ctx = InvokeContext(tool_call_id="redaction", background_jobs=registry.root_port())
    try:
        with bind_policy(
            ScrubPolicy(redaction_credentials=(("TEST_TOKEN", "synthetic-credential"),))
        ):
            start = await run(
                tools.get("bash_start"), BashStartArgs(command=command), ctx
            )
            job_id = start.job.job_id
            read = await run(
                tools.get("bash_read"), BashReadArgs(job_id=job_id, wait_seconds=2), ctx
            )
            stop = await run(tools.get("bash_stop"), BashStopArgs(job_id=job_id), ctx)
            listed = await run(
                tools.get("bash_list"), BashListArgs(include_finished=True), ctx
            )
            for result in (start, read, stop, listed):
                summary = result.jobs[0] if hasattr(result, "jobs") else result.job
                assert "[REDACTED]" in summary.command
                assert "credential" not in result.model_dump_json()
    finally:
        await registry.aclose()


@pytest.mark.asyncio
async def test_foreground_still_returns_separate_output(tmp_path):
    result = await run(manager(tmp_path).get("bash"), BashArgs(command="printf ok"))
    assert result.stdout == "ok" and result.stderr == ""
    assert result.exit_code == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("child", [False, True])
async def test_loop_invocation_hooks_persistence_and_generic_projection(
    tmp_path, monkeypatch, child
):
    root = build_test_agent_loop(cwd=tmp_path)
    assert root.background_jobs is not None
    loop = (
        build_test_agent_loop(
            cwd=tmp_path,
            is_subagent=True,
            inherited_workspace=root.tool_manager.workspace,
            background_jobs=root.background_jobs.borrow(),
        )
        if child
        else root
    )
    hook_payloads = []
    original = loop._run_post_tool_hooks

    async def collect(call, **kwargs):
        hook_payloads.append((call.tool_name, kwargs.get("tool_output")))
        async for event in original(call, **kwargs):
            yield event

    monkeypatch.setattr(loop, "_run_post_tool_hooks", collect)

    async def invoke(cls, args) -> Any:
        call = ResolvedToolCall(
            tool_name=cls.get_name(),
            tool_class=cls,
            validated_args=args,
            call_id=f"call-{len(hook_payloads)}",
        )
        events = [event async for event in loop._process_one_tool_call(call)]
        event = next(event for event in events if isinstance(event, ToolResultEvent))
        assert event.error is None and not event.skipped and event.result is not None
        assert hook_payloads[-1] == (
            cls.get_name(),
            event.result.model_dump(mode="json"),
        )
        message = [message for message in loop.messages if message.role is Role.tool][
            -1
        ]
        assert message.tool_result is not None
        stored = type(message.tool_result).model_validate_json(
            message.tool_result.model_dump_json()
        )
        assert stored.output == event.result.model_dump(mode="json")
        presentation = ToolUIDataAdapter(cls).get_result_presentation(event)
        projected_event = event.model_copy(update={"presentation": presentation})
        live = project_effect_output(projected_event)
        replay = project_effect_output_value(ToolEffectKind.TOOL, stored.output)
        assert live == replay == event.result.model_dump(mode="json")
        return event.result

    try:
        started = await invoke(BashStart, BashStartArgs(command="sleep 30"))
        await invoke(
            BashRead, BashReadArgs(job_id=started.job.job_id, wait_seconds=0.01)
        )
        # A distinct invocation context on a later call retains trusted ownership.
        await invoke(BashList, BashListArgs())
        stopped = await invoke(BashStop, BashStopArgs(job_id=started.job.job_id))
        assert stopped.job.state == "stopped"
        assert [name for name, _ in hook_payloads] == [
            "bash_start",
            "bash_read",
            "bash_list",
            "bash_stop",
        ]
    finally:
        if child:
            await loop.aclose()
        await root.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "command"),
    [
        ({"permission": "never"}, "sleep 30"),
        ({"denylist": ["sleep"]}, "sleep 30"),
        # The shipped default denylist denies these without any override.
        ({}, "git push"),
        ({}, "git -C . push"),
        ({}, "git switch --discard-changes"),
        ({}, "git reflog expire --all"),
    ],
)
async def test_loop_policy_denial_does_not_launch(tmp_path, policy, command):
    loop = build_test_agent_loop(
        cwd=tmp_path, config=build_test_vibe_config(tools={"bash": policy})
    )
    try:
        call = ResolvedToolCall(
            tool_name="bash_start",
            tool_class=BashStart,
            validated_args=BashStartArgs(command=command),
            call_id="denied-start",
        )
        events = [event async for event in loop._process_one_tool_call(call)]
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        assert result.skipped
        assert (
            loop.background_jobs is not None and loop.background_jobs.active_count == 0
        )
    finally:
        await loop.aclose()
