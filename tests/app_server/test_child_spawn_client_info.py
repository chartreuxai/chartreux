"""End-to-end regression: a process-backed child opens with a real identity.

Pre-F4 (B-spawn): every producer of a process child's spawn payload omitted
``client_info``/``client_capabilities``, while the child's
``HarnessChildLoopFactory.open`` validates ``ClientInfo`` from the payload —
so every real ``task`` spawn failed deterministically with a ClientInfo
ValidationError. The unit seam (test_child_transcripts.py) injected a payload
by hand, which is why CI never saw it.

F6 (E2) adds the same regression class for the parent workspace: the payload
producers omitted ``inherited_workspace`` while the child's open constructs
its loop as a subagent, which is illegal without it
("Subagent construction requires inherited_workspace"). The child double
now RECORDS the decoded open request and honors its subagent shape instead
of silently building a root loop.

F11 (Q2) adds the same class for the PARENT POLICY: the payload producers
omitted the restrictions/mode-restrictions/root-grants/agent-config keys
while the child's open REQUIRES them, so a restricted, budget-capped parent
spawned an unrestricted, unbudgeted child. The child double now builds its
loop through the REAL ``_RootRuntimeBlueprint`` (only the HarnessProcess
seam is replaced, exactly as before) and records the BUILT loop's policy, so
the proof lands on the constructed ``AgentRuntimePolicy`` -- not just the
payload.

This node spawns a REAL child OS process through ``ProcessSubagentSpawn``,
its spec published by a real ``AgentChildHost.publish`` carrying the root's
validated identity, and the child runs the REAL ``HarnessChildLoopFactory``
(seam: only its ``HarnessProcess`` is replaced, exactly the seam the unit
pin drives) served by the real ``ChildAgentServer``. ``publish`` succeeding
IS ``open_subagent`` succeeding: the spawn adapter surfaces any open
failure typed, so the child's own ClientInfo, workspace and policy decoding
are inside the proof.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from chartreux.app_server._connection_protocol import ClientCapabilities, ClientInfo
from chartreux.app_server._subagent_transcript import (
    AGENT_CONFIG_PAYLOAD_KEY,
    CLIENT_CAPABILITIES_PAYLOAD_KEY,
    CLIENT_INFO_PAYLOAD_KEY,
    INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY,
    INHERITED_RESTRICTIONS_PAYLOAD_KEY,
    INHERITED_ROOT_GRANTS_PAYLOAD_KEY,
    INHERITED_WORKSPACE_PAYLOAD_KEY,
    decode_inherited_root_grants,
    decode_inherited_workspace,
    decode_source_restrictions,
    encode_inherited_workspace,
)
from chartreux.app_server.child_agent_process import ProcessSubagentSpawn
from chartreux.app_server.protocol import AgentConfig
from chartreux.app_server.root_agent import AgentChildHost
from chartreux.app_server.transport import STDIO_TRANSPORT, TRANSPORT_ENV_VAR
from chartreux.core.config._restrictions import (
    PolicySourceIdentity,
    SourceRestrictions,
    ToolRestriction,
)
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.differential import egress
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CLIENT_INFO = ClientInfo(name="chartreux-tests", version="1.0")
_CLIENT_CAPABILITIES = ClientCapabilities(callback_kinds=["user_input"])

_PARENT_SOURCE_RESTRICTION = SourceRestrictions(
    layer_name="project policy",
    locator="AGENTS.md",
    kind="source",
    content_fingerprint="fp-e2e-source",
    store_fingerprint="store-fp-e2e-source",
    tools=(
        ToolRestriction(
            tool_name="bash", denied=True, denylist=("rm -rf",), sensitive_patterns=()
        ),
    ),
    identity=PolicySourceIdentity(
        owner=UUID("12345678-1234-5678-1234-567812345678"),
        source=UUID("87654321-4321-8765-4321-876543218765"),
    ),
    authorized_roots=(),
    dangerous_command_grants=(),
    remote_exec_grants=(),
    replaces_roots=False,
)

_PARENT_MODE_RESTRICTION = SourceRestrictions(
    layer_name="session mode",
    locator="review",
    kind="mode",
    content_fingerprint="fp-e2e-mode",
    store_fingerprint=None,
    tools=(
        ToolRestriction(
            tool_name="file_write", denied=True, denylist=(), sensitive_patterns=()
        ),
    ),
    identity=None,
    authorized_roots=(),
    dangerous_command_grants=(),
    remote_exec_grants=(),
    replaces_roots=False,
)

# Explicit parent budget (wins over the child's [subagents] default) and the
# config default the child's own config carries for the UNSET one.
_PARENT_RUN_MAX_PRICE = 1.5
_CHILD_DEFAULT_RUN_MAX_SESSION_TOKENS = 1024

_CHILD_AGENT_SCRIPT = """\
import asyncio
import json
import os

from chartreux.app_server import _runtime
from chartreux.app_server._subagent_transcript import (
    encode_inherited_workspace,
    encode_source_restrictions,
)
from chartreux.app_server.child_agent import (
    HarnessChildLoopFactory,
    serve_child_agent,
)
from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    init_harness_files_manager,
)
from chartreux.core.config.models import SubagentsConfig
from chartreux.core.hooks.models import HookConfigResult
from chartreux.utils.cache_store import FileSystemCacheStore
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

init_harness_files_manager("user", "project")


class _Process:
    # Only the HarnessProcess seam is replaced: the REAL factory code in
    # front of it (spec decode, root-dir fail-closed, AgentConfig and
    # ClientInfo validation, the inherited-workspace and F11 policy decodes,
    # the child transcript logging the request names, persist_empty_session,
    # handoff adoption) all runs here, on the spec a real publisher built.
    # The loop the seam returns is built through the REAL
    # _RootRuntimeBlueprint -- the process child's loop constructor -- so
    # the recorded policy is the BUILT loop's AgentRuntimePolicy, not the
    # bare request.
    async def open_root(self, request):
        if not request.is_subagent or request.inherited_workspace is None:
            raise RuntimeError(
                "the child open request carries no subagent shape or "
                "no inherited workspace"
            )
        blueprint = _runtime._RootRuntimeBlueprint(
            config_orchestrator=FakeConfigOrchestrator(
                build_test_vibe_config(
                    session_logging=request.session_logging,
                    subagents=SubagentsConfig(
                        run_max_price=0.05,
                        run_max_session_tokens=1024,
                    ),
                )
            ),
            harness_files=HarnessFilesManager(),
            options=request.options,
            client_info=request.client_info,
            client_capabilities=request.client_capabilities,
            hook_config_result=HookConfigResult(hooks=[], issues=[]),
            cache_store=FileSystemCacheStore(
                os.environ["F11_CHILD_CACHE"]
            ),
        )
        loop = blueprint.build(
            parent_session_id=request.parent_session_id,
            is_subagent=True,
            inherited_workspace=request.inherited_workspace,
            inherited_restrictions=request.inherited_restrictions,
            inherited_mode_restrictions=request.inherited_mode_restrictions,
            inherited_root_grants=request.inherited_root_grants,
        )
        policy = loop.runtime_policy
        record = {
            "workspace": encode_inherited_workspace(
                request.inherited_workspace
            ),
            "restrictions": encode_source_restrictions(
                policy.inherited_restrictions
            ),
            "mode_restrictions": encode_source_restrictions(
                policy.inherited_mode_restrictions
            ),
            "root_grants": [str(p) for p in loop._dispatch_root_grants],
            "run_max_price": policy.run_max_price,
            "run_max_session_tokens": policy.run_max_session_tokens,
            "options": request.options.model_dump(mode="json"),
        }
        with open(os.environ["F11_CHILD_POLICY_RECORD"], "w") as fh:
            fh.write(json.dumps(record))
        return loop

    async def close(self):
        pass


factory = HarnessChildLoopFactory()
factory._process = _Process()
asyncio.run(serve_child_agent(factory=factory, announce_ready=bootstrap_requested()))
"""


def _child_command() -> list[str]:
    return ["python", "-c", _CHILD_AGENT_SCRIPT]


def _child_env(guard_dir: Path, cache_path: Path, record_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env[TRANSPORT_ENV_VAR] = STDIO_TRANSPORT
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["F11_CHILD_CACHE"] = str(cache_path)
    env["F11_CHILD_POLICY_RECORD"] = str(record_path)
    paths = [str(_REPO_ROOT), str(guard_dir)]
    inherited = env.get("PYTHONPATH")
    if inherited:
        paths.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.update(egress.child_environment(egress.MODE_ALL))
    return env


class _PublishingSpawnAdapter:
    """Hands each spec to the real spawn and records what it carried."""

    agent_id = "recorded"

    def __init__(self, real: ProcessSubagentSpawn) -> None:
        self._real = real
        self.specs: list[Any] = []

    async def spawn(self, spec: Any) -> Any:
        self.specs.append(spec)
        return await self._real.spawn(spec)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_process_child_opens_with_the_payload_client_identity(
    tmp_path: Path,
) -> None:
    """Spawn -> subagent/open succeeds only with a populated client_info."""
    record_path = tmp_path / "child-policy-record.json"
    cache_path = tmp_path / "child-cache.toml"
    with egress.bootstrap_directory() as guard_dir:
        sessions = tmp_path / "sessions"
        # The parent is restricted and budget-capped: a pre-F11 child opened
        # with an EMPTY policy (the payload carried none of it); the F11
        # child must open with the parent's restrictions, mode restrictions,
        # dispatch root grants and budgets.
        loop = build_test_agent_loop(
            config=build_test_vibe_config(
                session_logging=SessionLoggingConfig(
                    enabled=True, save_dir=str(sessions), session_prefix="session"
                )
            ),
            backend=FakeBackend([[mock_llm_chunk(content="ack")]]),
            inherited_restrictions=(_PARENT_SOURCE_RESTRICTION,),
            inherited_mode_restrictions=(_PARENT_MODE_RESTRICTION,),
            run_max_price=_PARENT_RUN_MAX_PRICE,
        )
        expected_workspace = encode_inherited_workspace(loop.tool_manager.workspace)
        root_session_dir = loop.session_logger.session_dir
        assert root_session_dir is not None
        real_spawn = ProcessSubagentSpawn(
            env=_child_env(guard_dir, cache_path, record_path),
            cwd=str(_REPO_ROOT),
            ready_timeout=90,
            request_timeout=90,
            command=_child_command(),
        )
        port = _PublishingSpawnAdapter(real_spawn)
        host = AgentChildHost(
            loop,
            spawn=port,
            client_info=_CLIENT_INFO,
            client_capabilities=_CLIENT_CAPABILITIES,
        )
        handle = None
        try:
            handle = await host.publish(
                TaskArgs(task="Inspect the project", identity="coder"),
                InvokeContext(tool_call_id="task-1", session_id=loop.session_id),
                attempt_id="attempt-spawn-e2e-1",
                retained=False,
            )
        finally:
            if handle is not None:
                await handle.spawned.close()
            await loop.aclose()

    # The spawn handshake answered: the child's REAL ClientInfo validation
    # accepted the identity the payload carried (pre-F4 this publish raises
    # a ChildPublicationError wrapping the child's ClientInfo ValidationError).
    assert handle is not None
    assert handle.session_id

    payload = port.specs[0].payload
    info = payload[CLIENT_INFO_PAYLOAD_KEY]
    assert info["name"] == "chartreux-tests"
    assert ClientInfo.model_validate(info) == _CLIENT_INFO
    assert payload[CLIENT_CAPABILITIES_PAYLOAD_KEY] == _CLIENT_CAPABILITIES.model_dump(
        mode="json"
    )

    # E2/F6: the payload carries the parent workspace, and the child's REAL
    # open decoded exactly it (pre-F6 this publish raised a ChildPublicationError
    # wrapping RuntimeConfigurationError: a subagent loop was constructed
    # without inherited_workspace).
    assert decode_inherited_workspace(payload[INHERITED_WORKSPACE_PAYLOAD_KEY]) == (
        loop.tool_manager.workspace
    )

    # F11/Q2: the restricted, budget-capped parent's policy rides the payload,
    # and the expectations are the values the in-process route's
    # child_runtime_policy hands an in-process child of this loop.
    expected = loop.child_runtime_policy
    assert (
        decode_source_restrictions(payload[INHERITED_RESTRICTIONS_PAYLOAD_KEY])
        == expected.inherited_restrictions
        == (_PARENT_SOURCE_RESTRICTION,)
    )
    assert (
        decode_source_restrictions(payload[INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY])
        == expected.inherited_mode_restrictions
        == (_PARENT_MODE_RESTRICTION,)
    )
    assert (
        decode_inherited_root_grants(payload[INHERITED_ROOT_GRANTS_PAYLOAD_KEY])
        == expected.inherited_root_grants
        == ()
    )
    options = AgentConfig.model_validate(payload[AGENT_CONFIG_PAYLOAD_KEY])
    assert options.run_max_price == _PARENT_RUN_MAX_PRICE

    record = json.loads(record_path.read_text())
    assert record["workspace"] == expected_workspace
    # The child loop was BUILT with the parent's policy, not just handed it
    # (the recorded values come off the constructed loop's AgentRuntimePolicy).
    assert decode_source_restrictions(record["restrictions"]) == (
        _PARENT_SOURCE_RESTRICTION,
    )
    assert decode_source_restrictions(record["mode_restrictions"]) == (
        _PARENT_MODE_RESTRICTION,
    )
    assert record["root_grants"] == []
    # B2c, process-spawn counterpart of _create_like: the explicit parent
    # value wins, and the child's own [subagents] config default fills the
    # UNSET one (the run-budget middleware arms on these exact loop fields).
    assert record["run_max_price"] == _PARENT_RUN_MAX_PRICE
    assert record["run_max_session_tokens"] == _CHILD_DEFAULT_RUN_MAX_SESSION_TOKENS
    assert AgentConfig.model_validate(record["options"]) == options

    # A real child process ran: it persisted its own transcript directory
    # under the root session's agents/ subtree, named by its agent prefix.
    child_dirs = [entry for entry in (Path(root_session_dir) / "agents").iterdir()]
    assert len(child_dirs) == 1
    assert child_dirs[0].name.startswith("coder")
    assert handle.child_dir is not None
    assert Path(handle.child_dir) == child_dirs[0]
