"""S4a: remote-exec destination grants — schema, matching, and lifecycle.

The grant model is destination-keyed: authority is (scope, operation, exact
(user, host, port)). These tests pin canonicalization and matching, the
half-prompted lifecycle (deny/once/session/persist-project), the honest-scope
wording, operation distinctness in both directions, and the child semantics
pinned at the cited sites: session grants do not propagate, persisted grants
authorize parent and child, and children never prompt.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import cast

import pytest

from chartreux.core.agent_loop._dangerous_command_grants import (
    DANGEROUS_COMMAND_DECLINED_FEEDBACK,
    DANGEROUS_COMMAND_DENY_LABEL,
    DANGEROUS_COMMAND_ONCE_LABEL,
    DANGEROUS_COMMAND_PROJECT_LABEL,
    DANGEROUS_COMMAND_SESSION_LABEL,
    REMOTE_EXEC_SCOPE_STATEMENT,
    DangerousCommandGrantStore,
    RemoteExecGrantRequirement,
    dangerous_command_request,
    parse_dangerous_command_result,
)
from chartreux.core.agent_loop._request_broker import InteractionRequestBroker
from chartreux.core.agent_loop._root_grants import SessionRootGrants
from chartreux.core.agent_loop._tool_decisions import ToolDecisionCoordinator
from chartreux.core.config._dangerous_command_persistence import SavedGrantsRead
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config._root_persistence import SavedRootsRead
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.dangerous_commands import (
    OPERATION_REGISTRY,
    REMOTE_EXEC_OPERATION_ID,
    DangerousCommandAuthority,
    DangerousCommandProposal,
    RemoteDestination,
    RepositoryScope,
    authority_admits,
    canonical_remote_destination,
)
from chartreux.core.tools.base import BaseTool, BaseToolState, ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    resolve_repository_scope,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.models import PermissionContext
from chartreux.questions import UserAnswer, UserQuestionResult
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


def _destination(value: str) -> RemoteDestination:
    parsed = canonical_remote_destination(value)
    assert parsed is not None
    return parsed


def _scope(path: Path) -> RepositoryScope:
    scope = resolve_repository_scope(path, GitRepositoryRelocation())
    assert scope is not None
    return scope


def _repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def _remote_exec_proposal(
    project: Path, scope: RepositoryScope, destination: str
) -> DangerousCommandProposal:
    parsed = _destination(destination)
    return DangerousCommandProposal(
        command=("ssh", parsed.key, "uptime"),
        project=project,
        scope=scope,
        operations=(REMOTE_EXEC_OPERATION_ID,),
        consequence=OPERATION_REGISTRY[REMOTE_EXEC_OPERATION_ID].consequence,
        destination=parsed,
    )


def _bash(project: Path, *authorities: DangerousCommandAuthority) -> Bash:
    tool = Bash(
        config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=project
    )
    tool.dangerous_command_authorities_getter = lambda: tuple(authorities)
    return tool


class TestCanonicalDestination:
    """Canonical spelling: explicit user, lowercased host, port defaulting to 22."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("deploy@example.com", RemoteDestination("deploy", "example.com", 22)),
            ("deploy@example.com:22", RemoteDestination("deploy", "example.com", 22)),
            (
                "deploy@example.com:2222",
                RemoteDestination("deploy", "example.com", 2222),
            ),
            # The host is case-insensitive; the user is not (POSIX accounts).
            ("Deploy@Example.COM", RemoteDestination("Deploy", "example.com", 22)),
            (
                "deploy@[2001:db8:0:0::1]:2200",
                RemoteDestination("deploy", "2001:db8::1", 2200),
            ),
            ("deploy@192.0.2.7", RemoteDestination("deploy", "192.0.2.7", 22)),
        ],
    )
    def test_canonical_forms(self, value: str, expected: RemoteDestination):
        assert canonical_remote_destination(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            # An explicit user is mandatory.
            "example.com:22",
            "example.com",
            "@example.com",
            "deploy@",
            # Wildcards, ranges, and alias groups never parse.
            "deploy@*",
            "deploy@*.example.com",
            "deploy@host?.example.com",
            "deploy@[2001:db8::1-2001:db8::9]",
            "deploy@a.example.com,b.example.com",
            "deploy@host name",
            # Port range and syntax.
            "deploy@example.com:0",
            "deploy@example.com:65536",
            "deploy@example.com:ssh",
            # Not an IPv6 literal in brackets.
            "deploy@[not-an-address]:22",
        ],
    )
    def test_invalid_forms_are_rejected(self, value: str):
        assert canonical_remote_destination(value) is None

    def test_key_round_trips_through_the_canonicalizer(self):
        destination = _destination("Deploy@Example.COM:2222")
        assert destination.key == "Deploy@example.com:2222"
        assert str(destination) == destination.key
        assert canonical_remote_destination(destination.key) == destination


class TestExactMatching:
    """Grants match the exact canonical tuple; every near miss is denied."""

    def test_exact_destination_is_admitted(self, tmp_path: Path):
        scope = _scope(_repo(tmp_path / "repo"))
        destination = _destination("deploy@example.com:22")
        authority = DangerousCommandAuthority(
            operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
            scope=scope,
            source="runtime_grant",
            destination=destination,
        )
        assert authority_admits(
            authority, REMOTE_EXEC_OPERATION_ID, scope, destination=destination
        )

    @pytest.mark.parametrize(
        "other",
        [
            "deploy@example.com:2222",  # another port
            "deploy@other.example.com:22",  # another host
            "ops@example.com:22",  # another user
            "Deploy@example.com:22",  # user case is significant
        ],
    )
    def test_near_miss_destinations_are_denied(self, tmp_path: Path, other: str):
        scope = _scope(_repo(tmp_path / "repo"))
        authority = DangerousCommandAuthority(
            operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
            scope=scope,
            source="runtime_grant",
            destination=_destination("deploy@example.com"),
        )
        assert not authority_admits(
            authority, REMOTE_EXEC_OPERATION_ID, scope, destination=_destination(other)
        )

    def test_default_port_spelling_matches_the_explicit_22(self, tmp_path: Path):
        scope = _scope(_repo(tmp_path / "repo"))
        authority = DangerousCommandAuthority(
            operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
            scope=scope,
            source="protected_config",
            destination=_destination("deploy@example.com"),
        )
        assert authority_admits(
            authority,
            REMOTE_EXEC_OPERATION_ID,
            scope,
            destination=_destination("deploy@example.com:22"),
        )

    def test_another_scope_is_denied(self, tmp_path: Path):
        scope = _scope(_repo(tmp_path / "repo"))
        other_scope = _scope(_repo(tmp_path / "other"))
        destination = _destination("deploy@example.com")
        authority = DangerousCommandAuthority(
            operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
            scope=scope,
            source="runtime_grant",
            destination=destination,
        )
        assert not authority_admits(
            authority, REMOTE_EXEC_OPERATION_ID, other_scope, destination=destination
        )


class TestOperationDistinctness:
    """Egress and remote-exec authority never cross-authorize, either way."""

    def test_egress_grant_does_not_admit_a_remote_command(self, tmp_path: Path):
        scope = _scope(_repo(tmp_path / "repo"))
        egress = DangerousCommandAuthority(
            operations=frozenset({"network.egress"}),
            scope=scope,
            source="runtime_grant",
        )
        destination = _destination("deploy@example.com")
        assert not authority_admits(
            egress, REMOTE_EXEC_OPERATION_ID, scope, destination=destination
        )
        assert not authority_admits(egress, REMOTE_EXEC_OPERATION_ID, scope)

    def test_remote_exec_grant_does_not_admit_egress_or_publication(
        self, tmp_path: Path
    ):
        scope = _scope(_repo(tmp_path / "repo"))
        destination = _destination("deploy@example.com")
        remote_exec = DangerousCommandAuthority(
            operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
            scope=scope,
            source="runtime_grant",
            destination=destination,
        )
        assert not authority_admits(remote_exec, "network.egress", scope)
        assert not authority_admits(remote_exec, "git.push", scope)
        # A destination authority admits only its own exact destination.
        assert not authority_admits(
            remote_exec,
            REMOTE_EXEC_OPERATION_ID,
            scope,
            destination=_destination("deploy@other.example.com"),
        )

    @pytest.mark.parametrize(
        "granted",
        [frozenset({"network.egress"}), frozenset({REMOTE_EXEC_OPERATION_ID})],
    )
    def test_ssh_stays_denied_at_the_resolver_until_s4b(
        self, tmp_path: Path, granted: frozenset[str]
    ):
        repo = _repo(tmp_path / "repo")
        scope = _scope(repo)
        destination = _destination("deploy@example.com")
        authority = DangerousCommandAuthority(
            operations=granted,
            scope=scope,
            source="runtime_grant",
            destination=(destination if REMOTE_EXEC_OPERATION_ID in granted else None),
        )
        tool = _bash(repo, authority)
        result = tool.resolve_permission(
            BashArgs(command="ssh deploy@example.com uptime")
        )
        assert result is not None
        assert result.permission.value == "never"

    def test_egress_grant_admits_egress_but_remote_exec_grant_does_not(
        self, tmp_path: Path
    ):
        repo = _repo(tmp_path / "repo")
        scope = _scope(repo)
        destination = _destination("deploy@example.com")
        remote_exec = DangerousCommandAuthority(
            operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
            scope=scope,
            source="runtime_grant",
            destination=destination,
        )
        tool = _bash(repo, remote_exec)
        denied = tool.resolve_permission(
            BashArgs(command="curl https://example.invalid/")
        )
        assert denied is not None
        assert "network.egress" in (denied.reason or "")

        egress = DangerousCommandAuthority(
            operations=frozenset({"network.egress"}),
            scope=scope,
            source="runtime_grant",
        )
        tool = _bash(repo, egress)
        admitted = tool.resolve_permission(
            BashArgs(command="curl https://example.invalid/")
        )
        assert admitted is None or admitted.permission.value == "always"
        assert admitted is None or admitted.dangerous_command_proposal is None

    def test_remote_exec_grant_does_not_admit_a_push(self, tmp_path: Path):
        repo = _repo(tmp_path / "repo")
        authority = DangerousCommandAuthority(
            operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
            scope=_scope(repo),
            source="runtime_grant",
            destination=_destination("deploy@example.com"),
        )
        tool = _bash(repo, authority)
        denied = tool.resolve_permission(BashArgs(command="git push"))
        assert denied is not None
        assert "git.push" in (denied.reason or "")


class TestPrompt:
    """The approval question offers four choices and states the honest scope."""

    def test_remote_exec_question_offers_all_four_choices(self, tmp_path: Path):
        repo = _repo(tmp_path / "repo")
        request = dangerous_command_request(
            _remote_exec_proposal(repo, _scope(repo), "deploy@example.com")
        )
        question = request.questions[0]
        assert [option.label for option in question.options] == [
            DANGEROUS_COMMAND_DENY_LABEL,
            DANGEROUS_COMMAND_ONCE_LABEL,
            DANGEROUS_COMMAND_SESSION_LABEL,
            DANGEROUS_COMMAND_PROJECT_LABEL,
        ]
        assert question.hide_other is True
        assert len(question.header) <= 20

    def test_honest_scope_statement_is_verbatim(self, tmp_path: Path):
        repo = _repo(tmp_path / "repo")
        request = dangerous_command_request(
            _remote_exec_proposal(repo, _scope(repo), "deploy@example.com")
        )
        question = request.questions[0]
        assert REMOTE_EXEC_SCOPE_STATEMENT == (
            "Remote commands and their effects are outside chartreux's local "
            "guards — trusting the destination, not approving a contained "
            "payload."
        )
        assert REMOTE_EXEC_SCOPE_STATEMENT in question.question

    def test_question_names_destination_command_and_project(self, tmp_path: Path):
        repo = _repo(tmp_path / "repo")
        proposal = _remote_exec_proposal(repo, _scope(repo), "deploy@example.com:2222")
        request = dangerous_command_request(proposal)
        question = request.questions[0]
        assert "ssh deploy@example.com:2222 uptime" in question.question
        assert "Destination: deploy@example.com:2222" in question.question
        assert str(repo) in question.question
        assert REMOTE_EXEC_OPERATION_ID in question.question
        assert proposal.consequence in question.question
        # The session choice names the exact destination, not "these operations".
        assert "deploy@example.com:2222" in request.questions[0].options[2].description
        assert "destination" in (request.footer_note or "")

    def test_persistent_choice_is_offered_for_remote_exec(self, tmp_path: Path):
        repo = _repo(tmp_path / "repo")
        proposal = _remote_exec_proposal(repo, _scope(repo), "deploy@example.com")
        labels = {
            option.label
            for option in dangerous_command_request(proposal).questions[0].options
        }
        assert DANGEROUS_COMMAND_PROJECT_LABEL in labels


class TestParseResult:
    """Every label maps to its grant choice through the shared parser."""

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            (DANGEROUS_COMMAND_DENY_LABEL, "deny"),
            (DANGEROUS_COMMAND_ONCE_LABEL, "allow_once"),
            (DANGEROUS_COMMAND_SESSION_LABEL, "allow_session"),
            (DANGEROUS_COMMAND_PROJECT_LABEL, "allow_project"),
        ],
    )
    def test_remote_exec_answers(self, label: str, expected: str):
        result = UserQuestionResult(answers=[UserAnswer(question="q", answer=label)])
        assert (
            parse_dangerous_command_result(result, (REMOTE_EXEC_OPERATION_ID,))
            == expected
        )


class _FixedBroker:
    """A broker that answers every prompt with one fixed label and records it."""

    def __init__(self, label: str) -> None:
        self._label = label
        self.requests: list[object] = []

    async def request_user_input(self, args, tool_call_id, *, origin):
        self.requests.append(args)
        return UserQuestionResult(
            answers=[UserAnswer(question="q", answer=self._label)]
        )


class _AllowingTool:
    def resolve_permission(self, args):
        return None


class _GrantStorePort:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, str, str]] = []

    async def save_dangerous_command_grant(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        raise AssertionError("no scope-keyed save on the remote-exec path")

    async def save_remote_exec_grant(
        self, session_id: str, project: Path, destination: str, expected_revision: str
    ) -> ConfigSaveResult:
        self.calls.append((session_id, project, destination, expected_revision))
        return ConfigSaveResult("user", "saved", "unchanged", "rev-2")


def _coordinator(
    *,
    store: DangerousCommandGrantStore,
    broker: object,
    is_subagent: bool = False,
    port: object | None = None,
) -> ToolDecisionCoordinator:
    async def read_saved(_orchestrator, _project) -> SavedGrantsRead:
        return SavedGrantsRead(operations=(), destinations=(), user_revision="rev-1")

    async def never_read_roots(*_args: object) -> SavedRootsRead:
        raise AssertionError("no saved-roots read on this path")

    async def never_apply_root(*_args: object) -> None:
        raise AssertionError("no root grant on this path")

    manager = SimpleNamespace(
        scratchpad_roots=frozenset(),
        _invalidate_dangerous_command_authority=lambda: None,
    )
    return ToolDecisionCoordinator(
        grants_store=store,
        request_broker=cast(InteractionRequestBroker, broker),
        user_input_capability_getter=lambda: True,
        is_subagent_getter=lambda: is_subagent,
        grant_port_getter=lambda: cast(object, port),  # type: ignore[arg-type,return-value]
        tool_manager_getter=lambda: cast(ToolManager, manager),
        session_id_getter=lambda: "session",
        config_orchestrator_getter=lambda: FakeConfigOrchestrator(
            build_test_vibe_config()
        ),
        read_saved_grants=read_saved,
        root_grants_store=SessionRootGrants(),
        root_grant_port_getter=lambda: None,
        cwd_getter=lambda: Path("/tmp"),
        read_saved_roots=never_read_roots,
        apply_approved_root_grant=never_apply_root,
    )


def _args() -> BashArgs:
    """A shell-call payload; these lifecycle tests never inspect it."""
    return BashArgs(command="ssh deploy@example.com")


def _ctx(proposal: DangerousCommandProposal, reason: str = "needs authority"):
    return PermissionContext(
        permission=ToolPermission.NEVER,
        reason=reason,
        denial_kind="dangerous_command",
        dangerous_command_proposal=proposal,
    )


class TestLifecycle:
    """Deny / once / session / persist-project through the real coordinator."""

    @pytest.mark.asyncio
    async def test_deny_records_the_decision_and_never_reprompts(self, tmp_path: Path):
        repo = _repo(tmp_path / "repo")
        store = DangerousCommandGrantStore()
        broker = _FixedBroker(DANGEROUS_COMMAND_DENY_LABEL)
        coordinator = _coordinator(store=store, broker=broker)
        proposal = _remote_exec_proposal(repo, _scope(repo), "deploy@example.com")

        first = await coordinator.resolve_remote_exec_grant(
            cast(BaseTool, _AllowingTool()), _args(), _ctx(proposal), "bash", "c1"
        )
        assert first is not None
        assert first.feedback == DANGEROUS_COMMAND_DECLINED_FEEDBACK
        second = await coordinator.resolve_remote_exec_grant(
            cast(BaseTool, _AllowingTool()), _args(), _ctx(proposal), "bash", "c2"
        )
        assert second is not None
        assert second.feedback == DANGEROUS_COMMAND_DECLINED_FEEDBACK
        assert len(broker.requests) == 1
        assert store.dangerous_command_authorities() == ()

    @pytest.mark.asyncio
    async def test_allow_once_installs_then_consumes_a_destination_permit(
        self, tmp_path: Path
    ):
        repo = _repo(tmp_path / "repo")
        scope = _scope(repo)
        store = DangerousCommandGrantStore()
        coordinator = _coordinator(
            store=store, broker=_FixedBroker(DANGEROUS_COMMAND_ONCE_LABEL)
        )
        proposal = _remote_exec_proposal(repo, scope, "deploy@example.com")

        decision = await coordinator.resolve_remote_exec_grant(
            cast(BaseTool, _AllowingTool()), _args(), _ctx(proposal), "bash", "c1"
        )
        assert decision is not None
        assert decision.verdict.value == "execute"
        assert decision.admission is not None
        assert decision.admission.remote_exec_requirements == (
            RemoteExecGrantRequirement(
                scope=scope, destination=_destination("deploy@example.com")
            ),
        )
        # The permit was consumed before the call resumed: no reusable authority.
        assert store.dangerous_command_authorities() == ()

    @pytest.mark.asyncio
    async def test_allow_session_installs_exact_destination_authority(
        self, tmp_path: Path
    ):
        repo = _repo(tmp_path / "repo")
        scope = _scope(repo)
        store = DangerousCommandGrantStore()
        coordinator = _coordinator(
            store=store, broker=_FixedBroker(DANGEROUS_COMMAND_SESSION_LABEL)
        )
        proposal = _remote_exec_proposal(repo, scope, "deploy@example.com")

        decision = await coordinator.resolve_remote_exec_grant(
            cast(BaseTool, _AllowingTool()), _args(), _ctx(proposal), "bash", "c1"
        )
        assert decision is not None
        assert decision.verdict.value == "execute"
        authorities = store.dangerous_command_authorities()
        assert len(authorities) == 1
        authority = authorities[0]
        assert authority.operations == frozenset({REMOTE_EXEC_OPERATION_ID})
        assert authority.destination == _destination("deploy@example.com")
        assert authority.scope == scope
        assert authority.source == "runtime_grant"
        assert authority_admits(
            authority,
            REMOTE_EXEC_OPERATION_ID,
            scope,
            destination=_destination("deploy@example.com:22"),
        )
        # Only the granted destination, never a near miss.
        assert not authority_admits(
            authority,
            REMOTE_EXEC_OPERATION_ID,
            scope,
            destination=_destination("ops@example.com"),
        )
        assert not authority_admits(authority, "network.egress", scope)

    @pytest.mark.asyncio
    async def test_allow_project_persists_the_exact_destination(self, tmp_path: Path):
        repo = _repo(tmp_path / "repo")
        scope = _scope(repo)
        store = DangerousCommandGrantStore()
        port = _GrantStorePort()
        coordinator = _coordinator(
            store=store, broker=_FixedBroker(DANGEROUS_COMMAND_PROJECT_LABEL), port=port
        )
        proposal = _remote_exec_proposal(repo, scope, "deploy@EXAMPLE.com")

        decision = await coordinator.resolve_remote_exec_grant(
            cast(BaseTool, _AllowingTool()), _args(), _ctx(proposal), "bash", "c1"
        )
        assert decision is not None
        assert decision.verdict.value == "execute"
        assert port.calls == [("session", repo, "deploy@example.com:22", "rev-1")]
        assert decision.result_note is not None
        assert "saved in your user config" in decision.result_note
        # The live session grant is applied regardless of the save outcome.
        assert [a.destination for a in store.dangerous_command_authorities()] == [
            _destination("deploy@example.com")
        ]

    def test_session_grants_do_not_propagate_to_a_child_store(self, tmp_path: Path):
        # A child loop owns its own store (the same wiring the app server uses),
        # so a parent's session destination grant is invisible to it.
        scope = _scope(_repo(tmp_path / "repo"))
        parent = DangerousCommandGrantStore()
        assert parent.grant_remote_exec_session({
            RemoteExecGrantRequirement(scope, _destination("deploy@example.com"))
        })
        child = DangerousCommandGrantStore()
        assert child.dangerous_command_authorities() == ()
        assert parent.dangerous_command_authorities() != ()

    @pytest.mark.asyncio
    async def test_child_never_prompts_and_names_the_destination_knob(
        self, tmp_path: Path
    ):
        repo = _repo(tmp_path / "repo")
        store = DangerousCommandGrantStore()
        broker = _FixedBroker(DANGEROUS_COMMAND_PROJECT_LABEL)
        coordinator = _coordinator(
            store=store, broker=broker, is_subagent=True, port=_GrantStorePort()
        )
        proposal = _remote_exec_proposal(repo, _scope(repo), "deploy@example.com")

        decision = await coordinator.resolve_remote_exec_grant(
            cast(BaseTool, _AllowingTool()), _args(), _ctx(proposal), "bash", "c1"
        )
        assert decision is not None
        assert decision.verdict.value == "skip"
        assert decision.feedback is not None
        assert "remote_exec_grants_by_project" in decision.feedback
        assert "deploy@example.com:22" in decision.feedback
        assert "config.toml" in decision.feedback
        assert broker.requests == []
        assert store.dangerous_command_authorities() == ()

    def test_persisted_grants_authorize_parent_and_child(self, tmp_path: Path):
        # The parent assembles from its accepted restrictions; a child inherits
        # them through its inherited_restrictions. Both get the same
        # destination authority from the same assembly routine.
        repo = _repo(tmp_path / "repo")
        scope = _scope(repo)
        restrictions = SourceRestrictions.from_raw(
            {"remote_exec_grants_by_project": {str(repo): ["deploy@example.com"]}},
            root_layer=UserConfigLayer(path=tmp_path / "config.toml"),
            layer_name="user-toml",
            locator="user",
            kind="source",
            store_fingerprint=None,
        )

        def authorities(inherited: tuple[SourceRestrictions, ...], own):
            fake = SimpleNamespace(
                _authority_retired=False,
                _inherited_restrictions=inherited,
                _restriction_getter=own,
                _dangerous_command_authority_provider=None,
            )
            return ToolManager._dangerous_command_authorities(fake)  # type: ignore[arg-type]

        parent = authorities((), lambda: (restrictions,))
        child = authorities((restrictions,), None)
        for snapshot in (parent, child):
            assert len(snapshot) == 1
            authority = snapshot[0]
            assert authority.source == "protected_config"
            assert authority.destination == _destination("deploy@example.com")
            assert authority_admits(
                authority,
                REMOTE_EXEC_OPERATION_ID,
                scope,
                destination=_destination("deploy@example.com:22"),
            )
            assert not authority_admits(authority, "network.egress", scope)

    def test_clear_session_state_drops_destination_grants_and_permits(
        self, tmp_path: Path
    ):
        scope = _scope(_repo(tmp_path / "repo"))
        store = DangerousCommandGrantStore()
        store.grant_remote_exec_session({
            RemoteExecGrantRequirement(scope, _destination("deploy@example.com"))
        })
        store.install_remote_exec_once(
            frozenset({
                RemoteExecGrantRequirement(scope, _destination("ops@example.com"))
            })
        )
        assert len(store.dangerous_command_authorities()) == 2
        assert store.clear_session_state() is True
        assert store.dangerous_command_authorities() == ()

    def test_revoking_the_operation_removes_every_destination_in_scope(
        self, tmp_path: Path
    ):
        scope = _scope(_repo(tmp_path / "repo"))
        store = DangerousCommandGrantStore()
        store.grant_remote_exec_session({
            RemoteExecGrantRequirement(scope, _destination("deploy@example.com")),
            RemoteExecGrantRequirement(scope, _destination("ops@example.com")),
        })
        assert store.revoke_session({(REMOTE_EXEC_OPERATION_ID, scope)})
        assert store.dangerous_command_authorities() == ()
