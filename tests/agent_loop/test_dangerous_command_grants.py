"""Package P4 unit tests: grant stores, question shape, parsing, derivation."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from chartreux.core.agent_loop._dangerous_command_grants import (
    DANGEROUS_COMMAND_DECLINED_FEEDBACK,
    DANGEROUS_COMMAND_DENY_LABEL,
    DANGEROUS_COMMAND_ONCE_LABEL,
    DANGEROUS_COMMAND_PROJECT_LABEL,
    DANGEROUS_COMMAND_SESSION_LABEL,
    DangerousCommandCallIdentity,
    DangerousCommandGrantStore,
    dangerous_command_grant_save_note,
    dangerous_command_request,
    derive_call_requirements,
    parse_dangerous_command_result,
)
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.dangerous_commands import (
    OPERATION_REGISTRY,
    DangerousCommandProposal,
    RepositoryScope,
)
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    resolve_repository_scope,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    _split_command_tokens,
)
from chartreux.questions import UserAnswer, UserQuestionResult


def make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def scope_of(path: Path) -> RepositoryScope:
    scope = resolve_repository_scope(path, GitRepositoryRelocation())
    assert scope is not None
    return scope


def proposal_for(
    project: Path,
    scope: RepositoryScope,
    command: tuple[str, ...],
    operations: tuple[str, ...],
) -> DangerousCommandProposal:
    return DangerousCommandProposal(
        command=command,
        project=project,
        scope=scope,
        operations=operations,
        consequence="; ".join(
            OPERATION_REGISTRY[operation_id].consequence for operation_id in operations
        ),
    )


def answer_with(label: str, *, cancelled: bool = False) -> UserQuestionResult:
    return UserQuestionResult(
        answers=[] if cancelled else [UserAnswer(question="q", answer=label)],
        cancelled=cancelled,
    )


class TestSessionGrants:
    """Session authority is keyed by operation AND repository scope."""

    def test_grant_installs_authority_per_operation_and_scope(self, tmp_path: Path):
        repo = make_repo(tmp_path / "repo")
        scope = scope_of(repo)
        store = DangerousCommandGrantStore()
        assert store.dangerous_command_authorities() == ()
        assert store.grant_session({("git.push", scope), ("git.checkout", scope)})
        authorities = store.dangerous_command_authorities()
        assert len(authorities) == 1
        authority = authorities[0]
        assert authority.operations == frozenset({"git.push", "git.checkout"})
        assert authority.scope == scope
        assert authority.source == "runtime_grant"

    def test_grant_is_idempotent_per_operation_and_scope(self, tmp_path: Path):
        scope = scope_of(make_repo(tmp_path / "repo"))
        store = DangerousCommandGrantStore()
        assert store.grant_session({("git.push", scope)})
        assert not store.grant_session({("git.push", scope)})
        assert len(store.dangerous_command_authorities()) == 1

    def test_equivalent_scope_shares_authority_but_sibling_does_not(
        self, tmp_path: Path
    ):
        repo = make_repo(tmp_path / "repo")
        scope = scope_of(repo)
        other = scope_of(make_repo(tmp_path / "other"))
        store = DangerousCommandGrantStore()
        store.grant_session({("git.push", scope)})
        authorities = store.dangerous_command_authorities()
        assert all(authority.scope == scope for authority in authorities)
        assert all(authority.scope != other for authority in authorities)

    def test_revoke_session_removes_only_the_matching_authority(self, tmp_path: Path):
        repo = make_repo(tmp_path / "repo")
        scope = scope_of(repo)
        store = DangerousCommandGrantStore()
        store.grant_session({("git.push", scope), ("git.checkout", scope)})
        assert store.revoke_session({("git.push", scope)})
        authorities = store.dangerous_command_authorities()
        assert [authority.operations for authority in authorities] == [
            frozenset({"git.checkout"})
        ]
        assert not store.revoke_session({("git.push", scope)})
        assert store.revoke_session({("git.checkout", scope)})
        assert store.dangerous_command_authorities() == ()


class TestOncePermits:
    """A once permit is live authority for exactly one invocation."""

    def test_install_and_destroy_round_trip(self, tmp_path: Path):
        scope = scope_of(make_repo(tmp_path / "repo"))
        store = DangerousCommandGrantStore()
        permit = store.install_once(frozenset({("git.push", scope)}))
        assert [a.operations for a in store.dangerous_command_authorities()] == [
            frozenset({"git.push"})
        ]
        store.destroy_once(permit)
        assert store.dangerous_command_authorities() == ()
        # Destroying twice is a no-op, never an error.
        store.destroy_once(permit)

    def test_invalidate_pending_removes_only_matching_permits(self, tmp_path: Path):
        repo = make_repo(tmp_path / "repo")
        scope = scope_of(repo)
        other = scope_of(make_repo(tmp_path / "other"))
        store = DangerousCommandGrantStore()
        matching = store.install_once(frozenset({("git.push", scope)}))
        kept = store.install_once(frozenset({("git.checkout", other)}))
        assert store.invalidate_pending({("git.push", scope)}) == 1
        remaining = store.dangerous_command_authorities()
        assert [a.operations for a in remaining] == [frozenset({"git.checkout"})]
        assert all(a.scope == other for a in remaining)
        store.destroy_once(kept)
        store.destroy_once(matching)


class TestRefusalMemory:
    """Refusal memory is scoped to the decided call identity."""

    def test_recorded_refusal_is_recognized(self, tmp_path: Path):
        scope = scope_of(make_repo(tmp_path / "repo"))
        identity = DangerousCommandCallIdentity(scope=scope, command=("git", "push"))
        store = DangerousCommandGrantStore()
        assert not store.is_refused(identity)
        store.record_refusal(identity)
        assert store.is_refused(identity)

    def test_unrelated_identities_are_not_blocked(self, tmp_path: Path):
        scope = scope_of(make_repo(tmp_path / "repo"))
        store = DangerousCommandGrantStore()
        store.record_refusal(
            DangerousCommandCallIdentity(scope=scope, command=("git", "push"))
        )
        assert not store.is_refused(
            DangerousCommandCallIdentity(
                scope=scope, command=("git", "push", "--force")
            )
        )
        assert not store.is_refused(
            DangerousCommandCallIdentity(
                scope=scope_of(make_repo(tmp_path / "other")), command=("git", "push")
            )
        )
        assert store.dangerous_command_authorities() == ()

    def test_identity_is_the_whole_call_shlex_token_tuple(self, tmp_path: Path):
        """Canonicalization level: whole-call shlex tokens, nothing smarter.

        The proposal's command is the raw call's shlex token tuple, so
        whitespace and quoting variants of one command hit the same refusal
        record, while a semantically equivalent reformulation that
        tokenizes differently (-f vs --force) does not.
        """
        scope = scope_of(make_repo(tmp_path / "repo"))
        store = DangerousCommandGrantStore()

        def identity_of(command: str) -> DangerousCommandCallIdentity:
            return DangerousCommandCallIdentity(
                scope=scope, command=tuple(_split_command_tokens(command))
            )

        store.record_refusal(identity_of("git push"))
        for variant in ("git  push", "  git push  ", 'git "push"', "git 'push'"):
            assert store.is_refused(identity_of(variant))
        # A different token tuple is a different decision, even when the
        # shell meaning is identical.
        store.record_refusal(identity_of("git push -f"))
        assert store.is_refused(identity_of("git push -f"))
        assert not store.is_refused(identity_of("git push --force"))


class TestSessionStateClear:
    """An in-place session reset drops every session-scoped record."""

    def test_clear_drops_grants_permits_and_refusals(self, tmp_path: Path):
        scope = scope_of(make_repo(tmp_path / "repo"))
        store = DangerousCommandGrantStore()
        store.grant_session({("git.push", scope), ("git.checkout", scope)})
        store.install_once(frozenset({("git.push", scope)}))
        identity = DangerousCommandCallIdentity(scope=scope, command=("git", "push"))
        store.record_refusal(identity)
        assert store.clear_session_state() is True
        assert store.dangerous_command_authorities() == ()
        assert not store.is_refused(identity)

    def test_clear_keeps_the_store_object_and_bumps_the_generation(
        self, tmp_path: Path
    ):
        scope = scope_of(make_repo(tmp_path / "repo"))
        store = DangerousCommandGrantStore()
        store.grant_session({("git.push", scope)})
        generation = store.authority_generation
        store.clear_session_state()
        # The store object survives (the tool manager's provider wiring
        # references it); only its records go, and the generation always
        # bumps so consumers keying on it detect the boundary.
        assert store.authority_generation == generation + 1
        assert store.clear_session_state() is False
        assert store.authority_generation == generation + 2
        # The cleared store grants again exactly like a fresh one.
        assert store.grant_session({("git.push", scope)})
        assert [a.operations for a in store.dangerous_command_authorities()] == [
            frozenset({"git.push"})
        ]


class TestAuthorityGeneration:
    """The monotonic live-authority generation is exposed read-only."""

    def test_every_mutation_bumps_the_generation(self, tmp_path: Path):
        scope = scope_of(make_repo(tmp_path / "repo"))
        other = scope_of(make_repo(tmp_path / "other"))
        store = DangerousCommandGrantStore()
        assert store.authority_generation == 0
        store.grant_session({("git.push", scope)})
        assert store.authority_generation == 1
        store.revoke_session({("git.push", scope)})
        assert store.authority_generation == 2
        store.record_refusal(
            DangerousCommandCallIdentity(scope=scope, command=("git", "push"))
        )
        assert store.authority_generation == 3
        permit = store.install_once(frozenset({("git.push", scope)}))
        assert store.authority_generation == 4
        store.destroy_once(permit)
        assert store.authority_generation == 5
        permit = store.install_once(frozenset({("git.checkout", other)}))
        assert store.authority_generation == 6
        assert store.invalidate_pending({("git.checkout", other)}) == 1
        assert store.authority_generation == 7

    def test_mutation_free_calls_do_not_bump_the_generation(self, tmp_path: Path):
        scope = scope_of(make_repo(tmp_path / "repo"))
        store = DangerousCommandGrantStore()
        store.grant_session({("git.push", scope)})
        store.record_refusal(
            DangerousCommandCallIdentity(scope=scope, command=("git", "push"))
        )
        permit = store.install_once(frozenset({("git.push", scope)}))
        store.destroy_once(permit)
        generation = store.authority_generation
        # Repeats that change nothing are not mutations.
        assert not store.grant_session({("git.push", scope)})
        assert not store.revoke_session({("git.checkout", scope)})
        assert store.invalidate_pending({("git.checkout", scope)}) == 0
        store.record_refusal(
            DangerousCommandCallIdentity(scope=scope, command=("git", "push"))
        )
        store.destroy_once(permit)
        assert store.authority_generation == generation

    def test_generation_is_exposed_read_only(self):
        store = DangerousCommandGrantStore()
        descriptor = type(store).authority_generation
        assert isinstance(descriptor, property)
        assert descriptor.fset is None
        assert store.authority_generation == 0


class TestQuestionShape:
    """The frozen grant-choice display contract."""

    def test_publication_offers_deny_first_then_once_session_project(
        self, tmp_path: Path
    ):
        repo = make_repo(tmp_path / "repo")
        proposal = proposal_for(repo, scope_of(repo), ("git", "push"), ("git.push",))
        request = dangerous_command_request(proposal)
        question = request.questions[0]
        assert [option.label for option in question.options] == [
            DANGEROUS_COMMAND_DENY_LABEL,
            DANGEROUS_COMMAND_ONCE_LABEL,
            DANGEROUS_COMMAND_SESSION_LABEL,
            DANGEROUS_COMMAND_PROJECT_LABEL,
        ]
        assert question.options[0].label == DANGEROUS_COMMAND_DENY_LABEL
        assert question.multi_select is False
        assert question.hide_other is True
        assert len(question.header) <= 20

    def test_consequential_never_offers_always(self, tmp_path: Path):
        repo = make_repo(tmp_path / "repo")
        proposal = proposal_for(
            repo, scope_of(repo), ("git", "stash", "drop"), ("git.stash.drop",)
        )
        request = dangerous_command_request(proposal)
        question = request.questions[0]
        assert [option.label for option in question.options] == [
            DANGEROUS_COMMAND_DENY_LABEL,
            DANGEROUS_COMMAND_ONCE_LABEL,
            DANGEROUS_COMMAND_SESSION_LABEL,
        ]
        assert DANGEROUS_COMMAND_PROJECT_LABEL not in {
            option.label for option in question.options
        }

    def test_mixed_call_offers_only_the_consequential_choices(self, tmp_path: Path):
        repo = make_repo(tmp_path / "repo")
        proposal = proposal_for(
            repo,
            scope_of(repo),
            ("git", "push", "&&", "git", "push", "--force"),
            ("git.push", "git.push.force"),
        )
        request = dangerous_command_request(proposal)
        question = request.questions[0]
        assert [option.label for option in question.options] == [
            DANGEROUS_COMMAND_DENY_LABEL,
            DANGEROUS_COMMAND_ONCE_LABEL,
            DANGEROUS_COMMAND_SESSION_LABEL,
        ]

    def test_question_content_shows_command_project_repo_and_consequence(
        self, tmp_path: Path
    ):
        repo = make_repo(tmp_path / "repo")
        scope = scope_of(repo)
        proposal = proposal_for(repo, scope, ("git", "push"), ("git.push",))
        request = dangerous_command_request(proposal)
        question = request.questions[0]
        assert "git push" in question.question
        assert str(repo) in question.question
        assert "git.push" in question.question
        assert OPERATION_REGISTRY["git.push"].consequence in question.question
        assert "permission authority" in question.question
        assert "standing intent to publish" in question.question
        assert request.footer_note

    def test_consequential_question_carries_no_publication_warning(
        self, tmp_path: Path
    ):
        repo = make_repo(tmp_path / "repo")
        proposal = proposal_for(
            repo, scope_of(repo), ("git", "stash", "clear"), ("git.stash.clear",)
        )
        question = dangerous_command_request(proposal).questions[0]
        assert "standing intent to publish" not in question.question


class TestParseResult:
    """Answers map to choices; anything else fails closed to deny."""

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            (DANGEROUS_COMMAND_DENY_LABEL, "deny"),
            (DANGEROUS_COMMAND_ONCE_LABEL, "allow_once"),
            (DANGEROUS_COMMAND_SESSION_LABEL, "allow_session"),
            (DANGEROUS_COMMAND_PROJECT_LABEL, "allow_project"),
        ],
    )
    def test_publication_answers(self, label: str, expected: str):
        result = parse_dangerous_command_result(answer_with(label), ("git.push",))
        assert result == expected

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            (DANGEROUS_COMMAND_DENY_LABEL, "deny"),
            (DANGEROUS_COMMAND_ONCE_LABEL, "allow_once"),
            (DANGEROUS_COMMAND_SESSION_LABEL, "allow_session"),
        ],
    )
    def test_consequential_answers(self, label: str, expected: str):
        result = parse_dangerous_command_result(answer_with(label), ("git.stash.drop",))
        assert result == expected

    def test_forged_consequential_always_is_rejected(self):
        result = parse_dangerous_command_result(
            answer_with(DANGEROUS_COMMAND_PROJECT_LABEL), ("git.stash.drop",)
        )
        assert result == "deny"

    def test_forged_mixed_always_is_rejected(self):
        result = parse_dangerous_command_result(
            answer_with(DANGEROUS_COMMAND_PROJECT_LABEL), ("git.push", "git.push.force")
        )
        assert result == "deny"

    @pytest.mark.parametrize(
        "result",
        [
            UserQuestionResult(answers=[], cancelled=True),
            UserQuestionResult(answers=[UserAnswer(question="q", answer="yes")]),
            UserQuestionResult(answers=[UserAnswer(question="q", answer="")]),
            UserQuestionResult(answers=[]),
        ],
    )
    def test_non_choice_answers_deny(self, result: UserQuestionResult):
        assert parse_dangerous_command_result(result, ("git.push",)) == "deny"

    def test_declined_feedback_text(self):
        assert "user declined; do not retry" in DANGEROUS_COMMAND_DECLINED_FEEDBACK


class TestDeriveCallRequirements:
    """Approvals re-derive every (operation, scope) from the original call."""

    def test_single_repository_call(self, tmp_path: Path):
        repo = make_repo(tmp_path / "repo")
        requirements = derive_call_requirements("git push", repo, ("git.push",))
        assert requirements == frozenset({("git.push", scope_of(repo))})

    def test_multi_repository_call_covers_every_scope(self, tmp_path: Path):
        repo_a = make_repo(tmp_path / "a")
        repo_b = make_repo(tmp_path / "b")
        requirements = derive_call_requirements(
            "git -C a push && git -C b checkout", tmp_path, ("git.push", "git.checkout")
        )
        assert requirements == frozenset({
            ("git.push", scope_of(repo_a)),
            ("git.checkout", scope_of(repo_b)),
        })

    def test_cd_scoped_call_covers_every_possible_cwd(self, tmp_path: Path):
        repo = make_repo(tmp_path / "repo")
        sub = repo / "sub"
        sub.mkdir()
        requirements = derive_call_requirements(
            "cd sub && git push", repo, ("git.push",)
        )
        # The cwd set is monotonic: `cd` may fail, so both directories count.
        assert requirements == frozenset({
            ("git.push", scope_of(repo)),
            ("git.push", scope_of(sub)),
        })

    def test_operation_not_required_by_the_call_is_absent(self, tmp_path: Path):
        repo = make_repo(tmp_path / "repo")
        assert derive_call_requirements("git push", repo, ("git.checkout",)) == (
            frozenset()
        )

    def test_nested_shell_call_is_covered(self, tmp_path: Path):
        repo = make_repo(tmp_path / "repo")
        requirements = derive_call_requirements(
            "sh -c 'git push' && git restore file.txt",
            repo,
            ("git.push", "git.restore"),
        )
        assert requirements == frozenset({
            ("git.push", scope_of(repo)),
            ("git.restore", scope_of(repo)),
        })

    def test_unrelated_command_yields_no_requirements(self, tmp_path: Path):
        repo = make_repo(tmp_path / "repo")
        assert derive_call_requirements("git status", repo, ("git.push",)) == (
            frozenset()
        )

    def test_scratchpad_roots_do_not_change_derived_requirements(self, tmp_path: Path):
        # The re-derivation never reads approval_reasons — the only
        # roots-sensitive analysis output — so the roots parameter is
        # uniformity-only: an absolute scratchpad redirect target cannot
        # change the derived (operation, scope) requirements (ADR 0021).
        repo = make_repo(tmp_path / "repo")
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        command = f"git push > {scratch}/log.txt"
        without_roots = derive_call_requirements(command, repo, ("git.push",))
        with_roots = derive_call_requirements(
            command, repo, ("git.push",), scratchpad_roots=frozenset({scratch})
        )
        assert without_roots == with_roots == frozenset({("git.push", scope_of(repo))})


class TestSaveNote:
    """Persistence outcomes are reported apart from the live grant."""

    def test_saved_note(self, tmp_path: Path):
        repo = make_repo(tmp_path / "repo")
        note = dangerous_command_grant_save_note(
            tmp_path,
            scope_of(repo),
            (("git.push", ConfigSaveResult("user", "saved", "unchanged", "rev")),),
        )
        assert "applies to this session" in note
        assert "saved in your user config" in note
        assert "git.push" in note
        assert str(repo) in note

    def test_conflict_note_offers_a_retry(self, tmp_path: Path):
        repo = make_repo(tmp_path / "repo")
        note = dangerous_command_grant_save_note(
            tmp_path,
            scope_of(repo),
            (
                (
                    "git.push",
                    ConfigSaveResult(
                        "user", "not_saved", "unchanged", error="conflict"
                    ),
                ),
            ),
        )
        assert "applies to this session only" in note
        assert "retry" in note

    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (
                ConfigSaveResult("user", "not_saved", "unchanged", error="write"),
                "saving it to your user config failed",
            ),
            (None, "saving it to your user config failed"),
            (
                ConfigSaveResult("user", "durability_uncertain", "unchanged", "rev"),
                "could not be confirmed",
            ),
            (
                ConfigSaveResult("user", "saved", "unchanged", "rev", "cancelled"),
                "was cancelled",
            ),
            (
                ConfigSaveResult("user", "not_saved", "unchanged"),
                "does not save dangerous-command grants",
            ),
        ],
    )
    def test_failure_notes_are_honest(
        self, tmp_path: Path, result: ConfigSaveResult | None, expected: str
    ):
        repo = make_repo(tmp_path / "repo")
        note = dangerous_command_grant_save_note(
            tmp_path, scope_of(repo), (("git.push", result),)
        )
        assert "applies to this session" in note
        assert expected in note
