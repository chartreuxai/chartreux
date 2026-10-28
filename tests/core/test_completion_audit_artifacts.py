"""P7: the C1/C3 artifact audit tests.

The commit mandate's prefix matching, the git-command-result extraction
(consumes captured stdout, never a command echo), the submission-convention
touched-path collection, and the round-8 mutation-root semantics: a root
establishes conservative mutation scope only, so worker credit requires
file-specific evidence and a root-containment-only claim is unverified
(check-context) — with both an affected and an unaffected descendant under
the same root pinned. Custom/MCP writes are out of scope (documented).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
import pytest

from chartreux.core.completion_audit_artifacts import (
    ArtifactAuditDisposition,
    ArtifactEvidence,
    audit_artifacts,
    collect_touched_paths,
    git_result_hashes,
    hashes_match,
)
from chartreux.core.completion_contract import (
    CompletionArtifactKind,
    CompletionCheckTemplate,
    CompletionRequirement,
)
from chartreux.core.completion_evidence import snapshot_bytes
from chartreux.core.events import ToolResultEvent
from chartreux.core.subagent_submission import UNKNOWN_WRITE_FOOTPRINT
from chartreux.core.subagents import UNKNOWN_MUTATION_ROOT_FOOTPRINT
from chartreux.utils.tool_presentation import ToolEffectKind


class _ShellStub:
    effect_kind = ToolEffectKind.SHELL


class _EditStub:
    effect_kind = ToolEffectKind.FILE_EDIT


class _WriteStub:
    effect_kind = ToolEffectKind.FILE_WRITE


class _ShellOutcome(BaseModel):
    """The duck-typed slice of CapturedShellResult the audit reads."""

    command: str
    stdout: str = ""


class _FilePaths(BaseModel):
    """The duck-typed slice of the file tools' result models the submission
    conventions read.
    """

    file: str = ""
    file_path: str = ""


class _ShellFootprints(BaseModel):
    """The duck-typed slice of CapturedShellResult's footprint fields."""

    write_paths: list[str] = Field(default_factory=list)
    write_paths_unknown: bool = False


def _result_event(
    *,
    tool_class: type | None = None,
    result: BaseModel | None = None,
    skipped: bool = False,
    error_write_paths: list[str] | None = None,
    error_write_paths_unknown: bool = False,
    error_mutation_roots: list[str] | None = None,
    error_mutation_roots_unknown: bool = False,
) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="stub_tool",
        tool_class=tool_class,
        result=result,
        tool_call_id="call-1",
        error_write_paths=list(error_write_paths or []),
        error_write_paths_unknown=error_write_paths_unknown,
        error_mutation_roots=list(error_mutation_roots or []),
        error_mutation_roots_unknown=error_mutation_roots_unknown,
        skipped=skipped,
    )


def _requirement(
    requirement_id: str,
    *,
    artifact: CompletionArtifactKind | None = None,
    template: CompletionCheckTemplate | None = None,
) -> CompletionRequirement:
    return CompletionRequirement(
        requirement_id=requirement_id, artifact=artifact, template=template
    )


def _commit_requirement(requirement_id: str = "req-commit") -> CompletionRequirement:
    return _requirement(requirement_id, artifact=CompletionArtifactKind.COMMIT)


def _touched_path_requirement(
    requirement_id: str = "req-path",
) -> CompletionRequirement:
    return _requirement(requirement_id, artifact=CompletionArtifactKind.TOUCHED_PATH)


_FULL_HASH_A = "aaa1111abc999def222333444555666777888999"
_FULL_HASH_B = "bbb2222def888abc111222333444555666777888"
_FULL_HASH_C = "ccc3333abc777def111222333444555666777888"


def _audit(
    requirement: CompletionRequirement, claimed: str | None, evidence: ArtifactEvidence
) -> ArtifactAuditDisposition:
    findings = audit_artifacts(
        [requirement],
        [requirement.requirement_id],
        {} if claimed is None else {requirement.requirement_id: claimed},
        evidence,
    )
    assert len(findings) == 1
    assert findings[0].requirement_id == requirement.requirement_id
    assert findings[0].artifact == requirement.artifact
    return findings[0].disposition


# ---------------------------------------------------------------------------
# The prefix match (C1)


@pytest.mark.parametrize(
    ("claimed", "observed"),
    [
        ("aaa1111", _FULL_HASH_A),
        (_FULL_HASH_A, "aaa1111"),
        (_FULL_HASH_A, _FULL_HASH_A),
        (_FULL_HASH_A.upper(), _FULL_HASH_A),
        ("aaa1111abc999def", _FULL_HASH_A),
    ],
)
def test_prefix_matches_bind_in_both_directions(claimed: str, observed: str) -> None:
    assert hashes_match(claimed, observed)


@pytest.mark.parametrize(
    ("claimed", "observed"),
    [
        ("aaa111", _FULL_HASH_A),  # six characters: ambiguous, never evidence
        (_FULL_HASH_A, "aaa111"),
        ("abc1111", _FULL_HASH_B),  # prefix mismatch
        ("xyz1111", _FULL_HASH_A),  # not hex
        (_FULL_HASH_A + "0" * 20, _FULL_HASH_A),  # over-long fragment
    ],
)
def test_fragments_that_cannot_denote_one_object_never_match(
    claimed: str, observed: str
) -> None:
    assert not hashes_match(claimed, observed)


# ---------------------------------------------------------------------------
# The git-command-result extraction


def test_rev_parse_stdout_carries_hashes() -> None:
    event = _result_event(
        tool_class=_ShellStub,
        result=_ShellOutcome(command="git rev-parse HEAD", stdout=f"{_FULL_HASH_A}\n"),
    )
    assert git_result_hashes(event) == (_FULL_HASH_A,)


def test_log_stdout_hashes_are_deduped() -> None:
    event = _result_event(
        tool_class=_ShellStub,
        result=_ShellOutcome(
            command="git log --oneline -3",
            stdout=(
                f"{_FULL_HASH_A} first\n"
                f"{_FULL_HASH_A} second\n"
                f"short {_FULL_HASH_B[:8]} third\n"
            ),
        ),
    )
    assert git_result_hashes(event) == (_FULL_HASH_A, _FULL_HASH_B[:8])


def test_a_hash_inside_the_command_string_is_a_never_evidence_echo() -> None:
    event = _result_event(
        tool_class=_ShellStub,
        result=_ShellOutcome(command=f"git checkout {_FULL_HASH_A[:9]}", stdout=""),
    )
    assert git_result_hashes(event) == ()


def test_non_git_stdout_never_carries_hashes() -> None:
    event = _result_event(
        tool_class=_ShellStub,
        result=_ShellOutcome(command="echo done", stdout=f"{_FULL_HASH_A}\n"),
    )
    assert git_result_hashes(event) == ()


def test_git_status_output_is_never_treated_as_hash_bearing() -> None:
    event = _result_event(
        tool_class=_ShellStub,
        result=_ShellOutcome(
            command="git status --porcelain", stdout=f"?? {_FULL_HASH_A[:7]}\n"
        ),
    )
    assert git_result_hashes(event) == ()


def test_env_prefixed_and_directory_scoped_git_commands_still_extract() -> None:
    env_prefixed = _result_event(
        tool_class=_ShellStub,
        result=_ShellOutcome(
            command="GIT_DIR=/repo git rev-parse HEAD", stdout=f"{_FULL_HASH_B}\n"
        ),
    )
    directory_scoped = _result_event(
        tool_class=_ShellStub,
        result=_ShellOutcome(
            command="git -C /repo --no-pager rev-parse HEAD", stdout=f"{_FULL_HASH_C}\n"
        ),
    )
    assert git_result_hashes(env_prefixed) == (_FULL_HASH_B,)
    assert git_result_hashes(directory_scoped) == (_FULL_HASH_C,)


def test_skipped_and_error_events_carry_no_hashes() -> None:
    skipped = _result_event(tool_class=_ShellStub, skipped=True)
    error = _result_event(tool_class=_ShellStub, result=None)
    assert git_result_hashes(skipped) == ()
    assert git_result_hashes(error) == ()


# ---------------------------------------------------------------------------
# The submission-convention touched-path collection


def test_file_tool_and_shell_footprints_are_collected_and_deduped() -> None:
    events = [
        _result_event(
            tool_class=_EditStub, result=_FilePaths(file="chartreux/core/mod.py")
        ),
        _result_event(tool_class=_ShellStub, result=_ShellFootprints()),
    ]
    assert collect_touched_paths(events) == (("chartreux/core/mod.py",), False)


def test_shell_write_paths_and_error_footprints_join_the_collection() -> None:
    events = [
        _result_event(
            tool_class=_ShellStub,
            result=_ShellOutcome(command="python - <<'PY'", stdout="ignored"),
        ),
        _result_event(
            tool_class=_ShellStub,
            result=None,
            error_write_paths=["tests/core/out.txt"],
            error_write_paths_unknown=True,
        ),
    ]
    paths, unknown = collect_touched_paths(events)
    assert paths == ("tests/core/out.txt",)
    assert unknown


def test_skipped_events_contribute_nothing() -> None:
    events = [_result_event(tool_class=_EditStub, result=None, skipped=True)]
    assert collect_touched_paths(events) == ((), False)


# ---------------------------------------------------------------------------
# The commit mandate (C1)


def test_changed_head_with_matching_publication_hash_credits_the_worker() -> None:
    disposition = _audit(
        _commit_requirement(),
        _FULL_HASH_B,
        ArtifactEvidence(admission_head=_FULL_HASH_A, publication_head=_FULL_HASH_B),
    )
    assert disposition is ArtifactAuditDisposition.WORKER_CREDITED


def test_changed_head_with_a_matching_git_command_result_hash_credits() -> None:
    disposition = _audit(
        _commit_requirement(),
        _FULL_HASH_C[:7],
        ArtifactEvidence(
            observed_commit_hashes=(_FULL_HASH_C,),
            admission_head=_FULL_HASH_A,
            publication_head=_FULL_HASH_B,
        ),
    )
    assert disposition is ArtifactAuditDisposition.WORKER_CREDITED


def test_a_copied_hash_with_an_unchanged_head_cannot_satisfy_the_mandate() -> None:
    disposition = _audit(
        _commit_requirement(),
        _FULL_HASH_A,
        ArtifactEvidence(
            observed_commit_hashes=(_FULL_HASH_A,),
            admission_head=_FULL_HASH_A,
            publication_head=_FULL_HASH_A,
        ),
    )
    assert disposition is ArtifactAuditDisposition.UNVERIFIED


def test_matching_observed_hashes_without_a_changed_head_is_unverified() -> None:
    disposition = _audit(
        _commit_requirement(),
        _FULL_HASH_A,
        ArtifactEvidence(observed_commit_hashes=(_FULL_HASH_A,)),
    )
    assert disposition is ArtifactAuditDisposition.UNVERIFIED


def test_a_sibling_commit_with_a_changed_head_is_never_contradicted() -> None:
    disposition = _audit(
        _commit_requirement(),
        _FULL_HASH_C[:7],
        ArtifactEvidence(admission_head=_FULL_HASH_A, publication_head=_FULL_HASH_B),
    )
    assert disposition is ArtifactAuditDisposition.UNVERIFIED


def test_commit_evidence_absent_resolves_unavailable() -> None:
    disposition = _audit(_commit_requirement(), _FULL_HASH_A, ArtifactEvidence())
    assert disposition is ArtifactAuditDisposition.UNAVAILABLE


def test_a_claim_with_no_named_hash_is_unverified() -> None:
    disposition = _audit(
        _commit_requirement(),
        "not-a-hash",
        ArtifactEvidence(
            observed_commit_hashes=(_FULL_HASH_A,),
            admission_head=_FULL_HASH_A,
            publication_head=_FULL_HASH_B,
        ),
    )
    assert disposition is ArtifactAuditDisposition.UNVERIFIED


# ---------------------------------------------------------------------------
# The touched-path mandate (C3) and the round-8 mutation-root semantics


def test_touched_path_match_credits_the_worker() -> None:
    disposition = _audit(
        _touched_path_requirement(),
        "chartreux/core/mod.py",
        ArtifactEvidence(touched_paths=("chartreux/core/mod.py",)),
    )
    assert disposition is ArtifactAuditDisposition.WORKER_CREDITED


def test_collection_time_snapshot_credits_the_worker() -> None:
    snapshot = snapshot_bytes(b"artifact content at collection time")
    disposition = _audit(
        _touched_path_requirement(),
        "chartreux/core/mod.py",
        ArtifactEvidence(
            snapshots={"chartreux/core/./mod.py": snapshot},
            touched_paths=("chartreux/core/other.py",),
        ),
    )
    assert disposition is ArtifactAuditDisposition.WORKER_CREDITED


def test_affected_descendant_under_the_same_root_is_credited() -> None:
    disposition = _audit(
        _touched_path_requirement(),
        "chartreux/pkg/mod.py",
        ArtifactEvidence(
            touched_paths=("chartreux/pkg/mod.py",), mutation_roots=("/repo/worktree",)
        ),
    )
    assert disposition is ArtifactAuditDisposition.WORKER_CREDITED


def test_unaffected_descendant_under_the_same_root_stays_unverified() -> None:
    disposition = _audit(
        _touched_path_requirement(),
        "chartreux/pkg/other.py",
        ArtifactEvidence(
            touched_paths=("chartreux/pkg/mod.py",), mutation_roots=("/repo/worktree",)
        ),
    )
    assert disposition is ArtifactAuditDisposition.UNVERIFIED


def test_root_level_worktree_footprints_compare_like_any_other_root() -> None:
    disposition = _audit(
        _touched_path_requirement(),
        "/repo/worktree/pkg/mod.py",
        ArtifactEvidence(mutation_roots=("/repo/worktree",)),
    )
    assert disposition is ArtifactAuditDisposition.UNVERIFIED


def test_unknown_footprints_resolve_unavailable() -> None:
    write_surface = _audit(
        _touched_path_requirement(),
        "chartreux/core/mod.py",
        ArtifactEvidence(touched_paths=(UNKNOWN_WRITE_FOOTPRINT,)),
    )
    mutation_scope = _audit(
        _touched_path_requirement(),
        "chartreux/core/mod.py",
        ArtifactEvidence(mutation_roots=(UNKNOWN_MUTATION_ROOT_FOOTPRINT,)),
    )
    assert write_surface is ArtifactAuditDisposition.UNAVAILABLE
    assert mutation_scope is ArtifactAuditDisposition.UNAVAILABLE


def test_evidence_without_a_file_specific_match_is_unverified() -> None:
    disposition = _audit(
        _touched_path_requirement(),
        "chartreux/core/mod.py",
        ArtifactEvidence(
            touched_paths=("chartreux/core/other.py",),
            mutation_roots=("/repo/worktree",),
        ),
    )
    assert disposition is ArtifactAuditDisposition.UNVERIFIED


def test_custom_and_mcp_writes_are_documented_out_of_scope() -> None:
    """A claim about an artifact produced through a custom or MCP tool is
    never harness-observable: it resolves with nothing collected, never a
    contradiction (documented residual).
    """
    from chartreux.core import completion_audit_artifacts

    module_doc = completion_audit_artifacts.__doc__ or ""
    assert "Custom and MCP tool writes are out of scope" in module_doc
    disposition = _audit(
        _touched_path_requirement(), "custom/mcp/artifact.txt", ArtifactEvidence()
    )
    assert disposition is ArtifactAuditDisposition.UNAVAILABLE


# ---------------------------------------------------------------------------
# The batch resolution and its vocabulary


def test_only_claimed_artifact_requirements_are_resolved() -> None:
    claimed = _commit_requirement("req-claimed")
    unclaimed = _commit_requirement("req-unclaimed")
    findings = audit_artifacts(
        [claimed, unclaimed],
        ["req-claimed"],
        {"req-claimed": _FULL_HASH_B},
        ArtifactEvidence(admission_head=_FULL_HASH_A, publication_head=_FULL_HASH_B),
    )
    assert [finding.requirement_id for finding in findings] == ["req-claimed"]
    assert findings[0].disposition is ArtifactAuditDisposition.WORKER_CREDITED


def test_check_and_envelope_bindings_stay_out_of_the_artifact_audit() -> None:
    findings = audit_artifacts(
        [
            _requirement("req-check", template=CompletionCheckTemplate.PYTEST),
            _requirement("req-envelope", artifact=CompletionArtifactKind.ENVELOPE),
        ],
        ["req-check", "req-envelope"],
        {},
        ArtifactEvidence(),
    )
    assert findings == ()


def test_the_disposition_vocabulary_has_no_contradicted_member() -> None:
    assert {member.value for member in ArtifactAuditDisposition} == {
        "worker_credited",
        "unverified",
        "unavailable",
    }
