"""Documentation assertions for the tools-safety dangerous-command policy.

Keeps the security-normative claims of docs/guides/tools-safety.md honest:
the three policy classes, the grant choices, and the denylist breaking
change must stay documented as the runtime behaves.
"""

from __future__ import annotations

from pathlib import Path

DOC = (Path(__file__).parents[2] / "docs" / "guides" / "tools-safety.md").read_text()


def test_documents_three_policy_classes() -> None:
    assert "**Publication** (`git.push`" in DOC
    assert "**Consequential**" in DOC
    assert "git.push.force" in DOC
    assert "git.stash.clear" in DOC
    assert "**Hard guards**" in DOC
    assert "no operation ID and no grant path" in DOC


def test_documents_grant_choices() -> None:
    assert (
        "Deny (default), Allow once, Allow this session, or Always for this project"
        in DOC
    )
    assert (
        'Deny (default), Allow once, or Allow this session. **Never** "Always"' in DOC
    )
    assert (
        "a push combining consequential mechanisms requires all classified consequential operations"
        in DOC
    )


def test_documents_once_session_and_refusal_semantics() -> None:
    assert "exactly one invocation of the original call" in DOC
    assert "never shared across concurrent identical calls" in DOC
    assert "Refusal memory" in DOC
    assert "reformulated command re-prompts" in DOC
    assert "Recheck is the final gate" in DOC


def test_documents_breaking_change_and_absolute_denylist_veto() -> None:
    assert "### Breaking change" in DOC
    assert "`denylist = []` no longer implicitly authorizes" in DOC
    assert "Explicit `denylist` entries remain an absolute veto" in DOC


def test_documents_panel_and_no_public_grant_endpoint() -> None:
    assert "Settings > Shell Permissions" in DOC
    assert 'no public "grant an arbitrary operation" endpoint' in DOC
    assert "permission authority, not standing intent to publish" in DOC


def test_documents_bash_and_bash_start_parity() -> None:
    assert "`bash` and `bash_start` behave identically" in DOC


def test_documents_the_wget2_redirect_disable_prerequisite() -> None:
    # The recorded doc-grep target for wget2's loopback auto-allow: the
    # section exists, and it states the prerequisite that is NOT established,
    # which is why the denylist entry stays.
    assert "### Network-client egress modeling (v0.7.0)" in DOC
    assert (
        "no effective redirect-disable is established for wget2 in this release" in DOC
    )
    assert "every `wget2` spelling resolves denied" in DOC


def test_documents_the_ftp_pasv_residual_handling() -> None:
    # The recorded doc-grep target for the residual: it is disclosed for
    # wget and explicitly not inherited by a wget2 admission.
    assert "#### FTP PASV residual" in DOC
    assert "The wget FTP PASV residual is not inherited by wget2." in DOC
    assert "`wget2 ftp://...` operand is denied with the retained entry" in DOC


def test_documents_the_http_grammar_and_the_retained_denials() -> None:
    assert "`http [METHOD] URL ITEM...`" in DOC
    assert "no operation ID and no approval path" in DOC
    assert "`httpie` and `wget2` stay on the default denylist" in DOC


def test_documents_the_https_entry_point_denial() -> None:
    # The recorded doc-grep target for HTTPie's `https` entry point: it is the
    # unmodeled twin of the modeled `http`, denied fail-closed under the same
    # documented+corpus-tested prerequisite rule as wget2 and httpie, with
    # grammar modeling deferred.
    assert "### Network-client egress modeling (v0.7.0)" in DOC
    assert "**`https`.** HTTPie's `https` entry point" in DOC
    assert "it has no documented, corpus-tested grammar" in DOC
    assert "every `https` spelling resolves denied fail-closed" in DOC
    assert "deferred to the v0.8.0 backlog" in DOC


def test_documents_remote_exec_destination_grants() -> None:
    # The recorded doc-grep targets for the v0.7.0 S4a grant model: the class,
    # the destination-keyed lifecycle, the persistable field, and the honest
    # scope statement.
    assert "### Remote-exec destination grants (v0.7.0)" in DOC
    assert "**Remote-exec** (`network.remote-exec`" in DOC
    assert "one exact `user@host:port` destination" in DOC
    assert "a missing port canonicalizes to 22" in DOC
    assert "`remote_exec_grants_by_project`" in DOC
    assert "trusting the destination, not approving a contained payload" in DOC
    assert "an egress grant never admits a remote command" in DOC
    assert "Session grants do not propagate to children" in DOC
