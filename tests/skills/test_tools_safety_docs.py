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


def test_documents_the_https_entry_point_grammar() -> None:
    # The recorded doc-grep target for HTTPie's `https` entry point (v0.7.1
    # F2): it left the default denylist under the same restricted grammar as
    # `http`, with the scheme forced onto a schemeless operand and an explicit
    # non-https scheme hard-denied as cross-scheme. Each assert is a
    # discriminating literal of THIS passage, so the retired
    # "denied fail-closed until modeled" sentence cannot satisfy it.
    assert "### Network-client egress modeling (v0.7.0)" in DOC
    assert (
        "**`https`.** HTTPie's `https` entry point is the same request client as" in DOC
    )
    assert "`https [METHOD] URL ITEM...`" in DOC
    assert "forces `https://` onto a SCHEMELESS operand" in DOC
    assert "keeps its `http://` and really" in DOC
    assert "does issue a plain-HTTP request: it hard-denies as a cross-scheme" in DOC
    assert "resolves ask-class rather than riding a loopback" in DOC
    assert "every `https` spelling resolves denied fail-closed" not in DOC
    assert "deferred to the v0.8.0 backlog" not in DOC


def test_documents_the_executor_option_contract_and_extglob_residual() -> None:
    # F1pkg1(b): the executor shell-option contract landed, so the extglob
    # clause now names the pinned executor and the root-owned residual, not an
    # under-inspection hazard in the user's environment.
    assert "executor shell-option contract" in DOC
    assert "Extglob environment residual" in DOC
    assert "root-owned trust base" in DOC
    assert "stays literal under the pinned contract" in DOC


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


def test_documents_the_s4b2a_ssh_denylist_exit_clause() -> None:
    # S4b2a flip of the S4b1b passage-1 pin (the default-denylist paragraph;
    # flipped, never dropped): ssh left the shipped denylist in v0.7.1 under
    # the closed destination grammar — a positive destination is an IP literal
    # or a hostname the system resolver (honoring /etc/hosts) answers with
    # exactly one non-local address — classified as network.remote-exec with
    # the exact GRANTED destination attached, the pin lands in the spawned
    # argv at every execution transport, and for a hostname the pinned operand
    # is the resolved address. Each assert is a discriminating literal of THIS
    # passage.
    assert "denylist in v0.7.1 under the closed destination grammar" in DOC
    assert "exactly one non-local address" in DOC
    assert "exact granted destination" in DOC
    assert "pinned into the spawned argv at every execution" in DOC
    assert "the pinned operand is the resolved address" in DOC


def test_documents_the_s4b2a_ssh_retained_denials_clause() -> None:
    # S4b2a flip of the S4b1b passage-2 pin (the retained-denials paragraph;
    # flipped, never dropped): ssh is the exception since v0.7.1; a positive
    # destination is an IP literal or a uniquely-non-locally-resolved hostname,
    # the fixed HostKeyAlias/UserKnownHostsFile identity is pinned into the
    # actual argv per spawn point, the resolved address rides as the hostname's
    # destination operand while the HostKeyAlias stays bound to the granted
    # hostname, and an ambiguous/local/failed resolution stays denied. Each
    # assert is a discriminating literal of THIS passage.
    assert "`ssh` is the exception since v0.7.1" in DOC
    assert "fixed `HostKeyAlias`/`UserKnownHostsFile` host-key identity" in DOC
    assert "answers uniquely and non-locally" in DOC
    assert "the resolved address rides as the destination operand" in DOC
    assert "failed hostname resolution stays denied" in DOC
