from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.agents.identities import (
    Delegation,
    ToolPosture,
    UnknownIdentityError,
    builtin_identities,
    get_builtin_identity,
    validate_identity_selection,
)


def test_five_builtin_identities_have_typed_posture_and_metadata() -> None:
    identities = {identity.id: identity for identity in builtin_identities()}
    assert set(identities) == {
        "coder",
        "orchestrator",
        "researcher",
        "reviewer",
        "advisor",
    }
    assert all(
        identity.display_name and identity.safety for identity in identities.values()
    )
    assert identities["coder"].delegation is Delegation.OPTIONAL
    assert identities["coder"].personal_tools is ToolPosture.ALL_GRANTED
    assert identities["coder"].delegation_tools is ToolPosture.ALL_GRANTED
    assert identities["orchestrator"].delegation is Delegation.REQUIRED
    assert identities["orchestrator"].personal_tools is ToolPosture.NON_MUTATING
    assert identities["orchestrator"].delegation_tools is ToolPosture.ALL_GRANTED
    for name in ("researcher", "reviewer", "advisor"):
        # The declared delegation metadata matches the executable behavior:
        # these identities still delegate their non-mutating/control-plane
        # set (design §3: no shipped identity sets none in v1).
        assert identities[name].delegation is Delegation.OPTIONAL
        assert identities[name].personal_tools is ToolPosture.NON_MUTATING
        assert identities[name].delegation_tools is ToolPosture.NON_MUTATING
    assert get_builtin_identity("unknown") is None


@pytest.mark.parametrize(
    "reference",
    [
        "user:",  # empty custom name
        "user:.",  # traversal-ish single dot
        "user:..",  # traversal
        "user:../evil",  # traversal
        "user:a:b",  # multi-colon
        "user:a/b",  # path separator
        "user:a\\b",  # windows separator
        "user:bad name",  # whitespace
        "user:reviewer",  # reserved shipped name
        "user:coder",  # reserved shipped name
        "user:orchestrator",  # reserved shipped name
        "user:researcher",  # reserved shipped name
        "user:advisor",  # reserved shipped name
        "user:does-not-exist",  # no prompt resource
    ],
)
def test_user_qualified_references_fail_closed(tmp_path: Path, reference: str) -> None:
    """The loop-level check validates ``user:``-qualified selections.

    Malformed/traversal/empty/multi-colon forms, reserved shipped names
    (reserved names cannot shadow shipped definitions), and references whose
    ``<name>.md`` prompt resource does not exist in the user prompts
    directories are actionable errors — never a deferred rejection at prompt
    assembly (S2 round 2: unvalidated user:-qualified identities; design
    criterion 2; package-1 reserved-names acceptance).
    """
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    with pytest.raises(UnknownIdentityError) as raised:
        validate_identity_selection(reference, user_prompts_dirs=[prompts])
    message = str(raised.value)
    # Every error renders the reference with ``{reference!r}``, so the
    # message contains its repr (which also keeps backslash cases intact).
    assert repr(reference) in message
    if reference in {
        "user:reviewer",
        "user:coder",
        "user:orchestrator",
        "user:researcher",
        "user:advisor",
    }:
        assert "Reserved identity name" in message
    elif reference == "user:does-not-exist":
        assert "does-not-exist.md" in message
        assert str(prompts) in message
    else:
        assert "Malformed custom identity" in message


def test_user_qualified_reference_with_a_resource_is_valid(tmp_path: Path) -> None:
    """A ``user:`` selection whose prompt resource exists validates."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "my-custom.md").write_text("# My custom identity\n", encoding="utf-8")
    assert (
        validate_identity_selection("user:my-custom", user_prompts_dirs=[prompts])
        == "user:my-custom"
    )
    # A bare custom name resolves to the same resource.
    assert (
        validate_identity_selection("my-custom", user_prompts_dirs=[prompts])
        == "my-custom"
    )


@pytest.mark.parametrize(
    "reference",
    [
        "../x",  # traversal
        "sub/x",  # nested / multi-segment
        "we:ird",  # multi-segment (colon)
        "a\\b",  # windows separator
        "bad name",  # whitespace
    ],
)
def test_bare_custom_identity_references_fail_closed_with_a_matching_file(
    tmp_path: Path, reference: str
) -> None:
    """A bare custom identity must be a plain file stem — even when a file
    exists (S2 round 3 W4).

    The bare non-builtin branch used to accept ANY reference whose
    ``<reference>.md`` file existed in a user prompts directory, so
    traversal/nested/multi-segment names validated at loop construction (the
    ResourceLoader rejects them later, at prompt assembly). The ``_CUSTOM_ID``
    fullmatch (plus the reserved-name check) now applies before the existence
    check: a matching file cannot launder a malformed reference.
    """
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    # Place the file the OLD existence-only check would have matched.
    target = prompts / f"{reference}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# target\n", encoding="utf-8")
    assert target.is_file()  # the matching file really exists
    with pytest.raises(UnknownIdentityError) as raised:
        validate_identity_selection(reference, user_prompts_dirs=[prompts])
    message = str(raised.value)
    assert "Malformed custom identity" in message
    assert repr(reference) in message


def test_validate_identity_selection_rejects_unknown_references(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    with pytest.raises(UnknownIdentityError, match="builtin:reviever"):
        validate_identity_selection("builtin:reviever", user_prompts_dirs=[prompts])
    with pytest.raises(UnknownIdentityError, match="reviever"):
        validate_identity_selection("reviever", user_prompts_dirs=[prompts])
    with pytest.raises(UnknownIdentityError, match="non-empty text"):
        validate_identity_selection("", user_prompts_dirs=[prompts])
    # No user prompts directories at all: every custom reference fails closed.
    with pytest.raises(UnknownIdentityError, match="user:my-custom"):
        validate_identity_selection("user:my-custom", user_prompts_dirs=[])
    # Builtins keep validating without any directories.
    for reference in ("coder", "builtin:researcher"):
        assert validate_identity_selection(reference, user_prompts_dirs=[]) == reference
