"""Documentation assertions for the v0.5.8 operational contract.

Keeps the pinned operational contract of the v0.5.8 release honest: the root
run-budget intervals, the foreground attribution limits, the retained
shell-policy limitations, the expansion-failure provenance, and the
configured review behavior must stay documented as the runtime behaves.
The release-notes assertions below cover the shipped groups, the three
breaking changes, and the retained limitations; the version assertion pins
the version string agreement across the pyproject, the package, the
lockfile, and the release-note heading.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import re
import tomllib

import pytest

_DOCS = Path(__file__).parents[1] / "docs"
CONFIGURATION = (_DOCS / "reference" / "configuration.md").read_text()
TOOLS_SAFETY = (_DOCS / "guides" / "tools-safety.md").read_text()
SUBAGENTS = (_DOCS / "guides" / "subagents.md").read_text()
RELEASE_NOTES = (Path(__file__).parents[1] / "CHANGELOG.md").read_text()
RELEASE_NOTES_FLAT = " ".join(RELEASE_NOTES.split())

_RELEASE_HEADING = re.compile(r"^## (\S+) \(([^)]*)\)$", re.MULTILINE)
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_UNRELEASED = "unreleased"


def _release_headings(text: str) -> list[tuple[str, str]]:
    """Return the changelog version headings as (version, label) pairs."""
    return [(m.group(1), m.group(2)) for m in _RELEASE_HEADING.finditer(text)]


def _version_mismatch(
    pyproject_version: str, module_version: str, lock_version: str
) -> str | None:
    """Describe a disagreement across the version locations, else None."""
    if pyproject_version != module_version or pyproject_version != lock_version:
        return (
            f"pyproject {pyproject_version!r} != module {module_version!r} "
            f"!= lock {lock_version!r}"
        )
    return None


def _is_calendar_date(label: str) -> bool:
    """Return whether ``label`` is a real calendar date in ISO form."""
    if _ISO_DATE.fullmatch(label) is None:
        return False
    try:
        return datetime.date.fromisoformat(label).isoformat() == label
    except ValueError:
        return False


def _release_heading_mismatch(
    headings: list[tuple[str, str]], version: str
) -> str | None:
    """Describe why the release headings are invalid for ``version``, else None.

    At most one ``unreleased`` heading may appear anywhere in the changelog:
    the release transition is a single enforceable flip. The leading
    ``unreleased`` heading may be the current version's own or a pre-staged
    next version, and those two allowances cannot stack. Otherwise the
    current-version heading must carry a real calendar ISO date.
    """
    if sum(label == _UNRELEASED for _, label in headings) > 1:
        return "more than one unreleased heading"
    current = next(
        (
            index
            for index, (heading_version, _) in enumerate(headings)
            if heading_version == version
        ),
        None,
    )
    if current is None:
        return f"no heading names version {version}"
    current_label = headings[current][1]
    current_unreleased = current_label == _UNRELEASED
    if current_unreleased:
        if current == 0:
            return None
        return "the unreleased heading must lead the changelog"
    if not _is_calendar_date(current_label):
        return f"current-version heading lacks an ISO date: {current_label!r}"
    return None


def test_root_budget_interval_is_explicitly_documented() -> None:
    assert "arm the root loop, not only subagent runs" in CONFIGURATION
    assert "lazy self-anchor" in CONFIGURATION
    assert "from the first evaluation onward" in CONFIGURATION
    assert "spans pre-resume consumption" in CONFIGURATION
    assert "checkpoints, not reservations" in CONFIGURATION
    assert "Per-run budget default for subagent runs" in CONFIGURATION
    assert "measured relative to each run's admission baseline" in CONFIGURATION
    assert "budget-unverifiable" in CONFIGURATION


def test_foreground_attribution_limit_is_explicitly_documented() -> None:
    assert "Root intervals persist with their anchors" in SUBAGENTS
    assert "unanchored and unjoined" in SUBAGENTS
    assert "null run_id and its per-run coverage is unattributable" in SUBAGENTS
    assert "terminal budget stop" in SUBAGENTS
    assert "not a suspension" in SUBAGENTS
    assert "not a provider failure" in SUBAGENTS
    assert "budget-stopped" in SUBAGENTS


def test_shell_caps_and_retained_overdenials_are_documented() -> None:
    assert "### Retained shell-policy limitations" in TOOLS_SAFETY
    assert "Trailing-`**` admission" in TOOLS_SAFETY
    assert "never the recursive `**/` form" in TOOLS_SAFETY
    assert "executor shell-option contract" in TOOLS_SAFETY
    assert "Extglob environment residual" in TOOLS_SAFETY
    assert "root-owned trust base" in TOOLS_SAFETY
    assert "Quoted-variable conservative denial" in TOOLS_SAFETY
    assert 'double-quoted `"$NAME"` or `"${NAME}"` word' in TOOLS_SAFETY
    assert "prefixed assignment value" in TOOLS_SAFETY
    assert "directory-entry cap and a match-state cap" in TOOLS_SAFETY
    assert "fail-closed trade-off" in TOOLS_SAFETY
    assert "caps continued matching work," in TOOLS_SAFETY


def test_find_pattern_operand_carveout_is_documented() -> None:
    assert "Find pattern-operand carve-out" in TOOLS_SAFETY
    assert "find -name" in TOOLS_SAFETY
    assert "-ipath" in TOOLS_SAFETY
    assert "operand tokens bypass path-glob inspection" in TOOLS_SAFETY
    assert "out-of-root directory names only" in TOOLS_SAFETY


def test_expansion_failure_precision_is_documented() -> None:
    assert "### Expansion-failure provenance" in TOOLS_SAFETY
    assert "directory-entry budget" in TOOLS_SAFETY
    assert "match-state budget" in TOOLS_SAFETY
    assert "unsupported glob syntax" in TOOLS_SAFETY
    assert "attributed to a budget" in TOOLS_SAFETY
    assert "bounded, redacted" in TOOLS_SAFETY


def test_configured_review_behavior_is_documented() -> None:
    assert "Deep panels comprise all CONFIGURED seats" in SUBAGENTS
    assert "shipped route composes three seats" in SUBAGENTS
    assert "five-seat configured panel" in SUBAGENTS
    assert "launch only because the route configures them" in SUBAGENTS
    assert "no authorship substitution" in SUBAGENTS
    assert "applies only to quick single-seat reviews" in SUBAGENTS
    assert "LIVE route" in SUBAGENTS
    assert "applies to subsequent launches in the same" in SUBAGENTS
    assert "children already admitted keep their bindings" in SUBAGENTS
    assert "rejected at reload" in SUBAGENTS
    assert "leaving the prior table effective" in SUBAGENTS
    assert "FROZEN route" not in SUBAGENTS
    assert "configured multi-model route" in SUBAGENTS


def test_release_notes_cover_token_efficiency_user_changes() -> None:
    # The six user-visible groups shipped earlier in the cycle.
    assert (
        "Redirect targets that contain shell glob characters are denied"
        in RELEASE_NOTES_FLAT
    )
    assert "inspection surface of the `grep` family" in RELEASE_NOTES_FLAT
    assert "`rg`" in RELEASE_NOTES_FLAT
    assert "[subagents].run_max_price" in RELEASE_NOTES_FLAT
    assert "[subagents].run_max_session_tokens" in RELEASE_NOTES_FLAT
    assert "Usage ledger schema v2" in RELEASE_NOTES_FLAT
    assert "v1 ledgers still parse" in RELEASE_NOTES_FLAT
    assert "outline mode" in RELEASE_NOTES_FLAT
    assert "files-only mode" in RELEASE_NOTES_FLAT
    assert "UTF-8-byte-honest" in RELEASE_NOTES_FLAT
    # The changes landing with this release.
    assert "zai-glm-5-3-flash" in RELEASE_NOTES_FLAT
    assert "Expansion-failure provenance" in RELEASE_NOTES_FLAT
    assert "terminal budget stop" in RELEASE_NOTES_FLAT
    assert "characterization tests" in RELEASE_NOTES_FLAT
    assert "Review-system alignment" in RELEASE_NOTES_FLAT
    assert "receipt policy" in RELEASE_NOTES_FLAT
    assert "roster-neutral" in RELEASE_NOTES_FLAT


def test_release_notes_name_the_three_breaking_changes() -> None:
    assert "### Breaking" in RELEASE_NOTES
    assert (
        "Selecting thinking `off` for flash is now a typed launch-time "
        "configuration error" in RELEASE_NOTES_FLAT
    )
    assert "Selecting thinking `low` now wires real thinking" in RELEASE_NOTES_FLAT
    assert "The glob-denial message wording changed" in RELEASE_NOTES_FLAT


def test_release_notes_do_not_advertise_deferred_fixes() -> None:
    assert "### Retained limitations" in RELEASE_NOTES
    assert (
        "the trailing-`**` over-denial (until the executor shell-option "
        "contract lands)" in RELEASE_NOTES_FLAT
    )
    assert (
        "the extglob residual hazard (detection and modeling fast-follow"
        in RELEASE_NOTES_FLAT
    )
    assert (
        "the quoted-variable conservative denial (resolution fast-follow)"
        in RELEASE_NOTES_FLAT
    )
    # No deferred fix may be advertised as done.
    assert "trailing-`**` over-denial is fixed" not in RELEASE_NOTES_FLAT
    assert "extglob is now inspected" not in RELEASE_NOTES_FLAT
    assert "quoted variables are now resolved" not in RELEASE_NOTES_FLAT
    assert "no longer over-denies" not in RELEASE_NOTES_FLAT
    assert "caps are raised" not in RELEASE_NOTES_FLAT


def test_version_locations_and_release_heading_agree() -> None:
    import chartreux

    root = Path(__file__).parents[1]
    pyproject_version = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "version"
    ]
    lock = tomllib.loads((root / "uv.lock").read_text())
    lock_version = next(
        package["version"]
        for package in lock["package"]
        if package["name"] == "chartreux"
    )
    version_problem = _version_mismatch(
        pyproject_version, chartreux.__version__, lock_version
    )
    assert version_problem is None, version_problem
    heading_problem = _release_heading_mismatch(
        _release_headings(RELEASE_NOTES), pyproject_version
    )
    assert heading_problem is None, heading_problem


def test_version_locations_must_agree() -> None:
    assert _version_mismatch("0.7.8", "0.7.8", "0.7.8") is None
    assert _version_mismatch("0.7.8", "0.7.9", "0.7.8") is not None
    assert _version_mismatch("0.7.8", "0.7.8", "0.7.9") is not None


@pytest.mark.parametrize(
    ("headings", "version"),
    [
        ([("0.7.9", "unreleased"), ("0.7.8", "2026-11-12")], "0.7.8"),
        ([("0.7.8", "unreleased")], "0.7.8"),
        ([("0.7.8", "2026-11-12")], "0.7.8"),
        ([("0.7.8", "2028-02-29")], "0.7.8"),
        (
            [("0.7.9", "unreleased"), ("0.7.8", "2026-11-12"), ("0.4.0", "2026-10-02")],
            "0.7.9",
        ),
    ],
    ids=[
        "next-unreleased",
        "current-unreleased",
        "released",
        "leap-day",
        "historical-dated",
    ],
)
def test_release_heading_shapes_are_accepted(
    headings: list[tuple[str, str]], version: str
) -> None:
    assert _release_heading_mismatch(headings, version) is None


@pytest.mark.parametrize(
    ("headings", "version"),
    [
        ([("0.7.7", "2026-11-11")], "0.7.8"),
        ([("0.7.8", "2026-1-7")], "0.7.8"),
        ([("0.7.8", "2026-99-99")], "0.7.8"),
        ([("0.7.8", "2026-02-30")], "0.7.8"),
        ([("0.7.9", "unreleased"), ("0.7.8", "unreleased")], "0.7.8"),
        ([("0.7.9", "unreleased"), ("0.7.8", "unreleased")], "0.7.9"),
        (
            [("0.7.9", "unreleased"), ("0.7.8", "2026-11-12"), ("0.4.0", "unreleased")],
            "0.7.9",
        ),
    ],
    ids=[
        "missing-heading",
        "malformed-date",
        "impossible-month",
        "impossible-day",
        "two-unreleased",
        "two-unreleased-lead",
        "historical-unreleased",
    ],
)
def test_release_heading_shapes_are_rejected(
    headings: list[tuple[str, str]], version: str
) -> None:
    assert _release_heading_mismatch(headings, version) is not None
