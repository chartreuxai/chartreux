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

from pathlib import Path
import re
import tomllib

_DOCS = Path(__file__).parents[1] / "docs"
CONFIGURATION = (_DOCS / "reference" / "configuration.md").read_text()
TOOLS_SAFETY = (_DOCS / "guides" / "tools-safety.md").read_text()
SUBAGENTS = (_DOCS / "guides" / "subagents.md").read_text()
RELEASE_NOTES = (Path(__file__).parents[1] / "CHANGELOG.md").read_text()
RELEASE_NOTES_FLAT = " ".join(RELEASE_NOTES.split())


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
    assert "Trailing-`**` over-denial" in TOOLS_SAFETY
    assert "executor shell-option contract" in TOOLS_SAFETY
    assert "Extglob residual hazard" in TOOLS_SAFETY
    assert "under-inspection hazard" in TOOLS_SAFETY
    assert "Quoted-variable conservative denial" in TOOLS_SAFETY
    assert "conservatively denied" in TOOLS_SAFETY
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
    heading = re.search(r"^## (\S+ \([^)]*\))", RELEASE_NOTES, re.MULTILINE)
    assert heading is not None
    assert pyproject_version == chartreux.__version__ == lock_version
    assert heading.group(1) == f"{pyproject_version} (2026-10-25)"
