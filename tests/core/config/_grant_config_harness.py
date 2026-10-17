"""Shared parametrized suite for the grant-config authority twins (W6a, stage 12a).

The twin files (``test_root_persistence.py`` / ``test_root_authority.py`` /
``test_root_source_integration.py`` on the root side vs
``test_dangerous_command_persistence.py`` / ``test_dangerous_command_authority.py``
on the dangerous-command side) copy-adapted the same concerns twice: once for root
authority (``authorized_roots_by_project``) and once for dangerous-command grants
(``dangerous_command_grants_by_project``). This harness runs each migrated concern
once per feature. :class:`GrantConfigFeature` carries the feature's field, field
readers (restrictions values, config projection, item values, projection map),
record constructor (``make_read`` / ``read_values``), save/read adapters, and sample
values, so every case asserts the UNION of both twins' assertions. Feature-specific
concerns (grant removal, non-persistable operations, relative save arguments,
symlink reinterpretation, the settings-catalog exclusion, workspace allowance) stay
in the feature-specific modules. The suite is collected through
``test_grant_config_harness.py``, which re-exports these cases under stable node
IDs; per-pair difference audits and survival mappings live in the stage-12 report.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
import stat
import threading
import tomllib
from typing import IO, Any, Literal
from unittest.mock import patch

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._dangerous_command_authority import (
    GRANTS_FIELD,
    ProjectDangerousCommandGrant,
    validate_grant_source,
)
from chartreux.core.config._dangerous_command_persistence import (
    SavedGrantsRead,
    read_saved_dangerous_command_grants,
)
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config._root_authority import (
    ROOTS_FIELD,
    ProjectRootAuthority,
    validate_root_source,
)
from chartreux.core.config._root_persistence import SavedRootsRead, read_saved_roots
from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.fingerprint import create_file_fingerprint
from chartreux.core.config.layer import LayerImplementationError, RawConfig
from chartreux.core.config.layers import _base
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.environment import EnvironmentLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import (
    ConfigOrchestrator,
    ConfigPatchValidationError,
)
from chartreux.core.config.patch import (
    AddOperationPatch,
    PatchOp,
    RemoveOperationPatch,
    ReplaceOperationPatch,
)
from chartreux.core.config.types import (
    MISSING_BACKING_STORE_DATA_FINGERPRINT,
    ConfigSaveResult,
)
from chartreux.core.trusted_folders import trusted_folders_manager
from chartreux.core.workspace import Workspace

PUSH = "git.push"


async def _save_root(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    project: Path,
    value: Any,
    expected_revision: str,
) -> ConfigSaveResult:
    return await orchestrator.save_project_root_grant(
        project=project, root=value, expected_revision=expected_revision
    )


async def _save_grant(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    project: Path,
    value: Any,
    expected_revision: str,
) -> ConfigSaveResult:
    return await orchestrator.save_dangerous_command_grant(
        project=project, operation_id=value, expected_revision=expected_revision
    )


async def _read_roots(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], project: Path
) -> SavedRootsRead:
    return await read_saved_roots(orchestrator, project=project)


async def _read_grants(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], project: Path
) -> SavedGrantsRead:
    return await read_saved_dangerous_command_grants(orchestrator, project=project)


def _make_roots_read(
    values: tuple[str, ...], user_revision: str | None
) -> SavedRootsRead:
    return SavedRootsRead(roots=values, user_revision=user_revision)


def _make_grants_read(
    values: tuple[str, ...], user_revision: str | None
) -> SavedGrantsRead:
    return SavedGrantsRead(operations=values, user_revision=user_revision)


def _roots_unavailable(reason: Literal["no_user_source", "read"]) -> SavedRootsRead:
    return SavedRootsRead(unavailable=reason)


def _grants_unavailable(reason: Literal["no_user_source", "read"]) -> SavedGrantsRead:
    return SavedGrantsRead(unavailable=reason)


def _roots_values(record: SavedRootsRead) -> tuple[str, ...]:
    return record.roots


def _grants_values(record: SavedGrantsRead) -> tuple[str, ...]:
    return record.operations


def _roots_restrictions(
    restriction: SourceRestrictions,
) -> tuple[ProjectRootAuthority, ...]:
    return restriction.authorized_roots


def _grants_restrictions(
    restriction: SourceRestrictions,
) -> tuple[ProjectDangerousCommandGrant, ...]:
    return restriction.dangerous_command_grants


def _roots_projection(config: ChartreuxConfigSchema) -> dict[str, list[str]]:
    return config.authorized_roots_by_project


def _grants_projection(config: ChartreuxConfigSchema) -> dict[str, list[str]]:
    return config.dangerous_command_grants_by_project


def _root_item_values(item: ProjectRootAuthority) -> tuple[Path, ...]:
    return item.roots


def _grant_item_values(item: ProjectDangerousCommandGrant) -> tuple[str, ...]:
    return item.operations


def _roots_projection_map(restriction: SourceRestrictions) -> dict[str, list[str]]:
    return restriction.project_roots()


def _grants_projection_map(restriction: SourceRestrictions) -> dict[str, list[str]]:
    return restriction.project_grants()


def _root_sample(tmp_path: Path) -> Path:
    return tmp_path / "extra"


def _grant_sample(tmp_path: Path) -> str:
    return PUSH


def _root_authority_value(tmp_path: Path) -> str:
    return str(tmp_path.parent)


def _grant_authority_value(tmp_path: Path) -> str:
    return PUSH


def _root_named_value(tmp_path: Path, name: str) -> str:
    return str(tmp_path / name)


def _grant_named_value(tmp_path: Path, name: str) -> str:
    return PUSH


def _root_generic_injection(tmp_path: Path) -> dict[str, list[str]]:
    return {
        str(tmp_path): [str(tmp_path / "injected")],
        str(tmp_path / "other"): [str(tmp_path / "other-injected")],
    }


def _grant_generic_injection(tmp_path: Path) -> dict[str, list[str]]:
    return {str(tmp_path): [PUSH], str(tmp_path / "other"): [PUSH]}


def _root_patch_injected_value(tmp_path: Path) -> str:
    return str(tmp_path / "new-secret-root")


def _root_patch_injected_marker(tmp_path: Path) -> str:
    return "new-secret-root"


def _grant_patch_injected_value(tmp_path: Path) -> str:
    return PUSH


def _grant_patch_injected_marker(tmp_path: Path) -> str:
    return PUSH


def _root_concurrent_targets(
    tmp_path: Path,
) -> tuple[tuple[Path, Any], tuple[Path, Any]]:
    return (tmp_path, tmp_path / "first"), (tmp_path, tmp_path / "second")


def _root_concurrent_candidates(tmp_path: Path) -> frozenset[str]:
    return frozenset({str(tmp_path / "first"), str(tmp_path / "second")})


def _grant_concurrent_targets(
    tmp_path: Path,
) -> tuple[tuple[Path, Any], tuple[Path, Any]]:
    return (tmp_path, PUSH), (tmp_path / "other", PUSH)


def _grant_concurrent_candidates(tmp_path: Path) -> frozenset[str]:
    return frozenset({PUSH})


def _root_canonicalization_probe(tmp_path: Path) -> tuple[Path, Any]:
    return tmp_path, tmp_path / "sub" / ".." / "extra2"


def _grant_canonicalization_probe(tmp_path: Path) -> tuple[Path, Any]:
    return tmp_path / "sub" / "..", PUSH


def _root_accumulated_values(tmp_path: Path) -> list[str]:
    return [str(tmp_path / "extra"), str(tmp_path / "extra2")]


def _grant_accumulated_values(tmp_path: Path) -> list[str]:
    return [PUSH]


def _root_expected_authority_item_values(tmp_path: Path) -> tuple[Any, ...]:
    return (tmp_path.parent,)


def _grant_expected_authority_item_values(tmp_path: Path) -> tuple[Any, ...]:
    return (PUSH,)


@dataclass(frozen=True)
class GrantConfigFeature:
    """One grant-config authority feature as the shared suite parameterizes it.

    ``restrictions_reader`` / ``item_values`` / ``projection_map`` are the field
    readers, ``make_read`` / ``make_unavailable`` / ``read_values`` the record
    constructor, and the ``*_value`` factories derive the feature's sample values
    from ``tmp_path`` (roots sample paths under it; grants sample the one
    persistable operation).
    """

    name: str
    field: str
    validate_source: Callable[..., tuple[Any, ...]]
    save: Callable[..., Coroutine[Any, Any, ConfigSaveResult]]
    read: Callable[..., Awaitable[Any]]
    make_read: Callable[[tuple[str, ...], str | None], Any]
    make_unavailable: Callable[[Literal["no_user_source", "read"]], Any]
    read_values: Callable[[Any], tuple[str, ...]]
    restrictions_reader: Callable[[SourceRestrictions], tuple[Any, ...]]
    config_projection: Callable[[ChartreuxConfigSchema], dict[str, list[str]]]
    item_values: Callable[[Any], tuple[Any, ...]]
    projection_map: Callable[[SourceRestrictions], dict[str, list[str]]]
    item_values_attr: str
    invalid_definition_message: str
    provenance_match: str
    collision_match: str
    workspace_check: bool
    supports_remove: bool
    secret_marker: str
    invalid_definitions: tuple[Any, ...]
    malformed_disk_contents: tuple[str, ...]
    unreadable_disk_contents: tuple[str, ...]
    patch_operations: tuple[str, ...]
    sample_value: Callable[[Path], Any]
    authority_value: Callable[[Path], str]
    named_value: Callable[[Path, str], str]
    generic_injection: Callable[[Path], dict[str, list[str]]]
    patch_injected_value: Callable[[Path], str]
    patch_injected_marker: Callable[[Path], str]
    concurrent_targets: Callable[[Path], tuple[tuple[Path, Any], tuple[Path, Any]]]
    concurrent_candidates: Callable[[Path], frozenset[str]]
    canonicalization_probe: Callable[[Path], tuple[Path, Any]]
    accumulated_values: Callable[[Path], list[str]]
    expected_authority_item_values: Callable[[Path], tuple[Any, ...]]


ROOTS = GrantConfigFeature(
    name="roots",
    field=ROOTS_FIELD,
    validate_source=validate_root_source,
    save=_save_root,
    read=_read_roots,
    make_read=_make_roots_read,
    make_unavailable=_roots_unavailable,
    read_values=_roots_values,
    restrictions_reader=_roots_restrictions,
    config_projection=_roots_projection,
    item_values=_root_item_values,
    projection_map=_roots_projection_map,
    item_values_attr="roots",
    invalid_definition_message=f"Invalid {ROOTS_FIELD} definition (source: user)",
    provenance_match="root source provenance unavailable",
    collision_match=f"Invalid {ROOTS_FIELD}",
    workspace_check=True,
    supports_remove=False,
    secret_marker="secret",
    invalid_definitions=(
        None,
        [],
        {"relative": []},
        {"/project": ["relative"]},
        {"/project": [42]},
        {"/project": "secret-marker"},
    ),
    malformed_disk_contents=(
        'authorized_roots_by_project = "nope"\n',
        f'[{ROOTS_FIELD}]\n"relative-project" = []\n',
        "this is not valid = = = toml [[[\n",
    ),
    unreadable_disk_contents=(
        "this is not valid = = = toml [[[\n",
        f'[{ROOTS_FIELD}]\n"relative-project" = []\n',
    ),
    patch_operations=("set", "remove", "document", "erase-document", "set-remove"),
    sample_value=_root_sample,
    authority_value=_root_authority_value,
    named_value=_root_named_value,
    generic_injection=_root_generic_injection,
    patch_injected_value=_root_patch_injected_value,
    patch_injected_marker=_root_patch_injected_marker,
    concurrent_targets=_root_concurrent_targets,
    concurrent_candidates=_root_concurrent_candidates,
    canonicalization_probe=_root_canonicalization_probe,
    accumulated_values=_root_accumulated_values,
    expected_authority_item_values=_root_expected_authority_item_values,
)

GRANTS = GrantConfigFeature(
    name="grants",
    field=GRANTS_FIELD,
    validate_source=validate_grant_source,
    save=_save_grant,
    read=_read_grants,
    make_read=_make_grants_read,
    make_unavailable=_grants_unavailable,
    read_values=_grants_values,
    restrictions_reader=_grants_restrictions,
    config_projection=_grants_projection,
    item_values=_grant_item_values,
    projection_map=_grants_projection_map,
    item_values_attr="operations",
    invalid_definition_message=f"Invalid {GRANTS_FIELD} definition (source: user)",
    provenance_match="grant source provenance unavailable",
    collision_match=f"Invalid {GRANTS_FIELD}",
    workspace_check=False,
    supports_remove=True,
    secret_marker="git.push.force",
    invalid_definitions=(
        None,
        [],
        {"relative-project": [PUSH]},
        {"/project": ["git.push.force"]},
        {"/project": ["git.push.delete"]},
        {"/project": ["git.checkout"]},
        {"/project": ["git.push.unregistered"]},
        {"/project": [42]},
        {"/project": PUSH},
        {"/project": [PUSH, "git.push.force"]},
    ),
    malformed_disk_contents=(
        f'{GRANTS_FIELD} = "nope"\n',
        f'[{GRANTS_FIELD}]\n"relative-project" = ["{PUSH}"]\n',
        f'[{GRANTS_FIELD}]\n"/project" = ["git.push.force"]\n',
        "this is not valid = = = toml [[[\n",
    ),
    unreadable_disk_contents=(
        "this is not valid = = = toml [[[\n",
        f'[{GRANTS_FIELD}]\n"relative-project" = ["{PUSH}"]\n',
        f'[{GRANTS_FIELD}]\n"/project" = ["git.push.force"]\n',
    ),
    patch_operations=(
        "set",
        "replace",
        "remove",
        "document",
        "erase-document",
        "set-remove",
    ),
    sample_value=_grant_sample,
    authority_value=_grant_authority_value,
    named_value=_grant_named_value,
    generic_injection=_grant_generic_injection,
    patch_injected_value=_grant_patch_injected_value,
    patch_injected_marker=_grant_patch_injected_marker,
    concurrent_targets=_grant_concurrent_targets,
    concurrent_candidates=_grant_concurrent_candidates,
    canonicalization_probe=_grant_canonicalization_probe,
    accumulated_values=_grant_accumulated_values,
    expected_authority_item_values=_grant_expected_authority_item_values,
)

FEATURES: tuple[GrantConfigFeature, ...] = (ROOTS, GRANTS)
FEATURE_IDS = [feature.name for feature in FEATURES]


def sample_str(feature: GrantConfigFeature, tmp_path: Path) -> str:
    """The feature's default sample value as the string persisted to disk."""
    return str(feature.sample_value(tmp_path))


async def make_orchestrator(path: Path) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    user = UserConfigLayer(path=path)
    session = OverridesLayer(data={})
    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, session],
        default_layer_resolver=lambda: session,
    )


def live_state(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> tuple[object, object, object, object]:
    layer = orchestrator.get_layer("user-toml")
    return (
        orchestrator.config,
        orchestrator.restrictions,
        orchestrator.accepted_token,
        (layer.cached_data, layer.fingerprint),
    )


async def user_orchestrator(
    feature: GrantConfigFeature, tmp_path: Path, *, present: bool = True
) -> tuple[ConfigOrchestrator[ChartreuxConfigSchema], Path]:
    path = tmp_path / "user.toml"
    path.write_text(
        f'[{feature.field}]\n"{tmp_path}" = ["{sample_str(feature, tmp_path)}"]\n'
        if present
        else ""
    )
    user = UserConfigLayer(path=path, name="renamed-user")
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user],
        default_layer_resolver=lambda: user,
    )
    return orchestrator, path


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_harness_smoke_user_orchestrator_round_trip(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    """Harness smoke: the feature machinery drives a real accepted authority."""
    orchestrator, _ = await user_orchestrator(feature, tmp_path)
    accepted = feature.restrictions_reader(orchestrator.restrictions[-1])
    assert accepted[0].project == tmp_path
    assert feature.item_values(accepted[0]) == (feature.sample_value(tmp_path),)
    assert feature.config_projection(orchestrator.config) == {
        str(tmp_path): [sample_str(feature, tmp_path)]
    }


def _external_write_between_revision_reads(
    path: Path, external: str
) -> Callable[[IO[bytes]], str]:
    """Build a fingerprint hook that rewrites *path* on the second target read.

    The checked write fingerprints the target twice: once before the temp
    write and once under the lock immediately before the replace. The second
    read observes *external*, exactly as if a non-cooperating writer replaced
    the store during the temp write.
    """
    original = _base.create_file_fingerprint
    target_reads = 0

    def intervening(file: IO[bytes]) -> str:
        nonlocal target_reads
        if Path(file.name) == path:
            target_reads += 1
            if target_reads == 2:
                path.write_text(external)
        return original(file)

    return intervening


# ---------------------------------------------------------------------------
# Persistence pairs (migrated in 12a-ii from test_root_persistence.py and
# test_dangerous_command_persistence.py; each case asserts the union of both
# twins' assertions — see the stage-12 report for the per-pair audits).
# ---------------------------------------------------------------------------

_MALFORMED_DISK_CASES = [
    *[(ROOTS, content) for content in ROOTS.malformed_disk_contents],
    *[(GRANTS, content) for content in GRANTS.malformed_disk_contents],
]
_MALFORMED_DISK_IDS = [
    f"{feature.name}-{index}"
    for feature in FEATURES
    for index, _ in enumerate(feature.malformed_disk_contents)
]

_UNREADABLE_DISK_CASES = [
    *[(ROOTS, content) for content in ROOTS.unreadable_disk_contents],
    *[(GRANTS, content) for content in GRANTS.unreadable_disk_contents],
]
_UNREADABLE_DISK_IDS = [
    f"{feature.name}-{index}"
    for feature in FEATURES
    for index, _ in enumerate(feature.unreadable_disk_contents)
]


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_saves_grant_preserving_disk_content_without_publishing(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    other = tmp_path / "other-project"
    path = tmp_path / "config.toml"
    path.write_text(
        'theme = "dark"\n'
        '\n[tools.bash]\ndenylist = ["user-denial"]\n'
        "\n"
        f"[{feature.field}]\n"
        f'"{other}" = ["{feature.named_value(tmp_path, "other-root")}"]\n'
    )
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)
    projection_before = feature.config_projection(orchestrator.config)

    result = await feature.save(
        orchestrator, tmp_path, feature.sample_value(tmp_path), expected
    )

    assert result.error is None
    assert result.persistence == "saved"
    assert result.application == "unchanged"
    with path.open("rb") as file:
        assert result.revision == create_file_fingerprint(file)
    data = tomllib.loads(path.read_text())
    assert data["theme"] == "dark"
    assert data["tools"] == {"bash": {"denylist": ["user-denial"]}}
    assert data[feature.field] == {
        str(other): [feature.named_value(tmp_path, "other-root")],
        str(tmp_path): [sample_str(feature, tmp_path)],
    }
    # Runtime publication never happened: the accepted snapshot, restrictions,
    # token, live layer cache, and the effective projection all still describe
    # the pre-save state.
    assert live_state(orchestrator) == before
    assert feature.config_projection(orchestrator.config) == projection_before
    assert feature.restrictions_reader(orchestrator.restrictions[-1]) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_absent_user_config_file_is_created(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    assert layer.fingerprint == MISSING_BACKING_STORE_DATA_FINGERPRINT
    before = live_state(orchestrator)

    result = await feature.save(
        orchestrator,
        tmp_path,
        feature.sample_value(tmp_path),
        MISSING_BACKING_STORE_DATA_FINGERPRINT,
    )

    assert result.error is None
    assert result.persistence == "saved"
    assert result.application == "unchanged"
    assert path.exists()
    assert tomllib.loads(path.read_text()) == {
        feature.field: {str(tmp_path): [sample_str(feature, tmp_path)]}
    }
    # The live cache still describes the absent file: nothing was published.
    assert layer.fingerprint == MISSING_BACKING_STORE_DATA_FINGERPRINT
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_stale_expected_revision_fails_cleanly(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    stale = layer.fingerprint
    assert stale is not None
    before = live_state(orchestrator)
    external = 'theme = "light"\n'
    path.write_text(external)

    for revision in (stale, "vibe:bogus"):
        result = await feature.save(
            orchestrator, tmp_path, feature.sample_value(tmp_path), revision
        )
        assert result.persistence == "not_saved"
        assert result.application == "unchanged"
        assert result.error == "conflict"
        assert result.revision is None

    assert path.read_text() == external
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_intervening_write_between_revision_read_and_replace_conflicts(
    feature: GrantConfigFeature, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)
    external = 'theme = "light"\n# written between the revision reads\n'
    monkeypatch.setattr(
        _base,
        "create_file_fingerprint",
        _external_write_between_revision_reads(path, external),
    )

    result = await feature.save(
        orchestrator, tmp_path, feature.sample_value(tmp_path), expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "conflict"
    assert result.revision is None
    # The intervening writer's content survives byte-for-byte.
    assert path.read_text() == external
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "disk_content"), _MALFORMED_DISK_CASES, ids=_MALFORMED_DISK_IDS
)
async def test_malformed_disk_source_is_typed_error(
    feature: GrantConfigFeature, tmp_path: Path, disk_content: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)
    path.write_text(disk_content)
    with path.open("rb") as file:
        fresh = create_file_fingerprint(file)

    result = await feature.save(
        orchestrator, tmp_path, feature.sample_value(tmp_path), fresh
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert result.revision is None
    assert path.read_text() == disk_content
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_project_layer_is_not_a_user_target(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    project_dir = tmp_path / "project"
    config_file = project_dir / ".chartreux" / "config.toml"
    config_file.parent.mkdir(parents=True)
    trusted_folders_manager.trust_for_session(config_file.parent)
    project = ProjectConfigLayer(path=project_dir)
    session = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), project, session],
        default_layer_resolver=lambda: session,
    )

    result = await feature.save(
        orchestrator, project_dir, feature.sample_value(tmp_path), "any"
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert result.revision is None
    assert not config_file.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_subclass_cannot_persist_authority(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    class FakeUserLayer(UserConfigLayer):
        pass

    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    fake = FakeUserLayer(path=path)
    session = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), fake, session],
        default_layer_resolver=lambda: session,
    )
    expected = fake.fingerprint
    assert expected is not None

    result = await feature.save(
        orchestrator, tmp_path, feature.sample_value(tmp_path), expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert path.read_text() == 'theme = "dark"\n'


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_child_orchestrator_rejected(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before_bytes = path.read_bytes()
    children = [orchestrator._copy_for_child(), orchestrator._copy_for_child().copy()]

    for child in children:
        result = await feature.save(
            child, tmp_path, feature.sample_value(tmp_path), "any"
        )
        assert result.persistence == "not_saved"
        assert result.application == "unchanged"
        assert result.error == "validation"
        assert result.revision is None
        if feature.supports_remove:
            removed = await child.remove_dangerous_command_grant(
                project=tmp_path, operation_id=PUSH, expected_revision="any"
            )
            assert removed.persistence == "not_saved"
            assert removed.application == "unchanged"
            assert removed.error == "validation"
            assert removed.revision is None

    assert path.read_bytes() == before_bytes
    # The root orchestrator is unaffected and its mutation lock is free.
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    result = await feature.save(
        orchestrator, tmp_path, feature.sample_value(tmp_path), expected
    )
    assert result.persistence == "saved"
    assert not orchestrator._mutation_lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_concurrent_saves_serialize_under_the_lock(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    before = live_state(orchestrator)
    (first_project, first_value), (second_project, second_value) = (
        feature.concurrent_targets(tmp_path)
    )
    candidates = feature.concurrent_candidates(tmp_path)

    first = feature.save(orchestrator, first_project, first_value, expected)
    second = feature.save(orchestrator, second_project, second_value, expected)
    results = await asyncio.gather(first, second)

    # One save wins the fresh disk revision; the other conflicts cleanly
    # instead of interleaving a second write or a partial merge.
    assert sorted((r.persistence, r.error or "") for r in results) == [
        ("not_saved", "conflict"),
        ("saved", ""),
    ]
    data = tomllib.loads(path.read_text())
    # Exactly one project entry survives, holding exactly one of the two
    # candidate values (the roots twin asserted the entry directly; the grants
    # twin asserted the entry count).
    assert len(data[feature.field]) == 1
    entry = next(iter(data[feature.field].values()))
    assert len(entry) == 1
    assert entry[0] in candidates
    assert data["theme"] == "dark"
    assert not orchestrator._mutation_lock.locked()
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_saves_accumulate_and_deduplicate_canonical_values(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)

    first = await feature.save(
        orchestrator,
        tmp_path,
        feature.sample_value(tmp_path),
        MISSING_BACKING_STORE_DATA_FINGERPRINT,
    )
    assert first.persistence == "saved"
    assert first.revision is not None
    # A non-canonical spelling canonicalizes on save (the roots twin probed a
    # non-canonical root spelling; the grants twin a non-canonical project key).
    probe_project, probe_value = feature.canonicalization_probe(tmp_path)
    second = await feature.save(
        orchestrator, probe_project, probe_value, first.revision
    )
    assert second.persistence == "saved"
    assert second.revision is not None
    # A duplicate canonical value merges as a no-op.
    third = await feature.save(
        orchestrator, tmp_path, feature.sample_value(tmp_path), second.revision
    )
    assert third.persistence == "saved"

    assert tomllib.loads(path.read_text())[feature.field] == {
        str(tmp_path): feature.accumulated_values(tmp_path)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_saved_values_reload_through_user_layer(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)
    saved = await feature.save(
        orchestrator,
        tmp_path,
        feature.sample_value(tmp_path),
        MISSING_BACKING_STORE_DATA_FINGERPRINT,
    )
    assert saved.persistence == "saved"

    reloaded = await make_orchestrator(path)
    assert feature.config_projection(reloaded.config) == {
        str(tmp_path): [sample_str(feature, tmp_path)]
    }
    authority = next(r for r in reloaded.restrictions if r.layer_name == "user-toml")
    values = feature.restrictions_reader(authority)
    assert values[0].project == tmp_path
    assert feature.item_values(values[0]) == (feature.sample_value(tmp_path),)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_write", [False, True])
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_cancelled_save_reports_without_publishing(
    feature: GrantConfigFeature,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_write: bool,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = _base._write_toml_snapshot

    def blocked(
        path: Path, data: object, *, expected_revision: str | None = None
    ) -> str:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        if fail_write:
            raise OSError("secret-sentinel")
        return original(path, data, expected_revision=expected_revision)  # type: ignore[arg-type]

    monkeypatch.setattr(_base, "_write_toml_snapshot", blocked)
    task = asyncio.create_task(
        feature.save(orchestrator, tmp_path, feature.sample_value(tmp_path), expected)
    )
    await asyncio.wait_for(started.wait(), 5)
    try:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert orchestrator._mutation_lock.locked()
    finally:
        release.set()
    result = await asyncio.wait_for(task, 5)

    if fail_write:
        assert result.persistence == "not_saved"
        assert result.error == "write"
        assert result.revision is None
        assert path.read_text() == 'theme = "dark"\n'
    else:
        assert result.persistence == "saved"
        assert result.application == "unchanged"
        assert result.error == "cancelled"
        assert result.revision is not None
        assert tomllib.loads(path.read_text())[feature.field] == {
            str(tmp_path): [sample_str(feature, tmp_path)]
        }
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_durability_uncertainty_reported_without_publishing(
    feature: GrantConfigFeature, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    original_fsync = _base.os.fsync

    def fsync(fd: int) -> None:
        if stat.S_ISDIR(_base.os.fstat(fd).st_mode):
            raise OSError("secret-sentinel")
        original_fsync(fd)

    monkeypatch.setattr(_base.os, "fsync", fsync)
    result = await feature.save(
        orchestrator, tmp_path, feature.sample_value(tmp_path), expected
    )

    assert "secret-sentinel" not in repr(result)
    assert result.persistence == "durability_uncertain"
    assert result.application == "unchanged"
    assert result.error is None
    assert result.revision is not None
    # The atomic replacement completed; only directory durability is uncertain.
    with path.open("rb") as file:
        assert result.revision == create_file_fingerprint(file)
    data = tomllib.loads(path.read_text())
    assert data["theme"] == "dark"
    assert data[feature.field] == {str(tmp_path): [sample_str(feature, tmp_path)]}
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_write_failure_reports_write_error(
    feature: GrantConfigFeature, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None

    def deny_write(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(_base.tempfile, "NamedTemporaryFile", deny_write)
    result = await feature.save(
        orchestrator, tmp_path, feature.sample_value(tmp_path), expected
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "write"
    assert result.revision is None
    assert path.read_text() == 'theme = "dark"\n'
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_generic_field_rejection_preserved(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    expected = layer.fingerprint
    assert expected is not None
    saved = await feature.save(
        orchestrator, tmp_path, feature.sample_value(tmp_path), expected
    )
    assert saved.persistence == "saved"
    assert saved.revision is not None
    before_bytes = path.read_bytes()
    injected = feature.generic_injection(tmp_path)

    with pytest.raises(ConfigPatchValidationError, match=feature.field):
        await orchestrator.apply_patch(
            [AddOperationPatch(path=f"/{feature.field}", value=injected)],
            reason="generic",
        )
    result = await orchestrator.save(
        [AddOperationPatch(path=f"/{feature.field}", value=injected)],
        target="user",
        expected_revision=saved.revision,
        reason="generic",
    )

    assert result.persistence == "not_saved"
    assert result.application == "unchanged"
    assert result.error == "validation"
    assert path.read_bytes() == before_bytes
    assert tomllib.loads(path.read_text())[feature.field] == {
        str(tmp_path): [sample_str(feature, tmp_path)]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_read_reflects_a_freshly_saved_grant(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)
    saved = await feature.save(
        orchestrator,
        tmp_path,
        feature.sample_value(tmp_path),
        MISSING_BACKING_STORE_DATA_FINGERPRINT,
    )
    assert saved.persistence == "saved"
    layer = orchestrator.get_layer("user-toml")
    before = live_state(orchestrator)

    read = await feature.read(orchestrator, tmp_path)

    assert read == feature.make_read((sample_str(feature, tmp_path),), saved.revision)
    assert read.unavailable is None
    with path.open("rb") as file:
        assert read.user_revision == create_file_fingerprint(file)
    # The read published nothing and did not adopt the saved file into the
    # live cache: the layer still describes the absent file.
    assert live_state(orchestrator) == before
    assert layer.fingerprint == MISSING_BACKING_STORE_DATA_FINGERPRINT


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_read_reflects_disk_after_an_external_write(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        f'[{feature.field}]\n"{tmp_path}" = ["{feature.named_value(tmp_path, "cached")}"]\n'
    )
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")
    cached_revision = layer.fingerprint
    assert cached_revision is not None
    before = live_state(orchestrator)
    path.write_text(
        f"[{feature.field}]\n"
        f'"{tmp_path}" = ["{feature.named_value(tmp_path, "external")}"]\n'
        f'"{tmp_path / "other"}" = ["{feature.named_value(tmp_path, "other-root")}"]\n'
    )

    read = await feature.read(orchestrator, tmp_path)

    assert read.unavailable is None
    assert feature.read_values(read) == (feature.named_value(tmp_path, "external"),)
    with path.open("rb") as file:
        assert read.user_revision == create_file_fingerprint(file)
    assert read.user_revision != cached_revision
    # The live cache is stale and untouched: the read saw disk, not memory.
    assert layer.fingerprint == cached_revision
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_read_missing_file_is_empty_not_unavailable(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)

    read = await feature.read(orchestrator, tmp_path)

    assert read == feature.make_read((), MISSING_BACKING_STORE_DATA_FINGERPRINT)
    assert read.unavailable is None
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "disk_content"), _UNREADABLE_DISK_CASES, ids=_UNREADABLE_DISK_IDS
)
async def test_read_unreadable_source_is_unavailable_not_empty(
    feature: GrantConfigFeature, tmp_path: Path, disk_content: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)
    path.write_text(disk_content)

    read = await feature.read(orchestrator, tmp_path)

    assert read == feature.make_unavailable("read")
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_read_does_not_leak_other_projects(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    other = tmp_path / "other-project"
    path = tmp_path / "config.toml"
    path.write_text(
        f"[{feature.field}]\n"
        f'"{tmp_path}" = ["{feature.named_value(tmp_path, "mine")}"]\n'
        f'"{other}" = ["{feature.named_value(tmp_path, "theirs")}"]\n'
    )
    orchestrator = await make_orchestrator(path)
    layer = orchestrator.get_layer("user-toml")

    mine = await feature.read(orchestrator, tmp_path)
    theirs = await feature.read(orchestrator, other)
    absent = await feature.read(orchestrator, tmp_path / "absent")

    assert mine.unavailable is None
    assert feature.read_values(mine) == (feature.named_value(tmp_path, "mine"),)
    assert theirs.unavailable is None
    assert feature.read_values(theirs) == (feature.named_value(tmp_path, "theirs"),)
    assert absent.unavailable is None
    assert feature.read_values(absent) == ()
    # One force-loaded source: the same user-layer revision for every project.
    assert mine.user_revision == theirs.user_revision == absent.user_revision
    assert mine.user_revision == layer.fingerprint


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_read_without_actual_user_layer_is_unavailable(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    project_dir = tmp_path / "project"
    config_file = project_dir / ".chartreux" / "config.toml"
    config_file.parent.mkdir(parents=True)
    trusted_folders_manager.trust_for_session(config_file.parent)
    project = ProjectConfigLayer(path=project_dir)
    session = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), project, session],
        default_layer_resolver=lambda: session,
    )

    read = await feature.read(orchestrator, project_dir)

    assert read == feature.make_unavailable("no_user_source")
    assert not config_file.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_read_needs_no_lock_and_publishes_nothing(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    orchestrator = await make_orchestrator(path)
    before = live_state(orchestrator)

    async with orchestrator._mutation_lock:
        read = await feature.read(orchestrator, tmp_path)

    assert read.unavailable is None
    assert feature.read_values(read) == ()
    assert not orchestrator._mutation_lock.locked()
    assert live_state(orchestrator) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_read_relative_project_is_rejected(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    orchestrator = await make_orchestrator(path)

    with pytest.raises(ValueError, match=feature.field):
        await feature.read(orchestrator, Path("relative-project"))


# ---------------------------------------------------------------------------
# Authority pairs (migrated in 12a-iii from test_root_authority.py and the
# authority tests of test_dangerous_command_authority.py; each case asserts
# the union of both twins' assertions — see the stage-12 report).
# ---------------------------------------------------------------------------

_INVALID_DEFINITION_CASES = [
    *[(ROOTS, value) for value in ROOTS.invalid_definitions],
    *[(GRANTS, value) for value in GRANTS.invalid_definitions],
]
_INVALID_DEFINITION_IDS = [
    f"{feature.name}-{index}"
    for feature in FEATURES
    for index, _ in enumerate(feature.invalid_definitions)
]


@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
def test_unverified_contribution_fails_closed(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=feature.provenance_match):
        SourceRestrictions.from_raw(
            {feature.field: {str(tmp_path): [feature.authority_value(tmp_path)]}},
            layer_name="user-toml",
            locator="user-toml",
            kind="source",
            store_fingerprint=None,
        )


@pytest.mark.parametrize(
    "name", ["user-toml", "user", "default", "environment", "agent-profile"]
)
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
def test_names_never_authorize_authority(
    feature: GrantConfigFeature, tmp_path: Path, name: str, empty: bool
) -> None:
    data = {
        feature.field: (
            {} if empty else {str(tmp_path): [feature.authority_value(tmp_path)]}
        )
    }
    with pytest.raises(ValueError, match="actual user source"):
        feature.validate_source(data, layer=OverridesLayer(name=name, data=data))


@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
def test_only_empty_actual_default_is_tolerated(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    layer = DefaultConfigLayer(schema=ChartreuxConfigSchema, name="renamed")
    assert feature.validate_source({feature.field: {}}, layer=layer) == ()
    with pytest.raises(ValueError, match="actual user source"):
        feature.validate_source({feature.field: {str(tmp_path): []}}, layer=layer)


@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
def test_subclass_cannot_claim_user_authority(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    class FakeUserLayer(UserConfigLayer):
        pass

    with pytest.raises(ValueError, match="actual user source"):
        feature.validate_source(
            {feature.field: {}}, layer=FakeUserLayer(path=tmp_path / "user.toml")
        )


@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
def test_projection_canonical_immutable_and_separate_from_merge(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    value = sample_str(feature, tmp_path)
    data = {feature.field: {str(project): [value, value]}}
    layer = UserConfigLayer(path=tmp_path / "user.toml", name="renamed")
    source = SourceRestrictions.from_raw(
        data,
        layer_name=layer.name,
        locator=layer.source_locator,
        kind="source",
        store_fingerprint="accepted-source",
        root_layer=layer,
    )
    assert feature.restrictions_reader(source) == feature.validate_source(
        data, layer=layer
    )
    item = feature.restrictions_reader(source)[0]
    assert item.project == project
    assert feature.item_values(item) == (feature.sample_value(tmp_path),)
    assert feature.projection_map(source) == {str(project): [value]}
    frozen_field = feature.item_values_attr
    with pytest.raises(FrozenInstanceError):
        setattr(item, frozen_field, ())
    data[feature.field][str(project)].clear()
    config = ChartreuxConfigSchema.model_validate({
        feature.field: {str(project): [value]}
    })
    # Mutating the effective (mergeable) schema dict creates no accepted authority.
    feature.config_projection(config).clear()
    feature.config_projection(config)[str(tmp_path / "other")] = [
        feature.named_value(tmp_path, "other-root")
    ]
    assert feature.item_values(item) == (feature.sample_value(tmp_path),)
    if feature.workspace_check:
        assert not Workspace.for_session(project).allows(Path(value))
    assert not source.preserves(
        SourceRestrictions.from_raw(
            {},
            layer_name=layer.name,
            locator=layer.source_locator,
            kind="source",
            store_fingerprint=None,
        )
    )


@pytest.mark.parametrize(
    ("feature", "value"), _INVALID_DEFINITION_CASES, ids=_INVALID_DEFINITION_IDS
)
def test_invalid_definitions_are_sanitized(
    feature: GrantConfigFeature, tmp_path: Path, value: Any
) -> None:
    with pytest.raises(ValueError) as caught:
        feature.validate_source(
            {feature.field: value}, layer=UserConfigLayer(path=tmp_path / "user.toml")
        )
    assert str(caught.value) == feature.invalid_definition_message
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
def test_colliding_canonical_project_keys_rejected(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    alias = tmp_path / "alias"
    project = tmp_path / "project"
    project.mkdir()
    alias.symlink_to(project, target_is_directory=True)
    with pytest.raises(ValueError, match=feature.collision_match):
        feature.validate_source(
            {feature.field: {str(alias): [], str(project): []}},
            layer=UserConfigLayer(path=tmp_path / "user.toml"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_builder_accepts_real_user_authority(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    path = tmp_path / "user.toml"
    path.write_text(
        f'[{feature.field}]\n"{tmp_path}" = ["{feature.authority_value(tmp_path)}"]\n'
    )
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([
        DefaultConfigLayer(schema=ChartreuxConfigSchema),
        UserConfigLayer(path=path),
    ])
    candidate = await builder.build_candidate()
    accepted = feature.restrictions_reader(candidate.restrictions[-1])
    assert accepted[0].project == tmp_path
    assert feature.item_values(accepted[0]) == feature.expected_authority_item_values(
        tmp_path
    )
    assert feature.config_projection(await builder.build()) == {
        str(tmp_path): [feature.authority_value(tmp_path)]
    }


# ---------------------------------------------------------------------------
# Source-integration pairs (migrated in 12a-iv from
# test_root_source_integration.py and the source-integration tests of
# test_dangerous_command_authority.py; each case asserts the union of both
# twins' assertions — see the stage-12 report for the per-pair audits).
# ---------------------------------------------------------------------------

_PATCH_OPERATION_CASES = [
    *[(ROOTS, operation) for operation in ROOTS.patch_operations],
    *[(GRANTS, operation) for operation in GRANTS.patch_operations],
]
_PATCH_OPERATION_IDS = [
    f"{feature.name}-{operation}" for feature, operation in _PATCH_OPERATION_CASES
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["build", "build_candidate"])
@pytest.mark.parametrize("value", [{}, None, [], {"/private-project": ["secret"]}])
@pytest.mark.parametrize("prospective", [False, True])
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_forbidden_sources_fail_before_shadow_and_cache(
    feature: GrantConfigFeature,
    method: Literal["build", "build_candidate"],
    value: Any,
    prospective: bool,
) -> None:
    spoof = OverridesLayer(
        name="user-toml", data={} if prospective else {feature.field: value}
    )
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([spoof, DefaultConfigLayer(schema=ChartreuxConfigSchema)])
    overrides = {
        spoof.name: RawConfig.model_validate(
            {feature.field: value} if prospective else {}
        )
    }
    with pytest.raises(ValueError, match="actual user source") as caught:
        await getattr(builder, method)(layer_overrides=overrides)
    assert "secret" not in str(caught.value)
    assert "/private-project" not in str(caught.value)
    assert caught.value.__context__ is None
    assert all(layer.cached_data is None for layer in builder.layers)
    assert all(layer.fingerprint is None for layer in builder.layers)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["build", "build_candidate"])
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_malformed_real_user_cannot_be_hidden_by_override(
    feature: GrantConfigFeature,
    tmp_path: Path,
    method: Literal["build", "build_candidate"],
) -> None:
    path = tmp_path / "private-source.toml"
    path.write_text(
        f'[{feature.field}]\n"/private-project" = ["{feature.secret_marker}"]\n'
    )
    user = UserConfigLayer(path=path)
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([DefaultConfigLayer(schema=ChartreuxConfigSchema), user])
    with pytest.raises(ValueError, match=feature.field) as caught:
        await getattr(builder, method)(layer_overrides={user.name: RawConfig()})
    assert caught.value.__context__ is None
    assert "private" not in str(caught.value)
    assert feature.secret_marker not in str(caught.value)
    assert user.cached_data is None
    assert builder.layers[0].cached_data is None


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["", "__/private-project"])
@pytest.mark.parametrize("value", ["", "{}", "null", "secret-malformed-json"])
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_environment_field_rejected_before_decode(
    feature: GrantConfigFeature,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    value: str,
) -> None:
    monkeypatch.setenv(f"CHARTREUX_{feature.field.upper()}{suffix}", value)
    layer = EnvironmentLayer(schema=ChartreuxConfigSchema)
    with patch("chartreux.core.config.layers.environment.EnvSettingsSource") as decoder:
        with pytest.raises(LayerImplementationError) as caught:
            await layer.load()
    decoder.assert_not_called()
    cause = caught.value.__cause__
    assert cause is not None
    assert feature.field in str(cause)
    assert "secret" not in str(cause)
    assert "private" not in str(cause)
    assert cause.__context__ is None
    assert layer.cached_data is None
    assert layer.fingerprint is None


@pytest.mark.asyncio
@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    ("feature", "operation"), _PATCH_OPERATION_CASES, ids=_PATCH_OPERATION_IDS
)
async def test_ordinary_user_patch_cannot_change_authority(
    feature: GrantConfigFeature,
    operation: str,
    tmp_path: Path,
    save: bool,
    existing: bool,
) -> None:
    orchestrator, path = await user_orchestrator(feature, tmp_path, present=existing)
    layer = orchestrator.get_layer("renamed-user")
    injected = {str(tmp_path): [feature.patch_injected_value(tmp_path)]}
    ops: list[PatchOp]
    match operation:
        case "set":
            ops = [AddOperationPatch(path=f"/{feature.field}", value=injected)]
        case "replace":
            ops = [ReplaceOperationPatch(path=f"/{feature.field}", value=injected)]
        case "remove":
            ops = [RemoveOperationPatch(path=f"/{feature.field}")]
        case "document":
            ops = [AddOperationPatch(path="", value={feature.field: injected})]
        case "erase-document":
            if not existing:
                pytest.skip("No authority exists to remove")
            ops = [AddOperationPatch(path="", value={})]
        case _:
            ops = [
                AddOperationPatch(path=f"/{feature.field}", value=injected),
                RemoveOperationPatch(path=f"/{feature.field}"),
            ]
    ops = [op.model_copy(update={"target_layer_name": layer.name}) for op in ops]
    before = path.read_bytes()
    config, policy, token, cache = (
        orchestrator.config,
        orchestrator.restrictions,
        orchestrator.accepted_token,
        layer.cached_data,
    )
    if save:
        assert layer.fingerprint is not None
        result = await orchestrator.save(
            ops, target="user", expected_revision=layer.fingerprint, reason="ordinary"
        )
        assert result.persistence == "not_saved"
        assert result.error == "validation"
    else:
        with pytest.raises(ConfigPatchValidationError, match=feature.field) as caught:
            await orchestrator.apply_patch(ops, reason="ordinary")
        assert feature.patch_injected_marker(tmp_path) not in str(caught.value)
        assert caught.value.__context__ is None
    assert path.read_bytes() == before
    assert orchestrator.config is config
    assert orchestrator.restrictions is policy
    assert orchestrator.accepted_token is token
    assert layer.cached_data is cache


@pytest.mark.asyncio
@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_bad_reload_or_preview_preserves_accepted_snapshot(
    feature: GrantConfigFeature, tmp_path: Path, preview: bool
) -> None:
    orchestrator, path = await user_orchestrator(feature, tmp_path)
    layer = orchestrator.get_layer("renamed-user")
    config, policy, token, cache = (
        orchestrator.config,
        orchestrator.restrictions,
        orchestrator.accepted_token,
        layer.cached_data,
    )
    with pytest.raises(ValueError, match=feature.field) as caught:
        if preview:
            await orchestrator.preview_candidate(
                layer_overrides={
                    layer.name: RawConfig.model_validate({
                        feature.field: {"/private": [feature.secret_marker]}
                    })
                }
            )
        else:
            path.write_text(
                f'[{feature.field}]\n"/private" = ["{feature.secret_marker}"]\n'
            )
            await orchestrator.reload()
    assert caught.value.__context__ is None
    assert "private" not in str(caught.value)
    assert feature.secret_marker not in str(caught.value)
    assert orchestrator.config is config
    assert orchestrator.restrictions is policy
    assert orchestrator.accepted_token is token
    assert layer.cached_data is cache


@pytest.mark.parametrize("value", [{}, None, []])
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
def test_empty_unattributed_assertions_fail_closed(
    feature: GrantConfigFeature, value: Any
) -> None:
    with pytest.raises(ValueError, match="provenance"):
        SourceRestrictions.from_raw(
            {feature.field: value},
            layer_name="user-toml",
            locator="user-toml",
            kind="source",
            store_fingerprint=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", FEATURES, ids=FEATURE_IDS)
async def test_real_user_authority_is_accepted_and_immutable(
    feature: GrantConfigFeature, tmp_path: Path
) -> None:
    orchestrator, _ = await user_orchestrator(feature, tmp_path)
    accepted = feature.restrictions_reader(orchestrator.restrictions[-1])
    assert accepted[0].project == tmp_path
    assert feature.item_values(accepted[0]) == (feature.sample_value(tmp_path),)
    # Mutating the effective schema dict creates no accepted authority.
    projection = feature.config_projection(orchestrator.config)
    projection.clear()
    projection[str(tmp_path / "other")] = [feature.named_value(tmp_path, "other-root")]
    assert feature.restrictions_reader(orchestrator.restrictions[-1]) == accepted
    assert await orchestrator.set_field("/theme", "dark", reason="ordinary") == []
    assert feature.restrictions_reader(orchestrator.restrictions[-1]) == accepted
    staged = await orchestrator._stage_policy_replacement(
        source="renamed-user", tools={}, expected_token=orchestrator.accepted_token
    )
    assert feature.restrictions_reader(staged.restrictions[-1]) == accepted
    child = orchestrator._copy_for_child()
    refreshed = await child._stage_policy_refresh(
        previous=orchestrator.restrictions[-1],
        replacement=staged.restrictions[-1],
        expected_token=child.accepted_token,
    )
    assert feature.restrictions_reader(refreshed.restrictions[-1]) == accepted
