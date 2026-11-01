"""S4a: the remote-exec destination-grant field — schema and persistence.

The new protected user-layer field maps project keys to exact canonical
``user@host:port`` destinations. These tests pin canonicalization, the
user-source-only provenance rule, the revision-checked save/read/remove
round trip, and the catalog exclusion.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._dangerous_command_authority import (
    REMOTE_EXEC_GRANTS_FIELD,
    ProjectRemoteExecGrant,
    validate_remote_exec_grant_source,
)
from chartreux.core.config._dangerous_command_persistence import (
    read_saved_dangerous_command_grants,
)
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import normalize_remote_exec_grants
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.settings_catalog import EXCLUDED_SETTINGS
from chartreux.core.dangerous_commands import RemoteDestination


async def make_orchestrator(path: Path) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    user = UserConfigLayer(path=path)
    session = OverridesLayer(data={})
    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, session],
        default_layer_resolver=lambda: session,
    )


class TestNormalize:
    """The schema normalizer canonicalizes and rejects malformed entries."""

    def test_canonicalizes_and_dedupes(self, tmp_path: Path):
        repo = str(tmp_path / "repo")
        normalized = normalize_remote_exec_grants({
            repo: [
                "Deploy@Example.COM",
                "Deploy@example.com:22",
                "ops@example.com:2222",
            ]
        })
        assert normalized == {
            str((tmp_path / "repo").resolve()): [
                "Deploy@example.com:22",
                "ops@example.com:2222",
            ]
        }

    @pytest.mark.parametrize(
        "destination",
        [
            "example.com:22",  # no user
            "deploy@*.example.com",  # wildcard
            "deploy@a.example.com,b.example.com",  # alias group
            "deploy@example.com:0",  # out of range
            "deploy@example.com:ssh",  # not a port
        ],
    )
    def test_malformed_destination_rejects_the_field(
        self, tmp_path: Path, destination: str
    ):
        with pytest.raises(ValueError, match="remote_exec_grants_by_project"):
            normalize_remote_exec_grants({str(tmp_path / "repo"): [destination]})

    def test_relative_project_key_is_rejected(self):
        with pytest.raises(ValueError, match="remote_exec_grants_by_project"):
            normalize_remote_exec_grants({"relative/repo": ["deploy@example.com"]})


class TestSourceValidation:
    """Only an actual user source may carry the field."""

    def test_user_source_is_accepted(self, tmp_path: Path):
        repo = tmp_path / "repo"
        grants = validate_remote_exec_grant_source(
            {REMOTE_EXEC_GRANTS_FIELD: {str(repo): ["deploy@example.com"]}},
            layer=UserConfigLayer(path=tmp_path / "config.toml"),
        )
        assert grants == (
            ProjectRemoteExecGrant(
                Path(str(repo.resolve())),
                (RemoteDestination("deploy", "example.com", 22),),
            ),
        )

    def test_default_layer_tolerates_the_empty_default(self, tmp_path: Path):
        assert (
            validate_remote_exec_grant_source(
                {REMOTE_EXEC_GRANTS_FIELD: {}},
                layer=DefaultConfigLayer(schema=ChartreuxConfigSchema),
            )
            == ()
        )

    def test_non_user_source_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="actual user source"):
            validate_remote_exec_grant_source(
                {REMOTE_EXEC_GRANTS_FIELD: {str(tmp_path / "r"): ["a@example.com"]}},
                layer=DefaultConfigLayer(schema=ChartreuxConfigSchema),
            )

    def test_absent_field_contributes_nothing(self, tmp_path: Path):
        assert (
            validate_remote_exec_grant_source(
                {}, layer=UserConfigLayer(path=tmp_path / "config.toml")
            )
            == ()
        )


class TestPersistence:
    """Save, read, and remove round-trip through the user layer."""

    @pytest.mark.asyncio
    async def test_save_read_and_remove_round_trip(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_text('theme = "dark"\n')
        repo = tmp_path / "repo"
        orchestrator = await make_orchestrator(path)
        layer = orchestrator.get_layer("user-toml")
        expected = layer.fingerprint
        assert expected is not None

        result = await orchestrator.save_remote_exec_grant(
            project=repo, destination="Deploy@EXAMPLE.com", expected_revision=expected
        )
        assert result.persistence == "saved"
        assert result.error is None
        document = tomllib.loads(path.read_text())
        assert document["theme"] == "dark"
        assert document[REMOTE_EXEC_GRANTS_FIELD] == {
            str(repo.resolve()): ["Deploy@example.com:22"]
        }

        saved = await read_saved_dangerous_command_grants(orchestrator, project=repo)
        assert saved.unavailable is None
        assert saved.destinations == ("Deploy@example.com:22",)
        assert saved.user_revision is not None

        removed = await orchestrator.remove_remote_exec_grant(
            project=repo,
            destination="Deploy@example.com:22",
            expected_revision=saved.user_revision,
        )
        assert removed.persistence == "saved"
        document = tomllib.loads(path.read_text())
        assert REMOTE_EXEC_GRANTS_FIELD not in document
        assert document["theme"] == "dark"

    @pytest.mark.asyncio
    async def test_wildcard_destination_is_rejected_without_writing(
        self, tmp_path: Path
    ):
        path = tmp_path / "config.toml"
        path.write_text("")
        orchestrator = await make_orchestrator(path)
        layer = orchestrator.get_layer("user-toml")
        expected = layer.fingerprint
        assert expected is not None
        result = await orchestrator.save_remote_exec_grant(
            project=tmp_path / "repo",
            destination="deploy@*.example.com",
            expected_revision=expected,
        )
        assert result.persistence == "not_saved"
        assert result.error == "validation"
        assert path.read_text() == ""

    @pytest.mark.asyncio
    async def test_stale_revision_conflicts(self, tmp_path: Path):
        path = tmp_path / "config.toml"
        path.write_text("")
        orchestrator = await make_orchestrator(path)
        result = await orchestrator.save_remote_exec_grant(
            project=tmp_path / "repo",
            destination="deploy@example.com",
            expected_revision="stale",
        )
        assert result.persistence == "not_saved"
        assert result.error == "conflict"
        assert path.read_text() == ""


class TestCatalog:
    """The protected field is never an editable setting."""

    def test_field_is_excluded_from_the_settings_catalog(self):
        assert REMOTE_EXEC_GRANTS_FIELD in EXCLUDED_SETTINGS
