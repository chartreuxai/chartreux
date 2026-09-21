from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.config.fingerprint import (
    capture_stable_file,
    create_dict_fingerprint,
    create_file_fingerprint,
)
from chartreux.core.config.types import ConcurrencyConflictError


class TestCaptureStableFile:
    def test_captures_unchanged_file(self, tmp_working_directory: Path) -> None:
        path = tmp_working_directory / "config.toml"
        path.write_text("key = 1")

        with capture_stable_file(path) as (file, first_fingerprint):
            assert file.read() == b"key = 1"

        with capture_stable_file(path) as (file, second_fingerprint):
            assert file.read() == b"key = 1"

        assert isinstance(first_fingerprint, str)
        assert first_fingerprint
        assert first_fingerprint == second_fingerprint

    def test_rejects_in_place_mutation_during_read(
        self, tmp_working_directory: Path
    ) -> None:
        path = tmp_working_directory / "config.toml"
        path.write_text("key = 1")

        with pytest.raises(ConcurrencyConflictError) as exc_info:
            with capture_stable_file(path) as (file, _):
                assert file.read() == b"key = 1"
                path.write_text("key = 123")

        assert exc_info.value.actual_fp != exc_info.value.expected_fp

    def test_atomic_replace_during_read_preserves_opened_snapshot(
        self, tmp_working_directory: Path
    ) -> None:
        path = tmp_working_directory / "config.toml"
        replacement = tmp_working_directory / "replacement.toml"
        path.write_text("key = 1")
        replacement.write_text("key = 2")

        with capture_stable_file(path) as (file, fingerprint):
            replacement.replace(path)
            assert file.read() == b"key = 1"

        with path.open("rb") as replacement_file:
            assert fingerprint != create_file_fingerprint(replacement_file)

    def test_unlink_after_open_preserves_snapshot(
        self, tmp_working_directory: Path
    ) -> None:
        path = tmp_working_directory / "config.toml"
        path.write_text("key = 1")

        with capture_stable_file(path) as (file, fingerprint):
            path.unlink()
            assert file.read() == b"key = 1"

        assert fingerprint
        assert not path.exists()

    def test_raises_when_file_is_missing(self, tmp_working_directory: Path) -> None:
        path = tmp_working_directory / "missing.toml"

        with pytest.raises(FileNotFoundError):
            with capture_stable_file(path):
                pass


class TestCreateFileFingerprint:
    def test_captures_file_state(self, tmp_working_directory: Path) -> None:
        path = tmp_working_directory / "config.toml"
        path.write_text("key = 1")

        with path.open("rb") as file:
            first_fingerprint = create_file_fingerprint(file)
        with path.open("rb") as file:
            second_fingerprint = create_file_fingerprint(file)

        assert isinstance(first_fingerprint, str)
        assert first_fingerprint
        assert first_fingerprint == second_fingerprint

    def test_changes_when_file_changes(self, tmp_working_directory: Path) -> None:
        path = tmp_working_directory / "config.toml"
        path.write_text("key = 1")
        with path.open("rb") as file:
            first_fingerprint = create_file_fingerprint(file)

        path.write_text("key = 2")

        with path.open("rb") as file:
            assert create_file_fingerprint(file) != first_fingerprint


class TestCreateDictFingerprint:
    def test_empty_dict_returns_stable_non_empty_token(self) -> None:
        first_fingerprint = create_dict_fingerprint({})
        second_fingerprint = create_dict_fingerprint({})

        assert isinstance(first_fingerprint, str)
        assert first_fingerprint
        assert first_fingerprint == second_fingerprint

    def test_stable_for_same_dict(self) -> None:
        data = {
            "CHARTREUX_MODEL": "mistral-large",
            "CHARTREUX_THEME": "dark",
            "CHARTREUX_TOOLS": ["read", "write"],
        }
        fp1 = create_dict_fingerprint(data)
        fp2 = create_dict_fingerprint(data)
        assert fp1 == fp2

    def test_order_independent(self) -> None:
        fp1 = create_dict_fingerprint({"a": "1", "b": "2"})
        fp2 = create_dict_fingerprint({"b": "2", "a": "1"})
        assert fp1 == fp2

    def test_serializes_path_values(self) -> None:
        fp1 = create_dict_fingerprint({
            "tool_paths": [Path("/tmp/custom-tools")],
            "agent_paths": [Path("agents")],
        })
        fp2 = create_dict_fingerprint({
            "tool_paths": ["/tmp/custom-tools"],
            "agent_paths": ["agents"],
        })
        assert fp1 == fp2

    def test_changes_when_list_order_changes(self) -> None:
        fp1 = create_dict_fingerprint({"tools": ["read", "write"]})
        fp2 = create_dict_fingerprint({"tools": ["write", "read"]})
        assert fp1 != fp2

    def test_changes_when_value_changes(self) -> None:
        fp1 = create_dict_fingerprint({"CHARTREUX_MODEL": "mistral-large"})
        fp2 = create_dict_fingerprint({"CHARTREUX_MODEL": "devstral-2"})
        assert fp1 != fp2

    def test_changes_when_key_added(self) -> None:
        fp1 = create_dict_fingerprint({"a": "1"})
        fp2 = create_dict_fingerprint({"a": "1", "b": "2"})
        assert fp1 != fp2
