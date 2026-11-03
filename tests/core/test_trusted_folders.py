from __future__ import annotations

import fcntl
from pathlib import Path
import subprocess
import sys
import threading
import tomllib
from typing import Any
from unittest.mock import patch

import pytest
import tomli_w

from chartreux.core.config.layers import _base, _checked_store
from chartreux.core.config.layers._checked_store import CheckedWriteOutcome
from chartreux.core.paths import AGENTS_MD_FILENAME, TRUSTED_FOLDERS_FILE
from chartreux.core.trusted_folders import (
    TrustedFoldersManager,
    TrustStoreState,
    filesystem_identity,
    find_git_repo_ancestor,
    find_repo_trustable_files_for_cwd,
    find_trustable_files,
    find_untrusted_config_dirs,
    has_agents_md_file,
)

ORIGINAL_STORE = b'trusted = ["/a"]\nuntrusted = []\n'
EXTERNAL_STORE = b'trusted = ["/external"]\nuntrusted = []\n'
MALFORMED_STORE = b"trusted = [ not valid toml \n"
INVALID_STORE = b'trusted = "not-a-list"\n'

_CHILD_TRUST_WRITE = """
import sys
import tomllib
from pathlib import Path
import tomli_w
from chartreux.core.config.layers._checked_store import (
    checked_write_store,
    store_revision,
)
path = Path(sys.argv[1])
def edit(current):
    data = dict(current)
    trusted = list(data.get("trusted", []))
    if "/child" not in trusted:
        trusted.append("/child")
    data["trusted"] = trusted
    return tomli_w.dumps(data).encode()
def parse(data):
    return tomllib.loads(data.decode())
result = checked_write_store(
    path, edit, parse=parse, expected_revision=store_revision(path)
)
print(result.outcome.value, flush=True)
"""


def _store_tmp_files(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*.tmp"))


class TestTrustedFoldersManager:
    def test_initializes_with_empty_lists_when_file_does_not_exist(
        self, tmp_path: Path
    ) -> None:
        trusted_file = TRUSTED_FOLDERS_FILE.path
        assert not trusted_file.is_file()

        manager = TrustedFoldersManager()
        assert manager.is_trusted(tmp_path) is None
        assert manager.load_error == "trust store missing"
        assert not trusted_file.exists()

    def test_loads_existing_file(self, tmp_path: Path) -> None:
        trusted_file = TRUSTED_FOLDERS_FILE.path

        data = {"trusted": [str(tmp_path.resolve())], "untrusted": []}
        with trusted_file.open("wb") as f:
            tomli_w.dump(data, f)

        manager = TrustedFoldersManager()

        assert manager.is_trusted(tmp_path) is True

    def test_handles_corrupted_file(self, tmp_path: Path) -> None:
        trusted_file = TRUSTED_FOLDERS_FILE.path
        trusted_file.write_text("invalid toml content {[", encoding="utf-8")

        manager = TrustedFoldersManager()

        assert manager.is_trusted(tmp_path) is None
        assert manager.load_error == "trust store invalid or unreadable"
        assert trusted_file.read_text(encoding="utf-8") == "invalid toml content {["

    @pytest.mark.parametrize("trusted", [False, True])
    def test_successful_save_clears_load_error(
        self, tmp_path: Path, trusted: bool
    ) -> None:
        # Plan O4a2 amended pin: a MISSING store is created by the checked
        # write; the confirmed save clears the load error.
        manager = TrustedFoldersManager()
        assert manager.load_error is not None
        if trusted:
            result = manager.add_trusted(tmp_path)
        else:
            result = manager.add_untrusted(tmp_path)
        assert result.outcome == CheckedWriteOutcome.SAVED
        assert manager.load_error is None
        reloaded = TrustedFoldersManager()
        assert reloaded.load_error is None
        assert reloaded.is_trusted(tmp_path) is trusted

    @pytest.mark.parametrize(
        "contents,expected_state",
        [
            ("[malformed", TrustStoreState.MALFORMED),
            ('trusted = "invalid"', TrustStoreState.INVALID),
        ],
    )
    @pytest.mark.parametrize("trusted", [False, True])
    def test_malformed_or_invalid_save_is_refused_and_preserves_load_error(
        self,
        tmp_path: Path,
        contents: str,
        expected_state: TrustStoreState,
        trusted: bool,
    ) -> None:
        # Plan O4a2 amended pin: a malformed/invalid store is never rewritten.
        # The refused write is reported as MALFORMED, leaves the on-disk store
        # bytes byte-identical, keeps the load error, and grants nothing.
        trusted_file = TRUSTED_FOLDERS_FILE.path
        raw = contents.encode("utf-8")
        trusted_file.write_bytes(raw)
        manager = TrustedFoldersManager()
        assert manager.load_error is not None
        assert manager.store_state is expected_state
        if trusted:
            result = manager.add_trusted(tmp_path)
        else:
            result = manager.add_untrusted(tmp_path)
        assert result.outcome == CheckedWriteOutcome.MALFORMED
        assert trusted_file.read_bytes() == raw
        assert manager.load_error is not None
        assert manager.store_state is expected_state
        assert manager.is_trusted(tmp_path) is not True
        reloaded = TrustedFoldersManager()
        assert reloaded.load_error is not None
        assert reloaded.is_trusted(tmp_path) is not True

    def test_failed_save_preserves_load_error(self, tmp_path: Path) -> None:
        # Plan O4a2 amended pin: a save that cannot run is a typed NOT_SAVED
        # refusal, not a silent in-memory grant; the load error is preserved.
        manager = TrustedFoldersManager()
        error = manager.load_error
        with patch.object(Path, "open", side_effect=OSError("cannot write")):
            result = manager.add_trusted(tmp_path)
        assert result.outcome == CheckedWriteOutcome.NOT_SAVED
        assert manager.load_error == error
        assert manager.is_trusted(tmp_path) is not True

    def test_normalizes_paths_to_absolute(
        self, tmp_working_directory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = TrustedFoldersManager()

        manager.add_trusted(Path("."))
        assert manager.is_trusted(tmp_working_directory) is True
        assert manager.is_trusted(Path(".")) is True

    def test_expands_user_home_in_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        manager = TrustedFoldersManager()

        manager.add_trusted(Path("~/test"))
        assert manager.is_trusted(tmp_path / "test") is True

    def test_is_trusted_returns_true_for_trusted_path(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)

        assert manager.is_trusted(tmp_path) is True

    def test_is_trusted_returns_false_for_untrusted_path(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        manager.add_untrusted(tmp_path)

        assert manager.is_trusted(tmp_path) is False

    def test_is_trusted_returns_none_for_unknown_path(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()

        assert manager.is_trusted(tmp_path) is None

    def test_add_trusted_adds_path_to_trusted_list(self, tmp_path: Path) -> None:
        trusted_file = TRUSTED_FOLDERS_FILE.path
        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)

        assert manager.is_trusted(tmp_path) is True
        with trusted_file.open("rb") as f:
            data = tomllib.load(f)
        assert str(tmp_path.resolve()) in data["trusted"]

    def test_add_trusted_removes_path_from_untrusted(self, tmp_path: Path) -> None:
        trusted_file = TRUSTED_FOLDERS_FILE.path
        manager = TrustedFoldersManager()

        manager.add_untrusted(tmp_path)
        assert manager.is_trusted(tmp_path) is False

        manager.add_trusted(tmp_path)
        assert manager.is_trusted(tmp_path) is True

        with trusted_file.open("rb") as f:
            data = tomllib.load(f)
        assert str(tmp_path.resolve()) not in data["untrusted"]
        assert str(tmp_path.resolve()) in data["trusted"]

    def test_add_trusted_idempotent(self, tmp_path: Path) -> None:
        trusted_file = TRUSTED_FOLDERS_FILE.path

        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)
        manager.add_trusted(tmp_path)
        manager.add_trusted(tmp_path)

        assert manager.is_trusted(tmp_path) is True
        with trusted_file.open("rb") as f:
            data = tomllib.load(f)
        assert data["trusted"].count(str(tmp_path.resolve())) == 1

    def test_add_trusted_rejects_replaced_directory_identity(
        self, tmp_path: Path
    ) -> None:
        original = tmp_path / "workspace"
        replacement = tmp_path / "replacement"
        original.mkdir()
        replacement.mkdir()
        identity = filesystem_identity(original)
        original.rmdir()
        original.symlink_to(replacement, target_is_directory=True)
        manager = TrustedFoldersManager()

        with pytest.raises(ValueError, match="Workspace changed"):
            manager.add_trusted(original, expected_identity=identity)

        assert manager.is_trusted(replacement) is None

    def test_session_trust_rejects_swap_during_normalization(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "workspace"
        replacement = tmp_path / "replacement"
        workspace.mkdir()
        replacement.mkdir()
        manager = TrustedFoldersManager()
        validated_identity = filesystem_identity(workspace)
        monkeypatch.setattr(manager, "_normalize_path", lambda _path: str(replacement))

        with pytest.raises(ValueError, match="Workspace changed"):
            manager.trust_for_session(workspace, expected_identity=validated_identity)

        assert manager.is_trusted(replacement) is None

    def test_revoke_session_trust_is_noop_for_unmatched_redirected_grant(
        self, tmp_path: Path
    ) -> None:
        workspace = tmp_path / "workspace"
        replacement = tmp_path / "replacement"
        workspace.mkdir()
        replacement.mkdir()
        manager = TrustedFoldersManager()
        manager.trust_for_session(workspace)
        manager.trust_for_session(replacement)
        workspace.rmdir()
        workspace.symlink_to(replacement, target_is_directory=True)

        manager.revoke_session_trust(workspace)

        assert manager._session_trusted == [str(replacement)]
        assert manager._session_trust_grants == {str(replacement): [str(replacement)]}

        manager.revoke_session_trust(workspace)

        assert manager._session_trusted == [str(replacement)]
        assert manager._session_trust_grants == {str(replacement): [str(replacement)]}

    def test_decline_rejects_swap_during_normalization(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "workspace"
        replacement = tmp_path / "replacement"
        workspace.mkdir()
        replacement.mkdir()
        manager = TrustedFoldersManager()
        validated_identity = filesystem_identity(workspace)
        monkeypatch.setattr(manager, "_normalize_path", lambda _path: str(replacement))

        with pytest.raises(ValueError, match="Workspace changed"):
            manager.add_untrusted(workspace, expected_identity=validated_identity)

        assert manager.is_explicitly_untrusted(replacement) is False

    def test_add_untrusted_adds_path_to_untrusted_list(self, tmp_path: Path) -> None:
        trusted_file = TRUSTED_FOLDERS_FILE.path
        manager = TrustedFoldersManager()
        manager.add_untrusted(tmp_path)

        assert manager.is_trusted(tmp_path) is False
        with trusted_file.open("rb") as f:
            data = tomllib.load(f)
        assert str(tmp_path.resolve()) in data["untrusted"]

    def test_add_untrusted_removes_path_from_trusted(self, tmp_path: Path) -> None:
        trusted_file = TRUSTED_FOLDERS_FILE.path
        manager = TrustedFoldersManager()

        manager.add_trusted(tmp_path)
        assert manager.is_trusted(tmp_path) is True

        manager.add_untrusted(tmp_path)
        assert manager.is_trusted(tmp_path) is False

        with trusted_file.open("rb") as f:
            data = tomllib.load(f)
        assert str(tmp_path.resolve()) not in data["trusted"]
        assert str(tmp_path.resolve()) in data["untrusted"]

    def test_add_untrusted_idempotent(self, tmp_path: Path) -> None:
        trusted_file = TRUSTED_FOLDERS_FILE.path

        manager = TrustedFoldersManager()
        manager.add_untrusted(tmp_path)
        manager.add_untrusted(tmp_path)
        manager.add_untrusted(tmp_path)

        assert manager.is_trusted(tmp_path) is False
        with trusted_file.open("rb") as f:
            data = tomllib.load(f)
        assert data["untrusted"].count(str(tmp_path.resolve())) == 1

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        manager1 = TrustedFoldersManager()
        manager1.add_trusted(tmp_path)

        manager2 = TrustedFoldersManager()
        assert manager2.is_trusted(tmp_path) is True

    def test_handles_multiple_paths(self, tmp_path: Path) -> None:
        trusted1 = tmp_path / "trusted1"
        trusted2 = tmp_path / "trusted2"
        untrusted1 = tmp_path / "untrusted1"
        untrusted2 = tmp_path / "untrusted2"
        for p in [trusted1, trusted2, untrusted1, untrusted2]:
            p.mkdir()

        manager = TrustedFoldersManager()
        manager.add_trusted(trusted1)
        manager.add_trusted(trusted2)
        manager.add_untrusted(untrusted1)
        manager.add_untrusted(untrusted2)

        assert manager.is_trusted(trusted1) is True
        assert manager.is_trusted(trusted2) is True
        assert manager.is_trusted(untrusted1) is False
        assert manager.is_trusted(untrusted2) is False

    def test_handles_switching_between_trusted_and_untrusted(
        self, tmp_path: Path
    ) -> None:
        manager = TrustedFoldersManager()

        manager.add_trusted(tmp_path)
        assert manager.is_trusted(tmp_path) is True

        manager.add_untrusted(tmp_path)
        assert manager.is_trusted(tmp_path) is False

        manager.add_trusted(tmp_path)
        assert manager.is_trusted(tmp_path) is True

    def test_handles_missing_file_during_save(self, tmp_path: Path) -> None:
        # Plan O4a2 amended pin (design P): a save that cannot run is refused
        # as a typed NOT_SAVED outcome — the failed grant is NOT live in the
        # process and nothing is published to disk.
        trusted_file = TRUSTED_FOLDERS_FILE.path
        manager = TrustedFoldersManager()

        def mock_open(*args, **kwargs):
            raise OSError("Permission denied")

        with patch("pathlib.Path.open", side_effect=mock_open):
            result = manager.add_trusted(tmp_path)

        assert result.outcome == CheckedWriteOutcome.NOT_SAVED
        assert manager.is_trusted(tmp_path) is not True
        assert not trusted_file.exists()
        assert manager.load_error == "trust store missing"


class TestCheckedWriteWriters:
    """Plan O4a2: both add writers run on the O4a1 checked-store primitive."""

    @pytest.mark.parametrize("writer", ["add_trusted", "add_untrusted"])
    def test_missing_store_write_is_saved_and_published(
        self, tmp_path: Path, writer: str
    ) -> None:
        trusted_file = TRUSTED_FOLDERS_FILE.path
        manager = TrustedFoldersManager()
        assert manager.store_state is TrustStoreState.MISSING

        result = getattr(manager, writer)(tmp_path)

        assert result.outcome == CheckedWriteOutcome.SAVED
        assert trusted_file.is_file()
        assert manager.load_error is None
        assert manager.store_state is TrustStoreState.READY
        expected = writer == "add_trusted"
        assert manager.is_trusted(tmp_path) is expected
        assert TrustedFoldersManager().is_trusted(tmp_path) is expected

    @pytest.mark.parametrize("writer", ["add_trusted", "add_untrusted"])
    @pytest.mark.parametrize(
        "raw,state",
        [
            (MALFORMED_STORE, TrustStoreState.MALFORMED),
            (INVALID_STORE, TrustStoreState.INVALID),
        ],
    )
    def test_malformed_or_invalid_store_write_is_refused_byte_identical(
        self, tmp_path: Path, writer: str, raw: bytes, state: TrustStoreState
    ) -> None:
        # No malformed rewrite: a refused write leaves the on-disk store
        # bytes BYTE-IDENTICAL, and missing/malformed/invalid are distinct.
        trusted_file = TRUSTED_FOLDERS_FILE.path
        trusted_file.write_bytes(raw)
        manager = TrustedFoldersManager()
        assert manager.store_state is state

        result = getattr(manager, writer)(tmp_path)

        assert result.outcome == CheckedWriteOutcome.MALFORMED
        assert trusted_file.read_bytes() == raw
        assert manager.store_state is state
        assert manager.is_trusted(tmp_path) is not True
        assert _store_tmp_files(trusted_file) == []

    @pytest.mark.parametrize("writer", ["add_trusted", "add_untrusted"])
    def test_failed_pre_replace_candidate_is_never_published(
        self, tmp_path: Path, writer: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation target (plan O4a2, design P/JJJ): removing the candidate
        rollback — publishing the candidate on a refused write — fails the
        ``is_trusted(...) is not True`` assertion below.
        """
        trusted_file = TRUSTED_FOLDERS_FILE.path
        trusted_file.write_bytes(ORIGINAL_STORE)
        manager = TrustedFoldersManager()
        assert manager.store_state is TrustStoreState.READY

        def boom(fd: int) -> None:
            raise OSError("simulated disk full during temp fsync")

        monkeypatch.setattr(_checked_store, "_fsync", boom)
        result = getattr(manager, writer)(tmp_path)

        assert result.outcome == CheckedWriteOutcome.NOT_SAVED
        assert trusted_file.read_bytes() == ORIGINAL_STORE
        assert _store_tmp_files(trusted_file) == []
        # The failed pre-replace candidate is never published.
        assert manager.is_trusted(tmp_path) is not True
        assert manager.load_error is None

    @pytest.mark.parametrize("writer", ["add_trusted", "add_untrusted"])
    def test_forced_conflict_refuses_the_stale_writer(
        self, tmp_path: Path, writer: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plan O4a2 forced-conflict node: the revision changed while the writer
        waited on the held lock, so it is refused as CONFLICT (no stale
        resurrection) and effective trust is NOT granted for its payload.
        """
        trusted_file = TRUSTED_FOLDERS_FILE.path
        trusted_file.write_bytes(ORIGINAL_STORE)
        manager = TrustedFoldersManager()
        box: dict[str, Any] = {}
        pre_read = threading.Event()
        real_store_revision = _checked_store.store_revision

        def recording_store_revision(path: Path) -> str:
            revision = real_store_revision(path)
            pre_read.set()
            return revision

        monkeypatch.setattr(_checked_store, "store_revision", recording_store_revision)

        def stale_writer() -> None:
            box["result"] = getattr(manager, writer)(tmp_path)

        with _base._write_lock_path(trusted_file).open("a") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)
            writer_thread = threading.Thread(target=stale_writer)
            writer_thread.start()
            # The stale writer took its stable revision read, then waits for
            # the cooperating lock holder.
            assert pre_read.wait(timeout=5)
            assert writer_thread.is_alive()
            # An external writer moves the store while the stale writer waits.
            trusted_file.write_bytes(EXTERNAL_STORE)
            fcntl.flock(held.fileno(), fcntl.LOCK_UN)
            writer_thread.join(timeout=5)

        result = box["result"]
        assert result.outcome == CheckedWriteOutcome.CONFLICT
        # The external writer's content survives; no stale resurrection.
        assert trusted_file.read_bytes() == EXTERNAL_STORE
        # Effective trust is NOT granted for the refused payload.
        assert manager.is_trusted(tmp_path) is not True

    @pytest.mark.parametrize("writer", ["add_trusted", "add_untrusted"])
    def test_post_replace_uncertain_outcome_is_reported_and_not_granted(
        self, tmp_path: Path, writer: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plan round-5 arm split (design P): the post-replace UNCERTAIN arm
        asserts the REPORTED OUTCOME — a durability caveat, never "definitely
        not saved" — and effective trust is not falsely granted by it.
        """
        trusted_file = TRUSTED_FOLDERS_FILE.path
        trusted_file.write_bytes(ORIGINAL_STORE)
        manager = TrustedFoldersManager()

        def boom_dir(directory: Path) -> None:
            raise OSError("simulated directory fsync failure")

        monkeypatch.setattr(_checked_store, "_fsync_directory", boom_dir)
        result = getattr(manager, writer)(tmp_path)

        assert result.outcome == CheckedWriteOutcome.DURABILITY_UNCERTAIN
        assert result.outcome != CheckedWriteOutcome.NOT_SAVED
        assert result.error is not None and "fsync" in result.error
        # The replace did happen (only its durability is unconfirmed)...
        with trusted_file.open("rb") as f:
            stored = tomllib.load(f)
        key = "trusted" if writer == "add_trusted" else "untrusted"
        assert str(tmp_path.resolve()) in stored[key]
        # ...but the uncertain result does not falsely grant effective trust.
        assert manager.is_trusted(tmp_path) is not True
        assert _store_tmp_files(trusted_file) == []

    def test_two_process_write_requires_restart_characterization(
        self, tmp_path: Path
    ) -> None:
        """CHARACTERIZATION ONLY (plan O4a2): the restart-required two-process
        arm passes before and after this row and is NOT discriminating
        acceptance. A second process's checked write lands on the store; the
        already-loaded manager's in-memory authority stays stale until a
        restart, and only a fresh manager observes the other process's grant.
        """
        trusted_file = TRUSTED_FOLDERS_FILE.path
        trusted_file.write_bytes(ORIGINAL_STORE)
        manager = TrustedFoldersManager()
        assert manager.is_trusted(Path("/child")) is None

        child = subprocess.run(
            [sys.executable, "-c", _CHILD_TRUST_WRITE, str(trusted_file)],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        assert child.stdout.strip() == CheckedWriteOutcome.SAVED.value

        # The live manager's in-memory lists are stale: restart required.
        assert manager.is_trusted(Path("/child")) is None
        # A fresh read (restart) observes the other process's write.
        assert TrustedFoldersManager().is_trusted(Path("/child")) is True


class TestIsTrustedInheritance:
    """Tests for the walk-up trust inheritance behaviour."""

    def test_child_of_trusted_folder_returns_true(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)
        child = tmp_path / "sub" / "deep"
        child.mkdir(parents=True)
        assert manager.is_trusted(child) is True

    def test_child_of_untrusted_folder_returns_false(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        manager.add_untrusted(tmp_path)
        child = tmp_path / "sub"
        child.mkdir()
        assert manager.is_trusted(child) is False

    def test_most_specific_ancestor_wins(self, tmp_path: Path) -> None:
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)

        manager = TrustedFoldersManager()
        manager.add_trusted(parent)
        manager.add_untrusted(child)

        assert manager.is_trusted(parent) is True
        assert manager.is_trusted(child) is False
        assert manager.is_trusted(child / "grandchild") is False

    def test_untrusted_parent_trusted_child(self, tmp_path: Path) -> None:
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)

        manager = TrustedFoldersManager()
        manager.add_untrusted(parent)
        manager.add_trusted(child)

        assert manager.is_trusted(parent) is False
        assert manager.is_trusted(child) is True
        assert manager.is_trusted(child / "grandchild") is True

    def test_deep_nesting_inherits_trust(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        assert manager.is_trusted(deep) is True

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        assert manager.is_trusted(tmp_path / "unknown") is None


class TestFindTrustRoot:
    def test_returns_path_when_path_is_trusted(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)
        assert manager.find_trust_root(tmp_path) == tmp_path.resolve()

    def test_returns_ancestor_when_child(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)
        child = tmp_path / "sub" / "deep"
        child.mkdir(parents=True)
        assert manager.find_trust_root(child) == tmp_path.resolve()

    def test_returns_none_when_no_trusted_ancestor(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        assert manager.find_trust_root(tmp_path) is None

    def test_returns_closest_trusted_ancestor(self, tmp_path: Path) -> None:
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)
        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)
        manager.add_trusted(parent)
        # child should find parent (closest), not tmp_path
        assert manager.find_trust_root(child) == parent.resolve()

    def test_returns_none_when_closer_ancestor_is_untrusted(
        self, tmp_path: Path
    ) -> None:
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)
        manager = TrustedFoldersManager()
        manager.add_untrusted(parent)
        manager.add_trusted(tmp_path)
        # The closer untrusted ancestor blocks the higher trusted one.
        assert manager.find_trust_root(child) is None

    def test_returns_none_when_path_itself_is_untrusted(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        manager.add_untrusted(tmp_path)
        assert manager.find_trust_root(tmp_path) is None


class TestIsExplicitlyUntrusted:
    def test_returns_true_for_path_in_untrusted_list(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        manager.add_untrusted(tmp_path)
        assert manager.is_explicitly_untrusted(tmp_path) is True

    def test_returns_false_for_unknown_path(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        assert manager.is_explicitly_untrusted(tmp_path) is False

    def test_does_not_walk_ancestors(self, tmp_path: Path) -> None:
        child = tmp_path / "child"
        child.mkdir()
        manager = TrustedFoldersManager()
        manager.add_untrusted(tmp_path)
        # parent is untrusted, but child itself is not in the list
        assert manager.is_explicitly_untrusted(child) is False

    def test_returns_false_for_trusted_path(self, tmp_path: Path) -> None:
        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)
        assert manager.is_explicitly_untrusted(tmp_path) is False


class TestHasAgentsMdFile:
    def test_returns_false_for_empty_directory(self, tmp_path: Path) -> None:
        assert has_agents_md_file(tmp_path) is False

    def test_returns_true_when_agents_md_exists(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("# Agents", encoding="utf-8")
        assert has_agents_md_file(tmp_path) is True

    def test_returns_false_when_only_other_files_exist(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("", encoding="utf-8")
        (tmp_path / ".chartreux").mkdir()
        assert has_agents_md_file(tmp_path) is False

    def test_agents_md_filename_constant(self) -> None:
        assert AGENTS_MD_FILENAME == "AGENTS.md"

    def test_unreadable_agents_md_does_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        denied = (tmp_path / "AGENTS.md").resolve()

        def fake_is_file(self: Path) -> bool:
            if self.resolve() == denied:
                raise PermissionError(13, "Permission denied")
            return False

        monkeypatch.setattr(Path, "is_file", fake_is_file)
        assert has_agents_md_file(tmp_path) is False


class TestFindTrustableFiles:
    def test_returns_empty_for_clean_directory(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        assert find_trustable_files(tmp_path) == []

    def test_detects_vibe_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".chartreux" / "tools").mkdir(parents=True)
        result = find_trustable_files(tmp_path)
        assert ".chartreux/" in result

    def test_detects_agents_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".agents" / "skills").mkdir(parents=True)
        result = find_trustable_files(tmp_path)
        assert ".agents/" in result

    def test_ignores_empty_vibe_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".chartreux").mkdir()
        assert find_trustable_files(tmp_path) == []

    def test_ignores_empty_agents_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".agents").mkdir()
        assert find_trustable_files(tmp_path) == []

    def test_detects_agents_md(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("# Agent", encoding="utf-8")
        result = find_trustable_files(tmp_path)
        assert "AGENTS.md" in result

    def test_returns_empty_when_no_trustable_content(self, tmp_path: Path) -> None:
        (tmp_path / "other.txt").write_text("", encoding="utf-8")
        assert find_trustable_files(tmp_path) == []

    def test_ignores_vibe_config_in_subfolder(self, tmp_path: Path) -> None:
        (tmp_path / "sub" / ".chartreux" / "skills").mkdir(parents=True)
        assert find_trustable_files(tmp_path) == []

    def test_ignores_agents_skills_in_subfolder(self, tmp_path: Path) -> None:
        (tmp_path / "deep" / "nested" / ".agents" / "skills").mkdir(parents=True)
        assert find_trustable_files(tmp_path) == []

    def test_returns_empty_when_config_only_in_subfolder(self, tmp_path: Path) -> None:
        (tmp_path / "node_modules" / ".chartreux" / "skills").mkdir(parents=True)
        assert find_trustable_files(tmp_path) == []

    def test_detects_multiple_files_at_root(self, tmp_path: Path) -> None:
        (tmp_path / ".chartreux" / "skills").mkdir(parents=True)
        (tmp_path / ".agents" / "skills").mkdir(parents=True)
        (tmp_path / "AGENTS.md").write_text("# Agent", encoding="utf-8")
        result = find_trustable_files(tmp_path)
        assert ".chartreux/" in result
        assert ".agents/" in result
        assert "AGENTS.md" in result

    def test_no_duplicates_for_root_vibe_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".chartreux" / "tools").mkdir(parents=True)
        result = find_trustable_files(tmp_path)
        assert result.count(".chartreux/") == 1


class TestFindUntrustedConfigDirs:
    def test_detects_untrusted_vibe_under_trusted_cwd(self, tmp_path: Path) -> None:
        vibe_dir = tmp_path / ".chartreux"
        (vibe_dir / "tools").mkdir(parents=True)
        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)
        manager.add_untrusted(vibe_dir)

        assert find_untrusted_config_dirs(tmp_path, manager=manager) == [
            vibe_dir.resolve()
        ]

    def test_returns_empty_when_cwd_not_trusted(self, tmp_path: Path) -> None:
        vibe_dir = tmp_path / ".chartreux"
        (vibe_dir / "tools").mkdir(parents=True)
        manager = TrustedFoldersManager()
        manager.add_untrusted(vibe_dir)

        # cwd itself is undecided, so there is nothing to recover.
        assert find_untrusted_config_dirs(tmp_path, manager=manager) == []

    def test_returns_empty_when_cwd_explicitly_untrusted(self, tmp_path: Path) -> None:
        vibe_dir = tmp_path / ".chartreux"
        (vibe_dir / "tools").mkdir(parents=True)
        manager = TrustedFoldersManager()
        manager.add_untrusted(tmp_path)
        manager.add_untrusted(vibe_dir)

        assert find_untrusted_config_dirs(tmp_path, manager=manager) == []

    def test_ignores_config_dir_that_is_not_explicitly_untrusted(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".chartreux" / "tools").mkdir(parents=True)
        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)

        assert find_untrusted_config_dirs(tmp_path, manager=manager) == []

    def test_ignores_empty_config_dir(self, tmp_path: Path) -> None:
        vibe_dir = tmp_path / ".chartreux"
        vibe_dir.mkdir()
        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)
        manager.add_untrusted(vibe_dir)

        # No content means find_local_config_dirs ignores it.
        assert find_untrusted_config_dirs(tmp_path, manager=manager) == []

    def test_detects_multiple_untrusted_config_dirs_sorted(
        self, tmp_path: Path
    ) -> None:
        vibe_dir = tmp_path / ".chartreux"
        agents_dir = tmp_path / ".agents"
        (vibe_dir / "tools").mkdir(parents=True)
        (agents_dir / "skills").mkdir(parents=True)
        manager = TrustedFoldersManager()
        manager.add_trusted(tmp_path)
        manager.add_untrusted(vibe_dir)
        manager.add_untrusted(agents_dir)

        assert find_untrusted_config_dirs(tmp_path, manager=manager) == [
            agents_dir.resolve(),
            vibe_dir.resolve(),
        ]


def _make_git_repo(path: Path) -> None:
    git_dir = path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")


class TestFindRepoTrustableFilesForCwd:
    def test_returns_empty_when_repo_root_is_none(self, tmp_path: Path) -> None:
        assert find_repo_trustable_files_for_cwd(tmp_path, None) == []

    def test_returns_empty_when_repo_root_is_not_ancestor(self, tmp_path: Path) -> None:
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        repo = tmp_path / "repo"
        repo.mkdir()
        _make_git_repo(repo)
        assert find_repo_trustable_files_for_cwd(cwd, repo) == []

    def test_includes_root_trustable_files(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _make_git_repo(repo)
        (repo / ".chartreux" / "skills").mkdir(parents=True)
        (repo / ".agents" / "skills").mkdir(parents=True)
        (repo / "AGENTS.md").write_text("# Agent", encoding="utf-8")
        cwd = repo / "src" / "pkg"
        cwd.mkdir(parents=True)

        assert find_repo_trustable_files_for_cwd(cwd, repo) == [
            ".agents/",
            ".chartreux/",
            "AGENTS.md",
        ]

    def test_includes_agents_md_between_cwd_and_repo_root(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _make_git_repo(repo)
        cwd = repo / "src" / "pkg" / "deep"
        cwd.mkdir(parents=True)
        (repo / "src" / "AGENTS.md").write_text("# Source", encoding="utf-8")
        (repo / "src" / "pkg" / "AGENTS.md").write_text("# Package", encoding="utf-8")
        (cwd / "AGENTS.md").write_text("# Cwd", encoding="utf-8")

        assert find_repo_trustable_files_for_cwd(cwd, repo) == [
            "src/AGENTS.md",
            "src/pkg/AGENTS.md",
        ]


class TestFindGitRepoAncestor:
    def test_returns_path_when_directly_contains_git(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)
        assert find_git_repo_ancestor(tmp_path) == tmp_path.resolve()

    def test_ignores_git_file_pointer(self, tmp_path: Path) -> None:
        (tmp_path / ".git").write_text("gitdir: /elsewhere", encoding="utf-8")
        # Not a directory, so not treated as a repo root.
        assert find_git_repo_ancestor(tmp_path) is None

    def test_ignores_empty_git_directory(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        # Missing HEAD, so not a real repo.
        assert find_git_repo_ancestor(tmp_path) is None

    def test_returns_closest_ancestor_with_git(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        assert find_git_repo_ancestor(nested) == tmp_path.resolve()

    def test_returns_innermost_when_multiple_git_repos(self, tmp_path: Path) -> None:
        _make_git_repo(tmp_path)
        inner = tmp_path / "sub"
        inner.mkdir()
        _make_git_repo(inner)
        child = inner / "deep"
        child.mkdir()
        assert find_git_repo_ancestor(child) == inner.resolve()

    def test_returns_none_when_no_git_anywhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path.parent))
        assert find_git_repo_ancestor(tmp_path / "a") is None

    def test_excludes_home_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        _make_git_repo(home)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
        sub = home / "project"
        sub.mkdir()
        assert find_git_repo_ancestor(sub) is None

    def test_terminates_at_filesystem_root_without_git(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "nope"))
        assert find_git_repo_ancestor(tmp_path) is None

    def test_unreadable_ancestor_git_head_does_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ancestor = tmp_path / "mnt" / "vast"
        (ancestor / ".git").mkdir(parents=True)
        cwd = ancestor / "project"
        cwd.mkdir()

        real_is_file = Path.is_file
        denied = (ancestor / ".git" / "HEAD").resolve()

        def fake_is_file(self: Path) -> bool:
            if self.resolve() == denied:
                raise PermissionError(13, "Permission denied")
            return real_is_file(self)

        monkeypatch.setattr(Path, "is_file", fake_is_file)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))

        assert find_git_repo_ancestor(cwd) is None
