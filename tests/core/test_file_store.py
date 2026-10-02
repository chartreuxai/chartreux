from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import IO, Any
from unittest.mock import patch

import pytest

from chartreux.core.checkpoints import DiskFilesystem, FileState, FileStore
from tests.stubs.fake_filesystem import FakeFilesystem


class TestFileStoreRead:
    def test_read_present_file_returns_its_bytes(self) -> None:
        store = FileStore(FakeFilesystem({"a.txt": b"hello"}))
        assert store.read("a.txt") == FileState(b"hello")

    def test_read_missing_file_returns_absent(self) -> None:
        store = FileStore(FakeFilesystem())
        assert store.read("missing.txt") == FileState.absent()

    def test_read_unreadable_file_raises(self) -> None:
        fs = FakeFilesystem({"a.txt": b"hello"})
        fs.fail_reads.add("a.txt")
        store = FileStore(fs)
        with pytest.raises(OSError):
            store.read("a.txt")


class TestFileStoreApply:
    def test_writes_present_states(self) -> None:
        fs = FakeFilesystem()
        store = FileStore(fs)
        errors, restored = store.apply({"a.txt": FileState(b"new")})
        assert errors == []
        assert restored == ["a.txt"]
        assert fs.files["a.txt"] == b"new"

    def test_deletes_absent_state_when_file_exists(self) -> None:
        fs = FakeFilesystem({"a.txt": b"old"})
        store = FileStore(fs)
        errors, restored = store.apply({"a.txt": FileState.absent()})
        assert errors == []
        assert restored == ["a.txt"]
        assert "a.txt" not in fs.files

    def test_deleting_already_absent_file_is_a_noop(self) -> None:
        fs = FakeFilesystem()
        store = FileStore(fs)
        errors, restored = store.apply({"gone.txt": FileState.absent()})
        assert errors == []
        assert restored == []

    def test_skips_write_when_content_already_matches(self) -> None:
        fs = FakeFilesystem({"a.txt": b"same"})
        store = FileStore(fs)
        errors, restored = store.apply({"a.txt": FileState(b"same")})
        assert errors == []
        assert restored == []

    def test_reports_error_when_write_fails(self) -> None:
        fs = FakeFilesystem()
        fs.fail_writes.add("a.txt")
        store = FileStore(fs)
        errors, restored = store.apply({"a.txt": FileState(b"new")})
        assert restored == []
        assert errors == ["Failed to restore file: a.txt"]

    def test_reports_error_when_delete_fails(self) -> None:
        fs = FakeFilesystem({"a.txt": b"old"})
        fs.fail_removes.add("a.txt")
        store = FileStore(fs)
        errors, restored = store.apply({"a.txt": FileState.absent()})
        assert restored == []
        assert errors == ["Failed to delete file: a.txt"]

    @pytest.mark.parametrize("error_type", [PermissionError, OSError])
    def test_read_failure_is_reported_and_other_files_restore(
        self, error_type: type[OSError]
    ) -> None:
        fs = FakeFilesystem({"bad.txt": b"old", "good.txt": b"old"})
        read_bytes = fs.read_bytes

        def read(path: str) -> bytes | None:
            if path == "bad.txt":
                raise error_type("read failed")
            return read_bytes(path)

        with patch.object(fs, "read_bytes", side_effect=read):
            errors, restored = FileStore(fs).apply({
                "bad.txt": FileState(b"new"),
                "good.txt": FileState(b"new"),
            })
        assert errors == ["Failed to restore file: bad.txt"]
        assert restored == ["good.txt"]
        assert fs.files == {"bad.txt": b"old", "good.txt": b"new"}

    @pytest.mark.parametrize("error_type", [PermissionError, OSError])
    def test_stat_failure_is_reported_and_other_files_restore(
        self, error_type: type[OSError]
    ) -> None:
        fs = FakeFilesystem({"bad.txt": b"old", "good.txt": b"old"})
        with patch.object(fs, "exists", side_effect=error_type("stat failed")):
            errors, restored = FileStore(fs).apply({
                "bad.txt": FileState.absent(),
                "good.txt": FileState(b"new"),
            })
        assert errors == ["Failed to delete file: bad.txt"]
        assert restored == ["good.txt"]
        assert fs.files == {"bad.txt": b"old", "good.txt": b"new"}

    def test_aggregates_across_a_mixed_plan(self) -> None:
        fs = FakeFilesystem({"keep.txt": b"v", "del.txt": b"x"})
        fs.fail_writes.add("boom.txt")
        store = FileStore(fs)
        errors, restored = store.apply({
            "write.txt": FileState(b"created"),
            "del.txt": FileState.absent(),
            "keep.txt": FileState(b"v"),
            "boom.txt": FileState(b"nope"),
        })
        assert set(restored) == {"write.txt", "del.txt"}
        assert errors == ["Failed to restore file: boom.txt"]
        assert fs.files["write.txt"] == b"created"
        assert "del.txt" not in fs.files


class TestDiskFilesystem:
    def test_write_then_read_round_trips_and_creates_parents(
        self, tmp_path: Path
    ) -> None:
        fs = DiskFilesystem()
        target = tmp_path / "nested" / "dir" / "a.txt"
        fs.write_bytes(str(target), b"payload")
        assert fs.read_bytes(str(target)) == b"payload"
        assert fs.exists(str(target))

    def test_failed_staging_preserves_original_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "a.txt"
        target.write_bytes(b"original")
        open_file = open

        def failing_open(path: Path, mode: str = "r", **kwargs: Any) -> IO[Any]:
            staged = open_file(path, mode, **kwargs)
            if mode == "xb":
                write = staged.write

                def fail_write(data: bytes) -> int:
                    write(data[:2])
                    raise OSError(errno.ENOSPC, "no space left")

                monkeypatch.setattr(staged, "write", fail_write)
            return staged

        with patch("chartreux.core.checkpoints.fs.open", failing_open):
            errors, restored = FileStore().apply({
                str(target): FileState(b"replacement")
            })
        assert errors == [f"Failed to restore file: {target}"]
        assert restored == []
        assert target.read_bytes() == b"original"
        assert list(tmp_path.iterdir()) == [target]

    def test_failed_replacement_preserves_original_bytes(self, tmp_path: Path) -> None:
        target = tmp_path / "a.txt"
        target.write_bytes(b"original")
        with patch(
            "chartreux.core.checkpoints.fs.os.replace", side_effect=OSError("failed")
        ):
            errors, restored = FileStore().apply({
                str(target): FileState(b"replacement")
            })
        assert errors == [f"Failed to restore file: {target}"]
        assert restored == []
        assert target.read_bytes() == b"original"
        assert list(tmp_path.iterdir()) == [target]

    def test_restore_replaces_content_and_preserves_mode(self, tmp_path: Path) -> None:
        target = tmp_path / "a.txt"
        target.write_bytes(b"original")
        target.chmod(0o751)
        errors, restored = FileStore().apply({str(target): FileState(b"replacement")})
        assert errors == []
        assert restored == [str(target)]
        assert target.read_bytes() == b"replacement"
        assert target.stat().st_mode & 0o777 == 0o751
        assert list(tmp_path.iterdir()) == [target]

    @pytest.mark.parametrize("umask", [0o022, 0o027, 0o077])
    def test_restore_missing_file_uses_normal_creation_mode(
        self, tmp_path: Path, umask: int
    ) -> None:
        target = tmp_path / "missing.txt"
        previous_umask = os.umask(umask)
        try:
            errors, restored = FileStore().apply({str(target): FileState(b"restored")})
        finally:
            os.umask(previous_umask)
        assert errors == []
        assert restored == [str(target)]
        assert target.read_bytes() == b"restored"
        assert target.stat().st_mode & 0o777 == 0o666 & ~umask
        assert list(tmp_path.iterdir()) == [target]

    def test_private_staging_is_restricted_before_bytes_are_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "private.txt"
        target.write_bytes(b"original")
        target.chmod(0o600)
        import shutil

        copymode = shutil.copymode
        observed = False

        def check_mode(source: Path, staging: Path) -> None:
            nonlocal observed
            assert staging.stat().st_mode & 0o777 == 0o600
            assert staging.read_bytes() == b""
            observed = True
            copymode(source, staging)

        monkeypatch.setattr("chartreux.core.checkpoints.fs.shutil.copymode", check_mode)
        DiskFilesystem().write_bytes(str(target), b"private replacement")
        assert observed
        assert target.read_bytes() == b"private replacement"
        assert target.stat().st_mode & 0o777 == 0o600

    def test_restore_name_max_basename(self, tmp_path: Path) -> None:
        target = tmp_path / ("a" * os.pathconf(tmp_path, "PC_NAME_MAX"))
        target.write_bytes(b"original")
        errors, restored = FileStore().apply({str(target): FileState(b"replacement")})
        assert errors == []
        assert restored == [str(target)]
        assert target.read_bytes() == b"replacement"
        assert list(tmp_path.iterdir()) == [target]

    def test_read_missing_returns_none(self, tmp_path: Path) -> None:
        fs = DiskFilesystem()
        assert fs.read_bytes(str(tmp_path / "missing.txt")) is None

    def test_read_unreadable_path_raises_instead_of_absent(
        self, tmp_path: Path
    ) -> None:
        # A path that exists but cannot be read as a file (here, a directory)
        # must raise rather than be silently reported as absent.
        fs = DiskFilesystem()
        with pytest.raises(OSError):
            fs.read_bytes(str(tmp_path))

    def test_remove_deletes_the_file(self, tmp_path: Path) -> None:
        fs = DiskFilesystem()
        target = tmp_path / "a.txt"
        target.write_bytes(b"x")
        fs.remove(str(target))
        assert not target.exists()
