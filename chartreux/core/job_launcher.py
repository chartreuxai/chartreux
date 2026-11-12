"""The background-job launcher shim, its durable record, and the launch plan.

A background job may outlive the root agent process, so its leader guarantee is
kernel-enforced rather than protocol-ordered. A job spawn is therefore the
pinned executor argv prefixed with this shim::

    python <this file> <jobs-dir> <job-id> <parent-pid> <session-id> -- <argv...>

Before it execs ``argv`` the shim

1. writes its own pid and /proc-verified start time to
   ``<jobs-dir>/<job-id>.pgid`` -- durably, BEFORE anything is armed, so the
   coordinator can reconstruct ownership from disk even inside the
   spawn-to-registration window;
2. arms ``PR_SET_PDEATHSIG=SIGKILL`` on itself;
3. re-reads ``getppid()`` and exits if the root already died, which closes the
   arm race (no signal would come otherwise);
4. execs ``argv``, keeping this pid -- the recorded leader identity stays valid
   across the exec, and the inherited standard streams (the hook stdin=PIPE
   contract) are untouched.

Every refusal aborts the spawn visibly and before the arming step: a host with
no ``prctl`` never runs the job, and a jobs directory that cannot be created or
recorded into fails the launch instead of producing an unowned process. The
module is stdlib-only on purpose -- it must start without the package importable
-- and importing it spawns nothing; the caller that owns the cleanup course
binds the plan.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import sys
import uuid

JOB_GROUP_RECORD_SUFFIX = ".pgid"
"""The jobs-directory file that records one job's group identity."""

JOB_GROUP_RECORD_KEYS = ("job_id", "pid", "start_time", "session_id")
"""The record's members: the job, its leader, and the start time that pins it."""

JOB_OWNER_MARKER_ENV = "CHARTREUX_JOB_OWNER"
"""The id-bearing env var every job environment carries, inherited by its tree."""

EXIT_USAGE = 64
"""The shim was invoked without the launch plan's arguments."""

EXIT_PLATFORM_UNSUPPORTED = 71
"""The host has no parent-death signal, so a job cannot be covered."""

EXIT_JOBS_DIR_UNUSABLE = 72
"""The jobs directory could not be created: the launch aborts before arming."""

EXIT_RECORD_UNWRITABLE = 73
"""The pgid record could not be written durably: the launch aborts."""

EXIT_PARENT_GONE = 74
"""The root died between the spawn and the arming step: nothing is exec'd."""

_USAGE = (
    "usage: job_launcher <jobs-dir> <job-id> <parent-pid> <session-id> -- <argv...>"
)

_PR_SET_PDEATHSIG = 1


class JobLaunchRefused(RuntimeError):
    """A job spawn this host or this session cannot make fail-closed.

    ``reason`` is the machine-readable class (logging disabled, unsupported
    platform); the message names the requirement the caller must satisfy.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


JOB_LAUNCH_LOGGING_DISABLED = "logging-disabled"
JOB_LAUNCH_PLATFORM_UNSUPPORTED = "platform-unsupported"


@dataclass(frozen=True, slots=True)
class JobGroupRecord:
    """One job's recorded group identity: its leader pinned by start time."""

    job_id: str
    pid: int
    start_time: int
    session_id: str

    def to_payload(self) -> dict[str, str | int]:
        return {
            "job_id": self.job_id,
            "pid": self.pid,
            "start_time": self.start_time,
            "session_id": self.session_id,
        }


def job_group_record_path(jobs_dir: Path, job_id: str) -> Path:
    """The path of one job's group record inside its jobs directory."""
    return jobs_dir / f"{job_id}{JOB_GROUP_RECORD_SUFFIX}"


def write_job_group_record(jobs_dir: Path, record: JobGroupRecord) -> Path:
    """Write one record durably before anything is armed or exec'd.

    The payload lands through a fsynced temporary file and a rename, and the
    directory entry is fsynced too, so a crash cannot leave a readable record
    with unwritten content.
    """
    path = job_group_record_path(jobs_dir, record.job_id)
    payload = json.dumps(record.to_payload(), sort_keys=True).encode()
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(jobs_dir)
    return path


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_job_group_record(path: Path) -> JobGroupRecord | None:
    """Read one record; an unreadable or malformed file proves nothing."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or any(
        name not in payload for name in JOB_GROUP_RECORD_KEYS
    ):
        return None
    job_id = payload["job_id"]
    session_id = payload["session_id"]
    pid = payload["pid"]
    start_time = payload["start_time"]
    if not isinstance(job_id, str) or not isinstance(session_id, str):
        return None
    if not job_id or isinstance(pid, bool) or not isinstance(pid, int):
        return None
    if isinstance(start_time, bool) or not isinstance(start_time, int):
        return None
    return JobGroupRecord(
        job_id=job_id, pid=pid, start_time=start_time, session_id=session_id
    )


def process_start_time(pid: int) -> int | None:
    """Linux start-time of one pid (boot ticks), or ``None`` when it is absent."""
    try:
        with open(f"/proc/{pid}/stat") as stream:
            fields = stream.read().rsplit(")", 1)[1].split()
        return int(fields[19])
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def supports_parent_death_signal(platform: str | None = None) -> bool:
    """Whether this host can arm a parent-death signal (Linux ``prctl`` only)."""
    return (platform if platform is not None else sys.platform).startswith("linux")


@dataclass(frozen=True, slots=True)
class JobLaunchPlan:
    """The one job spawn a session may make, and how its argv is wrapped."""

    jobs_dir: Path
    session_id: str
    parent_pid: int

    def argv_for(self, job_id: str, argv: list[str] | tuple[str, ...]) -> list[str]:
        """The launch argv: this shim, the record's identity, then the target."""
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        return [
            sys.executable,
            str(Path(__file__).resolve()),
            str(self.jobs_dir),
            job_id,
            str(self.parent_pid),
            self.session_id,
            "--",
            *argv,
        ]

    def wrap(self, argv: list[str] | tuple[str, ...]) -> list[str]:
        """Wrap one argv under a freshly minted job id.

        The spawner that already minted its own id uses :meth:`argv_for`; the
        hook executor has no public job id, so it takes this one.
        """
        return self.argv_for(self.mint_job_id(), argv)

    def record_path(self, job_id: str) -> Path:
        return job_group_record_path(self.jobs_dir, job_id)

    @staticmethod
    def mint_job_id() -> str:
        return f"job-{os.getpid():x}-{uuid.uuid4().hex[:12]}"

    @property
    def owner_marker(self) -> dict[str, str]:
        """The env entry that marks every process in this session's job trees."""
        return {JOB_OWNER_MARKER_ENV: self.session_id}


def plan_job_launch(
    *,
    jobs_dir: Path | None,
    session_id: str,
    parent_pid: int,
    platform: str | None = None,
) -> JobLaunchPlan:
    """The plan one session's job spawns use, or the refusal that forbids them.

    A session with logging disabled publishes no jobs directory, so it cannot
    record a group identity and cannot make the leader guarantee: the admission
    is refused typed. A host without ``prctl`` can neither, so it is refused on
    the same terms -- the job is never spawned unwatched.
    """
    if not supports_parent_death_signal(platform):
        raise JobLaunchRefused(
            JOB_LAUNCH_PLATFORM_UNSUPPORTED,
            "background jobs need a parent-death signal, which only Linux "
            "provides on this build; the job was not started",
        )
    if jobs_dir is None:
        raise JobLaunchRefused(
            JOB_LAUNCH_LOGGING_DISABLED,
            "background jobs need a jobs directory to record their leader "
            "identity, and this session has logging disabled; enable session "
            "logging (session_logging.enabled) or run the work in the "
            "foreground",
        )
    return JobLaunchPlan(
        jobs_dir=jobs_dir, session_id=session_id, parent_pid=parent_pid
    )


def _arm_parent_death_signal() -> bool:
    """Arm ``PR_SET_PDEATHSIG=SIGKILL``; ``False`` when the host cannot."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return bool(libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) == 0)
    except (OSError, AttributeError):
        return False


_PLAN_ARGUMENTS = 6
"""The plan prefix plus its ``--`` separator: at least one target follows."""


def _parse(argv: list[str]) -> tuple[Path, str, int, str, list[str]] | None:
    """Split ``<jobs-dir> <job-id> <parent-pid> <session-id> -- <argv...>``."""
    if len(argv) < _PLAN_ARGUMENTS or argv[4] != "--":
        return None
    jobs_dir, job_id, parent_pid, session_id = argv[0], argv[1], argv[2], argv[3]
    try:
        parsed_parent = int(parent_pid)
    except ValueError:
        return None
    return Path(jobs_dir), job_id, parsed_parent, session_id, argv[5:]


def _record_identity(jobs_dir: Path, job_id: str, session_id: str) -> int:
    """Create the jobs directory and record this leader durably.

    Returns ``0`` once the record is on disk, or the exit code of the refusal
    that stopped the launch before anything was armed or exec'd.
    """
    try:
        jobs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sys.stderr.write(f"job_launcher: jobs directory unavailable: {exc}{chr(10)}")
        return EXIT_JOBS_DIR_UNUSABLE
    pid = os.getpid()
    start_time = process_start_time(pid)
    if start_time is None:
        sys.stderr.write("job_launcher: the leader identity is unreadable" + chr(10))
        return EXIT_RECORD_UNWRITABLE
    try:
        write_job_group_record(
            jobs_dir,
            JobGroupRecord(
                job_id=job_id, pid=pid, start_time=start_time, session_id=session_id
            ),
        )
    except OSError as exc:
        sys.stderr.write(
            f"job_launcher: the record could not be written: {exc}{chr(10)}"
        )
        return EXIT_RECORD_UNWRITABLE
    return 0


def main(argv: list[str]) -> int:
    """Run the shim: record, arm, re-check the parent, then exec.

    Every failure is a visible non-zero exit BEFORE the target runs, so a job
    that cannot be owned is never started.
    """
    parsed = _parse(argv)
    if parsed is None:
        sys.stderr.write(_USAGE + chr(10))
        return EXIT_USAGE
    jobs_dir, job_id, parent_pid, session_id, command = parsed
    if not supports_parent_death_signal():
        sys.stderr.write(
            "job_launcher: this host has no parent-death signal; nothing ran" + chr(10)
        )
        return EXIT_PLATFORM_UNSUPPORTED
    refused = _record_identity(jobs_dir, job_id, session_id)
    if refused:
        return refused
    if not _arm_parent_death_signal():
        sys.stderr.write(
            "job_launcher: the parent-death signal could not be armed; nothing ran"
            + chr(10)
        )
        return EXIT_PLATFORM_UNSUPPORTED
    if os.getppid() != parent_pid:
        sys.stderr.write("job_launcher: the root already died; nothing ran" + chr(10))
        return EXIT_PARENT_GONE
    os.execvp(command[0], command)
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
