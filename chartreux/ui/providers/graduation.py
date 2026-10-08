"""Graduation eligibility, independent of UI focus and session policy mutation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
import fcntl
from pathlib import Path
import tempfile
import tomllib

from chartreux.utils.paths import get_chartreux_home

MIN_USABLE_MODELS = 2
FAILURE_THRESHOLD = 2


@dataclass
class GraduationState:
    second_model_saved: bool = False
    need_signal: bool = False
    shown: bool = False
    dismissed: bool = False
    failures: dict[str, set[str]] = field(default_factory=dict)

    def model_saved(
        self,
        usable_models: frozenset[str],
        saved_models: frozenset[str],
        *,
        succeeded: bool = True,
        linked: bool = False,
    ) -> None:
        """Names are canonical identities; saved_models includes newly ready models."""
        if succeeded and not linked and len(usable_models) >= MIN_USABLE_MODELS:
            self.second_model_saved |= bool(saved_models & usable_models)

    def compacted(self, *, replayed: bool = False) -> None:
        if not replayed:
            self.need_signal = True

    def implementation_failed(
        self,
        task_id: str,
        attempt_id: str,
        *,
        attempt_budgeted: bool,
        replayed: bool = False,
        cancelled: bool = False,
        transient: bool = False,
    ) -> None:
        if not task_id or not attempt_id or not attempt_budgeted:
            return
        if replayed or cancelled or transient:
            return
        attempts = self.failures.setdefault(task_id, set())
        # This signal only needs two consumed attempts; never grow an unbounded
        # failure counter or reset the task's budget when a snapshot repeats.
        if len(attempts) < FAILURE_THRESHOLD:
            attempts.add(attempt_id)
        self.need_signal |= len(attempts) >= FAILURE_THRESHOLD

    def eligible(self, *, mode: str, idle: bool, headless: bool) -> bool:
        return (
            self.second_model_saved
            and self.need_signal
            and not self.shown
            and not self.dismissed
            and mode == "standalone"
            and idle
            and not headless
        )


class GraduationStore:
    """Small user-owned TOML file; loading never creates or repairs it."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_chartreux_home() / "graduation.toml"
        self.state = GraduationState()
        self.load_error = False
        try:
            with self.path.open("rb") as stream:
                data = tomllib.load(stream)
            for key in ("second_model_saved", "shown", "dismissed"):
                value = data.get(key, False)
                if not isinstance(value, bool):
                    raise ValueError("Invalid graduation state")
                setattr(self.state, key, value)
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            # Do not clobber an invalid user file or repeatedly show a nudge.
            self.load_error = True
            self.state.shown = True

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def save(self) -> bool:
        if self.load_error:
            return False
        temporary: Path | None = None
        try:
            with self._file_lock():
                # Another process may have shown or dismissed the notice since load.
                # Durable flags are monotonic; never replace them with stale false.
                durable = GraduationStore(self.path)
                if durable.load_error:
                    self.load_error = True
                    self.state.shown = True
                    return False
                for key in ("second_model_saved", "shown", "dismissed"):
                    setattr(
                        self.state,
                        key,
                        getattr(self.state, key) or getattr(durable.state, key),
                    )
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=self.path.parent, prefix=".graduation-", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    for key in ("second_model_saved", "shown", "dismissed"):
                        stream.write(
                            f"{key} = {str(getattr(self.state, key)).lower()}\n"
                        )
                temporary.replace(self.path)
            return True
        except OSError:
            return False
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
