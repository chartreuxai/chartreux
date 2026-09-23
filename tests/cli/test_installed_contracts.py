from __future__ import annotations

import asyncio
import configparser
import contextlib
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tomllib
from typing import Any
import zipfile

from acp import PROTOCOL_VERSION, Client, connect_to_agent
from acp.schema import ClientCapabilities, Implementation, TextContentBlock
from packaging.requirements import Requirement
from packaging.utils import NormalizedName, canonicalize_name
import pytest

from tests.stubs.fake_installed_provider import FakeInstalledProvider

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _AcpClient(Client):
    def __init__(self) -> None:
        self.updates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def on_connect(self, conn: Any) -> None:
        del conn

    async def request_permission(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("routine permission callback was not expected")

    async def write_text_file(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("file callback was not expected")

    async def read_text_file(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("file callback was not expected")

    async def create_terminal(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("terminal callback was not expected")

    async def create_elicitation(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("elicitation callback was not expected")

    async def complete_elicitation(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("elicitation callback was not expected")

    async def terminal_output(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("terminal callback was not expected")

    async def release_terminal(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("terminal callback was not expected")

    async def wait_for_terminal_exit(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("terminal callback was not expected")

    async def kill_terminal(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("terminal callback was not expected")

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        del params
        raise RuntimeError(f"unexpected extension request: {method}")

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        del params
        raise RuntimeError(f"unexpected extension notification: {method}")

    async def session_update(self, *args: Any, **kwargs: Any) -> None:
        self.updates.append((args, kwargs))


class _TeeStreamReader(asyncio.StreamReader):
    def __init__(self, source: asyncio.StreamReader, capture: bytearray) -> None:
        super().__init__()
        self._source = source
        self._capture = capture

    async def readuntil(self, separator: Any = b"\n") -> bytes:
        try:
            data = await self._source.readuntil(separator)
        except asyncio.IncompleteReadError as exc:
            self._capture.extend(exc.partial)
            raise
        else:
            self._capture.extend(data)
            return data

    async def readexactly(self, n: int) -> bytes:
        data = await self._source.readexactly(n)
        self._capture.extend(data)
        return data


_REMOVED_ROOTS = {
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-semantic-conventions",
    "rfc8785",
    "websockets",
    "zstandard",
    "twine",
}
_RETIRED_PATHS = (
    "vibe/",
    ".vscode/",
    "distribution/zed/",
    "chartreux/cli/vscode_extension_promo/",
    "chartreux/core/telemetry/",
    "chartreux/core/analytics/",
    "chartreux/observability/telemetry.py",
    "chartreux/observability/tracing.py",
    "chartreux/core/tracing.py",
    "chartreux/core/plugins/",
    "chartreux/app_server/_plugin_mcp.py",
    "chartreux/app_server/_plugins.py",
    "chartreux/app_server/plugin_catalog.py",
    "chartreux/cli/textual_ui/widgets/plugins_app.py",
    "chartreux/cli/textual_ui/widgets/skills_browser.py",
    "chartreux/setup/onboarding/screens/auth_method.py",
    "chartreux/setup/onboarding/screens/custom_domain.py",
    "chartreux/setup/onboarding/screens/sign_in_target.py",
)
_SHIPPED_RESOURCE_PATHS = {
    "chartreux/core/tools/builtins/prompts/ask_user_question.md",
    "chartreux/core/tools/builtins/prompts/bash.md",
    "chartreux/core/tools/builtins/prompts/check_agents.md",
    "chartreux/core/tools/builtins/prompts/edit.md",
    "chartreux/core/tools/builtins/prompts/get_agent_result.md",
    "chartreux/core/tools/builtins/prompts/grep.md",
    "chartreux/core/tools/builtins/prompts/read_file.md",
    "chartreux/core/tools/builtins/prompts/read_image.md",
    "chartreux/core/tools/builtins/prompts/release_agent.md",
    "chartreux/core/tools/builtins/prompts/skill.md",
    "chartreux/core/tools/builtins/prompts/task.md",
    "chartreux/core/tools/builtins/prompts/todo.md",
    "chartreux/core/tools/builtins/prompts/wait_for_agent.md",
    "chartreux/core/tools/builtins/prompts/web_fetch.md",
    "chartreux/core/tools/builtins/prompts/web_search.md",
    "chartreux/core/tools/builtins/prompts/write_file.md",
    "chartreux/core/skills/builtins/chartreux.py",
    "chartreux/core/skills/builtins/skill_creator.py",
    "chartreux/skills/main-debugging/SKILL.md",
    "chartreux/skills/workspace/SKILL.md",
    "chartreux/cli/textual_ui/app.tcss",
    "chartreux/setup/onboarding/onboarding.tcss",
    "chartreux/setup/trusted_folders/trust_folder_dialog.tcss",
}


def _assert_distribution_contents(wheel: Path, source: Path) -> None:
    project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any(name.startswith(_RETIRED_PATHS) for name in names)
        assert _SHIPPED_RESOURCE_PATHS <= set(names)
        (metadata_name,) = (
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        assert metadata["Name"] == project["name"]
        assert metadata["Version"] == project["version"]
        assert metadata["Author"] == "Chartreux"
        assert "Apache" in str(metadata["License"])
        requirements = [
            Requirement(raw) for raw in metadata.get_all("Requires-Dist", [])
        ]
        assert not _REMOVED_ROOTS & {
            canonicalize_name(req.name) for req in requirements
        }
        assert {str(req) for req in requirements} == {
            str(Requirement(raw)) for raw in project["dependencies"]
        }
        (license_name,) = (name for name in names if name.endswith("/licenses/LICENSE"))
        assert archive.read(license_name) == (source / "LICENSE").read_bytes()
        (entrypoint_name,) = (
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(entrypoint_name).decode())
        assert dict(parser["console_scripts"]) == project["scripts"]
        assert len(parser["console_scripts"]) == 3


def _assert_sdist_executable_modes(
    archive: tarfile.TarFile, source: Path, tracked: list[str]
) -> None:
    members = {
        member.name.partition("/")[2]: member
        for member in archive.getmembers()
        if "/" in member.name
    }
    for name in tracked:
        path = source / name
        if not path.is_file():
            continue
        source_mode = path.stat().st_mode & 0o111
        if not source_mode:
            continue
        member = members.get(name)
        if member is None:
            continue
        assert member.isfile(), name
        assert member.mode & 0o111 == source_mode, (
            f"{name}: sdist mode {member.mode:o} does not retain "
            f"source executable bits {source_mode:o}"
        )


@pytest.fixture(scope="module")
def installed_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("installed-contracts")
    dist = root / "dist"
    dist.mkdir()
    source = root / "source"
    source.mkdir()
    tracked = [
        name
        for name in (
            subprocess
            .check_output(["git", "ls-files", "-z"], cwd=_PROJECT_ROOT)
            .decode()
            .split("\0")
        )
        if name and (_PROJECT_ROOT / name).is_file()
    ]
    for name in tracked:
        relative = Path(name)
        assert not any(
            part == ".env" or part.startswith(".env.") for part in relative.parts
        )
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_PROJECT_ROOT / relative, destination)
    uv = shutil.which("uv")
    assert uv is not None, "installed contracts require uv on PATH"
    assert os.environ.get("UV_FIND_LINKS"), (
        "installed contracts require UV_FIND_LINKS pointing to a pre-provisioned "
        "local wheelhouse; builds and installs run offline with no index"
    )
    # Registry identity is part of lock freshness. The runner's Linux-only local
    # index is suitable for builds, not for checking the universal PyPI lock.
    check_env = dict(os.environ)
    for name in ("UV_CONFIG_FILE", "UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_FIND_LINKS"):
        check_env.pop(name, None)
    lock_before = (source / "uv.lock").read_bytes()
    lock_check = subprocess.run(
        [
            uv,
            "lock",
            "--check",
            "--offline",
            "--default-index",
            "https://pypi.org/simple",
        ],
        cwd=source,
        env=check_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (root / "lock-check.log").write_text(lock_check.stdout + lock_check.stderr)
    assert lock_check.returncode == 0, lock_check.stdout + lock_check.stderr
    assert (source / "uv.lock").read_bytes() == lock_before
    subprocess.run(
        [
            uv,
            "build",
            "--offline",
            "--no-index",
            "--wheel",
            "--sdist",
            "--out-dir",
            str(dist),
        ],
        cwd=source,
        check=True,
        timeout=120,
    )
    wheels = sorted(dist.glob("chartreux-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    sdists = sorted(dist.glob("chartreux-*.tar.gz"))
    assert len(sdists) == 1
    unpacked = root / "unpacked"
    with tarfile.open(sdists[0]) as archive:
        _assert_sdist_executable_modes(archive, source, tracked)
        assert not any(
            member.name.partition("/")[2].startswith(_RETIRED_PATHS)
            for member in archive.getmembers()
        )
        archive.extractall(unpacked, filter="data")
    (sdist_source,) = unpacked.iterdir()
    rebuilt = root / "rebuilt"
    subprocess.run(
        [uv, "build", "--offline", "--no-index", "--wheel", "--out-dir", str(rebuilt)],
        cwd=sdist_source,
        check=True,
        timeout=120,
    )
    (rebuilt_wheel,) = rebuilt.glob("chartreux-*.whl")
    _assert_distribution_contents(wheel, source)
    _assert_distribution_contents(rebuilt_wheel, source)
    assert (sdist_source / "LICENSE").read_bytes() == (source / "LICENSE").read_bytes()
    assert (sdist_source / "pyproject.toml").read_bytes() == (
        source / "pyproject.toml"
    ).read_bytes()
    with (
        zipfile.ZipFile(wheel) as direct,
        zipfile.ZipFile(rebuilt_wheel) as rebuilt_archive,
    ):
        assert {name: direct.read(name) for name in direct.namelist()} == {
            name: rebuilt_archive.read(name) for name in rebuilt_archive.namelist()
        }
    print(
        f"BUILD_EVIDENCE root={root} wheel_sha256={hashlib.sha256(wheel.read_bytes()).hexdigest()} sdist_sha256={hashlib.sha256(sdists[0].read_bytes()).hexdigest()}"
    )
    venv = root / "venv"
    subprocess.run(
        [
            uv,
            "venv",
            "--offline",
            "--no-project",
            "--python",
            sys.executable,
            str(venv),
        ],
        cwd=root,
        check=True,
        timeout=60,
    )
    python = venv / "bin" / "python"
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--offline",
            "--no-index",
            "--python",
            str(python),
            str(wheel),
        ],
        cwd=root,
        check=True,
        timeout=120,
    )
    subprocess.run(
        [uv, "pip", "check", "--offline", "--python", str(python)],
        cwd=root,
        check=True,
        timeout=60,
    )
    minimal = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            """
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys

# Match packaging.markers.default_environment using only the target's stdlib.
info = sys.implementation.version
implementation_version = f'{info.major}.{info.minor}.{info.micro}'
if info.releaselevel != 'final':
    implementation_version += info.releaselevel[0] + str(info.serial)
environment = {
    'implementation_name': sys.implementation.name,
    'implementation_version': implementation_version,
    'os_name': os.name,
    'platform_machine': platform.machine(),
    'platform_release': platform.release(),
    'platform_system': platform.system(),
    'platform_version': platform.version(),
    'python_full_version': platform.python_version(),
    'platform_python_implementation': platform.python_implementation(),
    'python_version': '.'.join(platform.python_version_tuple()[:2]),
    'sys_platform': sys.platform,
    'extra': '',
}
installed = []
for dist in metadata.distributions():
    origin = Path(dist._path).resolve()
    assert origin.is_relative_to(Path(sys.prefix).resolve()), (origin, sys.prefix)
    installed.append({
        'name': dist.metadata['Name'],
        'version': dist.version,
        'requires': dist.requires or [],
    })
print(json.dumps({'environment': environment, 'installed': installed}))
""",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert minimal.returncode == 0, (
        f"isolated installed metadata probe failed ({minimal.returncode}):\n"
        f"{minimal.stdout}{minimal.stderr}"
    )
    probe = json.loads(minimal.stdout)
    # Requirement parsing belongs to the dev environment, not the minimal install.
    installed = {canonicalize_name(dist["name"]): dist for dist in probe["installed"]}
    reachable: set[NormalizedName] = set()
    pending = [canonicalize_name("chartreux")]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        dist = installed[name]
        for raw in dist["requires"]:
            req = Requirement(raw)
            if req.marker is None or req.marker.evaluate(probe["environment"]):
                dependency = canonicalize_name(req.name)
                assert installed[dependency]["version"] in req.specifier, raw
                pending.append(dependency)
    assert reachable == set(installed), sorted(set(installed) - reachable)
    assert (
        not {
            "opentelemetry-sdk",
            "rfc8785",
            "websockets",
            "zstandard",
            "twine",
            "pytest",
            "hatchling",
        }
        & reachable
    )
    assert {"opentelemetry-api", "opentelemetry-semantic-conventions"} <= reachable
    dependencies = json.dumps(
        {name: dist["version"] for name, dist in installed.items()}, sort_keys=True
    )
    (root / "installed-dependencies.json").write_text(dependencies + "\n")
    print(f"MINIMAL_INSTALLED_DEPENDENCIES={dependencies}")
    return root, wheel


def _record_evidence(payload: dict[str, Any]) -> None:
    destination = Path("/test-output") / "installed-contracts-evidence.json"
    if not destination.parent.is_dir():
        destination = (
            Path(payload["wheel"]).parent / "installed-contracts-evidence.json"
        )
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"INSTALLED_CONTRACTS_EVIDENCE={destination}")


def _child_env(root: Path, helper: Path, *, home: Path) -> dict[str, str]:
    env = {
        "PATH": f"{root / 'venv' / 'bin'}:{os.defpath}",
        "HOME": str(home),
        "CHARTREUX_HOME": str(home / ".chartreux"),
        "XDG_CONFIG_HOME": str(home / "config"),
        "XDG_CACHE_HOME": str(home / "cache"),
        "XDG_DATA_HOME": str(home / "data"),
        "XDG_STATE_HOME": str(home / "state"),
        "PYTHONPATH": str(helper),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "LANG": "C.UTF-8",
        "TERM": "dumb",
    }
    return env


def _write_network_guard(
    helper: Path, guard: Path, *, allowed_port: int | None = None
) -> None:
    helper.mkdir(parents=True, exist_ok=True)
    helper.joinpath("sitecustomize.py").write_text(
        f"""
from __future__ import annotations

import os
from pathlib import Path
import sys

_guard = Path(os.environ["INSTALLED_NETWORK_GUARD"])
_allowed_port = {allowed_port!r}
_allowed_hosts = {{"127.0.0.1", "localhost", "::1"}}
_legacy_audit = os.environ.get("INSTALLED_LEGACY_READ_AUDIT")
_legacy_roots = tuple(
    Path(raw).resolve()
    for raw in os.environ.get("INSTALLED_LEGACY_ROOTS", "").split(os.pathsep)
    if raw
)


def _deny(event: str, address: object) -> None:
    with _guard.open("a", encoding="utf-8") as stream:
        stream.write(f"{{event}}:{{address!r}}\\n")
    raise OSError(f"non-fake network denied: {{address!r}}")


def _deny_legacy_read(event: str, raw_path: object) -> None:
    if _legacy_audit is not None:
        with open(_legacy_audit, "a", encoding="utf-8") as stream:
            stream.write(f"{{event}}:{{raw_path!r}}\\n")
    raise PermissionError(f"legacy read denied: {{raw_path!r}}")


def _is_allowed(host: object, port: object) -> bool:
    if host not in _allowed_hosts or _allowed_port is None:
        return False
    try:
        return int(port) == _allowed_port
    except (TypeError, ValueError):
        return False


def _is_legacy_path(raw_path: object) -> bool:
    if not _legacy_roots or not isinstance(raw_path, (str, bytes, os.PathLike)):
        return False
    try:
        candidate = Path(os.fsdecode(raw_path)).resolve()
    except (OSError, ValueError):
        return False
    return any(candidate == root or root in candidate.parents for root in _legacy_roots)


def _audit(event: str, args: tuple[object, ...]) -> None:
    if event == "socket.getaddrinfo" and len(args) > 1:
        if not _is_allowed(args[0], args[1]):
            _deny(event, (args[0], args[1]))
    elif event == "socket.connect" and len(args) > 1:
        address = args[1]
        if not isinstance(address, tuple) or len(address) < 2:
            _deny(event, address)
        if not _is_allowed(address[0], address[1]):
            _deny(event, address)
    elif event in {{"open", "os.access"}} and args and _is_legacy_path(args[0]):
        _deny_legacy_read(event, args[0])


sys.addaudithook(_audit)


_original_access = os.access


def _guarded_access(
    path: object,
    mode: int,
    dir_fd: int | None = None,
    effective_ids: bool = False,
    follow_symlinks: bool = True,
) -> bool:
    if _is_legacy_path(path):
        _deny_legacy_read("os.access", path)
    return _original_access(
        path,
        mode,
        dir_fd=dir_fd,
        effective_ids=effective_ids,
        follow_symlinks=follow_symlinks,
    )


os.access = _guarded_access
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_legacy_keyring_capture(helper: Path) -> None:
    helper.joinpath("installed_legacy_keyring_capture.py").write_text(
        """
from __future__ import annotations

import json
import os

from keyring.backend import KeyringBackend


class CaptureKeyring(KeyringBackend):
    priority = 1

    def _record(self, operation: str, service: str, username: str) -> None:
        with open(os.environ["INSTALLED_KEYRING_ATTEMPTS"], "a", encoding="utf-8") as stream:
            json.dump({"operation": operation, "service": service, "username": username}, stream)
            stream.write("\\n")

    def get_password(self, service: str, username: str) -> str | None:
        self._record("get", service, username)
        if service == "chartreux":
            return "synthetic-provider-credential"
        return None

    def set_password(self, service: str, username: str, password: str) -> None:
        del password
        self._record("set", service, username)

    def delete_password(self, service: str, username: str) -> None:
        self._record("delete", service, username)
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _read_audit_records(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _read_keyring_attempts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _disk_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    manifest: dict[str, dict[str, int | str]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        manifest[relative] = {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return manifest


def _assert_network_guard_clean(guard: Path) -> None:
    assert not guard.exists(), guard.read_text(encoding="utf-8")


def _assert_no_secret_output(guard: Path, *outputs: str | bytes) -> None:
    text = "".join(
        output.decode(errors="replace") if isinstance(output, bytes) else output
        for output in outputs
    )
    assert "synthetic-provider-credential" not in text
    _assert_network_guard_clean(guard)


def _assert_no_installed_secret_output(
    guard: Path, secrets: tuple[str, ...], *outputs: str | bytes
) -> None:
    _assert_no_secret_output(guard, *outputs)
    text = "".join(
        output.decode(errors="replace") if isinstance(output, bytes) else output
        for output in outputs
    )
    for secret in secrets:
        assert secret not in text


def _write_fake_config(home: Path, api_base: str, *, session_dir: Path) -> None:
    config = home / ".chartreux"
    config.mkdir(parents=True, exist_ok=True)
    (config / "config.toml").write_text(
        "\n".join([
            'active_model = "installed-fake-model"',
            "disable_welcome_banner_animation = true",
            "",
            "[session_logging]",
            "enabled = true",
            f'save_dir = "{session_dir}"',
        ])
        + "\n",
        encoding="utf-8",
    )
    (config / "models.toml").write_text(
        "\n".join([
            '[providers."installed-fake/default"]',
            f'api_base = "{api_base}"',
            'api_key_env_var = "FAKE_PROVIDER_KEY"',
            'backend = "generic"',
            "",
            '[models."installed-fake-model"]',
            'thinking = "off"',
            "",
            '[[models."installed-fake-model".deployments]]',
            'provider = "installed-fake/default"',
            'name = "installed-fake-model"',
        ])
        + "\n",
        encoding="utf-8",
    )


async def _read_json_rpc_response(
    reader: asyncio.StreamReader, expected_id: int, *, timeout: float = 30
) -> dict[str, Any]:
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        assert line, "installed protocol process closed before its response"
        message = json.loads(line)
        assert isinstance(message, dict), message
        assert message.get("jsonrpc") == "2.0", message
        if message.get("id") == expected_id:
            return message


def _run(
    executable: Path,
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


async def _drain_stderr(process: asyncio.subprocess.Process) -> bytes:
    assert process.stderr is not None
    return await process.stderr.read()


async def _close_protocol_process(
    process: asyncio.subprocess.Process, stderr_task: asyncio.Task[bytes]
) -> tuple[bytes, bytes]:
    if process.stdin is not None and not process.stdin.is_closing():
        process.stdin.close()
    try:
        await asyncio.wait_for(process.wait(), timeout=20)
    except TimeoutError:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=10)
    stdout = b""
    if process.stdout is not None:
        stdout = await asyncio.wait_for(process.stdout.read(), timeout=10)
    stderr = await asyncio.wait_for(stderr_task, timeout=10)
    return stdout, stderr


async def _read_until(
    reader: asyncio.StreamReader,
    *,
    response_id: int | None = None,
    terminal: Any = None,
    timeout: float = 30,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    response = None
    messages: list[dict[str, Any]] = []
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        assert line, "installed protocol process closed before terminal output"
        message = json.loads(line)
        assert isinstance(message, dict), message
        assert message.get("jsonrpc") == "2.0", message
        messages.append(message)
        if response_id is not None and message.get("id") == response_id:
            response = message
        if terminal is not None and terminal(message):
            return response, messages
        if response_id is not None and response is not None and terminal is None:
            return response, messages


def _validate_json_lines(output: bytes) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for line in output.splitlines():
        message = json.loads(line)
        assert isinstance(message, dict), message
        assert message.get("jsonrpc") == "2.0", message
        messages.append(message)
    return messages


async def _run_app_server_startup(
    root: Path, project: Path, env: dict[str, str]
) -> tuple[bytes, bytes, list[dict[str, Any]]]:
    process = await asyncio.create_subprocess_exec(
        str(root / "venv" / "bin" / "chartreux-app-server"),
        cwd=project,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=4 * 1024 * 1024,
    )
    assert process.stdin is not None and process.stdout is not None
    stderr_task = asyncio.create_task(_drain_stderr(process))
    messages: list[dict[str, Any]] = []
    stdout = b""
    stderr = b""
    try:
        process.stdin.write(
            (
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "clientInfo": {
                            "name": "installed-legacy-contract",
                            "version": "0",
                        },
                        "capabilities": {},
                    },
                })
                + "\n"
            ).encode()
        )
        await process.stdin.drain()
        initialize, received = await _read_until(
            process.stdout, response_id=1, timeout=30
        )
        messages.extend(received)
        assert initialize is not None
        assert initialize["result"]["serverInfo"]["name"] == "chartreux-app-server"
    finally:
        stdout, stderr = await _close_protocol_process(process, stderr_task)
    assert process.returncode == 0, stderr.decode(errors="replace")
    _validate_json_lines(stdout)
    return stdout, stderr, messages


@pytest.mark.timeout(240)
def test_exact_wheel_has_installed_identity_and_all_three_entrypoints(
    installed_bundle: tuple[Path, Path],
) -> None:
    root, wheel = installed_bundle
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    python = root / "venv" / "bin" / "python"
    home = root / "identity-home"
    guard = root / "identity-network-guard"
    helper = root / "identity-helper"
    _write_network_guard(helper, guard)
    env = _child_env(root, helper, home=home)
    env["INSTALLED_NETWORK_GUARD"] = str(guard)
    env["CHARTREUX_SOURCE_ROOT"] = str(_PROJECT_ROOT)
    env["FAKE_PROVIDER_KEY"] = "synthetic-provider-credential"
    identity = _run(
        python,
        [
            "-c",
            """
import importlib
import importlib.metadata as metadata
import importlib.util
import os
from pathlib import Path
import sys
import chartreux

source_root = Path(os.environ["CHARTREUX_SOURCE_ROOT"]).resolve()
dist = metadata.distribution("chartreux")
source = Path(chartreux.__file__).resolve()
prefix = Path(sys.prefix).resolve()
assert source.is_relative_to(prefix), (source, prefix)
assert Path.home() == Path(os.environ["HOME"])
assert os.environ["PYTHON_KEYRING_BACKEND"] == "keyring.backends.fail.Keyring"
assert "MISTRAL_API_KEY" not in os.environ
assert source.parent.name == "chartreux"
assert importlib.util.find_spec("vibe") is None
assert all(not str(file).replace("\\\\", "/").startswith("vibe/") for file in (dist.files or ()))
assert dist.metadata["Name"] == "chartreux"
assert chartreux.__version__ == "0.1.1"
assert dist.version == "0.1.1"
assert Path(dist._path).resolve().is_relative_to(prefix)
assert all(
    not Path(entry).resolve().is_relative_to(source_root)
    for entry in sys.path
    if entry
)
entrypoints = {
    entry.name: entry for entry in dist.entry_points if entry.group == "console_scripts"
}
expected = {
    "chartreux": "chartreux.cli.entrypoint",
    "chartreux-acp": "chartreux.acp.entrypoint",
    "chartreux-app-server": "chartreux.app_server.stdio",
}
assert set(expected) <= set(entrypoints)
for name, module_name in expected.items():
    entry = entrypoints[name]
    assert entry.value.split(":", 1)[0] == module_name
    module = importlib.import_module(module_name)
    origin = Path(module.__file__).resolve()
    assert origin.is_relative_to(prefix), (name, origin, prefix)
    loaded = entry.load()
    assert loaded.__module__ == module_name
print({
    "distribution": dist.metadata["Name"],
    "version": dist.version,
    "source": str(source),
    "prefix": str(prefix),
    "metadata": str(dist._path),
    "entrypoints": {name: str(Path(importlib.import_module(module).__file__).resolve()) for name, module in expected.items()},
})
""",
        ],
        cwd=root,
        env=env,
    )
    assert identity.returncode == 0, identity.stderr
    assert str(_PROJECT_ROOT) not in identity.stdout
    assert str(_PROJECT_ROOT) not in identity.stderr

    cli = root / "venv" / "bin" / "chartreux"
    acp = root / "venv" / "bin" / "chartreux-acp"
    app_server = root / "venv" / "bin" / "chartreux-app-server"
    assert cli.is_file() and acp.is_file() and app_server.is_file()
    for script in (cli, acp, app_server):
        script_text = script.read_text(encoding="utf-8")
        first_line = script_text.splitlines()[0]
        if first_line == "#!/bin/sh":
            assert f"'exec' '{python}'" in script_text
        else:
            assert first_line == f"#!{python}"
    for executable in (cli, acp):
        version = _run(executable, ["--version"], cwd=root, env=env)
        assert version.returncode == 0, version.stderr
        assert version.stdout.strip() == f"{executable.name} 0.1.1"
        _assert_no_secret_output(guard, version.stdout, version.stderr)
    negative_guard = root / "identity-negative-network-guard"
    negative_helper = root / "identity-negative-helper"
    _write_network_guard(negative_helper, negative_guard)
    negative_env = dict(env)
    negative_env.update({
        "PYTHONPATH": str(negative_helper),
        "INSTALLED_NETWORK_GUARD": str(negative_guard),
    })
    negative = _run(
        python,
        [
            "-c",
            "import socket\ntry:\n socket.create_connection(('example.invalid', 443), timeout=0.2)\nexcept OSError:\n pass\nelse:\n raise AssertionError('audit guard did not deny network attempt')",
        ],
        cwd=root,
        env=negative_env,
        timeout=10,
    )
    assert negative.returncode == 0, negative.stderr
    assert negative_guard.read_text(encoding="utf-8").startswith("socket.")
    _record_evidence({
        "wheel": str(wheel),
        "wheel_sha256": wheel_sha256,
        "installed_prefix": str(root / "venv"),
        "chartreux": str(cli),
        "chartreux_acp": str(acp),
        "chartreux_app_server": str(app_server),
        "checkout": str(_PROJECT_ROOT),
    })


@pytest.mark.timeout(180)
def test_installed_programmatic_cli_turn_uses_only_fake_provider(
    installed_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    root, wheel = installed_bundle
    fake_provider = FakeInstalledProvider()
    fake_provider.start()
    try:
        home = tmp_path / "home"
        project = tmp_path / "project"
        project.mkdir()
        helper = tmp_path / "helper"
        guard = tmp_path / "network-guard"
        _write_network_guard(helper, guard, allowed_port=fake_provider.server_port)
        _write_fake_config(
            home, fake_provider.api_base, session_dir=tmp_path / "sessions"
        )
        env = _child_env(root, helper, home=home)
        env.update({
            "INSTALLED_NETWORK_GUARD": str(guard),
            "FAKE_PROVIDER_KEY": "synthetic-provider-credential",
        })
        result = _run(
            root / "venv" / "bin" / "chartreux",
            [
                "--prompt",
                "Respond with the installed fake answer",
                "--output",
                "text",
                "--max-turns",
                "1",
            ],
            cwd=project,
            env=env,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert "Hello from installed fake provider" in result.stdout
        assert fake_provider.request_count == 1
        _assert_no_secret_output(guard, result.stdout, result.stderr)
    finally:
        fake_provider.stop()


def _protocol_context(
    root: Path, tmp_path: Path, provider: FakeInstalledProvider
) -> tuple[Path, dict[str, str], Path]:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    helper = tmp_path / "helper"
    guard = tmp_path / "network-guard"
    _write_network_guard(helper, guard, allowed_port=provider.server_port)
    _write_fake_config(home, provider.api_base, session_dir=tmp_path / "sessions")
    env = _child_env(root, helper, home=home)
    env.update({
        "INSTALLED_NETWORK_GUARD": str(guard),
        "FAKE_PROVIDER_KEY": "synthetic-provider-credential",
    })
    return project, env, guard


async def _run_acp_lifecycle(
    root: Path,
    tmp_path: Path,
    provider: FakeInstalledProvider,
    *,
    blocked: bool,
    context: tuple[Path, dict[str, str], Path] | None = None,
    expected_request_count: int = 1,
    secrets: tuple[str, ...] = (),
) -> None:
    project, env, guard = context or _protocol_context(root, tmp_path, provider)
    process = await asyncio.create_subprocess_exec(
        str(root / "venv" / "bin" / "chartreux-acp"),
        cwd=project,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=4 * 1024 * 1024,
    )
    assert process.stdin is not None and process.stdout is not None
    captured_stdout = bytearray()
    protocol_stdout = _TeeStreamReader(process.stdout, captured_stdout)
    stderr_task = asyncio.create_task(_drain_stderr(process))
    connection = None
    prompt_task: asyncio.Task[Any] | None = None
    stdout = b""
    stderr = b""
    client = _AcpClient()
    try:
        connection = connect_to_agent(client, process.stdin, protocol_stdout)
        initialized = await asyncio.wait_for(
            connection.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
                client_info=Implementation(name="installed-contract", version="0"),
            ),
            timeout=20,
        )
        assert initialized.agent_info is not None
        assert initialized.agent_info.name == "chartreux"
        session = await asyncio.wait_for(
            connection.new_session(cwd=str(project), mcp_servers=[]), timeout=20
        )
        prompt_task = asyncio.create_task(
            connection.prompt(
                session_id=session.session_id,
                prompt=[TextContentBlock(type="text", text="installed ACP turn")],
            )
        )
        await asyncio.to_thread(provider.wait_for_request, expected_request_count, 30)
        if blocked:
            assert await asyncio.to_thread(provider.streaming_started.wait, 30)
            await connection.cancel(session.session_id)
            response = await asyncio.wait_for(prompt_task, timeout=30)
            assert response.stop_reason == "cancelled"
            provider.release_streaming.set()
        else:
            response = await asyncio.wait_for(prompt_task, timeout=30)
            assert response.stop_reason == "end_turn"
        provider.wait_for_idle()
        assert provider.request_count == expected_request_count
        assert provider.authorization_matches == [True] * expected_request_count
        if not blocked:
            assert any(
                "Hello from installed fake provider" in repr(update)
                for update in client.updates
            )
    finally:
        provider.release_streaming.set()
        if prompt_task is not None and not prompt_task.done():
            await asyncio.wait_for(prompt_task, timeout=20)
        if connection is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(connection.close(), timeout=20)
        stdout, stderr = await _close_protocol_process(process, stderr_task)
    assert process.returncode == 0, stderr.decode(errors="replace")
    full_stdout = bytes(captured_stdout) + stdout
    _validate_json_lines(full_stdout)
    _assert_no_installed_secret_output(
        guard, secrets, repr(client.updates), full_stdout, stderr
    )
    provider.wait_for_idle()
    assert provider.request_count == expected_request_count
    assert provider.authorization_matches == [True] * expected_request_count


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_installed_acp_successful_fake_answer_lifecycle(
    installed_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    root, _wheel = installed_bundle
    provider = FakeInstalledProvider()
    provider.start()
    try:
        await _run_acp_lifecycle(root, tmp_path, provider, blocked=False)
    finally:
        provider.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_installed_acp_blocked_streaming_cancel_lifecycle(
    installed_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    root, _wheel = installed_bundle
    provider = FakeInstalledProvider(block_streaming=True)
    provider.start()
    try:
        await _run_acp_lifecycle(root, tmp_path, provider, blocked=True)
    finally:
        provider.release_streaming.set()
        provider.stop()


async def _run_app_server_lifecycle(
    root: Path,
    tmp_path: Path,
    provider: FakeInstalledProvider,
    *,
    blocked: bool,
    context: tuple[Path, dict[str, str], Path] | None = None,
    expected_request_count: int = 1,
    secrets: tuple[str, ...] = (),
) -> None:
    project, env, guard = context or _protocol_context(root, tmp_path, provider)
    process = await asyncio.create_subprocess_exec(
        str(root / "venv" / "bin" / "chartreux-app-server"),
        cwd=project,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=4 * 1024 * 1024,
    )
    assert process.stdin is not None and process.stdout is not None
    stdin = process.stdin
    stderr_task = asyncio.create_task(_drain_stderr(process))
    stdout = b""
    stderr = b""
    messages: list[dict[str, Any]] = []
    try:

        def send(payload: dict[str, Any]) -> None:
            stdin.write((json.dumps(payload) + "\n").encode())

        send({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "installed-contract", "version": "0"},
                "capabilities": {},
            },
        })
        await stdin.drain()
        initialize, received = await _read_until(
            process.stdout, response_id=1, timeout=20
        )
        messages.extend(received)
        assert initialize is not None
        assert initialize["result"]["serverInfo"]["name"] == "chartreux-app-server"
        send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        send({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/start",
            "params": {
                "agentConfig": {
                    "cwd": str(project),
                    "autoApprove": True,
                    "trustWorkspace": True,
                    "headless": True,
                }
            },
        })
        await stdin.drain()
        session_response, received = await _read_until(
            process.stdout, response_id=2, timeout=30
        )
        messages.extend(received)
        assert session_response is not None
        session_id = session_response["result"]["state"]["session"]["id"]
        send({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "turn/start",
            "params": {
                "sessionId": session_id,
                "message": [{"type": "text", "text": "installed app-server turn"}],
            },
        })
        await stdin.drain()
        turn_response, received = await _read_until(
            process.stdout, response_id=3, timeout=30
        )
        messages.extend(received)
        assert turn_response is not None
        turn_id = turn_response["result"]["turn"]["id"]
        await asyncio.to_thread(provider.wait_for_request, expected_request_count, 30)
        if blocked:
            assert await asyncio.to_thread(provider.streaming_started.wait, 30)
            send({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "turn/interrupt",
                "params": {"sessionId": session_id, "expectedTurnId": turn_id},
            })
            await stdin.drain()
            interrupt_response, received = await _read_until(
                process.stdout,
                response_id=4,
                terminal=lambda message: (
                    message.get("method") == "turn/completed"
                    and message.get("params", {}).get("turn", {}).get("id") == turn_id
                    and message.get("params", {}).get("turn", {}).get("status")
                    == "interrupted"
                ),
                timeout=30,
            )
            messages.extend(received)
            assert interrupt_response is not None
            assert interrupt_response.get("result", {}).get("accepted") is True
            provider.release_streaming.set()
        else:
            _, received = await _read_until(
                process.stdout,
                terminal=lambda message: (
                    message.get("method") == "turn/completed"
                    and message.get("params", {}).get("turn", {}).get("id") == turn_id
                    and message.get("params", {}).get("turn", {}).get("status")
                    == "completed"
                ),
                timeout=30,
            )
            messages.extend(received)
        if not blocked:
            assistant_texts = [
                block["text"]
                for message in messages
                if message.get("method") == "history/entryAdded"
                for entry in [message.get("params", {}).get("entry", {})]
                if entry.get("type") == "message" and entry.get("role") == "assistant"
                for block in entry.get("content", [])
                if block.get("type") == "text"
            ]
            assert "Hello from installed fake provider" in assistant_texts
        provider.wait_for_idle()
        assert provider.request_count == expected_request_count
        assert provider.authorization_matches == [True] * expected_request_count
    finally:
        provider.release_streaming.set()
        stdout, stderr = await _close_protocol_process(process, stderr_task)
    assert process.returncode == 0, stderr.decode(errors="replace")
    _validate_json_lines(stdout)
    for message in messages:
        assert message.get("jsonrpc") == "2.0"
    _assert_no_installed_secret_output(
        guard, secrets, json.dumps(messages), stdout, stderr
    )
    provider.wait_for_idle()
    assert provider.request_count == expected_request_count
    assert provider.authorization_matches == [True] * expected_request_count


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_installed_app_server_successful_fake_answer_lifecycle(
    installed_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    root, _wheel = installed_bundle
    provider = FakeInstalledProvider()
    provider.start()
    try:
        await _run_app_server_lifecycle(root, tmp_path, provider, blocked=False)
    finally:
        provider.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_installed_app_server_blocked_streaming_cancel_lifecycle(
    installed_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    root, _wheel = installed_bundle
    provider = FakeInstalledProvider(block_streaming=True)
    provider.start()
    try:
        await _run_app_server_lifecycle(root, tmp_path, provider, blocked=True)
    finally:
        provider.release_streaming.set()
        provider.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_installed_entrypoints_ignore_legacy_vibe_home_and_keyring(
    installed_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    root, wheel = installed_bundle
    provider = FakeInstalledProvider()
    provider.start()
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _write_fake_config(home, provider.api_base, session_dir=tmp_path / "sessions")

    legacy_home = home / ".vibe"
    legacy_project = project / ".vibe"
    legacy_home.mkdir(parents=True)
    legacy_project.mkdir(parents=True)
    legacy_home.joinpath("config.toml").write_text(
        'api_key = "legacy-home-installed-sentinel"\n', encoding="utf-8"
    )
    legacy_home.joinpath("credentials.json").write_text(
        '{"token": "legacy-home-installed-secret"}\n', encoding="utf-8"
    )
    legacy_project.joinpath("config.toml").write_text(
        'api_key = "legacy-project-installed-sentinel"\n', encoding="utf-8"
    )
    legacy_project.joinpath("credentials.json").write_text(
        '{"token": "legacy-project-installed-secret"}\n', encoding="utf-8"
    )
    legacy_roots = (legacy_home, legacy_project)
    legacy_manifest_before = {
        str(path.relative_to(tmp_path)): _disk_manifest(path) for path in legacy_roots
    }

    helper = tmp_path / "legacy-helper"
    _write_network_guard(
        helper, tmp_path / "network-guard", allowed_port=provider.server_port
    )
    _write_legacy_keyring_capture(helper)
    python = root / "venv" / "bin" / "python"
    secrets = (
        "legacy-home-installed-sentinel",
        "legacy-home-installed-secret",
        "legacy-project-installed-sentinel",
        "legacy-project-installed-secret",
        "synthetic-provider-credential",
    )
    entrypoint_names = ("chartreux", "chartreux-acp", "chartreux-app-server")
    evidence: dict[str, Any] = {
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "legacy_roots": [str(path) for path in legacy_roots],
        "legacy_manifest_before": legacy_manifest_before,
        "entrypoints": {},
    }

    for request_number, name in enumerate(entrypoint_names, start=1):
        network_guard = tmp_path / f"{name}-network-guard"
        read_audit = tmp_path / f"{name}-legacy-read-audit"
        keyring_attempts = tmp_path / f"{name}-keyring-attempts.jsonl"
        env = _child_env(root, helper, home=home)
        env.update({
            "INSTALLED_NETWORK_GUARD": str(network_guard),
            "INSTALLED_LEGACY_ROOTS": os.pathsep.join(
                str(path) for path in legacy_roots
            ),
            "INSTALLED_LEGACY_READ_AUDIT": str(read_audit),
            "INSTALLED_KEYRING_ATTEMPTS": str(keyring_attempts),
            "PYTHON_KEYRING_BACKEND": (
                "installed_legacy_keyring_capture.CaptureKeyring"
            ),
        })
        if name == "chartreux":
            result = _run(
                root / "venv" / "bin" / name,
                [
                    "--prompt",
                    "Respond with the installed fake answer",
                    "--output",
                    "text",
                    "--max-turns",
                    "1",
                ],
                cwd=project,
                env=env,
                timeout=60,
            )
            assert result.returncode == 0, result.stderr
            assert "Hello from installed fake provider" in result.stdout
            stdout = result.stdout.encode()
            stderr = result.stderr.encode()
            public_payload = result.stdout
        elif name == "chartreux-acp":
            await _run_acp_lifecycle(
                root,
                tmp_path,
                provider,
                blocked=False,
                context=(project, env, network_guard),
                expected_request_count=request_number,
                secrets=secrets,
            )
            stdout = b""
            stderr = b""
            public_payload = "acp-successful-fake-provider-turn"
        else:
            await _run_app_server_lifecycle(
                root,
                tmp_path,
                provider,
                blocked=False,
                context=(project, env, network_guard),
                expected_request_count=request_number,
                secrets=secrets,
            )
            stdout = b""
            stderr = b""
            public_payload = "app-server-successful-fake-provider-turn"
        provider.wait_for_idle()
        assert provider.request_count == request_number
        assert provider.authorization_matches == [True] * request_number

        attempts = _read_keyring_attempts(keyring_attempts)
        audit_records = _read_audit_records(read_audit)
        assert not audit_records
        assert all(
            attempt["service"] not in {"ai.mistral.vibe", "vibe"}
            for attempt in attempts
        )
        _assert_no_installed_secret_output(
            network_guard, secrets, stdout, stderr, public_payload, json.dumps(attempts)
        )
        evidence["entrypoints"][name] = {
            "keyring_attempts": attempts,
            "legacy_read_audit": audit_records,
            "public_payload_sha256": hashlib.sha256(
                public_payload.encode()
            ).hexdigest(),
        }

    negative_env = _child_env(root, helper, home=home)
    negative_network_guard = tmp_path / "negative-network-guard"
    negative_read_audit = tmp_path / "negative-legacy-read-audit"
    negative_keyring_attempts = tmp_path / "negative-keyring-attempts.jsonl"
    negative_env.update({
        "INSTALLED_NETWORK_GUARD": str(negative_network_guard),
        "INSTALLED_LEGACY_ROOTS": os.pathsep.join(str(path) for path in legacy_roots),
        "INSTALLED_LEGACY_READ_AUDIT": str(negative_read_audit),
        "INSTALLED_KEYRING_ATTEMPTS": str(negative_keyring_attempts),
        "PYTHON_KEYRING_BACKEND": "installed_legacy_keyring_capture.CaptureKeyring",
        "INSTALLED_LEGACY_PROBE": str(legacy_home / "credentials.json"),
    })
    negative_read = _run(
        python,
        [
            "-c",
            """
import os
from pathlib import Path

path = Path(os.environ["INSTALLED_LEGACY_PROBE"])
for operation in ("open", "access"):
    try:
        if operation == "open":
            path.read_text(encoding="utf-8")
        else:
            os.access(path, os.R_OK)
    except PermissionError:
        pass
    else:
        raise AssertionError(f"legacy {operation} probe was not denied")
""",
        ],
        cwd=project,
        env=negative_env,
    )
    assert negative_read.returncode == 0, negative_read.stderr
    negative_audit_records = _read_audit_records(negative_read_audit)
    assert len(negative_audit_records) == 2
    assert negative_audit_records[0].startswith("open:")
    assert negative_audit_records[1].startswith("os.access:")

    negative_keyring = _run(
        python,
        [
            "-c",
            """
import keyring

for service in ("ai.mistral.vibe", "vibe"):
    assert keyring.get_password(service, "MISTRAL_API_KEY") is None
""",
        ],
        cwd=project,
        env=negative_env,
    )
    assert negative_keyring.returncode == 0, negative_keyring.stderr
    negative_attempts = _read_keyring_attempts(negative_keyring_attempts)
    assert negative_attempts == [
        {
            "operation": "get",
            "service": "ai.mistral.vibe",
            "username": "MISTRAL_API_KEY",
        },
        {"operation": "get", "service": "vibe", "username": "MISTRAL_API_KEY"},
    ]
    _assert_no_installed_secret_output(
        negative_network_guard,
        secrets,
        negative_read.stdout,
        negative_read.stderr,
        negative_keyring.stdout,
        negative_keyring.stderr,
    )

    legacy_manifest_after = {
        str(path.relative_to(tmp_path)): _disk_manifest(path) for path in legacy_roots
    }
    assert legacy_manifest_after == legacy_manifest_before
    assert not (home / ".chartreux" / ".vibe").exists()
    assert not (project / ".chartreux" / ".vibe").exists()
    evidence["legacy_manifest_after"] = legacy_manifest_after
    evidence["inverse_disk_artifacts"] = {
        "legacy_manifest_unchanged": legacy_manifest_after == legacy_manifest_before,
        "home_chartreux_vibe_created": (home / ".chartreux" / ".vibe").exists(),
        "project_chartreux_vibe_created": (project / ".chartreux" / ".vibe").exists(),
    }
    evidence["installed_sentinels"] = [
        "legacy-home-installed-sentinel",
        "legacy-project-installed-sentinel",
    ]
    evidence["negative_probes"] = {
        "read_audit": negative_audit_records,
        "keyring_attempts": negative_attempts,
    }
    assert provider.request_count == len(entrypoint_names)
    assert provider.authorization_matches == [True] * len(entrypoint_names)
    evidence["installed_final_assertions"] = {
        "all_entrypoints_real_fake_provider_turns": True,
        "provider_request_count": provider.request_count,
        "authorization_matches": provider.authorization_matches,
        "legacy_read_audits_empty": all(
            not _read_audit_records(tmp_path / f"{name}-legacy-read-audit")
            for name in entrypoint_names
        ),
        "legacy_keyring_services_absent": all(
            all(
                attempt["service"] not in {"ai.mistral.vibe", "vibe"}
                for attempt in _read_keyring_attempts(
                    tmp_path / f"{name}-keyring-attempts.jsonl"
                )
            )
            for name in entrypoint_names
        ),
    }
    _record_evidence(evidence)
    provider.stop()
