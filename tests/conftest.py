from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator
from dataclasses import replace
import ipaddress
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from tempfile import TemporaryDirectory
import time
from typing import Any
import weakref

import keyring
from keyring.backend import KeyringBackend
import keyring.errors
import pytest
import tomli_w

from chartreux.cli.textual_ui.app import CORE_VERSION, ChartreuxApp, StartupOptions
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import (
    ChartreuxConfigSchema,
    ChartreuxConfigSchemaType,
    ModelConfig,
    ProviderConfig,
    SessionLoggingConfig,
    build_default_orchestrator,
)
from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    init_harness_files_manager,
    reset_harness_files_manager,
)
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.llm.types import BackendLike
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.utils.concurrency import run_sync
from chartreux.ui.theme import resolve_auto_theme
from chartreux.utils import keyring as keyring_utils
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator
from tests.stubs.fake_mcp_registry import FakeMCPRegistry

collect_ignore = ["perf"]

_TESTS_ROOT = Path(__file__).parent
_PROVIDER_CREDENTIALS = (
    "MISTRAL_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


@pytest.fixture(scope="session", autouse=True)
def _isolate_provider_transport() -> Generator[None]:
    """Remove real provider credentials and block non-loopback INET sockets."""
    guard_dir = Path(tempfile.mkdtemp(prefix="chartreux-provider-guard-"))
    guard = guard_dir / "sitecustomize.py"
    guard.write_text(
        "import ipaddress, socket\n"
        "_connect = socket.socket.connect\n"
        "_connect_ex = socket.socket.connect_ex\n"
        "_allowed = {'localhost', '127.0.0.1', '::1', '::', '0.0.0.0'}\n"
        "def _check(address):\n"
        "    if not isinstance(address, tuple): return False\n"
        "    if len(address) != 2: raise OSError('malformed socket address blocked by test isolation')\n"
        "    host = address[0]\n"
        "    if host in _allowed: return False\n"
        "    try: return not ipaddress.ip_address(host).is_loopback\n"
        "    except ValueError: return True\n"
        "def _blocked(self, address):\n"
        "    if _check(address): raise OSError('provider network blocked by test isolation')\n"
        "    return _connect(self, address)\n"
        "def _blocked_ex(self, address):\n"
        "    if _check(address): return 111\n"
        "    return _connect_ex(self, address)\n"
        "socket.socket.connect = _blocked\n"
        "socket.socket.connect_ex = _blocked_ex\n",
        encoding="utf-8",
    )
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    provider_env = {name: os.environ.get(name) for name in _PROVIDER_CREDENTIALS}
    original_pythonpath = os.environ.get("PYTHONPATH")
    for name in _PROVIDER_CREDENTIALS:
        os.environ.pop(name, None)

    def _is_blocked(address: Any) -> bool:
        if not isinstance(address, tuple):
            return False
        if len(address) != 2:
            raise OSError("malformed socket address blocked by test isolation")
        host = address[0]
        if host in {"localhost", "127.0.0.1", "::1", "::", "0.0.0.0"}:
            return False
        try:
            return not ipaddress.ip_address(host).is_loopback
        except ValueError:
            return True

    def _blocked(self: socket.socket, address: Any) -> Any:
        if _is_blocked(address):
            raise OSError("provider network blocked by test isolation")
        return original_connect(self, address)

    def _blocked_ex(self: socket.socket, address: Any) -> int:
        if _is_blocked(address):
            return 111
        return original_connect_ex(self, address)

    socket.socket.connect = _blocked
    socket.socket.connect_ex = _blocked_ex
    os.environ["PYTHONPATH"] = str(guard_dir) + os.pathsep + (original_pythonpath or "")
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        if original_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original_pythonpath
        for name, value in provider_env.items():
            if value is not None:
                os.environ[name] = value
        shutil.rmtree(guard_dir, ignore_errors=True)


_LOCAL_XDIST_GROUPS = {
    Path("core/test_history_properties.py"): "history_properties",
    Path("core/test_system_prompt.py"): "git_processes",
    Path("core/test_trusted_folders.py"): "git_processes",
    Path("core/test_worktree.py"): "git_processes",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    local_run = not os.environ.get("BUILDKITE") and not os.environ.get("GITHUB_ACTIONS")
    for item in items:
        try:
            relative_path = Path(item.path).relative_to(_TESTS_ROOT)
        except ValueError:
            continue
        if local_run and (group := _LOCAL_XDIST_GROUPS.get(relative_path)):
            item.add_marker(pytest.mark.xdist_group(name=group))
            continue
        if relative_path.parts[0] != "e2e":
            continue
        if relative_path.name == "test_mock_server.py":
            continue
        group = (
            "subprocess_characterization"
            if relative_path.parts[1] == "agent_loop_characterization"
            else "subprocess_cli"
        )
        item.add_marker(pytest.mark.xdist_group(name=group))


def pytest_addoption(parser: pytest.Parser) -> None:
    pass


class _EmptyKeyring(KeyringBackend):
    """A keyring backend that stores nothing, used to keep tests off the real OS keyring."""

    priority = 1  # type: ignore[assignment]

    def get_password(self, service: str, username: str) -> str | None:
        return None

    def set_password(self, service: str, username: str, password: str) -> None:
        return None

    def delete_password(self, service: str, username: str) -> None:
        raise keyring.errors.PasswordDeleteError()


@pytest.fixture(autouse=True)
def _isolate_git_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite off the developer's git config.

    Tests create throwaway repos and real commits; without this, settings from
    ~/.gitconfig leak in: commit.gpgsign=true makes every test commit ping the
    developer's signing agent (and fail where the agent can't prompt). Repos
    that need an identity set user.name/email themselves.

    Redirecting the global/system config is not enough on every git build, so we
    also inject commit.gpgsign=false through GIT_CONFIG_COUNT, which git applies
    after all config files and thus overrides any leaked signing setting.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "commit.gpgsign")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "false")


@pytest.fixture(autouse=True)
def _disable_os_keyring(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Keep the suite off the real OS keyring.

    ``mcp_oauth`` tests use the keyring helpers, so without this they could touch
    the real Keychain. We install an empty backend (rather than patching
    ``keyring.get_password``) so tests that swap in their own backend via
    ``keyring.set_keyring`` still work. Tests that exercise keyring behaviour opt in by
    patching ``keyring.get_password`` / ``set_password`` directly.
    """
    original = keyring.get_keyring()
    monkeypatch.setattr(keyring_utils, "_should_use_macos_security", lambda: False)
    keyring.set_keyring(_EmptyKeyring())
    try:
        yield
    finally:
        keyring.set_keyring(original)


def get_base_config() -> dict[str, Any]:
    return {}


@pytest.fixture(autouse=True)
def tmp_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    tmp_working_directory = tmp_path_factory.mktemp("test_cwd")
    monkeypatch.chdir(tmp_working_directory)
    return tmp_working_directory


@pytest.fixture(autouse=True)
def config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    tmp_path = tmp_path_factory.mktemp("vibe")
    config_dir = tmp_path / ".chartreux"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    config_file.write_text(tomli_w.dumps(get_base_config()), encoding="utf-8")

    monkeypatch.setattr("chartreux.utils.paths._DEFAULT_CHARTREUX_HOME", config_dir)
    monkeypatch.setenv("CHARTREUX_HOME", str(config_dir))
    agents_dir = tmp_path / ".agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "chartreux.core.paths._agents_home._DEFAULT_AGENTS_HOME", agents_dir
    )

    return config_dir


@pytest.fixture(autouse=True)
def _reset_trusted_folders_manager(config_dir: Path) -> None:
    """Prevent the singleton from writing to the real ~/.chartreux/trusted_folders.toml.

    The module-level ``trusted_folders_manager`` captures its file path at import
    time (before any monkeypatch), so it would otherwise target the real home
    directory.  Redirect it to the temp config dir used by the ``config_dir``
    fixture.
    """
    from chartreux.core.trusted_folders import trusted_folders_manager

    trusted_folders_manager._file_path = config_dir / "trusted_folders.toml"
    trusted_folders_manager._trusted = []
    trusted_folders_manager._untrusted = []
    trusted_folders_manager._session_trusted = []


@pytest.fixture(autouse=True)
def _init_harness_files_manager():
    reset_harness_files_manager()
    init_harness_files_manager("user", "project")
    yield
    reset_harness_files_manager()


@pytest.fixture(autouse=True)
def _scratchpad_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Generator[Path]:
    scratchpad_root = tmp_path_factory.mktemp("scratchpad")
    _counter = 0
    real_mkdtemp = tempfile.mkdtemp

    def _fake_mkdtemp(
        suffix: str | None = None, prefix: str | None = None, dir: str | None = None
    ) -> str:
        nonlocal _counter
        if not (prefix or "").startswith(("chartreux-scratchpad-", "vibe-scratchpad-")):
            return real_mkdtemp(suffix, prefix, dir)
        _counter += 1
        d = scratchpad_root / f"{prefix}{_counter}{suffix or ''}"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr("chartreux.core.scratchpad.tempfile.mkdtemp", _fake_mkdtemp)

    yield scratchpad_root


@pytest.fixture(autouse=True)
def _mock_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "mock")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock")
    monkeypatch.setenv("OPENAI_API_KEY", "mock")


@pytest.fixture(autouse=True)
def _mock_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock platform to be Linux with /bin/sh shell for consistent test behavior.

    This ensures that platform-specific system prompt generation is consistent
    across all tests regardless of the actual platform running the tests.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("SHELL", "/bin/sh")
    resolve_auto_theme.cache_clear()
    monkeypatch.setattr(
        "chartreux.ui._theme_detection.detect_terminal_dark", lambda: None
    )
    monkeypatch.setattr(
        "chartreux.ui._theme_detection.detect_system_preferred_dark", lambda: None
    )


@pytest.fixture(autouse=True)
def _disable_auto_title_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Auto-title generation fires a background model call after turns; off by
    # default so it never consumes mocked responses. Title tests re-patch this.
    async def _noop(messages: Any, *, config: Any, previous_title: Any = None) -> None:
        return None

    monkeypatch.setattr(
        "chartreux.core.session.title_model.generate_session_title", _noop
    )


@pytest.fixture(autouse=True)
def _disable_input_grace_periods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.widgets.question_app._INPUT_GRACE_PERIOD_S", 0
    )
    monkeypatch.setattr("chartreux.cli.textual_ui.app._DEFAULT_TYPING_DEBOUNCE_MS", 0)
    monkeypatch.delenv("CHARTREUX_TYPING_GRACE_PERIOD_MS", raising=False)


@pytest.fixture
def mock_prompts_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    project = tmp_path / "project" / ".chartreux" / "prompts"
    user = tmp_path / "home" / ".chartreux" / "prompts"
    project.mkdir(parents=True)
    user.mkdir(parents=True)

    class _MockManager(HarnessFilesManager):
        @property
        def project_prompts_dirs(self) -> list[Path]:
            # Project prompt directories are retired (package 16a): the
            # production adapter reports them through retired_project_prompt_dirs
            # and never consumes them, so the mock mirrors that. The project
            # directory stays in the fixture so tests can pin that a file
            # written there is never picked up.
            return []

        @property
        def user_prompts_dirs(self) -> list[Path]:
            return [user]

    monkeypatch.setattr(
        "chartreux.core.prompts.get_harness_files_manager",
        lambda: _MockManager(sources=("user",)),
    )
    return project, user


@pytest.fixture
def chartreux_app() -> ChartreuxApp:
    return build_test_chartreux_app()


@pytest.fixture
def agent_loop() -> AgentLoop:
    return build_test_agent_loop()


@pytest.fixture
def vibe_config() -> ChartreuxConfigSchema:
    return build_test_vibe_config()


@pytest.fixture(params=[ChartreuxConfigSchema], ids=["vibe_config_schema"])
def config_cls(request: pytest.FixtureRequest) -> ChartreuxConfigSchemaType:
    return request.param


@pytest.fixture
def make_config(
    config_cls: ChartreuxConfigSchemaType,
) -> Callable[..., ChartreuxConfigSchema]:
    def _make(**kwargs: Any) -> ChartreuxConfigSchema:
        return build_test_vibe_config(config_cls=config_cls, **kwargs)

    return _make


@pytest.fixture
def make_orchestrator() -> Callable[
    [], Awaitable[ConfigOrchestrator[ChartreuxConfigSchema]]
]:
    """Build the default config orchestrator lazily.

    The factory is async because the orchestrator builds its layer stack lazily;
    call it after seeding the on-disk config.
    """

    async def _make() -> ConfigOrchestrator[ChartreuxConfigSchema]:
        return await build_default_orchestrator()

    return _make


def make_test_models(auto_compact_threshold: int) -> list[ModelConfig]:
    definition = SHIPPED_CATALOG.models["glm-5-3"]
    deployment = definition.deployments[0]
    return [
        ModelConfig.model_validate({
            "name": deployment.name,
            "provider": deployment.provider,
            "alias": "glm-5-3",
            "thinking": definition.thinking,
            "temperature": (
                definition.temperature if definition.temperature is not None else 1.0
            ),
            "input_price": (
                deployment.prices.input if deployment.prices.input is not None else 0.0
            ),
            "output_price": (
                deployment.prices.output
                if deployment.prices.output is not None
                else 0.0
            ),
            "cached_input_price": deployment.prices.cached_input,
            "input_price_known": deployment.prices.input is not None,
            "output_price_known": deployment.prices.output is not None,
            "cached_input_price_known": deployment.prices.cached_input is not None,
            "supports_images": deployment.supports_images,
            "supported_thinking_levels": deployment.supported_thinking_levels,
            "auto_compact_threshold": auto_compact_threshold,
        })
    ]


def set_agent_config(agent: AgentLoop, config: ChartreuxConfigSchema) -> None:
    orchestrator = agent.config_orchestrator
    match orchestrator:
        case FakeConfigOrchestrator():
            orchestrator._config = config
        case ConfigOrchestrator():
            orchestrator._snapshot = replace(orchestrator._snapshot, config=config)
        case _:
            raise TypeError(f"unexpected orchestrator {orchestrator!r}")


def stub_config_reload(
    monkeypatch: pytest.MonkeyPatch, config: ChartreuxConfigSchema
) -> None:
    """Make orchestrator reloads resolve to ``config`` instead of reading disk."""

    async def _reload(
        self: Any,
        *,
        preflight: Callable[[ChartreuxConfigSchema], Awaitable[None]] | None = None,
        apply: Callable[[ChartreuxConfigSchema], None] | None = None,
    ) -> None:
        if preflight is not None:
            await preflight(config)
        if apply is not None:
            apply(config)
        if isinstance(self, FakeConfigOrchestrator):
            self._config = config
        else:
            self._snapshot = replace(self._snapshot, config=config)

    monkeypatch.setattr(ConfigOrchestrator, "reload", _reload)
    monkeypatch.setattr(FakeConfigOrchestrator, "reload", _reload)


def _prepare_test_config_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    session_logging = kwargs.pop("session_logging", None)
    kwargs["session_logging"] = (
        SessionLoggingConfig(enabled=False)
        if session_logging is None
        else session_logging
    )
    if models := kwargs.get("models"):
        if isinstance(models, dict):
            kwargs.setdefault("active_model", next(iter(models)))
        else:
            kwargs.setdefault("active_model", models[0].alias)
    # Keep the test prompt minimal: skip project-context discovery and prompt
    # detail unless a test opts in.
    kwargs.setdefault("include_project_context", False)
    kwargs.setdefault("include_prompt_detail", False)
    return kwargs


async def wait_until(
    pilot: Any, predicate: Callable[[], bool], timeout: float = 2.0
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.05)
    return predicate()


def _test_catalog_snapshot(kwargs: dict[str, Any]) -> CatalogSnapshot:
    """Materialize legacy test inputs as an in-memory catalog fixture."""
    providers = kwargs.pop("providers", None)
    models = kwargs.pop("models", None)
    providers = [
        provider
        if isinstance(provider, ProviderConfig)
        else ProviderConfig.model_validate(provider)
        for provider in providers or []
    ]
    model_values = models.values() if isinstance(models, dict) else models or []
    models = [
        model if isinstance(model, ModelConfig) else ModelConfig.model_validate(model)
        for model in model_values
    ]
    raw = SHIPPED_CATALOG.model_dump(mode="json")
    provider_ids: dict[str, str] = {}
    for provider in providers or []:
        provider_id = provider.name.strip()
        provider_ids[provider.name] = provider_id
        values = provider.model_dump(mode="json")
        values.pop("name")
        raw["providers"][provider_id] = values
    for model in models or []:
        provider_id = provider_ids.get(
            model.provider or "", model.provider or ""
        ).strip()
        values = model.model_dump(mode="json")
        raw["models"][model.alias] = {
            "thinking": values["thinking"],
            "temperature": values["temperature"],
            "deployments": [
                {
                    "provider": provider_id,
                    "name": values["name"],
                    "prices": {
                        "input": values["input_price"],
                        "output": values["output_price"],
                        "cached_input": values["cached_input_price"],
                    },
                    "supports_images": values["supports_images"],
                    "supported_thinking_levels": values["supported_thinking_levels"],
                    "auto_compact_threshold": values["auto_compact_threshold"],
                }
            ],
        }
    return CatalogSnapshot(ModelCatalog.model_validate(raw), "test-fixture")


def build_test_vibe_config(
    config_cls: ChartreuxConfigSchemaType = ChartreuxConfigSchema, **kwargs: Any
) -> ChartreuxConfigSchema:
    snapshot = _test_catalog_snapshot(kwargs)
    return config_cls(**_prepare_test_config_kwargs(kwargs)).attach_catalog_snapshot(
        snapshot
    )


def build_test_vibe_config_schema(**kwargs: Any) -> ChartreuxConfigSchema:
    return build_test_vibe_config(ChartreuxConfigSchema, **kwargs)


def multi_model_catalog() -> ModelCatalog:
    """A deterministic two-canonical-model catalog with the roster roles.

    The shipped dispatch slots bind ``@scout``/``@worker`` to one base model
    and ``@heavy`` to another, so the roster shape is multi-model and the
    orchestrated rendering keeps its tier routing.
    """
    return ModelCatalog.model_validate({
        "providers": {
            "alpha": {"api_base": "https://alpha.test"},
            "beta": {"api_base": "https://beta.test"},
        },
        "models": {
            "alpha-model": {
                "thinking": "medium",
                "deployments": [{"provider": "alpha", "name": "alpha-wire"}],
            },
            "beta-model": {
                "thinking": "high",
                "deployments": [{"provider": "beta", "name": "beta-wire"}],
            },
        },
        "roles": {
            "main": {
                "model": "beta-model",
                "thinking": "high",
                "description": "main assistant preset",
            },
            "scout": {
                "model": "alpha-model",
                "thinking": "low",
                "description": "scout preset",
            },
            "worker": {
                "model": "alpha-model",
                "thinking": "medium",
                "description": "worker preset",
            },
            "heavy": {
                "model": "beta-model",
                "thinking": "high",
                "description": "heavy preset",
            },
        },
    })


def single_model_catalog() -> ModelCatalog:
    """A deterministic one-canonical-model catalog with the roster roles.

    Every role the shipped policies reference binds the same canonical model,
    so the roster shape is single-model regardless of how the shipped catalog
    roster evolves. Kept named and isolated from ``SHIPPED_CATALOG`` so the
    single-model rendering fixtures stay stable.
    """
    return ModelCatalog.model_validate({
        "providers": {"solo": {"api_base": "https://solo.test"}},
        "models": {
            "solo-model": {
                "thinking": "medium",
                "deployments": [{"provider": "solo", "name": "solo-wire"}],
            }
        },
        "roles": {
            "main": {
                "model": "solo-model",
                "thinking": "high",
                "description": "main assistant preset",
            },
            "worker": {
                "model": "solo-model",
                "thinking": "medium",
                "description": "worker preset",
            },
            "scout": {
                "model": "solo-model",
                "thinking": "low",
                "description": "scout preset",
            },
            "heavy": {
                "model": "solo-model",
                "thinking": "high",
                "description": "heavy preset",
            },
        },
    })


def cost_tier_catalog() -> ModelCatalog:
    """The P1-c1 catalog fixture: prices pin the capability criterion.

    Three models on one provider: ``budget-model`` is the cheapest priced
    model, ``premium-model`` is priced higher, and ``mystery-model`` carries
    no price data. The shipped cost-tier route defaults (``implementation``,
    ``mechanical-edit``, ``search``, ``verification``) must resolve to
    ``budget-model`` — the fixture names the concrete per-tier default, so
    the route tests stay environment-independent (no live catalog, no user
    configuration), and the unpriced model is never selected.
    """
    return ModelCatalog.model_validate({
        "providers": {"alpha": {"api_base": "https://alpha.test"}},
        "models": {
            "budget-model": {
                "thinking": "medium",
                "deployments": [
                    {
                        "provider": "alpha",
                        "name": "budget-wire",
                        "prices": {"input": 0.3, "output": 1.2},
                    }
                ],
            },
            "premium-model": {
                "thinking": "high",
                "deployments": [
                    {
                        "provider": "alpha",
                        "name": "premium-wire",
                        "prices": {"input": 3.0, "output": 9.0},
                    }
                ],
            },
            "mystery-model": {
                "thinking": "medium",
                "deployments": [{"provider": "alpha", "name": "mystery-wire"}],
            },
        },
    })


type ConfigBuilder = Callable[..., ChartreuxConfigSchema]


@pytest.fixture(params=[build_test_vibe_config_schema], ids=["vibe_config_schema"])
def build_config(request: pytest.FixtureRequest) -> ConfigBuilder:
    return request.param


type ConfigLoader = Callable[[], ChartreuxConfigSchema]


def _load_vibe_config_schema() -> ChartreuxConfigSchema:
    return run_sync(build_default_orchestrator()).config


@pytest.fixture(params=[_load_vibe_config_schema], ids=["vibe_config_schema"])
def load_config(request: pytest.FixtureRequest) -> ConfigLoader:
    """Loader that reads the persisted config through the orchestrator layer stack."""
    return request.param


type OrchestratorLoader[C: ChartreuxConfigSchema] = Callable[[C], ConfigOrchestrator[C]]


async def _load_orchestrator(
    config: ChartreuxConfigSchema,
) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    return FakeConfigOrchestrator(config)


@pytest.fixture
def load_orchestrator() -> OrchestratorLoader[ChartreuxConfigSchema]:
    """Wraps a config into the default orchestrator by injecting its set fields
    into the highest-priority OverridesLayer, exposed via ConfigOrchestrator.
    """

    def load(
        config: ChartreuxConfigSchema,
    ) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        return run_sync(_load_orchestrator(config))

    return load


def build_test_agent_loop(
    *,
    config: ChartreuxConfigSchema | None = None,
    backend: BackendLike | None = None,
    enable_streaming: bool = False,
    **kwargs,
) -> AgentLoop:

    resolved_config = config or build_test_vibe_config()
    orchestrator = run_sync(_load_orchestrator(resolved_config))
    return AgentLoop(
        config_orchestrator=orchestrator,
        backend=backend or FakeBackend(),
        enable_streaming=enable_streaming,
        mcp_registry=kwargs.pop("mcp_registry", FakeMCPRegistry()),
        **kwargs,
    )


def make_test_history_file() -> tuple[Path, TemporaryDirectory[str]]:
    history_dir = TemporaryDirectory(prefix="chartreux-test-history-")
    return Path(history_dir.name) / "history.jsonl", history_dir


def build_test_chartreux_app(
    *,
    config: ChartreuxConfigSchema | None = None,
    agent_loop: AgentLoop | None = None,
    **kwargs,
) -> ChartreuxApp:
    app_config = config or build_test_vibe_config()

    resolved_agent_loop = agent_loop or build_test_agent_loop(config=app_config)

    kwargs.pop("update_notifier", None)
    kwargs.pop("update_cache_repository", None)
    current_version = kwargs.pop("current_version", None)
    resolved_current_version = (
        CORE_VERSION if current_version is None else current_version
    )
    app_server = kwargs.pop("app_server", None)
    app_server_source = (
        app_server
        if app_server is not None
        else lambda: create_test_app_server_session(resolved_agent_loop)
    )
    history_file = kwargs.pop("history_file", None)
    history_dir = None
    if history_file is None:
        history_file, history_dir = make_test_history_file()
    startup = kwargs.pop("startup", None) or StartupOptions(
        initial_prompt=kwargs.pop("initial_prompt", None)
    )

    app = ChartreuxApp(
        app_server=app_server_source,
        history_file=history_file,
        startup=startup,
        current_version=resolved_current_version,
        **kwargs,
    )
    if history_dir is not None:
        weakref.finalize(app, history_dir.cleanup)
    return app
