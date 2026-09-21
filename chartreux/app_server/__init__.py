from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chartreux.app_server.client import AppServerConnectionClosed
    from chartreux.app_server.client_tools import ClientToolHandler
    from chartreux.app_server.host import AppServerHost
    from chartreux.app_server.session import AppServerSession, SessionExitSummary

__all__ = [
    "AppServerConnectionClosed",
    "AppServerHost",
    "AppServerSession",
    "ClientToolHandler",
    "SessionExitSummary",
]

_MAPPING: dict[str, tuple[str, str]] = {
    "AppServerConnectionClosed": (
        "chartreux.app_server.client",
        "AppServerConnectionClosed",
    ),
    "AppServerHost": ("chartreux.app_server.host", "AppServerHost"),
    "AppServerSession": ("chartreux.app_server.session", "AppServerSession"),
    "ClientToolHandler": ("chartreux.app_server.client_tools", "ClientToolHandler"),
    "SessionExitSummary": ("chartreux.app_server.session", "SessionExitSummary"),
}


def __getattr__(name: str) -> object:
    if name in _MAPPING:
        import importlib

        module_name, attr_name = _MAPPING[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
