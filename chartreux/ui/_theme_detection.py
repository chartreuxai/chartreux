from __future__ import annotations

from functools import lru_cache
import os
import re
import select
import subprocess
import sys
import time

from chartreux.config_values import AUTO_THEME, DARK_THEME, FALLBACK_THEME, LIGHT_THEME

_OS_APPEARANCE_TIMEOUT_SECONDS = 1.0
_OSC11_QUERY = b"\x1b]11;?\x07"
_OSC11_RESPONSE_PREFIX = b"\x1b]11;"
_OSC11_TIMEOUT_SECONDS = 0.5
_OSC11_RESPONSE_RE = re.compile(
    rb"\x1b\]11;rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)"
)
_LUMINANCE_DARK_THRESHOLD = 0.5
_SRGB_LINEARIZE_CUTOFF = 0.03928


def detect_system_preferred_dark() -> bool | None:
    match sys.platform:
        case "darwin":
            return _detect_macos_dark()
        case "linux":
            return _detect_linux_dark()
        case _:
            return None


def _detect_macos_dark() -> bool | None:
    try:
        result = subprocess.run(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True,
            text=True,
            timeout=_OS_APPEARANCE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return False
    return "Dark" in result.stdout


def _detect_linux_dark() -> bool | None:
    try:
        result = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True,
            text=True,
            timeout=_OS_APPEARANCE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    match result.stdout.strip().strip("'"):
        case "prefer-dark":
            return True
        case "prefer-light":
            return False
        case _:
            return None


def detect_terminal_dark() -> bool | None:
    if _is_inside_multiplexer():
        return None
    return _detect_terminal_dark_posix()


def _is_inside_multiplexer() -> bool:
    return any(os.environ.get(name) for name in ("TMUX", "STY", "ZELLIJ"))


def _detect_terminal_dark_posix() -> bool | None:
    import termios
    import tty

    try:
        fd = os.open("/dev/tty", os.O_RDWR)
    except OSError:
        return None

    try:
        previous_attributes = termios.tcgetattr(fd)
    except termios.error:
        os.close(fd)
        return None

    try:
        # tty.setraw defaults to TCSAFLUSH, which discards queued user input.
        tty.setraw(fd, termios.TCSANOW)
        if select.select([fd], [], [], 0)[0]:
            return None
        os.write(fd, _OSC11_QUERY)
        response = _read_osc11_response(fd)
    except (OSError, termios.error):
        return None
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous_attributes)
        except termios.error:
            pass
        os.close(fd)

    return _classify_osc11_response(response)


def _read_osc11_response(fd: int) -> bytes:
    response = b""
    deadline = time.monotonic() + _OSC11_TIMEOUT_SECONDS
    while not _is_complete_osc11_response(response):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            break
        chunk = os.read(fd, 1)
        if not chunk:
            break
        response += chunk
    return response


def _is_complete_osc11_response(response: bytes) -> bool:
    start = response.find(_OSC11_RESPONSE_PREFIX)
    if start == -1:
        return False
    payload = response[start + len(_OSC11_RESPONSE_PREFIX) :]
    return b"\x07" in payload or b"\x1b\\" in payload


def _classify_osc11_response(response: bytes) -> bool | None:
    match = _OSC11_RESPONSE_RE.search(response)
    if not match:
        return None
    red, green, blue = (_parse_hex_component(raw) for raw in match.groups())
    if red is None or green is None or blue is None:
        return None
    return _relative_luminance(red, green, blue) < _LUMINANCE_DARK_THRESHOLD


def _parse_hex_component(raw: bytes) -> float | None:
    try:
        return int(raw, 16) / (16 ** len(raw) - 1)
    except (ValueError, ZeroDivisionError):
        return None


def _relative_luminance(red: float, green: float, blue: float) -> float:
    def linearize(component: float) -> float:
        if component <= _SRGB_LINEARIZE_CUTOFF:
            return component / 12.92
        return ((component + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * linearize(red) + 0.7152 * linearize(green) + 0.0722 * linearize(blue)
    )


@lru_cache(maxsize=1)
def resolve_auto_theme() -> str:
    """Detect once because probing blocks and temporarily owns terminal input."""
    dark = detect_terminal_dark()
    if dark is None:
        dark = detect_system_preferred_dark()
    if dark is None:
        return FALLBACK_THEME
    return DARK_THEME if dark else LIGHT_THEME


def resolve_theme(theme: str) -> str:
    return resolve_auto_theme() if theme == AUTO_THEME else theme
