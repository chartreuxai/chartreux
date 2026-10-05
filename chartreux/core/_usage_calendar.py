from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from typing import Literal

from dateutil import tz


def utc_now() -> datetime:
    return datetime.now(UTC)


def resolve_local_timezone() -> tzinfo:
    """Resolve local rules afresh, including changes to TZ or /etc/localtime."""
    zone = tz.gettz.nocache(None)
    if zone is None:
        raise ValueError("Cannot resolve the system-local timezone")
    return zone


def _utc(instant: datetime) -> datetime:
    if instant.utcoffset() is None:
        raise ValueError("Usage calendar instants must be timezone-aware")
    return instant.astimezone(UTC)


def _zone_name(zone: tzinfo) -> str:
    if key := getattr(zone, "key", None):
        return str(key)
    # dateutil tzfile exposes the resolved file rather than an IANA key.
    if filename := getattr(zone, "_filename", None):
        return str(filename).split("/zoneinfo/")[-1]
    return str(zone)


@dataclass(frozen=True, slots=True)
class TimezoneFingerprint:
    """Opaque equality token, not a serialized identifier or a fixed UTC offset.

    dateutil tzfile equality compares transition rules; a freshly resolved zone
    therefore detects rule changes as well as zone-name changes. Keep this token
    in memory and compare it for equality, rather than relying on its hash.
    """

    name: str
    zone: tzinfo


def timezone_fingerprint(zone: tzinfo) -> TimezoneFingerprint:
    return TimezoneFingerprint(name=_zone_name(zone), zone=zone)


@dataclass(frozen=True, slots=True)
class CalendarWindow:
    kind: Literal["day", "week", "month"]
    start_local: datetime
    end_local: datetime
    start_utc: datetime
    end_utc: datetime

    def contains(self, instant: datetime) -> bool:
        """Attribute an instant using half-open UTC comparisons (including folds)."""
        return self.start_utc <= _utc(instant) < self.end_utc


@dataclass(frozen=True, slots=True)
class CalendarWindows:
    as_of: datetime
    zone_name: str
    timezone_fingerprint: TimezoneFingerprint
    day: CalendarWindow
    week: CalendarWindow
    month: CalendarWindow


def _midnight(day: date, zone: tzinfo) -> datetime:
    # Resolve each date independently, never carry the current instant's offset.
    # At an ambiguous midnight the first occurrence starts the calendar date;
    # at a skipped midnight advance through the gap to the first valid time.
    boundary = datetime.combine(day, time.min, tzinfo=zone).replace(fold=0)
    return tz.resolve_imaginary(boundary)


def _window(
    kind: Literal["day", "week", "month"], start: date, end: date, zone: tzinfo
) -> CalendarWindow:
    start_local, end_local = _midnight(start, zone), _midnight(end, zone)
    return CalendarWindow(
        kind=kind,
        start_local=start_local,
        end_local=end_local,
        start_utc=_utc(start_local),
        end_utc=_utc(end_local),
    )


def calendar_windows(
    as_of: datetime | None = None,
    timezone: tzinfo | str | None = None,
    *,
    clock: Callable[[], datetime] = utc_now,
    timezone_resolver: Callable[[], tzinfo] = resolve_local_timezone,
) -> CalendarWindows:
    """Rebuild day, Monday-start week and month from one shared aware instant.

    Explicit zones may be dateutil/zoneinfo objects or IANA names. Omitted zones
    use the injectable local resolver; omitted instants call the clock once.
    The result is an immutable snapshot with UTC storage/comparison boundaries
    and local boundaries for display. No ledger or cache state is accessed.
    """
    instant = _utc(as_of if as_of is not None else clock())
    zone = timezone_resolver() if timezone is None else timezone
    if isinstance(zone, str):
        resolved = tz.gettz.nocache(zone)
        if resolved is None:
            raise ValueError(f"Unknown timezone: {zone}")
        zone = resolved
    today = instant.astimezone(zone).date()
    monday = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    month_end = (month_start + timedelta(days=31)).replace(day=1)
    fingerprint = timezone_fingerprint(zone)
    return CalendarWindows(
        as_of=instant,
        zone_name=fingerprint.name,
        timezone_fingerprint=fingerprint,
        day=_window("day", today, today + timedelta(days=1), zone),
        week=_window("week", monday, monday + timedelta(days=7), zone),
        month=_window("month", month_start, month_end, zone),
    )
