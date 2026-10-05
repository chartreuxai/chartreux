from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from dateutil import tz
import pytest

from chartreux.core._usage_calendar import calendar_windows, timezone_fingerprint


def instant(value: str) -> datetime:
    return datetime.fromisoformat(value)


@pytest.mark.parametrize("kind", ["day", "week", "month"])
def test_exact_half_open_boundaries(kind):
    window = getattr(calendar_windows(instant("2024-02-29T12:00:00Z"), UTC), kind)
    assert window.contains(window.start_utc)
    assert window.contains(window.end_utc - timedelta(microseconds=1))
    assert not window.contains(window.start_utc - timedelta(microseconds=1))
    assert not window.contains(window.end_utc)
    rebuilt = getattr(calendar_windows(window.end_utc, UTC), kind)
    assert rebuilt.start_utc == window.end_utc
    assert rebuilt.contains(window.end_utc)


@pytest.mark.parametrize(
    ("as_of", "monday", "next_monday"),
    [
        ("2024-01-07T23:59:59Z", "2024-01-01", "2024-01-08"),
        ("2024-01-08T00:00:00Z", "2024-01-08", "2024-01-15"),
        ("2024-01-01T00:00:00Z", "2024-01-01", "2024-01-08"),
    ],
)
def test_monday_start_weeks(as_of, monday, next_monday):
    week = calendar_windows(instant(as_of), UTC).week
    assert week.start_local.date() == date.fromisoformat(monday)
    assert week.end_local.date() == date.fromisoformat(next_monday)


@pytest.mark.parametrize(
    ("as_of", "start", "end", "days"),
    [
        ("2023-02-28T23:59:59Z", "2023-02-01", "2023-03-01", 28),
        ("2024-02-29T23:59:59Z", "2024-02-01", "2024-03-01", 29),
        ("2000-02-29T12:00:00Z", "2000-02-01", "2000-03-01", 29),
        ("2100-02-28T12:00:00Z", "2100-02-01", "2100-03-01", 28),
        ("2024-04-30T23:59:59Z", "2024-04-01", "2024-05-01", 30),
        ("2024-12-31T23:59:59Z", "2024-12-01", "2025-01-01", 31),
        ("2025-01-01T00:00:00Z", "2025-01-01", "2025-02-01", 31),
    ],
)
def test_month_and_year_boundaries(as_of, start, end, days):
    month = calendar_windows(instant(as_of), UTC).month
    assert month.start_local.date() == date.fromisoformat(start)
    assert month.end_local.date() == date.fromisoformat(end)
    assert month.end_utc - month.start_utc == timedelta(days=days)


@pytest.mark.parametrize("zone_factory", [ZoneInfo, tz.gettz.nocache])
@pytest.mark.parametrize(
    ("as_of", "start", "end", "hours"),
    [
        ("2024-03-10T16:00:00Z", "2024-03-10T05:00:00Z", "2024-03-11T04:00:00Z", 23),
        ("2024-11-03T17:00:00Z", "2024-11-03T04:00:00Z", "2024-11-04T05:00:00Z", 25),
    ],
)
def test_dst_day_uses_each_boundary_offset(zone_factory, as_of, start, end, hours):
    snapshot = calendar_windows(instant(as_of), zone_factory("America/New_York"))
    day = snapshot.day
    assert day.start_utc == instant(start)
    assert day.end_utc == instant(end)
    assert day.end_utc - day.start_utc == timedelta(hours=hours)
    assert day.start_local.hour == day.end_local.hour == 0
    assert day.start_local.utcoffset() != day.end_local.utcoffset()
    assert snapshot.zone_name == "America/New_York"


def test_week_and_month_span_dst_rules_not_current_offset():
    snapshot = calendar_windows(instant("2024-03-10T16:00:00Z"), "America/New_York")
    assert snapshot.week.start_utc == instant("2024-03-04T05:00:00Z")
    assert snapshot.week.end_utc == instant("2024-03-11T04:00:00Z")
    assert snapshot.month.start_utc == instant("2024-03-01T05:00:00Z")
    assert snapshot.month.end_utc == instant("2024-04-01T04:00:00Z")


def test_both_occurrences_of_fall_back_hour_are_included():
    zone = ZoneInfo("America/New_York")
    first = datetime(2024, 11, 3, 1, 30, tzinfo=zone, fold=0)
    second = first.replace(fold=1)
    snapshot = calendar_windows(second, zone)
    assert snapshot.as_of == instant("2024-11-03T06:30:00Z")
    assert snapshot.day.contains(first)
    assert snapshot.day.contains(second)


def test_non_hour_offset_and_local_date_attribution():
    snapshot = calendar_windows(instant("2024-12-31T20:00:00Z"), "Asia/Kathmandu")
    assert snapshot.day.start_local.date() == date(2025, 1, 1)
    assert snapshot.day.start_local.utcoffset() == timedelta(hours=5, minutes=45)
    assert snapshot.day.start_utc == instant("2024-12-31T18:15:00Z")
    assert snapshot.day.end_utc == instant("2025-01-01T18:15:00Z")
    assert snapshot.month.start_utc == snapshot.day.start_utc
    assert snapshot.month.end_utc == instant("2025-01-31T18:15:00Z")


@pytest.mark.parametrize("zone_factory", [ZoneInfo, tz.gettz.nocache])
def test_midnight_gap_and_fold(zone_factory):
    zone = zone_factory("America/Havana")
    spring = calendar_windows(instant("2024-03-10T12:00:00Z"), zone).day
    assert spring.start_local.hour == 1
    assert spring.start_utc == instant("2024-03-10T05:00:00Z")
    assert spring.end_utc == instant("2024-03-11T04:00:00Z")
    fall = calendar_windows(instant("2024-11-03T12:00:00Z"), zone).day
    assert fall.start_local.fold == 0
    assert fall.start_utc == instant("2024-11-03T04:00:00Z")
    assert fall.end_utc == instant("2024-11-04T05:00:00Z")


def test_injectable_clock_and_timezone_resolver():
    calls = []

    def clock():
        calls.append("clock")
        return instant("2024-01-07T20:00:00Z")

    def resolver():
        calls.append("timezone")
        return ZoneInfo("Asia/Kathmandu")

    snapshot = calendar_windows(clock=clock, timezone_resolver=resolver)
    assert calls == ["clock", "timezone"]
    assert snapshot.as_of == instant("2024-01-07T20:00:00Z")
    assert snapshot.week.start_local.date() == date(2024, 1, 8)

    calls.clear()
    calendar_windows(snapshot.as_of, UTC, clock=clock, timezone_resolver=resolver)
    assert not calls


def test_timezone_change_fingerprint_rebuilds_same_instant():
    zones = iter([ZoneInfo("UTC"), ZoneInfo("Asia/Kathmandu")])
    as_of = instant("2024-01-07T20:00:00Z")
    before = calendar_windows(as_of, timezone_resolver=lambda: next(zones))
    after = calendar_windows(as_of, timezone_resolver=lambda: next(zones))
    assert before.timezone_fingerprint != after.timezone_fingerprint
    assert before.week.start_local.date() == date(2024, 1, 1)
    assert after.week.start_local.date() == date(2024, 1, 8)
    assert before.as_of == after.as_of


def test_fingerprint_is_stable_across_dst_and_fresh_resolutions():
    before = calendar_windows(instant("2024-03-09T12:00:00Z"), "America/New_York")
    after = calendar_windows(instant("2024-03-10T12:00:00Z"), "America/New_York")
    assert before.timezone_fingerprint == after.timezone_fingerprint
    assert timezone_fingerprint(ZoneInfo("UTC")) != timezone_fingerprint(
        ZoneInfo("Africa/Abidjan")
    )


def test_default_resolver_observes_tz_change(monkeypatch):
    as_of = instant("2024-01-07T20:00:00Z")
    monkeypatch.setenv("TZ", "America/New_York")
    before = calendar_windows(as_of)
    monkeypatch.setenv("TZ", "Asia/Kathmandu")
    after = calendar_windows(as_of)
    assert before.zone_name == "America/New_York"
    assert after.zone_name == "Asia/Kathmandu"
    assert before.timezone_fingerprint != after.timezone_fingerprint


def test_invalid_zone_and_naive_instants_are_rejected():
    with pytest.raises(ValueError, match="Unknown timezone"):
        calendar_windows(instant("2024-01-01T00:00:00Z"), "Not/AZone")
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar_windows(datetime(2024, 1, 1), UTC)
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar_windows(instant("2024-01-01T00:00:00Z"), UTC).day.contains(
            datetime(2024, 1, 1)
        )
