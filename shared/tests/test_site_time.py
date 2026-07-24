"""Tests for shared site time / location helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from hotbox_shared import (
    SitePose,
    as_site_local,
    ensure_utc,
    load_system_constants,
    local_times_sunrise_to_sunset,
    site_local_datetime,
    utc_now,
)


def test_system_yaml_loads_timezone_id() -> None:
    system = load_system_constants()
    assert system.default_site.timezone_id == "America/Los_Angeles"


def test_ensure_utc_rejects_naive() -> None:
    with pytest.raises(ValueError, match="naive"):
        ensure_utc(datetime(2026, 9, 7, 14, 0, 0))


def test_ensure_utc_converts_site_local() -> None:
    local = site_local_datetime("America/Los_Angeles", 2026, 9, 7, 14, 0, 0)
    utc = ensure_utc(local)
    assert utc.tzinfo == timezone.utc
    assert utc.hour == 21  # PDT in September


def test_site_local_datetime_uses_iana_zone_not_host() -> None:
    when = site_local_datetime("Europe/Paris", 2026, 8, 30, 12, 0, 0)
    assert when.tzinfo == ZoneInfo("Europe/Paris")
    assert when.hour == 12


def test_as_site_local_roundtrip() -> None:
    site = SitePose.from_constants(load_system_constants().default_site)
    utc = datetime(2026, 9, 7, 21, 0, 0, tzinfo=timezone.utc)
    local = as_site_local(utc, site)
    assert local.tzinfo == site.zone
    assert local.hour == 14


def test_utc_now_is_aware_utc() -> None:
    now = utc_now()
    assert now.tzinfo == timezone.utc


def test_gps_position_override_keeps_timezone() -> None:
    site = SitePose.from_constants(load_system_constants().default_site)
    moved = site.with_position(latitude_deg=40.0, longitude_deg=-119.0, altitude_m=1200.0)
    assert moved.timezone_id == site.timezone_id
    assert moved.latitude_deg == 40.0


def test_local_times_sunrise_to_sunset_site_pose() -> None:
    site = SitePose(
        latitude_deg=37.8716,
        longitude_deg=-122.2585,
        altitude_m=186.0,
        timezone_id="America/Los_Angeles",
    )
    times, sunrise, sunset = local_times_sunrise_to_sunset(
        site, year=2026, month=6, day=21, step_minutes=60
    )
    assert sunrise is not None and sunset is not None
    assert len(times) >= 8
