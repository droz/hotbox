"""Tests for site-local sunrise–sunset sampling (site TZ, not host TZ)."""

from __future__ import annotations

from hotbox_shared import SitePose, local_times_sunrise_to_sunset

from src.main import spot_pattern_sample_times


def test_local_times_match_requested_civil_day_in_paris() -> None:
    site = SitePose(
        latitude_deg=45.5885,
        longitude_deg=5.0904,
        altitude_m=380.0,
        timezone_id="Europe/Paris",
    )
    times, sunrise, sunset = local_times_sunrise_to_sunset(
        site,
        year=2026,
        month=8,
        day=30,
        step_minutes=20,
    )
    assert sunrise is not None and sunset is not None
    assert sunrise.date().isoformat() == "2026-08-30"
    assert sunset.date().isoformat() == "2026-08-30"
    assert times
    assert times[0] >= sunrise
    assert times[-1] <= sunset
    assert all(str(t.tzinfo) == "Europe/Paris" or getattr(t.tzinfo, "key", None) == "Europe/Paris" for t in times)


def test_spot_pattern_sample_times_returns_multiple_panels() -> None:
    site = SitePose(
        latitude_deg=45.5885,
        longitude_deg=5.0904,
        altitude_m=380.0,
        timezone_id="Europe/Paris",
    )
    times = spot_pattern_sample_times(
        site,
        year=2026,
        month=9,
        day=7,
        step_minutes=20,
        num_panels=12,
    )
    assert len(times) >= 8
    assert all(t.date().isoformat() == "2026-09-07" for t in times)


def test_berkeley_day_curve_uses_america_los_angeles() -> None:
    site = SitePose(
        latitude_deg=37.8716,
        longitude_deg=-122.2585,
        altitude_m=186.0,
        timezone_id="America/Los_Angeles",
    )
    times, sunrise, sunset = local_times_sunrise_to_sunset(
        site,
        year=2026,
        month=9,
        day=7,
        step_minutes=30,
    )
    assert sunrise is not None and sunset is not None
    assert sunrise.tzinfo is not None
    assert getattr(sunrise.tzinfo, "key", None) == "America/Los_Angeles" or str(sunrise.tzinfo) == "America/Los_Angeles"
    assert times
    assert all(
        getattr(t.tzinfo, "key", None) == "America/Los_Angeles" or str(t.tzinfo) == "America/Los_Angeles"
        for t in times
    )
