"""Tests for local sunrise–sunset sampling used by day curves / spot grids."""

from __future__ import annotations

from zoneinfo import ZoneInfo

from src.main import local_times_sunrise_to_sunset, spot_pattern_sample_times


def test_local_times_match_requested_civil_day_in_paris() -> None:
    times, sunrise, sunset = local_times_sunrise_to_sunset(
        45.5885,
        5.0904,
        380.0,
        2026,
        8,
        30,
        ZoneInfo("Europe/Paris"),
        20,
    )
    assert sunrise is not None and sunset is not None
    assert sunrise.date().isoformat() == "2026-08-30"
    assert sunset.date().isoformat() == "2026-08-30"
    assert times
    assert times[0] >= sunrise
    assert times[-1] <= sunset


def test_spot_pattern_sample_times_returns_multiple_panels() -> None:
    times = spot_pattern_sample_times(
        45.5885,
        5.0904,
        380.0,
        2026,
        9,
        7,
        ZoneInfo("Europe/Paris"),
        20,
        12,
    )
    assert len(times) >= 8
    assert all(t.date().isoformat() == "2026-09-07" for t in times)
