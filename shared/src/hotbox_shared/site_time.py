"""
Site time and location — never use the computer's local timezone.

All civil-day work (sunrise–sunset sampling, plot axes, scene timestamps) uses the
plant site's IANA timezone from ``config/system.yaml`` (``default_site.timezone_id``).
Physics / solar-position callers should pass **timezone-aware** datetimes; convert to
UTC with :func:`ensure_utc` before handing them to pvlib.

Live clock: prefer GPS UTC time when available; otherwise :func:`utc_now`. Display in
site local via :func:`as_site_local`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from pvlib.location import Location

from .system import SiteConstants


@dataclass(frozen=True, slots=True)
class SitePose:
    """Geographic plant site + IANA timezone for civil-time work."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    timezone_id: str

    @classmethod
    def from_constants(cls, site: SiteConstants) -> SitePose:
        return cls(
            latitude_deg=float(site.latitude_deg),
            longitude_deg=float(site.longitude_deg),
            altitude_m=float(site.altitude_m),
            timezone_id=str(site.timezone_id),
        )

    def with_position(
        self,
        *,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m: float,
    ) -> SitePose:
        """GPS (or other) position override; keeps the plant timezone_id."""
        return SitePose(
            latitude_deg=float(latitude_deg),
            longitude_deg=float(longitude_deg),
            altitude_m=float(altitude_m),
            timezone_id=self.timezone_id,
        )

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_id)

    def pvlib_location(self, *, tz: str | None = None) -> Location:
        """
        ``pvlib.location.Location`` for this site.

        Default ``tz`` is the site IANA id (civil-day / rise–set). Pass ``\"UTC\"`` for
        solar-position calls that already convert timestamps to UTC.
        """
        return Location(
            self.latitude_deg,
            self.longitude_deg,
            tz if tz is not None else self.timezone_id,
            altitude=self.altitude_m,
        )


def utc_now() -> datetime:
    """Current UTC time (host clock). Prefer GPS UTC when a valid fix is available."""
    return datetime.now(timezone.utc)


def ensure_utc(when: datetime) -> datetime:
    """
    Return ``when`` in UTC.

    Raises ``ValueError`` if ``when`` is naive — never silently interpret host-local time.
    """
    if when.tzinfo is None:
        raise ValueError(
            "naive datetime is not allowed; pass a timezone-aware value "
            "(e.g. site_local_datetime(...) or utc_now())"
        )
    return when.astimezone(timezone.utc)


def as_site_local(when: datetime, site: SitePose | SiteConstants | str) -> datetime:
    """Convert an aware datetime to the site's local timezone."""
    zone = _zone_of(site)
    return ensure_utc(when).astimezone(zone)


def site_local_datetime(
    site: SitePose | SiteConstants | str,
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    """Build an aware civil datetime in the site's timezone (not the computer's)."""
    return datetime(year, month, day, hour, minute, second, tzinfo=_zone_of(site))


def site_local_date(site: SitePose | SiteConstants | str, when: datetime | None = None) -> date:
    """Civil calendar date at the site for ``when`` (default: now UTC)."""
    stamp = utc_now() if when is None else when
    return as_site_local(stamp, site).date()


def format_site_local(
    when: datetime,
    site: SitePose | SiteConstants | str,
    fmt: str = "%Y-%m-%d %H:%M %Z",
) -> str:
    """Format ``when`` in site local time for logs and plot labels."""
    return as_site_local(when, site).strftime(fmt)


def hours_since_site_midnight(when: datetime, site: SitePose | SiteConstants | str) -> float:
    """Fractional hours since site-local midnight (for same-day plot overlays)."""
    local = as_site_local(when, site)
    return (
        local.hour
        + local.minute / 60.0
        + local.second / 3600.0
        + local.microsecond / 3_600_000_000.0
    )


def local_times_sunrise_to_sunset(
    site: SitePose | SiteConstants,
    *,
    year: int,
    month: int,
    day: int,
    step_minutes: int,
) -> tuple[list[datetime], datetime | None, datetime | None]:
    """
    Site-local samples from first step on/after sunrise through last on/before sunset.

    Sunrise/sunset from pvlib SPA for the site's lat/lon/alt and ``timezone_id``.
    Uses local noon as the SPA reference (midnight is ambiguous across timezones).
    """
    pose = site if isinstance(site, SitePose) else SitePose.from_constants(site)
    loc = pose.pvlib_location()
    day_noon = pd.Timestamp(year=year, month=month, day=day, hour=12, tz=pose.timezone_id)
    rs = loc.get_sun_rise_set_transit(pd.DatetimeIndex([day_noon]), method="spa")
    sunrise_ts = rs["sunrise"].iloc[0]
    sunset_ts = rs["sunset"].iloc[0]
    if pd.isna(sunrise_ts) or pd.isna(sunset_ts):
        return [], None, None

    sunrise_ts = sunrise_ts.floor("s")
    sunset_ts = sunset_ts.floor("s")
    sunrise = sunrise_ts.to_pydatetime()
    sunset = sunset_ts.to_pydatetime()
    step = timedelta(minutes=int(step_minutes))
    midnight = site_local_datetime(pose, year, month, day)
    t = midnight
    while t < sunrise:
        t += step
    out: list[datetime] = []
    while t <= sunset:
        out.append(t)
        t += step
    return out, sunrise, sunset


def _zone_of(site: SitePose | SiteConstants | str) -> ZoneInfo:
    if isinstance(site, str):
        return ZoneInfo(site)
    if isinstance(site, SitePose):
        return site.zone
    return ZoneInfo(site.timezone_id)
