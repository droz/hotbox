from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from hotbox_shared import ensure_utc
from pvlib.location import Location

from .config import SiteConfig


def pvlib_to_world_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    return np.array(
        [
            np.cos(el) * np.sin(az),
            np.cos(el) * np.cos(az),
            np.sin(el),
        ],
        dtype=float,
    )


@dataclass(slots=True)
class SunVector:
    azimuth_deg: float
    elevation_deg: float
    world_vector: np.ndarray


class SunService:
    def __init__(self, site: SiteConfig, cache_ttl_s: float = 1.0) -> None:
        self._site = site
        self._location = Location(site.latitude_deg, site.longitude_deg, tz="UTC", altitude=site.altitude_m)
        self._cache_ttl_s = cache_ttl_s
        self._cached_when: datetime | None = None
        self._cached_vector: SunVector | None = None

    def sun_vector(self, when: datetime) -> SunVector:
        when_utc = ensure_utc(when)
        if (
            self._cached_vector is not None
            and self._cached_when is not None
            and abs((when_utc - self._cached_when).total_seconds()) < self._cache_ttl_s
        ):
            return self._cached_vector

        frame = self._location.get_solarposition([when_utc])
        azimuth_deg = float(frame["azimuth"].iloc[0])
        zenith_deg = float(frame["zenith"].iloc[0])
        elevation_deg = 90.0 - zenith_deg
        vector = SunVector(
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            world_vector=pvlib_to_world_vector(azimuth_deg, elevation_deg),
        )
        self._cached_when = when_utc
        self._cached_vector = vector
        return vector
