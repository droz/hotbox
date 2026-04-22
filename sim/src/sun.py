from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

from src.geometry import az_el_to_unit, orthonormal_basis_from_direction
from src.rays import RayBundle


@dataclass(slots=True)
class SunModel:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float

    def ray_direction(self, when_utc: datetime) -> np.ndarray:
        ts = when_utc.astimezone(timezone.utc)
        solpos = pvlib.solarposition.get_solarposition(
            time=ts,
            latitude=self.latitude_deg,
            longitude=self.longitude_deg,
            altitude=self.altitude_m,
        )
        zenith_deg = float(solpos["apparent_zenith"].iloc[0])
        azimuth_deg = float(solpos["azimuth"].iloc[0])
        elevation_deg = 90.0 - zenith_deg
        sun_to_world = az_el_to_unit(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)
        return -sun_to_world

    def clear_sky_dni_w_per_m2(self, when_utc: datetime) -> float:
        """
        Direct normal irradiance (W/m²) from pvlib's clear-sky model (Ineichen, via
        ``Location.get_clearsky``), for this site's lat/lon/altitude at ``when_utc``.
        """
        ts = when_utc.astimezone(timezone.utc)
        idx = pd.DatetimeIndex([ts])
        loc = Location(
            self.latitude_deg,
            self.longitude_deg,
            tz="UTC",
            altitude=self.altitude_m,
        )
        cs = loc.get_clearsky(idx)
        dni = float(cs["dni"].iloc[0])
        if not np.isfinite(dni) or dni < 0.0:
            return 0.0
        return dni

    def sample_parallel_bundle(
        self,
        when_utc: datetime,
        center: np.ndarray,
        ray_direction: np.ndarray,
        samples_u: int,
        samples_v: int,
        *,
        cylinder_radius_m: float | None = None,
        half_extent_u_m: float | None = None,
        half_extent_v_m: float | None = None,
        upstream_distance_m: float = 50.0,
    ) -> RayBundle:
        """
        Parallel rays on a regular grid in the plane through ``center`` orthogonal to
        ``ray_direction``.

        Provide either ``cylinder_radius_m`` (square of side ``2 * radius``) or both
        ``half_extent_u_m`` and ``half_extent_v_m`` (half-widths along the ``(u, v)`` basis from
        ``orthonormal_basis_from_direction``). The latter avoids sampling empty corners when the
        mirror footprint is elongated in projection.
        """
        u_axis, v_axis = orthonormal_basis_from_direction(ray_direction)
        if half_extent_u_m is not None and half_extent_v_m is not None:
            hu = float(half_extent_u_m)
            hv = float(half_extent_v_m)
        elif cylinder_radius_m is not None:
            hu = hv = float(cylinder_radius_m)
        else:
            raise TypeError(
                "Provide cylinder_radius_m or both half_extent_u_m and half_extent_v_m"
            )
        hu = max(hu, 1e-6)
        hv = max(hv, 1e-6)

        u = np.linspace(-hu, hu, samples_u)
        v = np.linspace(-hv, hv, samples_v)
        uu, vv = np.meshgrid(u, v, indexing="xy")

        c = np.asarray(center, dtype=float).reshape(3)
        starts_plane = c + uu[..., None] * u_axis + vv[..., None] * v_axis
        origins = starts_plane - upstream_distance_m * ray_direction
        origins = origins.reshape(-1, 3)

        num = origins.shape[0]
        directions = np.repeat(ray_direction.reshape(1, 3), num, axis=0)

        du = (2.0 * hu) / max(samples_u - 1, 1)
        dv = (2.0 * hv) / max(samples_v - 1, 1)
        dni = self.clear_sky_dni_w_per_m2(when_utc)
        ray_power = dni * du * dv
        powers_w = np.full(num, ray_power, dtype=float)
        return RayBundle(origins=origins, directions=directions, powers_w=powers_w)
