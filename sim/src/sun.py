from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pvlib

from src.geometry import az_el_to_unit, orthonormal_basis_from_direction
from src.rays import RayBundle


@dataclass(slots=True)
class SunModel:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    dni_w_per_m2: float = 1000.0

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

    def sample_parallel_bundle(
        self,
        center: np.ndarray,
        ray_direction: np.ndarray,
        cylinder_radius_m: float,
        samples_u: int,
        samples_v: int,
        upstream_distance_m: float = 50.0,
    ) -> RayBundle:
        u_axis, v_axis = orthonormal_basis_from_direction(ray_direction)
        u = np.linspace(-cylinder_radius_m, cylinder_radius_m, samples_u)
        v = np.linspace(-cylinder_radius_m, cylinder_radius_m, samples_v)
        uu, vv = np.meshgrid(u, v, indexing="xy")

        starts_plane = center + uu[..., None] * u_axis + vv[..., None] * v_axis
        origins = starts_plane - upstream_distance_m * ray_direction
        origins = origins.reshape(-1, 3)

        num = origins.shape[0]
        directions = np.repeat(ray_direction.reshape(1, 3), num, axis=0)

        du = (2.0 * cylinder_radius_m) / max(samples_u - 1, 1)
        dv = (2.0 * cylinder_radius_m) / max(samples_v - 1, 1)
        ray_power = self.dni_w_per_m2 * du * dv
        powers_w = np.full(num, ray_power, dtype=float)
        return RayBundle(origins=origins, directions=directions, powers_w=powers_w)
