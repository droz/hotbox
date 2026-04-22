from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.absorber import SolarAbsorber
from src.flat_mirror_grid import AltAzFlatMirrorGrid
from src.mirror import CylindricalMirror
from src.rays import RayBundle
from src.sun import SunModel

MirrorLike = CylindricalMirror | AltAzFlatMirrorGrid

# Ignore shadow ordering when hits are within this (m) along the ray (numerical tie-break).
_SHADOW_TOL_M = 1e-5


@dataclass(slots=True)
class MirrorResult:
    mirror: MirrorLike
    incoming: RayBundle
    mirror_hit_mask: np.ndarray
    mirror_hit_points: np.ndarray
    reflected: RayBundle
    absorber_hit_mask: np.ndarray
    absorber_hit_points: np.ndarray

    @property
    def incident_power_w(self) -> float:
        return self.incoming.total_power_w

    @property
    def intercepted_power_w(self) -> float:
        return float(np.sum(self.incoming.powers_w[self.mirror_hit_mask]))

    @property
    def delivered_power_w(self) -> float:
        return float(np.sum(self.reflected.powers_w[self.absorber_hit_mask]))


@dataclass(slots=True)
class SimulationResult:
    sun_direction: np.ndarray
    per_mirror: list[MirrorResult]

    @property
    def total_delivered_power_w(self) -> float:
        return float(sum(m.delivered_power_w for m in self.per_mirror))


class HotboxSimulation:
    def __init__(
        self,
        sun: SunModel,
        absorber: SolarAbsorber,
        mirrors: list[MirrorLike],
        samples_u: int = 60,
        samples_v: int = 60,
    ) -> None:
        self.sun = sun
        self.absorber = absorber
        self.mirrors = mirrors
        self.samples_u = samples_u
        self.samples_v = samples_v

    def run(
        self,
        when_utc: datetime,
        *,
        samples_u: int | None = None,
        samples_v: int | None = None,
    ) -> SimulationResult:
        su = self.samples_u if samples_u is None else samples_u
        sv = self.samples_v if samples_v is None else samples_v
        sun_dir = self.sun.ray_direction(when_utc)
        per_mirror: list[MirrorResult] = []
        for i, mirror in enumerate(self.mirrors):
            extents = getattr(mirror, "incoming_ray_bundle_extents", None)
            if callable(extents):
                bundle_c, hu, hv = extents(sun_dir)
                incoming = self.sun.sample_parallel_bundle(
                    when_utc=when_utc,
                    center=bundle_c,
                    ray_direction=sun_dir,
                    samples_u=su,
                    samples_v=sv,
                    half_extent_u_m=hu,
                    half_extent_v_m=hv,
                )
            else:
                incoming = self.sun.sample_parallel_bundle(
                    when_utc=when_utc,
                    center=mirror.center,
                    ray_direction=sun_dir,
                    samples_u=su,
                    samples_v=sv,
                    cylinder_radius_m=mirror.sampling_radius_m,
                )
            mirror_hit_mask, mirror_hit_points, reflected = mirror.intersect_and_reflect(incoming)

            # Mutual shadowing: another mirror patch closer to the sun along the same sun ray
            # blocks this hit (incoming path intersects the other mirror first).
            o = incoming.origins
            d = incoming.directions
            t_i = np.sum((mirror_hit_points - o) * d, axis=1)
            shadowed = np.zeros_like(mirror_hit_mask, dtype=bool)
            for j, other in enumerate(self.mirrors):
                if j == i:
                    continue
                t_j = other.incoming_first_patch_hit_t(o, d)
                shadowed |= mirror_hit_mask & np.isfinite(t_j) & (t_j + _SHADOW_TOL_M < t_i)
            mirror_hit_mask = mirror_hit_mask & ~shadowed
            reflected.powers_w[shadowed] = 0.0

            absorber_hit_mask, absorber_hit_points = self.absorber.intersect(reflected)
            absorber_hit_mask &= reflected.powers_w > 0.0
            per_mirror.append(
                MirrorResult(
                    mirror=mirror,
                    incoming=incoming,
                    mirror_hit_mask=mirror_hit_mask,
                    mirror_hit_points=mirror_hit_points,
                    reflected=reflected,
                    absorber_hit_mask=absorber_hit_mask,
                    absorber_hit_points=absorber_hit_points,
                )
            )
        return SimulationResult(sun_direction=sun_dir, per_mirror=per_mirror)
