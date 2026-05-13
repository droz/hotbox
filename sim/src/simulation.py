from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time

import numpy as np

from src.absorber import SolarAbsorber
from src.flat_mirror_grid import AltAzFlatMirrorGrid
from src.rays import RayBundle
from src.sun import SunModel

# Ignore shadow ordering when hits are within this (m) along the ray (numerical tie-break).
_SHADOW_TOL_M = 1e-5


@dataclass(slots=True)
class MirrorResult:
    mirror: AltAzFlatMirrorGrid
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
        mirrors: list[AltAzFlatMirrorGrid],
        samples_u: int = 60,
        samples_v: int = 60,
    ) -> None:
        """
        ``samples_u`` / ``samples_v`` are the number of **cells** along each axis on every square
        facet (one ray per cell at the cell center); see
        ``AltAzFlatMirrorGrid.incoming_ray_bundle_facet_grid``.
        """
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
        verbose: bool = False,
    ) -> SimulationResult:
        su = self.samples_u if samples_u is None else samples_u
        sv = self.samples_v if samples_v is None else samples_v
        t_run = time.perf_counter()
        sun_dir = self.sun.ray_direction(when_utc)
        per_mirror: list[MirrorResult] = []
        if verbose:
            print(
                f"[sim] run start — mirrors={len(self.mirrors)} samples_u/v={su}×{sv} "
                f"utc={when_utc.isoformat()}",
                flush=True,
            )
        for i, mirror in enumerate(self.mirrors):
            t0 = time.perf_counter()
            incoming = mirror.incoming_ray_bundle_facet_grid(
                when_utc=when_utc,
                samples_u=su,
                samples_v=sv,
            )
            t1 = time.perf_counter()
            mirror_hit_mask, mirror_hit_points, reflected = mirror.intersect_and_reflect(incoming)
            t2 = time.perf_counter()

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
            t3 = time.perf_counter()

            absorber_hit_mask, absorber_hit_points = self.absorber.intersect(reflected)
            absorber_hit_mask &= reflected.powers_w > 0.0
            # Outgoing-path occlusion: reflected rays that would hit the absorber can still be
            # blocked by another mirror patch first.
            t_abs = np.sum((absorber_hit_points - reflected.origins) * reflected.directions, axis=1)
            blocked_out = np.zeros_like(absorber_hit_mask, dtype=bool)
            for j, other in enumerate(self.mirrors):
                if j == i:
                    continue
                t_j_out = other.incoming_first_patch_hit_t(reflected.origins, reflected.directions)
                blocked_out |= (
                    absorber_hit_mask
                    & np.isfinite(t_j_out)
                    & (t_j_out > 1e-8)
                    & (t_j_out + _SHADOW_TOL_M < t_abs)
                )
            absorber_hit_mask &= ~blocked_out
            reflected.powers_w[blocked_out] = 0.0
            t4 = time.perf_counter()
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
            if verbose:
                n_rays = int(incoming.origins.shape[0])
                print(
                    f"[sim]   mirror {i}: rays={n_rays} "
                    f"facet_grid={t1 - t0:.4f}s intersect={t2 - t1:.4f}s "
                    f"shadow={t3 - t2:.4f}s absorber+occlusion={t4 - t3:.4f}s "
                    f"subtotal={t4 - t0:.4f}s",
                    flush=True,
                )
        if verbose:
            print(f"[sim] run done — total {time.perf_counter() - t_run:.4f}s", flush=True)
        return SimulationResult(sun_direction=sun_dir, per_mirror=per_mirror)
