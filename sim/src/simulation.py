from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.absorber import SolarAbsorber
from src.mirror import CylindricalMirror
from src.rays import RayBundle
from src.sun import SunModel


@dataclass(slots=True)
class MirrorResult:
    mirror: CylindricalMirror
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
        mirrors: list[CylindricalMirror],
        samples_u: int = 60,
        samples_v: int = 60,
    ) -> None:
        self.sun = sun
        self.absorber = absorber
        self.mirrors = mirrors
        self.samples_u = samples_u
        self.samples_v = samples_v

    def run(self, when_utc: datetime) -> SimulationResult:
        sun_dir = self.sun.ray_direction(when_utc)
        per_mirror: list[MirrorResult] = []
        for mirror in self.mirrors:
            incoming = self.sun.sample_parallel_bundle(
                center=mirror.center,
                ray_direction=sun_dir,
                cylinder_radius_m=mirror.sampling_radius_m,
                samples_u=self.samples_u,
                samples_v=self.samples_v,
            )
            mirror_hit_mask, mirror_hit_points, reflected = mirror.intersect_and_reflect(incoming)
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
