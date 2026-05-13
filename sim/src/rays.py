from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RayBundle:
    origins: np.ndarray  # (N, 3)
    directions: np.ndarray  # (N, 3), normalized
    powers_w: np.ndarray  # (N,)
    # If set, ray i targets facet ``target_facet[i]`` on the originating mirror grid (see
    # ``AltAzFlatMirrorGrid.incoming_ray_bundle_facet_grid``); used to skip cross-facet tests.
    target_facet: np.ndarray | None = None

    def alive(self) -> np.ndarray:
        return self.powers_w > 0.0

    def subset(self, mask: np.ndarray) -> "RayBundle":
        tf = None if self.target_facet is None else self.target_facet[mask]
        return RayBundle(
            origins=self.origins[mask],
            directions=self.directions[mask],
            powers_w=self.powers_w[mask],
            target_facet=tf,
        )

    @property
    def total_power_w(self) -> float:
        return float(np.sum(self.powers_w))
