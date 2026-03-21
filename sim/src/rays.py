from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RayBundle:
    origins: np.ndarray  # (N, 3)
    directions: np.ndarray  # (N, 3), normalized
    powers_w: np.ndarray  # (N,)

    def alive(self) -> np.ndarray:
        return self.powers_w > 0.0

    def subset(self, mask: np.ndarray) -> "RayBundle":
        return RayBundle(
            origins=self.origins[mask],
            directions=self.directions[mask],
            powers_w=self.powers_w[mask],
        )

    @property
    def total_power_w(self) -> float:
        return float(np.sum(self.powers_w))
