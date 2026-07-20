from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actuator import ActuatorModel
from .optics import RayResult, reflect_toward_target


@dataclass(slots=True)
class MirrorScenario:
    node_id: int
    mount_world: np.ndarray
    facet_offset_world: np.ndarray
    altitude_axis: ActuatorModel = field(default_factory=ActuatorModel)
    azimuth_axis: ActuatorModel = field(default_factory=ActuatorModel)

    def step(self, pwm_az: float, pwm_alt: float, dt_s: float, sun_vector_world: np.ndarray, target_world: np.ndarray) -> RayResult:
        self.azimuth_axis.step(pwm_az, dt_s)
        self.altitude_axis.step(pwm_alt, dt_s)
        facet_center = np.asarray(self.mount_world, dtype=float).reshape(3) + np.asarray(self.facet_offset_world, dtype=float).reshape(3)
        return reflect_toward_target(sun_vector_world, facet_center, target_world)
