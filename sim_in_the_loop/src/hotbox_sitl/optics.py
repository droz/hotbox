from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


@dataclass(slots=True)
class RayResult:
    facet_center_world: np.ndarray
    reflected_direction_world: np.ndarray
    hit_error_m: float


def reflect_toward_target(sun_vector_world: np.ndarray, facet_center_world: np.ndarray, target_world: np.ndarray) -> RayResult:
    incoming = -normalize(sun_vector_world)
    outgoing = normalize(np.asarray(target_world, dtype=float).reshape(3) - np.asarray(facet_center_world, dtype=float).reshape(3))
    normal = normalize(incoming + outgoing)
    reflected = incoming - 2.0 * float(np.dot(incoming, normal)) * normal
    hit_error = float(np.linalg.norm(reflected - outgoing))
    return RayResult(
        facet_center_world=np.asarray(facet_center_world, dtype=float).reshape(3),
        reflected_direction_world=reflected,
        hit_error_m=hit_error,
    )
