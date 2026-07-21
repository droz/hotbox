"""
Mirror pointing — single source of truth for bisector tracking.

Both the live controller and the full raytrace simulation call :func:`solve_bisector_tracking`
to decide where each alt-az mount should point. The algorithm is **flat heliostat at the pivot**:

1. Form the unit sun ray **toward the plant** (from the sun toward the mirrors).
2. Form the unit ray from the mount pivot toward the target (typically the absorber center).
3. Compute the specular bisector normal ``n_W`` that reflects (1) toward (2).
4. Solve mount angles so the **pivot facet** body normal aligns with ``n_W`` under
   ``R_mount = R_z(azimuth) @ R_x(elevation)``.

Frames (right-handed, meters):

- **World W**: ENU fixed to the site — +x east, +y north, +z up.
- **Mount body B**: rigid with the mirror assembly; at ``(az, el) = (0, 0)`` the body axes match world.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mount import (
    heading_and_tilt_from_normal,
    mount_az_el_align_body_normal_to_world,
    mount_rotation_matrix,
    pivot_facet_normal_body,
)
from .vectors import bisector_normal, normalize


@dataclass(frozen=True, slots=True)
class MirrorGridSpec:
    """Facet grid parameters that define the pivot facet normal at identity mount."""

    grid_nx: int
    grid_ny: int
    pitch_m: float
    radius_of_curvature_m: float

    def pivot_normal_body(self) -> np.ndarray:
        return pivot_facet_normal_body(
            grid_nx=self.grid_nx,
            grid_ny=self.grid_ny,
            pitch_m=self.pitch_m,
            radius_of_curvature_m=self.radius_of_curvature_m,
        )


@dataclass(frozen=True, slots=True)
class MountAngles:
    """Commanded alt-az mount angles sent to firmware or applied in simulation."""

    azimuth_deg: float
    elevation_deg: float

    def pivot_normal_world(self, pivot_normal_body: np.ndarray) -> np.ndarray:
        """Pivot facet reflective normal in world frame at these mount angles."""
        return normalize(mount_rotation_matrix(self.azimuth_deg, self.elevation_deg) @ pivot_normal_body.reshape(3))

    def display_heading_and_tilt(self, pivot_normal_body: np.ndarray) -> tuple[float, float]:
        """Physical azimuth [deg] and tilt from horizontal [deg] of the pivot facet normal."""
        return heading_and_tilt_from_normal(self.pivot_normal_world(pivot_normal_body))


def bisector_normal_at_mount(
    incoming_toward_scene: np.ndarray,
    mount_world: np.ndarray,
    target_world: np.ndarray,
) -> np.ndarray:
    """
    Bisector normal for a flat mirror at ``mount_world`` reflecting sunlight toward ``target_world``.

    Args:
        incoming_toward_scene: Unit vector from the sun toward the plant.
        mount_world: Mount pivot position in world coordinates [m].
        target_world: Target point in world coordinates [m] (e.g. absorber center).
    """
    mount = np.asarray(mount_world, dtype=float).reshape(3)
    target = np.asarray(target_world, dtype=float).reshape(3)
    incoming = normalize(incoming_toward_scene)
    outgoing = normalize(target - mount)
    return bisector_normal(incoming, outgoing)


def solve_bisector_tracking(
    *,
    sun_direction_toward_scene: np.ndarray,
    mount_world: np.ndarray,
    target_world: np.ndarray,
    pivot_facet_normal_body: np.ndarray,
) -> MountAngles:
    """
    Compute mount angles for bisector tracking at the pivot.

    This is the primary API used by both the controller and raytrace simulation.

    Args:
        sun_direction_toward_scene: Unit vector from the sun toward the plant.
        mount_world: Mount pivot position in world coordinates [m].
        target_world: Point to reflect toward (typically absorber center) [m].
        pivot_facet_normal_body: Unit normal of the center facet in mount body frame at (0, 0).

    Returns:
        ``MountAngles`` with azimuth in ``[0, 360)`` and elevation in ``[-90, 90]``.
    """
    n_bisector = bisector_normal_at_mount(sun_direction_toward_scene, mount_world, target_world)
    pivot = normalize(pivot_facet_normal_body)
    azimuth_deg, elevation_deg = mount_az_el_align_body_normal_to_world(pivot, n_bisector)
    return MountAngles(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)


def solve_bisector_tracking_for_grid(
    *,
    sun_direction_toward_scene: np.ndarray,
    mount_world: np.ndarray,
    target_world: np.ndarray,
    grid: MirrorGridSpec,
) -> MountAngles:
    """Convenience wrapper that derives the pivot facet normal from grid geometry."""
    return solve_bisector_tracking(
        sun_direction_toward_scene=sun_direction_toward_scene,
        mount_world=mount_world,
        target_world=target_world,
        pivot_facet_normal_body=grid.pivot_normal_body(),
    )
