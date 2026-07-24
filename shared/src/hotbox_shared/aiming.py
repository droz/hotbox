"""
Mirror pointing — single source of truth for heliostat aiming.

Both the live controller and the full raytrace simulation call :func:`solve_tracking`
to decide where each alt-az mount should point.

Algorithm
---------
1. **Bisector seed** (:func:`solve_bisector_tracking`): treat the mirror as a flat
   heliostat at the **mount pivot** (ignores ``mount_offset_d_m``). Fast closed form.
2. **Optional offset refine** (:func:`refine_tracking_for_mount_offset`): use
   :func:`evaluate_center_ray` as the forward model and ``scipy.optimize.least_squares``
   to nudge ``(az, el)`` so the **center-facet** reflected ray aims at the target.
   Controlled by ``control.solve_for_mount_offset`` in ``config/system.yaml``.
3. **Night stow**: if the sun is at or below the horizon, skip the solve and return
   :func:`horizontal_stow_angles` (pivot facet normal → world +Z, mirror face horizontal).
4. **Joint limits**: every command is mapped into physical mount limits relative to the
   oven-facing azimuth (see :class:`~hotbox_shared.mount.MountJointLimits`).

Frames (right-handed, meters):

- **World W**: ENU fixed to the site — +x east, +y north, +z up.
- **Mount body B**: rigid with the mirror assembly; at ``(az, el) = (0, 0)`` the body axes match world.
  The center facet sits at ``(0, 0, mount_offset_d_m)_B`` (Mn), offset from pivot An along +Z body.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .mount import (
    MountJointLimits,
    apply_mount_joint_limits,
    heading_and_tilt_from_normal,
    mount_az_el_align_body_normal_to_world,
    mount_rotation_matrix,
    normalize_mount_az_el,
    pivot_facet_normal_body,
)
from .vectors import bisector_normal, normalize, reflect_ray


@dataclass(frozen=True, slots=True)
class MirrorGridSpec:
    """Facet grid parameters that define the pivot facet normal at identity mount."""

    grid_nx: int
    grid_ny: int
    pitch_m: float
    radius_of_curvature_m: float
    mount_offset_d_m: float = 0.0

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
    night_stow: bool = False
    """True when the sun is below the horizon and angles are the horizontal stow pose."""

    def pivot_normal_world(self, pivot_normal_body: np.ndarray) -> np.ndarray:
        """Pivot facet reflective normal in world frame at these mount angles."""
        return normalize(mount_rotation_matrix(self.azimuth_deg, self.elevation_deg) @ pivot_normal_body.reshape(3))

    def display_heading_and_tilt(self, pivot_normal_body: np.ndarray) -> tuple[float, float]:
        """Physical azimuth [deg] and tilt from horizontal [deg] of the pivot facet normal."""
        return heading_and_tilt_from_normal(self.pivot_normal_world(pivot_normal_body))


def sun_elevation_deg(sun_direction_toward_scene: np.ndarray) -> float:
    """
    Geometric sun elevation [deg] from an incoming unit ray (sun → plant).

    Positive = above the horizon. Uses ``arcsin((-incoming)_z)`` in the ENU world frame.
    """
    incoming = normalize(sun_direction_toward_scene)
    toward_sun_z = float(-incoming[2])
    return float(np.rad2deg(np.arcsin(np.clip(toward_sun_z, -1.0, 1.0))))


def sun_is_above_horizon(sun_direction_toward_scene: np.ndarray) -> bool:
    """True iff the geometric sun elevation is strictly above the horizon."""
    return sun_elevation_deg(sun_direction_toward_scene) > 0.0


def _limited_angles(
    azimuth_deg: float,
    elevation_deg: float,
    *,
    mount_world: np.ndarray,
    target_world: np.ndarray,
    joint_limits: MountJointLimits | None,
    night_stow: bool = False,
) -> MountAngles:
    az, el = apply_mount_joint_limits(
        azimuth_deg,
        elevation_deg,
        mount_world=mount_world,
        absorber_world=target_world,
        limits=joint_limits,
    )
    return MountAngles(azimuth_deg=az, elevation_deg=el, night_stow=night_stow)


def horizontal_stow_angles(
    pivot_facet_normal_body: np.ndarray,
    *,
    mount_world: np.ndarray | None = None,
    target_world: np.ndarray | None = None,
    joint_limits: MountJointLimits | None = None,
) -> MountAngles:
    """
    Face-up stow: mount ``(azimuth, elevation) = (0, 0)``.

    At identity the body axes match world, so a +Z body normal points at zenith and the
    mirror face is horizontal. Fixed angles (not joint-limited) so Park / night stow are
    unambiguous. Unused kwargs kept for call-site compatibility.
    """
    _ = pivot_facet_normal_body, mount_world, target_world, joint_limits
    return MountAngles(azimuth_deg=0.0, elevation_deg=0.0, night_stow=True)


@dataclass(frozen=True, slots=True)
class CenterRay:
    """Forward geometry of the sun ray reflected off the center (pivot) facet."""

    facet_center_world: np.ndarray
    """World position of the center facet [m] (Mn)."""

    normal_world: np.ndarray
    """Unit reflective normal of the center facet in world frame."""

    reflected_direction: np.ndarray
    """Unit reflected ray leaving the facet."""

    def miss_vector_to_point(self, target_world: np.ndarray) -> np.ndarray:
        """
        Perpendicular miss from the reflected ray to ``target_world`` [m].

        Zero iff the ray passes through the target. Magnitude equals the closest-approach distance.
        """
        target = np.asarray(target_world, dtype=float).reshape(3)
        delta = target - self.facet_center_world
        return delta - self.reflected_direction * float(np.dot(delta, self.reflected_direction))

    def miss_m(self, target_world: np.ndarray) -> float:
        """Closest-approach distance from the reflected ray to ``target_world`` [m]."""
        return float(np.linalg.norm(self.miss_vector_to_point(target_world)))

    def impact_on_plane(self, plane_point_world: np.ndarray, plane_normal_world: np.ndarray) -> np.ndarray:
        """
        Intersection of the reflected ray with a plane.

        Raises ``ValueError`` if the ray is parallel to the plane (or hits from the wrong side
        with zero length to the plane along the ray).
        """
        p0 = np.asarray(plane_point_world, dtype=float).reshape(3)
        n = normalize(plane_normal_world)
        denom = float(np.dot(self.reflected_direction, n))
        if abs(denom) < 1e-12:
            raise ValueError("reflected ray is parallel to the plane")
        t = float(np.dot(p0 - self.facet_center_world, n) / denom)
        return self.facet_center_world + t * self.reflected_direction


def pivot_facet_center_world(
    mount_world: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    mount_offset_d_m: float,
) -> np.ndarray:
    """World position of the center facet: ``mount + R(az, el) @ (0, 0, d)``."""
    mount = np.asarray(mount_world, dtype=float).reshape(3)
    r = mount_rotation_matrix(azimuth_deg, elevation_deg)
    return mount + r @ np.array([0.0, 0.0, float(mount_offset_d_m)], dtype=float)


def evaluate_center_ray(
    *,
    sun_direction_toward_scene: np.ndarray,
    mount_world: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    mount_offset_d_m: float,
    pivot_facet_normal_body: np.ndarray,
) -> CenterRay:
    """
    Forward model: center-facet pose and reflected sun ray at the given mount angles.

    This is the shared geometry used for display miss metrics and for offset refinement.
    """
    r = mount_rotation_matrix(azimuth_deg, elevation_deg)
    pivot = normalize(pivot_facet_normal_body)
    normal_world = normalize(r @ pivot)
    facet = pivot_facet_center_world(mount_world, azimuth_deg, elevation_deg, mount_offset_d_m)
    incoming = normalize(sun_direction_toward_scene)
    reflected = reflect_ray(incoming, normal_world)
    return CenterRay(
        facet_center_world=facet,
        normal_world=normal_world,
        reflected_direction=reflected,
    )


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
    joint_limits: MountJointLimits | None = None,
) -> MountAngles:
    """
    Closed-form bisector seed: flat heliostat at the **pivot** (ignores mount offset).

    Prefer :func:`solve_tracking` for production pointing unless you explicitly want the seed.
    """
    n_bisector = bisector_normal_at_mount(sun_direction_toward_scene, mount_world, target_world)
    pivot = normalize(pivot_facet_normal_body)
    azimuth_deg, elevation_deg = mount_az_el_align_body_normal_to_world(pivot, n_bisector)
    return _limited_angles(
        azimuth_deg,
        elevation_deg,
        mount_world=mount_world,
        target_world=target_world,
        joint_limits=joint_limits,
    )


def refine_tracking_for_mount_offset(
    *,
    sun_direction_toward_scene: np.ndarray,
    mount_world: np.ndarray,
    target_world: np.ndarray,
    pivot_facet_normal_body: np.ndarray,
    mount_offset_d_m: float,
    initial: MountAngles,
    joint_limits: MountJointLimits | None = None,
) -> MountAngles:
    """
    Nudge ``initial`` mount angles so the center-facet reflected ray aims at ``target_world``.

    Uses ``scipy.optimize.least_squares`` on the 3D ray–target miss vector from
    :meth:`CenterRay.miss_vector_to_point`. Joint limits are applied after the optimizer.
    """
    if abs(float(mount_offset_d_m)) < 1e-12:
        return initial

    sun = normalize(sun_direction_toward_scene)
    mount = np.asarray(mount_world, dtype=float).reshape(3)
    target = np.asarray(target_world, dtype=float).reshape(3)
    pivot = normalize(pivot_facet_normal_body)
    d = float(mount_offset_d_m)

    def residual(x: np.ndarray) -> np.ndarray:
        az, el = normalize_mount_az_el(float(x[0]), float(x[1]))
        ray = evaluate_center_ray(
            sun_direction_toward_scene=sun,
            mount_world=mount,
            azimuth_deg=az,
            elevation_deg=el,
            mount_offset_d_m=d,
            pivot_facet_normal_body=pivot,
        )
        return ray.miss_vector_to_point(target)

    result = least_squares(
        residual,
        x0=np.array([initial.azimuth_deg, initial.elevation_deg], dtype=float),
        method="lm",
    )
    az, el = normalize_mount_az_el(float(result.x[0]), float(result.x[1]))
    return _limited_angles(
        az,
        el,
        mount_world=mount_world,
        target_world=target_world,
        joint_limits=joint_limits,
    )


def solve_tracking(
    *,
    sun_direction_toward_scene: np.ndarray,
    mount_world: np.ndarray,
    target_world: np.ndarray,
    pivot_facet_normal_body: np.ndarray,
    mount_offset_d_m: float = 0.0,
    solve_for_mount_offset: bool = True,
    joint_limits: MountJointLimits | None = None,
) -> MountAngles:
    """
    Compute mount angles that aim the center-facet reflected ray at ``target_world``.

    This is the primary API used by both the controller and raytrace simulation.

    When the sun is at or below the horizon, returns :func:`horizontal_stow_angles`
    (pivot facet normal → world +Z) instead of attempting a reflection solve.

    Args:
        sun_direction_toward_scene: Unit vector from the sun toward the plant.
        mount_world: Mount pivot position in world coordinates [m].
        target_world: Point to aim at (typically absorber center) [m].
        pivot_facet_normal_body: Unit normal of the center facet in mount body frame at (0, 0).
        mount_offset_d_m: Pivot-to-facet offset along +Z body [m].
        solve_for_mount_offset: If true (and offset ≠ 0), refine the bisector seed with least squares.
        joint_limits: Physical travel limits. ``None`` uses default :class:`MountJointLimits`.

    Returns:
        ``MountAngles`` with azimuth/elevation inside the joint limits.
        ``night_stow`` is True when the sun is down.
    """
    pivot = normalize(pivot_facet_normal_body)
    if not sun_is_above_horizon(sun_direction_toward_scene):
        return horizontal_stow_angles(
            pivot,
            mount_world=mount_world,
            target_world=target_world,
            joint_limits=joint_limits,
        )

    seed = solve_bisector_tracking(
        sun_direction_toward_scene=sun_direction_toward_scene,
        mount_world=mount_world,
        target_world=target_world,
        pivot_facet_normal_body=pivot,
        joint_limits=joint_limits,
    )
    if not solve_for_mount_offset:
        return seed
    return refine_tracking_for_mount_offset(
        sun_direction_toward_scene=sun_direction_toward_scene,
        mount_world=mount_world,
        target_world=target_world,
        pivot_facet_normal_body=pivot,
        mount_offset_d_m=mount_offset_d_m,
        initial=seed,
        joint_limits=joint_limits,
    )


def solve_tracking_for_grid(
    *,
    sun_direction_toward_scene: np.ndarray,
    mount_world: np.ndarray,
    target_world: np.ndarray,
    grid: MirrorGridSpec,
    solve_for_mount_offset: bool = True,
    joint_limits: MountJointLimits | None = None,
) -> MountAngles:
    """Convenience wrapper that derives the pivot facet normal and offset from ``grid``."""
    return solve_tracking(
        sun_direction_toward_scene=sun_direction_toward_scene,
        mount_world=mount_world,
        target_world=target_world,
        pivot_facet_normal_body=grid.pivot_normal_body(),
        mount_offset_d_m=grid.mount_offset_d_m,
        solve_for_mount_offset=solve_for_mount_offset,
        joint_limits=joint_limits,
    )


def solve_bisector_tracking_for_grid(
    *,
    sun_direction_toward_scene: np.ndarray,
    mount_world: np.ndarray,
    target_world: np.ndarray,
    grid: MirrorGridSpec,
    joint_limits: MountJointLimits | None = None,
) -> MountAngles:
    """Bisector-only convenience wrapper (no mount-offset refine)."""
    return solve_bisector_tracking(
        sun_direction_toward_scene=sun_direction_toward_scene,
        mount_world=mount_world,
        target_world=target_world,
        pivot_facet_normal_body=grid.pivot_normal_body(),
        joint_limits=joint_limits,
    )
