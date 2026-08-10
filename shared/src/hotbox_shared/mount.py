"""Alt-az mount kinematics shared by the live controller and raytrace simulation.

World frame is ENU (+X east, +Y north, +Z up). Mount angles follow the usual
astronomical convention:

* **Elevation** ``0°`` = horizon, ``90°`` = zenith (face-up).
* **Azimuth** ``0°`` = north, ``90°`` = east, ``180°`` = south, ``270°`` = west
  (clockwise from north when viewed from above).

The body → world rotation is ``R = R_z(−az) @ R_x(el − 90°)``. At
``(az, el) = (0, 90)`` the body axes match world (identity). At ``el = 0`` a
body ``+Z`` normal lies in the horizontal plane at heading ``az``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vectors import normalize


@dataclass(frozen=True, slots=True)
class MountJointLimits:
    """Physical joint limits for commanded mount angles.

    Azimuth limits are **relative** to oven-facing: ``0`` is the absolute mount azimuth
    that aims the mirror toward the absorber at low elevation (horizon tip; see
    :func:`oven_facing_azimuth_deg`).
    """

    elevation_min_deg: float = 0.0
    elevation_max_deg: float = 90.0
    azimuth_min_deg: float = -150.0
    azimuth_max_deg: float = 150.0


def mount_rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """
    Active rotation **body → world**: ``p_W = R @ p_B``.

    ``R = R_z(−az) @ R_x(el − 90°)`` so that:

    * ``(0, 90)`` → identity (face-up, body = world)
    * ``el = 0`` → body ``+Z`` points to heading ``az`` (0=N, 90=E, …)
    """
    # Implement as R_z(phi) @ R_x(theta) with phi = -az, theta = el - 90.
    phi = np.deg2rad(-float(azimuth_deg))
    theta = np.deg2rad(float(elevation_deg) - 90.0)
    cx, sx = np.cos(theta), np.sin(theta)
    cz, sz = np.cos(phi), np.sin(phi)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return r_z @ r_x


def normalize_mount_az_el(az_deg: float, el_deg: float) -> tuple[float, float]:
    """Wrap azimuth to ``[0, 360)`` and clip elevation to ``[-90, 90]`` (pre-limit normalize)."""
    az = float(az_deg % 360.0)
    el = float(el_deg)
    if not np.isfinite(el) or abs(el) > 720.0:
        el = 45.0
    el = float(np.clip(el, -90.0, 90.0))
    return az, el


def wrapped_azimuth_delta_deg(a_deg: float, b_deg: float) -> float:
    """Signed shortest azimuth difference ``a - b`` in ``(-180, 180]``."""
    return ((float(a_deg) - float(b_deg) + 180.0) % 360.0) - 180.0


def dual_mount_angles(azimuth_deg: float, elevation_deg: float) -> tuple[float, float]:
    """
    Optically equivalent alt-az branch: ``(az, el) ↔ (az+180°, 180°−el)``.

    Both map the same body ``+Z`` (and the same general body normal under the
    mount twist) to the same world normal; they differ by a tip past zenith.

    Elevation is **not** clipped into ``[-90, 90]``: for ``el ∈ (0, 90)`` the dual
    has ``el ∈ (90, 180)``. Callers with ``elevation_max_deg = 90`` should treat
    that dual as out of range rather than clipping it to face-up (which is not
    optically equivalent).
    """
    return float(azimuth_deg + 180.0) % 360.0, 180.0 - float(elevation_deg)


def oven_facing_azimuth_deg(mount_world: np.ndarray, absorber_world: np.ndarray) -> float:
    """
    Absolute mount azimuth [deg] that aims toward the absorber at the horizon.

    At ``elevation = 0°``, body ``+Z`` maps to horizontal
    ``(sin(az), cos(az), 0)`` (0=N, 90=E). That direction is aligned with the
    horizontal ``mount → absorber`` vector.
    """
    d = np.asarray(absorber_world, dtype=float).reshape(3) - np.asarray(mount_world, dtype=float).reshape(3)
    return float(np.rad2deg(np.arctan2(d[0], d[1]))) % 360.0


def relative_azimuth_deg(absolute_azimuth_deg: float, oven_facing_azimuth_deg: float) -> float:
    """Mount azimuth relative to oven-facing, in ``(-180, 180]``."""
    return wrapped_azimuth_delta_deg(absolute_azimuth_deg, oven_facing_azimuth_deg)


def limited_azimuth_error_deg(
    target_azimuth_deg: float,
    position_azimuth_deg: float,
    *,
    oven_facing_azimuth_deg: float,
) -> float:
    """Signed azimuth error that stays inside the joint travel window.

    Unlike a shortest-path wrap to ``(-180, 180]``, this subtracts relative
    azimuths so motion from ``-150°`` to ``+149°`` goes through ``0°`` instead of
    the forbidden back side near ``±180°``.
    """
    target_rel = relative_azimuth_deg(target_azimuth_deg, oven_facing_azimuth_deg)
    position_rel = relative_azimuth_deg(position_azimuth_deg, oven_facing_azimuth_deg)
    return float(target_rel - position_rel)


def within_mount_joint_limits(
    azimuth_deg: float,
    elevation_deg: float,
    *,
    oven_facing_azimuth_deg: float,
    limits: MountJointLimits,
    eps_deg: float = 1e-6,
) -> bool:
    # Do not clip elevation into range before testing — that would treat a past-zenith
    # dual (e.g. el=135°) as face-up (el=90°) and falsely accept it.
    az = float(azimuth_deg % 360.0)
    el = float(elevation_deg)
    if not np.isfinite(el):
        return False
    if el < limits.elevation_min_deg - eps_deg or el > limits.elevation_max_deg + eps_deg:
        return False
    rel = relative_azimuth_deg(az, oven_facing_azimuth_deg)
    return limits.azimuth_min_deg - eps_deg <= rel <= limits.azimuth_max_deg + eps_deg


def apply_mount_joint_limits(
    azimuth_deg: float,
    elevation_deg: float,
    *,
    mount_world: np.ndarray,
    absorber_world: np.ndarray,
    limits: MountJointLimits | None = None,
) -> tuple[float, float]:
    """
    Choose ``(az, el)`` or its dual so the pose lies in the joint limits when possible.

    Prefer the in-limit candidate with elevation in range and smallest ``|relative az|``.
    If neither dual is valid, clamp elevation and relative azimuth into the box.
    """
    lim = limits or MountJointLimits()
    oven_az = oven_facing_azimuth_deg(mount_world, absorber_world)
    # Keep the primary elev unclipped so dual math stays exact; only wrap azimuth.
    az0 = float(azimuth_deg % 360.0)
    el0 = float(elevation_deg)
    if not np.isfinite(el0):
        el0 = 45.0
    candidates = [(az0, el0), dual_mount_angles(az0, el0)]
    valid = [
        c
        for c in candidates
        if within_mount_joint_limits(c[0], c[1], oven_facing_azimuth_deg=oven_az, limits=lim)
    ]
    if valid:
        def score(c: tuple[float, float]) -> tuple[float, float]:
            # Prefer the in-range branch with elevation closer to mid-travel, then
            # smaller |rel az|. Do not prefer zenith — that selected a clipped dual.
            mid = 0.5 * (lim.elevation_min_deg + lim.elevation_max_deg)
            return (abs(c[1] - mid), abs(relative_azimuth_deg(c[0], oven_az)))

        az, el = min(valid, key=score)
        return float(az % 360.0), float(el)

    return clamp_to_mount_joint_limits(
        az0, el0, oven_facing_azimuth_deg=oven_az, limits=lim
    )


def clamp_to_mount_joint_limits(
    azimuth_deg: float,
    elevation_deg: float,
    *,
    oven_facing_azimuth_deg: float,
    limits: MountJointLimits | None = None,
) -> tuple[float, float]:
    """Clip elevation and relative azimuth into the joint-limit box (no dual flip)."""
    lim = limits or MountJointLimits()
    az0, el0 = normalize_mount_az_el(azimuth_deg, elevation_deg)
    el = float(np.clip(el0, lim.elevation_min_deg, lim.elevation_max_deg))
    rel = float(
        np.clip(
            relative_azimuth_deg(az0, oven_facing_azimuth_deg),
            lim.azimuth_min_deg,
            lim.azimuth_max_deg,
        )
    )
    az = (float(oven_facing_azimuth_deg) + rel) % 360.0
    return normalize_mount_az_el(az, el)


def pivot_facet_normal_body(
    *,
    grid_nx: int,
    grid_ny: int,
    pitch_m: float,
    radius_of_curvature_m: float,
) -> np.ndarray:
    """
    Unit reflective normal of the center facet in mount body frame at face-up identity.

    Assumes an odd facet grid with the center tile at the mount pivot and spherical design
    about ``(0, 0, R)`` in body coordinates. At ``(az, el) = (0, 90)`` body = world.
    """
    _ = grid_nx, grid_ny, pitch_m  # grid dimensions affect off-center facets only
    center = np.array([0.0, 0.0, 0.0], dtype=float)
    sphere = np.array([0.0, 0.0, float(radius_of_curvature_m)], dtype=float)
    return normalize(sphere - center)


def _solve_rz_rx_align(body_normal: np.ndarray, target_normal_world: np.ndarray) -> tuple[float, float]:
    """Closed-form ``(phi_deg, theta_deg)`` for ``R_z(phi) @ R_x(theta) @ n_B ≈ n_W``."""
    nb = normalize(body_normal)
    nw = normalize(target_normal_world)
    nx, ny, nz = float(nb[0]), float(nb[1]), float(nb[2])
    tx, ty, tz = float(nw[0]), float(nw[1]), float(nw[2])

    def sqerr(phi_deg: float, theta_deg: float) -> float:
        # Evaluate with the intermediate (phi, theta) parameterization.
        d = _rz_rx(phi_deg, theta_deg) @ nb - nw
        return float(np.dot(d, d))

    def finish(phi_deg: float, theta_deg: float) -> tuple[float, float]:
        return float(phi_deg % 360.0), float(np.clip(theta_deg, -180.0, 180.0))

    r_yz = float(np.hypot(ny, nz))
    candidates: list[tuple[float, float]] = []

    if r_yz < 1e-12:
        vx, vy = nx, 0.0
        r_xy = float(np.hypot(tx, ty))
        if r_xy < 1e-12:
            return finish(0.0, 0.0)
        phi_deg = float(np.rad2deg(np.arctan2(ty, tx) - np.arctan2(vy, vx)))
        return finish(phi_deg, 0.0)

    phi_nb = float(np.arctan2(nz, ny))
    arg = float(np.clip(tz / r_yz, -1.0, 1.0))
    for delta in (float(np.arcsin(arg)), float(np.pi - np.arcsin(arg))):
        theta_rad = delta - phi_nb
        theta_deg = float(np.rad2deg(theta_rad))
        if not (-180.0 - 1e-9 <= theta_deg <= 180.0 + 1e-9):
            continue
        theta_deg = float(np.clip(theta_deg, -180.0, 180.0))
        elr = np.deg2rad(theta_deg)
        cr, sr = np.cos(elr), np.sin(elr)
        vx = nx
        vy = cr * ny - sr * nz
        vz = sr * ny + cr * nz
        r_xy_v = float(np.hypot(vx, vy))
        r_xy_t = float(np.hypot(tx, ty))
        if r_xy_v < 1e-12 or r_xy_t < 1e-12:
            if abs(vz - tz) < 1e-6 and r_xy_t < 1e-12:
                candidates.append((0.0, theta_deg))
            continue
        phi_deg = float(np.rad2deg(np.arctan2(ty, tx) - np.arctan2(vy, vx)))
        candidates.append((phi_deg, theta_deg))

    if not candidates:
        theta_cands = [-90.0, 90.0, 0.0, -180.0, 180.0]
        theta_crit = float(np.rad2deg(np.arctan2(ny, nz)))
        for shift in (-360.0, 0.0, 360.0):
            e = theta_crit + shift
            if -180.0 <= e <= 180.0:
                theta_cands.append(e)
        best_phi_f, best_theta_f, best_e_f = 0.0, 0.0, 1e30
        for theta_deg in theta_cands:
            elr = np.deg2rad(float(theta_deg))
            cr, sr = np.cos(elr), np.sin(elr)
            vy = cr * ny - sr * nz
            vx = nx
            r_xy_v = float(np.hypot(vx, vy))
            r_xy_t = float(np.hypot(tx, ty))
            if r_xy_v < 1e-12:
                phi_deg = 0.0
            else:
                phi_deg = float(np.rad2deg(np.arctan2(ty, tx) - np.arctan2(vy, vx)))
            e = sqerr(phi_deg, theta_deg)
            if e < best_e_f:
                best_e_f, best_phi_f, best_theta_f = e, phi_deg, theta_deg
        return finish(best_phi_f, best_theta_f)

    best_phi, best_theta = candidates[0]
    best_e = sqerr(best_phi, best_theta)
    for phi_deg, theta_deg in candidates[1:]:
        e = sqerr(phi_deg, theta_deg)
        if e < best_e:
            best_e, best_phi, best_theta = e, phi_deg, theta_deg
    return finish(best_phi, best_theta)


def _rz_rx(phi_deg: float, theta_deg: float) -> np.ndarray:
    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    cx, sx = np.cos(theta), np.sin(theta)
    cz, sz = np.cos(phi), np.sin(phi)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return r_z @ r_x


def mount_az_el_align_body_normal_to_world(
    body_normal: np.ndarray,
    target_normal_world: np.ndarray,
) -> tuple[float, float]:
    """
    Solve ``(azimuth_deg, elevation_deg)`` so ``R_mount(az, el) @ n_B ≈ n_W``.

    Uses ``R = R_z(−az) @ R_x(el − 90°)``. Callers should pass the result through
    :func:`apply_mount_joint_limits` to resolve the dual-branch ambiguity.
    """
    nb = normalize(body_normal)
    nw = normalize(target_normal_world)

    def sqerr(az_deg: float, el_deg: float) -> float:
        d = mount_rotation_matrix(az_deg, el_deg) @ nb - nw
        return float(np.dot(d, d))

    # Fast closed form when the body axis is +Z (center facet / identity normal).
    if abs(nb[0]) < 1e-8 and abs(nb[1]) < 1e-8 and nb[2] > 0.999:
        el = float(np.rad2deg(np.arcsin(np.clip(nw[2], -1.0, 1.0))))
        az = float(np.rad2deg(np.arctan2(nw[0], nw[1]))) % 360.0
        return normalize_mount_az_el(az, el)

    # General body normal: seed from R_z@R_x IK, then dense search (2-DOF fit).
    candidates: list[tuple[float, float]] = []
    phi0, theta0 = _solve_rz_rx_align(nb, nw)
    for phi_deg, theta_deg in ((phi0, theta0), (phi0 + 180.0, -theta0), (phi0 - 180.0, -theta0)):
        az = float((-phi_deg) % 360.0)
        el = float(theta_deg + 90.0)
        while el > 90.0 + 1e-9:
            az = (az + 180.0) % 360.0
            el = 180.0 - el
        while el < -90.0 - 1e-9:
            az = (az + 180.0) % 360.0
            el = -180.0 - el
        candidates.append(normalize_mount_az_el(az, el))
        candidates.append(dual_mount_angles(az, el))

    best = candidates[0]
    best_e = sqerr(*best)
    for el in np.linspace(-90.0, 90.0, 37):
        for az in np.linspace(0.0, 360.0, 73, endpoint=False):
            e = sqerr(az, el)
            if e < best_e:
                best_e, best = e, (float(az), float(el))

    # Local polish around the best grid / seed hit.
    az0, el0 = best
    for el in np.linspace(el0 - 5.0, el0 + 5.0, 21):
        if el < -90.0 or el > 90.0:
            continue
        for az in np.linspace(az0 - 5.0, az0 + 5.0, 21):
            e = sqerr(az % 360.0, el)
            if e < best_e:
                best_e, best = e, normalize_mount_az_el(az, el)
    return best


def facet_normal_world(azimuth_deg: float, elevation_deg: float, pivot_normal_body: np.ndarray) -> np.ndarray:
    """World-frame pivot facet normal at the given mount angles."""
    return normalize(mount_rotation_matrix(azimuth_deg, elevation_deg) @ np.asarray(pivot_normal_body, dtype=float).reshape(3))


def heading_and_tilt_from_normal(normal_world: np.ndarray) -> tuple[float, float]:
    """
    Spherical heading of a unit direction in world frame.

    Returns ``(azimuth_deg, elevation_from_horizon_deg)`` — display angles using the
    same astronomical sense as mount commands (0=N/horizon, 90=E/zenith).
    """
    n = normalize(normal_world)
    azimuth_deg = float(np.rad2deg(np.arctan2(n[0], n[1]))) % 360.0
    elevation_deg = float(np.rad2deg(np.arcsin(np.clip(n[2], -1.0, 1.0))))
    return azimuth_deg, elevation_deg
