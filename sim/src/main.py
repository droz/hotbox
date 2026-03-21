from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src.absorber import SolarAbsorber
from src.controller import Controller
from src.mirror import CylindricalMirror
from src.simulation import HotboxSimulation
from src.sun import SunModel
from src.visualizer import SceneVisualizer

# --- Layout (automated mirror arc) ---
NUM_MIRRORS = 3
MIRROR_RADIUS_OF_CURVATURE_M = 10
# Arc length along the circle (radius R/2) between adjacent mirror rotation points.
MIRROR_SPACING_M = 0.75

# Mirror sheet / mount (shared by all mirrors in this demo)
MIRROR_WIDTH_M = 0.305  # ~12 in
MIRROR_HEIGHT_M = 1.22  # ~48 in
MIRROR_POST_HEIGHT_M = 0.95
MIRROR_BACK_TO_ROTATION_OFFSET_M = 0.10

SIM_SAMPLES_U = 65
SIM_SAMPLES_V = 65


def mirror_rotation_xy_on_arc(
    absorber: SolarAbsorber,
    num_mirrors: int,
    radius_of_curvature_m: float,
    spacing_along_arc_m: float,
) -> list[tuple[float, float]]:
    """
    Place mirror rotation points on a horizontal arc in front of the absorber.

    All points lie on a circle of radius R/2 in the horizontal plane, centered on
    the absorber axis (x,y of absorber center). The arc is symmetric about the
    absorber outward normal; spacing is uniform arc length between neighbors.
    """
    r = 0.5 * radius_of_curvature_m
    center_xy = absorber.center[:2]
    forward_xy = absorber.normal[:2]
    perp_xy = absorber.horizontal_axis[:2]

    if num_mirrors < 1:
        return []

    if num_mirrors == 1:
        thetas = np.array([0.0], dtype=float)
    else:
        arc_total = (num_mirrors - 1) * spacing_along_arc_m
        phi = arc_total / r
        thetas = np.linspace(-0.5 * phi, 0.5 * phi, num_mirrors)

    positions: list[tuple[float, float]] = []
    for th in thetas:
        direction_xy = np.cos(th) * forward_xy + np.sin(th) * perp_xy
        offset = r * direction_xy
        xy = center_xy + offset
        positions.append((float(xy[0]), float(xy[1])))

    return positions


def build_default_simulation() -> HotboxSimulation:
    sun = SunModel(
        latitude_deg=40.7864,  # Black Rock City area
        longitude_deg=-119.2065,
        altitude_m=1190.0,
        dni_w_per_m2=1000.0,
    )
    absorber = SolarAbsorber(
        width_m=0.230,
        height_m=0.230,
        center_height_m=1.20,
        normal_angle_from_x_deg=180.0,  # Facing approximately -x
    )

    xy_list = mirror_rotation_xy_on_arc(
        absorber,
        NUM_MIRRORS,
        MIRROR_RADIUS_OF_CURVATURE_M,
        MIRROR_SPACING_M,
    )
    mirrors = [
        CylindricalMirror(
            radius_of_curvature_m=MIRROR_RADIUS_OF_CURVATURE_M,
            width_m=MIRROR_WIDTH_M,
            height_m=MIRROR_HEIGHT_M,
            post_height_m=MIRROR_POST_HEIGHT_M,
            back_to_rotation_offset_m=MIRROR_BACK_TO_ROTATION_OFFSET_M,
            position_xy_m=xy,
            azimuth_deg=0.0,
            elevation_deg=0.0,
        )
        for xy in xy_list
    ]

    return HotboxSimulation(
        sun=sun,
        absorber=absorber,
        mirrors=mirrors,
        samples_u=SIM_SAMPLES_U,
        samples_v=SIM_SAMPLES_V,
    )


def main() -> None:
    sim = build_default_simulation()
    when = datetime(2026, 8, 28, 20, 0, 0, tzinfo=timezone.utc)

    mirror_rel_positions = [m.rotation_point - sim.absorber.center for m in sim.mirrors]
    absorber_orientation_from_north_deg = 90.0 - sim.absorber.normal_angle_from_x_deg
    controller = Controller(
        mirror_positions_relative_to_absorber=mirror_rel_positions,
        absorber_orientation_relative_to_north_deg=absorber_orientation_from_north_deg,
    )
    orientations = controller.apply_for_time(
        when_utc=when,
        sun=sim.sun,
        absorber_center=sim.absorber.center,
        mirrors=sim.mirrors,
    )

    result = sim.run(when)

    print(f"Sun ray direction (world xyz): {result.sun_direction}")
    for idx, (az, el) in enumerate(orientations):
        print(f"Controller mirror {idx}: azimuth={az:.2f} deg, elevation={el:.2f} deg")
    for idx, mr in enumerate(result.per_mirror):
        print(
            f"Mirror {idx}: incident={mr.incident_power_w:.1f} W, "
            f"intercepted={mr.intercepted_power_w:.1f} W, delivered={mr.delivered_power_w:.1f} W"
        )
    print(f"Total delivered power: {result.total_delivered_power_w:.1f} W")

    viz = SceneVisualizer(sim.absorber, sim.mirrors)
    scene_fig = viz.build_scene_figure(result, ray_stride=100)
    spot_fig = viz.build_absorber_spot_figure(result)
    # Prevent the browser from resizing the plot div (which distorts 3D aspect).
    _plotly_config = {"responsive": False}
    scene_fig.show(config=_plotly_config)
    spot_fig.show(config=_plotly_config)


if __name__ == "__main__":
    main()
