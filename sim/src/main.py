from __future__ import annotations

from datetime import datetime, timezone

from src.absorber import SolarAbsorber
from src.controller import Controller
from src.mirror import CylindricalMirror
from src.simulation import HotboxSimulation
from src.sun import SunModel
from src.visualizer import SceneVisualizer


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
    mirrors = [
        CylindricalMirror(
            radius_of_curvature_m=1.5,
            width_m=0.61,  # ~24 in
            height_m=1.22,  # ~48 in
            post_height_m=0.95,
            back_to_rotation_offset_m=0.10,
            position_xy_m=(3.0, -0.7),
            azimuth_deg=248.0,
            elevation_deg=30.0,
        ),
        CylindricalMirror(
            radius_of_curvature_m=1.5,
            width_m=0.61,
            height_m=1.22,
            post_height_m=0.95,
            back_to_rotation_offset_m=0.10,
            position_xy_m=(3.2, 0.8),
            azimuth_deg=235.0,
            elevation_deg=31.0,
        ),
    ]
    return HotboxSimulation(sun=sun, absorber=absorber, mirrors=mirrors, samples_u=65, samples_v=65)


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
    scene_fig.show()
    spot_fig.show()


if __name__ == "__main__":
    main()
