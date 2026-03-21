from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pvlib.location import Location

from src.absorber import SolarAbsorber
from src.controller import Controller
from src.mirror import CylindricalMirror
from src.simulation import HotboxSimulation
from src.sun import SunModel
from src.visualizer import SceneVisualizer, build_day_delivered_power_figure

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

# Solar absorber: vertical rectangle, center at (0, 0, center_height); normal in horizontal plane.
# normal_angle_from_x_deg: 0° = +x (east), 90° = +y (north), 180° = −x (west), 270° = −y (south).
ABSORBER_WIDTH_M = 0.30
ABSORBER_HEIGHT_M = 0.30
ABSORBER_CENTER_HEIGHT_M = 1.20
ABSORBER_NORMAL_ANGLE_FROM_X_DEG = 90.0

SIM_SAMPLES_U = 250
SIM_SAMPLES_V = 250

# Site (must match SunModel in build_default_simulation)
SITE_LATITUDE_DEG = 40.7864
SITE_LONGITUDE_DEG = -119.2065
SITE_ALTITUDE_M = 1190.0

# Day curve: local sunrise → sunset on this calendar day (Pacific TZ), every N minutes.
DAY_CURVE_YEAR = 2026
DAY_CURVE_MONTH = 9
DAY_CURVE_DAY = 3
DAY_CURVE_TZ = ZoneInfo("America/Los_Angeles")
DAY_CURVE_STEP_MINUTES = 10


def local_times_sunrise_to_sunset(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    year: int,
    month: int,
    day: int,
    tz: ZoneInfo,
    step_minutes: int,
) -> tuple[list[datetime], datetime | None, datetime | None]:
    """
    10-minute samples from first step on/after sunrise through last on/before sunset.
    Sunrise/sunset from pvlib SPA for the given site and local date.
    """
    tz_key = tz.key
    loc = Location(latitude_deg, longitude_deg, tz_key, altitude=altitude_m)
    day_midnight = pd.Timestamp(year=year, month=month, day=day, tz=tz_key).normalize()
    idx = pd.DatetimeIndex([day_midnight])
    rs = loc.get_sun_rise_set_transit(idx, method="spa")
    sunrise_ts = rs["sunrise"].iloc[0]
    sunset_ts = rs["sunset"].iloc[0]
    if pd.isna(sunrise_ts) or pd.isna(sunset_ts):
        return [], None, None

    sunrise_ts = sunrise_ts.floor("s")
    sunset_ts = sunset_ts.floor("s")
    sunrise = sunrise_ts.to_pydatetime()
    sunset = sunset_ts.to_pydatetime()
    step = timedelta(minutes=step_minutes)
    midnight = datetime(year, month, day, 0, 0, 0, tzinfo=tz)
    t = midnight
    while t < sunrise:
        t += step
    out: list[datetime] = []
    while t <= sunset:
        out.append(t)
        t += step
    return out, sunrise, sunset


def simulate_delivered_power_over_times(
    sim: HotboxSimulation,
    controller: Controller,
    times: list[datetime],
) -> tuple[list[datetime], list[float]]:
    powers: list[float] = []
    for when in times:
        controller.apply_for_time(
            when_utc=when,
            sun=sim.sun,
            absorber_center=sim.absorber.center,
            mirrors=sim.mirrors,
        )
        result = sim.run(when)
        powers.append(result.total_delivered_power_w)
    return times, powers


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
        latitude_deg=SITE_LATITUDE_DEG,
        longitude_deg=SITE_LONGITUDE_DEG,
        altitude_m=SITE_ALTITUDE_M,
        dni_w_per_m2=1000.0,
    )
    absorber = SolarAbsorber(
        width_m=ABSORBER_WIDTH_M,
        height_m=ABSORBER_HEIGHT_M,
        center_height_m=ABSORBER_CENTER_HEIGHT_M,
        normal_angle_from_x_deg=ABSORBER_NORMAL_ANGLE_FROM_X_DEG,
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

    day_times, sr, ss = local_times_sunrise_to_sunset(
        SITE_LATITUDE_DEG,
        SITE_LONGITUDE_DEG,
        SITE_ALTITUDE_M,
        DAY_CURVE_YEAR,
        DAY_CURVE_MONTH,
        DAY_CURVE_DAY,
        DAY_CURVE_TZ,
        DAY_CURVE_STEP_MINUTES,
    )
    if sr is not None and ss is not None:
        print(
            f"Day curve: sunrise {sr.strftime('%Y-%m-%d %H:%M:%S %Z')}, "
            f"sunset {ss.strftime('%Y-%m-%d %H:%M:%S %Z')} ({len(day_times)} samples)"
        )
    if day_times:
        _, day_powers = simulate_delivered_power_over_times(sim, controller, day_times)
        sr_s = sr.strftime("%H:%M") if sr else "?"
        ss_s = ss.strftime("%H:%M") if ss else "?"
        day_fig = build_day_delivered_power_figure(
            day_times,
            day_powers,
            title=(
                f"Delivered power — {DAY_CURVE_MONTH}/{DAY_CURVE_DAY}/{DAY_CURVE_YEAR} "
                f"(sunrise–sunset {sr_s}–{ss_s} {DAY_CURVE_TZ.key}, "
                f"every {DAY_CURVE_STEP_MINUTES} min)"
            ),
        )
        day_fig.show(config=_plotly_config)
    else:
        print("Day curve: no daylight samples (polar night or missing rise/set).")


if __name__ == "__main__":
    main()
