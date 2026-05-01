from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pvlib.location import Location

from src.absorber import SolarAbsorber
from src.controller import mirror_orientations_for_time
from src.flat_mirror_grid import AltAzFlatMirrorGrid
from src.simulation import HotboxSimulation, SimulationResult
from src.sun import SunModel
from src.visualizer import SceneVisualizer, build_day_delivered_power_figure

# --- Rigid flat mirror grid (one alt-az mount) ---
MIRROR_GRID_NX = 3
MIRROR_GRID_NY = 5
MIRROR_TILE_SIDE_M = 0.254
MIRROR_GRID_PITCH_M = 0.254 + 0.01  # center-to-center spacing [m]
MIRROR_ASSEMBLY_COUNT = 3
MIRROR_LOCATION_RING_RADIUS_M = 3.0  # mount pivot offset from absorber along absorber normal
MIRROR_ASSEMBLY_SPACING_M = 1.0  # fixed center-to-center spacing between assemblies [m]
MIRROR_GRID_MOUNT_HEIGHT_M = 0.85  # mount pivot z [m]; facet centers lie on the design tilted plane through the pivot

# Solar absorber: vertical rectangle, center at (0, 0, center_height); normal in horizontal plane.
# normal_angle_from_x_deg: 0° = +x (east), 90° = +y (north), 180° = −x (west), 270° = −y (south).
ABSORBER_WIDTH_M = 0.40
ABSORBER_HEIGHT_M = 0.40
ABSORBER_CENTER_HEIGHT_M = 1.0
ABSORBER_NORMAL_ANGLE_FROM_X_DEG = 90.0

SIM_SAMPLES_U = 100
SIM_SAMPLES_V = 100

# Higher ray count for the multi-panel absorber spot figure (main scene uses SIM_SAMPLES_*).
SPOT_GRID_NUM_PANELS = 12
SPOT_GRID_NCOLS = 4
SPOT_GRID_SAMPLES_U = 400
SPOT_GRID_SAMPLES_V = 400
SPOT_GRID_BINS = 80

# Site (must match SunModel in build_default_simulation)
SITE_LATITUDE_DEG = 40.7864
SITE_LONGITUDE_DEG = -119.2065
SITE_ALTITUDE_M = 1190.0

# Day curve: local sunrise → sunset (Pacific TZ), every N minutes. Int or same-length lists.
DAY_CURVE_YEAR = 2026
DAY_CURVE_MONTH = [8, 9]
DAY_CURVE_DAY = [30, 7]
DAY_CURVE_TZ = ZoneInfo("America/Los_Angeles")
DAY_CURVE_STEP_MINUTES = 20

# Facet tilts are chosen so each center ray reflects to the absorber at this instant (mount at 0,0).
MIRROR_GRID_DESIGN_WHEN = datetime(2026, 8, 31, 14, 0, 0, tzinfo=DAY_CURVE_TZ)

# Local wall time for the 3D scene / absorber spot figures and printed snapshot
# (mount solve, mirror angles, ray bundle). Independent of DAY_CURVE_* curve list.
SCENE_VIS_WHEN = datetime(2026, 9, 7, 9, 0, 0, tzinfo=DAY_CURVE_TZ)


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


def day_curve_month_day_pairs(
    month: int | list[int] | tuple[int, ...],
    day: int | list[int] | tuple[int, ...],
) -> list[tuple[int, int]]:
    """Expand scalar or parallel lists into (month, day) pairs for each simulated curve."""
    months = list(month) if isinstance(month, (list, tuple)) else [month]
    days = list(day) if isinstance(day, (list, tuple)) else [day]
    if len(months) == 1 and len(days) != 1:
        months = months * len(days)
    elif len(days) == 1 and len(months) != 1:
        days = days * len(months)
    if len(months) != len(days):
        raise ValueError(
            "DAY_CURVE_MONTH and DAY_CURVE_DAY must be the same length, "
            "or one must be a single int to broadcast."
        )
    return list(zip(months, days))


def spot_pattern_sample_times(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    year: int,
    month: int,
    day: int,
    tz: ZoneInfo,
    step_minutes: int,
    num_panels: int,
) -> list[datetime]:
    """
    ``num_panels`` local times spread across sunrise–sunset for spot-pattern visualization.
    Falls back to empty if there is no daylight interval.
    """
    day_times, _, _ = local_times_sunrise_to_sunset(
        latitude_deg,
        longitude_deg,
        altitude_m,
        year,
        month,
        day,
        tz,
        step_minutes,
    )
    if not day_times:
        return []
    if len(day_times) <= num_panels:
        return list(day_times)
    idx = np.linspace(0, len(day_times) - 1, num=num_panels, dtype=int)
    idx = np.unique(idx)
    return [day_times[int(j)] for j in idx]


def simulate_delivered_power_over_times(
    sim: HotboxSimulation,
    times: list[datetime],
) -> tuple[list[datetime], list[float], list[float], list[list[tuple[float, float]]]]:
    """For each time: total delivered power, total power hitting mirrors, orientations [deg]."""
    delivered_w: list[float] = []
    intercepted_w: list[float] = []
    orientations_per_time: list[list[tuple[float, float]]] = []
    for when in times:
        az_el = mirror_orientations_for_time(
            when_utc=when,
            sun=sim.sun,
            absorber_center=sim.absorber.center,
            mirrors=sim.mirrors,
            absorber=sim.absorber,
        )
        result = sim.run(when)
        delivered_w.append(result.total_delivered_power_w)
        intercepted_w.append(float(sum(m.intercepted_power_w for m in result.per_mirror)))
        orientations_per_time.append(list(az_el))
    return times, delivered_w, intercepted_w, orientations_per_time


def build_default_simulation() -> HotboxSimulation:
    sun = SunModel(
        latitude_deg=SITE_LATITUDE_DEG,
        longitude_deg=SITE_LONGITUDE_DEG,
        altitude_m=SITE_ALTITUDE_M,
    )
    absorber = SolarAbsorber(
        width_m=ABSORBER_WIDTH_M,
        height_m=ABSORBER_HEIGHT_M,
        center_height_m=ABSORBER_CENTER_HEIGHT_M,
        normal_angle_from_x_deg=ABSORBER_NORMAL_ANGLE_FROM_X_DEG,
    )

    a = np.asarray(absorber.center, dtype=float)
    fw = np.asarray(absorber.normal, dtype=float)
    fw_xy = np.array([fw[0], fw[1], 0.0], dtype=float)
    fw_xy /= max(float(np.linalg.norm(fw_xy)), 1e-12)
    # Horizontal tangent direction to lay out multiple assemblies on one side of the absorber.
    tw_xy = np.array([-fw_xy[1], fw_xy[0], 0.0], dtype=float)
    base_mount = a + MIRROR_LOCATION_RING_RADIUS_M * fw_xy
    tile_half_m = 0.5 * MIRROR_TILE_SIDE_M
    pitch_m = MIRROR_GRID_PITCH_M
    grids: list[AltAzFlatMirrorGrid] = []
    for i in range(MIRROR_ASSEMBLY_COUNT):
        offset = (i - 0.5 * (MIRROR_ASSEMBLY_COUNT - 1)) * MIRROR_ASSEMBLY_SPACING_M
        mount_world = np.array(
            [
                base_mount[0] + offset * tw_xy[0],
                base_mount[1] + offset * tw_xy[1],
                MIRROR_GRID_MOUNT_HEIGHT_M,
            ],
            dtype=float,
        )
        grids.append(
            AltAzFlatMirrorGrid(
                mount_world=mount_world,
                design_when_utc=MIRROR_GRID_DESIGN_WHEN,
                absorber_center=a.copy(),
                grid_nx=MIRROR_GRID_NX,
                grid_ny=MIRROR_GRID_NY,
                pitch_m=pitch_m,
                tile_half_m=tile_half_m,
                sun=sun,
            )
        )

    return HotboxSimulation(
        sun=sun,
        absorber=absorber,
        mirrors=list(grids),
        samples_u=SIM_SAMPLES_U,
        samples_v=SIM_SAMPLES_V,
    )


def main() -> None:
    sim = build_default_simulation()
    print(f"Mirror assemblies: {len(sim.mirrors)}")
    day_specs = day_curve_month_day_pairs(DAY_CURVE_MONTH, DAY_CURVE_DAY)
    when = SCENE_VIS_WHEN

    orientations = mirror_orientations_for_time(
        when_utc=when,
        sun=sim.sun,
        absorber_center=sim.absorber.center,
        mirrors=sim.mirrors,
        absorber=sim.absorber,
    )

    result = sim.run(when)

    print(f"Sun ray direction (world xyz): {result.sun_direction}")
    for idx, (az, tilt) in enumerate(orientations):
        print(
            f"Mirror {idx} mount: azimuth={az:.2f} deg, "
            f"lattice tilt={tilt:.2f} deg (0=vertical plane, 90=horizontal toward zenith)"
        )
    for idx, mr in enumerate(result.per_mirror):
        print(
            f"Mirror {idx}: incident={mr.incident_power_w:.1f} W, "
            f"intercepted={mr.intercepted_power_w:.1f} W, delivered={mr.delivered_power_w:.1f} W"
        )
    print(f"Total delivered power: {result.total_delivered_power_w:.1f} W")

    viz = SceneVisualizer(sim.absorber, sim.mirrors)
    scene_fig = viz.build_scene_figure(result, scene_when_local=when)

    # Spot grid: same calendar day as SCENE_VIS_WHEN, sunrise→sunset (see SPOT_GRID_*).
    spot_times = spot_pattern_sample_times(
        SITE_LATITUDE_DEG,
        SITE_LONGITUDE_DEG,
        SITE_ALTITUDE_M,
        SCENE_VIS_WHEN.year,
        SCENE_VIS_WHEN.month,
        SCENE_VIS_WHEN.day,
        DAY_CURVE_TZ,
        DAY_CURVE_STEP_MINUTES,
        SPOT_GRID_NUM_PANELS,
    )
    if not spot_times:
        spot_times = [when]
    spot_labeled: list[tuple[str, SimulationResult]] = []
    for t_spot in spot_times:
        mirror_orientations_for_time(
            when_utc=t_spot,
            sun=sim.sun,
            absorber_center=sim.absorber.center,
            mirrors=sim.mirrors,
            absorber=sim.absorber,
        )
        r_spot = sim.run(
            t_spot,
            samples_u=SPOT_GRID_SAMPLES_U,
            samples_v=SPOT_GRID_SAMPLES_V,
        )
        label = t_spot.strftime("%H:%M")
        spot_labeled.append((label, r_spot))
    spot_fig = viz.build_absorber_spot_figure_grid(
        spot_labeled,
        bins=SPOT_GRID_BINS,
        ncols=SPOT_GRID_NCOLS,
    )
    # Prevent the browser from resizing the plot div (which distorts 3D aspect).
    _plotly_config = {"responsive": False}
    scene_fig.show(config=_plotly_config)
    spot_fig.show(config=_plotly_config)

    day_series: list[
        tuple[str, list[datetime], list[float], list[float], list[list[tuple[float, float]]]]
    ] = []
    single_curve_sr_ss: tuple[datetime | None, datetime | None] | None = None
    for month_i, day_i in day_specs:
        day_times, sr, ss = local_times_sunrise_to_sunset(
            SITE_LATITUDE_DEG,
            SITE_LONGITUDE_DEG,
            SITE_ALTITUDE_M,
            DAY_CURVE_YEAR,
            month_i,
            day_i,
            DAY_CURVE_TZ,
            DAY_CURVE_STEP_MINUTES,
        )
        label = f"{month_i}/{day_i}/{DAY_CURVE_YEAR}"
        if sr is not None and ss is not None:
            print(
                f"Day curve {label}: sunrise {sr.strftime('%Y-%m-%d %H:%M:%S %Z')}, "
                f"sunset {ss.strftime('%Y-%m-%d %H:%M:%S %Z')} ({len(day_times)} samples)"
            )
        else:
            print(f"Day curve {label}: no sunrise/sunset (polar night or missing rise/set).")
        if day_times:
            _, day_delivered, day_intercepted, day_orients = simulate_delivered_power_over_times(
                sim, day_times
            )
            day_series.append((label, day_times, day_delivered, day_intercepted, day_orients))
            if len(day_specs) == 1:
                single_curve_sr_ss = (sr, ss)

    if day_series:
        x_axis_title = f"Local time ({DAY_CURVE_TZ.key})"
        if len(day_series) == 1 and len(day_specs) == 1 and single_curve_sr_ss is not None:
            sr0, ss0 = single_curve_sr_ss
            month_i, day_i = day_specs[0]
            sr_s = sr0.strftime("%H:%M") if sr0 else "?"
            ss_s = ss0.strftime("%H:%M") if ss0 else "?"
            day_title = (
                f"Delivered & mirror-intercepted power — {month_i}/{day_i}/{DAY_CURVE_YEAR} "
                f"(sunrise–sunset {sr_s}–{ss_s}, every {DAY_CURVE_STEP_MINUTES} min)"
            )
        else:
            dates_s = ", ".join(name for name, _, _, _, _ in day_series)
            day_title = (
                f"Delivered & mirror-intercepted power — {DAY_CURVE_YEAR} ({dates_s}), "
                f"sunrise–sunset local, every {DAY_CURVE_STEP_MINUTES} min"
            )
        day_fig = build_day_delivered_power_figure(
            day_series,
            title=day_title,
            x_axis_title=(
                "Local time of day [h] (wall clock)"
                if len(day_series) > 1
                else x_axis_title
            ),
            same_day_time_scale=len(day_series) > 1,
        )
        day_fig.show(config=_plotly_config)
    elif day_specs:
        print("Day curve: no daylight samples for any selected day.")


if __name__ == "__main__":
    main()
