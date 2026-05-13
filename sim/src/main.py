from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pvlib.location import Location

from src.absorber import SolarAbsorber
from src.controller import mirror_orientations_for_time
from src.flat_mirror_grid import AltAzFlatMirrorGrid, FacetDesignStrategy
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
# Default **signed** distance along horizontal absorber normal ``fw_xy`` (same unit vector as
# from absorber to mirror ring) to the spherical **center of curvature** ``O``: ``O = a + dist * fw_xy``.
# Negative = ``O`` on the **opposite** side of the absorber from the mirrors (typical converging
# cap); positive = ``O`` past the absorber toward the mirror field (often divergent for sun from
# the opposite hemisphere).
MIRROR_SPHERICAL_FOCUS_DISTANCE_FROM_ABSORBER_M = -2.0 * MIRROR_LOCATION_RING_RADIUS_M
MIRROR_ASSEMBLY_SPACING_M = 1.0  # fixed center-to-center spacing between assemblies [m]
MIRROR_GRID_MOUNT_HEIGHT_M = 0.85  # mount pivot z [m]; facet centers lie on the design tilted plane through the pivot

# Solar absorber: vertical rectangle, center at (0, 0, center_height); normal in horizontal plane.
# normal_angle_from_x_deg: 0° = +x (east), 90° = +y (north), 180° = −x (west), 270° = −y (south).
ABSORBER_WIDTH_M = 0.40
ABSORBER_HEIGHT_M = 0.40
ABSORBER_CENTER_HEIGHT_M = 1.0
ABSORBER_NORMAL_ANGLE_FROM_X_DEG = 90.0

# Cell count along each axis on **each** square facet (rays per mirror ≈ grid_nx * grid_ny * U * V).
SIM_SAMPLES_U = 8
SIM_SAMPLES_V = 8

# Same per-facet cell counts for the multi-panel absorber spot figure.
SPOT_GRID_NUM_PANELS = 12
SPOT_GRID_NCOLS = 4
SPOT_GRID_SAMPLES_U = 32
SPOT_GRID_SAMPLES_V = 32
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

# Terminal progress: high-level phases (timed); mirror-level timings from HotboxSimulation.run.
SHOW_PROGRESS_STEPS = True
SHOW_MIRROR_TIMING = False


@contextmanager
def timed_step(label: str) -> None:
    if not SHOW_PROGRESS_STEPS:
        yield
        return
    print(f"[hotbox] {label} …", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[hotbox] {label} — done in {time.perf_counter() - t0:.3f}s", flush=True)


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
    *,
    progress_label: str = "day power curve",
    sim_verbose: bool = False,
) -> tuple[list[datetime], list[float], list[float], list[list[tuple[float, float]]]]:
    """For each time: total delivered power, total power hitting mirrors, orientations [deg]."""
    delivered_w: list[float] = []
    intercepted_w: list[float] = []
    orientations_per_time: list[list[tuple[float, float]]] = []
    n = len(times)
    if SHOW_PROGRESS_STEPS and n > 1:
        print(
            f"[hotbox] {progress_label}: {n} timesteps (mirror timing={'on' if sim_verbose else 'off'})",
            flush=True,
        )
    t_curve = time.perf_counter()
    report_every = max(1, n // 10) if n > 10 else 1
    for idx, when in enumerate(times):
        if SHOW_PROGRESS_STEPS and n > 1 and (
            idx % report_every == 0 or idx == n - 1
        ):
            print(
                f"[hotbox] {progress_label}: timestep {idx + 1}/{n} "
                f"{when.strftime('%Y-%m-%d %H:%M')} …",
                flush=True,
            )
        t_step = time.perf_counter()
        az_el = mirror_orientations_for_time(
            when_utc=when,
            sun=sim.sun,
            absorber_center=sim.absorber.center,
            mirrors=sim.mirrors,
            absorber=sim.absorber,
        )
        t_after_mount = time.perf_counter()
        result = sim.run(when, verbose=sim_verbose)
        t_after_ray = time.perf_counter()
        if SHOW_PROGRESS_STEPS and n > 1 and (
            idx % report_every == 0 or idx == n - 1
        ):
            dt_ori = t_after_mount - t_step
            dt_ray = t_after_ray - t_after_mount
            print(
                f"[hotbox] {progress_label}: timestep {idx + 1}/{n} "
                f"mount_solve={dt_ori:.3f}s raytrace={dt_ray:.3f}s "
                f"(step total {t_after_ray - t_step:.3f}s)",
                flush=True,
            )
        delivered_w.append(result.total_delivered_power_w)
        intercepted_w.append(float(sum(m.intercepted_power_w for m in result.per_mirror)))
        orientations_per_time.append(list(az_el))
    if SHOW_PROGRESS_STEPS and n > 1:
        print(
            f"[hotbox] {progress_label}: finished all {n} timesteps in "
            f"{time.perf_counter() - t_curve:.3f}s",
            flush=True,
        )
    return times, delivered_w, intercepted_w, orientations_per_time


def build_default_simulation(
    mirror_design: FacetDesignStrategy = "optimized",
    *,
    spherical_focus_distance_from_absorber_m: float | None = None,
) -> HotboxSimulation:
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
    if mirror_design == "spherical":
        dist_m = (
            spherical_focus_distance_from_absorber_m
            if spherical_focus_distance_from_absorber_m is not None
            else MIRROR_SPHERICAL_FOCUS_DISTANCE_FROM_ABSORBER_M
        )
        spherical_target_world = (a + dist_m * fw_xy).astype(float)
    else:
        spherical_target_world = None
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
                facet_design=mirror_design,
                spherical_target_world=spherical_target_world,
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
    parser = argparse.ArgumentParser(description="Hotbox heliostat-style ray simulation.")
    parser.add_argument(
        "--mirror-design",
        choices=("optimized", "spherical"),
        default="optimized",
        help=(
            'Facet canting: "optimized" — each facet specular toward the absorber at the design '
            'instant; "spherical" — facet normals are outward radials from a shared sphere center '
            "(see --spherical-focus-distance-m)."
        ),
    )
    parser.add_argument(
        "--spherical-focus-distance-m",
        type=float,
        default=None,
        metavar="M",
        help=(
            "Signed distance [m] from absorber center to sphere center of curvature along the "
            "horizontal absorber normal (same axis as mirror ring offset): **positive** = toward "
            "the mirror field, **negative** = opposite side of the absorber. "
            "Default for spherical design: −2× mirror ring radius. Ignored for optimized design."
        ),
    )
    args = parser.parse_args()
    mirror_design: FacetDesignStrategy = args.mirror_design
    with timed_step("Build default simulation (geometry + mirror grids)"):
        sim = build_default_simulation(
            mirror_design,
            spherical_focus_distance_from_absorber_m=args.spherical_focus_distance_m,
        )
    print(f"Mirror assemblies: {len(sim.mirrors)} (design={mirror_design})")
    if mirror_design == "spherical" and sim.mirrors:
        o = sim.mirrors[0].spherical_target_world
        if o is not None:
            print(f"Spherical center of curvature (world xyz): {o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f} m")
    day_specs = day_curve_month_day_pairs(DAY_CURVE_MONTH, DAY_CURVE_DAY)
    when = SCENE_VIS_WHEN

    with timed_step("Solve mount angles for scene snapshot"):
        orientations = mirror_orientations_for_time(
            when_utc=when,
            sun=sim.sun,
            absorber_center=sim.absorber.center,
            mirrors=sim.mirrors,
            absorber=sim.absorber,
        )

    with timed_step("Raytrace snapshot (scene time)"):
        result = sim.run(when, verbose=SHOW_MIRROR_TIMING)

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
    with timed_step("Build 3D scene figure (Plotly)"):
        scene_fig = viz.build_scene_figure(result, scene_when_local=when)

    # Spot grid: same calendar day as SCENE_VIS_WHEN, sunrise→sunset (see SPOT_GRID_*).
    with timed_step("Compute spot-pattern sample times"):
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
    with timed_step(f"Spot figure: raytrace {len(spot_times)} panel(s)"):
        for i_spot, t_spot in enumerate(spot_times):
            if SHOW_PROGRESS_STEPS and len(spot_times) > 1:
                print(
                    f"[hotbox] spot panel {i_spot + 1}/{len(spot_times)} "
                    f"{t_spot.strftime('%Y-%m-%d %H:%M')} …",
                    flush=True,
                )
            t_panel = time.perf_counter()
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
                verbose=SHOW_MIRROR_TIMING,
            )
            if SHOW_PROGRESS_STEPS and len(spot_times) > 1:
                print(
                    f"[hotbox] spot panel {i_spot + 1}/{len(spot_times)} "
                    f"— done in {time.perf_counter() - t_panel:.3f}s",
                    flush=True,
                )
            label = t_spot.strftime("%H:%M")
            spot_labeled.append((label, r_spot))
    with timed_step("Build absorber spot figure grid (Plotly)"):
        spot_fig = viz.build_absorber_spot_figure_grid(
            spot_labeled,
            bins=SPOT_GRID_BINS,
            ncols=SPOT_GRID_NCOLS,
        )
    # Prevent the browser from resizing the plot div (which distorts 3D aspect).
    _plotly_config = {"responsive": False}
    with timed_step("Open 3D scene in browser (Plotly)"):
        scene_fig.show(config=_plotly_config)
    with timed_step("Open spot figure in browser (Plotly)"):
        spot_fig.show(config=_plotly_config)

    day_series: list[
        tuple[str, list[datetime], list[float], list[float], list[list[tuple[float, float]]]]
    ] = []
    single_curve_sr_ss: tuple[datetime | None, datetime | None] | None = None
    for month_i, day_i in day_specs:
        with timed_step(f"Sunrise/sunset & timestep list for {month_i}/{day_i}/{DAY_CURVE_YEAR}"):
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
                sim,
                day_times,
                progress_label=f"Day curve {label}",
                sim_verbose=SHOW_MIRROR_TIMING,
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
        with timed_step("Build day power Plotly figure"):
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
        with timed_step("Open day power figure in browser (Plotly)"):
            day_fig.show(config=_plotly_config)
    elif day_specs:
        print("Day curve: no daylight samples for any selected day.")


if __name__ == "__main__":
    main()
