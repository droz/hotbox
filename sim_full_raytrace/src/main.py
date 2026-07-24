from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import time

import numpy as np
from hotbox_shared import (
    SitePose,
    SystemConstants,
    format_site_local,
    load_system_constants,
    local_times_sunrise_to_sunset,
    site_local_datetime,
)

from src.absorber import SolarAbsorber
from src.controller import mirror_orientations_for_time
from src.flat_mirror_grid import AltAzFlatMirrorGrid, grid_mount_rotation_matrix
from src.simulation import HotboxSimulation, SimulationResult
from src.sun import SunModel
from src.visualizer import SceneVisualizer, build_day_delivered_power_figure

# --- Sim-only knobs (plant geometry / site TZ come from config/system.yaml via hotbox_shared) ---

# Cell count along each axis on **each** square facet (rays per mirror ≈ grid_nx * grid_ny * U * V).
SIM_SAMPLES_U = 8
SIM_SAMPLES_V = 8

# Same per-facet cell counts for the multi-panel absorber spot figure.
SPOT_GRID_NUM_PANELS = 12
SPOT_GRID_NCOLS = 4
SPOT_GRID_SAMPLES_U = 32
SPOT_GRID_SAMPLES_V = 32
SPOT_GRID_BINS = 80

# Day curve: site-local sunrise → sunset, every N minutes. Int or same-length lists.
# Civil times use default_site.timezone_id from config/system.yaml (not the laptop TZ).
DAY_CURVE_YEAR = 2026
DAY_CURVE_MONTH = [8, 9]
DAY_CURVE_DAY = [30, 7]
DAY_CURVE_STEP_MINUTES = 20

# Site-local wall clock for the 3D scene / absorber spot figures and printed snapshot
# (mount solve, mirror angles, ray bundle). Built with the site timezone after load.
SCENE_VIS_YEAR = 2026
SCENE_VIS_MONTH = 9
SCENE_VIS_DAY = 7
SCENE_VIS_HOUR = 14
SCENE_VIS_MINUTE = 0

# 3D scene figure only: coarser facet grid and shorter incoming segments (sun-ward) than the
# snapshot raytrace used for printed power (still uses SIM_SAMPLES_* via ``sim.run`` defaults).
SCENE_VIS_SAMPLES_U = 3
SCENE_VIS_SAMPLES_V = 3
SCENE_VIS_UPSTREAM_DISTANCE_M = 6.0

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
    site: SitePose,
    *,
    year: int,
    month: int,
    day: int,
    step_minutes: int,
    num_panels: int,
) -> list[datetime]:
    """
    ``num_panels`` site-local times spread across sunrise–sunset for spot-pattern visualization.
    Falls back to empty if there is no daylight interval.
    """
    day_times, _, _ = local_times_sunrise_to_sunset(
        site,
        year=year,
        month=month,
        day=day,
        step_minutes=step_minutes,
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
    bypass_mirror_occlusion: bool = False,
    solve_for_mount_offset: bool = True,
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
            solve_for_mount_offset=solve_for_mount_offset,
        )
        t_after_mount = time.perf_counter()
        result = sim.run(when, verbose=sim_verbose, bypass_mirror_occlusion=bypass_mirror_occlusion)
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
    *,
    system: SystemConstants | None = None,
    sphere_center_offset_m: float | None = None,
) -> HotboxSimulation:
    """Build the optical plant from ``config/system.yaml`` (via ``hotbox_shared``)."""
    system = system or load_system_constants()
    site = system.default_site
    absorber_c = system.absorber
    mirror_c = system.mirror

    sun = SunModel(
        latitude_deg=site.latitude_deg,
        longitude_deg=site.longitude_deg,
        altitude_m=site.altitude_m,
    )
    absorber = SolarAbsorber(
        width_m=absorber_c.width_m,
        height_m=absorber_c.height_m,
        center_height_m=absorber_c.center_height_m,
        normal_angle_from_x_deg=absorber_c.normal_angle_from_x_deg,
    )

    dist_m = (
        sphere_center_offset_m
        if sphere_center_offset_m is not None
        else mirror_c.radius_of_curvature_m
    )
    tile_half_m = 0.5 * mirror_c.tile_side_m
    mounts = system.fleet.mounts
    if len(mounts) != system.fleet.assembly_count:
        raise ValueError(
            f"fleet.assembly_count={system.fleet.assembly_count} but "
            f"len(fleet.mounts)={len(mounts)}"
        )

    grids: list[AltAzFlatMirrorGrid] = []
    for mount in mounts:
        grids.append(
            AltAzFlatMirrorGrid(
                mount_world=system.mount_world(mount.node_id),
                grid_nx=mirror_c.grid_nx,
                grid_ny=mirror_c.grid_ny,
                pitch_m=mirror_c.pitch_m,
                tile_half_m=tile_half_m,
                sun=sun,
                sphere_center_offset_m=float(dist_m),
                mount_offset_d_m=float(mirror_c.mount_offset_d_m),
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
        "--sphere-center-offset-m",
        type=float,
        default=None,
        metavar="M",
        help=(
            "Sphere center z in mirror assembly frame [m]: sphere at (0, 0, z) with facet grid "
            "offset by mount_offset_d_m along +z. Default: mirror.radius_of_curvature_m from "
            "config/system.yaml."
        ),
    )
    parser.add_argument(
        "--bypass-mirror-occlusion",
        action="store_true",
        help=(
            "Skip mutual mirror shadowing and outgoing mirror-on-absorber occlusion in every "
            "raytrace (testing only; power and spot figures are not physically self-consistent)."
        ),
    )
    args = parser.parse_args()
    bypass_occ = bool(args.bypass_mirror_occlusion)
    if bypass_occ:
        print(
            "[hotbox] --bypass-mirror-occlusion: incoming shadowing and outgoing occlusion disabled "
            "(geometric facet/absorber hits only).",
            flush=True,
        )
    with timed_step("Load shared plant constants (config/system.yaml)"):
        system = load_system_constants()
        site = SitePose.from_constants(system.default_site)
        solve_for_mount_offset = bool(system.control.solve_for_mount_offset)
        print(
            f"[hotbox] site lat={site.latitude_deg}, lon={site.longitude_deg}, "
            f"alt={site.altitude_m} m; tz={site.timezone_id}; "
            f"fleet={system.fleet.assembly_count} mounts; "
            f"solve_for_mount_offset={solve_for_mount_offset}",
            flush=True,
        )
    with timed_step("Build default simulation (geometry + mirror grids)"):
        sim = build_default_simulation(
            system=system,
            sphere_center_offset_m=args.sphere_center_offset_m,
        )
    print(f"Mirror assemblies: {len(sim.mirrors)} (facet normals toward assembly-frame sphere)")
    if sim.mirrors:
        g0 = sim.mirrors[0]
        r0 = grid_mount_rotation_matrix(g0.azimuth_deg, g0.elevation_deg)
        o_w = g0.mount_world + r0 @ np.array([0.0, 0.0, float(g0.sphere_center_offset_m)], dtype=float)
        print(
            f"Sphere center: assembly (0, 0, {g0.sphere_center_offset_m:.4f}) m; "
            f"world xyz at current mount {o_w[0]:.4f}, {o_w[1]:.4f}, {o_w[2]:.4f} m"
        )
    day_specs = day_curve_month_day_pairs(DAY_CURVE_MONTH, DAY_CURVE_DAY)
    when = site_local_datetime(
        site,
        SCENE_VIS_YEAR,
        SCENE_VIS_MONTH,
        SCENE_VIS_DAY,
        SCENE_VIS_HOUR,
        SCENE_VIS_MINUTE,
    )
    print(f"[hotbox] scene snapshot at site-local {format_site_local(when, site)}", flush=True)

    with timed_step("Solve mount angles for scene snapshot"):
        orientations = mirror_orientations_for_time(
            when_utc=when,
            sun=sim.sun,
            absorber_center=sim.absorber.center,
            mirrors=sim.mirrors,
            absorber=sim.absorber,
            solve_for_mount_offset=solve_for_mount_offset,
        )

    with timed_step("Raytrace snapshot (scene time)"):
        result = sim.run(when, verbose=SHOW_MIRROR_TIMING, bypass_mirror_occlusion=bypass_occ)

    with timed_step("Raytrace snapshot (3D scene display)"):
        result_scene = sim.run(
            when,
            samples_u=SCENE_VIS_SAMPLES_U,
            samples_v=SCENE_VIS_SAMPLES_V,
            upstream_distance_m=SCENE_VIS_UPSTREAM_DISTANCE_M,
            verbose=False,
            bypass_mirror_occlusion=bypass_occ,
        )

    print(f"Sun ray direction (world xyz): {result.sun_direction}")
    for idx, (az, tilt) in enumerate(orientations):
        print(
            f"Mirror {idx} mount: azimuth={az:.2f} deg, "
            f"pivot facet tilt={tilt:.2f} deg (0=vertical plane, 90=horizontal toward zenith)"
        )
    for idx, mr in enumerate(result.per_mirror):
        print(
            f"Mirror {idx}: incident={mr.incident_power_w:.1f} W, "
            f"intercepted={mr.intercepted_power_w:.1f} W, delivered={mr.delivered_power_w:.1f} W"
        )
    print(f"Total delivered power: {result.total_delivered_power_w:.1f} W")

    viz = SceneVisualizer(sim.absorber, sim.mirrors)
    with timed_step("Build 3D scene figure (Plotly)"):
        scene_fig = viz.build_scene_figure(result_scene, scene_when_local=when)

    # Spot grid: same calendar day as scene snapshot, sunrise→sunset (see SPOT_GRID_*).
    with timed_step("Compute spot-pattern sample times"):
        spot_times = spot_pattern_sample_times(
            site,
            year=when.year,
            month=when.month,
            day=when.day,
            step_minutes=DAY_CURVE_STEP_MINUTES,
            num_panels=SPOT_GRID_NUM_PANELS,
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
                solve_for_mount_offset=solve_for_mount_offset,
            )
            r_spot = sim.run(
                t_spot,
                samples_u=SPOT_GRID_SAMPLES_U,
                samples_v=SPOT_GRID_SAMPLES_V,
                verbose=SHOW_MIRROR_TIMING,
                bypass_mirror_occlusion=bypass_occ,
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
                site,
                year=DAY_CURVE_YEAR,
                month=month_i,
                day=day_i,
                step_minutes=DAY_CURVE_STEP_MINUTES,
            )
        label = f"{month_i}/{day_i}/{DAY_CURVE_YEAR}"
        if sr is not None and ss is not None:
            print(
                f"Day curve {label}: sunrise {format_site_local(sr, site, '%Y-%m-%d %H:%M:%S %Z')}, "
                f"sunset {format_site_local(ss, site, '%Y-%m-%d %H:%M:%S %Z')} ({len(day_times)} samples)"
            )
        else:
            print(f"Day curve {label}: no sunrise/sunset (polar night or missing rise/set).")
        if day_times:
            _, day_delivered, day_intercepted, day_orients = simulate_delivered_power_over_times(
                sim,
                day_times,
                progress_label=f"Day curve {label}",
                sim_verbose=SHOW_MIRROR_TIMING,
                bypass_mirror_occlusion=bypass_occ,
                solve_for_mount_offset=solve_for_mount_offset,
            )
            day_series.append((label, day_times, day_delivered, day_intercepted, day_orients))
            if len(day_specs) == 1:
                single_curve_sr_ss = (sr, ss)

    if day_series:
        x_axis_title = f"Site local time ({site.timezone_id})"
        if len(day_series) == 1 and len(day_specs) == 1 and single_curve_sr_ss is not None:
            sr0, ss0 = single_curve_sr_ss
            month_i, day_i = day_specs[0]
            sr_s = sr0.strftime("%H:%M") if sr0 else "?"
            ss_s = ss0.strftime("%H:%M") if ss0 else "?"
            day_title = (
                f"Delivered & mirror-intercepted power — {month_i}/{day_i}/{DAY_CURVE_YEAR} "
                f"({site.timezone_id}; sunrise–sunset {sr_s}–{ss_s}, every {DAY_CURVE_STEP_MINUTES} min)"
            )
        else:
            dates_s = ", ".join(name for name, _, _, _, _ in day_series)
            day_title = (
                f"Delivered & mirror-intercepted power — {DAY_CURVE_YEAR} ({dates_s}), "
                f"sunrise–sunset {site.timezone_id}, every {DAY_CURVE_STEP_MINUTES} min"
            )
        with timed_step("Build day power Plotly figure"):
            day_fig = build_day_delivered_power_figure(
                day_series,
                title=day_title,
                x_axis_title=(
                    f"Site local time of day [h] ({site.timezone_id})"
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
