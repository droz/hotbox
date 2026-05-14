from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import convolve

from src.absorber import SolarAbsorber
from src.flat_mirror_grid import AltAzFlatMirrorGrid
from src.simulation import MirrorResult, SimulationResult

# Distinct colors per mirror assembly (Plotly merges legend entries when trace names repeat).
# Solar disk angular diameter (nominal ≈0.53°); blur radius on absorber uses limb half-angle θ/2.
SUN_APPARENT_ANGULAR_DIAMETER_DEG = 0.5


def sun_disc_radius_on_absorber_m(
    mirror_to_absorber_distance_m: float,
    *,
    sun_angular_diameter_deg: float = SUN_APPARENT_ANGULAR_DIAMETER_DEG,
) -> float:
    """Physical radius [m] of sun image from ``distance * tan(angular_radius)``."""
    half_angle_rad = 0.5 * np.deg2rad(float(sun_angular_diameter_deg))
    return float(mirror_to_absorber_distance_m) * float(np.tan(half_angle_rad))


_SCENE_MIRROR_ASSEMBLY_COLORS = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#b279a2",
    "#9467bd",
)


class SceneVisualizer:
    def __init__(self, absorber: SolarAbsorber, mirrors: list[AltAzFlatMirrorGrid]) -> None:
        self.absorber = absorber
        self.mirrors = mirrors

    def _scene_xy_limits(
        self,
        incoming_ray_length_m: float,
        *,
        ray_result: SimulationResult | None = None,
    ) -> tuple[list[float], list[float]]:
        """
        Tight x/y span for the ground patch only (not used to force scene axis ranges).

        When ``ray_result`` is set, includes simulation ray origins and hit points so upstream
        bundle extent (e.g. 50 m sun-ward) is in frame.

        Returns ``(x_range, y_range)`` in meters.
        """
        pts_xy: list[np.ndarray] = [self.absorber.corners()[:, :2]]
        for mirror in self.mirrors:
            for surf in mirror.tile_surface_grids(nu=2, nv=2):
                pts_xy.append(surf.reshape(-1, 3)[:, :2])

        if ray_result is not None:
            for mres in ray_result.per_mirror:
                inn = mres.incoming.origins
                if inn.size:
                    pts_xy.append(inn[:, :2])
                mh = mres.mirror_hit_mask
                if np.any(mh):
                    pts_xy.append(mres.mirror_hit_points[mh, :2])
                ah = mres.absorber_hit_mask
                if np.any(ah):
                    pts_xy.append(mres.absorber_hit_points[ah, :2])

        pts = np.vstack(pts_xy)
        x_min = float(np.min(pts[:, 0]))
        x_max = float(np.max(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))
        pad = float(max(incoming_ray_length_m, 0.0))
        return [x_min - pad, x_max + pad], [y_min - pad, y_max + pad]

    @staticmethod
    def _nan_break_segment_polylines(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Join segment pairs ``(a[k], b[k])`` into one line with NaN breaks for Plotly."""
        k = int(a.shape[0])
        if k == 0:
            return (
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
            )
        x = np.empty(k * 3, dtype=float)
        y = np.empty(k * 3, dtype=float)
        z = np.empty(k * 3, dtype=float)
        x[0::3] = a[:, 0]
        x[1::3] = b[:, 0]
        x[2::3] = np.nan
        y[0::3] = a[:, 1]
        y[1::3] = b[:, 1]
        y[2::3] = np.nan
        z[0::3] = a[:, 2]
        z[1::3] = b[:, 2]
        z[2::3] = np.nan
        return x, y, z

    @staticmethod
    def _subsample_flat_indices(indices: np.ndarray, max_count: int) -> np.ndarray:
        """Keep at most ``max_count`` entries from sorted 1D ``indices`` (deterministic stride)."""
        n = int(indices.size)
        if max_count <= 0 or n <= max_count:
            return indices
        step = int(np.ceil(n / max_count))
        return indices[::step]

    def _add_simulation_ray_segments(
        self,
        fig: go.Figure,
        mirror_result: MirrorResult,
        *,
        assembly_index: int,
        max_rays_per_assembly: int,
        incoming_legend: bool,
        reflected_legend: bool,
    ) -> tuple[bool, bool]:
        """
        Draw incoming and reflected segments from the same ray bundle used in ``HotboxSimulation.run``.

        Only rays counted as intercepted / delivered after shadowing and outgoing occlusion are
        drawn (``mirror_hit_mask`` / ``absorber_hit_mask``).
        """
        mr = mirror_result
        inc_idx = self._subsample_flat_indices(np.flatnonzero(mr.mirror_hit_mask), max_rays_per_assembly)
        abs_idx = self._subsample_flat_indices(np.flatnonzero(mr.absorber_hit_mask), max_rays_per_assembly)

        color = _SCENE_MIRROR_ASSEMBLY_COLORS[assembly_index % len(_SCENE_MIRROR_ASSEMBLY_COLORS)]

        any_incoming = False
        any_reflected = False

        if inc_idx.size > 0:
            o0 = mr.incoming.origins[inc_idx]
            h0 = mr.mirror_hit_points[inc_idx]
            xi, yi, zi = self._nan_break_segment_polylines(o0, h0)
            fig.add_trace(
                go.Scatter3d(
                    x=xi,
                    y=yi,
                    z=zi,
                    mode="lines",
                    line={"color": color, "width": 1.25},
                    opacity=0.4,
                    name="Incoming (simulation)",
                    showlegend=incoming_legend,
                )
            )
            any_incoming = True

        if abs_idx.size > 0:
            r0 = mr.reflected.origins[abs_idx]
            a0 = mr.absorber_hit_points[abs_idx]
            xr, yr, zr = self._nan_break_segment_polylines(r0, a0)
            fig.add_trace(
                go.Scatter3d(
                    x=xr,
                    y=yr,
                    z=zr,
                    mode="lines",
                    line={"color": color, "width": 1.25},
                    opacity=0.45,
                    name="Reflected (simulation)",
                    showlegend=reflected_legend,
                )
            )
            any_reflected = True

        return any_incoming, any_reflected

    def build_scene_figure(
        self,
        result: SimulationResult,
        incoming_ray_length_m: float = 2.0,
        scene_when_local: datetime | None = None,
        *,
        max_simulation_rays_per_assembly: int = 4000,
    ) -> go.Figure:
        x_range, y_range = self._scene_xy_limits(incoming_ray_length_m, ray_result=result)
        scene_title = "Hot-box optical scene"
        if scene_when_local is not None:
            ts = scene_when_local.strftime("%Y-%m-%d %H:%M")
            tz_s = scene_when_local.tzname() or ""
            scene_title = f"Hot-box optical scene — {ts} {tz_s}".strip()
        fig = go.Figure()
        self._add_ground(fig, x_range, y_range)
        self._add_absorber(fig)
        for i, mirror in enumerate(self.mirrors):
            self._add_mirror(fig, mirror, assembly_index=i)

        incoming_added = False
        reflected_added = False
        for assembly_index, mirror_result in enumerate(result.per_mirror):
            inc_here, ref_here = self._add_simulation_ray_segments(
                fig,
                mirror_result,
                assembly_index=assembly_index,
                max_rays_per_assembly=max_simulation_rays_per_assembly,
                incoming_legend=not incoming_added,
                reflected_legend=not reflected_added,
            )
            incoming_added = incoming_added or inc_here
            reflected_added = reflected_added or ref_here

        # Square figure + autosize=False avoids non-uniform div stretching in the page.
        # aspectmode="data": one data unit is the same length on screen along x, y, and z (physical
        # proportions). "auto" can inflate the short axis to fill a cube when x/y span is large,
        # which reads as z being stretched vertically relative to x and y.
        fig.update_layout(
            title=scene_title,
            autosize=False,
            width=720,
            height=720,
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
            scene={
                "xaxis_title": "x (east) [m]",
                "yaxis_title": "y (north) [m]",
                "aspectmode": "data",
                "camera": {
                    "projection": {"type": "orthographic"},
                    "eye": {"x": 1.35, "y": -1.35, "z": 0.9},
                },
                "xaxis": {
                    "showbackground": False,
                    "showgrid": False,
                    "zeroline": True,
                },
                "yaxis": {
                    "showbackground": False,
                    "showgrid": False,
                    "zeroline": True,
                },
                "zaxis": {
                    "visible": False,
                    "showbackground": False,
                    "showgrid": False,
                    "zeroline": False,
                },
            },
            legend={"x": 0.01, "y": 0.99},
        )
        return fig

    def _spot_uv_and_powers(self, result: SimulationResult) -> tuple[np.ndarray, np.ndarray] | None:
        pts: list[np.ndarray] = []
        powers: list[np.ndarray] = []
        c = self.absorber.center
        u_axis = self.absorber.horizontal_axis
        v_axis = self.absorber.vertical_axis

        for mres in result.per_mirror:
            mask = mres.absorber_hit_mask
            if not np.any(mask):
                continue
            p = mres.absorber_hit_points[mask]
            rel = p - c
            u = rel @ u_axis
            v = rel @ v_axis
            pts.append(np.column_stack([u, v]))
            powers.append(mres.reflected.powers_w[mask])

        if not pts:
            return None
        return np.vstack(pts), np.hstack(powers)

    def _spot_power_heatmap_z(
        self, uv: np.ndarray, pw: np.ndarray, bins: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = 0.5 * self.absorber.width_m
        h = 0.5 * self.absorber.height_m
        x_edges = np.linspace(-w, w, bins + 1)
        y_edges = np.linspace(-h, h, bins + 1)
        power_grid, _, _ = np.histogram2d(uv[:, 0], uv[:, 1], bins=[x_edges, y_edges], weights=pw)
        bin_area_m2 = ((2.0 * w) / bins) * ((2.0 * h) / bins)
        irradiance_grid = power_grid / max(bin_area_m2, 1e-12)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        return x_centers, y_centers, irradiance_grid.T

    def _mean_mirror_absorber_distance_m(self, result: SimulationResult) -> float:
        """Representative path length [m] for sun-blur scaling (power-weighted mean mount–absorber)."""
        ac = np.asarray(self.absorber.center, dtype=float).reshape(3)
        dists: list[float] = []
        weights: list[float] = []
        for mres in result.per_mirror:
            mw = np.asarray(mres.mirror.mount_world, dtype=float).reshape(3)
            dists.append(float(np.linalg.norm(mw - ac)))
            weights.append(max(float(mres.delivered_power_w), 0.0))
        ws = float(sum(weights))
        if ws > 1e-18:
            return float(np.average(dists, weights=weights))
        if dists:
            return float(np.mean(dists))
        return float(np.linalg.norm(ac))

    @staticmethod
    def _uniform_disc_convolution_kernel(
        radius_m: float,
        du_m: float,
        dv_m: float,
    ) -> np.ndarray:
        """Normalized uniform kernel over bin centers within ``radius_m`` in absorber (u, v) [m]."""
        if radius_m <= 0.0 or not np.isfinite(radius_m):
            return np.ones((1, 1), dtype=float)
        du_m = max(float(du_m), 1e-15)
        dv_m = max(float(dv_m), 1e-15)
        nx = int(np.ceil(radius_m / du_m))
        ny = int(np.ceil(radius_m / dv_m))
        ni = np.arange(-nx, nx + 1, dtype=float)[:, None]
        nj = np.arange(-ny, ny + 1, dtype=float)[None, :]
        dist_m = np.sqrt((ni * du_m) ** 2 + (nj * dv_m) ** 2)
        mask = dist_m <= radius_m + 1e-12
        k = mask.astype(float)
        s = float(np.sum(k))
        if s <= 0.0:
            return np.ones((1, 1), dtype=float)
        k /= s
        return k

    def _blur_spot_irradiance_sun_disc(
        self,
        z: np.ndarray,
        result: SimulationResult,
        *,
        bins: int,
        width_half_m: float,
        height_half_m: float,
        sun_angular_diameter_deg: float,
        mirror_absorber_distance_m: float | None,
    ) -> np.ndarray:
        """Convolve irradiance with a uniform disk ~sun angular diameter (preserves total bin sum)."""
        if z.size == 0:
            return z
        du_m = (2.0 * width_half_m) / bins
        dv_m = (2.0 * height_half_m) / bins
        dist_m = (
            float(mirror_absorber_distance_m)
            if mirror_absorber_distance_m is not None
            else self._mean_mirror_absorber_distance_m(result)
        )
        radius_m = sun_disc_radius_on_absorber_m(
            dist_m, sun_angular_diameter_deg=sun_angular_diameter_deg
        )
        k = self._uniform_disc_convolution_kernel(radius_m, du_m, dv_m)
        z_blur = convolve(z, k, mode="nearest")
        tb = float(np.sum(z_blur))
        ta = float(np.sum(z))
        if tb > 1e-20 and ta > 1e-20:
            z_blur *= ta / tb
        return z_blur

    def build_absorber_spot_figure(
        self,
        result: SimulationResult,
        bins: int = 60,
        *,
        sun_angular_diameter_deg: float = SUN_APPARENT_ANGULAR_DIAMETER_DEG,
        mirror_absorber_distance_m: float | None = None,
    ) -> go.Figure:
        fig = go.Figure()
        spot = self._spot_uv_and_powers(result)
        if spot is not None:
            uv, pw = spot
            x_centers, y_centers, z = self._spot_power_heatmap_z(uv, pw, bins)
            w_h = 0.5 * self.absorber.width_m
            h_h = 0.5 * self.absorber.height_m
            z = self._blur_spot_irradiance_sun_disc(
                z,
                result,
                bins=bins,
                width_half_m=w_h,
                height_half_m=h_h,
                sun_angular_diameter_deg=sun_angular_diameter_deg,
                mirror_absorber_distance_m=mirror_absorber_distance_m,
            )
            fig.add_trace(
                go.Heatmap(
                    x=x_centers,
                    y=y_centers,
                    z=z,
                    colorscale="Inferno",
                    colorbar={"title": "Irradiance [W/m²]"},
                    name="Spot heatmap",
                    hovertemplate="u %{x:.4f} m<br>v %{y:.4f} m<br>%{z:.3g} W/m²<extra></extra>",
                )
            )

        w = 0.5 * self.absorber.width_m
        h = 0.5 * self.absorber.height_m
        fig.add_trace(
            go.Scatter(
                x=[-w, w, w, -w, -w],
                y=[-h, -h, h, h, -h],
                mode="lines",
                line={"color": "black", "width": 2},
                name="Absorber boundary",
            )
        )
        lim = 0.55 * max(self.absorber.width_m, self.absorber.height_m)
        total_w = float(result.total_delivered_power_w)
        fig.update_layout(
            title=f"Spot pattern on absorber — total delivered {total_w:.1f} W",
            template="plotly_white",
            width=640,
            height=640,
            margin={"l": 60, "r": 20, "t": 50, "b": 55},
            xaxis_title="absorber horizontal axis [m]",
            yaxis_title="z [m]",
            xaxis={
                "range": [-lim, lim],
                "scaleanchor": "y",
                "scaleratio": 1,
                "constrain": "domain",
            },
            yaxis={
                "range": [-lim, lim],
                "scaleanchor": "x",
                "scaleratio": 1,
                "constrain": "domain",
            },
        )
        return fig

    def build_absorber_spot_figure_grid(
        self,
        labeled_results: list[tuple[str, SimulationResult]],
        *,
        bins: int = 72,
        ncols: int = 4,
        sun_angular_diameter_deg: float = SUN_APPARENT_ANGULAR_DIAMETER_DEG,
        mirror_absorber_distance_m: float | None = None,
    ) -> go.Figure:
        """
        Small multiples of absorber spot heatmaps (local time in subplot titles).

        All panels share one color scale so irradiation patterns are comparable across the day.
        Irradiance is convolved with a uniform disk whose radius is
        ``distance * tan((sun_angular_diameter_deg/2))`` in absorber coordinates (see
        ``sun_disc_radius_on_absorber_m``). By default ``distance`` is power-weighted mean
        mirror-mount to absorber distance per panel; pass ``mirror_absorber_distance_m`` to fix one
        value for every panel.
        """
        n = len(labeled_results)
        if n == 0:
            return go.Figure()
        nrows = int(np.ceil(n / ncols))
        titles = [
            f"{lab}<br>{res.total_delivered_power_w:.1f} W"
            for lab, res in labeled_results
        ] + [""] * (nrows * ncols - n)
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=titles[: nrows * ncols],
            vertical_spacing=0.09,
            horizontal_spacing=0.06,
        )

        lim = 0.55 * max(self.absorber.width_m, self.absorber.height_m)
        w = 0.5 * self.absorber.width_m
        h = 0.5 * self.absorber.height_m
        bx = [-w, w, w, -w, -w]
        by = [-h, -h, h, h, -h]

        zmax = 0.0
        cell_z: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray] | None] = []
        for i, (_lab, res) in enumerate(labeled_results):
            row = i // ncols + 1
            col = i % ncols + 1
            spot = self._spot_uv_and_powers(res)
            if spot is None:
                cell_z.append(None)
                continue
            uv, pw = spot
            xc, yc, z = self._spot_power_heatmap_z(uv, pw, bins)
            w_h = 0.5 * self.absorber.width_m
            h_h = 0.5 * self.absorber.height_m
            z = self._blur_spot_irradiance_sun_disc(
                z,
                res,
                bins=bins,
                width_half_m=w_h,
                height_half_m=h_h,
                sun_angular_diameter_deg=sun_angular_diameter_deg,
                mirror_absorber_distance_m=mirror_absorber_distance_m,
            )
            zm = float(np.nanmax(z)) if z.size else 0.0
            zmax = max(zmax, zm)
            cell_z.append((row, col, xc, yc, z))

        zmax = max(zmax, 1e-30)

        colorbar_shown = False
        for i, cz in enumerate(cell_z):
            row = i // ncols + 1
            col = i % ncols + 1
            if cz is None:
                fig.add_trace(
                    go.Scatter(
                        x=bx,
                        y=by,
                        mode="lines",
                        line={"color": "black", "width": 1},
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
                continue
            _, _, xc, yc, z = cz
            show_cbar = not colorbar_shown
            if show_cbar:
                colorbar_shown = True
            fig.add_trace(
                go.Heatmap(
                    x=xc,
                    y=yc,
                    z=z,
                    zmin=0.0,
                    zmax=zmax,
                    colorscale="Inferno",
                    showscale=show_cbar,
                    colorbar=(
                        {"title": "Irradiance [W/m²]", "len": 0.92, "thickness": 14}
                        if show_cbar
                        else None
                    ),
                    hovertemplate="u %{x:.4f} m<br>v %{y:.4f} m<br>%{z:.3g} W/m²<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=bx,
                    y=by,
                    mode="lines",
                    line={"color": "black", "width": 1},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        cell_px = 220
        fig.update_layout(
            title="Absorber spot pattern through the day (shared color scale)",
            template="plotly_white",
            width=min(40 + ncols * cell_px, 1400),
            height=min(70 + nrows * cell_px, 1100),
            margin={"l": 50, "r": 30, "t": 80, "b": 40},
        )
        for r in range(1, nrows + 1):
            for c in range(1, ncols + 1):
                ref = fig.get_subplot(row=r, col=c)
                fig.update_xaxes(
                    title_text="u [m]" if r == nrows else "",
                    range=[-lim, lim],
                    scaleanchor=ref.xaxis.anchor,
                    scaleratio=1,
                    constrain="domain",
                    row=r,
                    col=c,
                )
                fig.update_yaxes(
                    title_text="v [m]" if c == 1 else "",
                    range=[-lim, lim],
                    scaleanchor=ref.yaxis.anchor,
                    scaleratio=1,
                    constrain="domain",
                    row=r,
                    col=c,
                )
        return fig

    @staticmethod
    def _add_ground(fig: go.Figure, x_range: list[float], y_range: list[float]) -> None:
        x = np.array(x_range)
        y = np.array(y_range)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=zz,
                opacity=0.15,
                showscale=False,
                colorscale=[[0.0, "#d8d8d8"], [1.0, "#d8d8d8"]],
                name="Ground",
            )
        )

    def _add_absorber(self, fig: go.Figure) -> None:
        # Explicit planar quad (two triangles) so the patch is exactly the absorber rectangle.
        c = self.absorber.corners()
        x = c[:, 0].tolist()
        y = c[:, 1].tolist()
        z = c[:, 2].tolist()
        # Winding: (0,1,2) and (0,2,3) matches corners order from SolarAbsorber.corners().
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color="#2ca02c",
                opacity=0.5,
                flatshading=True,
                name="Absorber",
            )
        )

    @staticmethod
    def _add_mirror(fig: go.Figure, mirror: AltAzFlatMirrorGrid, assembly_index: int) -> None:
        c = _SCENE_MIRROR_ASSEMBLY_COLORS[assembly_index % len(_SCENE_MIRROR_ASSEMBLY_COLORS)]
        colorscale = [[0.0, c], [1.0, c]]
        trace_name = f"Mirror assembly {assembly_index}"
        first = True
        for surf in mirror.tile_surface_grids():
            fig.add_trace(
                go.Surface(
                    x=surf[..., 0],
                    y=surf[..., 1],
                    z=surf[..., 2],
                    opacity=0.45,
                    showscale=False,
                    colorscale=colorscale,
                    name=trace_name,
                    showlegend=first,
                )
            )
            first = False


def _local_hours_since_midnight(dt: datetime) -> float:
    """Wall-clock time of day in hours (uses dt's own tz/calendar components)."""
    return (
        dt.hour
        + dt.minute / 60.0
        + dt.second / 3600.0
        + dt.microsecond / 3_600_000_000.0
    )


def build_day_delivered_power_figure(
    series: list[
        tuple[str, list[datetime], list[float], list[float], list[list[tuple[float, float]]]]
    ],
    *,
    title: str = "Delivered & mirror-intercepted power vs time",
    y_axis_title: str = "Power [W]",
    x_axis_title: str = "Local time",
    same_day_time_scale: bool = False,
) -> go.Figure:
    """Two stacked panels: power (top), mirror orientation per mirror (bottom).

    Each series entry is ``(label, local times, delivered_w, intercepted_w, orientations)``.
    ``intercepted_w`` is total power striking all mirrors (sum of per-mirror intercepted).
    ``orientations``
    has the same length as ``times``; each element is ``[(az_deg, second_deg), ...]`` per mirror.
    For each mirror, ``second_deg`` is pivot-facet tilt in world (0° vertical, 90° horizontal toward zenith),
    not the raw ``(azimuth_deg, elevation_deg)`` joint tuple on the grid.

    If ``same_day_time_scale`` is True, x is hours since local midnight for overlaying different
    calendar days; hover still shows the actual timestamp.
    """
    palette = (
        "#d62728",
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.5, 0.5],
        subplot_titles=(
            "Delivered (solid) & power on mirrors (dashed)",
            "Lattice plane (solid = azimuth, dashed = tilt 0°=vertical, 90°=horizontal zenith)",
        ),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    for series_i, (name, times_local, delivered_w, intercepted_w, orient_data) in enumerate(series):
        if not (
            len(orient_data) == len(times_local)
            == len(delivered_w)
            == len(intercepted_w)
        ):
            raise ValueError(
                "times, delivered_w, intercepted_w, and orientations must have the same length."
            )
        n_m = len(orient_data[0]) if orient_data else 0

        if same_day_time_scale:
            x_plot = [_local_hours_since_midnight(t) for t in times_local]
            stamp = [
                t.strftime("%Y-%m-%d %H:%M")
                + (f" {t.tzname()}" if t.tzinfo is not None else "")
                for t in times_local
            ]
        else:
            x_plot = list(times_local)
            stamp = None

        # Slight x shift per overlay curve so traces at the same wall time do not paint on top
        # of each other (otherwise a bad/zero series can hide a good one).
        x_shift_h = 0.012 * series_i if (same_day_time_scale and len(series) > 1) else 0.0
        x_plot_used = [x + x_shift_h for x in x_plot] if x_shift_h else x_plot

        c_power = palette[series_i % len(palette)]
        delivered_trace = go.Scatter(
            x=x_plot_used,
            y=delivered_w,
            mode="lines+markers",
            name=f"{name} delivered",
            legendgroup=f"{name}_pwr",
            line={"color": c_power, "width": 2},
            marker={"size": 5},
        )
        intercept_trace = go.Scatter(
            x=x_plot_used,
            y=intercepted_w,
            mode="lines+markers",
            name=f"{name} on mirrors",
            legendgroup=f"{name}_pwr",
            line={"color": c_power, "width": 2, "dash": "dash"},
            marker={"size": 5, "symbol": "diamond"},
        )
        if stamp is not None:
            delivered_trace.update(
                customdata=stamp,
                hovertemplate="%{customdata}<br>delivered %{y:.1f} W<extra></extra>",
            )
            intercept_trace.update(
                customdata=stamp,
                hovertemplate="%{customdata}<br>on mirrors %{y:.1f} W<extra></extra>",
            )
        fig.add_trace(delivered_trace, row=1, col=1)
        fig.add_trace(intercept_trace, row=1, col=1)

        for m in range(n_m):
            az = [orient_data[t][m][0] for t in range(len(times_local))]
            el = [orient_data[t][m][1] for t in range(len(times_local))]
            c_m = palette[(series_i * max(n_m, 1) + m) % len(palette)]
            lg = f"{name}_m{m}"
            az_trace = go.Scatter(
                x=x_plot_used,
                y=az,
                mode="lines+markers",
                name=f"{name} M{m} az",
                legendgroup=lg,
                line={"color": c_m, "width": 1.5},
                marker={"size": 3},
            )
            el_trace = go.Scatter(
                x=x_plot_used,
                y=el,
                mode="lines+markers",
                name=f"{name} M{m} tilt",
                legendgroup=lg,
                line={"color": c_m, "width": 1.5, "dash": "dash"},
                marker={"size": 3},
            )
            if stamp is not None:
                az_trace.update(
                    customdata=stamp,
                    hovertemplate="%{customdata}<br>azimuth %{y:.2f}°<extra></extra>",
                )
                el_trace.update(
                    customdata=stamp,
                    hovertemplate="%{customdata}<br>tilt %{y:.2f}°<extra></extra>",
                )
            fig.add_trace(az_trace, row=2, col=1, secondary_y=False)
            fig.add_trace(el_trace, row=2, col=1, secondary_y=True)

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        width=920,
        height=780,
        margin={"l": 58, "r": 58, "t": 90, "b": 60},
        legend={
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.01,
        },
    )
    fig.update_yaxes(title_text=y_axis_title, row=1, col=1)
    fig.update_yaxes(title_text="Azimuth [deg]", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Tilt [deg]", secondary_y=True, row=2, col=1)
    fig.update_xaxes(title_text=x_axis_title, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    if same_day_time_scale:
        tick_h = list(range(0, 25, 2))
        tick_labels = [f"{h:02d}:00" for h in tick_h]
        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_h,
            ticktext=tick_labels,
            row=1,
            col=1,
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_h,
            ticktext=tick_labels,
            row=2,
            col=1,
        )
    else:
        fig.update_xaxes(tickformat="%H:%M", row=1, col=1)
        fig.update_xaxes(tickformat="%H:%M", row=2, col=1)

    return fig
