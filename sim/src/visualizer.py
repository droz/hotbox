from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter

from src.absorber import SolarAbsorber
from src.flat_mirror_grid import AltAzFlatMirrorGrid
from src.geometry import normalize
from src.simulation import SimulationResult

# Distinct colors per mirror assembly (Plotly merges legend entries when trace names repeat).
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

    def _scene_xy_limits(self, incoming_ray_length_m: float) -> tuple[list[float], list[float]]:
        """
        Tight x/y span for the ground patch only (not used to force scene axis ranges).

        Returns ``(x_range, y_range)`` in meters.
        """
        pts_xy: list[np.ndarray] = [self.absorber.corners()[:, :2]]
        for mirror in self.mirrors:
            for surf in mirror.tile_surface_grids(nu=2, nv=2):
                pts_xy.append(surf.reshape(-1, 3)[:, :2])

        pts = np.vstack(pts_xy)
        x_min = float(np.min(pts[:, 0]))
        x_max = float(np.max(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))
        return [x_min, x_max], [y_min, y_max]

    def _add_flat_grid_facet_center_rays(
        self,
        fig: go.Figure,
        grid: AltAzFlatMirrorGrid,
        sun_direction: np.ndarray,
        *,
        incoming_ray_length_m: float,
        reflected_stub_m: float = 4.0,
        incoming_legend: bool,
        reflected_legend: bool,
    ) -> tuple[bool, bool]:
        """
        One incoming + one reflected segment per facet, through the facet center (parallel sun).

        Returns ``(drew_any_incoming, drew_any_reflected)`` for legend bookkeeping upstream.
        """
        d = normalize(np.asarray(sun_direction, dtype=float).reshape(1, 3))[0]
        c_w, n_w, _, _ = grid._world_facets()
        na = self.absorber.normal.reshape(3)
        ca = self.absorber.center.reshape(3)

        inc_leg = incoming_legend
        ref_leg = reflected_legend
        any_incoming = False
        any_reflected = False

        for f in range(c_w.shape[0]):
            c = c_w[f]
            n = n_w[f]
            dn = float(np.dot(d, n))
            if abs(dn) < 1e-10:
                continue
            if dn > 0.0:
                continue

            p_hit = c
            p_in0 = c - incoming_ray_length_m * d
            fig.add_trace(
                go.Scatter3d(
                    x=[p_in0[0], p_hit[0]],
                    y=[p_in0[1], p_hit[1]],
                    z=[p_in0[2], p_hit[2]],
                    mode="lines",
                    line={"color": "lightskyblue", "width": 1.5},
                    opacity=0.45,
                    name="Incoming (facet centers)",
                    showlegend=inc_leg,
                )
            )
            inc_leg = False
            any_incoming = True

            r = d - 2.0 * dn * n
            rn = float(np.linalg.norm(r))
            if rn < 1e-12:
                continue
            r /= rn
            denom = float(np.dot(r, na))
            if abs(denom) < 1e-12:
                p_out = c + reflected_stub_m * r
            else:
                t = float(np.dot(ca - c, na) / denom)
                if t > 1e-6:
                    p_out = c + t * r
                else:
                    p_out = c + reflected_stub_m * r
            fig.add_trace(
                go.Scatter3d(
                    x=[c[0], p_out[0]],
                    y=[c[1], p_out[1]],
                    z=[c[2], p_out[2]],
                    mode="lines",
                    line={"color": "tomato", "width": 1.5},
                    opacity=0.5,
                    name="Reflected (facet centers)",
                    showlegend=ref_leg,
                )
            )
            ref_leg = False
            any_reflected = True

        return any_incoming, any_reflected

    def build_scene_figure(
        self,
        result: SimulationResult,
        incoming_ray_length_m: float = 2.0,
        scene_when_local: datetime | None = None,
    ) -> go.Figure:
        x_range, y_range = self._scene_xy_limits(incoming_ray_length_m)
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
        for mirror_result in result.per_mirror:
            mir = mirror_result.mirror
            inc_here, ref_here = self._add_flat_grid_facet_center_rays(
                fig,
                mir,
                result.sun_direction,
                incoming_ray_length_m=incoming_ray_length_m,
                incoming_legend=not incoming_added,
                reflected_legend=not reflected_added,
            )
            incoming_added = incoming_added or inc_here
            reflected_added = reflected_added or ref_here

        # Square figure + autosize=False avoids non-uniform div stretching in the page.
        # aspectmode="auto" (Plotly default): like "data" (proportional axis box) unless one
        # axis span is >4× the other two, then "cube" — avoids an ultra-thin vertical slab when
        # x/y footprint ≫ height (which reads as "flattened" with aspectmode="data" alone).
        fig.update_layout(
            title=scene_title,
            autosize=False,
            width=720,
            height=720,
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
            scene={
                "xaxis_title": "x (east) [m]",
                "yaxis_title": "y (north) [m]",
                "aspectmode": "auto",
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

    @staticmethod
    def _smoothed_spot_power_grid(z: np.ndarray, sigma_px: float) -> np.ndarray:
        """Apply light Gaussian smoothing while preserving total bin power."""
        if sigma_px <= 0.0 or z.size == 0:
            return z
        total_before = float(np.sum(z))
        z_smooth = gaussian_filter(z, sigma=float(sigma_px), mode="nearest")
        total_after = float(np.sum(z_smooth))
        if total_after > 1e-20:
            z_smooth *= total_before / total_after
        return z_smooth

    def build_absorber_spot_figure(
        self,
        result: SimulationResult,
        bins: int = 60,
        smooth_sigma_px: float = 0.8,
    ) -> go.Figure:
        fig = go.Figure()
        spot = self._spot_uv_and_powers(result)
        if spot is not None:
            uv, pw = spot
            x_centers, y_centers, z = self._spot_power_heatmap_z(uv, pw, bins)
            z = self._smoothed_spot_power_grid(z, smooth_sigma_px)
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
        smooth_sigma_px: float = 1.2,
    ) -> go.Figure:
        """
        Small multiples of absorber spot heatmaps (local time in subplot titles).

        All panels share one color scale so irradiation patterns are comparable across the day.
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
            z = self._smoothed_spot_power_grid(z, smooth_sigma_px)
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
    For each mirror, ``second_deg`` is lattice-plane tilt (0° vertical, 90° horizontal toward zenith),
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
