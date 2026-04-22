from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.absorber import SolarAbsorber
from src.flat_mirror_grid import AltAzFlatMirrorGrid
from src.mirror import CylindricalMirror
from src.simulation import SimulationResult


class SceneVisualizer:
    def __init__(self, absorber: SolarAbsorber, mirrors: list[Any]) -> None:
        self.absorber = absorber
        self.mirrors = mirrors

    def build_scene_figure(
        self,
        result: SimulationResult,
        ray_stride: int = 120,
        incoming_ray_length_m: float = 2.0,
    ) -> go.Figure:
        fig = go.Figure()
        self._add_ground(fig, size=8.0)
        self._add_absorber(fig)
        for mirror in self.mirrors:
            self._add_mirror(fig, mirror)

        incoming_added = False
        reflected_added = False
        for mirror_result in result.per_mirror:
            incoming = mirror_result.incoming
            mirror_hits = mirror_result.mirror_hit_points
            mask = mirror_result.mirror_hit_mask
            hit_indices = np.where(mask)[0][::ray_stride]
            for i in hit_indices:
                p1 = mirror_hits[i]
                d = incoming.directions[i]
                p0 = p1 - incoming_ray_length_m * d
                fig.add_trace(
                    go.Scatter3d(
                        x=[p0[0], p1[0]],
                        y=[p0[1], p1[1]],
                        z=[p0[2], p1[2]],
                        mode="lines",
                        line={"color": "lightskyblue", "width": 2},
                        opacity=0.5,
                        name="Incoming rays",
                        showlegend=not incoming_added,
                    )
                )
                incoming_added = True

            reflected = mirror_result.reflected
            absorber_hits = mirror_result.absorber_hit_points
            h2 = mirror_result.absorber_hit_mask
            hit2_indices = np.where(h2)[0][::ray_stride]
            for i in hit2_indices:
                p0 = reflected.origins[i]
                p1 = absorber_hits[i]
                fig.add_trace(
                    go.Scatter3d(
                        x=[p0[0], p1[0]],
                        y=[p0[1], p1[1]],
                        z=[p0[2], p1[2]],
                        mode="lines",
                        line={"color": "tomato", "width": 2},
                        opacity=0.55,
                        name="Reflected rays",
                        showlegend=not reflected_added,
                    )
                )
                reflected_added = True

        # Square figure + autosize=False limits div stretching. aspectmode="data" is Plotly's
        # 3D equivalent of matplotlib axis("equal"): one meter along x, y, or z is the same
        # length on screen (orthogonal to camera view).
        fig.update_layout(
            title="Hot-box optical scene",
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

    def build_absorber_spot_figure(self, result: SimulationResult, bins: int = 60) -> go.Figure:
        pts = []
        powers = []
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

        fig = go.Figure()
        if pts:
            uv = np.vstack(pts)
            pw = np.hstack(powers)
            w = 0.5 * self.absorber.width_m
            h = 0.5 * self.absorber.height_m
            x_edges = np.linspace(-w, w, bins + 1)
            y_edges = np.linspace(-h, h, bins + 1)
            power_grid, _, _ = np.histogram2d(uv[:, 0], uv[:, 1], bins=[x_edges, y_edges], weights=pw)
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            fig.add_trace(
                go.Heatmap(
                    x=x_centers,
                    y=y_centers,
                    z=power_grid.T,
                    colorscale="Inferno",
                    colorbar={"title": "Bin power [W]"},
                    name="Spot heatmap",
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
        # Equal data units on both axes and a square figure so a square absorber stays square.
        lim = 0.55 * max(self.absorber.width_m, self.absorber.height_m)
        fig.update_layout(
            title="Spot pattern on absorber",
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

    @staticmethod
    def _add_ground(fig: go.Figure, size: float) -> None:
        x = np.array([-size, size])
        y = np.array([-size, size])
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
    def _add_mirror(fig: go.Figure, mirror: Any) -> None:
        if isinstance(mirror, AltAzFlatMirrorGrid):
            first = True
            for surf in mirror.tile_surface_grids():
                fig.add_trace(
                    go.Surface(
                        x=surf[..., 0],
                        y=surf[..., 1],
                        z=surf[..., 2],
                        opacity=0.45,
                        showscale=False,
                        colorscale=[[0.0, "#4c78a8"], [1.0, "#4c78a8"]],
                        name="Mirror tiles",
                        showlegend=first,
                    )
                )
                first = False
            return
        surf = mirror.surface_grid()
        fig.add_trace(
            go.Surface(
                x=surf[..., 0],
                y=surf[..., 1],
                z=surf[..., 2],
                opacity=0.45,
                showscale=False,
                colorscale=[[0.0, "#4c78a8"], [1.0, "#4c78a8"]],
                name="Mirror",
            )
        )


def _local_hours_since_midnight(dt: datetime) -> float:
    """Wall-clock time of day in hours (uses dt's own tz/calendar components)."""
    return (
        dt.hour
        + dt.minute / 60.0
        + dt.second / 3600.0
        + dt.microsecond / 3_600_000_000.0
    )


def build_day_delivered_power_figure(
    series: list[tuple[str, list[datetime], list[float], list[list[tuple[float, float]]]]],
    *,
    title: str = "Delivered optical power vs time",
    y_axis_title: str = "Delivered power [W]",
    x_axis_title: str = "Local time",
    same_day_time_scale: bool = False,
) -> go.Figure:
    """Two stacked panels: delivered power (top), mirror orientation per mirror (bottom).

    Each series entry is ``(label, local times, powers_w, orientations)`` where ``orientations``
    has the same length as ``times``; each element is ``[(az_deg, second_deg), ...]`` per mirror.
    For ``AltAzFlatMirrorGrid``, ``second_deg`` is lattice-plane tilt (0° vertical, 90° horizontal
    toward zenith), not the raw ``(azimuth_deg, elevation_deg)`` joint tuple on the grid. For cylindrical mirrors, ``second_deg`` is
    elevation of the mirror normal [deg].

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
            "Delivered power",
            "Lattice plane (solid = azimuth, dashed = tilt 0°=vertical, 90°=horizontal zenith)",
        ),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    for series_i, (name, times_local, powers_w, orient_data) in enumerate(series):
        if len(orient_data) != len(times_local) or len(powers_w) != len(times_local):
            raise ValueError("times, powers, and orientations must have the same length.")
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
        power_trace = go.Scatter(
            x=x_plot_used,
            y=powers_w,
            mode="lines+markers",
            name=name,
            line={"color": c_power, "width": 2},
            marker={"size": 5},
        )
        if stamp is not None:
            power_trace.update(
                customdata=stamp,
                hovertemplate="%{customdata}<br>%{y:.1f} W<extra></extra>",
            )
        fig.add_trace(power_trace, row=1, col=1)

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
