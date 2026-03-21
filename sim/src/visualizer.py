from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.absorber import SolarAbsorber
from src.mirror import CylindricalMirror
from src.simulation import SimulationResult


class SceneVisualizer:
    def __init__(self, absorber: SolarAbsorber, mirrors: list[CylindricalMirror]) -> None:
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

        sun_dir = result.sun_direction
        sun_start = np.array([0.0, 0.0, 5.5])
        sun_end = sun_start - 2.0 * sun_dir
        fig.add_trace(
            go.Scatter3d(
                x=[sun_start[0], sun_end[0]],
                y=[sun_start[1], sun_end[1]],
                z=[sun_start[2], sun_end[2]],
                mode="lines",
                line={"color": "gold", "width": 6},
                name="Sun direction",
            )
        )

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

        fig.update_layout(
            title="Hot-box optical scene",
            scene={
                "xaxis_title": "x (east) [m]",
                "yaxis_title": "y (north) [m]",
                "aspectmode": "cube",
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
        fig.update_layout(
            title="Spot pattern on absorber",
            xaxis_title="absorber horizontal axis [m]",
            yaxis_title="z [m]",
            xaxis={"scaleanchor": "y", "scaleratio": 1},
            yaxis={"constrain": "domain"},
            template="plotly_white",
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
        c = self.absorber.corners()
        x = np.array([[c[0, 0], c[1, 0]], [c[3, 0], c[2, 0]]])
        y = np.array([[c[0, 1], c[1, 1]], [c[3, 1], c[2, 1]]])
        z = np.array([[c[0, 2], c[1, 2]], [c[3, 2], c[2, 2]]])
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                opacity=0.5,
                showscale=False,
                colorscale=[[0.0, "#2ca02c"], [1.0, "#2ca02c"]],
                name="Absorber",
            )
        )

    @staticmethod
    def _add_mirror(fig: go.Figure, mirror: CylindricalMirror) -> None:
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
