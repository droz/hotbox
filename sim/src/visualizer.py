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

        # Square layout + autosize off avoids the browser stretching a 3D subplot. Plotly's
        # WebGL viewer often makes Z look ~2× taller than X/Y at 1:1:1; aspectratio.z=0.5
        # compresses displayed Z so one meter vertically matches one meter horizontally.
        fig.update_layout(
            title="Hot-box optical scene",
            autosize=False,
            width=720,
            height=720,
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
            scene={
                "xaxis_title": "x (east) [m]",
                "yaxis_title": "y (north) [m]",
                "aspectmode": "manual",
                "aspectratio": {"x": 1, "y": 1, "z": 0.5},
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
