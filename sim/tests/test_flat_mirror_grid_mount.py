"""Geometry tests: mount R_z @ R_x and lattice plane / facet normals built from vectors."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import numpy as np

from src.flat_mirror_grid import (
    AltAzFlatMirrorGrid,
    coarse_mount_angles_align_lattice_normal,
    grid_mount_rotation_matrix,
    unit_mirror_normal_at_point,
)
from src.geometry import normalize
from src.sun import SunModel


def _R_x(el_deg: float) -> np.ndarray:
    el = np.deg2rad(el_deg)
    c, s = np.cos(el), np.sin(el)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _R_z(az_deg: float) -> np.ndarray:
    az = np.deg2rad(az_deg)
    c, s = np.cos(az), np.sin(az)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


class TestUnitMirrorNormalAtPoint(unittest.TestCase):
    def test_pivot_normal_reflects_sun_toward_absorber(self) -> None:
        d_sun = normalize(np.array([[0.3, -0.2, -0.9]], dtype=float))[0]
        m = np.array([1.0, 0.5, 0.3], dtype=float)
        a = np.array([0.0, 0.0, 1.2], dtype=float)
        n = unit_mirror_normal_at_point(d_sun, m, a)
        dn = float(np.dot(d_sun, n))
        refl = d_sun - 2.0 * dn * n
        want = normalize((a - m).reshape(1, 3))[0]
        np.testing.assert_allclose(refl, want, atol=1e-9, rtol=0.0)


class TestCoarseMountAnglesAlignLatticeNormal(unittest.TestCase):
    def test_recovers_known_rotation(self) -> None:
        n0 = normalize(np.array([[0.15, -0.72, 0.67]], dtype=float))[0]
        az_t, el_t = 118.0, 41.0
        n_target = grid_mount_rotation_matrix(az_t, el_t) @ n0
        az, el = coarse_mount_angles_align_lattice_normal(n0, n_target)
        n_got = grid_mount_rotation_matrix(az, el) @ n0
        np.testing.assert_allclose(n_got, n_target, atol=0.02, rtol=0.0)


class TestGridMountRotationMatrix(unittest.TestCase):
    def test_identity_at_zero(self) -> None:
        r = grid_mount_rotation_matrix(0.0, 0.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-14)

    def test_matches_manual_rz_rx_product(self) -> None:
        for az in (0.0, 17.0, -40.0, 350.0):
            for el in (0.0, 12.5, 45.0, 90.0):
                got = grid_mount_rotation_matrix(az, el)
                want = _R_z(az) @ _R_x(el)
                np.testing.assert_allclose(got, want, atol=1e-12, err_msg=f"az={az} el={el}")

    def test_elevation_tips_north_normal_to_zenith(self) -> None:
        ey = np.array([0.0, 1.0, 0.0])
        n = grid_mount_rotation_matrix(0.0, 90.0) @ ey
        np.testing.assert_allclose(n, np.array([0.0, 0.0, 1.0]), atol=1e-12)

    def test_azimuth_at_zero_elevation_rotates_north_normal_in_xy(self) -> None:
        ey = np.array([0.0, 1.0, 0.0])
        for az in (30.0, 90.0, 180.0):
            n = grid_mount_rotation_matrix(az, 0.0) @ ey
            self.assertAlmostEqual(float(n[2]), 0.0, places=12)
            self.assertAlmostEqual(float(np.linalg.norm(n[:2])), 1.0, places=12)

    def test_composition_order_is_rx_then_rz_on_body_vectors(self) -> None:
        v = np.array([0.0, 0.0, 1.0], dtype=float)
        az, el = 40.0, 25.0
        step1 = _R_x(el) @ v
        step2 = _R_z(az) @ step1
        full = grid_mount_rotation_matrix(az, el) @ v
        np.testing.assert_allclose(full, step2, atol=1e-12)

    def test_not_equal_to_wrong_order_r_x_r_z(self) -> None:
        v = np.array([1.0, 1.0, 0.3], dtype=float)
        v /= np.linalg.norm(v)
        az, el = 35.0, 20.0
        wrong = _R_x(el) @ _R_z(az) @ v
        right = grid_mount_rotation_matrix(az, el) @ v
        self.assertGreater(np.linalg.norm(wrong - right), 0.05)


class TestAltAzFlatMirrorGridBodyLayout(unittest.TestCase):
    def _make_grid(self) -> AltAzFlatMirrorGrid:
        sun = SunModel(latitude_deg=40.7864, longitude_deg=-119.2065, altitude_m=1190.0)
        when = datetime(2026, 6, 21, 19, 0, 0, tzinfo=timezone.utc)
        return AltAzFlatMirrorGrid(
            mount_world=np.array([2.0, -1.5, 0.9], dtype=float),
            design_when_utc=when,
            absorber_center=np.array([0.0, 0.0, 1.0], dtype=float),
            grid_n=5,
            pitch_m=0.2,
            tile_half_m=0.08,
            sun=sun,
        )

    def test_facet_centers_orthogonal_to_lattice_normal(self) -> None:
        g = self._make_grid()
        n_pi = g._lattice_plane_normal_body
        for c in g._c_body:
            self.assertAlmostEqual(float(np.dot(c, n_pi)), 0.0, places=10)

    def test_lattice_plane_normal_matches_bisector_at_pivot(self) -> None:
        g = self._make_grid()
        d_sun = g.sun.ray_direction(g.design_when_utc)
        m = g.mount_world.reshape(3)
        a = g.absorber_center.reshape(3)
        want = unit_mirror_normal_at_point(d_sun, m, a)
        np.testing.assert_allclose(g._lattice_plane_normal_body, want, atol=1e-12)

    def test_lattice_plane_as_single_mirror_reflects_at_pivot(self) -> None:
        """If the whole grid were one flat mirror with normal n_π at M, design sun aims at A."""
        g = self._make_grid()
        d_sun = g.sun.ray_direction(g.design_when_utc)
        m = g.mount_world.reshape(3)
        a = g.absorber_center.reshape(3)
        n_pi = g._lattice_plane_normal_body
        dn = float(np.dot(d_sun, n_pi))
        refl = d_sun - 2.0 * dn * n_pi
        want = normalize((a - m).reshape(1, 3))[0]
        np.testing.assert_allclose(refl, want, atol=1e-9, rtol=0.0)

    def test_incoming_ray_bundle_extents_are_finite(self) -> None:
        g = self._make_grid()
        g.azimuth_deg = 10.0
        g.elevation_deg = 15.0
        d = np.array([0.2, -0.3, -0.9], dtype=float)
        d /= np.linalg.norm(d)
        c, hu, hv = g.incoming_ray_bundle_extents(d)
        self.assertEqual(c.shape, (3,))
        self.assertGreater(hu, 0.0)
        self.assertGreater(hv, 0.0)

    def test_world_positions_at_zero_mount_match_mount_plus_body(self) -> None:
        g = self._make_grid()
        g.azimuth_deg = 0.0
        g.elevation_deg = 0.0
        r = grid_mount_rotation_matrix(0.0, 0.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-15)
        c_w, _, _, _ = g._world_facets()
        m = g.mount_world.reshape(3)
        np.testing.assert_allclose(c_w, m.reshape(1, 3) + g._c_body, atol=1e-12)

    def test_each_facet_normal_reflects_design_sun_toward_absorber(self) -> None:
        g = self._make_grid()
        d_sun = g.sun.ray_direction(g.design_when_utc)
        a = g.absorber_center.reshape(3)
        g.azimuth_deg = 0.0
        g.elevation_deg = 0.0
        c_w, n_w, _, _ = g._world_facets()
        for f in range(c_w.shape[0]):
            c = c_w[f]
            n = n_w[f]
            dn = float(np.dot(d_sun, n))
            refl = d_sun - 2.0 * dn * n
            want = normalize((a - c).reshape(1, 3))[0]
            np.testing.assert_allclose(refl, want, atol=1e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
