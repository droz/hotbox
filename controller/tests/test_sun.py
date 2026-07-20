from __future__ import annotations

import numpy as np

from hotbox_controller.sun import pvlib_to_world_vector


def test_pvlib_to_world_vector_points_north_at_zero_azimuth() -> None:
    got = pvlib_to_world_vector(0.0, 0.0)
    np.testing.assert_allclose(got, np.array([0.0, 1.0, 0.0]), atol=1e-12)
