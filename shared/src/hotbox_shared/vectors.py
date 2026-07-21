"""Vector primitives for Hot-Box optics (world frame: ENU, +x east, +y north, +z up)."""

from __future__ import annotations

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector; raise ``ValueError`` on zero length."""
    v = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        raise ValueError("cannot normalize zero-length vector")
    return v / norm


def bisector_normal(
    incoming_toward_mirror: np.ndarray,
    outgoing_from_mirror: np.ndarray,
) -> np.ndarray:
    """
    Unit mirror normal for specular reflection.

    Args:
        incoming_toward_mirror: Unit ray direction from the sun toward the mirror.
        outgoing_from_mirror: Unit ray direction leaving the mirror toward the target.

    Returns:
        Unit normal ``n`` with ``n · incoming < 0`` (front face toward the sun).
    """
    incoming = normalize(incoming_toward_mirror)
    outgoing = normalize(outgoing_from_mirror)
    normal = normalize(incoming - outgoing)
    if float(np.dot(incoming, normal)) > 0.0:
        normal = -normal
    return normal


def reflect_ray(incoming_toward_mirror: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect an incoming unit ray about a unit surface normal."""
    n = normalize(normal)
    d = normalize(incoming_toward_mirror)
    return normalize(d - 2.0 * float(np.dot(d, n)) * n)
