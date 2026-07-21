"""Shared Hot-Box plant constants and mirror pointing geometry."""

from .aiming import (
    MirrorGridSpec,
    MountAngles,
    bisector_normal_at_mount,
    solve_bisector_tracking,
    solve_bisector_tracking_for_grid,
)
from .load import default_system_yaml_path, load_system_constants
from .mount import (
    facet_normal_world,
    heading_and_tilt_from_normal,
    mount_az_el_align_body_normal_to_world,
    mount_rotation_matrix,
    normalize_mount_az_el,
    pivot_facet_normal_body,
)
from .system import (
    AbsorberConstants,
    ControlConstants,
    FleetConstants,
    MirrorConstants,
    MountDesign,
    SiteConstants,
    SystemConstants,
)
from .vectors import bisector_normal, normalize, reflect_ray

__all__ = [
    "AbsorberConstants",
    "ControlConstants",
    "FleetConstants",
    "MirrorConstants",
    "MirrorGridSpec",
    "MountAngles",
    "MountDesign",
    "SiteConstants",
    "SystemConstants",
    "bisector_normal",
    "bisector_normal_at_mount",
    "default_system_yaml_path",
    "facet_normal_world",
    "heading_and_tilt_from_normal",
    "load_system_constants",
    "mount_az_el_align_body_normal_to_world",
    "mount_rotation_matrix",
    "normalize",
    "normalize_mount_az_el",
    "pivot_facet_normal_body",
    "reflect_ray",
    "solve_bisector_tracking",
    "solve_bisector_tracking_for_grid",
]
