"""Shared Hot-Box plant constants and mirror pointing geometry."""

from .aiming import (
    CenterRay,
    MirrorGridSpec,
    MountAngles,
    bisector_normal_at_mount,
    evaluate_center_ray,
    pivot_facet_center_world,
    refine_tracking_for_mount_offset,
    solve_bisector_tracking,
    solve_bisector_tracking_for_grid,
    solve_tracking,
    solve_tracking_for_grid,
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
    "CenterRay",
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
    "evaluate_center_ray",
    "facet_normal_world",
    "heading_and_tilt_from_normal",
    "load_system_constants",
    "mount_az_el_align_body_normal_to_world",
    "mount_rotation_matrix",
    "normalize",
    "normalize_mount_az_el",
    "pivot_facet_center_world",
    "pivot_facet_normal_body",
    "refine_tracking_for_mount_offset",
    "reflect_ray",
    "solve_bisector_tracking",
    "solve_bisector_tracking_for_grid",
    "solve_tracking",
    "solve_tracking_for_grid",
]
