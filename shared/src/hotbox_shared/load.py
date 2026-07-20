from __future__ import annotations

from pathlib import Path

import yaml

from .system import (
    AbsorberConstants,
    ControlConstants,
    FleetConstants,
    MirrorConstants,
    MountDesign,
    SiteConstants,
    SystemConstants,
)


def default_system_yaml_path() -> Path:
    """Repo-level config/system.yaml when running from a checkout."""
    here = Path(__file__).resolve()
    # shared/src/hotbox_shared/load.py -> repo root
    repo_root = here.parents[3]
    return repo_root / "config" / "system.yaml"


def load_system_constants(path: Path | None = None) -> SystemConstants:
    yaml_path = path or default_system_yaml_path()
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"system constants file not found: {yaml_path}. "
            "Expected config/system.yaml at the repository root."
        )
    raw = yaml.safe_load(yaml_path.read_text()) or {}
    return system_constants_from_dict(raw)


def system_constants_from_dict(raw: dict) -> SystemConstants:
    site = SiteConstants(**raw["default_site"])
    absorber = AbsorberConstants(**raw["absorber"])
    mirror = MirrorConstants(**raw["mirror"])
    mounts = tuple(MountDesign(**item) for item in raw["fleet"]["mounts"])
    fleet = FleetConstants(
        assembly_count=int(raw["fleet"]["assembly_count"]),
        assembly_spacing_m=float(raw["fleet"]["assembly_spacing_m"]),
        mounts=mounts,
    )
    control = ControlConstants(**raw["control"])
    if fleet.assembly_count != len(fleet.mounts):
        raise ValueError(
            f"fleet.assembly_count ({fleet.assembly_count}) != number of mounts ({len(fleet.mounts)})"
        )
    return SystemConstants(default_site=site, absorber=absorber, mirror=mirror, fleet=fleet, control=control)
