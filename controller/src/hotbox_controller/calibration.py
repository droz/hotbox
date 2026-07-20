from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from .geometry import MirrorCalibration


def load_calibrations(path: Path) -> dict[int, MirrorCalibration]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    calibrations: dict[int, MirrorCalibration] = {}
    for item in raw.get("mirrors", []):
        calibration = MirrorCalibration(**item)
        calibrations[calibration.node_id] = calibration
    return calibrations


def save_calibrations(path: Path, calibrations: dict[int, MirrorCalibration]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mirrors": [asdict(calibration) for calibration in calibrations.values()]}
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
