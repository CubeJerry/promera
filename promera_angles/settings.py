"""Versioned study settings: historical angles always remain placement-ray tilts."""

from __future__ import annotations

from typing import Any

import numpy as np

SURFACE_DEFINITION = "surface_elevation_v1"


def validate_settings(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate without interpreting any historical interval as elevation."""
    result = dict(spec)
    allowed = {
        "enabled",
        "paired",
        "interval_deg",
        "backbones",
        "replicates",
        "seed",
        "target_structure",
        "standoff_A",
        "definition",
        "three_arms",
        "elevation_interval_deg",
        "legacy_tilt_interval_deg",
        "cdr_lateral_offset_A",
    }
    if set(result) - allowed:
        raise ValueError(
            f"unrecognised angle settings: {sorted(set(result) - allowed)}"
        )
    definition = result.get("definition")
    if definition not in (None, "legacy_placement_ray_v1", SURFACE_DEFINITION):
        raise ValueError(f"unrecognised angle definition: {definition}")
    surface = definition == SURFACE_DEFINITION or "elevation_interval_deg" in result
    for key in ("enabled", "paired", "three_arms"):
        if type(result.get(key, False)) is not bool:
            raise ValueError(f"{key} must be boolean")
    enabled = result.get("enabled", False)
    if (result.get("paired") or result.get("three_arms")) and not enabled:
        raise ValueError("paired or three_arms requires enabled")
    if result.get("paired") and result.get("three_arms"):
        raise ValueError("choose paired or three_arms, not both")
    if result.get("three_arms") and not surface:
        raise ValueError("three_arms requires an explicit elevation_interval_deg")
    if surface:
        if "interval_deg" in result:
            raise ValueError(
                "historical interval_deg is legacy tilt; specify elevation_interval_deg independently"
            )
        if definition == "legacy_placement_ray_v1":
            raise ValueError("legacy definition cannot carry elevation_interval_deg")
        result["definition"] = SURFACE_DEFINITION
    elif "legacy_tilt_interval_deg" in result:
        if "interval_deg" in result:
            raise ValueError("specify only one legacy tilt interval")
        result["interval_deg"] = result.pop("legacy_tilt_interval_deg")
    primary = "elevation_interval_deg" if surface else "interval_deg"
    required = [primary] if enabled else []
    if result.get("three_arms"):
        required.append("legacy_tilt_interval_deg")
    for key in set(required) | {
        "interval_deg",
        "elevation_interval_deg",
        "legacy_tilt_interval_deg",
    }:
        values = result.get(key)
        if key not in required and values is None:
            continue
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"provide two {key} endpoints")
        low, high = map(float, values)
        minimum, maximum = (-90, 90) if key == "elevation_interval_deg" else (0, 180)
        if not np.isfinite([low, high]).all() or not minimum <= low <= high <= maximum:
            raise ValueError(
                f"{key} requires {minimum} <= minimum <= maximum <= {maximum}"
            )
        result[key] = [low, high]
    standoff = float(result.get("standoff_A", 6))
    if not np.isfinite(standoff) or standoff <= 0:
        raise ValueError("standoff_A must be positive and finite")
    result["standoff_A"] = standoff
    offset = np.asarray(result.get("cdr_lateral_offset_A", [0, 0]), dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("cdr_lateral_offset_A requires two finite tangent offsets")
    if not surface and np.any(offset):
        raise ValueError("independent CDR lateral offsets require surface elevation")
    result["cdr_lateral_offset_A"] = offset.tolist()
    return result


def study_arms(spec: dict[str, Any]) -> list[str]:
    """Return explicit new arms while preserving historical arm identifiers."""
    if spec.get("definition") == SURFACE_DEFINITION:
        if spec.get("three_arms"):
            return ["ordinary", "legacy_tilt", "surface_elevation"]
        if spec.get("paired"):
            return ["ordinary", "surface_elevation"]
        return ["surface_elevation" if spec.get("enabled") else "ordinary"]
    if spec.get("paired"):
        return ["ordinary", "guided"]
    return ["guided" if spec.get("enabled") else "ordinary"]
