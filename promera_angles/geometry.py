"""Rigid-motion equivariant geometry. Coordinates in Å; residue keys are external."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from experiments.pose_guidance.pose_geometry import (
    MIN_ANCHOR_SEPARATION_A,
    AnchorClash,
    farthest_points,
    xyz,
)


class GeometryError(ValueError):
    """The selected coordinates do not determine an interpretable interface."""


def unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-7:
        raise GeometryError("undefined direction")
    return vector / length


def degrees(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def extreme(values, largest=True):
    """Break equal geometric choices by residue order even after a rigid rotation."""
    best = np.max(values) if largest else np.min(values)
    return int(np.flatnonzero(np.isclose(values, best, atol=1e-9, rtol=1e-10))[0])


def patch_frame(coords, selected):
    """Estimate the local empty half-space at each selected Cα, then pool signs."""
    coords = xyz(coords)
    ids = np.asarray(selected, dtype=int)
    if (
        not len(ids)
        or min(ids) < 0
        or max(ids) >= len(coords)
        or len(set(ids)) != len(ids)
    ):
        raise GeometryError("invalid patch ordinals")
    sites = coords[ids]
    tree = cKDTree(coords)
    local_normals = []
    for point in sites:
        normals = []
        for radius in (10.0, 15.0):
            delta = coords[tree.query_ball_point(point, radius + 1e-8)] - point
            lengths = np.linalg.norm(delta, axis=1)
            delta = delta[lengths > 1e-6]
            if len(delta) < 6:
                raise GeometryError("under-supported patch or exposed terminus")
            directions = delta / np.linalg.norm(delta, axis=1)[:, None]
            resultant = -directions.mean(0)
            if np.linalg.norm(resultant) < 0.22:
                raise GeometryError(
                    "buried, two-sided or ambiguously signed neighbourhood"
                )
            normal = unit(resultant)
            # A chain end or ridge cannot determine a surface from one strand.
            lateral = directions - np.outer(directions @ normal, normal)
            values = np.linalg.eigvalsh(lateral.T @ lateral / len(lateral))
            if values[-2] < 0.11:
                raise GeometryError("insufficient lateral support at patch or terminus")
            normals.append(normal)
        if degrees(*normals) > 40:
            raise GeometryError("surface normal changes with neighbourhood radius")
        local_normals.append(unit(sum(normals)))
    # Do not average disconnected sites into an unoccupied centre.
    visited, pending = {0}, [0]
    neighbours = cKDTree(sites).query_ball_tree(cKDTree(sites), 14.0)
    while pending:
        for index in neighbours[pending.pop()]:
            if index not in visited:
                visited.add(index)
                pending.append(index)
    if len(visited) != len(sites):
        raise GeometryError("disconnected epitope patches")
    normal = unit(np.mean(local_normals, axis=0))
    spread = max(degrees(n, normal) for n in local_normals)
    if spread > 45:
        raise GeometryError("incompatible local surface normals across patch")
    origin = sites.mean(0)
    offset = coords - origin
    height = offset @ normal
    lateral = offset - np.outer(height, normal)
    if np.any(
        (height > 4.0) & (height < 24.0) & (np.linalg.norm(lateral, axis=1) < 4.0)
    ):
        raise GeometryError("recessed or occluded central approach")
    near = np.flatnonzero(np.linalg.norm(offset, axis=1) <= 15.0 + 1e-8)
    if len(near) < 4:
        raise GeometryError("too few resolved local landmarks")
    # Noncoplanar anchors distinguish mirrored placements at a planar surface.
    first = int(near[extreme(np.linalg.norm(offset[near], axis=1), False)])
    a = coords[first]
    diff = coords[near] - a
    second = int(near[extreme(np.linalg.norm(diff, axis=1))])
    line = unit(coords[second] - a)
    third = int(near[extreme(np.linalg.norm(np.cross(diff, line), axis=1))])
    plane = unit(np.cross(line, coords[third] - a))
    fourth = int(near[extreme(abs(diff @ plane))])
    if (
        len({first, second, third, fourth}) != 4
        or abs((coords[fourth] - a) @ plane) < 2.0
    ):
        raise GeometryError(
            "near-coplanar target anchors cannot resolve a mirrored guide"
        )
    # Use structural coordinates for azimuth zero to avoid a lab-frame bias.
    tangent = unit(lateral[near[extreme(np.linalg.norm(lateral[near], axis=1))]])
    return dict(
        origin=origin.tolist(),
        normal=normal.tolist(),
        tangent=tangent.tolist(),
        landmarks=[first, second, third, fourth],
        neighbourhood=near.tolist(),
        normal_spread_deg=spread,
        site_normals=np.asarray(local_normals).tolist(),
    )


def basis(axis, transverse):
    z = unit(axis)
    x = unit(transverse - (transverse @ z) * z)
    return np.stack((x, np.cross(z, x), z))


def scaffold_shape(ca, fixed, cdr):
    ca = xyz(ca)
    framework = ca[fixed]
    tip = ca[cdr].mean(0)
    inward = unit(tip - framework.mean(0))
    d = framework - tip
    lateral = d - np.outer(d @ inward, inward)
    axes = basis(inward, lateral[extreme(np.linalg.norm(lateral, axis=1))])
    anchors = farthest_points(framework, tie_tolerance=1e-8)
    return dict(
        fixed_indices=anchors,
        local_xyz=((framework[anchors] - tip) @ axes.T).tolist(),
        scaffold_ca_local=((ca - tip) @ axes.T).tolist(),
        framework_indices=list(map(int, fixed)),
        cdr_indices=list(map(int, cdr)),
    )


def placement(frame, shape, interval, seed, standoff=6.0, *, target_anchors=None):
    lo, hi = interval
    if (
        not np.isfinite([lo, hi, standoff]).all()
        or not 0 <= lo <= hi <= 180
        or standoff <= 0
    ):
        raise ValueError("invalid angular bounds or CDR-centroid standoff")
    rng = np.random.default_rng(seed)
    cos_theta = rng.uniform(np.cos(np.radians(hi)), np.cos(np.radians(lo)))
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
    phi, roll = rng.uniform(0, 2 * np.pi, size=2)
    x, y, z = basis(np.asarray(frame["normal"]), np.asarray(frame["tangent"]))
    radial = np.cos(phi) * x + np.sin(phi) * y
    ray = cos_theta * z + sin_theta * radial
    transverse = -np.sin(phi) * x + np.cos(phi) * y
    tip = np.asarray(frame["origin"]) + standoff * ray
    local = xyz(shape["local_xyz"])
    targets = None if target_anchors is None else xyz(target_anchors)
    # Test the original roll first, then traverse a circle in 15-degree steps.
    # All candidates have the same tilt, azimuth, and CDR-centroid standoff.
    for step in range(24):
        candidate = (roll + step * np.pi / 12) % (2 * np.pi)
        rotation = basis(
            -ray,
            np.cos(candidate) * transverse
            + np.sin(candidate) * np.cross(-ray, transverse),
        )
        anchors = local @ rotation + tip
        if targets is not None:
            distances = np.linalg.norm(anchors[:, None] - targets[None, :], axis=-1)
            if distances.min() <= MIN_ANCHOR_SEPARATION_A:
                continue
        return anchors, dict(
            requested_deg=float(np.degrees(np.arccos(cos_theta))),
            ray=ray.tolist(),
            azimuth_deg=float(np.degrees(phi)),
            roll_deg=float(np.degrees(candidate)),
            roll_attempts=step + 1,
            **scaffold_placement_record(shape, rotation, tip, frame),
            guidance_definition="legacy_placement_ray",
            requested_legacy_tilt_deg=float(np.degrees(np.arccos(cos_theta))),
            standoff_A=float(standoff),
        )
    raise AnchorClash(
        "all 24 sampled rolls have a cross-chain anchor separation <= 1 A"
    )


def orientation(frame, fw, cdr):
    origin, outward = np.asarray(frame["origin"]), np.asarray(frame["normal"])
    center, tip = xyz(fw).mean(0), xyz(cdr).mean(0)
    return dict(
        spatial_deg=degrees(center - origin, outward),
        facing_deg=degrees(tip - center, -outward),
        pointing_to_patch_deg=degrees(tip - center, origin - center),
        framework_to_patch_A=float(np.linalg.norm(center - origin)),
    )


def signed_elevation(framework_ca: Any, cdr_ca: Any, normal: Any) -> float:
    """Measure signed C→F elevation from all supplied framework and CDR Cαs."""
    axis = unit(xyz(framework_ca).mean(0) - xyz(cdr_ca).mean(0))
    return float(np.degrees(np.arcsin(np.clip(axis @ unit(np.asarray(normal)), -1, 1))))


def scaffold_placement_record(
    shape: dict[str, Any], rotation: Any, tip: Any, frame: dict[str, Any]
) -> dict[str, Any]:
    """Verify full scaffold centroids; old four-anchor plans remain legacy."""
    if "scaffold_ca_local" not in shape:
        return {}
    local = xyz(shape["scaffold_ca_local"])
    fw_array = np.asarray(shape["framework_indices"])
    cdr_array = np.asarray(shape["cdr_indices"])
    marks = np.asarray(shape["fixed_indices"])
    for indices in (fw_array, cdr_array):
        if (
            indices.ndim != 1
            or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
            or (indices < 0).any()
            or (indices >= len(local)).any()
        ):
            raise GeometryError("invalid full-scaffold framework/CDR mapping")
    combined = np.concatenate((fw_array, cdr_array))
    if len(combined) != len(local) or len(np.unique(combined)) != len(local):
        raise GeometryError(
            "framework/CDR mapping must partition the complete scaffold"
        )
    if (
        marks.shape != (4,)
        or not np.issubdtype(marks.dtype, np.integer)
        or len(np.unique(marks)) != 4
        or (marks < 0).any()
        or (marks >= len(fw_array)).any()
    ):
        raise GeometryError("invalid framework landmark mapping")
    encoded_local = xyz(shape["local_xyz"])
    expected_local = local[fw_array[marks]]
    if encoded_local.shape != expected_local.shape or not np.allclose(
        encoded_local, expected_local, atol=1e-7, rtol=0
    ):
        raise GeometryError(
            "framework landmarks disagree with full scaffold coordinates"
        )
    transformed = local @ rotation + tip
    fw, cdr = fw_array.tolist(), cdr_array.tolist()
    center = transformed[cdr].mean(0)
    if not np.allclose(center, tip, atol=1e-7, rtol=0):
        raise GeometryError(
            "scaffold CDR centroid is inconsistent with the placement origin"
        )
    return dict(
        realized_scaffold_elevation_deg=signed_elevation(
            transformed[fw], transformed[cdr], frame["normal"]
        ),
        scaffold_framework_centroid_xyz=transformed[fw].mean(0).tolist(),
        scaffold_cdr_centroid_xyz=center.tolist(),
        scaffold_ca_xyz=transformed.tolist(),
        scaffold_framework_indices=fw,
        scaffold_cdr_indices=cdr,
        patch_frame=frame,
    )


def elevation_placement(
    frame: dict[str, Any],
    shape: dict[str, Any],
    interval: Any,
    seed: int,
    standoff: float = 6.0,
    *,
    target_anchors: Any | None = None,
    cdr_lateral_offset: Any = (0.0, 0.0),
) -> tuple[Any, dict[str, Any]]:
    """Rotate the body about fixed C; sample elevation, azimuth and axial roll.

    C = origin + standoff * n + optional tangent offsets, independently of body
    inclination. Elevations are uniform in degrees; azimuth/roll are uniform on
    the circle. Roll-only clash retries preserve elevation, azimuth and C.
    """
    lo, hi = map(float, interval)
    offset = np.asarray(cdr_lateral_offset, dtype=float)
    if (
        not np.isfinite([lo, hi, standoff]).all()
        or not -90 <= lo <= hi <= 90
        or standoff <= 0
        or offset.shape != (2,)
        or not np.isfinite(offset).all()
    ):
        raise ValueError(
            "require -90 <= elevation minimum <= maximum <= 90 and positive standoff"
        )
    if "scaffold_ca_local" not in shape:
        raise ValueError(
            "surface elevation requires a scaffold with complete framework/CDR mapping"
        )
    rng = np.random.default_rng(seed)
    requested = float(rng.uniform(lo, hi))
    phi, roll = rng.uniform(0, 2 * np.pi, size=2)
    x, y, normal = basis(np.asarray(frame["normal"]), np.asarray(frame["tangent"]))
    tip = (
        np.asarray(frame["origin"]) + standoff * normal + offset[0] * x + offset[1] * y
    )
    radial = np.cos(phi) * x + np.sin(phi) * y
    outward_axis = (
        np.sin(np.radians(requested)) * normal + np.cos(np.radians(requested)) * radial
    )
    inward_axis = -outward_axis  # scaffold local +z is F→C, body axis is C→F.
    transverse = -np.sin(phi) * x + np.cos(phi) * y
    targets = None if target_anchors is None else xyz(target_anchors)
    for step in range(24):
        candidate = (roll + step * np.pi / 12) % (2 * np.pi)
        rotation = basis(
            inward_axis,
            np.cos(candidate) * transverse
            + np.sin(candidate) * np.cross(inward_axis, transverse),
        )
        anchors = xyz(shape["local_xyz"]) @ rotation + tip
        if targets is not None:
            distances = np.linalg.norm(anchors[:, None] - targets[None, :], axis=-1)
            if distances.min() <= MIN_ANCHOR_SEPARATION_A:
                continue
        record = scaffold_placement_record(shape, rotation, tip, frame)
        error = float(record["realized_scaffold_elevation_deg"] - requested)
        if abs(error) > 2e-6:
            raise GeometryError(
                f"transformed scaffold elevation verification failed: {error} degrees"
            )
        return anchors, dict(
            record,
            guidance_definition="surface_elevation",
            requested_elevation_deg=requested,
            requested_elevation_interval_deg=[lo, hi],
            scaffold_elevation_signed_error_deg=error,
            scaffold_elevation_absolute_error_deg=abs(error),
            elevation_sampling="uniform_degrees",
            body_axis_c_to_f=outward_axis.tolist(),
            cdr_placement_definition="fixed_centroid_at_patch_normal_standoff",
            cdr_lateral_offset_A=offset.tolist(),
            standoff_A=float(standoff),
            azimuth_deg=float(np.degrees(phi)),
            roll_deg=float(np.degrees(candidate)),
            roll_attempts=step + 1,
        )
    raise AnchorClash(
        "all 24 sampled rolls have a cross-chain anchor separation <= 1 A"
    )


def prospective_patch_frame(
    coords: Any,
    selected: Any,
    antigen_heavy_xyz: Any,
    sampling_points: Any,
    *,
    landmarks: list[int] | None = None,
) -> dict[str, Any]:
    """Build a prospective plane from one heavy-atom centroid per patch residue.

    The origin is the unweighted mean of these sites. Density uses all represented
    atoms on selected antigen chains. No scaffold or observed contacts enter this
    estimate. Nonplanar landmarks are preferred; supported planar patches retain
    numeric elevation with a reflection-ambiguity diagnostic.
    """
    from .footprint import estimate_surface_normal

    target, sites = xyz(coords), xyz(sampling_points)
    if len(sites) != len(selected) or not len(selected):
        raise GeometryError(
            "one prospective sampling point is required per selected residue"
        )
    diagnostics = estimate_surface_normal(antigen_heavy_xyz, sites)
    if diagnostics["status"] != "ok":
        raise GeometryError(f"prospective patch: {diagnostics['status']}")
    normal = np.asarray(diagnostics["normal"], dtype=float)
    origin = sites.mean(0)
    offset = target - origin
    near = np.flatnonzero(np.linalg.norm(offset, axis=1) <= 15.0 + 1e-8)
    if len(near) < 4:
        raise GeometryError("too few local target landmarks for sparse conditioning")
    lateral = offset - np.outer(offset @ normal, normal)
    tangent = unit(lateral[near[extreme(np.linalg.norm(lateral[near], axis=1))]])
    if landmarks is None:
        first = int(near[extreme(np.linalg.norm(offset[near], axis=1), False)])
        difference = target[near] - target[first]
        second = int(near[extreme(np.linalg.norm(difference, axis=1))])
        line = unit(target[second] - target[first])
        third = int(near[extreme(np.linalg.norm(np.cross(difference, line), axis=1))])
        plane = unit(np.cross(line, target[third] - target[first]))
        fourth = int(near[extreme(abs(difference @ plane))])
        landmarks = [first, second, third, fourth]
        if len(set(landmarks)) != 4:
            landmarks = [
                int(near[i]) for i in farthest_points(target[near], tie_tolerance=1e-8)
            ]
    rank = int(
        np.linalg.matrix_rank(target[landmarks] - target[landmarks].mean(0), tol=1e-6)
    )
    return dict(
        origin=origin.tolist(),
        normal=normal.tolist(),
        tangent=tangent.tolist(),
        landmarks=landmarks,
        neighbourhood=near.tolist(),
        origin_definition="mean_of_selected_residue_heavy_atom_centroids",
        sampling_definition="one_all_heavy_atom_centroid_per_selected_antigen_residue",
        sampling_points=sites.tolist(),
        density_scales_A=[4.0, 6.0],
        normal_estimator="revised_footprint_angle_audit_v1",
        diagnostics=diagnostics,
        target_landmark_rank=rank,
        target_landmark_reflection_ambiguous=rank < 3,
    )
