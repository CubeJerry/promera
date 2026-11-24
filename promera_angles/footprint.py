"""Signed VHH elevation using the revised footprint_angle_audit estimator.

The same antigen-only density estimator supports prospective patch construction
and independent observed-footprint measurements. Atom inputs are represented
heavy atoms. Numerical reference fixtures live under the experiment tests.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

CONTACT_A = 5.0
SCALES_A = (4.0, 6.0)
MIN_SITES = 4
MIN_GRADIENT = 0.15
MIN_COHERENCE = 0.75
MAX_SCALE_DEG = 20.0
MAX_P80_DEG = 55.0
MIN_LATERAL_RMS_A = 1.5
CONNECT_A = 12.0
ESTIMATOR = "revised_footprint_density_4_6A_v1"
ResidueKey: TypeAlias = tuple[str, int]
FloatArray: TypeAlias = NDArray[np.float64]


def unit(vector: ArrayLike) -> FloatArray:
    """Normalize a vector, rejecting missing or numerically cancelled axes."""
    value = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(length) or length < 1e-9:
        raise ValueError("undefined direction")
    return value / length


def angle(a: ArrayLike, b: ArrayLike) -> float:
    """Return the angle between two directed vectors in degrees."""
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def elevation(axis_to_framework: ArrayLike, outward: ArrayLike) -> float:
    """Return signed C-to-F elevation; positive means CDR end faces inward."""
    return float(
        np.degrees(np.arcsin(np.clip(unit(axis_to_framework) @ unit(outward), -1, 1)))
    )


def _surface_geometry(
    tree: cKDTree, sites: FloatArray, axis_to_framework: ArrayLike | None
) -> dict[str, Any]:
    """Keep the revised audit's arithmetic, thresholds and diagnostic meanings."""
    if not len(sites) or not len(tree.data) or not np.isfinite(sites).all():
        return {"status": "missing_coordinates"}
    tree_sites = cKDTree(sites)
    links = tree_sites.sparse_distance_matrix(
        tree_sites, CONNECT_A, output_type="coo_matrix"
    )
    components = int(connected_components(links, directed=False, return_labels=False))
    center = sites.mean(0)
    qc: dict[str, Any] = dict(
        zip(("footprint_center_x", "footprint_center_y", "footprint_center_z"), center)
    )
    qc.update(components=components)
    flags: list[str] = []
    if len(sites) < MIN_SITES:
        flags.append("few_contact_residues")
    if components > 1:
        flags.append("disconnected_footprint")
    axis = None if axis_to_framework is None else unit(axis_to_framework)
    scales: list[FloatArray] = []
    coherence: list[float] = []
    spread: list[float] = []
    gradients: list[float] = []
    sensitivity: list[float] = []
    sensitivity_normals: list[FloatArray] = []
    for sigma in SCALES_A:
        local_vectors = []
        for site, ids in zip(sites, tree.query_ball_point(sites, 3 * sigma + 1e-8)):
            if not len(ids):
                local_vectors.append(np.zeros(3))
                gradients.append(0.0)
                continue
            displacement = (site - tree.data[ids]) / sigma
            weights = np.exp(-0.5 * np.einsum("ij,ij->i", displacement, displacement))
            gradient = np.average(displacement, axis=0, weights=weights)
            strength = float(np.linalg.norm(gradient))
            gradients.append(strength)
            # Strong sites get one vote; weak sites contribute proportionately.
            local_vectors.append(gradient / max(strength, MIN_GRADIENT))
        vectors = np.asarray(local_vectors)
        lengths = np.linalg.norm(vectors, axis=1)
        total = vectors.sum(0)
        if lengths.mean() < 0.15:
            return dict(qc, status="no_resolved_outward_direction")
        coherence.append(float(np.linalg.norm(total) / lengths.sum()))
        if coherence[-1] < 0.15:
            return dict(
                qc, status="opposing_surface_normals", coherence_min=min(coherence)
            )
        normal = unit(total)
        valid = lengths > 1e-8
        deviations = np.degrees(
            np.arccos(np.clip((vectors[valid] @ normal) / lengths[valid], -1, 1))
        )
        spread.append(float(np.percentile(deviations, 80)))
        scales.append(normal)
        sensitivity_normals.append(normal)
        if axis is not None:
            sensitivity.append(elevation(axis, normal))
        if len(sites) > 1:
            omitted = total - vectors
            norms = np.linalg.norm(omitted, axis=1)
            valid = norms > 1e-8
            if not np.all(valid):
                flags.append("single_site_determines_normal")
            if valid.any():
                sensitivity_normals.extend(omitted[valid] / norms[valid, None])
                if axis is not None:
                    sensitivity.extend(
                        np.degrees(
                            np.arcsin(
                                np.clip((omitted[valid] @ axis) / norms[valid], -1, 1)
                            )
                        ).tolist()
                    )
    qc.update(
        coherence_min=min(coherence),
        normal_p80_max_deg=max(spread),
        min_local_gradient=min(gradients),
        scale_disagreement_deg=angle(*scales),
    )
    if axis is not None:
        qc.update(
            elevation_sigma4_deg=elevation(axis, scales[0]),
            elevation_sigma6_deg=elevation(axis, scales[1]),
        )
    if scales[0] @ scales[1] <= 0:
        return dict(qc, status="incompatible_directions_across_scales")
    if min(gradients) < MIN_GRADIENT:
        flags.append("weak_local_gradient")
    if min(coherence) < MIN_COHERENCE or max(spread) > MAX_P80_DEG:
        flags.append("divergent_surface_normals")
    if qc["scale_disagreement_deg"] > MAX_SCALE_DEG:
        flags.append("scale_sensitive_normal")
    normal = unit(scales[0] + scales[1])
    offsets = sites - center
    projected = offsets - np.outer(offsets @ normal, normal)
    widths = np.linalg.svd(projected, compute_uv=False) / np.sqrt(len(sites))
    qc["lateral_rms_min_A"] = float(widths[1]) if len(widths) > 1 else 0.0
    if qc["lateral_rms_min_A"] < MIN_LATERAL_RMS_A:
        flags.append("narrow_footprint")
    qc.update(zip(("outward_nx", "outward_ny", "outward_nz"), normal))
    sensitivity_normals.append(normal)
    qc.update(
        normal=normal.tolist(),
        scale_normals=np.asarray(scales).tolist(),
        sensitivity_normals=np.asarray(sensitivity_normals).tolist(),
        estimator=ESTIMATOR,
    )
    if axis is None:
        return dict(qc, status="ok", detail="|".join(dict.fromkeys(flags)))
    signed = elevation(axis, normal)
    sensitivity.append(signed)
    low, high = float(min(sensitivity)), float(max(sensitivity))
    flat = max(abs(low), abs(high)) <= 5.0 + 1e-8 and not {
        "few_contact_residues",
        "single_site_determines_normal",
        "divergent_surface_normals",
        "scale_sensitive_normal",
    }.intersection(flags)
    return dict(
        qc,
        status="ok",
        detail="|".join(dict.fromkeys(flags)),
        sensitivity_min_deg=low,
        sensitivity_max_deg=high,
        pose_label="flat" if flat else "",
        footprint_elevation_deg=signed,
        inclination_deg=abs(signed),
    )


def regions(
    seq: str, pattern: str, saved: Sequence[int] | None
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Use run labels, or a unique scaffold mapping around the three CDRs."""
    if re.findall(r"<cdrh[123]>", pattern) != ["<cdrh1>", "<cdrh2>", "<cdrh3>"]:
        raise ValueError("invalid framework pattern")
    parts = re.split(r"<cdrh[123]>", pattern)
    if not all(parts):
        raise ValueError("empty framework segment")
    if saved is not None:
        if any(type(p) is not int for p in saved):
            raise ValueError("expected integer saved CDR positions")
        mask = {p - 1 for p in saved}
        if (
            not mask
            or len(mask) != len(saved)
            or min(mask) < 0
            or max(mask) >= len(seq)
        ):
            raise ValueError("invalid saved CDR positions")
        matches = re.findall(
            r"[^|]+", "".join("|" if i in mask else aa for i, aa in enumerate(seq))
        )
        if matches != parts or 0 in mask or len(seq) - 1 in mask:
            raise ValueError("saved CDR mask does not match observed scaffold")
    else:
        p0, p1, p2, p3 = parts
        if not seq.startswith(p0) or not seq.endswith(p3):
            raise ValueError("sequence does not match framework ends")
        stop = len(seq) - len(p3)
        solutions = [
            (i, j)
            for i in range(len(p0) + 1, stop)
            if seq.startswith(p1, i)
            for j in range(i + len(p1) + 1, stop - len(p2))
            if seq.startswith(p2, j)
        ]
        if len(solutions) != 1:
            raise ValueError("framework mapping is absent or ambiguous")
        i, j = solutions[0]
        mask = (
            set(range(len(p0), i))
            | set(range(i + len(p1), j))
            | set(range(j + len(p2), stop))
        )
    return (
        np.array([i for i in range(len(seq)) if i not in mask], dtype=np.int64),
        np.array(sorted(mask), dtype=np.int64),
    )


def _coordinates(values: ArrayLike) -> FloatArray:
    """Normalize coordinates without silently dropping missing atoms."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("expected an N by 3 coordinate array")
    return array


def footprint_geometry(
    tree: cKDTree, sites: ArrayLike, axis_to_framework: ArrayLike
) -> dict[str, Any]:
    """Measure elevation from one contacted-antigen-atom centroid per residue.

    Args:
        tree: All represented heavy atoms of the selected antigen chains.
            This array must not contain binder atoms.
        sites: Contacted atom centroid for each chain-qualified antigen residue.
        axis_to_framework: Complete framework Cα centroid minus complete CDR Cα
            centroid, including CDR residues without contacts.

    Returns:
        Revised audit measurements, support diagnostics and sensitivity range.
        Undefined normals have a reason-specific status and no numeric angle.
    """
    try:
        return _surface_geometry(tree, _coordinates(sites), axis_to_framework)
    except ValueError as error:
        if str(error) == "undefined direction":
            return {"status": "undefined_vhh_axis"}
        raise


def estimate_surface_normal(antigen_xyz: ArrayLike, sites: ArrayLike) -> dict[str, Any]:
    """Estimate a prospective normal without a VHH or laboratory-frame axis.

    The caller chooses and records patch sampling points. Density scales,
    weak-gradient weighting, pooling and support checks match the observed
    footprint estimator. This function assigns no angle or flat pose label.
    """
    atoms, points = _coordinates(antigen_xyz), _coordinates(sites)
    if (
        not len(atoms)
        or not len(points)
        or not np.isfinite(atoms).all()
        or not np.isfinite(points).all()
    ):
        return {"status": "missing_coordinates"}
    return _surface_geometry(cKDTree(atoms), points, None)


def measure_footprint(
    antigen_xyz: ArrayLike,
    antigen_keys: Sequence[ResidueKey],
    binder_xyz: ArrayLike,
    binder_positions: Sequence[int],
    framework_ca: ArrayLike,
    cdr_ca: ArrayLike,
    cdr_positions: Sequence[int],
    *,
    antigen_ca: Mapping[ResidueKey, ArrayLike] | None = None,
) -> dict[str, Any]:
    """Independently measure the actual footprint of one backbone or refold.

    Positions are one-based runtime positions. Antigen keys retain chains.
    Coordinates must already exclude hydrogen, zero-occupancy and duplicate
    alternate atoms. Framework/CDR assignment is the validated run mapping;
    all provided Cα coordinates enter the unweighted centroids. Contacts
    include framework and CDR atoms. Nonfinite coordinates are not dropped.
    """
    at, bt = _coordinates(antigen_xyz), _coordinates(binder_xyz)
    fw, cdr = _coordinates(framework_ca), _coordinates(cdr_ca)
    if len(at) != len(antigen_keys) or len(bt) != len(binder_positions):
        raise ValueError("atom coordinates and residue labels have different lengths")
    if (
        not len(at)
        or not len(bt)
        or not all(np.isfinite(array).all() for array in (at, bt, fw, cdr))
    ):
        return {"status": "missing_coordinates"}
    keys = [(str(chain), int(position)) for chain, position in antigen_keys]
    tree = cKDTree(at)
    hits = tree.query_ball_point(bt, CONTACT_A)
    contact_atoms: set[int] = set()
    contacting_binder: set[int] = set()
    for position, indices in zip(binder_positions, hits):
        if indices:
            contact_atoms.update(indices)
            contacting_binder.add(int(position))
    groups: dict[ResidueKey, list[int]] = defaultdict(list)
    for index in sorted(contact_atoms):
        groups[keys[index]].append(index)
    cdr_count = len(contacting_binder & {int(position) for position in cdr_positions})
    result: dict[str, Any] = dict(
        footprint_residues=len(groups),
        contacting_cdr_residues=cdr_count,
        contacting_framework_residues=len(contacting_binder) - cdr_count,
        footprint_residue_keys=[list(key) for key in sorted(groups)],
        contacting_binder_positions=sorted(contacting_binder),
    )
    if not groups:
        return dict(result, status="no_contacts")
    result["cdr_contact_fraction"] = cdr_count / len(contacting_binder)
    sites = np.array([tree.data[groups[key]].mean(0) for key in sorted(groups)])
    result["sampling_points"] = sites.tolist()
    if not len(fw) or not len(cdr):
        return dict(result, status="missing_coordinates")
    f, c = fw.mean(0), cdr.mean(0)
    result.update(framework_centroid=f.tolist(), cdr_centroid=c.tolist())
    try:
        axis = unit(f - c)
    except ValueError:
        return dict(result, status="undefined_vhh_axis")
    if antigen_ca is not None and all(key in antigen_ca for key in groups):
        interface = np.mean([antigen_ca[key] for key in sorted(groups)], axis=0)
        if np.isfinite(interface).all() and np.linalg.norm(interface - f) > 1e-9:
            result["centroid_pointing_deg"] = angle(c - f, interface - f)
    return dict(result, **footprint_geometry(tree, sites, axis))
