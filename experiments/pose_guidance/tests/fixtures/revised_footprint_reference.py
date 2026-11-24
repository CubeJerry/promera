"""Test-only numerical reference from the revised footprint_angle_audit.zip.

The constants and geometry functions below retain the archive's arithmetic.
They are deliberately independent of the production module. grid_surface is
the supplied audit-test fixture. This is not the superseded MILTON estimator.
Source: footprint_angle_audit.zip, ChatGPT Library version 1, file
file_000000001098820b88565bee34a9fdfd, last modified 2026-11-24T11:41:02Z.
The archive includes historical smoke CSVs but not those coordinate files.
"""
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

CONTACT_A = 5.0
SCALES_A = (4.0, 6.0)
MIN_SITES = 4
MIN_GRADIENT = 0.15       # dimensionless |sigma * grad(log density)|
MIN_COHERENCE = 0.75     # length of the mean of unit outward normals
MAX_SCALE_DEG = 20.0
MAX_P80_DEG = 55.0
MIN_LATERAL_RMS_A = 1.5
CONNECT_A = 12.0


def unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("undefined direction")
    return vector / length


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def elevation(axis_to_framework, outward):
    return float(np.degrees(np.arcsin(np.clip(unit(axis_to_framework) @ unit(outward), -1, 1))))


def footprint_geometry(tree, sites, axis_to_framework):
    """Patch tangent approximation from the antigen density field at contact sites."""
    tree_sites = cKDTree(sites)
    links = tree_sites.sparse_distance_matrix(tree_sites, CONNECT_A, output_type="coo_matrix")
    components = int(connected_components(links, directed=False, return_labels=False))
    center = sites.mean(0)
    qc = dict(zip(("footprint_center_x", "footprint_center_y", "footprint_center_z"), center))
    qc.update(components=components)
    flags = []
    if len(sites) < MIN_SITES:
        flags.append("few_contact_residues")
    if components > 1:
        flags.append("disconnected_footprint")
    axis = unit(axis_to_framework)
    scales, coherence, spread, gradients, sensitivity = [], [], [], [], []
    for sigma in SCALES_A:
        vectors = []
        for site, ids in zip(sites, tree.query_ball_point(sites, 3 * sigma + 1e-8)):
            if not len(ids):
                vectors.append(np.zeros(3))
                gradients.append(0.)
                continue
            displacement = (site - tree.data[ids]) / sigma
            w = np.exp(-0.5 * np.einsum("ij,ij->i", displacement, displacement))
            g = np.average(displacement, axis=0, weights=w)
            strength = float(np.linalg.norm(g))
            gradients.append(strength)
            # Strong sites still get one vote. A weak gradient contributes
            # proportionately instead of vetoing the rest of the interface.
            vectors.append(g / max(strength, MIN_GRADIENT))
        vectors = np.asarray(vectors)
        lengths = np.linalg.norm(vectors, axis=1)
        total = vectors.sum(0)
        if lengths.mean() < 0.15:
            return dict(qc, status="no_resolved_outward_direction")
        coherence.append(float(np.linalg.norm(total) / lengths.sum()))
        if coherence[-1] < 0.15:
            return dict(qc, status="opposing_surface_normals", coherence_min=min(coherence))
        normal = unit(total)
        valid = lengths > 1e-8
        deviations = np.degrees(np.arccos(np.clip((vectors[valid] @ normal) / lengths[valid], -1, 1)))
        spread.append(float(np.percentile(deviations, 80)))
        scales.append(normal)
        sensitivity.append(elevation(axis, normal))
        if len(sites) > 1:
            omitted = total - vectors
            norms = np.linalg.norm(omitted, axis=1)
            valid = norms > 1e-8
            if not np.all(valid):
                flags.append("single_site_determines_normal")
            if valid.any():
                sensitivity.extend(np.degrees(np.arcsin(np.clip((omitted[valid] @ axis) / norms[valid], -1, 1))))

    qc.update(coherence_min=min(coherence), normal_p80_max_deg=max(spread),
              min_local_gradient=min(gradients), scale_disagreement_deg=angle(*scales),
              elevation_sigma4_deg=elevation(axis, scales[0]), elevation_sigma6_deg=elevation(axis, scales[1]))
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
    qc["lateral_rms_min_A"] = float(widths[1]) if len(widths) > 1 else 0.
    if qc["lateral_rms_min_A"] < MIN_LATERAL_RMS_A:
        flags.append("narrow_footprint")
    signed = elevation(axis, normal)
    sensitivity.append(signed)
    low, high = float(min(sensitivity)), float(max(sensitivity))
    flat = (max(abs(low), abs(high)) <= 5.0 + 1e-8
            and not {"few_contact_residues", "single_site_determines_normal",
                     "divergent_surface_normals", "scale_sensitive_normal"}.intersection(flags))
    qc.update(zip(("outward_nx", "outward_ny", "outward_nz"), normal))
    return dict(qc, status="ok", detail="|".join(dict.fromkeys(flags)),
                sensitivity_min_deg=low, sensitivity_max_deg=high, pose_label="flat" if flat else "",
                footprint_elevation_deg=signed, inclination_deg=abs(signed))


def grid_surface(curve=0):
    # Smooth a uniformly populated solid with a known (possibly curved) top.
    x, y, depth = np.meshgrid(np.arange(-30, 31, 2.), np.arange(-30, 31, 2.),
                              np.arange(-28, 1, 2.), indexing="ij")
    z = depth + curve * (x*x + y*y)
    sites = np.array([[a, b, curve*(a*a+b*b)] for a in (-4., 0., 4.) for b in (-4., 0., 4.)])
    return np.column_stack((x.ravel(), y.ravel(), z.ravel())), sites
