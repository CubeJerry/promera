"""Measure signed surface elevation while preserving every requested parent.

The positional interval argument is always legacy placement-ray tilt. New
elevation endpoints require explicit elevation metadata; old results are never
silently reinterpreted.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from experiments.pose_guidance.pose_geometry import AA3, align, rmsd

from .footprint import elevation, measure_footprint, regions
from .geometry import GeometryError, degrees, orientation, patch_frame
from .report import elevation_tables, experiment_report
from .structures import atom_table

Row = dict[str, Any]


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON with unavailable measurements represented as null."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[Row], required: list[str]) -> None:
    """Serialize structured diagnostics as JSON within CSV cells."""
    fields = list(dict.fromkeys([*required, *(key for row in rows for key in row)]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    json.dumps(value, separators=(",", ":"), allow_nan=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        )


def positions(table: Any) -> np.ndarray:
    """Extract represented coordinates in row order."""
    return table[["Cartn_x", "Cartn_y", "Cartn_z"]].to_numpy(float)


def finite_number(value: Any) -> float | None:
    """Accept finite measurements, excluding booleans."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if np.isfinite(number) else None


def in_interval(value: Any, interval: Any) -> bool | None:
    """Preserve missing measurements separately from out-of-band values."""
    number = finite_number(value)
    return (
        None
        if number is None or interval is None
        else bool(interval[0] - 1e-7 <= number <= interval[1] + 1e-7)
    )


def measurements(path: Path, plan: Row, cdr_ids: Any) -> tuple[Row, list[Row], Any]:
    """Independently measure the footprint and compare the prospective plane."""
    atoms = atom_table(path)
    antigen = atoms[atoms.label_asym_id.isin(plan["sequences"])]
    binder = atoms[atoms.label_asym_id == plan["binder_chain"]]
    bca = binder[binder.label_atom_id == "CA"].sort_values("pos")
    if not len(bca) or list(bca.pos) != list(range(1, len(bca) + 1)):
        raise ValueError("missing coordinates: incomplete binder Cα trace")
    sequence = "".join(AA3.get(value, "X") for value in bca.label_comp_id)
    fw, cdr = regions(sequence, plan["framework"], cdr_ids)
    bc = positions(bca)
    if not np.isfinite(bc).all():
        raise ValueError("missing coordinates: nonfinite binder Cα trace")
    tca = antigen[antigen.label_atom_id == "CA"]
    for chain, pos, aa in zip(tca.label_asym_id, tca.pos, tca.label_comp_id):
        if (
            pos < 1
            or pos > len(plan["sequences"][chain])
            or AA3.get(aa, "X") != plan["sequences"][chain][pos - 1]
        ):
            raise ValueError(f"unexpected target residue {chain}:{pos}")
    tlookup = dict(zip(zip(tca.label_asym_id, tca.pos), positions(tca)))
    keys = [(str(chain), int(pos)) for chain, pos in plan["keys"]]
    selected = {keys[i] for i in plan["selected"]}
    at, bt = positions(antigen), positions(binder)
    akeys = [
        (str(chain), int(pos)) for chain, pos in zip(antigen.label_asym_id, antigen.pos)
    ]
    cdr_set = {int(index) + 1 for index in cdr}
    fp = measure_footprint(
        at,
        akeys,
        bt,
        [int(p) for p in binder.pos],
        bc[fw],
        bc[cdr],
        sorted(cdr_set),
        antigen_ca=tlookup,
    )
    metrics = {
        key: value for key, value in fp.items() if key not in {"status", "detail"}
    }
    metrics.update(
        footprint_status=fp["status"],
        footprint_diagnostics=fp.get("detail", ""),
        geometry_status="ok",
        framework_ca_count=len(fw),
        cdr_ca_count=len(cdr),
        represented_antigen_heavy_atoms=len(at),
        represented_vhh_heavy_atoms=len(bt),
        missing_antigen_chains=sorted(
            set(plan["sequences"]) - set(antigen.label_asym_id)
        ),
        patch_size=len(selected),
    )
    usable = bool(
        len(at) and len(bt) and np.isfinite(at).all() and np.isfinite(bt).all()
    )
    pairs = set()
    if usable:
        tree = cKDTree(at)
        within, nearest = tree.query_ball_point(bt, 5.0), tree.query(bt)[0]
        pairs = {
            (int(pos), akeys[i]) for pos, hits in zip(binder.pos, within) for i in hits
        }
        metrics.update(
            min_heavy_atom_A=float(np.min(nearest)),
            binder_atoms_with_clash_2A=int(np.sum(nearest < 2)),
        )
    contacted = {key for _, key in pairs}
    bcontacts = {pos for pos, _ in pairs}
    metrics.update(
        footprint_residue_keys=[list(key) for key in sorted(contacted)],
        requested_site_contact=bool(selected & contacted) if usable else None,
        patch_residues_contacted=len(selected & contacted) if usable else None,
        requested_site_residue_keys=[list(key) for key in sorted(selected & contacted)],
        heavy_atom_residue_pairs=len(pairs) if usable else None,
        cdr_residue_pairs=sum(pos in cdr_set for pos, _ in pairs) if usable else None,
        cdr_patch_contact=(
            any(pos in cdr_set and key in selected for pos, key in pairs)
            if usable
            else None
        ),
        contacting_cdr_residues=len(bcontacts & cdr_set) if usable else None,
        contacting_framework_residues=len(bcontacts - cdr_set) if usable else None,
        cdr_contact_fraction=(
            len(bcontacts & cdr_set) / len(bcontacts) if bcontacts else None
        ),
        pose_class="no_contacts" if usable and not contacted else None,
    )
    lookup = dict(zip(keys, plan.get("labels", [{}] * len(keys))))
    metrics["footprint_author_residues"] = [
        dict(chain=c, runtime_position=p, **lookup.get((c, p), {}))
        for c, p in sorted(contacted)
    ]
    rows = []
    for i in plan["selected"]:
        chain, pos = keys[i]
        hits = {b for b, key in pairs if key == (chain, pos)}
        rows.append(
            dict(
                chain=chain,
                runtime_position=pos,
                **lookup.get((chain, pos), {}),
                coordinate_present=(chain, pos) in set(akeys),
                cdr_residues=len(hits & cdr_set) if usable else None,
                framework_residues=len(hits - cdr_set) if usable else None,
            )
        )
    patch_ca = [tlookup[key] for key in selected if key in tlookup]
    metrics["patch_ca_pairs_8A"] = (
        sum(map(len, cKDTree(patch_ca).query_ball_point(bc, 8)))
        if len(patch_ca) == len(selected)
        else None
    )
    # Existing badges retain their old centroid-pointing and contact definition.
    pointing = finite_number(metrics.get("centroid_pointing_deg"))
    if pointing is not None and metrics["cdr_contact_fraction"] is not None:
        share = metrics["cdr_contact_fraction"]
        metrics.update(
            interface_axis_deg=pointing, legacy_centroid_pointing_deg=pointing
        )
        if pointing <= 20 and share > 0.60:
            metrics["pose_class"] = "cdr_driven"
        elif pointing >= 30 or (pointing > 20 and share < 0.50):
            metrics["pose_class"] = "framework_driven"
        else:
            metrics["pose_class"] = "mixed"
    near = plan["frame"].get("neighbourhood", list(range(len(keys))))
    ids = [
        i for i in near if keys[i] in tlookup and np.isfinite(tlookup[keys[i]]).all()
    ]
    reference = np.asarray(plan["coords"], float)
    shape = None
    metrics["patch_geometry_status"] = (
        "missing coordinates: insufficient local antigen Cα support"
    )
    if len(ids) >= 3:
        seen, expected = np.asarray([tlookup[keys[i]] for i in ids]), reference[ids]
        if (
            np.linalg.matrix_rank(seen - seen.mean(0), tol=1e-7) >= 2
            and np.linalg.matrix_rank(expected - expected.mean(0), tol=1e-7) >= 2
        ):
            rotation, translation = align(seen, expected)
            normal = np.asarray(plan["frame"]["normal"], float)
            metrics.update(
                local_target_rmsd_A=rmsd(seen @ rotation + translation, expected),
                patch_fit_ca_count=len(ids),
                patch_fit_expected_ca_count=len(near),
                patch_geometry_status=(
                    "ok" if len(ids) == len(near) else "partial local antigen Cα fit"
                ),
                patch_elevation_deg=elevation(
                    (bc[fw].mean(0) - bc[cdr].mean(0)) @ rotation, normal
                ),
                requested_patch_normal_observed=(normal @ rotation.T).tolist(),
            )
            if all(fp.get("outward_n" + axis) is not None for axis in "xyz"):
                actual = np.array([fp["outward_n" + axis] for axis in "xyz"])
                metrics["patch_footprint_normal_difference_deg"] = degrees(
                    normal, actual @ rotation
                )
            shape = (
                bc[fw] @ rotation + translation,
                (bc[cdr].mean(0) - bc[fw].mean(0)) @ rotation,
                bc[cdr] @ rotation + translation,
            )
    if all(key in tlookup for key in keys):
        try:
            historical = orientation(
                patch_frame(np.array([tlookup[key] for key in keys]), plan["selected"]),
                bc[fw],
                bc[cdr],
            )
        except (GeometryError, ValueError) as exc:
            metrics["legacy_geometry_status"] = str(exc)
        else:
            metrics.update(
                historical,
                legacy_geometry_status="ok",
                legacy_placement_spatial_deg=historical["spatial_deg"],
                legacy_cdr_facing_patch_deg=historical["facing_deg"],
                legacy_pointing_to_requested_patch_deg=historical[
                    "pointing_to_patch_deg"
                ],
            )
    if metrics["missing_antigen_chains"]:
        metrics["partial_footprint_status"] = metrics["footprint_status"]
        metrics["partial_footprint_diagnostics"] = metrics["footprint_diagnostics"]
        # The audit requires every declared antigen chain. Keep the useful
        # observed-subset diagnostics explicitly partial, never as a complete
        # footprint or a negative requested-site-contact observation.
        partial_keys = {
            *fp.keys(),
            "requested_site_contact",
            "patch_residues_contacted",
            "requested_site_residue_keys",
            "footprint_author_residues",
            "heavy_atom_residue_pairs",
            "cdr_residue_pairs",
            "cdr_patch_contact",
            "contacting_cdr_residues",
            "contacting_framework_residues",
            "cdr_contact_fraction",
            "pose_class",
            "interface_axis_deg",
            "legacy_centroid_pointing_deg",
            "patch_footprint_normal_difference_deg",
        } - {"status", "detail"}
        for key in partial_keys:
            if key in metrics:
                metrics["partial_" + key] = metrics[key]
                metrics[key] = None
        for key in (
            "footprint_elevation_deg",
            "inclination_deg",
            "sensitivity_min_deg",
            "sensitivity_max_deg",
            "pose_label",
        ):
            metrics[key] = None
        metrics["footprint_status"] = "missing_coordinates"
        metrics["footprint_diagnostics"] = (
            "missing_selected_antigen_chains:"
            + ",".join(metrics["missing_antigen_chains"])
        )
        for row in rows:
            row["antigen_coordinates_complete"] = False
    return metrics, rows, shape


def scdockq_from_confidence(confidence: Any, plan: Row) -> Row:
    """Use the binder–antigen union score, never the maximum arbitrary pair."""
    missing = dict(scDockQ=None, scDockQ_status="missing score")
    if not isinstance(confidence, dict):
        return dict(missing, scDockQ_status="invalid confidence object")
    raw = confidence.get("scDockQ")
    number = finite_number(raw)
    if number is not None:
        return (
            dict(
                scDockQ=number,
                scDockQ_status="ok",
                scDockQ_source="scalar",
                scDockQ_raw=raw,
            )
            if 0 <= number <= 1
            else dict(missing, scDockQ_status="invalid score")
        )
    if not isinstance(raw, dict):
        return missing
    selected, source = None, ""
    if "binder_antigen" in raw:
        selected, source = raw["binder_antigen"], "binder_antigen_union"
    elif len(plan["sequences"]) == 1:
        binder, target = str(plan["binder_chain"]), str(next(iter(plan["sequences"])))
        candidates = [
            key for key in (f"{binder}_{target}", f"{target}_{binder}") if key in raw
        ]
        if len(candidates) == 1:
            source, selected = candidates[0], raw[candidates[0]]
        elif set(raw) <= {"0_1", "1_0"} and len(raw) == 1:
            source = next(iter(raw))
            selected = raw[source]
        elif any(key in raw for key in ("DockQ", "dockq")):
            selected, source = raw, "single_interface"
    if selected is None:
        return dict(
            missing,
            scDockQ_status="missing unambiguous binder–antigen union score",
            scDockQ_raw=raw,
        )
    value = (
        finite_number(selected.get("DockQ", selected.get("dockq")))
        if isinstance(selected, dict)
        else finite_number(selected)
    )
    if value is None or not 0 <= value <= 1:
        return dict(missing, scDockQ_status="invalid score", scDockQ_raw=raw)
    result = dict(
        scDockQ=value, scDockQ_status="ok", scDockQ_source=source, scDockQ_raw=raw
    )
    if isinstance(selected, dict):
        for key in ("fnat", "iRMSD", "LRMSD"):
            result["scDockQ_" + key] = finite_number(selected.get(key))
    return result


def confidence_metrics(path: Path, plan: Row) -> Row:
    """Read the confidence record belonging to this exact refold."""
    source = path.with_name(path.stem + "_confidence.json")
    try:
        confidence = json.loads(source.read_text())
    except (ValueError, OSError) as exc:
        return dict(
            scDockQ=None,
            scDockQ_status="missing or invalid confidence",
            confidence_detail=str(exc),
        )
    result = scdockq_from_confidence(confidence, plan)
    result["confidence_path"] = str(source)
    if isinstance(confidence, dict):
        for key in ("complex_iptm", "binder_scRMSD", "complex_plddt"):
            value = finite_number(confidence.get(key))
            if value is not None:
                result[key] = value
    return result


def summarise(rows: list[Row], interval: Any, *, elevation_interval: Any = None) -> Row:
    groups = {}
    paired = {}
    for row in rows:
        groups.setdefault(row["arm"], []).append(row)
        paired.setdefault((row["replicate"], row["backbone"]), {})[row["arm"]] = row
    report = {}
    for arm, items in groups.items():
        stats = dict(
            expected=len(items),
            generated=sum(r["file_exists"] for r in items),
            rejected_guides=sum(
                r.get("geometry_status") == "rejected_guide" for r in items
            ),
        )
        for metric in (
            "spatial_deg",
            "facing_deg",
            "pointing_to_patch_deg",
            "cdr_contact_fraction",
        ):
            vals = [r[metric] for r in items if r.get(metric) is not None]
            dist = dict(measured=len(vals))
            if vals:
                dist.update(
                    zip(
                        ["min", "q25", "median", "q75", "max"],
                        np.percentile(vals, [0, 25, 50, 75, 100]).tolist(),
                    )
                )
                dist["mean"] = float(np.mean(vals))
                if metric.endswith("_deg") and interval:
                    hits = sum(
                        interval[0] - 1e-7 <= x <= interval[1] + 1e-7 for x in vals
                    )
                    dist.update(
                        in_interval=hits,
                        occupancy=hits / len(vals),
                        occupancy_bounds_all_expected=[
                            hits / len(items),
                            (hits + len(items) - len(vals)) / len(items),
                        ],
                    )
            stats[metric] = dist
        for metric in (
            "patch_elevation_deg",
            "footprint_elevation_deg",
            "scaffold_signed_error_deg",
            "scaffold_absolute_error_deg",
            "patch_signed_error_deg",
            "patch_absolute_error_deg",
            "footprint_signed_error_deg",
            "footprint_absolute_error_deg",
            "patch_footprint_normal_difference_deg",
        ):
            values = [
                number
                for row in items
                if (number := finite_number(row.get(metric))) is not None
            ]
            dist = dict(measured=len(values), missing=len(items) - len(values))
            if values:
                dist.update(
                    zip(
                        ("min", "q25", "median", "q75", "max"),
                        np.percentile(values, [0, 25, 50, 75, 100]).tolist(),
                    )
                )
                dist["mean"] = float(np.mean(values))
                if (
                    metric in ("patch_elevation_deg", "footprint_elevation_deg")
                    and elevation_interval is not None
                ):
                    hits = sum(
                        bool(in_interval(value, elevation_interval)) for value in values
                    )
                    dist.update(
                        in_interval=hits,
                        occupancy=hits / len(values),
                        occupancy_bounds_all_expected=[
                            hits / len(items),
                            (hits + len(items) - len(values)) / len(items),
                        ],
                    )
            stats[metric] = dist
        stats["footprint_statuses"] = dict(
            Counter(row.get("footprint_status", "missing structure") for row in items)
        )
        report[arm] = stats
    effects = {}
    if len(groups) == 2:
        for metric in ("spatial_deg", "facing_deg", "cdr_contact_fraction"):
            by_rep = {}
            for (rep, idx), arms in paired.items():
                a, b = (arms.get(k, {}) for k in ("ordinary", "guided"))
                if a.get(metric) is None or b.get(metric) is None:
                    continue
                av, bv = a[metric], b[metric]
                vals = [bv - av]
                if metric.endswith("_deg") and interval:

                    def inside(x):
                        return int(interval[0] - 1e-7 <= x <= interval[1] + 1e-7)

                    vals.append(inside(bv) - inside(av))
                by_rep.setdefault(rep, []).append(vals)
            if not by_rep:
                effects[metric] = {"matched_pairs": 0}
                continue
            blocks = [np.asarray(v) for v in by_rep.values()]
            totals = np.stack([x.sum(0) for x in blocks])
            sizes = np.array([len(x) for x in blocks])
            est = totals.sum(0) / sizes.sum()
            entry = dict(
                matched_pairs=int(sizes.sum()),
                matched_replicates=len(blocks),
                mean_difference=float(est[0]),
                bootstrap95=None,
            )
            if len(est) > 1:
                entry["occupancy_difference"] = float(est[1])
            if len(blocks) >= 2:
                draw = np.random.default_rng(0).integers(
                    len(blocks), size=(4000, len(blocks))
                )
                estimates = totals[draw].sum(1) / sizes[draw].sum(1)[:, None]
                bounds = np.percentile(estimates, [2.5, 97.5], axis=0).T.tolist()
                entry["bootstrap95"] = bounds[0]
                if len(bounds) > 1:
                    entry["occupancy_bootstrap95"] = bounds[1]
            effects[metric] = entry
    return dict(
        arms=report,
        guided_minus_ordinary=effects,
        uncertainty_unit="paired replicate; percentile bootstrap",
    )


def figures(
    folder: Path, rows: list[Row], interval: Any, *, elevation_interval: Any = None
) -> None:
    """Compare requested, placed and independently observed signed elevations."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 4), layout="constrained")
    axes = fig.subplots(1, 3)
    for key, label, marker, color in (
        ("realized_scaffold_elevation_deg", "Placed scaffold", "x", "#405c6d"),
        ("patch_elevation_deg", "Generated / requested patch", "o", "#b46c50"),
        ("footprint_elevation_deg", "Generated / actual footprint", "^", "#69866b"),
    ):
        pts = [
            (r["requested_elevation_deg"], r[key])
            for r in rows
            if finite_number(r.get("requested_elevation_deg")) is not None
            and finite_number(r.get(key)) is not None
        ]
        if pts:
            axes[0].scatter(
                *zip(*pts),
                label=f"{label} (n={len(pts)})",
                marker=marker,
                color=color,
                s=16,
                alpha=0.65,
            )
    axes[0].plot([-90, 90], [-90, 90], "--", color=".65", lw=0.8)
    axes[0].set(
        xlabel="Requested elevation (°)",
        ylabel="Measured elevation (°)",
        xlim=(-90, 90),
        ylim=(-90, 90),
        title="Did generation obey the request?",
    )
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7, frameon=False)
    for arm in sorted({r["arm"] for r in rows}):
        values = [
            v
            for r in rows
            if r["arm"] == arm
            and (v := finite_number(r.get("footprint_elevation_deg"))) is not None
        ]
        if values:
            axes[1].hist(
                values,
                bins=np.arange(-90, 91, 5),
                histtype="step",
                label=f"{arm} (n={len(values)})",
            )
        pts = [
            (
                r["patch_footprint_normal_difference_deg"],
                r["footprint_elevation_deg"] - r["patch_elevation_deg"],
            )
            for r in rows
            if r["arm"] == arm
            and all(
                finite_number(r.get(k)) is not None
                for k in (
                    "patch_footprint_normal_difference_deg",
                    "footprint_elevation_deg",
                    "patch_elevation_deg",
                )
            )
        ]
        if pts:
            axes[2].scatter(*zip(*pts), s=13, alpha=0.6, label=arm)
    if elevation_interval is not None:
        axes[1].axvspan(*elevation_interval, color="#b6ceb8", alpha=0.35)
    axes[1].set(
        xlabel="Actual-footprint elevation (°)",
        ylabel="Backbones",
        xlim=(-90, 90),
        title="Signed body inclination",
    )
    axes[2].axhline(0, color=".65", lw=0.8)
    axes[2].set(
        xlabel="Patch–footprint normal difference (°)",
        ylabel="Footprint − patch elevation (°)",
        xlim=(0, 180),
        title="Did the contacted surface differ?",
    )
    for axis in axes[1:]:
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7, frameon=False)
    fig.savefig(folder / "orientations.png", dpi=160)


def _angle_metrics(row: Row, interval: Any) -> None:
    requested = finite_number(row.get("requested_elevation_deg"))
    for name, key in (
        ("scaffold", "realized_scaffold_elevation_deg"),
        ("patch", "patch_elevation_deg"),
        ("footprint", "footprint_elevation_deg"),
    ):
        observed = finite_number(row.get(key))
        row[name + "_in_band"] = in_interval(observed, interval)
        if observed is not None and requested is not None:
            row[name + "_signed_error_deg"] = observed - requested
            row[name + "_absolute_error_deg"] = abs(observed - requested)


def _pair_refold(row: Row, parent: Row, shape: Any, before_shape: Any) -> None:
    row.update(parent_path=parent["path"], parent_file_exists=parent["file_exists"])
    for prefix, key in (
        ("elevation", "footprint_elevation_deg"),
        ("patch_elevation", "patch_elevation_deg"),
    ):
        before, after = finite_number(parent.get(key)), finite_number(row.get(key))
        row["parent_" + key] = before
        if before is not None and after is not None:
            row[prefix + "_change_deg"] = after - before
            row["absolute_" + prefix + "_change_deg"] = abs(after - before)
    if (
        parent.get("requested_site_contact") is not None
        and row.get("requested_site_contact") is not None
    ):
        original = {tuple(k) for k in parent.get("footprint_residue_keys", [])}
        current = {tuple(k) for k in row.get("footprint_residue_keys", [])}
        union, overlap = original | current, original & current
        row.update(
            footprint_intersection_residues=len(overlap),
            footprint_union_residues=len(union),
            footprint_jaccard=len(overlap) / len(union) if union else None,
            parent_footprint_recovered_fraction=(
                len(overlap) / len(original) if original else None
            ),
            refold_footprint_shared_with_parent_fraction=(
                len(overlap) / len(current) if current else None
            ),
        )
    if shape is None or before_shape is None:
        row["pose_recovery_geometry_status"] = "missing parent or local target fit"
        return
    try:
        row.update(
            framework_rmsd_after_target_fit_A=rmsd(before_shape[0], shape[0]),
            cdr_axis_change_deg=degrees(before_shape[1], shape[1]),
        )
        fit, shift = align(shape[0], before_shape[0])
        row["self_fit_framework_rmsd_A"] = rmsd(shape[0] @ fit + shift, before_shape[0])
        row["cdr_rmsd_after_framework_fit_A"] = rmsd(
            shape[2] @ fit + shift, before_shape[2]
        )
        row["pose_recovery_geometry_status"] = "ok"
    except (ValueError, GeometryError) as exc:
        row["pose_recovery_geometry_status"] = str(exc)
    for key in ("spatial_deg", "facing_deg"):
        if (
            finite_number(row.get(key)) is not None
            and finite_number(parent.get(key)) is not None
        ):
            row["delta_" + key] = row[key] - parent[key]


def analyse(
    folder: Path,
    jobs: list[Row],
    plan: Row,
    interval: Any,
    *,
    elevation_interval: Any = None,
    legacy_interval: Any = None,
) -> Row:
    """Write structure, parent and replicate tables plus a standalone report."""
    folder.mkdir(parents=True, exist_ok=True)
    if (
        elevation_interval is None
        and plan.get("experiment_definition") == "surface_elevation_v1"
    ):
        elevation_interval = plan.get("elevation_interval_deg")
    if legacy_interval is None:
        legacy_interval = plan.get("legacy_tilt_interval_deg", interval)
    budget = plan.get("refolds_per_parent")
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("refolds_per_parent must be a nonnegative integer")
    backbones, refolds, contacts = [], [], []
    for job in jobs:
        for index in range(job["start"], job["start"] + job["count"]):
            sample = Path(job["output"]) / plan["target"] / f"sample{index}"
            base = dict(
                arm=job["arm"],
                replicate=job["replicate"],
                backbone=index,
                replicate_seed=job.get("seed"),
            )
            try:
                record = json.loads((sample / "angle.json").read_text())
                if not isinstance(record, dict):
                    raise ValueError("expected an angle record object")
            except (ValueError, OSError) as exc:
                record = {"record_status": str(exc)}
            definition = record.get("guidance_definition")
            requested = (
                record.get("requested_elevation_deg")
                if definition == "surface_elevation"
                else None
            )
            legacy = record.get("requested_legacy_tilt_deg")
            if legacy is None and definition != "surface_elevation":
                legacy = record.get("requested_deg")
            parent, parent_shape = None, None
            paths = [
                sample / "backbone.cif",
                *sorted((sample / "refolds").glob("*_refold*.cif")),
            ]
            for path in paths:
                is_parent = path.name == "backbone.cif"
                row = dict(
                    base,
                    stage="backbone" if is_parent else "refold",
                    path=(
                        str(path.relative_to(folder.parent))
                        if path.is_relative_to(folder.parent)
                        else str(path)
                    ),
                    refold_id=None if is_parent else path.stem,
                    file_exists=path.is_file(),
                    guide_record_status=record.get("record_status", "ok"),
                    guidance_definition=definition
                    or {
                        "ordinary": "ordinary",
                        "surface_elevation": "surface_elevation",
                    }.get(job["arm"], "legacy_placement_ray"),
                    requested_elevation_deg=finite_number(requested),
                    realized_scaffold_elevation_deg=finite_number(
                        record.get("realized_scaffold_elevation_deg")
                    ),
                    requested_legacy_tilt_deg=finite_number(legacy),
                    requested_deg=finite_number(legacy),
                    roll_attempts=record.get("roll_attempts"),
                    ray_obstructed_ca_proxy=record.get("ray_obstructed_ca_proxy"),
                    standoff_A=record.get("standoff_A", record.get("standoff")),
                    azimuth_deg=record.get("azimuth_deg"),
                    roll_deg=record.get("roll_deg"),
                    expected_refolds=budget if is_parent else None,
                )
                (backbones if is_parent else refolds).append(row)
                if is_parent:
                    parent = row
                else:
                    row.update(confidence_metrics(path, plan))
                    row.update(
                        parent_path=parent["path"],
                        parent_file_exists=parent["file_exists"],
                    )
                if not row["file_exists"]:
                    if is_parent and record.get("record_status") == "rejected_guide":
                        row.update(
                            geometry_status="rejected_guide",
                            reason=record.get("reason"),
                        )
                    else:
                        row["geometry_status"] = "missing structure"
                    row["footprint_status"] = "missing_coordinates"
                    _angle_metrics(row, elevation_interval)
                    continue
                try:
                    metrics, per_site, shape = measurements(
                        path, plan, record.get("cdr_positions")
                    )
                    row.update(metrics)
                    contacts.extend(
                        dict(base, structure=path.name, **site) for site in per_site
                    )
                    if is_parent:
                        parent_shape = shape
                    else:
                        _pair_refold(row, parent, shape, parent_shape)
                        for key in ("spatial_deg", "facing_deg"):
                            if (
                                legacy_interval is not None
                                and finite_number(parent.get(key)) is not None
                                and finite_number(row.get(key)) is not None
                            ):
                                row["lost_interval_" + key] = bool(
                                    in_interval(parent[key], legacy_interval)
                                ) and not bool(in_interval(row[key], legacy_interval))
                except (ValueError, KeyError, IndexError, OSError) as exc:
                    row["geometry_status"] = f"unmeasurable: {exc}"
                    row.setdefault(
                        "footprint_status",
                        (
                            "missing_coordinates"
                            if "coordinates" in str(exc)
                            else "unmeasurable"
                        ),
                    )
                _angle_metrics(row, elevation_interval)
            count = len(paths) - 1
            parent.update(
                available_refolds=count,
                missing_refolds=max(0, budget - count) if budget is not None else None,
                excess_refolds=max(0, count - budget) if budget is not None else None,
            )
    # Populate each exact refold's joint criterion before serializing rows.
    elevation_tables(backbones, refolds, elevation_interval)
    fields = [
        "arm",
        "replicate",
        "backbone",
        "stage",
        "geometry_status",
        "footprint_status",
        "requested_elevation_deg",
        "realized_scaffold_elevation_deg",
        "patch_elevation_deg",
        "footprint_elevation_deg",
        "footprint_in_band",
        "path",
    ]
    write_csv(folder / "backbones.csv", backbones, fields)
    write_csv(
        folder / "refolds.csv",
        refolds,
        [
            *fields,
            "parent_path",
            "elevation_change_deg",
            "absolute_elevation_change_deg",
            "footprint_jaccard",
            "scDockQ",
            "joint_useful_refold",
        ],
    )
    write_csv(
        folder / "patch_contacts.csv",
        contacts,
        ["arm", "replicate", "backbone", "structure", "chain", "runtime_position"],
    )
    summary = summarise(
        backbones, legacy_interval, elevation_interval=elevation_interval
    )
    summary.update(
        experiment_definition=plan.get("experiment_definition", "legacy_placement_ray"),
        primary_angle_definition="signed_surface_elevation_C_to_F",
        elevation_interval_deg=(
            list(elevation_interval) if elevation_interval is not None else None
        ),
        legacy_tilt_interval_deg=(
            list(legacy_interval) if legacy_interval is not None else None
        ),
        interval_deg=list(legacy_interval) if legacy_interval is not None else None,
        expected_refolds_per_parent=budget,
        available_refolds=len(refolds),
        backbones_with_refolds=len(
            {(r["arm"], r["replicate"], r["backbone"]) for r in refolds}
        ),
        measurement_units={"angles": "degrees", "lengths": "angstrom"},
        contact_definition="selected antigen-chain heavy atoms within 5 Å of any represented VHH heavy atom; one sampling point per chain-qualified contacted antigen residue",
        patch_normal_definition="saved prospective normal transported by local antigen Cα alignment",
        footprint_definition="revised audit Gaussian density normal at contacted-atom residue centroids, scales 4 and 6 Å",
        compatibility_aliases={
            "requested_deg": "requested_legacy_tilt_deg",
            "spatial_deg": "legacy_placement_spatial_deg",
            "facing_deg": "legacy_cdr_facing_patch_deg",
            "interface_axis_deg": "legacy_centroid_pointing_deg",
        },
        interpretation="predicted geometry; binding requires independent experimental evidence",
    )
    figures(folder, backbones, legacy_interval, elevation_interval=elevation_interval)
    summary["pose_endpoints"] = experiment_report(
        folder,
        backbones,
        refolds,
        legacy_interval,
        elevation_interval=elevation_interval,
        plan=plan,
    )
    write_json(folder / "summary.json", summary)
    return summary
