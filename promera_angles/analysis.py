"""Measure every expected backbone and every available refold, including failures."""

import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.pose_guidance.pose_geometry import AA3, align, framework_map, rmsd

from .geometry import GeometryError, degrees, orientation, patch_frame
from .structures import atom_table


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, required):
    keys = list(dict.fromkeys([*required, *(k for row in rows for k in row)]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def positions(table):
    return table[["Cartn_x", "Cartn_y", "Cartn_z"]].to_numpy(float)


def measurements(path, plan, cdr_ids):
    atoms = atom_table(path)
    antigen = atoms[atoms.label_asym_id.isin(plan["sequences"])]
    binder = atoms[atoms.label_asym_id == plan["binder_chain"]]
    bca = binder[binder.label_atom_id == "CA"].sort_values("pos")
    if list(bca.pos) != list(range(1, len(bca) + 1)):
        raise ValueError("incomplete binder Cα trace")
    sequence = "".join(AA3.get(v, "X") for v in bca.label_comp_id)
    if cdr_ids is None:
        fw, cdr = framework_map(sequence, plan["framework"])
    else:
        cdr = [int(i) - 1 for i in cdr_ids]
        if (
            len(cdr) != len(set(cdr))
            or not cdr
            or not all(0 <= i < len(sequence) for i in cdr)
        ):
            raise ValueError("invalid saved CDR positions")
        cdr_mask = set(cdr)
        fw = [i for i in range(len(sequence)) if i not in cdr_mask]
        if "".join(sequence[i] for i in fw) != re.sub(
            r"<cdrh[123]>", "", plan["framework"]
        ):
            raise ValueError("saved CDR positions do not match the observed framework")
    bc = positions(bca)
    target_ca = antigen[antigen.label_atom_id == "CA"]
    tkeys = list(zip(target_ca.label_asym_id, target_ca.pos))
    for chain, pos, aa in zip(
        target_ca.label_asym_id, target_ca.pos, target_ca.label_comp_id
    ):
        if (
            pos < 1
            or pos > len(plan["sequences"][chain])
            or AA3.get(aa, "X") != plan["sequences"][chain][pos - 1]
        ):
            raise ValueError(f"unexpected target residue {chain}:{pos}")
    tlookup = dict(zip(tkeys, positions(target_ca)))
    keys = list(map(tuple, plan["keys"]))
    tc = np.array([tlookup[key] for key in keys])
    at, bt = positions(antigen), positions(binder)
    if (
        not len(at)
        or not len(bt)
        or not np.isfinite(at).all()
        or not np.isfinite(bt).all()
    ):
        raise ValueError("invalid interface coordinates")
    tree = cKDTree(at)
    within = tree.query_ball_point(bt, 5.0)
    nearest = tree.query(bt)[0]
    antigen_keys = list(zip(antigen.label_asym_id, antigen.pos))
    pairs = {
        (int(b), antigen_keys[i]) for b, hits in zip(binder.pos, within) for i in hits
    }
    cdr_set = {i + 1 for i in cdr}
    contacting = {b for b, k in pairs}
    cdr_count = len(contacting & cdr_set)
    selected = {keys[i] for i in plan["selected"]}
    contacted = {k for b, k in pairs}
    rows = []
    for i in plan["selected"]:
        chain, pos = keys[i]
        hits = {b for b, k in pairs if k == (chain, pos)}
        rows.append(
            dict(
                chain=chain,
                runtime_position=pos,
                **plan["labels"][i],
                cdr_residues=len(hits & cdr_set),
                framework_residues=len(hits - cdr_set),
            )
        )
    metrics = dict(
        contacting_cdr_residues=cdr_count,
        contacting_framework_residues=len(contacting) - cdr_count,
        cdr_contact_fraction=cdr_count / len(contacting) if contacting else None,
        heavy_atom_residue_pairs=len(pairs),
        cdr_residue_pairs=sum(b in cdr_set for b, k in pairs),
        patch_residues_contacted=len(selected & contacted),
        patch_size=len(selected),
        patch_ca_pairs_8A=sum(
            len(v) for v in cKDTree(tc[plan["selected"]]).query_ball_point(bc, 8.0)
        ),
        min_heavy_atom_A=float(min(nearest)),
        binder_atoms_with_clash_2A=int(sum(nearest < 2.0)),
    )
    # A local distortion may destroy the normal; preserve contact results.
    try:
        frame = patch_frame(tc, plan["selected"])
        metrics.update(orientation(frame, bc[fw], bc[cdr]), geometry_status="ok")
    except GeometryError as exc:
        metrics.update(geometry_status=str(exc))
    near = plan["frame"]["neighbourhood"]
    rotation, translation = align(tc[near], np.asarray(plan["coords"])[near])
    metrics["local_target_rmsd_A"] = rmsd(
        tc[near] @ rotation + translation, np.asarray(plan["coords"])[near]
    )
    return (
        metrics,
        rows,
        (
            bc[fw] @ rotation + translation,
            (bc[cdr].mean(0) - bc[fw].mean(0)) @ rotation,
        ),
    )


def summarise(rows, interval):
    groups = {}
    paired = {}
    for row in rows:
        groups.setdefault(row["arm"], []).append(row)
        paired.setdefault((row["replicate"], row["backbone"]), {})[row["arm"]] = row
    report = {}
    for arm, items in groups.items():
        stats = dict(
            expected=len(items), generated=sum(r["file_exists"] for r in items),
            rejected_guides=sum(r.get("geometry_status") == "rejected_guide" for r in items),
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
                if metric.endswith("_deg"):
                    inside = lambda x: int(
                        interval[0] - 1e-7 <= x <= interval[1] + 1e-7
                    )
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


def figures(folder, rows, interval):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, key, label in zip(
        axs[:2], ["spatial_deg", "facing_deg"], ["Spatial approach", "CDR-facing axis"]
    ):
        for arm, color in (("ordinary", "#447795"), ("guided", "#b5572e")):
            vals = [r[key] for r in rows if r["arm"] == arm and key in r]
            if vals:
                ax.hist(
                    vals,
                    bins=np.arange(0, 181, 5),
                    histtype="step",
                    label=f"{arm} (n={len(vals)})",
                    color=color,
                )
        if interval:
            ax.axvspan(*interval, color="#79aa8a", alpha=0.18)
        ax.set(xlabel=label + " (degrees)", ylabel="Backbones", xlim=(0, 180))
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
    for key, label in (("spatial_deg", "Spatial"), ("facing_deg", "CDR-facing")):
        points = [
            (r["requested_deg"], r[key])
            for r in rows
            if r.get("requested_deg") is not None and key in r
        ]
        if points:
            axs[2].scatter(*zip(*points), s=10, alpha=0.5, label=label)
    axs[2].plot([0, 180], [0, 180], "--", color=".65", lw=0.7)
    if axs[2].get_legend_handles_labels()[0]:
        axs[2].legend(frameon=False, fontsize=8)
    axs[2].set(
        xlabel="Requested tilt (degrees)",
        ylabel="Recovered angle (degrees)",
        ylim=(0, 180),
        xlim=(max(0, interval[0] - 2), min(180, interval[1] + 2))
        if interval
        else (0, 180),
    )
    fig.savefig(folder / "orientations.png", dpi=160)
    plt.close(fig)


def analyse(folder, jobs, plan, interval):
    folder.mkdir(parents=True, exist_ok=True)
    backbones, refolds, contacts = [], [], []
    for job in jobs:
        for i in range(job["start"], job["start"] + job["count"]):
            sample = Path(job["output"]) / plan["target"] / f"sample{i}"
            base = dict(arm=job["arm"], replicate=job["replicate"], backbone=i)
            record = {}
            try:
                record = json.loads((sample / "angle.json").read_text())
                if not isinstance(record, dict):
                    raise ValueError("expected an angle record object")
            except (ValueError, OSError) as exc:
                record = {"record_status": str(exc)}
            gen_shape, generated = None, {}
            for path in [
                sample / "backbone.cif",
                *sorted((sample / "refolds").glob("*_refold*.cif")),
            ]:
                is_gen = path.name == "backbone.cif"
                row = dict(
                    base,
                    path=str(path.relative_to(folder.parent))
                    if path.is_relative_to(folder.parent)
                    else str(path),
                    guide_record_status=record.get("record_status", "ok"),
                    file_exists=path.is_file(),
                    requested_deg=record.get("requested_deg"),
                    ray_obstructed_ca_proxy=record.get("ray_obstructed_ca_proxy"),
                )
                (backbones if is_gen else refolds).append(row)
                if not row["file_exists"]:
                    if is_gen and record.get("record_status") == "rejected_guide":
                        row.update(geometry_status="rejected_guide", reason=record.get("reason"))
                    else:
                        row["geometry_status"] = "missing structure"
                    continue
                try:
                    metrics, per_site, shape = measurements(
                        path, plan, record.get("cdr_positions")
                    )
                    row.update(metrics)
                    contacts.extend(
                        dict(base, structure=path.name, **s) for s in per_site
                    )
                    if is_gen:
                        gen_shape, generated = shape, metrics
                    elif gen_shape is not None:
                        row.update(
                            framework_rmsd_after_target_fit_A=rmsd(
                                gen_shape[0], shape[0]
                            ),
                            cdr_axis_change_deg=degrees(gen_shape[1], shape[1]),
                        )
                        for key in ("spatial_deg", "facing_deg"):
                            if key in metrics and key in generated:
                                row["delta_" + key] = metrics[key] - generated[key]
                                if interval:
                                    inside = lambda x: (
                                        interval[0] - 1e-7 <= x <= interval[1] + 1e-7
                                    )
                                    row["lost_interval_" + key] = inside(
                                        generated[key]
                                    ) and not inside(metrics[key])
                except (ValueError, KeyError, IndexError, OSError) as exc:
                    row["geometry_status"] = f"unmeasurable: {exc}"
    fields = [
        "arm",
        "replicate",
        "backbone",
        "geometry_status",
        "requested_deg",
        "spatial_deg",
        "facing_deg",
        "path",
    ]
    write_csv(folder / "backbones.csv", backbones, fields)
    write_csv(folder / "refolds.csv", refolds, fields)
    write_csv(
        folder / "patch_contacts.csv",
        contacts,
        ["arm", "replicate", "backbone", "structure", "chain", "runtime_position"],
    )
    summary = summarise(backbones, interval)
    summary.update(
        interval_deg=interval,
        available_refolds=len(refolds),
        backbones_with_refolds=len(
            {(r["arm"], r["replicate"], r["backbone"]) for r in refolds}
        ),
        measurement_units={"angles": "degrees", "lengths": "angstrom"},
        contact_definition="unique represented binder residues within 5 Å of a represented antigen heavy atom",
        interpretation="predicted structural geometry; binding requires independent experimental evidence",
    )
    write_json(folder / "summary.json", summary)
    figures(folder, backbones, interval)
    return summary
