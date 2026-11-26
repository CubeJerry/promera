"""Parent-based elevation report; historical tables retain their old semantics."""

from __future__ import annotations

import base64
import io
from collections import Counter, defaultdict
from html import escape
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ENDPOINTS = {
    "cdr": "CDR-driven interface",
    "patch": "CDR-driven at selected patch",
    "pose": "CDR-driven at patch, within interval",
}


def qualifies(row, endpoint, interval):
    category = row.get("pose_class")
    if category is None:
        return None
    if category != "cdr_driven":
        return False
    if endpoint == "cdr":
        return True
    if not row.get("cdr_patch_contact"):
        return False
    if endpoint == "patch":
        return True
    angle = row.get("spatial_deg")
    if angle is None:
        return None
    return interval[0] - 1e-7 <= angle <= interval[1] + 1e-7


def tables(backbones, refolds, interval):
    def key(row):
        return tuple(row[k] for k in ("arm", "replicate", "backbone"))

    families = defaultdict(list)
    for row in refolds:
        families[key(row)].append(row)
    parents = []
    for b in backbones:
        for endpoint in ENDPOINTS:
            if endpoint == "pose" and not interval:
                continue
            qualified = qualifies(b, endpoint, interval)
            outcomes = [qualifies(r, endpoint, interval) for r in families[key(b)]]
            hit = any(outcomes)
            parents.append(
                dict(
                    arm=b["arm"],
                    replicate=b["replicate"],
                    backbone=b["backbone"],
                    endpoint=endpoint,
                    requested=1,
                    generated=int(b["file_exists"]),
                    measurable=int(qualified is not None),
                    qualified=int(bool(qualified)),
                    with_refolds=int(any(v is not None for v in outcomes)),
                    qualified_with_refolds=int(
                        bool(qualified) and any(v is not None for v in outcomes)
                    ),
                    recovered=int(bool(qualified) and hit),
                    final_hit=int(hit),
                )
            )
    parents = pd.DataFrame(parents)
    reps = (
        parents.drop(columns="backbone").groupby(["endpoint", "replicate", "arm"]).sum()
    )
    totals = reps.groupby(["endpoint", "arm"]).sum()
    effects = []
    for endpoint, block in reps.groupby(level="endpoint"):
        wide = block.droplevel("endpoint").unstack("arm")
        if not {"ordinary", "guided"} <= set(wide.columns.get_level_values("arm")):
            continue
        for metric, num, den in [
            ("Backbone yield", "qualified", "requested"),
            ("Conditional recovery", "recovered", "qualified_with_refolds"),
            ("Final yield", "final_hit", "requested"),
        ]:
            a = (
                wide[
                    [
                        (num, "guided"),
                        (den, "guided"),
                        (num, "ordinary"),
                        (den, "ordinary"),
                    ]
                ]
                .dropna()
                .to_numpy()
            )
            if len(a) == 0:
                continue

            def diff(x):
                with np.errstate(divide="ignore", invalid="ignore"):
                    return 100 * (x[..., 0] / x[..., 1] - x[..., 2] / x[..., 3])

            draws = np.random.default_rng(101).integers(len(a), size=(5000, len(a)))
            sims = diff(a[draws].sum(axis=1))
            valid = sims[np.isfinite(sims)]
            ci = (
                np.percentile(valid, [2.5, 97.5])
                if len(a) > 1 and len(valid)
                else [np.nan] * 2
            )
            effects.append(
                dict(
                    endpoint=endpoint,
                    metric=metric,
                    difference_pp=diff(a.sum(0)),
                    low95_pp=ci[0],
                    high95_pp=ci[1],
                    paired_replicates=len(a),
                )
            )
    return (
        parents,
        reps.reset_index(),
        totals.reset_index(),
        pd.DataFrame(
            effects,
            columns=[
                "endpoint",
                "metric",
                "difference_pp",
                "low95_pp",
                "high95_pp",
                "paired_replicates",
            ],
        ),
    )


Row = dict[str, Any]
ARM_LABELS = {
    "ordinary": "Ordinary targeted",
    "legacy_tilt": "Legacy placement-ray guide",
    "surface_elevation": "Surface-elevation guide",
    "guided": "Historical placement-ray guide",
}
COUNT_COLUMNS = [
    "requested",
    "generated",
    "rejected_guides",
    "measured_parents",
    "inband_parents",
    "with_refolds",
    "with_evaluable_refolds",
    "complete_refold_budgets",
    "known_refold_budget_parents",
    "unknown_refold_budget_parents",
    "expected_refolds",
    "available_refolds",
    "missing_refolds",
    "excess_refolds",
    "refolds_measured_elevation",
    "refolds_measured_site_contact",
    "refolds_measured_scdockq",
    "evaluable_refolds",
    "known_outcome_refolds",
    "successful_refolds",
    "useful_parents",
    "unresolved_parents",
    "conditional_recovered",
    "conditional_total",
    "conditional_unresolved",
    "conditional_with_evaluable_refold",
]
EFFECT_COLUMNS = [
    "comparison",
    "arm",
    "reference",
    "metric",
    "difference_pp",
    "low95_pp",
    "high95_pp",
    "paired_replicates",
    "paired_replicates_with_both_denominators",
    "valid_bootstrap_draws",
]


def _finite(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if np.isfinite(number) else None


def _inside(value: Any, interval: Any) -> bool | None:
    number = _finite(value)
    return (
        None
        if number is None or interval is None
        else bool(interval[0] - 1e-7 <= number <= interval[1] + 1e-7)
    )


def _refold_criteria(
    row: Row, interval: Any
) -> tuple[bool | None, bool | None, bool | None]:
    """Separate criterion availability from the logical outcome."""
    angle = _inside(row.get("footprint_elevation_deg"), interval)
    site = row.get("requested_site_contact")
    if "requested_site_contact" not in row:
        contacts = _finite(row.get("patch_residues_contacted"))
        site = None if contacts is None else contacts > 0
    score = _finite(row.get("scDockQ"))
    return (
        angle,
        None if site is None else bool(site),
        None if score is None else score >= 0.23,
    )


def useful_refold(row: Row, interval: Any) -> bool | None:
    """Apply three-valued AND to criteria from this same refold."""
    criteria = _refold_criteria(row, interval)
    if any(value is False for value in criteria):
        return False
    if all(value is True for value in criteria):
        return True
    return None


def _paired_effects(reps: pd.DataFrame) -> pd.DataFrame:
    """Resample matched replicate blocks, never individual sibling refolds."""
    effects = []
    order = {"ordinary": 0, "legacy_tilt": 1, "surface_elevation": 2, "guided": 3}
    arms = sorted(reps.arm.unique(), key=lambda name: (order.get(name, 4), name))
    for reference, arm in combinations(arms, 2):
        left = reps[reps.arm == arm].set_index("replicate")
        right = reps[reps.arm == reference].set_index("replicate")
        paired = sorted(set(left.index) & set(right.index))
        for metric, num, den in (
            ("Useful parents / requested", "useful_parents", "requested"),
            ("Conditional recovery", "conditional_recovered", "conditional_total"),
            ("In-band parents / requested", "inband_parents", "requested"),
        ):
            blocks = np.array(
                [
                    [
                        left.at[rep, num],
                        left.at[rep, den],
                        right.at[rep, num],
                        right.at[rep, den],
                    ]
                    for rep in paired
                ],
                dtype=float,
            )
            if not len(blocks):
                continue
            total = blocks.sum(0)
            estimate = (
                100 * (total[0] / total[1] - total[2] / total[3])
                if total[1] and total[3]
                else None
            )
            informative = int(np.sum((blocks[:, 1] > 0) & (blocks[:, 3] > 0)))
            low = high = None
            valid_count = 0
            if informative >= 2:
                draws = np.random.default_rng(101).integers(
                    len(blocks), size=(5000, len(blocks))
                )
                samples = blocks[draws].sum(1)
                valid = (samples[:, 1] > 0) & (samples[:, 3] > 0)
                values = 100 * (
                    samples[valid, 0] / samples[valid, 1]
                    - samples[valid, 2] / samples[valid, 3]
                )
                valid_count = len(values)
                if valid_count:
                    low, high = map(float, np.percentile(values, [2.5, 97.5]))
            effects.append(
                dict(
                    comparison=f"{arm} − {reference}",
                    arm=arm,
                    reference=reference,
                    metric=metric,
                    difference_pp=estimate,
                    low95_pp=low,
                    high95_pp=high,
                    paired_replicates=len(paired),
                    paired_replicates_with_both_denominators=informative,
                    valid_bootstrap_draws=valid_count,
                )
            )
    return pd.DataFrame(effects, columns=EFFECT_COLUMNS)


def elevation_tables(
    backbones: list[Row], refolds: list[Row], interval: Any
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reduce final outcomes to one opportunity per requested parent."""
    families = defaultdict(list)
    fields = ("arm", "replicate", "backbone")
    for row in refolds:
        families[tuple(row[key] for key in fields)].append(row)
    parents, seen = [], set()
    for parent in backbones:
        key = tuple(parent[field] for field in fields)
        if key in seen:
            raise ValueError(f"duplicate requested parent {key}")
        seen.add(key)
        siblings = families[key]
        outcomes = [useful_refold(row, interval) for row in siblings]
        hit = any(value is True for value in outcomes)
        evaluable = sum(
            all(value is not None for value in _refold_criteria(row, interval))
            for row in siblings
        )
        budget = parent.get("expected_refolds")
        known = type(budget) is int and budget >= 0
        expected = int(budget) if known else 0
        missing = max(0, expected - len(siblings)) if known else 0
        excess = max(0, len(siblings) - expected) if known else 0
        inband = _inside(parent.get("footprint_elevation_deg"), interval)
        unknown = not hit and (
            not known or missing > 0 or any(value is None for value in outcomes)
        )
        status = "success" if hit else "incomplete" if unknown else "failure"
        if interval is None:
            status = "no_elevation_interval"
        parents.append(
            dict(
                arm=key[0],
                replicate=key[1],
                backbone=key[2],
                replicate_seed=parent.get("replicate_seed"),
                parent_path=parent.get("path"),
                parent_elevation_deg=parent.get("footprint_elevation_deg"),
                parent_in_band=inband,
                endpoint_status=status,
                requested_refolds=budget if known else None,
                requested=1,
                generated=int(bool(parent.get("file_exists"))),
                rejected_guides=int(parent.get("geometry_status") == "rejected_guide"),
                measured_parents=int(
                    _finite(parent.get("footprint_elevation_deg")) is not None
                ),
                inband_parents=int(inband is True),
                with_refolds=int(bool(siblings)),
                with_evaluable_refolds=int(evaluable > 0),
                complete_refold_budgets=int(known and len(siblings) == expected),
                known_refold_budget_parents=int(known),
                unknown_refold_budget_parents=int(not known),
                expected_refolds=expected,
                available_refolds=len(siblings),
                missing_refolds=missing,
                excess_refolds=excess,
                refolds_measured_elevation=sum(
                    _finite(row.get("footprint_elevation_deg")) is not None
                    for row in siblings
                ),
                refolds_measured_site_contact=sum(
                    row.get("requested_site_contact") is not None for row in siblings
                ),
                refolds_measured_scdockq=sum(
                    _finite(row.get("scDockQ")) is not None for row in siblings
                ),
                evaluable_refolds=evaluable,
                known_outcome_refolds=sum(value is not None for value in outcomes),
                successful_refolds=sum(value is True for value in outcomes),
                useful_parents=int(hit),
                unresolved_parents=int(unknown),
                conditional_recovered=int(inband is True and hit),
                conditional_total=int(inband is True),
                conditional_unresolved=int(inband is True and unknown),
                conditional_with_evaluable_refold=int(inband is True and evaluable > 0),
            )
        )
        for sibling, outcome in zip(siblings, outcomes, strict=True):
            sibling["joint_useful_refold"] = outcome
    columns = [
        "arm",
        "replicate",
        "backbone",
        "replicate_seed",
        "parent_path",
        "parent_elevation_deg",
        "parent_in_band",
        "endpoint_status",
        "requested_refolds",
        *COUNT_COLUMNS,
    ]
    parents = pd.DataFrame(parents, columns=columns)
    if parents.empty:
        return (
            parents,
            pd.DataFrame(columns=["arm", "replicate", *COUNT_COLUMNS]),
            pd.DataFrame(columns=["arm", *COUNT_COLUMNS]),
            pd.DataFrame(columns=EFFECT_COLUMNS),
        )
    reps = (
        parents.groupby(["arm", "replicate"], sort=True)[COUNT_COLUMNS]
        .sum()
        .reset_index()
    )
    totals = reps.groupby("arm", sort=True)[COUNT_COLUMNS].sum().reset_index()
    for frame in (reps, totals):
        frame["useful_parent_rate"] = frame.useful_parents / frame.requested
        frame["useful_parent_rate_upper_missing"] = (
            frame.useful_parents + frame.unresolved_parents
        ) / frame.requested
        frame["conditional_recovery_rate"] = (
            frame.conditional_recovered / frame.conditional_total.replace(0, np.nan)
        )
        frame["conditional_recovery_upper_missing"] = (
            frame.conditional_recovered + frame.conditional_unresolved
        ) / frame.conditional_total.replace(0, np.nan)
    return parents, reps, totals, _paired_effects(reps)


def _summary(rows: list[Row], key: str, *, signed: bool = False) -> str:
    values = [value for row in rows if (value := _finite(row.get(key))) is not None]
    if not values:
        return "— (n=0)"
    low, median, high = np.percentile(values, [10, 50, 90])
    spec = "+.2f" if signed else ".2f"
    return f"{median:{spec}} [{low:{spec}}, {high:{spec}}]; n={len(values)}"


def _ratio(n: int, d: int) -> str:
    return f"{n:,} / {d:,} ({100*n/d:.1f}%)" if d else "— (0 eligible)"


def _table(rows: list[list[Any]], columns: list[str]) -> str:
    return pd.DataFrame(rows, columns=columns).to_html(
        index=False, border=0, na_rep="—", escape=True
    )


def _json_records(frame: pd.DataFrame) -> list[Row]:
    return [
        {
            key: None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in row.items()
        }
        for row in frame.to_dict("records")
    ]


def graphics(
    folder: Path,
    backbones: list[Row],
    refolds: list[Row],
    interval: Any,
    effects: pd.DataFrame,
) -> str:
    """Draw signed stage distributions and refold versus its own parent."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 4), layout="constrained")
    axes = fig.subplots(1, 3)
    colors = {
        "ordinary": "#456a88",
        "legacy_tilt": "#ba8464",
        "surface_elevation": "#6b8b74",
        "guided": "#ba8464",
    }
    for rows, axis, title in zip(
        (backbones, refolds), axes[:2], ("Generated backbones", "Refolds"), strict=True
    ):
        for arm in sorted({row["arm"] for row in rows}):
            values = sorted(
                v
                for row in rows
                if row["arm"] == arm
                and (v := _finite(row.get("footprint_elevation_deg"))) is not None
            )
            if values:
                axis.step(
                    [-90, *values, 90],
                    [0, *((np.arange(len(values)) + 1) / len(values)), 1],
                    where="post",
                    color=colors.get(arm),
                    label=f"{arm} (n={len(values)})",
                )
        if interval is not None:
            axis.axvspan(*interval, color="#dcebdc", zorder=-1)
        axis.set(
            title=title,
            xlabel="Actual-footprint elevation (°)",
            ylabel="Cumulative fraction",
            xlim=(-90, 90),
            ylim=(0, 1.02),
        )
        if axis.lines:
            axis.legend(fontsize=7, loc="lower right", frameon=False)
    for arm in sorted({row["arm"] for row in refolds}):
        points = [
            (row["parent_footprint_elevation_deg"], row["footprint_elevation_deg"])
            for row in refolds
            if row["arm"] == arm
            and _finite(row.get("parent_footprint_elevation_deg")) is not None
            and _finite(row.get("footprint_elevation_deg")) is not None
        ]
        if points:
            axes[2].scatter(
                *zip(*points), s=12, alpha=0.55, color=colors.get(arm), label=arm
            )
    axes[2].plot([-90, 90], [-90, 90], "--", lw=0.8, color=".6")
    axes[2].set(
        title="Each refold versus its own parent",
        xlabel="Parent elevation (°)",
        ylabel="Refold elevation (°)",
        xlim=(-90, 90),
        ylim=(-90, 90),
    )
    if axes[2].get_legend_handles_labels()[0]:
        axes[2].legend(fontsize=7, frameon=False)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    (folder / "pose_comparison.png").write_bytes(buffer.getvalue())
    return base64.b64encode(buffer.getvalue()).decode()


def experiment_report(
    folder: Path,
    backbones: list[Row],
    refolds: list[Row],
    interval: Any,
    *,
    elevation_interval: Any = None,
    plan: Row | None = None,
) -> Row:
    """Write a standalone report answering compliance, recovery and parent yield."""
    parents, reps, totals, effects = elevation_tables(
        backbones, refolds, elevation_interval
    )
    for name, frame in (
        ("parents", parents),
        ("parent_recovery", parents),
        ("replicate_yields", reps),
        ("endpoint_totals", totals),
        ("paired_effects", effects),
    ):
        frame.to_csv(folder / f"{name}.csv", index=False)
    image = graphics(folder, backbones, refolds, elevation_interval, effects)
    compliance = []
    fidelity = []
    outcomes = []
    coverage = []
    diagnostics = []
    pair_counts = []
    for arm in sorted({row["arm"] for row in backbones}):
        bb = [row for row in backbones if row["arm"] == arm]
        rf = [row for row in refolds if row["arm"] == arm]
        name = ARM_LABELS.get(arm, arm)
        if plan and plan.get("gfp_pair_ablation"):
            pair_counts.append([
                name,
                _summary(bb, "nominal_guidance_pairs"),
                _summary(bb, "requested_guidance_pairs"),
                _summary(bb, "informative_guidance_pairs"),
            ])
        patch_measured = sum(
            _finite(row.get("patch_elevation_deg")) is not None for row in bb
        )
        fp_measured = sum(
            _finite(row.get("footprint_elevation_deg")) is not None for row in bb
        )
        patch_hits = sum(
            _inside(row.get("patch_elevation_deg"), elevation_interval) is True
            for row in bb
        )
        fp_hits = sum(
            _inside(row.get("footprint_elevation_deg"), elevation_interval) is True
            for row in bb
        )
        compliance.append(
            [
                name,
                len(bb),
                _summary(bb, "scaffold_signed_error_deg", signed=True),
                _summary(bb, "scaffold_absolute_error_deg"),
                _summary(bb, "patch_signed_error_deg", signed=True),
                _summary(bb, "patch_absolute_error_deg"),
                _summary(bb, "footprint_signed_error_deg", signed=True),
                _summary(bb, "footprint_absolute_error_deg"),
                (
                    _ratio(patch_hits, patch_measured)
                    if elevation_interval is not None
                    else "No elevation interval"
                ),
                (
                    _ratio(fp_hits, fp_measured)
                    if elevation_interval is not None
                    else "No elevation interval"
                ),
                _summary(bb, "patch_footprint_normal_difference_deg"),
            ]
        )
        fidelity.append(
            [
                name,
                len(rf),
                _summary(rf, "elevation_change_deg", signed=True),
                _summary(rf, "absolute_elevation_change_deg"),
                _summary(rf, "footprint_jaccard"),
                _ratio(
                    sum(row.get("requested_site_contact") is True for row in rf),
                    sum(row.get("requested_site_contact") is not None for row in rf),
                ),
                _summary(rf, "scDockQ"),
                _summary(rf, "cdr_contact_fraction"),
                _summary(rf, "framework_rmsd_after_target_fit_A"),
                _summary(rf, "self_fit_framework_rmsd_A"),
                _summary(rf, "cdr_rmsd_after_framework_fit_A"),
            ]
        )
        for stage, rows in (("backbone", bb), ("refold", rf)):
            for status, count in Counter(
                row.get("footprint_status", "missing structure") for row in rows
            ).items():
                diagnostics.append([name, stage, status, count])
    for row in totals.itertuples():
        outcomes.append(
            [
                ARM_LABELS.get(row.arm, row.arm),
                row.requested,
                row.generated,
                _ratio(row.useful_parents, row.requested),
                row.unresolved_parents,
                _ratio(row.conditional_recovered, row.conditional_total),
                row.conditional_unresolved,
                row.conditional_with_evaluable_refold,
            ]
        )
        coverage.append(
            [
                ARM_LABELS.get(row.arm, row.arm),
                row.requested,
                row.generated,
                row.rejected_guides,
                row.measured_parents,
                row.expected_refolds,
                row.available_refolds,
                row.missing_refolds,
                row.excess_refolds,
                row.refolds_measured_elevation,
                row.refolds_measured_site_contact,
                row.refolds_measured_scdockq,
                row.evaluable_refolds,
                row.unknown_refold_budget_parents,
            ]
        )
    rep_rows = [
        [
            ARM_LABELS.get(row.arm, row.arm),
            row.replicate,
            row.requested,
            _ratio(row.useful_parents, row.requested),
            _ratio(row.conditional_recovered, row.conditional_total),
            row.unresolved_parents,
            row.missing_refolds,
        ]
        for row in reps.itertuples()
    ]
    effect_rows = []
    for row in effects.itertuples():
        values = [
            f"{number:+.2f}" if (number := _finite(value)) is not None else "—"
            for value in (row.difference_pp, row.low95_pp, row.high95_pp)
        ]
        effect_rows.append(
            [
                row.comparison,
                row.metric,
                *values,
                row.paired_replicates,
                row.paired_replicates_with_both_denominators,
                row.valid_bootstrap_draws,
            ]
        )
    interval_label = (
        f"{elevation_interval[0]:g}° to {elevation_interval[1]:g}°"
        if elevation_interval is not None
        else "No elevation interval was requested; historical tilt bounds were not reinterpreted."
    )
    legacy_label = (
        f"{interval[0]:g}° to {interval[1]:g}°"
        if interval is not None
        else "not specified"
    )
    orientation_path = folder / "orientations.png"
    orientation_html = (
        f'<img alt="Requested versus scaffold and generated elevations, footprint distribution and normal differences" src="data:image/png;base64,{base64.b64encode(orientation_path.read_bytes()).decode()}">'
        if orientation_path.exists()
        else ""
    )
    files = [
        "backbones.csv",
        "refolds.csv",
        "parents.csv",
        "patch_contacts.csv",
        "parent_recovery.csv",
        "replicate_yields.csv",
        "endpoint_totals.csv",
        "paired_effects.csv",
        "summary.json",
        "orientations.png",
        "pose_comparison.png",
    ]
    links = " · ".join(f'<a href="{name}">{name}</a>' for name in files)
    fixture_note = (
        '<p class="note"><strong>Synthetic verification fixture.</strong> '
        "These coordinates and scores are invented to test the analysis and report. "
        "They are not model predictions or remeasurements of the archived MILTON run.</p>"
        if plan and plan.get("synthetic_fixture") is True
        else ""
    )
    pair_note = (
        '<h2>GFP distance-pair ablation</h2><p>Matched ordinary, 4, 8, 12 and 16 pair arms. All guided arms use nested subsets of the same fully validated 16-pair placement; the shared scaffold, seeds and evaluation band are unchanged. Counts are median [10th, 90th percentile] with measured denominator.</p>'
        + _table(pair_counts, ["Arm", "Nominal pairs", "Encoded pairs", "Finite-distance pairs"])
        if pair_counts else ""
    )
    css = """body{margin:0 auto;max-width:1250px;padding:36px 4%;font:15px/1.6 system-ui,sans-serif;color:#20272b;background:#f8fafb}h1,h2{line-height:1.2}h2{margin-top:2.1em}p{max-width:100ch}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{padding:9px;text-align:left;border-bottom:1px solid #d7e0e4}th{background:#eaf0f3}img{width:100%;height:auto}a{color:#335d79}.note{background:#edf2ed;padding:14px 18px;border-radius:5px}code{font-size:.92em}"""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Surface-elevation guidance experiment</title><style>{css}</style></head><body>
<h1>Surface-elevation guidance experiment</h1>
{fixture_note}
{pair_note}
<p><strong>Elevation evaluation band:</strong> {escape(interval_label)}<br><strong>Legacy-control placement-ray tilt:</strong> {escape(legacy_label)}. These intervals are independent experimental choices.</p>
<p>The signed approach angle is asin(unit(F − C) · n): +90° is CDR-end-on, 0° is parallel, and −90° is framework-end-on. F and C are the unweighted Cα centroids of all framework and CDR residues in the validated run mapping, including CDR residues without contacts.</p>
<h2>1. Did generation obey the request?</h2>
<p>The placed-scaffold error checks geometry before distance encoding. Generated coordinates are then measured relative to the recorded prospective patch plane and independently against the surface they actually contact. Errors are observed minus requested; angles retain their sign.</p>
{orientation_html}<div class="scroll">{_table(compliance,["Arm","Requested parents","Scaffold signed error °","Scaffold absolute error °","Generated patch signed error °","Generated patch absolute error °","Generated footprint signed error °","Generated footprint absolute error °","Patch in band / measured","Footprint in band / measured","Patch–footprint normal difference °"])}</div>
<p>Continuous summaries are median [10th, 90th percentile], followed by the measured denominator. Band occupancies use measured angles; requested-parent denominators and missing values remain in the accounting below. Ordinary and legacy-guide samples have no requested elevation point, so their point-error fields stay empty.</p>
<h2>2. What survived refolding?</h2>
<img alt="Backbone and refold signed elevation distributions and each refold paired with its own parent" src="data:image/png;base64,{image}">
<div class="scroll">{_table(fidelity,["Arm","Refold files","Signed elevation change °","Absolute elevation change °","Footprint Jaccard","Requested-site contact / measured","scDockQ","CDR contact fraction","Framework RMSD after target fit Å","Framework self-fit RMSD Å","CDR RMSD after framework fit Å"])}</div>
<p>Every refold is compared with its own backbone. Footprint overlap uses chain-qualified antigen residue identities; A:42 and B:42 are different residues. Similar elevation alone does not establish recovery of the same docking pose. scDockQ and RMSDs assess additional aspects of pose recovery; the CDR contact fraction is shown independently.</p>
<h2>3. Which guide produced more useful parents?</h2>
<div class="note">A useful parent has at least one refold that simultaneously has an actual-footprint elevation in the evaluation band, contacts at least one requested antigen residue, and has scDockQ ≥0.23. All three criteria must hold for the same refold. Five successful siblings still count as one parent opportunity.</div>
<div class="scroll">{_table(outcomes,["Arm","Requested parents","Generated parents","Useful parents / requested","Unresolved parent outcomes","Recovered / initially in-band parents","Unresolved initially in-band","In-band parents with an evaluable refold"])}</div>
<p>Conditional recovery uses all parents whose own generated footprint elevation was in-band, including those with missing refolds. Missing parent elevations are outside that conditional denominator and remain visible in accounting. A refold fails the joint criterion as soon as one measured criterion is false; missing values in its other criteria still remain in the coverage counts. Confirmed successes form a lower bound; CSV tables also give upper bounds if every unresolved parent could succeed.</p>
<h2>Matched replicates and uncertainty</h2>
<div class="scroll">{_table(effect_rows,["Comparison","Endpoint","Difference pp","Lower 95%","Upper 95%","Matched replicates","Replicates with both denominators","Valid bootstrap draws"])}</div>
<p>Effects compare pooled parent rates with 5,000 paired-replicate bootstrap resamples. All sibling refolds stay within their parent and replicate. Intervals require at least two matched replicates with nonzero denominators in both arms. Sparse conditional denominators can leave uncertainty unavailable.</p>
<div class="scroll">{_table(rep_rows,["Arm","Replicate","Requested parents","Useful / requested","Conditional recovery","Unresolved parents","Missing refolds"])}</div>
<h2>Missing-data denominators and run accounting</h2>
<div class="scroll">{_table(coverage,["Arm","Requested","Generated","Rejected guides","Parent elevations measured","Expected refolds (known budgets)","Available refolds","Missing refolds","Excess refolds","Refold elevations measured","Refold site contact measured","Refold scDockQ measured","Refolds with all criteria measured","Parents with unknown refold budget"])}</div>
<div class="scroll">{_table(diagnostics,["Arm","Stage","Footprint measurement status","Structures"])}</div>
<h2>Measurement and compatibility notes</h2>
<p>Each footprint includes all selected antigen-chain heavy atoms within 5 Å of any represented VHH heavy atom, including framework contacts. Contacted atom coordinates are averaged within each chain-qualified antigen residue. The outward density direction uses Gaussian scales of 4 and 6 Å, the revised audit’s weak-gradient weighting, equal-scale pooling, and single-residue-omission sensitivity. A weak local site does not veto a supported footprint.</p>
<p>The numeric angle is retained for flat poses. The separate <code>flat</code> display label requires the complete supported sensitivity range to lie within ±5° under the audit’s support conditions. A flat antigen surface can support any VHH elevation. Missing coordinates, absent contacts, unresolved gradients, opposing normals and incompatible scales have distinct statuses.</p>
<p>The prospective patch normal is transported into each observed complex by a local antigen Cα fit; the actually contacted footprint is estimated independently, even after a site change. Patch-normal fit coverage and RMSD are recorded. Multichain scDockQ uses the explicit binder–antigen union score; a maximum of separate pair scores is never substituted.</p>
<p>Historical placement-ray tilt, CDR-facing angle, and centroid-pointing are retained as legacy diagnostics. Untagged historical <code>requested_deg</code> and old tilt intervals never become elevation requests. Existing CDR/framework badge rules are unchanged. Represented atom coverage may differ between masked backbones and refolds.</p>
<h2>Tables and figures</h2><p>{links}</p></body></html>"""
    (folder / "report.html").write_text(html, encoding="utf-8")
    return dict(
        totals=_json_records(totals),
        replicates=_json_records(reps),
        paired_effects=_json_records(effects),
        endpoint="same-refold in-band footprint + requested-site contact + scDockQ>=0.23",
        sampling_unit="requested parent",
        uncertainty_unit="paired replicate",
    )
