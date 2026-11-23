"""Standalone paired-study report using the same measured rows as the CSV export."""

import base64
import io
from collections import defaultdict
from html import escape

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
            parents.append(dict(arm=b["arm"], replicate=b["replicate"], backbone=b["backbone"],
                                endpoint=endpoint, requested=1, generated=int(b["file_exists"]),
                                measurable=int(qualified is not None), qualified=int(bool(qualified)),
                                with_refolds=int(any(v is not None for v in outcomes)),
                                qualified_with_refolds=int(bool(qualified) and any(v is not None for v in outcomes)),
                                recovered=int(bool(qualified) and hit), final_hit=int(hit)))
    parents = pd.DataFrame(parents)
    reps = parents.drop(columns="backbone").groupby(["endpoint", "replicate", "arm"]).sum()
    totals = reps.groupby(["endpoint", "arm"]).sum()
    effects = []
    for endpoint, block in reps.groupby(level="endpoint"):
        wide = block.droplevel("endpoint").unstack("arm")
        if not {"ordinary", "guided"} <= set(wide.columns.get_level_values("arm")):
            continue
        for metric, num, den in [("Backbone yield", "qualified", "requested"),
                                 ("Conditional recovery", "recovered", "qualified_with_refolds"),
                                 ("Final yield", "final_hit", "requested")]:
            a = wide[[(num,"guided"),(den,"guided"),(num,"ordinary"),(den,"ordinary")]].dropna().to_numpy()
            if len(a) == 0:
                continue
            def diff(x):
                with np.errstate(divide="ignore", invalid="ignore"):
                    return 100 * (x[...,0]/x[...,1] - x[...,2]/x[...,3])
            draws = np.random.default_rng(101).integers(len(a), size=(5000,len(a)))
            sims = diff(a[draws].sum(axis=1))
            valid = sims[np.isfinite(sims)]
            ci = np.percentile(valid, [2.5,97.5]) if len(a)>1 and len(valid) else [np.nan]*2
            effects.append(dict(endpoint=endpoint, metric=metric, difference_pp=diff(a.sum(0)),
                                low95_pp=ci[0], high95_pp=ci[1], paired_replicates=len(a)))
    return parents, reps.reset_index(), totals.reset_index(), pd.DataFrame(effects, columns=[
        "endpoint","metric","difference_pp","low95_pp","high95_pp","paired_replicates"])


def graphics(folder, backbones, refolds, interval, effects):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(11, 3.8), layout="constrained")
    axes = fig.subplots(1, 3)
    colors = {"ordinary":"#305f88", "guided":"#a64b29"}
    for rows, ax, title in zip([backbones, refolds], axes, ["Generated backbones", "Refold predictions"]):
        for arm in sorted({r["arm"] for r in rows}):
            v = sorted(r["spatial_deg"] for r in rows if r["arm"]==arm and r.get("spatial_deg") is not None)
            if v:
                ax.step([0,*v,180],[0,*((np.arange(len(v))+1)/len(v)),1],where="post",
                        color=colors.get(arm),label=f"{arm.title()} (n={len(v):,})")
        if interval:
            ax.axvspan(*interval, color="#dcebdc",zorder=-1)
        ax.set(title=title,xlabel="Spatial approach angle (°)",ylabel="Cumulative fraction",xlim=(0,180),ylim=(0,1.02))
        if ax.lines:
            ax.legend(fontsize=8,loc="lower right")
    ax=axes[2]
    panel=effects[effects.endpoint==("pose" if interval else "patch")]
    ax.axvline(0,color="#888",lw=1)
    for i,row in enumerate(panel.itertuples()):
        ax.plot([row.low95_pp,row.high95_pp],[i,i],color="#305f88",lw=2)
        ax.plot(row.difference_pp,i,"o",color="#172d40")
    ax.set(yticks=list(range(len(panel))),yticklabels=panel.metric, xlabel="Guided − ordinary (pp)",
           title="Requested pose" if interval else "CDR-driven at patch")
    ax.invert_yaxis()
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=160)
    data=buf.getvalue()
    (folder/"pose_comparison.png").write_bytes(data)
    return base64.b64encode(data).decode()


def experiment_report(folder, backbones, refolds, interval):
    parents,reps,totals,effects=tables(backbones,refolds,interval)
    for name,frame in [("parent_recovery",parents),("replicate_yields",reps),
                       ("endpoint_totals",totals),("paired_effects",effects)]:
        frame.to_csv(folder/f"{name}.csv",index=False)
    image=graphics(folder,backbones,refolds,interval,effects)
    def ratio(n,d):
        return f"{n:,} / {d:,} ({100*n/d:.1f}%)" if d else "—"
    def table(rows,columns):
        return pd.DataFrame(rows,columns=columns).to_html(index=False,border=0,na_rep="—")
    outcomes=[]
    for t in totals.itertuples():
        outcomes.append([ENDPOINTS[t.endpoint],t.arm.title(),ratio(t.qualified,t.requested),
                         ratio(t.recovered,t.qualified_with_refolds),ratio(t.final_hit,t.requested)])
    coverage=[]
    rmsds=[]
    for arm in sorted({r["arm"] for r in backbones}):
        b=[r for r in backbones if r["arm"]==arm]
        f=[r for r in refolds if r["arm"]==arm]
        generated=sum(r["file_exists"] for r in b)
        rejected=sum(r.get("geometry_status")=="rejected_guide" for r in b)
        coverage.append([arm.title(),len(b),generated,rejected,len(b)-generated-rejected,
                         sum(r.get("spatial_deg") is not None for r in b),len(f)])
        measures=[]
        for metric in ("self_fit_framework_rmsd_A","cdr_rmsd_after_framework_fit_A","framework_rmsd_after_target_fit_A"):
            v=[r[metric] for r in f if r.get(metric) is not None]
            if v:
                q=np.percentile(v,[10,50,90])
                measures.append(f"{q[1]:.2f} [{q[0]:.2f}–{q[2]:.2f}]")
            else:
                measures.append("—")
        rmsds.append([arm.title(),*measures])
    title="Target-relative approach experiment"
    subtitle=(f"Preferred spatial approach {interval[0]:g}–{interval[1]:g}°" if interval else "No spatial interval specified")
    css="""body{margin:0 auto;max-width:1140px;padding:40px 5%;font:16px/1.55 system-ui;color:#192c38;background:#fafbfc}h1,h2{line-height:1.2}h2{margin-top:2em}p{max-width:90ch}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{text-align:left;padding:8px;border-bottom:1px solid #d4dee4}th{background:#eaf0f4}img{width:100%;height:auto}.scroll{overflow:auto}a{color:#305f88}"""
    values=effects.copy()
    values["endpoint"]=values.endpoint.map(ENDPOINTS)
    for c in ["difference_pp","low95_pp","high95_pp"]:
        values[c]=values[c].map(lambda x:f"{x:+.1f}" if np.isfinite(x) else "—")
    values.columns=["Endpoint","Measure","Difference (pp)","Lower 95%","Upper 95%","Replicate pairs"]
    files=["backbones.csv","refolds.csv","patch_contacts.csv","parent_recovery.csv","replicate_yields.csv",
           "endpoint_totals.csv","paired_effects.csv","summary.json","pose_comparison.png","orientations.png"]
    links=" · ".join(f'<a href="{f}">{f}</a>' for f in files)
    html=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{title}</title><style>{css}</style>
<h1>{title}</h1><p>{escape(subtitle)}</p>
<h2>What did each arm produce?</h2><div class="scroll">{table(outcomes,["Endpoint","Arm","Backbones / requested","Recovery among qualifying parents¹","Final hits / requested²"])}</div>
<p>¹ A qualifying parent with a measurable refold counts once if any refold meets the same endpoint. ² Final hits include parents that first meet the endpoint after refolding. Recovery here describes predicted geometry. Experimental binding requires separate evidence.</p>
<h2>Angle distributions and paired differences</h2><img alt="Cumulative angle distributions for each stage and paired yield and recovery differences" src="data:image/png;base64,{image}">
<p>The shaded interval marks the requested spatial angle. At its upper edge the curve gives the fraction at or below that angle. Refold curves include every measurable prediction from each parent.</p>
<div class="scroll">{values.to_html(index=False,border=0)}</div><p>Effects are percentage-point differences of pooled rates. Intervals are paired-replicate percentile bootstrap estimates from 5,000 resamples. Sparse events can produce wide or unstable intervals.</p>
<h2>Did the framework change shape or move relative to the antigen?</h2><div class="scroll">{table(rmsds,["Arm","Framework after self-fit","CDRs after framework fit","Framework after local antigen fit"])}</div><p>Refold Cα RMSD in Å, median [10th–90th percentile]. A small self-fit RMSD with a large antigen-fit RMSD indicates that the framework retains its shape while its docking position or orientation changes.</p>
<h2>Run accounting</h2><div class="scroll">{table(coverage,["Arm","Requested","Generated","Rejected guides","Other missing","Spatial angles measured","Refold files"])}</div>
<h2>Measurement definitions</h2><p>The spatial angle compares the selected patch’s outward normal to the patch-to-framework direction. The CDR-driven classification compares the framework-to-CDR axis to the direction from the framework to the centroid of the actually contacted antigen residues. It requires an angle ≤20° and a CDR share above 60% of contacting binder residues. At-patch qualification additionally requires a CDR contact with a selected residue. Contacts use represented heavy atoms within 5 Å. The separate 8 Å Cα pair count is a proximity proxy.</p>
<p>Backbone yield uses all requested IDs. Missing and unmeasurable poses contribute no confirmed hit. The structure tables retain diagnostics and out-of-interval measurements.</p>
<h2>Data and figures</h2><p>{links}</p></html>"""
    (folder/"report.html").write_text(html,encoding="utf-8")
    return totals.to_dict("records")
