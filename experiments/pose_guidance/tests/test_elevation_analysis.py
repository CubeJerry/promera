"""Synthetic regression tests for elevation analysis and parent-level endpoints."""
from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from promera_angles import analysis
from promera_angles.analysis import analyse, measurements, scdockq_from_confidence
from promera_angles.demo import PATTERN, SEQUENCE, binder_coords, fixture, write_cif
from promera_angles.report import elevation_tables, useful_refold


def parent(index: int, angle: float | None=80, *, arm: str="surface_elevation",
           rep: int=1, budget: int=5, exists: bool=True) -> dict:
    return dict(arm=arm,replicate=rep,backbone=index,footprint_elevation_deg=angle,
                file_exists=exists,path=f"{arm}/{rep}/sample{index}/backbone.cif",
                expected_refolds=budget)


def fold(index: int, angle: float | None=80, *, arm: str="surface_elevation",
         rep: int=1, site: bool | None=True, score: float | None=.5) -> dict:
    return dict(arm=arm,replicate=rep,backbone=index,footprint_elevation_deg=angle,
                requested_site_contact=site,scDockQ=score,file_exists=True)


def test_same_refold_must_meet_all_criteria() -> None:
    rows=[fold(0,score=.1),fold(0,angle=60),fold(0,site=False)]
    assert all(useful_refold(row,[70,90]) is False for row in rows)
    assert useful_refold(fold(0,score=.23),[70,90]) is True
    assert useful_refold(fold(0,score=None),[70,90]) is None
    _,_,totals,_=elevation_tables([parent(0,budget=3)],rows,[70,90])
    assert totals.iloc[0].useful_parents == 0
    assert totals.iloc[0].evaluable_refolds == 3


def test_siblings_count_once_and_missing_parents_stay_in_denominator() -> None:
    bb=[parent(0),parent(1),parent(2,angle=20),parent(3,angle=None,exists=False),parent(4)]
    rf=([fold(0) for _ in range(5)]+[fold(1,score=.1),fold(1,angle=60),fold(1,site=False)]
        +[fold(2)]+[fold(4,score=None) for _ in range(5)])
    parents,reps,totals,effects=elevation_tables(bb,rf,[70,90])
    total=totals.iloc[0]
    assert (total.requested,total.generated,total.measured_parents)==(5,4,4)
    assert (total.useful_parents,total.conditional_recovered,total.conditional_total)==(2,1,3)
    assert (total.expected_refolds,total.available_refolds,total.missing_refolds)==(25,14,11)
    assert total.unresolved_parents == 3
    assert total.conditional_unresolved == 2
    assert len(parents)==5 and len(reps)==1 and effects.empty
    assert parents.set_index("backbone").loc[0,"successful_refolds"]==5
    assert all(row.get("joint_useful_refold") is True for row in rf[:5])


def test_uncertainty_uses_replicates_not_sibling_predictions() -> None:
    bb=[]; rf=[]
    for rep in (1,2):
        for arm in ("ordinary","legacy_tilt","surface_elevation"):
            bb.append(parent(0,arm=arm,rep=rep,budget=1))
            success=arm=="surface_elevation" or (arm=="legacy_tilt" and rep==1)
            rf.append(fold(0,arm=arm,rep=rep,score=.5 if success else .1))
    _,_,_,effects=elevation_tables(bb,rf,[70,90])
    duplicated=[deepcopy(row) for row in rf for _ in range(5)]
    repeated_bb=[dict(row,expected_refolds=5) for row in bb]
    _,_,_,again=elevation_tables(repeated_bb,duplicated,[70,90])
    pd.testing.assert_frame_equal(effects,again)
    effect=effects[(effects.arm=="surface_elevation") & (effects.reference=="ordinary")
                   & (effects.metric=="Useful parents / requested")].iloc[0]
    assert effect.difference_pp==100 and effect.low95_pp==100 and effect.high95_pp==100
    assert effect.paired_replicates==2
    _,_,_,single=elevation_tables([r for r in bb if r["replicate"]==1],
                                  [r for r in rf if r["replicate"]==1],[70,90])
    assert single.low95_pp.isna().all() and single.high95_pp.isna().all()


def test_unknown_elevation_interval_is_not_legacy_tilt() -> None:
    bb=[parent(0)]
    rf=[fold(0)]
    parents,_,totals,_=elevation_tables(bb,rf,None)
    assert parents.iloc[0].endpoint_status=="no_elevation_interval"
    assert totals.iloc[0].conditional_total==0
    assert totals.iloc[0].useful_parents==0
    assert rf[0]["joint_useful_refold"] is None


def test_scdockq_uses_union_without_selecting_best_target_pair() -> None:
    plan=dict(binder_chain="VHH",sequences={"A":"AAA","B":"AAA"})
    raw={"A_B":{"DockQ":1.0},"A_VHH":{"DockQ":.99},"binder_antigen":{"DockQ":.25,"iRMSD":2.0}}
    observed=scdockq_from_confidence({"scDockQ":raw},plan)
    assert observed["scDockQ"]==.25 and observed["scDockQ_iRMSD"]==2
    assert observed["scDockQ_source"]=="binder_antigen_union"
    raw.pop("binder_antigen")
    assert scdockq_from_confidence({"scDockQ":raw},plan)["scDockQ"] is None
    single=dict(binder_chain="VHH",sequences={"A":"AAA"})
    assert scdockq_from_confidence({"scDockQ":{"A_VHH":{"DockQ":.3}}},single)["scDockQ"]==.3
    assert scdockq_from_confidence({"scDockQ":{"0_1":{"DockQ":.4}}},single)["scDockQ"]==.4
    for value in (None,True,float("nan"),1.1,-.1):
        assert scdockq_from_confidence({"scDockQ":value},single)["scDockQ"] is None


def atom_frame(plan: dict, bc: np.ndarray) -> pd.DataFrame:
    names={"A":"ALA","G":"GLY","S":"SER","X":"UNK"}
    records=[]
    target=list(zip(plan["keys"],"A"*len(plan["keys"]),plan["coords"]))
    nano=[(("VHH",i),aa,xyz) for i,(aa,xyz) in enumerate(zip(SEQUENCE,bc),1)]
    for (chain,pos),aa,xyz in target+nano:
        for atom,offset in (("CA",[0,0,0]),("N",[-1,.4,0]),("C",[1,.5,0]),("O",[1.3,.9,.3])):
            p=np.asarray(xyz)+offset
            records.append(dict(label_asym_id=chain,pos=pos,label_atom_id=atom,label_comp_id=names[aa],
                                Cartn_x=p[0],Cartn_y=p[1],Cartn_z=p[2]))
    return pd.DataFrame(records)


def test_measurement_is_rigid_invariant_and_missing_patch_does_not_veto_footprint(monkeypatch: pytest.MonkeyPatch) -> None:
    plan=fixture()
    atoms=atom_frame(plan,binder_coords(plan,0,20))
    monkeypatch.setattr(analysis,"atom_table",lambda path:atoms.copy())
    first,_,_=measurements(Path("synthetic.cif"),plan,None)
    assert first["patch_elevation_deg"]==pytest.approx(70,abs=1e-6)
    assert first["footprint_status"]=="ok"
    rotated=atoms.copy()
    xyz=["Cartn_x","Cartn_y","Cartn_z"]
    rotation=Rotation.from_rotvec([.3,-.7,.2]).as_matrix()
    rotated[xyz]=rotated[xyz].to_numpy()@rotation+[4,16,-9]
    monkeypatch.setattr(analysis,"atom_table",lambda path:rotated.copy())
    moved,_,_=measurements(Path("rotated.cif"),plan,None)
    for key in ("patch_elevation_deg","footprint_elevation_deg","patch_footprint_normal_difference_deg"):
        assert moved[key]==pytest.approx(first[key],abs=1e-6)
    assert moved["footprint_residue_keys"]==first["footprint_residue_keys"]
    # The actual heavy-atom footprint survives a missing local Cα alignment.
    nearby={tuple(plan["keys"][i]) for i in plan["frame"]["neighbourhood"]}
    keep=[not (row.label_atom_id=="CA" and (row.label_asym_id,row.pos) in nearby)
          for row in atoms.itertuples()]
    partial=atoms[keep]
    monkeypatch.setattr(analysis,"atom_table",lambda path:partial.copy())
    missing,_,shape=measurements(Path("missing_patch.cif"),plan,None)
    assert missing["footprint_status"]=="ok"
    assert "patch_elevation_deg" not in missing and shape is None


def test_refold_overlap_is_chain_qualified_and_angle_sign_is_preserved() -> None:
    before=dict(path="parent.cif",file_exists=True,footprint_elevation_deg=80,patch_elevation_deg=78,
                requested_site_contact=True,footprint_residue_keys=[["A",1],["A",2]])
    after=dict(footprint_elevation_deg=-20,patch_elevation_deg=-18,requested_site_contact=True,
               footprint_residue_keys=[["B",1],["A",2]])
    analysis._pair_refold(after,before,None,None)
    assert after["elevation_change_deg"]==-100
    assert after["absolute_elevation_change_deg"]==100
    assert after["footprint_jaccard"]==pytest.approx(1/3)
    assert after["parent_footprint_recovered_fraction"]==.5


def test_analysis_report_real_synthetic_cifs_and_historical_compatibility(tmp_path: Path) -> None:
    plan=fixture()
    plan.update(experiment_definition="surface_elevation_v1",elevation_interval_deg=[70,90],
                legacy_tilt_interval_deg=[0,20],refolds_per_parent=5)
    coords=np.array(plan["coords"])
    chains={c:(seq,coords[[i for i,key in enumerate(plan["keys"]) if key[0]==c]])
            for c,seq in plan["sequences"].items()}
    jobs=[]
    for rep in (1,2):
        for arm in ("ordinary","legacy_tilt","surface_elevation"):
            root=tmp_path/arm/f"rep-{rep}"
            jobs.append(dict(arm=arm,replicate=rep,seed=100+rep,start=0,count=1,output=str(root)))
            sample=root/plan["target"]/"sample0"
            write_cif(sample/"backbone.cif",dict(chains,VHH=(SEQUENCE,binder_coords(plan,0,10))))
            record=dict(guidance_definition={"ordinary":"ordinary","legacy_tilt":"legacy_placement_ray",
                                             "surface_elevation":"surface_elevation"}[arm])
            if arm=="surface_elevation":
                record.update(requested_elevation_deg=80,realized_scaffold_elevation_deg=80)
            elif arm=="legacy_tilt":
                record["requested_deg"]=10
            (sample/"angle.json").write_text(json.dumps(record))
            for index in range(5):
                path=sample/"refolds"/f"design_refold{index}.cif"
                write_cif(path,dict(chains,VHH=(SEQUENCE,binder_coords(plan,0,10))))
                score=.4 if arm=="surface_elevation" else .1
                path.with_name(path.stem+"_confidence.json").write_text(
                    json.dumps({"scDockQ":{"binder_antigen":{"DockQ":score}}}))
    output=tmp_path/"analysis"
    summary=analyse(output,jobs,plan,[0,20])
    assert set(summary["arms"])=={"ordinary","legacy_tilt","surface_elevation"}
    assert summary["available_refolds"]==30
    assert summary["elevation_interval_deg"]==[70,90]
    totals={row["arm"]:row for row in summary["pose_endpoints"]["totals"]}
    assert totals["surface_elevation"]["useful_parents"]==2
    assert totals["surface_elevation"]["requested"]==2
    assert totals["ordinary"]["useful_parents"]==0
    assert totals["legacy_tilt"]["useful_parents"]==0
    with (output/"refolds.csv").open() as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==30 and all(row["parent_path"] for row in rows)
    assert sum(row["joint_useful_refold"]=="True" for row in rows)==10
    assert (output/"orientations.png").read_bytes().startswith(b"\x89PNG")
    html=(output/"report.html").read_text()
    for text in ("Did generation obey","What survived","useful parents","same refold","scDockQ","±5"):
        assert text.lower() in html.lower()
    historical=fixture()
    old=analyse(tmp_path/"historical",[jobs[1]],historical,[0,20])
    assert old["elevation_interval_deg"] is None
    with (tmp_path/"historical"/"backbones.csv").open() as handle:
        row=next(csv.DictReader(handle))
    assert row["requested_elevation_deg"]=="" and float(row["requested_legacy_tilt_deg"])==10
    assert row["footprint_in_band"]==""
