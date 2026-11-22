import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.pose_guidance.pose_geometry import framework_map
from promera_angles.analysis import analyse
from promera_angles.demo import (
    PATTERN,
    SEQUENCE,
    binder_coords,
    demonstrate,
    fixture,
    write_cif,
)
from promera_angles.geometry import (
    GeometryError,
    degrees,
    orientation,
    patch_frame,
    placement,
    scaffold_shape,
)
from promera_angles.run import split_jobs


def test_frame_is_local_multichain_and_rigid_motion_equivariant():
    plan = fixture()
    coords = np.array(plan["coords"])
    r = Rotation.from_rotvec([0.3, -0.7, 0.1]).as_matrix()
    t = [14, -9, 7]
    seen = patch_frame(coords @ r + t, plan["selected"])
    assert seen["landmarks"] == plan["frame"]["landmarks"]
    np.testing.assert_allclose(
        seen["normal"], np.array(plan["frame"]["normal"]) @ r, atol=1e-8
    )
    far = patch_frame(np.vstack([coords, coords + [80, 0, 100]]), plan["selected"])
    np.testing.assert_allclose(far["normal"], plan["frame"]["normal"], atol=1e-10)
    assert {plan["keys"][i][0] for i in plan["selected"]} == set(plan["sequences"])
    assert patch_frame(coords, plan["selected"][:1])["normal"][2] > 0.9


def test_invalid_or_incompatible_geometry():
    plan = fixture()
    coords = np.array(plan["coords"])
    for values, selection in (
        (coords[:4], [0]),
        (coords, [0, len(coords) - 1]),
        (coords[coords[:, 2] == 0], [24]),
        (np.vstack([coords, [2, 0, 9]]), plan["selected"]),
        (np.array([[i, 0.0, 0.0] for i in range(15)]), [0]),
    ):
        with pytest.raises(GeometryError):
            patch_frame(values, selection)


def test_oriented_scaffold_uses_no_external_pose_and_preserves_angles():
    plan = fixture()
    fw, cdr = framework_map(SEQUENCE, PATTERN)
    coords = binder_coords(plan, 15, 30)
    shape = scaffold_shape(coords, fw, cdr)
    r = Rotation.from_rotvec([0.7, 0.25, -0.1]).as_matrix()
    moved = scaffold_shape(coords @ r + [9, 3, 44], fw, cdr)
    assert shape["fixed_indices"] == moved["fixed_indices"]
    np.testing.assert_allclose(shape["local_xyz"], moved["local_xyz"], atol=1e-8)
    for lo, hi in ((0, 0), (12, 20), (90, 90)):
        placed, rec = placement(plan["frame"], shape, [lo, hi], 12)
        assert lo - 1e-7 <= rec["requested_deg"] <= hi + 1e-7
        assert degrees(
            np.array(rec["ray"]), np.array(plan["frame"]["normal"])
        ) == pytest.approx(rec["requested_deg"], abs=1e-6)
        assert placed.shape == (4, 3)
    metrics = orientation(plan["frame"], coords[fw], coords[cdr])
    assert metrics["facing_deg"] - metrics["spatial_deg"] == pytest.approx(15)


def test_budget_and_matched_replicates(tmp_path):
    jobs = split_jobs(tmp_path, 10, 3, True, True, 31)
    for arm in ("ordinary", "guided"):
        assert [(j["start"], j["count"]) for j in jobs if j["arm"] == arm] == [
            (0, 4),
            (4, 3),
            (7, 3),
        ]
    assert [j["seed"] for j in jobs] == [31, 31, 32, 32, 33, 33]
    with pytest.raises(ValueError):
        split_jobs(tmp_path, 2, 3, True, True, 0)


def test_report_all_structures_and_missingness(tmp_path):
    summary = demonstrate(tmp_path)
    assert summary["arms"]["guided"]["spatial_deg"]["measured"] == 6
    assert summary["arms"]["guided"]["spatial_deg"]["in_interval"] == 4
    assert summary["arms"]["guided"]["facing_deg"]["in_interval"] == 2
    assert summary["guided_minus_ordinary"]["spatial_deg"]["matched_pairs"] == 6
    assert summary["guided_minus_ordinary"]["spatial_deg"]["bootstrap95"] is not None
    folder = tmp_path / "analysis"
    assert (folder / "orientations.png").read_bytes().startswith(b"\x89PNG")
    with (folder / "refolds.csv").open() as f:
        r = list(csv.DictReader(f))
    assert len(r) == 4 and float(r[0]["framework_rmsd_after_target_fit_A"]) > 0
    assert any(row["lost_interval_spatial_deg"] == "True" for row in r)
    with (folder / "patch_contacts.csv").open() as f:
        assert {r["chain"] for r in csv.DictReader(f)} == {"Left", "Right"}
    jobs = split_jobs(tmp_path / "missing", 2, 1, True, True, 0)
    missing = analyse(tmp_path / "partial", jobs, fixture(), [0, 20])
    assert missing["arms"]["ordinary"]["generated"] == 0
    assert missing["guided_minus_ordinary"]["spatial_deg"]["matched_pairs"] == 0


def test_feature_parity_and_batched_collation(monkeypatch):
    # Real collator and task adapter; tinyprot is only needed to import the native Design class.
    from promera.data import utils
    from promera_angles.task import condition

    plan = fixture()
    fw, cdr = framework_map(SEQUENCE, PATTERN)
    plan["shape"] = scaffold_shape(binder_coords(plan, 0, 0), fw, cdr)

    def setup():
        codes = {"A": "ALA", "S": "SER", "G": "GLY", "X": "UNK"}
        chains = {
            c: SimpleNamespace(rname=[codes[a] for a in seq])
            for c, seq in dict(plan["sequences"], VHH=SEQUENCE).items()
        }
        # Deliberately interleave the binder and the two antigen chains.
        chains = {c: chains[c] for c in ("Right", "VHH", "Left")}
        n = sum(len(c.rname) for c in chains.values())
        flags = np.zeros(n, int)
        starts = {}
        offset = 0
        for name, c in chains.items():
            starts[name] = offset
            offset += len(c.rname)
        for i in plan["selected"]:
            c, p = plan["keys"][i]
            flags[starts[c] + p - 1] = 1
        mask = np.eye(n, dtype=bool)
        return dict(
            struct=SimpleNamespace(chains=chains),
            is_epitope=flags,
            restype=np.zeros(n),
            distogram_mask=mask,
            distogram_emb=np.eye(n, dtype=int),
        )

    control = setup()
    features = setup()
    original = control["distogram_mask"].copy()
    condition(control, plan, 17, None, 6)
    np.testing.assert_array_equal(control["distogram_mask"], original)
    assert (control["distogram_emb"] == np.eye(len(original))).all()
    guide = condition(features, plan, 17, [0, 20], 6)
    assert len(guide["pairs"]) == 16
    assert np.sum(features["distogram_mask"] ^ original) == 32
    assert np.array_equal(control["is_epitope"], features["is_epitope"])
    assert np.all(features["distogram_emb"][original] == 1)
    monkeypatch.setattr(
        utils,
        "_token_pair_keys",
        [*utils._token_pair_keys, "distogram_mask", "distogram_emb"],
    )
    collated = utils.collate([features, control], token_multiple=32)
    assert collated["distogram_mask"].shape[1] == collated["distogram_mask"].shape[2]
    features["is_epitope"][:] = 0
    with pytest.raises(ValueError, match="epitope flags"):
        condition(features, plan, 17, [0, 20], 6)


def test_native_mapping_and_preparation(tmp_path):
    from promera_angles.run import execute
    from promera_angles.structures import resolve_target

    plan = fixture()
    coords = np.array(plan["coords"])
    chains = {
        c: (seq, coords[[i for i, k in enumerate(plan["keys"]) if k[0] == c]])
        for c, seq in plan["sequences"].items()
    }
    cif = tmp_path / "target.cif"
    write_cif(cif, chains)
    schema = tmp_path / "input.json"
    schema.write_text(
        json.dumps(
            {c: {"type": "protein", "sequence": v[0]} for c, v in chains.items()}
        )
    )
    site = {}
    for i in plan["selected"]:
        c, p = plan["keys"][i]
        site.setdefault(c, []).append(p)
    cfg = dict(
        input=str(schema),
        binder=dict(type="vhh", chain="VHH", framework=PATTERN),
        epitope_residue_map=site,
        target_template={"path": str(cif), "chains": list(chains)},
        num_backbones=3,
    )
    result = resolve_target(cfg)
    assert [list(plan["keys"][i]) for i in plan["selected"]] == [
        list(result["keys"][i]) for i in result["selected"]
    ]
    assert all(
        int(result["labels"][i]["auth_seq_id"]) == result["keys"][i][1] + 37
        for i in result["selected"]
    )
    execute(
        cfg,
        tmp_path / "preview",
        {"enabled": True, "interval_deg": [0, 20]},
        preview=True,
    )
    assert not (tmp_path / "preview" / "analysis").exists()
    assert (
        json.loads(
            (tmp_path / "preview" / "approach_angle" / "preview.json").read_text()
        )["status"]
        == "valid"
    )
    cfg["epitope_residue_map"] = {"Unknown": [1]}
    with pytest.raises(ValueError, match="native epitope"):
        execute(cfg, tmp_path / "bad", {}, preview=True)
    assert (
        json.loads((tmp_path / "bad" / "approach_angle" / "preview.json").read_text())[
            "status"
        ]
        == "unavailable"
    )


def test_paired_lifecycle_writes_analysis_on_success_and_failure(tmp_path, monkeypatch):
    from promera_angles import run

    plan = fixture()
    target = tmp_path / "schema.json"
    target.write_text("{}")
    # This lifecycle fixture bypasses model execution. Separate tests exercise native mapping.
    plan.update(
        input=str(target),
        structure=str(target),
        schema_sha256=run.digest(target),
        structure_sha256=run.digest(target),
    )
    monkeypatch.setattr(run, "resolve_target", lambda cfg, path: plan.copy())
    monkeypatch.setattr(run, "build_shape", lambda *args: {})
    tc = np.array(plan["coords"])
    chains = {
        c: (s, tc[[i for i, k in enumerate(plan["keys"]) if k[0] == c]])
        for c, s in plan["sequences"].items()
    }

    def produce(task, cfg, dest, seed, model, weights):
        assert task == "promera_angles.task.TargetedDesign"
        spec = cfg["angle_trial"]
        angle = 12 if spec["interval"] else 40
        for i in range(spec["start"], spec["start"] + cfg["num_backbones"]):
            path = Path(cfg["output"]) / plan["target"] / f"sample{i}" / "backbone.cif"
            write_cif(
                path, dict(chains, VHH=(SEQUENCE, binder_coords(plan, angle, angle)))
            )

    cfg = dict(binder={"type": "vhh"}, num_backbones=4)
    spec = dict(enabled=True, paired=True, replicates=2, interval_deg=[0, 20])
    monkeypatch.setattr(run, "infer", produce)
    root = tmp_path / "success"
    run.execute(cfg, root, spec)
    summary = json.loads((root / "analysis" / "summary.json").read_text())
    assert summary["guided_minus_ordinary"]["spatial_deg"]["matched_pairs"] == 4
    assert len(summary["job_status"]) == 4

    def fail(*args):
        raise RuntimeError("injected model failure")

    monkeypatch.setattr(run, "infer", fail)
    root = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="injected model failure"):
        run.execute(cfg, root, spec)
    partial = json.loads((root / "analysis" / "summary.json").read_text())
    assert partial["arms"]["ordinary"]["expected"] == 4
    assert partial["arms"]["guided"]["generated"] == 0


@pytest.mark.parametrize("multichain", [False, True])
def test_installed_native_features(tmp_path, multichain):
    """Run with a TinyProt cache containing standard amino acids, UNK and taxonomy.

    Multichain native masks require the NOMINEE-patched Design runtime.
    """
    root = tmp_path
    import hashlib
    import inspect

    import numpy as np
    from omegaconf import OmegaConf

    from experiments.pose_guidance.pose_geometry import digest, framework_map
    from promera.data.utils import collate
    from promera.inference.cofolding import Cofolding
    from promera.inference.design import Design
    from promera_angles.analysis import write_json
    from promera_angles.demo import PATTERN, SEQUENCE, binder_coords, fixture, write_cif
    from promera_angles.geometry import scaffold_shape
    from promera_angles.structures import resolve_target
    from promera_angles.task import TargetedDesign

    if multichain and "epitope_residue_map" not in inspect.getsource(
        Design.__getitem__
    ):
        pytest.skip("requires NOMINEE multichain Design adapter")
    p = fixture()
    coords = np.array(p["coords"])
    chains = {
        c: (seq, coords[[i for i, k in enumerate(p["keys"]) if k[0] == c]])
        for c, seq in p["sequences"].items()
    }
    if not multichain:
        chains = {"Plane": ("A" * len(coords), coords)}
    target = root / "target.cif"
    write_cif(target, chains)
    schema = root / "synthetic.json"
    write_json(
        schema,
        {
            c: {"type": "protein", "sequence": v[0], "use_msa": True}
            for c, v in chains.items()
        },
    )
    msa = root / "msa"
    msa.mkdir()
    for seq, c in chains.values():
        key = hashlib.sha256((seq + "\n").encode()).hexdigest()
        (msa / f"{key}.a3m").write_text(">query\n" + seq + "\n")
    cfg = dict(
        input=str(schema),
        output=str(root / "generated"),
        msa_dir=str(msa),
        binder=dict(
            type="vhh",
            chain="VHH",
            framework=PATTERN,
            cdr_lengths=dict(cdrh1=1, cdrh2=2, cdrh3=1),
            paratope_from_cdrs=True,
        ),
        num_backbones=2,
        skip_existing=False,
        epitope_chain=next(iter(chains)),
        epitope_residues=[i + 1 for i in p["selected"]],
        target_chains=list(chains),
        target_template=dict(
            path=str(target), chain=next(iter(chains)), chains=list(chains)
        ),
        inverse_folder=dict(type="none"),
    )
    if multichain:
        cfg["epitope_residue_map"] = {
            c: [
                pos
                for i, (name, pos) in enumerate(p["keys"])
                if i in p["selected"] and name == c
            ]
            for c in chains
        }
        cfg["epitope_residues"] = []
    plan = resolve_target(cfg)
    fw, cdr = framework_map(SEQUENCE, PATTERN)
    plan["shape"] = scaffold_shape(binder_coords(p, 0, 0), fw, cdr)
    path = root / "plan.json"
    write_json(path, plan)
    trial = dict(
        plan=str(path),
        sha256=digest(path),
        seed=0,
        start=0,
        interval=[0, 20],
        standoff=6,
    )
    ds = TargetedDesign(OmegaConf.create(dict(cfg, angle_trial=trial)))
    guided = ds[0]
    ds0 = TargetedDesign(
        OmegaConf.create(dict(cfg, angle_trial=dict(trial, interval=None)))
    )
    baseline = ds0[0]
    native = Design(OmegaConf.create(cfg))[0]
    assert np.array_equal(native["is_epitope"], baseline["is_epitope"])
    assert np.array_equal(native["distogram_mask"], baseline["distogram_mask"])
    assert np.sum(guided["distogram_mask"] ^ baseline["distogram_mask"]) == 32
    collate([guided, ds[1]], token_multiple=32)
    mono = root / "mono"
    mono.mkdir()
    write_json(
        mono / "scaffold.json",
        {"VHH": {"type": "protein", "sequence": SEQUENCE, "use_msa": False}},
    )
    m = Cofolding(
        OmegaConf.create(
            dict(
                input=str(mono),
                output=str(root),
                skip_existing=False,
                num_seeds=1,
                diffusion_samples=1,
                msa_dir=str(msa),
                assert_msa=False,
            )
        )
    )
    assert len(m[0]["restype"]) == len(SEQUENCE)
