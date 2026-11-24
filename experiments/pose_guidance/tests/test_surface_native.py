"""Real native featurization/refolding boundary on a small synthetic protein.

No checkpoint inference is performed. Only the model forward call is replaced,
after the real native refold featurizer has built its fresh validation features.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="CPU native contract needs PyTorch")
pytest.importorskip("tinyprot", reason="CPU native contract needs TinyProt")


def test_native_elevation_features_and_unrestrained_refold(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from experiments.pose_guidance.pose_geometry import digest, framework_map
    from promera.data.utils import collate
    from promera_angles.demo import PATTERN, SEQUENCE, binder_coords, fixture, write_cif
    from promera_angles.geometry import scaffold_shape
    from promera_angles.structures import resolve_target
    from promera_angles.task import TargetedDesign

    original = fixture()
    coords = np.asarray(original["coords"])
    antigen_sequence = "A" * len(coords)
    target = tmp_path / "antigen.cif"
    write_cif(target, {"Plane": (antigen_sequence, coords)})
    schema = tmp_path / "synthetic.json"
    schema.write_text(
        json.dumps(
            {
                "Plane": {
                    "type": "protein",
                    "sequence": antigen_sequence,
                    "use_msa": True,
                }
            }
        )
    )
    msa = tmp_path / "msa"
    msa.mkdir()
    key = hashlib.sha256((antigen_sequence + "\n").encode()).hexdigest()
    (msa / f"{key}.a3m").write_text(">query\n" + antigen_sequence + "\n")
    cfg = dict(
        input=str(schema),
        output=str(tmp_path / "generated"),
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
        epitope_chain="Plane",
        epitope_residues=[i + 1 for i in original["selected"]],
        target_chains=["Plane"],
        target_template=dict(path=str(target), chain="Plane"),
        inverse_folder=dict(type="none"),
    )
    plan = resolve_target(cfg, surface_elevation=True)
    assert plan["experiment_definition"] == "surface_elevation_v1"
    assert plan["prospective_antigen_heavy_atom_count"] == 4 * len(coords)
    fw, cdr = framework_map(SEQUENCE, PATTERN)
    plan["shape"] = scaffold_shape(binder_coords(original, 0, 0), fw, cdr)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    trial = dict(
        plan=str(plan_path),
        sha256=digest(plan_path),
        seed=37,
        start=0,
        interval=[70, 90],
        standoff=6,
        guidance_definition="surface_elevation",
        cdr_lateral_offset=[0, 0],
    )
    guided_task = TargetedDesign(OmegaConf.create(dict(cfg, angle_trial=trial)))
    control_task = TargetedDesign(
        OmegaConf.create(
            dict(
                cfg,
                angle_trial=dict(trial, interval=None, guidance_definition="ordinary"),
            )
        )
    )
    guided, control = guided_task[0], control_task[0]
    record = guided["angle_record"]
    assert record["guidance_definition"] == "surface_elevation"
    assert 70 <= record["requested_elevation_deg"] <= 90
    assert record["realized_scaffold_elevation_deg"] == pytest.approx(
        record["requested_elevation_deg"], abs=2e-6
    )
    assert len(record["pairs"]) == 16
    np.testing.assert_array_equal(guided["is_epitope"], control["is_epitope"])
    delta = guided["distogram_mask"] ^ control["distogram_mask"]
    assert delta.sum() == 32
    np.testing.assert_array_equal(delta, delta.T)
    np.testing.assert_array_equal(
        guided["distogram_emb"][control["distogram_mask"]],
        control["distogram_emb"][control["distogram_mask"]],
    )
    collate([guided, guided_task[1]], token_multiple=32)
    generated_mask = guided["distogram_mask"].copy()
    captured = {}

    class ForwardReached(Exception):
        pass

    def inspect_forward(feats, **kwargs):
        captured.update(feats)
        raise ForwardReached

    model = SimpleNamespace(pairformer_forward=inspect_forward)
    with pytest.raises(ForwardReached):
        guided_task._refold_batch(
            model,
            [(guided["struct"], "VHH", SEQUENCE.replace("X", "S"))],
            1,
            SimpleNamespace(diffusion_samples=5),
            torch.device("cpu"),
        )
    assert captured
    for name in ("distogram_mask", "is_epitope"):
        value = captured.get(name)
        assert value is None or not bool(value.any())
    assert model.pairformer_forward is inspect_forward
    np.testing.assert_array_equal(guided["distogram_mask"], generated_mask)
