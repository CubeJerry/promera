"""CPU orchestration checks; invented coordinates stand in for model inference."""

from __future__ import annotations

import builtins
import csv
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from promera_angles import run
from promera_angles.demo import PATTERN, SEQUENCE, binder_coords, fixture, write_cif
from promera_angles.settings import SURFACE_DEFINITION, study_arms, validate_settings


def surface_spec(**overrides):
    result = dict(
        enabled=True, definition=SURFACE_DEFINITION, three_arms=True,
        elevation_interval_deg=[70, 90], legacy_tilt_interval_deg=[0, 20],
        replicates=1, seed=19,
    )
    result.update(overrides)
    return result


def csv_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def test_three_arm_budget_and_seed_matching(tmp_path):
    spec = validate_settings(surface_spec())
    jobs = run.split_jobs(
        tmp_path, 7, 3, False, True, 19, arms=study_arms(spec)
    )
    assert len(jobs) == 9
    for arm in ("ordinary", "legacy_tilt", "surface_elevation"):
        rows = [job for job in jobs if job["arm"] == arm]
        assert [(job["start"], job["count"]) for job in rows] == [(0, 3), (3, 2), (5, 2)]
        assert [job["seed"] for job in rows] == [19, 20, 21]
        assert sum(job["count"] for job in rows) == 7
        assert len({job["output"] for job in rows}) == 3


def test_legacy_split_and_definition_remain_legacy(tmp_path):
    spec = validate_settings(dict(enabled=True, paired=True, interval_deg=[0, 20]))
    assert "elevation_interval_deg" not in spec
    assert study_arms(spec) == ["ordinary", "guided"]
    jobs = run.split_jobs(tmp_path, 7, 3, True, True, 19)
    assert [job["seed"] for job in jobs] == [19, 19, 20, 20, 21, 21]
    assert jobs[0]["output"] == str(tmp_path / "ordinary/rep-001")


@pytest.mark.parametrize("interval", [[-91, 10], [80, 70], [70, 91], [float("nan"), 90]])
def test_invalid_elevation_ranges_fail(interval):
    spec = surface_spec()
    spec["elevation_interval_deg"] = interval
    with pytest.raises(ValueError):
        validate_settings(spec)


@pytest.mark.parametrize("change", [
    {"interval_deg": [0, 20]},
    {"definition": "legacy_placement_ray_v1"},
    {"enabled": "true"},
    {"paired": True},
    {"cdr_lateral_offset_A": [0, float("inf")]},
])
def test_settings_reject_ambiguous_or_invalid_values(change):
    spec = surface_spec()
    spec.update(change)
    with pytest.raises(ValueError):
        validate_settings(spec)


def test_separate_namespaces_and_refold_budget():
    with pytest.raises(ValueError, match="both be enabled"):
        run.study_settings({
            "approach_angle": {"enabled": True, "interval_deg": [0, 20]},
            "surface_elevation": {"enabled": True, "elevation_interval_deg": [70, 90]},
        })
    with pytest.raises(ValueError, match="surface_elevation block"):
        run.study_settings({
            "approach_angle": {"enabled": True, "elevation_interval_deg": [70, 90]},
        })
    spec = run.study_settings({
        "surface_elevation": {"enabled": True, "elevation_interval_deg": [70, 90]},
    })
    assert validate_settings(spec)["definition"] == SURFACE_DEFINITION
    assert run.refold_budget({"inverse_folder": {"type": "solublempnn", "num_seqs": 2}}) == {
        "sequences_per_parent": 2, "refolds_per_sequence": 5, "refolds_per_parent": 10,
    }
    assert run.refold_budget({"inverse_folder": {"type": "none", "num_seqs": 2}})["refolds_per_parent"] == 0


@pytest.fixture
def study_fixture(tmp_path, monkeypatch):
    plan = deepcopy(fixture())
    schema = tmp_path / "target.json"
    schema.write_text("{}")
    plan.update(
        input=str(schema), structure=str(schema),
        schema_sha256=run.digest(schema), structure_sha256=run.digest(schema),
        legacy_frame=deepcopy(plan["frame"]),
    )
    calls = []

    def resolve(cfg, path, *, surface_elevation=False):
        assert surface_elevation
        return deepcopy(plan)

    monkeypatch.setattr(run, "resolve_target", resolve)
    monkeypatch.setattr(run, "build_shape", lambda *args: {})
    coords = np.asarray(plan["coords"], float)
    antigen = {
        chain: (sequence, coords[[i for i, key in enumerate(plan["keys"]) if key[0] == chain]])
        for chain, sequence in plan["sequences"].items()
    }

    def produce(task, cfg, dest, seed, model, weights):
        assert task == "promera_angles.task.TargetedDesign"
        assert "approach_angle" not in cfg and "surface_elevation" not in cfg
        trial = cfg["angle_trial"]
        calls.append(deepcopy(trial))
        definition = trial["guidance_definition"]
        if definition == "surface_elevation":
            assert trial["interval"] == [70.0, 90.0]
        elif definition == "legacy_placement_ray":
            assert trial["interval"] == [0.0, 20.0]
        else:
            assert definition == "ordinary" and trial["interval"] is None
        for index in range(trial["start"], trial["start"] + cfg["num_backbones"]):
            sample = Path(cfg["output"]) / plan["target"] / f"sample{index}"
            binder = binder_coords(plan, 10, 15)
            chains = dict(antigen, VHH=(SEQUENCE, binder))
            write_cif(sample / "backbone.cif", chains)
            record = {
                "guidance_definition": definition, "record_status": "ok",
                "requested_elevation_deg": 75 if definition == "surface_elevation" else None,
                "realized_scaffold_elevation_deg": 75,
                "requested_legacy_tilt_deg": 10 if definition == "legacy_placement_ray" else None,
            }
            run.write_json(sample / "angle.json", record)
            for refold_index in range(5):
                path = sample / "refolds" / f"sample{index}_design0_refold{refold_index}.cif"
                write_cif(path, chains)
                run.write_json(path.with_name(path.stem + "_confidence.json"), {
                    "scDockQ": {"binder_antigen": {
                        "DockQ": 0.8, "fnat": 1.0, "iRMSD": 0.0, "LRMSD": 0.0,
                    }}
                })

    monkeypatch.setattr(run, "infer", produce)
    cfg = {
        "binder": {"type": "vhh", "framework": PATTERN, "chain": "VHH"},
        "num_backbones": 2, "inverse_folder": {"type": "solublempnn", "num_seqs": 1},
    }
    return cfg, plan, calls, produce


def test_actual_three_arm_lifecycle_and_html_six_parents_thirty_refolds(tmp_path, study_fixture):
    cfg, plan, calls, produce = study_fixture
    root = tmp_path / "study"
    run.execute(cfg, root, surface_spec())
    assert len(calls) == 3
    manifest = json.loads((root / "approach_angle/manifest.json").read_text())
    resolved = json.loads((root / "approach_angle/plan.json").read_text())
    assert resolved["experiment_definition"] == SURFACE_DEFINITION
    assert resolved["refolds_per_parent"] == 5
    assert manifest["version"] == 2 and "interval_deg" not in manifest
    assert manifest["elevation_interval_deg"] == [70.0, 90.0]
    assert manifest["legacy_tilt_interval_deg"] == [0.0, 20.0]
    backbones = csv_rows(root / "analysis/backbones.csv")
    refolds = csv_rows(root / "analysis/refolds.csv")
    parents = csv_rows(root / "analysis/parents.csv")
    assert len(backbones) == len(parents) == 6
    assert len(refolds) == 30
    assert {row["arm"] for row in backbones} == {"ordinary", "legacy_tilt", "surface_elevation"}
    assert (root / "analysis/report.html").is_file()
    summary = json.loads((root / "analysis/summary.json").read_text())
    assert summary["available_refolds"] == 30
    assert summary["expected_refolds_per_parent"] == 5
    assert all(value["status"] == "completed" for value in summary["job_status"].values())
    assert all(job["seed"] == 19 for job in manifest["jobs"])


def test_new_study_continues_after_one_failed_arm_and_reports_every_parent(
    tmp_path, monkeypatch, study_fixture
):
    cfg, plan, calls, produce = study_fixture
    attempted = []

    def fail_first(task, config, *args):
        attempted.append(config["angle_trial"]["guidance_definition"])
        if len(attempted) == 1:
            raise RuntimeError("injected first-arm failure")
        return produce(task, config, *args)

    monkeypatch.setattr(run, "infer", fail_first)
    root = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="injected first-arm failure"):
        run.execute(cfg, root, surface_spec())
    assert attempted == ["ordinary", "legacy_placement_ray", "surface_elevation"]
    rows = csv_rows(root / "analysis/backbones.csv")
    assert len(rows) == 6
    summary = json.loads((root / "analysis/summary.json").read_text())
    assert summary["job_status"]["ordinary-1"]["status"] == "failed"
    assert summary["job_status"]["legacy_tilt-1"]["status"] == "completed"
    assert summary["job_status"]["surface_elevation-1"]["status"] == "completed"
    assert summary["arms"]["ordinary"]["expected"] == 2
    assert (root / "analysis/report.html").is_file()


def test_failed_scaffold_preparation_still_writes_requested_parent_accounting(
    tmp_path, monkeypatch, study_fixture
):
    cfg, plan, calls, produce = study_fixture

    def fail(*args):
        raise RuntimeError("injected scaffold failure")

    monkeypatch.setattr(run, "build_shape", fail)
    root = tmp_path / "failed-prepare"
    with pytest.raises(RuntimeError, match="injected scaffold failure"):
        run.execute(cfg, root, surface_spec(), prepare_only=True)
    assert calls == []
    assert len(csv_rows(root / "analysis/backbones.csv")) == 6
    summary = json.loads((root / "analysis/summary.json").read_text())
    assert summary["preparation_status"]["status"] == "failed"
    assert all(value["status"] == "missing" for value in summary["job_status"].values())


def test_preview_has_explicit_units_and_both_independent_intervals(tmp_path, study_fixture):
    cfg, plan, calls, produce = study_fixture
    root = tmp_path / "preview"
    spec = surface_spec()
    spec["cdr_lateral_offset_A"] = [2, -3]
    run.execute(cfg, root, spec, preview=True)
    preview = json.loads((root / "approach_angle/preview.json").read_text())
    assert preview["definition"] == SURFACE_DEFINITION
    assert preview["elevation_interval_deg"] == [70, 90]
    assert preview["legacy_tilt_interval_deg"] == [0, 20]
    assert "interval_deg" not in preview
    assert preview["cdr_lateral_offset_A"] == [2, -3]
    assert calls == []
    assert not (root / "analysis").exists()


def test_tampered_plan_has_a_failed_receipt_and_cannot_run(tmp_path, study_fixture):
    cfg, plan, calls, produce = study_fixture
    root = tmp_path / "tampered"
    run.execute(cfg, root, surface_spec(), prepare_only=True)
    manifest = json.loads((root / "approach_angle/manifest.json").read_text())
    path = root / "approach_angle/plan.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="plan digest"):
        run.run_job(root, manifest, 0)
    receipt = json.loads((root / "approach_angle/ordinary-1-status.json").read_text())
    assert receipt["status"] == "failed"
    assert calls == []


def test_cli_layout_and_ambiguous_old_flags(tmp_path, capsys):
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"binder": {"type": "vhh"}, "num_backbones": 7}))
    base = ["--config", str(path), "--output", str(tmp_path / "out"), "--layout-only"]
    run.main(base + [
        "--elevation", "70", "90", "--legacy-tilt", "0", "20",
        "--three-arms", "--backbones", "7", "--replicates", "3",
    ])
    assert capsys.readouterr().out.strip() == "9"
    run.main(base + ["--angle", "0", "20", "--paired", "--replicates", "3"])
    assert capsys.readouterr().out.strip() == "6"
    with pytest.raises(SystemExit) as error:
        run.main(base + ["--angle", "0", "20", "--elevation", "70", "90"])
    assert error.value.code == 2
    path.write_text(json.dumps({
        "binder": {"type": "vhh"}, "num_backbones": 7,
        "approach_angle": {"enabled": True, "interval_deg": [0, 20]},
    }))
    with pytest.raises(SystemExit) as error:
        run.main(base + ["--elevation", "70", "90"])
    assert error.value.code == 2


@pytest.mark.parametrize("block,expected", [
    ({"surface_elevation": {
        "enabled": True, "elevation_interval_deg": [70, 90],
        "legacy_tilt_interval_deg": [0, 20], "three_arms": True,
    }}, SURFACE_DEFINITION),
    ({"approach_angle": {"enabled": True, "interval_deg": [0, 20]}}, None),
])
def test_native_entry_dispatches_before_importing_torch(tmp_path, monkeypatch, block, expected):
    task = tmp_path / "task.json"
    task.write_text(json.dumps({
        "binder": {"type": "vhh"}, "num_backbones": 2, "output": str(tmp_path / "out"), **block,
    }))
    model = tmp_path / "model.json"
    model.write_text("{}")
    seen = []
    monkeypatch.setattr(run, "execute", lambda cfg, root, spec, **kw: seen.append(validate_settings(spec)))
    monkeypatch.setattr(sys, "argv", [
        "promera", "--task", "promera.inference.Design",
        "--task_config", str(task), "--model_config", str(model),
    ])
    script = Path(__file__).resolve().parents[3] / "promera/__main__.py"
    with pytest.raises(SystemExit) as exit_code:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_code.value.code == 0
    assert seen[0].get("definition") == expected


def test_disabled_native_experiment_preserves_the_ordinary_entry(tmp_path, monkeypatch):
    task = tmp_path / "ordinary.json"
    task.write_text(json.dumps({
        "binder": {"type": "vhh"}, "num_backbones": 2,
        "surface_elevation": {"enabled": False},
    }))
    model = tmp_path / "model.json"
    model.write_text("{}")
    called = []
    monkeypatch.setattr(run, "execute", lambda *a, **k: called.append(True))
    monkeypatch.setattr(sys, "argv", [
        "promera", "--task", "promera.inference.Design",
        "--task_config", str(task), "--model_config", str(model),
    ])
    native_import = builtins.__import__

    def observe(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("ordinary entry reached")
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", observe)
    script = Path(__file__).resolve().parents[3] / "promera/__main__.py"
    with pytest.raises(ImportError, match="ordinary entry reached"):
        runpy.run_path(str(script), run_name="__main__")
    assert called == []
