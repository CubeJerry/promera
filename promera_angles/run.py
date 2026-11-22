"""Launch a native targeted run or a paired study. Report on exit, even on failure."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from experiments.pose_guidance.pose_geometry import digest

from .analysis import analyse, write_json
from .structures import resolve_target, scaffold_from_cif


def read_config(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def config_paths(cfg, base):
    def absolute(value):
        p = Path(os.path.expandvars(str(value))).expanduser()
        return str((base / p).resolve())

    for key in ("input", "msa_dir"):
        if cfg.get(key):
            cfg[key] = absolute(cfg[key])
    for item in (cfg.get("target_template"), cfg["binder"].get("framework_template")):
        if item and item.get("path"):
            item["path"] = absolute(item["path"])
    return cfg


def split_jobs(root, count, reps, paired, enabled, seed):
    if type(count) is not int or type(reps) is not int or not 1 <= reps <= count:
        raise ValueError(
            "need integer budgets with 1 <= replicates <= backbones per condition"
        )
    if type(seed) is not int or not 0 <= seed < 2**32 - reps:
        raise ValueError("seed must fit an unsigned 32-bit stream for each replicate")
    jobs = []
    offset = 0
    for rep in range(reps):
        size = count // reps + (rep < count % reps)
        for arm in (
            ["ordinary", "guided"] if paired else ["guided" if enabled else "ordinary"]
        ):
            output = root / arm / f"rep-{rep + 1:03}" if paired else root
            jobs.append(
                dict(
                    arm=arm,
                    replicate=rep + 1,
                    seed=seed + rep,
                    count=size,
                    start=offset,
                    output=str(output),
                )
            )
        offset += size
    return jobs


def infer(task, cfg, dest, seed, model, weights):
    write_json(dest, cfg)
    cmd = [
        sys.executable,
        "-m",
        "promera_angles.worker",
        str(seed),
        "--task",
        task,
        "--task_config",
        str(dest),
        "data.workers=0",
        "trainer.devices=1",
    ]
    if weights:
        cmd += ["--weights", str(weights)]
    if model:
        cmd += ["--model_config", str(model)]
    subprocess.run(cmd, check=True)


def build_shape(cfg, plan, root, seed, model, weights):
    # Only intramolecular coordinates are used. No example scaffold or docked pose.
    import random

    from promera.inference.design import _build_binder_chain

    chain = cfg["binder"]["chain"]
    entry = _build_binder_chain(OmegaConf.create(cfg["binder"]), random.Random(seed))
    entry["use_msa"] = False
    inputs = root / "monomer-input"
    inputs.mkdir()
    write_json(inputs / "scaffold.json", {chain: entry})
    out = root / "monomer"
    mono = dict(
        input=str(inputs),
        output=str(out),
        msa_dir=cfg["msa_dir"],
        assert_msa=False,
        diffusion_samples=1,
        num_seeds=1,
        recycling_steps=cfg["recycling_steps"],
        diffusion_steps=cfg["diffusion_steps"],
        diffusion=cfg["diffusion"],
        skip_existing=False,
        save_distogram=False,
        save_traj=False,
        save_full_confidence=False,
    )
    for key in ("amp", "amp_diffusion"):
        if key in cfg:
            mono[key] = cfg[key]
    infer(
        "promera.inference.Cofolding",
        mono,
        root / "monomer-config.json",
        seed,
        model,
        weights,
    )
    path = out / "scaffold" / "scaffold_seed0_samp0.cif"
    if not path.is_file():
        raise RuntimeError(
            "target-free scaffold inference returned without a structure"
        )
    return dict(
        scaffold_from_cif(path, chain, plan["framework"]), scaffold_sha256=digest(path)
    )


def run_job(root, manifest, index):
    meta = root / "approach_angle"
    job = manifest["jobs"][index]
    cfg = manifest["base_config"]
    path = meta / "plan.json"
    if digest(path) != manifest["plan_sha256"]:
        raise ValueError("prepared plan digest changed before sampling")
    plan = json.loads(path.read_text())
    for field, sha in (("input", "schema_sha256"), ("structure", "structure_sha256")):
        if digest(plan[field]) != plan[sha]:
            raise ValueError(f"prepared {field} changed before sampling")
    trial = dict(
        plan=str(path),
        sha256=manifest["plan_sha256"],
        seed=job["seed"],
        start=job["start"],
        interval=manifest["interval_deg"] if job["arm"] == "guided" else None,
        standoff=manifest["settings"].get("standoff_A", 6.0),
    )
    config = dict(
        cfg,
        output=job["output"],
        num_backbones=job["count"],
        skip_existing=False,
        angle_trial=trial,
    )
    tag = f"{job['arm']}-{job['replicate']}"
    if config.get("nominee"):
        config["nominee"] = dict(
            config["nominee"], summary_csv=str(meta / f"{tag}-summary.csv")
        )
    receipt = meta / f"{tag}-status.json"
    write_json(receipt, {"status": "running"})
    try:
        infer(
            "promera_angles.task.TargetedDesign",
            config,
            meta / f"{tag}.json",
            job["seed"],
            manifest["model_config"],
            manifest["weights"],
        )
    except Exception as exc:
        write_json(receipt, {"status": "failed", "reason": str(exc)})
        raise
    write_json(receipt, {"status": "completed"})


def finish(root, manifest, plan):
    result = analyse(
        root / "analysis", manifest["jobs"], plan, manifest["interval_deg"]
    )
    statuses = {}
    for j in manifest["jobs"]:
        tag = f"{j['arm']}-{j['replicate']}"
        receipt = root / "approach_angle" / f"{tag}-status.json"
        statuses[tag] = (
            json.loads(receipt.read_text())
            if receipt.exists()
            else {"status": "missing"}
        )
    result["job_status"] = statuses
    write_json(root / "analysis" / "summary.json", result)
    print(f"Analysis: {root / 'analysis' / 'summary.json'}", flush=True)
    if any(r["status"] != "completed" for r in statuses.values()):
        raise RuntimeError(
            "one or more design jobs did not complete; see analysis/summary.json"
        )
    if any(a["generated"] != a["expected"] for a in result["arms"].values()):
        raise RuntimeError("incomplete backbone generation; see analysis/backbones.csv")


def execute(
    cfg,
    root,
    spec,
    *,
    model=None,
    weights=None,
    preview=False,
    model_settings=None,
    prepare_only=False,
):
    cfg = deepcopy(cfg)
    cfg.pop("approach_angle", None)
    if cfg["binder"]["type"] != "vhh":
        raise ValueError("the CDR-facing definition requires a tagged VHH framework")
    allowed = {
        "enabled",
        "paired",
        "interval_deg",
        "backbones",
        "replicates",
        "seed",
        "target_structure",
        "standoff_A",
    }
    if set(spec) - allowed:
        raise ValueError(f"unrecognised angle settings: {sorted(set(spec) - allowed)}")
    enabled, paired = spec.get("enabled", False), spec.get("paired", False)
    if (
        type(enabled) is not bool
        or type(paired) is not bool
        or (paired and not enabled)
    ):
        raise ValueError("enabled and paired must be booleans; paired requires enabled")
    interval = spec.get("interval_deg")
    if enabled or interval is not None:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("provide two interval_deg endpoints")
        interval = list(map(float, interval))
        if (
            not np.isfinite(interval).all()
            or not 0 <= interval[0] <= interval[1] <= 180
        ):
            raise ValueError("require 0 <= minimum <= maximum <= 180 degrees")
    standoff = float(spec.get("standoff_A", 6))
    if not np.isfinite(standoff) or standoff <= 0:
        raise ValueError("standoff_A must be positive and finite")
    root = Path(root).resolve()
    jobs = split_jobs(
        root,
        spec.get("backbones", cfg["num_backbones"]),
        spec.get("replicates", 1),
        paired,
        enabled,
        spec.get("seed", 0),
    )
    # Ordinary NOMINEE may have created the batch directory and its inputs.
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "approach_angle"
    meta.mkdir(exist_ok=False)
    if model_settings is not None:
        model = meta / "model-config.json"
        write_json(model, model_settings)
    try:
        plan = resolve_target(cfg, spec.get("target_structure"))
    except (KeyError, ValueError, OSError) as exc:
        write_json(meta / "preview.json", dict(status="unavailable", reason=str(exc)))
        raise
    collision = [
        Path(j["output"]) / plan["target"]
        for j in jobs
        if (Path(j["output"]) / plan["target"]).exists()
    ]
    if collision:
        raise FileExistsError(
            f"existing target outputs at {collision[0]}; select a fresh output directory"
        )
    preview_data = dict(
        version=1,
        status="valid",
        kind="intended angular region",
        units="angstrom, degrees",
        interval_deg=interval,
        standoff_A=standoff,
        **plan["frame"],
        selected=[
            dict(
                chain=plan["keys"][i][0],
                position=plan["keys"][i][1],
                **plan["labels"][i],
            )
            for i in plan["selected"]
        ],
    )
    write_json(meta / "preview.json", preview_data)
    if preview:
        return
    if (
        int(os.environ.get("SLURM_NNODES", "1")) != 1
        or int(os.environ.get("SLURM_NTASKS", "1")) != 1
    ):
        raise ValueError("run one experiment driver on one GPU per Slurm job")
    cfg["input"] = plan["input"]
    manifest = dict(
        version=1,
        jobs=jobs,
        settings=spec,
        interval_deg=interval,
        base_config=cfg,
        model_config=str(model) if model else None,
        weights=str(weights or os.environ.get("PROMERA_WEIGHTS", "")),
        image=os.environ.get("POSE_RUNTIME_IMAGE"),
        commit=os.environ.get("POSE_EXPERIMENT_COMMIT"),
    )
    write_json(meta / "manifest.json", manifest)
    plan_path = meta / "plan.json"
    write_json(plan_path, plan)
    try:
        if enabled:
            plan["shape"] = build_shape(
                cfg, plan, meta, spec.get("seed", 0), model, weights
            )
            write_json(plan_path, plan)
        manifest["plan_sha256"] = digest(plan_path)
        write_json(meta / "manifest.json", manifest)
        if not prepare_only:
            for index in range(len(jobs)):
                run_job(root, manifest, index)
    except Exception:
        if not prepare_only:
            analyse(root / "analysis", jobs, plan, interval)
        raise
    if not prepare_only:
        finish(root, manifest, plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layout-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--stage",
        choices=["all", "prepare", "job", "finish"],
        default="all",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--config", type=Path, help="ordinary targeted Promera task YAML/JSON"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--angle",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="enable the angular preference",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--paired", action="store_true", help="also run a matched ordinary control"
    )
    mode.add_argument(
        "--ordinary", action="store_true", help="measure an ordinary control only"
    )
    p.add_argument(
        "--backbones", type=int, help="total per condition, divided across replicates"
    )
    p.add_argument("--replicates", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--target-structure",
        type=Path,
        help="geometry-only mmCIF if distinct from target_template",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="write the intended geometry with no inference",
    )
    p.add_argument("--weights", type=Path)
    p.add_argument("--model-config", type=Path)
    args = p.parse_args()
    if args.stage in ("job", "finish"):
        root = args.output.resolve()
        meta = root / "approach_angle"
        manifest = json.loads((meta / "manifest.json").read_text())
        if args.stage == "job":
            run_job(root, manifest, int(os.environ["SLURM_ARRAY_TASK_ID"]))
        else:
            finish(root, manifest, json.loads((meta / "plan.json").read_text()))
        return
    if args.config is None:
        p.error("--config is required")
    cfg = config_paths(read_config(args.config), args.config.resolve().parent)
    spec = dict(cfg.pop("approach_angle", None) or {})
    if spec.get("target_structure"):
        spec["target_structure"] = str(
            (args.config.resolve().parent / spec["target_structure"]).resolve()
        )
    if args.angle is not None:
        spec.update(enabled=True, interval_deg=args.angle)
    if args.paired:
        spec.update(paired=True)
    if args.ordinary:
        spec.update(enabled=False, paired=False)
    for key in ("backbones", "replicates", "seed"):
        if getattr(args, key) is not None:
            spec[key] = getattr(args, key)
    if args.target_structure:
        spec["target_structure"] = str(args.target_structure.resolve())
    if args.layout_only:
        jobs = split_jobs(
            args.output,
            spec.get("backbones", cfg["num_backbones"]),
            spec.get("replicates", 1),
            spec.get("paired", False),
            spec.get("enabled", False),
            spec.get("seed", 0),
        )
        print(len(jobs))
        return
    execute(
        cfg,
        args.output,
        spec,
        model=args.model_config,
        weights=args.weights,
        preview=args.preview,
        prepare_only=args.stage == "prepare",
    )


if __name__ == "__main__":
    main()
