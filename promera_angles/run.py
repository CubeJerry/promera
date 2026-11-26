"""Launch versioned targeted-angle studies and report every requested parent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf

from experiments.pose_guidance.pose_geometry import digest

from .analysis import analyse, write_json
from .gfp_pair_ablation import GFP_PAIR_COUNTS, validate_gfp_plan
from .settings import SURFACE_DEFINITION, study_arms, validate_settings
from .structures import resolve_target, scaffold_from_cif


def study_settings(cfg):
    """Read explicit namespaces without migrating historical angle values."""
    legacy, surface = cfg.get("approach_angle"), cfg.get("surface_elevation")
    for name, value in (("approach_angle", legacy), ("surface_elevation", surface)):
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")
        if value is not None and type(value.get("enabled", False)) is not bool:
            raise ValueError(f"{name}.enabled must be boolean")
    legacy, surface = dict(legacy or {}), dict(surface or {})
    if legacy.get("enabled") and surface.get("enabled"):
        raise ValueError("approach_angle and surface_elevation cannot both be enabled")
    if surface and not legacy.get("enabled"):
        if surface.get("definition") not in (None, SURFACE_DEFINITION):
            raise ValueError(
                "surface_elevation requires the surface_elevation_v1 definition"
            )
        surface["definition"] = SURFACE_DEFINITION
        return surface
    if (
        legacy.get("definition") == SURFACE_DEFINITION
        or "elevation_interval_deg" in legacy
    ):
        raise ValueError(
            "Use the surface_elevation block for elevation; approach_angle is legacy tilt"
        )
    return legacy


def refold_budget(cfg):
    """Record the native five-sample budget; inverse_folder=none skips refolding."""
    inverse = cfg.get("inverse_folder") or {"type": "none"}
    if not isinstance(inverse, dict):
        raise ValueError("inverse_folder must be a mapping")
    if inverse.get("type", "none") == "none":
        return dict(
            sequences_per_parent=0, refolds_per_sequence=5, refolds_per_parent=0
        )
    count = inverse.get("num_seqs", 1)
    if type(count) is not int or count <= 0:
        raise ValueError("inverse_folder.num_seqs must be a positive integer")
    return dict(
        sequences_per_parent=count, refolds_per_sequence=5, refolds_per_parent=5 * count
    )


def cli_settings(cfg, args):
    """Apply explicit CLI overrides and reject ambiguous old/new combinations."""
    spec = study_settings(cfg)
    surface_cli = (
        args.elevation is not None
        or args.three_arms
        or getattr(args, "gfp_pair_ablation", False)
        or args.cdr_lateral_offset is not None
    )
    if args.angle is not None and (surface_cli or args.legacy_tilt is not None):
        raise ValueError(
            "--angle is historical placement-ray tilt; do not combine it with new guide flags"
        )
    if args.angle is not None and spec.get("definition") == SURFACE_DEFINITION:
        raise ValueError("--angle cannot override a surface_elevation configuration")
    if surface_cli and (
        "interval_deg" in spec or spec.get("definition") == "legacy_placement_ray_v1"
    ):
        raise ValueError(
            "A saved legacy interval cannot become elevation; remove the approach_angle block "
            "and specify --elevation and --legacy-tilt independently"
        )
    if surface_cli:
        spec.update(definition=SURFACE_DEFINITION, enabled=True)
    if args.elevation is not None:
        spec["elevation_interval_deg"] = args.elevation
    if args.legacy_tilt is not None:
        if spec.get("definition") != SURFACE_DEFINITION and "interval_deg" in spec:
            raise ValueError("Do not supply two legacy tilt intervals")
        spec["legacy_tilt_interval_deg"] = args.legacy_tilt
        spec["enabled"] = True
    if args.angle is not None:
        spec.update(enabled=True, interval_deg=args.angle)
    if getattr(args, "gfp_pair_ablation", False):
        spec.update(gfp_pair_ablation=True, three_arms=False, paired=False)
    elif args.three_arms:
        spec.update(three_arms=True, paired=False)
    elif args.paired:
        spec.update(paired=True, three_arms=False)
    elif args.ordinary:
        spec.update(enabled=False, paired=False, three_arms=False)
    for key in ("backbones", "replicates", "seed"):
        if getattr(args, key) is not None:
            spec[key] = getattr(args, key)
    if args.standoff is not None:
        spec["standoff_A"] = args.standoff
    if args.cdr_lateral_offset is not None:
        spec["cdr_lateral_offset_A"] = args.cdr_lateral_offset
    if args.target_structure:
        spec["target_structure"] = str(args.target_structure.resolve())
    elif spec.get("target_structure"):
        target = Path(os.path.expandvars(str(spec["target_structure"]))).expanduser()
        spec["target_structure"] = str(
            (args.config.resolve().parent / target).resolve()
        )
    return validate_settings(spec)


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


def split_jobs(root, count, reps, paired, enabled, seed, *, arms=None):
    """Allocate an equal total budget and matching replicate seeds to every arm."""
    if type(count) is not int or type(reps) is not int or not 1 <= reps <= count:
        raise ValueError(
            "need integer budgets with 1 <= replicates <= backbones per condition"
        )
    if type(seed) is not int or not 0 <= seed < 2**32 - reps:
        raise ValueError("seed must fit an unsigned 32-bit stream for each replicate")
    selected = (
        (["ordinary", "guided"] if paired else ["guided" if enabled else "ordinary"])
        if arms is None
        else list(arms)
    )
    known_arms = {"ordinary", "guided", "legacy_tilt", "surface_elevation"} | set(
        GFP_PAIR_COUNTS
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - known_arms
    ):
        raise ValueError("provide distinct recognized study arms")
    root = Path(root)
    jobs, offset = [], 0
    for rep in range(reps):
        size = count // reps + (rep < count % reps)
        for arm in selected:
            output = (
                root / arm / f"rep-{rep + 1:03}" if paired or arms is not None else root
            )
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
    if set(selected) == set(GFP_PAIR_COUNTS):
        for job in jobs:
            job["gfp_pair_arm"] = job["arm"]
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
    """Run one requested arm/replicate and persist failures including stale inputs."""
    root = Path(root)
    if type(index) is not int or not 0 <= index < len(manifest["jobs"]):
        raise ValueError("job index is outside the prepared study")
    meta = root / "approach_angle"
    job = manifest["jobs"][index]
    tag = f"{job['arm']}-{job['replicate']}"
    receipt = meta / f"{tag}-status.json"
    write_json(receipt, {"status": "running"})
    try:
        cfg = deepcopy(manifest["base_config"])
        path = meta / "plan.json"
        if digest(path) != manifest["plan_sha256"]:
            raise ValueError("prepared plan digest changed before sampling")
        plan = json.loads(path.read_text())
        for field, sha in (
            ("input", "schema_sha256"),
            ("structure", "structure_sha256"),
        ):
            if digest(plan[field]) != plan[sha]:
                raise ValueError(f"prepared {field} changed before sampling")
        arm = job["arm"]
        gfp_ablation = manifest["settings"].get("gfp_pair_ablation", False)
        if gfp_ablation:
            validate_gfp_plan(plan)
            if job.get("gfp_pair_arm") != arm or arm not in GFP_PAIR_COUNTS:
                raise ValueError("invalid GFP pair ablation arm")
        elif arm.startswith("gfp_pairs_"):
            raise ValueError("GFP arms require the GFP-only ablation study")
        if arm == "surface_elevation" or arm.startswith("gfp_pairs_"):
            definition, interval = (
                "surface_elevation",
                manifest["elevation_interval_deg"],
            )
        elif arm in ("legacy_tilt", "guided"):
            definition = "legacy_placement_ray"
            interval = (
                manifest["legacy_tilt_interval_deg"]
                if arm == "legacy_tilt"
                else manifest["interval_deg"]
            )
        else:
            definition, interval = "ordinary", None
        trial = dict(
            plan=str(path),
            sha256=manifest["plan_sha256"],
            seed=job["seed"],
            start=job["start"],
            interval=interval,
            guidance_definition=definition,
            standoff=manifest["settings"].get("standoff_A", 6.0),
            cdr_lateral_offset=manifest["settings"].get(
                "cdr_lateral_offset_A", [0.0, 0.0]
            ),
        )
        if gfp_ablation:
            trial["gfp_pair_arm"] = arm
        config = dict(
            cfg,
            output=job["output"],
            num_backbones=job["count"],
            skip_existing=False,
            angle_trial=trial,
        )
        if config.get("nominee"):
            config["nominee"] = dict(
                config["nominee"], summary_csv=str(meta / f"{tag}-summary.csv")
            )
        infer(
            "promera_angles.task.TargetedDesign",
            config,
            meta / f"{tag}.json",
            job["seed"],
            manifest.get("model_config"),
            manifest.get("weights"),
        )
    except Exception as exc:
        write_json(receipt, {"status": "failed", "reason": str(exc)})
        raise
    write_json(receipt, {"status": "completed"})


def finish(root, manifest, plan, *, strict=True):
    """Write the report before raising for missing or failed requested work."""
    root = Path(root)
    # This positional argument is always historical tilt. New analysis reads the
    # explicitly versioned elevation fields from plan instead.
    result = analyse(
        root / "analysis", manifest["jobs"], plan, manifest.get("interval_deg")
    )
    statuses = {}
    for job in manifest["jobs"]:
        tag = f"{job['arm']}-{job['replicate']}"
        receipt = root / "approach_angle" / f"{tag}-status.json"
        try:
            statuses[tag] = (
                json.loads(receipt.read_text())
                if receipt.exists()
                else {"status": "missing"}
            )
        except (OSError, ValueError) as exc:
            statuses[tag] = {
                "status": "failed",
                "reason": f"unreadable job receipt: {exc}",
            }
    result["job_status"] = statuses
    if "preparation_status" in manifest:
        result["preparation_status"] = manifest["preparation_status"]
    write_json(root / "analysis" / "summary.json", result)
    print(f"Analysis: {root / 'analysis' / 'summary.json'}", flush=True)
    if not strict:
        return result
    if any(value.get("status") != "completed" for value in statuses.values()):
        raise RuntimeError(
            "one or more design jobs did not complete; see analysis/summary.json"
        )
    if any(
        arm["generated"] + arm.get("rejected_guides", 0) != arm["expected"]
        for arm in result["arms"].values()
    ):
        raise RuntimeError("incomplete backbone generation; see analysis/backbones.csv")
    return result


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
    """Prepare a versioned study, execute its jobs, and account for every parent."""
    cfg = deepcopy(cfg)
    # Validate conflicting native blocks even for direct Python callers.
    study_settings(cfg)
    cfg.pop("approach_angle", None)
    cfg.pop("surface_elevation", None)
    if cfg["binder"]["type"] != "vhh":
        raise ValueError("VHH body-axis measurements require a tagged VHH framework")
    spec = validate_settings(spec)
    surface = spec.get("definition") == SURFACE_DEFINITION
    enabled, paired = spec.get("enabled", False), spec.get("paired", False)
    interval = spec.get("interval_deg")
    root = Path(root).resolve()
    jobs = split_jobs(
        root,
        spec.get("backbones", cfg["num_backbones"]),
        spec.get("replicates", 1),
        paired,
        enabled,
        spec.get("seed", 0),
        **({"arms": study_arms(spec)} if surface else {}),
    )
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "approach_angle"
    meta.mkdir(exist_ok=False)
    if model_settings is not None:
        model = meta / "model-config.json"
        write_json(model, model_settings)
    try:
        # Keep the old positional API intact for legacy callers and reproducibility.
        plan = (
            resolve_target(cfg, spec.get("target_structure"), surface_elevation=True)
            if surface
            else resolve_target(cfg, spec.get("target_structure"))
        )
    except (KeyError, ValueError, OSError) as exc:
        write_json(
            meta / "preview.json",
            dict(
                status="unavailable",
                definition=SURFACE_DEFINITION if surface else "legacy_placement_ray_v1",
                reason=str(exc),
            ),
        )
        raise
    if spec.get("gfp_pair_ablation"):
        validate_gfp_plan(plan)
        plan["gfp_pair_ablation"] = True
    if surface:
        plan.update(
            version=2,
            experiment_definition=SURFACE_DEFINITION,
            elevation_interval_deg=spec.get("elevation_interval_deg"),
            legacy_tilt_interval_deg=spec.get("legacy_tilt_interval_deg"),
            **refold_budget(cfg),
        )
    collision = [
        Path(job["output"]) / plan["target"]
        for job in jobs
        if (Path(job["output"]) / plan["target"]).exists()
    ]
    if collision:
        raise FileExistsError(
            f"existing target outputs at {collision[0]}; select a fresh output directory"
        )
    preview_data = dict(
        version=2 if surface else 1,
        status="valid",
        definition=SURFACE_DEFINITION if surface else "legacy_placement_ray_v1",
        kind=(
            "surface elevation above prospective patch"
            if surface
            else "legacy placement-ray tilt"
        ),
        units="angstrom, degrees",
        standoff_A=spec["standoff_A"],
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
    if surface:
        preview_data.update(
            elevation_interval_deg=spec.get("elevation_interval_deg"),
            legacy_tilt_interval_deg=spec.get("legacy_tilt_interval_deg"),
            cdr_lateral_offset_A=spec["cdr_lateral_offset_A"],
            arms=study_arms(spec),
            axis="CDR centroid to framework centroid (C to F)",
            legacy_patch_frame=plan.get("legacy_frame"),
            legacy_frame_error=plan.get("legacy_frame_error"),
        )
    else:
        preview_data["interval_deg"] = interval
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
        version=2 if surface else 1,
        experiment_definition=(
            SURFACE_DEFINITION if surface else "legacy_placement_ray_v1"
        ),
        jobs=jobs,
        settings=spec,
        base_config=cfg,
        model_config=str(model) if model else None,
        weights=str(weights or os.environ.get("PROMERA_WEIGHTS", "")),
        image=os.environ.get("POSE_RUNTIME_IMAGE"),
        commit=os.environ.get("POSE_EXPERIMENT_COMMIT"),
    )
    if surface:
        manifest.update(
            elevation_interval_deg=spec.get("elevation_interval_deg"),
            legacy_tilt_interval_deg=spec.get("legacy_tilt_interval_deg"),
            **refold_budget(cfg),
        )
    else:
        manifest["interval_deg"] = interval
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
        manifest["preparation_status"] = {"status": "completed"}
        write_json(meta / "manifest.json", manifest)
    except Exception as exc:
        manifest["preparation_status"] = {"status": "failed", "reason": str(exc)}
        write_json(meta / "manifest.json", manifest)
        if surface:
            finish(root, manifest, plan, strict=False)
        elif not prepare_only:
            analyse(root / "analysis", jobs, plan, interval)
        raise
    if prepare_only:
        return
    if not surface:
        # Preserve early-failure behavior for historical experiments.
        try:
            for index in range(len(jobs)):
                run_job(root, manifest, index)
        except Exception:
            analyse(root / "analysis", jobs, plan, interval)
            raise
        finish(root, manifest, plan)
        return
    failures = []
    for index in range(len(jobs)):
        try:
            run_job(root, manifest, index)
        except Exception as exc:
            failures.append(exc)
    # One failed arm must not prevent the remaining matched jobs from running.
    # Finish writes all counts before propagating a nonzero outcome to Slurm.
    try:
        finish(root, manifest, plan)
    except Exception:
        if failures:
            raise RuntimeError(
                f"{len(failures)} study job(s) failed; first failure: {failures[0]}; "
                "see analysis/summary.json"
            ) from failures[0]
        raise
    if failures:
        raise RuntimeError(
            f"{len(failures)} study job(s) failed: {failures[0]}"
        ) from failures[0]


def main(argv=None):
    """CLI with explicit elevation and legacy-tilt namespaces."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--auto-msa", action="store_true", help=argparse.SUPPRESS)
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
        help="historical placement-ray tilt in degrees; never surface elevation",
    )
    p.add_argument(
        "--elevation",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="signed C-to-F elevation above the prospective patch plane, -90 to +90",
    )
    p.add_argument(
        "--legacy-tilt",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="explicit, independent legacy placement-ray tilt interval",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--paired", action="store_true", help="also run a matched ordinary control"
    )
    mode.add_argument(
        "--ordinary", action="store_true", help="measure an ordinary control only"
    )
    mode.add_argument(
        "--three-arms",
        action="store_true",
        help="ordinary, legacy tilt and surface elevation with equal budgets",
    )
    mode.add_argument(
        "--gfp-pair-ablation",
        action="store_true",
        help="GFP-only benchmark: matched ordinary, 4, 8, 12 and 16 pair arms",
    )
    p.add_argument(
        "--backbones", type=int, help="total per arm, divided across replicates"
    )
    p.add_argument("--replicates", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--standoff", type=float, help="CDR-end standoff from the patch in angstrom"
    )
    p.add_argument(
        "--cdr-lateral-offset",
        nargs=2,
        type=float,
        metavar=("DX", "DY"),
        help="independent CDR-end displacement along patch tangents, in angstrom",
    )
    p.add_argument(
        "--target-structure",
        type=Path,
        help="geometry-only mmCIF if distinct from target_template",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="write intended geometry with no inference",
    )
    p.add_argument("--weights", type=Path)
    p.add_argument("--model-config", type=Path)
    args = p.parse_args(argv)
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
    try:
        spec = cli_settings(cfg, args)
        surface = spec.get("definition") == SURFACE_DEFINITION
        jobs = split_jobs(
            args.output,
            spec.get("backbones", cfg["num_backbones"]),
            spec.get("replicates", 1),
            spec.get("paired", False),
            spec.get("enabled", False),
            spec.get("seed", 0),
            **({"arms": study_arms(spec)} if surface else {}),
        )
    except ValueError as exc:
        p.error(str(exc))
    if args.layout_only:
        print(len(jobs))
        return
    if args.auto_msa and not args.preview:
        from .cif_input import prepare_msas

        prepare_msas(cfg)
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
