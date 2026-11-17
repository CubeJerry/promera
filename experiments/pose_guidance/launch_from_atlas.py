#!/usr/bin/env python3
"""Launch a source-verified Pose Atlas angular-range smoke on current NOMINEE.

Host-only standard library. Scientific featurisation/inference stays inside the
managed patched Promera image. Never modify the original completed scan.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from nominee_paths import installation_parent

DEFAULT_POSE = "B004__G047__5_389"
POSE_PATTERN = re.compile(r"^([A-Za-z0-9._-]+)__([A-Za-z0-9._-]+)__([0-9]+_[0-9]+)$")


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _checked(path: str, expected: str, label: str) -> Path:
    if not path or not expected:
        raise ValueError(f"Missing catalogued {label} path or checksum")
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise ValueError(f"Missing catalogued {label}: {value}")
    if _sha256(value) != expected:
        raise ValueError(f"STALE {label}: SHA-256 differs from Pose Atlas catalog: {value}")
    return value


def select_source(run: Path, pose_id: str = DEFAULT_POSE) -> dict:
    """Select exactly one CDR-driven canonical *backbone*, not viewer refold."""
    run = run.expanduser().resolve()
    if not run.is_dir():
        raise ValueError(f"Completed run directory not found: {run}")
    match = POSE_PATTERN.fullmatch(pose_id)
    if not match:
        raise ValueError(f"Use full Pose Atlas seed ID BASIN__GROUP__BATCH_CANDIDATE: {pose_id}")
    basin, group, candidate = match.groups()
    catalogs = [run / "analysis/epitope_scan/pose_catalog.json",
                run / "epitope_scan/pose_catalog.json"]
    found = [p for p in catalogs if p.is_file()]
    if len(found) != 1:
        raise ValueError("Expected one Pose Atlas catalog. Checked: " + ", ".join(map(str, catalogs)))
    catalog = _load_json(found[0])
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("records"), list):
        raise ValueError("Unsupported Pose Atlas catalog format")
    selections = [x for x in catalog["records"]
                  if isinstance(x, dict) and x.get("pose_seed_id") == pose_id]
    if len(selections) != 1:
        raise ValueError(f"Expected exactly one Pose Atlas seed {pose_id}; found {len(selections)}")
    row = selections[0]
    for key, expected in (
        ("basin_id", basin), ("geometry_group_id", group),
        ("source_candidate_id", candidate), ("classification", "cdr_driven"),
    ):
        if str(row.get(key) or "") != expected:
            raise ValueError(f"Pose Atlas seed {pose_id}: {key} mismatch ({row.get(key)!r})")

    seed_manifest_path = _checked(str(row.get("pose_seed_manifest_path") or ""),
                                  str(row.get("pose_seed_manifest_sha256") or ""),
                                  "pose seed manifest")
    seed = _load_json(seed_manifest_path)
    if seed.get("pose_seed_id") != pose_id:
        raise ValueError("Pose seed manifest identity differs from catalog")
    canonical = _checked(str(seed.get("canonical_backbone_path") or ""),
                         str(seed.get("canonical_backbone_sha256") or ""),
                         "canonical source backbone")
    if _sha256(canonical) != str(row.get("backbone_sha256") or ""):
        raise ValueError("Canonical backbone differs from the catalogued original backbone")
    original = _checked(str(row.get("backbone_path") or ""),
                        str(row.get("backbone_sha256") or ""),
                        "original source backbone")
    if not canonical.is_relative_to(run) or not seed_manifest_path.is_relative_to(run):
        raise ValueError("Selected canonical pose does not belong to the supplied run")

    # NOMINEE source IDs use SLURM array number as the first numeric component.
    # Do not silently select promera_batch_1.yaml for a seed from batch 5.
    task_num = int(candidate.split("_", 1)[0])
    config = run / "promera" / f"promera_batch_{task_num}.yaml"
    if not config.is_file():
        raise ValueError(
            f"Missing ORIGINAL Promera backend task config for {candidate}: {config}. "
            "Do not substitute the flat promera_run.json or a different batch."
        )
    backend = _load_json(config)  # The rendered .yaml contains JSON.
    binder = backend.get("binder")
    if not isinstance(binder, dict):
        raise ValueError(f"Backend task is not a Promera VHH generation config: {config}")
    if binder.get("type") != "vhh" or binder.get("framework") != row.get("framework"):
        raise ValueError("The generated batch config has a different VHH framework than Pose Atlas")
    if str(binder.get("chain")) != str(row.get("binder_chain")):
        raise ValueError("The generated batch config binder chain differs from Pose Atlas")
    target_names = backend.get("target_chains")
    if isinstance(target_names, str):
        target_names = [x.strip() for x in target_names.split(",") if x.strip()]
    if target_names:
        catalog_chains = [x for x in str(row.get("antigen_chains") or "").split(",") if x]
        if set(target_names) != set(catalog_chains):
            raise ValueError(f"Target chain mismatch: backend={target_names}, catalog={catalog_chains}")
    if not backend.get("input"):
        raise ValueError(f"Backend task input schema is unspecified: {config}")
    return {
        "run": run, "pose_id": pose_id, "source_record": row,
        "catalog": found[0], "canonical_backbone": canonical,
        "original_backbone": original, "seed_manifest": seed_manifest_path,
        "base_config": config, "backend": backend, "batch_number": task_num,
    }


def _contains(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def extra_binds(selection: dict, parent: Path, work: Path, prior: str = "") -> str:
    """Expose external source locations without mounting arbitrary sibling trees."""
    extra = [prior] if prior else []
    sources = [selection["run"], selection["base_config"].parent,
               selection["canonical_backbone"].parent]
    cfg = selection["backend"]
    paths = [cfg.get("input"), cfg.get("msa_dir"),
             (cfg.get("target_template") or {}).get("path")]
    for value in paths:
        if not value or str(value).startswith("/opt/nominee/"):
            continue
        source = Path(str(value)).expanduser().resolve()
        sources.append(source if source.is_dir() else source.parent)
    included = set()
    for item in sources:
        item = item.resolve()
        if _contains(item, parent) or _contains(item, work) or item in included:
            continue
        if not item.is_dir():
            raise ValueError(f"Required external source directory does not exist: {item}")
        if any(ch in str(item) for ch in ",:\n"):
            raise ValueError(f"Cannot create safe Apptainer bind for {item}")
        included.add(item)
        extra.append(f"{item}:{item}:ro")
    return ",".join(extra)


def launch(args: argparse.Namespace) -> None:
    selection = select_source(Path(args.run), args.pose_id)
    repo = Path(__file__).resolve().parents[1]
    here = Path(__file__).resolve().parent
    config = Path(args.nominee_config).expanduser().resolve()
    if not config.is_file():
        raise ValueError(f"Missing configured NOMINEE site file: {config}")
    parent = installation_parent(config)
    work = Path(args.work).expanduser().resolve() if args.work else parent / "pose-guidance-work"
    if _contains(work, selection["run"]):
        raise ValueError("Experiment output must not be inside the completed source run")
    work.mkdir(parents=True, exist_ok=True)
    interval = args.angle_range
    match = re.fullmatch(r"(\d{1,3}(?:\.\d+)?):(\d{1,3}(?:\.\d+)?)", interval)
    if not match or not 0 <= float(match[1]) <= float(match[2]) <= 180:
        raise ValueError("Set a degree interval like --angle-range 15:35 (0 <= min <= max <= 180)")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = re.sub(r"[^A-Za-z0-9-]+", "-", interval)
    name = f"{selection['pose_id']}-angle-{tag}-smoke-{stamp}"
    experiment = work / name
    if experiment.exists():
        raise ValueError(f"Experiment directory already exists: {experiment}")
    env = dict(os.environ)
    env["NOMINEE_CONFIG"] = str(config)
    env["POSE_SOURCE"] = str(repo)
    env["POSE_WORK"] = str(work)
    env["POSE_EXPERIMENT"] = str(experiment)
    env["POSE_EXTRA_BINDS"] = extra_binds(
        selection, parent, work, env.get("POSE_EXTRA_BINDS", "")
    )
    print("Verified Pose Atlas seed:", selection["pose_id"], flush=True)
    print("Exact canonical backbone:", selection["canonical_backbone"], flush=True)
    print("Backbone SHA256:", _sha256(selection["canonical_backbone"]), flush=True)
    print("Matching batch configuration:", selection["base_config"], flush=True)
    print("Source-relative angle interval:", interval, "degrees", flush=True)
    print("New experiment directory:", experiment, flush=True)
    if args.dry_run:
        print("DRY RUN: no experiment prepared and no SLURM jobs submitted.", flush=True)
        return
    launcher = here / "container.sh"
    subprocess.run([
        "bash", str(launcher), "prepare",
        "--base-config", str(selection["base_config"]),
        "--pose", str(selection["canonical_backbone"]),
        "--out", str(experiment),
        "--profile", "smoke",
        "--angle-range", interval,
    ], check=True, env=env)
    (experiment / "source_atlas_selection.json").write_text(json.dumps({
        "pose_id": selection["pose_id"],
        "source_run": str(selection["run"]),
        "pose_catalog": str(selection["catalog"]),
        "pose_seed_manifest": str(selection["seed_manifest"]),
        "canonical_backbone": str(selection["canonical_backbone"]),
        "canonical_backbone_sha256": _sha256(selection["canonical_backbone"]),
        "source_promera_backend_config": str(selection["base_config"]),
        "source_promera_backend_config_sha256": _sha256(selection["base_config"]),
        "approach_angle_range_degrees": list(map(float, interval.split(":"))),
    }, indent=2) + "\n")
    for index in range(3):
        print(f"Preflight {index}/2:", flush=True)
        subprocess.run(["bash", str(launcher), "preflight",
                        "--experiment", str(experiment), "--job", str(index)],
                       check=True, env=env)
    print("All three CPU preflights passed.", flush=True)
    if args.no_submit:
        print("No job submitted (--no-submit).", flush=True)
        print(f"To submit: POSE_SOURCE={repo} POSE_WORK={work} "
              f"POSE_EXPERIMENT={experiment} sbatch --array=0-2%1 {here / 'run.slurm'}")
        return
    if not shutil_which_sbatch():
        raise ValueError("sbatch not found on PATH; run from the HPC login node")
    subprocess.run(["sbatch", "--array=0-2%1", str(here / "run.slurm")],
                   check=True, env=env, cwd=str(work))
    print("Submitted: 3 arms x 2 backbones = 6 backbones, up to 30 refolds.", flush=True)
    print(f"After completion: bash {launcher} analyse --experiment {experiment}", flush=True)


def shutil_which_sbatch() -> bool:
    from shutil import which
    return which("sbatch") is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed NOMINEE run root containing analysis/epitope_scan")
    parser.add_argument("--pose-id", default=DEFAULT_POSE)
    parser.add_argument("--angle-range", default="15:35", help="Deviation from selected VHH approach (degrees)")
    parser.add_argument("--nominee-config", default=os.environ.get(
        "NOMINEE_CONFIG", "/vast/home/users/chen.je/dev/config.yaml"))
    parser.add_argument("--work", default=os.environ.get("POSE_WORK"),
                        help="Experiment output directory (default: <installation_parent>/pose-guidance-work)")
    parser.add_argument("--dry-run", action="store_true", help="Verify sources without starting anything")
    parser.add_argument("--no-submit", action="store_true", help="Prepare and preflight but do not submit")
    args = parser.parse_args()
    try:
        launch(args)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
