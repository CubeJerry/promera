"""Prepare one paired study through NOMINEE's ordinary saved-setup machinery."""

import argparse
import json
import os
import shutil
from pathlib import Path

from .analysis import write_json


def referenced_paths(setup_file):
    from utils.nominee_core.promera_setup import load_setup

    raw = load_setup(setup_file)
    paths = {}
    for field in ("antigen_cif", "structure", "template_cif", "promera_input"):
        if raw.get(field):
            value = os.path.expandvars(str(raw[field]))
            if value.startswith("~/"):
                value = str(
                    Path(os.environ.get("ANGLE_SUBMIT_HOME", Path.home())) / value[2:]
                )
            paths[field] = str((setup_file.parent / value).resolve())
    return paths


def prepare_setups(antigen_file, nano_file, residues, root):
    from utils.nominee_core.promera import (
        PromeraRunConfig,
        build_promera_backend_config,
        write_promera_run_config,
    )
    from utils.nominee_core.promera_setup import (
        load_setup,
        normalise_antigen_setup,
        normalise_nanobody_setup,
        promera_length_spec,
    )
    from utils.nominee_core.promera_residue_mapping import (
        finalise_promera_run_epitope_mapping,
    )

    inputs = root / "inputs"
    inputs.mkdir()
    raw = load_setup(antigen_file)
    raw.update(referenced_paths(antigen_file))
    raw.pop("epitope_residue_map", None)
    raw["epitope_residues"] = residues
    raw["epitope_numbering"] = raw.get("epitope_numbering") or "author"
    antigen = normalise_antigen_setup(raw)
    nano_raw = load_setup(nano_file)
    for key, fallback in (
        ("cdrh1_length", "6-8"),
        ("cdrh2_length", "6-8"),
        ("cdrh3_length", "8-20"),
    ):
        nano_raw[key] = nano_raw.get(key) or fallback
    nano = normalise_nanobody_setup(nano_raw)
    config = PromeraRunConfig(
        **{
            k: v
            for k, v in antigen.items()
            if k not in ("schema_version", "kind", "antigen_symmetry_validation")
        },
        **{
            k: promera_length_spec(nano[k])
            for k in ("cdrh1_length", "cdrh2_length", "cdrh3_length")
        },
        framework=nano["framework"],
        binder_chain=nano["binder_chain"],
        run_name="approach-angle",
        output_dir=str(root),
        msa_dir=str(inputs / "msa"),
        inverse_folder="solublempnn",
        num_backbones=1000,
    )
    path = write_promera_run_config(config, inputs / "nominee_run.json")
    finalise_promera_run_epitope_mapping(path, numbering=config.epitope_numbering)
    run = json.loads(path.read_text())
    cfg = build_promera_backend_config(
        run,
        batch_id=1,
        batch_backbones=1000,
        output_dir=root,
        summary_csv=root / "summary.csv",
    )
    cfg["input"] = run["promera_runtime_input"]
    # Keep the referenced template readable by later jobs regardless of its source mount.
    template = Path(cfg["target_template"]["path"])
    owned = inputs / "angle-template.cif"
    shutil.copyfile(template, owned)
    cfg["target_template"]["path"] = str(owned)
    cfg["approach_angle"] = dict(
        enabled=True, paired=True, interval_deg=[0, 20], replicates=10
    )
    for src, name in ((antigen_file, "antigen_setup"), (nano_file, "nanobody_setup")):
        shutil.copyfile(src, inputs / (name + src.suffix))
    write_json(inputs / "targeted.json", cfg)
    print(
        "Target chains:",
        ", ".join(cfg["target_chains"]),
        "| residues:",
        antigen["epitope_residues"],
    )
    print("CDR lengths:", cfg["binder"]["cdr_lengths"])
    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("antigen", type=Path)
    p.add_argument("nanobody", type=Path)
    p.add_argument("residues")
    p.add_argument("output", type=Path)
    p.add_argument("--list-inputs", action="store_true")
    args = p.parse_args()
    if args.list_inputs:
        for path in dict.fromkeys(referenced_paths(args.antigen).values()):
            print(path)
    else:
        prepare_setups(args.antigen, args.nanobody, args.residues, args.output)
