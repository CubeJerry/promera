"""Prepare experiments through NOMINEE's validated saved-setup machinery.

The legacy positional invocation retains its original numbering convention and
placement-ray settings. ``--surface-study`` explicitly selects a new entry point
whose replacement residue selection uses chain-qualified AUTHOR numbering.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .analysis import write_json

DEFAULT_CDR_LENGTHS = {
    "cdrh1_length": "6-8",
    "cdrh2_length": "6-8",
    "cdrh3_length": "8-20",
}


def referenced_paths(
    setup_file: Path, antigen_cif: Path | None = None
) -> dict[str, str]:
    """Resolve paths beside the YAML and account for an explicit antigen override."""
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
    if antigen_cif is not None:
        old_antigen = paths.get("antigen_cif") or paths.get("structure")
        replacement = str(antigen_cif.resolve())
        paths["antigen_cif"] = replacement
        paths.pop("structure", None)
        if not paths.get("template_cif") or paths["template_cif"] == old_antigen:
            paths["template_cif"] = replacement
        paths.pop("promera_input", None)
    return paths


def validate_author_selection(residues: str) -> None:
    """Require explicit chain groups before NOMINEE validates the residue IDs."""
    groups = residues.strip().split(";")
    if not groups or any(
        re.fullmatch(r"\s*[^\s,:;]+\s*:\s*\S.*", group) is None for group in groups
    ):
        raise ValueError(
            "Use chain-qualified AUTHOR residues, for example A:10,12-15;B:7"
        )


def prepare_setups(
    antigen_file: Path,
    nano_file: Path | None,
    residues: str,
    root: Path,
    *,
    surface_study: bool = False,
    defaults: Path | None = None,
    antigen_cif: Path | None = None,
    sequences_per_parent: int = 1,
    inverse_folder: str = "solublempnn",
    recycling_steps: int = 5,
    diffusion_steps: int = 200,
) -> dict[str, Any]:
    """Build one ordinary target config shared by every arm of a study.

    NOMINEE owns chain selection, author/runtime conversion, exclusions and
    template validation. Saved antigen setups contain no VHH framework or model
    settings. A new surface study may use the shipped VHH framework explicitly;
    legacy setup invocations still require their nanobody setup.
    """
    from utils.nominee_core.promera import (
        PromeraRunConfig,
        build_promera_backend_config,
        write_promera_run_config,
    )
    from utils.nominee_core.promera_residue_mapping import (
        finalise_promera_run_epitope_mapping,
    )
    from utils.nominee_core.promera_setup import (
        load_setup,
        normalise_antigen_setup,
        normalise_nanobody_setup,
        promera_length_spec,
    )

    if surface_study:
        validate_author_selection(residues)
    if nano_file is None and (not surface_study or defaults is None):
        raise ValueError(
            "A nanobody setup or explicit surface-study defaults is required"
        )
    if any(
        type(value) is not int or value <= 0
        for value in (sequences_per_parent, recycling_steps, diffusion_steps)
    ):
        raise ValueError(
            "Sequence, recycling and diffusion counts must be positive integers"
        )
    root = root.resolve()
    inputs = root / "inputs"
    inputs.mkdir()
    raw = load_setup(antigen_file)
    raw.update(referenced_paths(antigen_file, antigen_cif))
    if antigen_cif is not None:
        raw.pop("structure", None)
        raw["promera_input"] = ""
    raw.pop("epitope_residue_map", None)
    raw["epitope_residues"] = residues
    raw["epitope_numbering"] = (
        "author" if surface_study else raw.get("epitope_numbering") or "author"
    )
    antigen = normalise_antigen_setup(raw)
    if nano_file is None:
        assert defaults is not None
        nano_raw = {"framework": load_setup(defaults)["binder"]["framework"]}
    else:
        nano_raw = load_setup(nano_file)
    for key, fallback in DEFAULT_CDR_LENGTHS.items():
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
        run_name="surface-elevation" if surface_study else "approach-angle",
        output_dir=str(root),
        msa_dir=str(inputs / "msa"),
        inverse_folder=inverse_folder,
        num_backbones=1000,
        num_sequences=sequences_per_parent,
        recycling_steps=recycling_steps,
        diffusion_steps=diffusion_steps,
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
    if not surface_study:
        cfg["approach_angle"] = dict(
            enabled=True, paired=True, interval_deg=[0, 20], replicates=10
        )
    for src, name in ((antigen_file, "antigen_setup"), (nano_file, "nanobody_setup")):
        if src is not None:
            shutil.copyfile(src, inputs / (name + src.suffix))
    if surface_study:
        write_json(
            inputs / "setup_provenance.json",
            {
                "antigen_setup": str(antigen_file),
                "antigen_cif_override": str(antigen_cif) if antigen_cif else None,
                "residue_selection": residues,
                "residue_numbering": "author",
                "nanobody_setup": str(nano_file) if nano_file else None,
                "framework_source": str(nano_file or defaults),
                "framework_default_used": nano_file is None,
                "framework": nano["framework"],
                "cdr_lengths": cfg["binder"]["cdr_lengths"],
                "sequences_per_parent": sequences_per_parent,
                "native_refolds_per_sequence": 5,
            },
        )
    write_json(inputs / "targeted.json", cfg)
    print(
        "Target chains:",
        ", ".join(cfg["target_chains"]),
        "| residues:",
        antigen["epitope_residues"],
    )
    print("CDR lengths:", cfg["binder"]["cdr_lengths"])
    if nano_file is None:
        print(
            "Framework: shipped caplacizumab/7eow default; use --nanobody-setup to override"
        )
    return cfg


def main() -> None:
    """Support the legacy setup API and explicit surface-study preparation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("antigen", type=Path)
    p.add_argument("nanobody", type=Path)
    p.add_argument("residues")
    p.add_argument("output", type=Path)
    p.add_argument("--list-inputs", action="store_true")
    p.add_argument("--surface-study", action="store_true")
    p.add_argument("--defaults", type=Path)
    p.add_argument("--antigen-cif", type=Path)
    p.add_argument("--sequences-per-parent", type=int, default=1)
    p.add_argument("--inverse-folder", default="solublempnn")
    p.add_argument("--recycling-steps", type=int, default=5)
    p.add_argument("--diffusion-steps", type=int, default=200)
    args = p.parse_args()
    if args.list_inputs:
        for path in dict.fromkeys(
            referenced_paths(args.antigen, args.antigen_cif).values()
        ):
            print(path)
    else:
        prepare_setups(
            args.antigen,
            None if str(args.nanobody) == "-" else args.nanobody,
            args.residues,
            args.output,
            surface_study=args.surface_study,
            defaults=args.defaults,
            antigen_cif=args.antigen_cif,
            sequences_per_parent=args.sequences_per_parent,
            inverse_folder=args.inverse_folder,
            recycling_steps=args.recycling_steps,
            diffusion_steps=args.diffusion_steps,
        )


if __name__ == "__main__":
    main()
