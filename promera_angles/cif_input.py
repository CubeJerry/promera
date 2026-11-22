"""Prepare an ordinary targeted task from an antigen CIF and author residues."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.pose_guidance.pose_geometry import chain_sequence
from .analysis import write_json
from .structures import atom_table


def prepare(cif, residues, root, defaults, chains=None):
    from tinyprot.structure import Structure
    from utils.nominee_core.antigen import choose_binder_chain, residue_map
    from .run import read_config

    inputs = root / "inputs"
    inputs.mkdir()
    template = inputs / "antigen.cif"
    shutil.copyfile(cif, template)
    structure = Structure.from_mmcif(str(template))
    proteins = {
        str(k): v for k, v in structure.chains.items() if "polypeptide" in v.type
    }
    ca = atom_table(template)
    ca = ca[(ca.label_atom_id == "CA") & ca.label_asym_id.isin(proteins)].copy()
    ca["author"] = (
        ca.auth_asym_id.astype(str) if "auth_asym_id" in ca else ca.label_asym_id
    )
    aliases = ca[["author", "label_asym_id"]].drop_duplicates()
    author_ids = list(aliases.author.unique()) if chains is None else chains.split(",")
    if not author_ids or len(set(author_ids)) != len(author_ids):
        raise ValueError("expected distinct protein chain IDs")
    mapping = {}
    for author in author_ids:
        labels = aliases.loc[aliases.author == author, "label_asym_id"].tolist()
        if len(labels) != 1:
            raise ValueError(
                f"author chain {author!r} has {len(labels)} matching protein chains"
            )
        mapping[author] = labels[0]
    if len(mapping) > 1 and any(
        ":" not in group.split(",")[0] for group in residues.split(";")
    ):
        raise ValueError(
            "qualify the residues with an author chain (available: "
            + ", ".join(mapping)
            + ")"
        )
    selection = residue_map(residues, author_ids[0], author_ids)
    if not selection:
        raise ValueError("empty epitope selection")
    schema, runtime, correspondences = {}, {}, {}
    for author, label in mapping.items():
        chain = proteins[label]
        lookup = {int(n): i for i, n in enumerate(chain.ridx, 1)}
        correspondences[label] = {v: k for k, v in lookup.items()}
        schema[label] = dict(
            type="protein", sequence=chain_sequence(chain), use_msa=True
        )
        for number in selection.get(author, []):
            rows = ca[
                (ca.author == author) & (ca.auth_seq_id.astype(str) == str(number))
            ]
            if len(rows) != 1:
                raise ValueError(
                    f"{author}:{number} has {len(rows)} resolved Cα matches; check author numbering and insertion codes"
                )
            runtime.setdefault(label, []).append(lookup[int(rows.iloc[0].pos)])
    target_dir = inputs / "schemas"
    target_dir.mkdir()
    source = target_dir / "target.json"
    write_json(source, schema)
    cfg = read_config(defaults)
    labels = list(schema)
    cfg["binder"].update(
        chain=choose_binder_chain(labels),
        cdr_lengths={"cdrh1": [6, 8], "cdrh2": [6, 8], "cdrh3": [8, 20]},
    )
    cfg.update(
        input=str(source),
        msa_dir=str(inputs / "msa"),
        num_backbones=1000,
        target_chains=labels,
        epitope_chain=labels[0],
        epitope_residues=[],
        epitope_residue_map=runtime,
        target_template=dict(
            path=str(template),
            chain=labels[0],
            chains=labels,
            runtime_residue_map=correspondences,
        ),
        approach_angle=dict(
            enabled=True, paired=True, interval_deg=[0, 20], replicates=10
        ),
    )
    write_json(inputs / "targeted.json", cfg)
    return cfg


def prepare_msas(cfg):
    from tinyprot.msa import msa_exists_in_dir

    dest = Path(cfg["msa_dir"])
    dest.mkdir(exist_ok=True)
    schema = Path(cfg["input"])
    seqs = {v["sequence"] for v in json.loads(schema.read_text()).values()}
    caches = [
        Path("/opt/nominee/cache/msa"),
        Path(os.environ["TINYPROT_CACHE"]) / "msa",
    ]
    for seq in seqs:
        for cache in caches:
            hit = msa_exists_in_dir(str(cache), seq)
            if hit:
                shutil.copyfile(hit, dest / Path(hit).name)
                break
    missing = sorted(s for s in seqs if not msa_exists_in_dir(str(dest), s))
    if missing:
        request = dest.parent / "missing_msa.fasta"
        request.write_text(
            "".join(f">target_{i}\n{seq}\n" for i, seq in enumerate(missing))
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tinyprot.mmseqs2",
                "--url",
                "https://api.colabfold.com",
                "--input",
                str(request),
                "--msa_dir",
                str(dest),
            ],
            check=True,
        )
    if any(not msa_exists_in_dir(str(dest), s) for s in seqs):
        raise RuntimeError("target MSA absent after TinyProt fetch")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cif", type=Path)
    p.add_argument("residues")
    p.add_argument("output", type=Path)
    p.add_argument("defaults", type=Path)
    p.add_argument("--chains")
    args = p.parse_args()
    prepare(args.cif, args.residues, args.output, args.defaults, args.chains)
