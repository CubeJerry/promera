"""Resolve one ordinary targeted input using the installed runtime's numbering."""

import gzip
import json
from pathlib import Path

import numpy as np

from experiments.pose_guidance.pose_geometry import AA3, digest, read_chains

from .geometry import GeometryError, patch_frame, prospective_patch_frame


def atom_table(path):
    try:
        from tinyprot.mmcif import read_mmcif
    except ModuleNotFoundError:
        import gemmi
        import pandas as pd

        category = (
            gemmi.cif.read(str(path)).sole_block().find_mmcif_category("_atom_site.")
        )
        table = pd.DataFrame(
            list(category), columns=[t.split(".", 1)[1] for t in category.tags]
        )
    else:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as handle:
            table = read_mmcif(handle)["_atom_site"].copy()
    if "pdbx_PDB_model_num" in table and table.pdbx_PDB_model_num.nunique() != 1:
        raise ValueError("expected one coordinate model per file")
    table["label_asym_id"] = table.label_asym_id.astype(str)
    table = table[~table.type_symbol.str.upper().isin(["H", "D"])]
    table = table[~table.label_seq_id.isin([".", "?"])].copy()
    table["pos"] = table.label_seq_id.astype(int)
    if "occupancy" in table:
        # Match the revised audit: unspecified occupancy means one; explicit
        # nonfinite occupancy is an input error, never a silently omitted atom.
        occupancy = table.occupancy.map(
            lambda value: 1.0 if value in (None, "", ".", "?") else float(value)
        ).to_numpy(float)
        if not np.isfinite(occupancy).all():
            raise ValueError("nonfinite occupancy")
        table = table.assign(_occ=occupancy)
        table = table[table._occ > 0].sort_values(
            "_occ", ascending=False, kind="stable"
        )
    return table.drop_duplicates(["label_asym_id", "pos", "label_atom_id"])


def resolve_target(cfg, geometry_path=None, *, surface_elevation=False):
    from omegaconf import OmegaConf

    from promera.inference.design import _map_epitope_residues
    from promera.inference.utils import (
        align_schema_to_template,
        validate_alignment_or_raise,
    )

    input_path = Path(cfg["input"])
    if input_path.is_dir():
        names = cfg.get("targets") or cfg.get("target_list")
        files = [
            p for p in sorted(input_path.glob("*.json")) if not names or p.stem in names
        ]
        if len(files) != 1:
            raise ValueError("select one target schema per angular experiment")
        input_path = files[0]
    schema = json.loads(input_path.read_text())
    selected = cfg.get("target_chains") or [
        key for key in schema if key != "connections"
    ]
    if isinstance(selected, str):
        selected = [s.strip() for s in selected.split(",")]
    if len(set(selected)) != len(selected) or cfg["binder"]["chain"] in selected:
        raise ValueError("invalid target chain selection")
    sequences = {key: schema[key]["sequence"] for key in selected}
    if any(schema[k].get("type") != "protein" for k in selected):
        raise ValueError("angle geometry requires resolved protein target chains")
    template = cfg.get("target_template") or {}
    path = geometry_path or template.get("path")
    if not path:
        raise ValueError(
            "supply --target-structure or a target_template in the ordinary design config"
        )
    same_template = (
        template.get("path")
        and Path(path).resolve() == Path(template["path"]).resolve()
    )
    chains = read_chains(path)
    expected = cfg.get("epitope_residue_map")
    if expected is None:
        ec = cfg["epitope_chain"]
        expected = {ec: _map_epitope_residues(schema, OmegaConf.create(cfg), ec)}
    site = {(c, int(p)) for c, positions in expected.items() for p in positions}
    if not site or any(
        c not in sequences or not 1 <= p <= len(sequences[c]) for c, p in site
    ):
        raise ValueError(
            "expected a nonempty native epitope on selected runtime chains"
        )

    coords, keys, labels = [], [], []
    heavy_coords, heavy_keys = [], []
    table = atom_table(path)
    for key, seq in sequences.items():
        source_id = (
            template.get("chain", key) if same_template and len(selected) == 1 else key
        )
        source = chains[source_id]
        src_ids = source["residue_ids"]
        explicit = (
            (template.get("runtime_residue_map") or {}).get(key)
            if same_template
            else None
        )
        if explicit:
            lookup = {v: i for i, v in enumerate(src_ids)}
            mapping = {int(k) - 1: lookup[str(v)] for k, v in explicit.items()}
            if (
                len(lookup) != len(src_ids)
                or set(mapping) != set(range(len(seq)))
                or len(set(mapping.values())) != len(mapping)
            ):
                raise ValueError(f"ambiguous or incomplete template map for {key}")
        else:
            if source["sequence"].find(seq) != source["sequence"].rfind(seq):
                raise ValueError(
                    f"ambiguous repeated sequence crop for {key}; supply runtime_residue_map"
                )
            mapping, stats = align_schema_to_template(seq, source["sequence"])
            validate_alignment_or_raise(
                stats,
                template.get("min_identity", 0.8),
                template.get("min_coverage", 0.5),
            )
        atoms = table[
            (table.label_asym_id == source_id) & (table.label_atom_id == "CA")
        ].set_index("pos")
        for i, j in sorted(mapping.items()):
            if seq[i] != source["sequence"][j]:
                raise ValueError(
                    f"sequence mismatch at {key}:{i + 1}; verify the structural correspondence"
                )
            rid = int(src_ids[j])
            if surface_elevation:
                residue_atoms = table[
                    (table.label_asym_id == source_id) & (table.pos == rid)
                ]
                values = residue_atoms[["Cartn_x", "Cartn_y", "Cartn_z"]].to_numpy(
                    float
                )
                if not np.isfinite(values).all():
                    raise ValueError(f"nonfinite antigen coordinates at {key}:{i + 1}")
                heavy_coords.extend(values.tolist())
                heavy_keys.extend([(key, i + 1)] * len(values))
            ca = source["coords"][j]
            if not np.isfinite(ca).all():
                continue
            rid = int(src_ids[j])
            row = atoms.loc[rid]
            keys.append((key, i + 1))
            coords.append(ca.tolist())
            labels.append(
                dict(
                    template_chain=source_id,
                    label_seq_id=rid,
                    auth_chain=str(row.get("auth_asym_id", source_id)),
                    auth_seq_id=str(row.get("auth_seq_id", rid)),
                    insertion_code=str(row.get("pdbx_PDB_ins_code", "?")),
                )
            )
    missing = site.difference(keys)
    if missing:
        raise ValueError(
            f"epitope lacks mapped resolved Cα coordinates: {sorted(missing)}"
        )
    indices = [i for i, k in enumerate(keys) if k in site]
    legacy_frame, legacy_error = None, None
    try:
        legacy_frame = patch_frame(coords, indices)
    except GeometryError as exc:
        if not surface_elevation:
            raise
        legacy_error = str(exc)
    frame = legacy_frame
    if surface_elevation:
        heavy = np.asarray(heavy_coords)
        samples = [
            heavy[[j for j, key in enumerate(heavy_keys) if key == keys[i]]].mean(0)
            for i in indices
        ]
        frame = prospective_patch_frame(
            coords,
            indices,
            heavy,
            samples,
            landmarks=legacy_frame["landmarks"] if legacy_frame is not None else None,
        )
    result = dict(
        version=1,
        target=input_path.stem,
        input=str(input_path.resolve()),
        schema_sha256=digest(input_path),
        structure=str(Path(path).resolve()),
        structure_sha256=digest(path),
        sequences=sequences,
        keys=keys,
        labels=labels,
        coords=coords,
        selected=indices,
        frame=frame,
        binder_chain=cfg["binder"]["chain"],
        framework=cfg["binder"]["framework"],
    )
    if surface_elevation:
        result.update(
            version=2,
            experiment_definition="surface_elevation_v1",
            legacy_frame=legacy_frame,
            legacy_frame_status="ok" if legacy_frame else "unavailable",
            legacy_frame_error=legacy_error,
            prospective_antigen_heavy_atom_count=len(heavy_coords),
            prospective_sampling_residues=[keys[i] for i in indices],
        )
    return result


def scaffold_from_cif(path, chain, pattern):
    from experiments.pose_guidance.pose_geometry import framework_map

    from .geometry import scaffold_shape

    rows = atom_table(path)
    ca = rows[(rows.label_asym_id == chain) & (rows.label_atom_id == "CA")].sort_values(
        "pos"
    )
    if list(ca.pos) != list(range(1, len(ca) + 1)):
        raise ValueError("unbound scaffold has incomplete residue coordinates")
    fixed, cdr = framework_map(
        "".join(AA3.get(k, "X") for k in ca.label_comp_id), pattern
    )
    return scaffold_shape(
        ca[["Cartn_x", "Cartn_y", "Cartn_z"]].to_numpy(float), fixed, cdr
    )
