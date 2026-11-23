"""Generate a deterministic analysis fixture. Coordinates are invented, not model output."""

import argparse
from pathlib import Path

import numpy as np

from experiments.pose_guidance.pose_geometry import framework_map

from .analysis import analyse, write_json
from .geometry import patch_frame

PATTERN = "ASG<cdrh1>GAS<cdrh2>SGA<cdrh3>AGS"
SEQUENCE = "ASGXGASXXSGAXAGS"


def fixture():
    coords = np.array(
        [
            (x, y, z)
            for z in (0, -4, -8)
            for x in range(-12, 13, 4)
            for y in range(-12, 13, 4)
        ],
        dtype=float,
    )
    keys, seqs = [], {"Left": "", "Right": ""}
    for x, y, z in coords:
        chain = "Left" if x < 1 else "Right"
        seqs[chain] += "A"
        keys.append((chain, len(seqs[chain])))
    selected = [
        i for i, (x, y, z) in enumerate(coords) if z == 0 and y == 0 and x in (0, 4)
    ]
    labels = [
        dict(
            template_chain=c,
            label_seq_id=i,
            auth_chain=c,
            auth_seq_id=str(i + 37),
            insertion_code="?",
        )
        for c, i in keys
    ]
    return dict(
        target="synthetic",
        sequences=seqs,
        binder_chain="VHH",
        framework=PATTERN,
        coords=coords.tolist(),
        keys=keys,
        selected=selected,
        labels=labels,
        frame=patch_frame(coords, selected),
    )


def write_cif(path, chains):
    """Minimal polymer mmCIF with separate label/author numbers and declared sequences."""
    name = {"A": "ALA", "G": "GLY", "S": "SER", "X": "UNK"}

    def loop(tags, rows):
        return (
            "loop_\n"
            + "\n".join(tags)
            + "\n"
            + "\n".join(" ".join(map(str, r)) for r in rows)
            + "\n#\n"
        )

    parts = ["data_synthetic\n#\n"]
    entities = list(enumerate(chains.items(), 1))
    parts.append(
        loop(["_entity.id", "_entity.type"], [(e, "polymer") for e, _ in entities])
    )
    parts.append(
        loop(
            ["_entity_poly.entity_id", "_entity_poly.type"],
            [(e, "polypeptide(L)") for e, _ in entities],
        )
    )
    parts.append(
        loop(
            ["_struct_asym.id", "_struct_asym.entity_id"],
            [(c, e) for e, (c, _) in entities],
        )
    )
    parts.append(
        loop(
            [
                "_entity_poly_seq.entity_id",
                "_entity_poly_seq.num",
                "_entity_poly_seq.mon_id",
            ],
            [
                (e, i, name[a])
                for e, (c, (seq, xyz)) in entities
                for i, a in enumerate(seq, 1)
            ],
        )
    )
    rows = []
    for e, (c, (seq, xyz)) in entities:
        for i, (aa, p) in enumerate(zip(seq, xyz, strict=True), 1):
            for at, element, off in (
                ("CA", "C", [0, 0, 0]),
                ("N", "N", [-1, 0.4, 0]),
                ("C", "C", [1, 0.5, 0]),
                ("O", "O", [1.3, 0.9, 0.3]),
            ):
                rows.append(
                    [
                        "ATOM",
                        len(rows) + 1,
                        element,
                        at,
                        ".",
                        name[aa],
                        c,
                        e,
                        i,
                        "?",
                        *[f"{v:.5f}" for v in p + off],
                        "1",
                        i + 37,
                        c,
                        1,
                    ]
                )
    cols = "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id label_asym_id label_entity_id label_seq_id pdbx_PDB_ins_code Cartn_x Cartn_y Cartn_z occupancy auth_seq_id auth_asym_id pdbx_PDB_model_num".split()
    parts.append(loop(["_atom_site." + k for k in cols], rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts))


def binder_coords(plan, spatial, facing):
    fixed, cdr = framework_map(SEQUENCE, PATTERN)

    def direction(angle):
        a = np.radians(angle)
        return np.array([np.sin(a), 0.0, np.cos(a)])

    center = np.asarray(plan["frame"]["origin"]) + 13 * direction(spatial)
    tip = center - 9 * direction(facing)
    coords = np.empty((len(SEQUENCE), 3))
    for ids, origin in ((fixed, center), (cdr, tip)):
        phi = np.arange(len(ids)) * 2 * np.pi / len(ids)
        coords[ids] = origin + np.stack(
            [2 * np.cos(phi), 2 * np.sin(phi), 0.4 * np.sin(2 * phi)], axis=1
        )
    return coords


def demonstrate(root):
    plan = fixture()
    tc = np.array(plan["coords"])
    chains = {
        c: (seq, tc[[i for i, k in enumerate(plan["keys"]) if k[0] == c]])
        for c, seq in plan["sequences"].items()
    }
    jobs = []
    for rep in (1, 2):
        for arm in ("ordinary", "guided"):
            out = root / arm / f"rep-{rep}"
            jobs.append(dict(arm=arm, replicate=rep, start=0, count=3, output=str(out)))
            for i in range(3):
                # Include deliberate disagreement and an out-of-band guided structure.
                spatial = ([18, 58, 72] if arm == "ordinary" else [5, 12, 40])[i] + rep
                facing = ([77, 15, 64] if arm == "ordinary" else [8, 75, 30])[i] + rep
                base = out / "synthetic" / f"sample{i}"
                write_cif(
                    base / "backbone.cif",
                    dict(chains, VHH=(SEQUENCE, binder_coords(plan, spatial, facing))),
                )
                write_json(
                    base / "angle.json",
                    dict(requested_deg=(5 * i + 2 if arm == "guided" else None)),
                )
                if i == 0:
                    write_cif(
                        base / "refolds" / "demo_refold0.cif",
                        dict(
                            chains,
                            VHH=(
                                SEQUENCE,
                                binder_coords(plan, spatial + 30, facing + 10),
                            ),
                        ),
                    )
    result = analyse(root / "analysis", jobs, plan, [0, 20])
    write_json(
        root / "fixture.json",
        {"origin": "hand-placed synthetic coordinate fixture", "plan": plan},
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    demonstrate(parser.parse_args().output)
