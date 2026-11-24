"""Missing requested-site coordinates must remain explicit missing data."""

from pathlib import Path

import numpy as np
import pandas as pd

from promera_angles import analysis


def measure_partial(monkeypatch, *, nonfinite=False, contact=False):
    x, y, z = np.meshgrid(
        np.arange(-6, 7, 2.0),
        np.arange(-6, 7, 2.0),
        [-4.0, -2.0, 0.0],
        indexing="ij",
    )
    antigen = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    binder = np.array(
        [
            [-3, -3, 15],
            [-2, 0, 3.17],
            [3, -3, 16],
            [0, 0, 3.17],
            [-3, 3, 16],
            [2, 0, 3.17],
            [3, 3, 15],
        ]
    )
    if not contact:
        binder[:, 2] += 100
    selected = [
        i
        for i, point in enumerate(antigen)
        if point[2] == 0 and abs(point[0]) == 2 and abs(point[1]) == 2
    ]
    plan = dict(
        sequences={"A": "A" * len(antigen)},
        binder_chain="B",
        framework="A<cdrh1>A<cdrh2>A<cdrh3>A",
        coords=antigen.tolist(),
        keys=[["A", i + 1] for i in range(len(antigen))],
        selected=selected,
        frame=dict(
            normal=[0, 0, 1], origin=[0, 0, 0], neighbourhood=list(range(len(antigen)))
        ),
    )
    rows = [
        dict(
            label_asym_id=chain,
            pos=i + 1,
            label_comp_id="ALA",
            label_atom_id="CA",
            Cartn_x=point[0],
            Cartn_y=point[1],
            Cartn_z=point[2],
        )
        for chain, coords in (("A", antigen), ("B", binder))
        for i, point in enumerate(coords)
    ]
    atoms = pd.DataFrame(rows)
    missing = (atoms.label_asym_id == "A") & (atoms.pos == selected[0] + 1)
    if nonfinite:
        atoms.loc[missing, "Cartn_x"] = np.nan
    else:
        atoms = atoms[~missing]
    monkeypatch.setattr(analysis, "atom_table", lambda path: atoms.copy())
    measured, rows, _ = analysis.measurements(Path("unused.cif"), plan, [2, 4, 6])
    return measured, rows


def test_absent_selected_residue_is_not_a_negative_contact_observation(monkeypatch):
    measured, rows = measure_partial(monkeypatch)
    assert measured["requested_site_contact"] is None
    assert measured["cdr_patch_contact"] is None
    assert measured["requested_site_coordinate_status"] == "missing_coordinates"
    assert len(measured["missing_requested_site_residues"]) == 1
    missing = [row for row in rows if not row["coordinate_present"]]
    assert len(missing) == 1
    assert missing[0]["cdr_residues"] is None
    assert missing[0]["framework_residues"] is None


def test_known_contact_remains_true_when_another_requested_residue_is_absent(
    monkeypatch,
):
    measured, _ = measure_partial(monkeypatch, contact=True)
    assert measured["requested_site_contact"] is True
    assert measured["requested_site_coordinate_status"] == "missing_coordinates"


def test_nonfinite_selected_ca_preserves_missing_coordinate_status(monkeypatch):
    measured, _ = measure_partial(monkeypatch, nonfinite=True)
    assert measured["footprint_status"] == "missing_coordinates"
    assert measured["requested_site_contact"] is None
    assert measured["patch_ca_pairs_8A"] is None
