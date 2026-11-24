"""Edge contracts found by independent review of the revised audit estimator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from promera_angles.footprint import (
    angle,
    elevation,
    footprint_geometry,
    measure_footprint,
)
from promera_angles.geometry import elevation_placement, scaffold_shape
from promera_angles.structures import atom_table


def solid(curve: float = 0.0) -> np.ndarray:
    x, y, depth = np.meshgrid(
        np.arange(-30, 31, 2.0),
        np.arange(-30, 31, 2.0),
        np.arange(-28, 1, 2.0),
        indexing="ij",
    )
    return np.column_stack(
        (x.ravel(), y.ravel(), (depth + curve * (x * x + y * y)).ravel())
    )


def test_flat_label_requires_the_complete_sensitivity_range() -> None:
    curve = -0.04
    sites = np.array(
        [[x, y, curve * (x * x + y * y)] for x in (-6.0, 6.0) for y in (-6.0, 6.0)]
    )
    result = footprint_geometry(cKDTree(solid(curve)), sites, [1, 0, 0])
    assert result["status"] == "ok"
    assert result["detail"] == ""
    assert result["footprint_elevation_deg"] == pytest.approx(0, abs=2e-6)
    assert result["sensitivity_min_deg"] < -8
    assert result["sensitivity_max_deg"] > 8
    assert result["pose_label"] == ""


def test_conflicting_scales_are_distinct_from_a_cancelled_pooled_footprint() -> None:
    x, y = np.meshgrid(np.arange(-30, 31, 2.0), np.arange(-30, 31, 2.0), indexing="ij")
    # A nearby plane dominates sigma 4; the farther plane dominates sigma 6.
    atoms = np.vstack(
        [
            np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))
            for z in (2.0, -8.0)
        ]
    )
    sites = np.array([[x, y, 0.0] for x in (-2.0, 2.0) for y in (-2.0, 2.0)])
    result = footprint_geometry(cKDTree(atoms), sites, [0, 0, 1])
    assert result["status"] == "incompatible_directions_across_scales"
    assert result["coherence_min"] == pytest.approx(1.0)
    assert result["scale_disagreement_deg"] == pytest.approx(180.0)
    assert result["elevation_sigma4_deg"] == pytest.approx(-90.0)
    assert result["elevation_sigma6_deg"] == pytest.approx(90.0)
    assert "footprint_elevation_deg" not in result


def test_actual_footprint_follows_a_different_face_without_using_requested_patch() -> (
    None
):
    original = solid()
    rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    wall = original @ rotation.T + [80, 0, 0]
    antigen = np.vstack((original, wall))
    keys = [("A", i + 1) for i in range(len(antigen))]
    scaffold = np.array(
        [
            [-3, -3, 15],
            [-2, 0, 3.17],
            [3, -3, 16],
            [0, 0, 3.17],
            [-3, 3, 16],
            [2, 0, 3.17],
            [3, 3, 15],
        ]
    ) @ rotation.T + [80, 0, 0]
    framework, cdr = scaffold[[0, 2, 4, 6]], scaffold[[1, 3, 5]]
    result = measure_footprint(
        antigen, keys, scaffold, list(range(1, 8)), framework, cdr, [2, 4, 6]
    )
    requested_normal = [0, 0, 1]
    assert result["status"] == "ok"
    assert result["cdr_contact_fraction"] == 1.0
    assert result["footprint_elevation_deg"] == pytest.approx(90, abs=2e-6)
    assert elevation(
        framework.mean(0) - cdr.mean(0), requested_normal
    ) == pytest.approx(0, abs=2e-6)
    assert angle(requested_normal, result["normal"]) == pytest.approx(90, abs=2e-6)
    # All actual contacts belong to the other face, not the originally selected patch.
    assert result["footprint_residue_keys"]
    assert all(
        position > len(original) for _, position in result["footprint_residue_keys"]
    )


def body_shape() -> dict:
    ca = np.array(
        [
            [-3, -3, 12],
            [3, -3, 13],
            [-2, 0, 0.4],
            [-4, 1, 14.2],
            [3, 3, 12.5],
            [0.2, 1, -0.3],
            [-2, 4, 15],
            [4, 0, 14.6],
            [2, -0.8, 0.8],
            [-1, -4, 13.8],
            [2, 4, 15.7],
        ]
    )
    return scaffold_shape(ca, [0, 1, 3, 4, 6, 7, 9, 10], [2, 5, 8])


def test_encoded_landmarks_must_belong_to_the_verified_full_scaffold() -> None:
    shape = body_shape()
    shape["local_xyz"] = (np.asarray(shape["local_xyz"]) + [8, 0, 0]).tolist()
    frame = dict(origin=[0.0, 0, 0], normal=[0.0, 0, 1], tangent=[1.0, 0, 0])
    with pytest.raises(ValueError, match="landmarks"):
        elevation_placement(frame, shape, [80, 80], 31)


def test_full_scaffold_mapping_must_partition_residues() -> None:
    shape = body_shape()
    shape["framework_indices"][0] = shape["framework_indices"][1]
    frame = dict(origin=[0.0, 0, 0], normal=[0.0, 0, 1], tangent=[1.0, 0, 0])
    with pytest.raises(ValueError, match="partition"):
        elevation_placement(frame, shape, [80, 80], 31)


def write_atoms(path: Path, occupancy: str) -> None:
    columns = [
        "label_asym_id",
        "label_seq_id",
        "label_comp_id",
        "label_atom_id",
        "type_symbol",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "pdbx_PDB_model_num",
    ]
    lines = [
        "data_occupancy",
        "loop_",
        *["_atom_site." + column for column in columns],
        f"A 1 ALA CA C 0 0 0 {occupancy} 1",
        "A 1 ALA CB C 0 0 1 ? 1",
        "A 1 ALA H H 0 0 2 1 1",
        "A 2 ALA CA C 1 0 0 0 1",
    ]
    path.write_text("\n".join(lines) + "\n#\n")


def test_unspecified_occupancies_match_the_revised_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "tinyprot.mmcif", None)
    path = tmp_path / "atoms.cif"
    write_atoms(path, ".")
    atoms = atom_table(path)
    assert list(atoms.label_atom_id) == ["CA", "CB"]
    assert list(atoms.pos) == [1, 1]


@pytest.mark.parametrize("occupancy", ["nan", "inf"])
def test_explicit_nonfinite_occupancy_is_not_silently_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, occupancy: str
) -> None:
    monkeypatch.setitem(sys.modules, "tinyprot.mmcif", None)
    path = tmp_path / "atoms.cif"
    write_atoms(path, occupancy)
    with pytest.raises(ValueError, match="nonfinite occupancy"):
        atom_table(path)
