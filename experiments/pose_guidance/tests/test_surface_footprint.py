"""Benign synthetic geometry regressions and revised-audit numerical parity."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from promera_angles.footprint import (
    angle,
    elevation,
    estimate_surface_normal,
    footprint_geometry,
    measure_footprint,
    regions,
    unit,
)


def grid_surface(
    curve: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reproduce grid_surface from the revised audit's supplied test fixtures."""
    x, y, depth = np.meshgrid(
        np.arange(-30, 31, 2.0),
        np.arange(-30, 31, 2.0),
        np.arange(-28, 1, 2.0),
        indexing="ij",
    )
    z = depth + curve * (x * x + y * y)
    sites = np.array(
        [
            [a, b, curve * (a * a + b * b)]
            for a in (-4.0, 0.0, 4.0)
            for b in (-4.0, 0.0, 4.0)
        ]
    )
    return np.column_stack((x.ravel(), y.ravel(), z.ravel())), sites


def recipe_arrays(
    case: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Expand audit-derived synthetic fixture recipes."""
    atoms, sites = grid_surface(case.get("curve", 0.0))
    if case.get("kind") == "weak_site":
        sites = np.vstack((sites, [0.0, 0.0, -12.0]))
    elif case.get("kind") == "narrow":
        sites = np.array([[x, 0.0, 0.0] for x in (-6, -2, 2, 6)])
    elif case.get("kind") == "disconnected":
        sites = np.array([[x, y, 0.0] for x in (-14, -12, 12, 14) for y in (0, 4)])
    elif case.get("kind") == "opposing":
        rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        left = atoms @ rotation.T - [5, 0, 0]
        sites_left = sites @ rotation.T - [5, 0, 0]
        right, sites_right = left.copy(), sites_left.copy()
        right[:, 0] *= -1
        sites_right[:, 0] *= -1
        atoms, sites = np.vstack((left, right)), np.vstack((sites_left, sites_right))
    elif case.get("kind") == "buried":
        sites = sites - [0, 0, 14]
    elif case.get("kind") == "irregular":
        rng = np.random.default_rng(629)
        atoms = atoms + rng.normal(0, 0.1, atoms.shape)
        sites = sites + rng.normal(0, 0.1, sites.shape)
    return atoms, sites


def test_revised_audit_numerical_parity() -> None:
    path = Path(__file__).parent / "fixtures/revised_footprint_reference.py"
    spec = importlib.util.spec_from_file_location("revised_audit_reference", path)
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    cases: list[dict[str, Any]] = []
    for value in (0, 30, 90, -40, -90):
        for azimuth in (0.1, 1.3, 2.7):
            radians = np.radians(value)
            cases.append(
                {
                    "axis": [
                        np.cos(radians) * np.cos(azimuth),
                        np.cos(radians) * np.sin(azimuth),
                        np.sin(radians),
                    ]
                }
            )
    cases.extend({"curve": curve, "axis": [0, 0, 1]} for curve in (-0.025, 0.025))
    cases.extend(
        {"kind": kind, "axis": [0, 0, 1]}
        for kind in (
            "weak_site",
            "narrow",
            "disconnected",
            "opposing",
            "buried",
            "irregular",
        )
    )
    for case in cases:
        atoms, sites = recipe_arrays(case)
        expected = reference.footprint_geometry(cKDTree(atoms), sites, case["axis"])
        observed = footprint_geometry(cKDTree(atoms), sites, case["axis"])
        for key, value in expected.items():
            if isinstance(value, (float, int, np.number)):
                assert observed[key] == pytest.approx(value, abs=2e-6, rel=0), (
                    case,
                    key,
                )
            else:
                assert observed[key] == value, (case, key)
    # The fixture construction itself must agree with the supplied audit test.
    for curve in (-0.025, 0, 0.025):
        actual = grid_surface(curve)
        original = reference.grid_surface(curve)
        for a, b in zip(actual, original):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("value", [-90, -40, -4, 0, 4, 30, 90])
@pytest.mark.parametrize("azimuth", [0.1, 1.3, 2.7])
def test_planar_surface_supports_every_signed_body_elevation(
    value: float, azimuth: float
) -> None:
    atoms, sites = grid_surface()
    radians = np.radians(value)
    axis = [
        np.cos(radians) * np.cos(azimuth),
        np.cos(radians) * np.sin(azimuth),
        np.sin(radians),
    ]
    measured = footprint_geometry(cKDTree(atoms), sites, axis)
    assert measured["status"] == "ok"
    assert measured["footprint_elevation_deg"] == pytest.approx(value, abs=2e-6)
    assert measured["pose_label"] == ("flat" if abs(value) <= 5 else "")


@pytest.mark.parametrize("curve", [-0.025, 0.025])
def test_convex_and_concave_apex(curve: float) -> None:
    atoms, sites = grid_surface(curve)
    result = footprint_geometry(cKDTree(atoms), sites, [0, 0, 1])
    assert result["status"] == "ok"
    assert result["footprint_elevation_deg"] == pytest.approx(90.0, abs=2e-6)


def test_rotation_translation_and_prospective_normal_share_the_estimator() -> None:
    atoms, sites = grid_surface(-0.014)
    rotation = Rotation.from_rotvec([0.53, -0.19, 0.68]).as_matrix()
    translation = np.array([34.0, -160.0, 9.0])
    axis = np.array([1.0, 2.0, 1.0])
    before = footprint_geometry(cKDTree(atoms), sites, axis)
    after = footprint_geometry(
        cKDTree(atoms @ rotation + translation),
        sites @ rotation + translation,
        axis @ rotation,
    )
    prospective = estimate_surface_normal(atoms, sites)
    moved = estimate_surface_normal(
        atoms @ rotation + translation, sites @ rotation + translation
    )
    assert "footprint_elevation_deg" not in prospective
    assert "pose_label" not in prospective
    for key in (
        "footprint_elevation_deg",
        "sensitivity_min_deg",
        "sensitivity_max_deg",
    ):
        assert after[key] == pytest.approx(before[key], abs=2e-6)
    np.testing.assert_allclose(prospective["normal"], before["normal"], atol=1e-12)
    np.testing.assert_allclose(
        moved["normal"], np.array(prospective["normal"]) @ rotation, atol=1e-10
    )
    sensitivity = [elevation(axis, n) for n in prospective["sensitivity_normals"]]
    assert min(sensitivity) == pytest.approx(before["sensitivity_min_deg"], abs=2e-6)
    assert max(sensitivity) == pytest.approx(before["sensitivity_max_deg"], abs=2e-6)


def test_one_weak_site_is_weighted_instead_of_vetoing_the_footprint() -> None:
    atoms, sites = grid_surface()
    result = footprint_geometry(
        cKDTree(atoms), np.vstack((sites, [0, 0, -12])), [0, 0, 1]
    )
    assert result["status"] == "ok"
    assert "weak_local_gradient" in result["detail"]
    assert result["footprint_elevation_deg"] > 89.9


def test_opposing_walls_and_buried_sites_are_distinct_failures() -> None:
    atoms, sites = recipe_arrays({"kind": "opposing"})
    opposing = footprint_geometry(cKDTree(atoms), sites, [0, 0, 1])
    assert opposing["status"] == "opposing_surface_normals"
    assert "footprint_elevation_deg" not in opposing
    atoms, sites = recipe_arrays({"kind": "buried"})
    buried = footprint_geometry(cKDTree(atoms), sites, [0, 0, 1])
    assert buried["status"] == "no_resolved_outward_direction"
    assert "footprint_elevation_deg" not in buried


def test_centroid_pointing_stays_zero_while_body_elevation_changes() -> None:
    atoms, sites = grid_surface()
    measured = []
    for value in (15.0, 80.0):
        radians = np.radians(value)
        framework = 15 * np.array([np.cos(radians), 0, np.sin(radians)])
        cdr = framework / 2
        assert angle(cdr - framework, -framework) == pytest.approx(0, abs=2e-6)
        measured.append(
            footprint_geometry(cKDTree(atoms), sites, framework - cdr)[
                "footprint_elevation_deg"
            ]
        )
    np.testing.assert_allclose(measured, [15, 80], atol=2e-6)


def complex_atoms(
    curve: float = 0.0,
) -> tuple[NDArray[np.float64], list[tuple[str, int]], NDArray[np.float64]]:
    """Use a synthetic solid and seven-residue body, away from cutoff ties."""
    atoms, _ = grid_surface(curve)
    counts = {"A": 0, "D": 0}
    keys = []
    for point in atoms:
        chain = "A" if point[0] < 0 else "D"
        counts[chain] += 1
        keys.append((chain, counts[chain]))
    # Avoid exact 3-4-5 distance ties in rigid-motion contact-set tests.
    height = 3.17
    binder = np.array(
        [
            [5, -3, 15],
            [-6, 0, height + curve * 36],
            [11, -3, 16],
            [0, 0, height],
            [5, 3, 16],
            [6, 0, height + curve * 36],
            [11, 3, 15],
        ],
        dtype=float,
    )
    return atoms, keys, binder


def measure_complex(
    atoms: NDArray[np.float64],
    keys: list[tuple[str, int]],
    binder: NDArray[np.float64],
) -> dict[str, Any]:
    return measure_footprint(
        atoms,
        keys,
        binder,
        list(range(1, 8)),
        binder[[0, 2, 4, 6]],
        binder[[1, 3, 5]],
        [2, 4, 6],
        antigen_ca=dict(zip(keys, atoms)),
    )


@pytest.mark.parametrize("curve", [-0.012, -0.025, -0.06])
def test_cdr_contacts_follow_a_convex_surface_with_an_oblique_body(
    curve: float,
) -> None:
    atoms, keys, binder = complex_atoms(curve)
    result = measure_complex(atoms, keys, binder)
    assert result["status"] == "ok"
    assert result["cdr_contact_fraction"] == 1.0
    assert result["contacting_cdr_residues"] == 3
    assert 45 < result["footprint_elevation_deg"] < 75
    assert {key[0] for key in result["footprint_residue_keys"]} == {"A", "D"}
    rotation = Rotation.from_rotvec([0.2, 0.6, -0.4]).as_matrix()
    shifted = measure_complex(
        atoms @ rotation + [100, -4, 8],
        keys,
        binder @ rotation + [100, -4, 8],
    )
    assert shifted["footprint_elevation_deg"] == pytest.approx(
        result["footprint_elevation_deg"], abs=2e-6
    )
    assert shifted["footprint_residue_keys"] == result["footprint_residue_keys"]


def test_all_cdr_residues_enter_axis_even_without_contacts() -> None:
    atoms, keys, binder = complex_atoms()
    extra_cdr = np.array([[20.0, 0.0, 30.0]])
    full_cdr = np.vstack((binder[[1, 3, 5]], extra_cdr))
    result = measure_footprint(
        atoms,
        keys,
        np.vstack((binder, extra_cdr)),
        list(range(1, 9)),
        binder[[0, 2, 4, 6]],
        full_cdr,
        [2, 4, 6, 8],
    )
    assert result["status"] == "ok"
    assert result["contacting_cdr_residues"] == 3
    assert 8 not in result["contacting_binder_positions"]
    np.testing.assert_allclose(result["cdr_centroid"], full_cdr.mean(0))
    assert result["footprint_elevation_deg"] == pytest.approx(
        elevation(binder[[0, 2, 4, 6]].mean(0) - full_cdr.mean(0), result["normal"]),
        abs=1e-9,
    )


def test_contact_sampling_uses_every_contacted_atom_once_per_chain_residue() -> None:
    atoms, keys, binder = complex_atoms()
    keys = [(chain, (pos + 1) // 2) for chain, pos in keys]
    binder[0] = [-8, -2, 3.17]  # A framework atom contributes to the footprint.
    result = measure_complex(atoms, keys, binder)
    assert result["status"] == "ok"
    assert result["contacting_framework_residues"] == 1
    assert result["cdr_contact_fraction"] == 0.75
    distances = np.linalg.norm(atoms[:, None] - binder[None, :], axis=2)
    contact = distances.min(axis=1) <= 5.0
    groups: dict[tuple[str, int], list[NDArray[np.float64]]] = {}
    for key, xyz, hit in zip(keys, atoms, contact):
        if hit:
            groups.setdefault(key, []).append(xyz)
    assert result["footprint_residue_keys"] == [list(key) for key in sorted(groups)]
    np.testing.assert_allclose(
        result["sampling_points"],
        [np.mean(groups[key], axis=0) for key in sorted(groups)],
    )


def test_absent_contacts_missing_coordinates_and_degenerate_axes() -> None:
    atoms, keys, binder = complex_atoms()
    assert measure_complex(atoms, keys, binder + [0, 0, 100])["status"] == "no_contacts"
    missing = binder.copy()
    missing[0, 0] = np.nan
    assert measure_complex(atoms, keys, missing)["status"] == "missing_coordinates"
    assert (
        measure_footprint(
            [], [], binder, list(range(1, 8)), binder[:1], binder[1:2], [2]
        )["status"]
        == "missing_coordinates"
    )
    assert (
        measure_footprint(atoms, keys, binder, list(range(1, 8)), [], binder[1:2], [2])[
            "status"
        ]
        == "missing_coordinates"
    )
    assert (
        measure_footprint(
            atoms, keys, binder, list(range(1, 8)), binder[1:2], binder[1:2], [2]
        )["status"]
        == "undefined_vhh_axis"
    )
    assert (
        footprint_geometry(cKDTree(atoms), [], [0, 0, 1])["status"]
        == "missing_coordinates"
    )
    assert (
        footprint_geometry(cKDTree(atoms), [[0, 0, 0]], [0, 0, 0])["status"]
        == "undefined_vhh_axis"
    )
    assert estimate_surface_normal([], [[0, 0, 0]])["status"] == "missing_coordinates"
    with pytest.raises(ValueError, match="different lengths"):
        measure_footprint(
            atoms, keys[:-1], binder, list(range(1, 8)), binder[:1], binder[1:2], [2]
        )
    with pytest.raises(ValueError, match="N by 3"):
        estimate_surface_normal([1, 2, 3], [[0, 0, 0]])
    with pytest.raises(ValueError, match="undefined direction"):
        unit([0, 0, 0])


def test_validated_mapping_preserves_framework_segments_and_all_cdrs() -> None:
    pattern = "AA<cdrh1>CC<cdrh2>DD<cdrh3>EE"
    sequence = "AAGGCCDDDEEE"
    saved = [3, 4, 7, 10]
    fw, cdr = regions(sequence, pattern, saved)
    assert cdr.tolist() == [2, 3, 6, 9]
    assert "".join(sequence[i] for i in fw) == "AACCDDEE"
    fw2, cdr2 = regions(sequence, pattern, None)
    np.testing.assert_array_equal(fw, fw2)
    np.testing.assert_array_equal(cdr, cdr2)
    for wrong in ([1, 4, 7, 10], [3, 3, 7, 10], [3, 4, 7, 12], [], [3.0, 4, 7, 10]):
        with pytest.raises(ValueError):
            regions(sequence, pattern, wrong)
    with pytest.raises(ValueError, match="ambiguous"):
        regions("AAAAAAAAAA", "A<cdrh1>A<cdrh2>A<cdrh3>A", None)
    with pytest.raises(ValueError, match="pattern"):
        regions(sequence, "AA<cdrh2>CC<cdrh1>DD<cdrh3>EE", saved)
    with pytest.raises(ValueError, match="empty"):
        regions(sequence, "<cdrh1>CC<cdrh2>DD<cdrh3>EE", saved)
    with pytest.raises(ValueError, match="ends"):
        regions("ZAAGGCCDDDEEE", pattern, None)
