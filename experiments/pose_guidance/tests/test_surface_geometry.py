"""Geometry acceptance tests using synthetic scaffold and antigen coordinates.

The legacy reference is frozen at PR #15's original remote head. These tests
need no checkpoints, Torch, real antigen sequences or GPU inference.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from promera_angles.footprint import elevation, estimate_surface_normal
from promera_angles.geometry import (
    elevation_placement,
    placement,
    prospective_patch_frame,
    scaffold_shape,
)


def scaffold() -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Six asymmetric framework and three CDR points avoid landmark ties."""
    framework = np.array(
        [
            [-3.2, -2.4, 12.5],
            [4.7, -1.8, 14.4],
            [-2.1, 3.6, 15.2],
            [3.4, 2.7, 11.9],
            [0.3, -4.5, 13.8],
            [1.8, 5.2, 16.1],
        ]
    )
    cdr = np.array([[-2.0, 0.0, 1.3], [0.4, 1.2, -0.2], [2.1, -0.7, 0.4]])
    return (
        np.vstack((framework, cdr)),
        np.arange(6, dtype=np.int64),
        np.arange(6, 9, dtype=np.int64),
    )


def patch() -> dict[str, Any]:
    return {
        "origin": [7.0, -3.0, 2.0],
        "normal": [0.0, 0.0, 1.0],
        "tangent": [1.0, 0.0, 0.0],
    }


def record_geometry(record: dict[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    coordinates = np.asarray(record["scaffold_ca_xyz"], dtype=float)
    framework = np.asarray(record["scaffold_framework_indices"], dtype=int)
    cdr = np.asarray(record["scaffold_cdr_indices"], dtype=int)
    return coordinates, coordinates[framework].mean(0), coordinates[cdr].mean(0)


@pytest.mark.parametrize("requested", [-90.0, -60.0, 0.0, 30.0, 70.0, 90.0])
def test_signed_elevation_verified_from_full_centroids_with_c_fixed(requested: float) -> None:
    coordinates, framework, cdr = scaffold()
    shape = scaffold_shape(coordinates, framework, cdr)
    frame = patch()
    anchors, record = elevation_placement(
        frame, shape, [requested, requested], 17, 6.0,
        cdr_lateral_offset=(1.2, -0.6),
    )
    actual, f, c = record_geometry(record)
    assert record["requested_elevation_deg"] == pytest.approx(requested, abs=2e-6)
    assert record["realized_scaffold_elevation_deg"] == pytest.approx(requested, abs=2e-6)
    assert elevation(f - c, frame["normal"]) == pytest.approx(requested, abs=2e-6)
    np.testing.assert_allclose(c, [8.2, -3.6, 8.0], atol=1e-9)
    np.testing.assert_allclose(record["scaffold_cdr_centroid_xyz"], c, atol=1e-9)
    marks = framework[np.asarray(shape["fixed_indices"], dtype=int)]
    np.testing.assert_allclose(anchors, actual[marks], atol=1e-9)
    # All framework/CDR coordinates, not just four landmarks, survive one rigid transform.
    old_distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    new_distances = np.linalg.norm(actual[:, None] - actual[None, :], axis=2)
    np.testing.assert_allclose(new_distances, old_distances, atol=1e-9)


def test_elevation_changes_body_inclination_without_changing_cdr_end_or_random_roll() -> None:
    coordinates, framework, cdr = scaffold()
    shape = scaffold_shape(coordinates, framework, cdr)
    records = [
        elevation_placement(patch(), shape, [value, value], 123, 6.0)[1]
        for value in (-60, 0, 45, 90)
    ]
    centroids = [record_geometry(record)[2] for record in records]
    for centroid in centroids:
        np.testing.assert_allclose(centroid, centroids[0], atol=1e-9)
    assert len({round(record["azimuth_deg"], 10) for record in records}) == 1
    assert len({round(record["roll_deg"], 10) for record in records}) == 1
    vectors = [record_geometry(record)[1] - record_geometry(record)[2] for record in records]
    assert all(not np.allclose(vectors[0], vector) for vector in vectors[1:])


def test_cdr_position_and_body_inclination_are_independent_choices() -> None:
    coordinates, framework, cdr = scaffold()
    shape = scaffold_shape(coordinates, framework, cdr)
    first = elevation_placement(
        patch(), shape, [35, 35], 42, 6.0, cdr_lateral_offset=(0, 0)
    )[1]
    moved = elevation_placement(
        patch(), shape, [35, 35], 42, 9.0, cdr_lateral_offset=(2, -1)
    )[1]
    a, fa, ca = record_geometry(first)
    b, fb, cb = record_geometry(moved)
    np.testing.assert_allclose(b - a, np.tile([2, -1, 3], (len(a), 1)), atol=1e-9)
    np.testing.assert_allclose(cb - ca, [2, -1, 3], atol=1e-9)
    np.testing.assert_allclose(fb - cb, fa - ca, atol=1e-9)


@pytest.mark.parametrize("bounds", [[-90, -70], [-30, 30], [70, 90]])
def test_sampled_signed_intervals_and_seed_reproducibility(bounds: list[float]) -> None:
    coordinates, framework, cdr = scaffold()
    shape = scaffold_shape(coordinates, framework, cdr)
    for seed in (0, 31, 2026):
        anchors, record = elevation_placement(patch(), shape, bounds, seed)
        again, repeated = elevation_placement(patch(), shape, bounds, seed)
        assert bounds[0] - 1e-7 <= record["requested_elevation_deg"] <= bounds[1] + 1e-7
        actual, f, c = record_geometry(record)
        assert elevation(f - c, patch()["normal"]) == pytest.approx(
            record["requested_elevation_deg"], abs=2e-6
        )
        np.testing.assert_array_equal(anchors, again)
        np.testing.assert_array_equal(actual, repeated["scaffold_ca_xyz"])


def test_new_placement_is_equivariant_to_patch_rigid_motion() -> None:
    coordinates, framework, cdr = scaffold()
    shape = scaffold_shape(coordinates, framework, cdr)
    frame = patch()
    rotation = Rotation.from_rotvec([0.3, -0.7, 0.2]).as_matrix()
    translation = np.array([41.0, -8.0, 13.0])
    moved_frame = {
        "origin": (np.asarray(frame["origin"]) @ rotation + translation).tolist(),
        "normal": (np.asarray(frame["normal"]) @ rotation).tolist(),
        "tangent": (np.asarray(frame["tangent"]) @ rotation).tolist(),
    }
    before, rec_before = elevation_placement(frame, shape, [-25, 80], 31, 6.0)
    after, rec_after = elevation_placement(moved_frame, shape, [-25, 80], 31, 6.0)
    np.testing.assert_allclose(after, before @ rotation + translation, atol=1e-8)
    np.testing.assert_allclose(
        rec_after["scaffold_ca_xyz"],
        np.asarray(rec_before["scaffold_ca_xyz"]) @ rotation + translation,
        atol=1e-8,
    )
    assert rec_after["realized_scaffold_elevation_deg"] == pytest.approx(
        rec_before["realized_scaffold_elevation_deg"], abs=2e-6
    )


def test_framework_landmarks_and_local_scaffold_have_no_laboratory_frame_bias() -> None:
    coordinates, framework, cdr = scaffold()
    rotation = Rotation.from_rotvec([0.7, 0.25, -0.1]).as_matrix()
    first = scaffold_shape(coordinates, framework, cdr)
    moved = scaffold_shape(coordinates @ rotation + [9, 3, 44], framework, cdr)
    assert first["fixed_indices"] == moved["fixed_indices"]
    np.testing.assert_allclose(first["local_xyz"], moved["local_xyz"], atol=1e-8)
    np.testing.assert_allclose(
        first["scaffold_ca_local"], moved["scaffold_ca_local"], atol=1e-8
    )


@pytest.mark.parametrize("bounds", [[-91, 0], [0, 91], [30, 20], [float("nan"), 80]])
def test_invalid_signed_elevation_interval_is_rejected(bounds: list[float]) -> None:
    coordinates, framework, cdr = scaffold()
    shape = scaffold_shape(coordinates, framework, cdr)
    with pytest.raises(ValueError):
        elevation_placement(patch(), shape, bounds, 31)


def test_corrupted_scaffold_is_rejected_before_encoding() -> None:
    coordinates, framework, cdr = scaffold()
    shape = scaffold_shape(coordinates, framework, cdr)
    corrupted = copy.deepcopy(shape)
    xyz = np.asarray(corrupted["scaffold_ca_local"], dtype=float)
    xyz[framework] += [4.0, 0.0, 0.0]
    corrupted["scaffold_ca_local"] = xyz.tolist()
    with pytest.raises(ValueError):
        elevation_placement(patch(), corrupted, [70, 90], 31)
    corrupted = copy.deepcopy(shape)
    corrupted["scaffold_ca_local"][0][0] = float("nan")
    with pytest.raises(ValueError):
        elevation_placement(patch(), corrupted, [70, 90], 31)


def antigen_patch() -> tuple[NDArray[np.float64], list[int], NDArray[np.float64], NDArray[np.float64]]:
    ca = np.array(
        [
            [-4, -4, 0], [-4, 4, 0], [4, -4, 0], [4, 4, 0],
            [-8, 0, 0], [8, 0, 0], [0, -8, 0], [0, 8, 0],
        ],
        dtype=float,
    )
    x, y, z = np.meshgrid(
        np.arange(-24, 25, 2.0),
        np.arange(-24, 25, 2.0),
        np.arange(-24, 1, 2.0),
        indexing="ij",
    )
    heavy = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    return ca, [0, 1, 2, 3], heavy, ca[:4].copy()


def test_prospective_patch_uses_shared_density_and_explicit_residue_sampling_origin() -> None:
    ca, selected, heavy, sites = antigen_patch()
    sites += [0.3, -0.2, 0.7]
    expected = estimate_surface_normal(heavy, sites)
    frame = prospective_patch_frame(
        ca, selected, heavy, sites, landmarks=[0, 1, 2, 3]
    )
    np.testing.assert_allclose(frame["origin"], sites.mean(0), atol=1e-10)
    np.testing.assert_allclose(frame["normal"], expected["normal"], atol=1e-10)
    assert frame["landmarks"] == [0, 1, 2, 3]
    assert np.linalg.norm(frame["normal"]) == pytest.approx(1)
    assert np.linalg.norm(frame["tangent"]) == pytest.approx(1)
    assert np.dot(frame["normal"], frame["tangent"]) == pytest.approx(0, abs=1e-9)


def test_planar_patch_does_not_restrict_body_elevation() -> None:
    ca, selected, heavy, sites = antigen_patch()
    frame = prospective_patch_frame(
        ca, selected, heavy, sites, landmarks=[0, 1, 2, 3]
    )
    coordinates, framework, cdr = scaffold()
    shape = scaffold_shape(coordinates, framework, cdr)
    for value in (-90, 0, 35, 90):
        _, record = elevation_placement(frame, shape, [value, value], 12)
        assert record["realized_scaffold_elevation_deg"] == pytest.approx(value, abs=2e-6)


def test_prospective_patch_rigid_motion_preserves_its_frame() -> None:
    ca, selected, heavy, sites = antigen_patch()
    rotation = Rotation.from_rotvec([0.2, -0.4, 0.3]).as_matrix()
    translation = np.array([3.0, 47.0, -9.0])
    first = prospective_patch_frame(ca, selected, heavy, sites, landmarks=[0, 1, 2, 3])
    moved = prospective_patch_frame(
        ca @ rotation + translation,
        selected,
        heavy @ rotation + translation,
        sites @ rotation + translation,
        landmarks=[0, 1, 2, 3],
    )
    np.testing.assert_allclose(
        moved["origin"], np.asarray(first["origin"]) @ rotation + translation, atol=1e-8
    )
    for key in ("normal", "tangent"):
        np.testing.assert_allclose(moved[key], np.asarray(first[key]) @ rotation, atol=1e-8)
    assert first["landmarks"] == moved["landmarks"]


def legacy_reference() -> Any:
    path = Path(__file__).parent / "fixtures/legacy_placement_reference.py"
    spec = importlib.util.spec_from_file_location("legacy_placement_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bounds", [[0, 20], [30, 90], [135, 180]])
@pytest.mark.parametrize("seed", [0, 31, 2026])
def test_legacy_seeded_placement_exactly_matches_pr15(bounds: list[float], seed: int) -> None:
    old = legacy_reference()
    coordinates, framework, cdr = scaffold()
    shape = scaffold_shape(coordinates, framework, cdr)
    old_shape = old.scaffold_shape(coordinates, framework, cdr)
    expected, old_record = old.placement(patch(), old_shape, bounds, seed, 6.0)
    observed, record = placement(patch(), shape, bounds, seed, 6.0)
    np.testing.assert_array_equal(observed, expected)
    for key in ("requested_deg", "azimuth_deg", "roll_deg", "roll_attempts"):
        assert record[key] == old_record[key]
    np.testing.assert_array_equal(record["ray"], old_record["ray"])


def test_legacy_collision_roll_retry_keeps_original_seeded_coordinates() -> None:
    old = legacy_reference()
    coordinates, framework, cdr = scaffold()
    shape = scaffold_shape(coordinates, framework, cdr)
    old_shape = old.scaffold_shape(coordinates, framework, cdr)
    first, _ = old.placement(patch(), old_shape, [0, 20], 31, 6.0)
    targets = np.vstack((first[0], [100, 100, 100], [100, 110, 100], [110, 100, 110]))
    expected, old_record = old.placement(
        patch(), old_shape, [0, 20], 31, 6.0, target_anchors=targets
    )
    observed, record = placement(
        patch(), shape, [0, 20], 31, 6.0, target_anchors=targets
    )
    assert old_record["roll_attempts"] > 1
    np.testing.assert_array_equal(observed, expected)
    assert record["roll_attempts"] == old_record["roll_attempts"]
    assert record["roll_deg"] == old_record["roll_deg"]
