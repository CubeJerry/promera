"""Exercise sparse guidance with NumPy features and real geometry/encoding.

Only the runtime chain container is represented by a SimpleNamespace. Placement,
framework mapping, pair selection and distance-bin encoding are the production
implementations. No model inference or biological target is involved.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from experiments.pose_guidance.pose_geometry import framework_map
from promera_angles.conditioning import condition
from promera_angles.footprint import elevation
from promera_angles.geometry import scaffold_shape

PATTERN = "AA<cdrh1>SS<cdrh2>GG<cdrh3>TT"
SEQUENCE = "AAKSSNGGPTT"
CODES = {
    "A": "ALA",
    "S": "SER",
    "G": "GLY",
    "T": "THR",
    "K": "LYS",
    "N": "ASN",
    "P": "PRO",
}


def fixture(
    *, native_distances: bool = True
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    coordinates = np.array(
        [
            [-4, -4, 0],
            [4, -4, 0],
            [-4, 4, 0],
            [3, 3, -5],
            [-8, 0, -4],
            [0, -8, -4],
            [8, 0, -8],
            [0, 8, -8],
        ],
        dtype=float,
    )
    keys = [
        ["Left", 1],
        ["Right", 1],
        ["Left", 2],
        ["Right", 2],
        ["Left", 3],
        ["Right", 3],
        ["Left", 4],
        ["Right", 4],
    ]
    frame = {
        "origin": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "tangent": [1.0, 0.0, 0.0],
        "landmarks": [0, 1, 2, 3],
        "neighbourhood": list(range(8)),
        "normal_spread_deg": 0.0,
    }
    tilt = np.radians(20.0)
    legacy_frame = dict(
        frame,
        normal=[float(np.sin(tilt)), 0.0, float(np.cos(tilt))],
        tangent=[float(np.cos(tilt)), 0.0, float(-np.sin(tilt))],
    )
    binder_ca = np.array(
        [
            [-3, -3, 12.0],
            [3, -3, 13.0],
            [-2, 0, 0.4],
            [-4, 1, 14.2],
            [3, 3, 12.5],
            [0.2, 1, -0.3],
            [-2, 4, 15.0],
            [4, 0, 14.6],
            [2, -0.8, 0.8],
            [-1, -4, 13.8],
            [2, 4, 15.7],
        ],
        dtype=float,
    )
    framework, cdr = framework_map(SEQUENCE, PATTERN)
    plan = {
        "experiment_definition": "surface_elevation_v1",
        "coords": coordinates.tolist(),
        "keys": keys,
        "selected": [0, 3],
        "sequences": {"Left": "AAAA", "Right": "SSSS"},
        "binder_chain": "VHH",
        "framework": PATTERN,
        "frame": frame,
        "legacy_frame": legacy_frame,
        "shape": scaffold_shape(binder_ca, framework, cdr),
    }
    sequences = {"Right": "SSSS", "VHH": SEQUENCE, "Left": "AAAA"}
    chains = {
        name: SimpleNamespace(rname=[CODES[aa] for aa in sequence])
        for name, sequence in sequences.items()
    }
    starts, count = {}, 0
    for name, chain in chains.items():
        starts[name] = count
        count += len(chain.rname)
    flags = np.zeros(count, dtype=int)
    for i in plan["selected"]:
        chain, position = keys[i]
        flags[starts[chain] + position - 1] = 1
    features: dict[str, Any] = {
        "struct": SimpleNamespace(chains=chains),
        "restype": np.arange(count, dtype=int),
        "is_epitope": flags,
        "unchanged_payload": np.arange(count * 3).reshape(count, 3),
    }
    if native_distances:
        mask = np.eye(count, dtype=bool)
        # Existing antigen-internal and framework-internal conditioning.
        old_pairs = [
            (starts["Right"], starts["Right"] + 1),
            (starts["Left"] + 1, starts["Left"] + 2),
            (starts["VHH"] + framework[0], starts["VHH"] + framework[1]),
        ]
        for first, second in old_pairs:
            mask[first, second] = mask[second, first] = True
        features["distogram_mask"] = mask
        features["distogram_emb"] = np.where(mask, 7, 0).astype(np.int64)
    return features, plan, starts


def snapshot(features: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in features.items()
    }


def assert_feature_noop(before: dict[str, Any], after: dict[str, Any]) -> None:
    assert set(before) == set(after)
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(after[key], value)
        else:
            assert after[key] is value


@pytest.mark.parametrize("native_distances", [False, True])
@pytest.mark.parametrize("definition", ["legacy_placement_ray", "surface_elevation"])
def test_ordinary_targeted_condition_is_a_complete_feature_noop(
    native_distances: bool, definition: str
) -> None:
    features, plan, _ = fixture(native_distances=native_distances)
    before = snapshot(features)
    record = condition(features, plan, 17, None, 6.0, guidance_definition=definition)
    assert_feature_noop(before, features)
    assert record["guidance_definition"] == "ordinary"
    assert record["generation_only"] is True
    assert record["guide_seed"] == 17
    assert record["cdr_positions"] == [3, 6, 9]
    assert record["framework_positions"] == [1, 2, 4, 5, 7, 8, 10, 11]
    assert not record.get("pairs")
    assert record.get("requested_elevation_deg") is None


def test_surface_guide_encodes_exact_symmetric_pairs_with_interleaved_target_chains() -> (
    None
):
    features, plan, starts = fixture()
    before = snapshot(features)
    record = condition(
        features, plan, 31, [70, 90], 6.0, guidance_definition="surface_elevation"
    )
    assert record.get("record_status") != "rejected_guide"
    assert record["generation_only"] is True
    assert record["guidance_definition"] == "surface_elevation"
    assert record["requested_guidance_pairs"] == 16
    assert record["informative_guidance_pairs"] == 16
    assert record["cdr_positions"] == [3, 6, 9]
    assert len(record["pairs"]) == 16
    assert record["distance_encoding"] == "promera_51_bins_1_to_50_with_overflow"
    assert np.count_nonzero(features["distogram_mask"] ^ before["distogram_mask"]) == 32
    np.testing.assert_array_equal(
        features["distogram_mask"], features["distogram_mask"].T
    )
    np.testing.assert_array_equal(
        features["distogram_emb"], features["distogram_emb"].T
    )
    np.testing.assert_array_equal(
        features["distogram_emb"][before["distogram_mask"]],
        before["distogram_emb"][before["distogram_mask"]],
    )
    for key in ("restype", "is_epitope", "unchanged_payload"):
        np.testing.assert_array_equal(features[key], before[key])
    assert features["struct"] is before["struct"]
    marks = plan["frame"]["landmarks"]
    framework, _ = framework_map(SEQUENCE, PATTERN)
    target_tokens = {
        starts[plan["keys"][i][0]] + plan["keys"][i][1] - 1: plan["keys"][i]
        for i in marks
    }
    binder_tokens = {
        starts["VHH"] + framework[i]: ["VHH", framework[i] + 1]
        for i in plan["shape"]["fixed_indices"]
    }
    assert {(p["target_token"], p["binder_token"]) for p in record["pairs"]} == {
        (target, binder) for target in target_tokens for binder in binder_tokens
    }
    target_xyz = np.asarray(record["target_landmark_xyz"])
    binder_xyz = np.asarray(record["framework_landmark_xyz"])
    t_index = {token: i for i, token in enumerate(target_tokens)}
    b_index = {token: i for i, token in enumerate(binder_tokens)}
    for pair in record["pairs"]:
        target, binder = pair["target_token"], pair["binder_token"]
        assert pair["target_residue"] == target_tokens[target]
        assert pair["binder_residue"] == binder_tokens[binder]
        assert pair["distance_overflow"] is False
        measured = np.linalg.norm(
            target_xyz[t_index[target]] - binder_xyz[b_index[binder]]
        )
        assert pair["distance_angstrom"] == pytest.approx(measured, abs=1e-10)
        assert pair["bin"] == int(np.sum(measured > np.linspace(1, 50, 50)))
        assert features["distogram_mask"][target, binder]
        assert features["distogram_emb"][target, binder] == pair["bin"]
    placed = np.asarray(record["scaffold_ca_xyz"])
    fw = np.asarray(record["scaffold_framework_indices"], dtype=int)
    cdr = np.asarray(record["scaffold_cdr_indices"], dtype=int)
    observed = elevation(
        placed[fw].mean(0) - placed[cdr].mean(0), plan["frame"]["normal"]
    )
    assert 70 <= record["requested_elevation_deg"] <= 90
    assert observed == pytest.approx(record["requested_elevation_deg"], abs=2e-6)
    assert observed == pytest.approx(
        record["realized_scaffold_elevation_deg"], abs=2e-6
    )
    np.testing.assert_allclose(placed[cdr].mean(0), [0, 0, 6], atol=1e-9)


def test_surface_guide_can_add_features_when_no_native_distogram_is_present() -> None:
    features, plan, _ = fixture(native_distances=False)
    epitope = features["is_epitope"].copy()
    record = condition(
        features, plan, 31, [70, 90], 6.0, guidance_definition="surface_elevation"
    )
    assert len(record["pairs"]) == 16
    assert features["distogram_mask"].sum() == 32
    np.testing.assert_array_equal(features["is_epitope"], epitope)


def test_default_and_explicit_legacy_modes_are_identical_and_keep_tilt_semantics() -> (
    None
):
    default_features, plan, _ = fixture()
    explicit_features, _, _ = fixture()
    first = condition(default_features, plan, 31, [0, 20], 6.0)
    explicit = condition(
        explicit_features,
        plan,
        31,
        [0, 20],
        6.0,
        guidance_definition="legacy_placement_ray",
    )
    assert first["guidance_definition"] == "legacy_placement_ray"
    assert first.get("requested_elevation_deg") is None
    assert 0 <= first["requested_deg"] <= 20
    for key in ("distogram_mask", "distogram_emb"):
        np.testing.assert_array_equal(default_features[key], explicit_features[key])
    np.testing.assert_array_equal(
        first["framework_landmark_xyz"], explicit["framework_landmark_xyz"]
    )
    assert first["pairs"] == explicit["pairs"]
    placed = np.asarray(first["scaffold_ca_xyz"])
    shape = plan["shape"]
    axis = placed[shape["framework_indices"]].mean(0) - placed[
        shape["cdr_indices"]
    ].mean(0)
    old_normal_angle = elevation(axis, plan["legacy_frame"]["normal"])
    new_normal_angle = elevation(axis, plan["frame"]["normal"])
    assert first["legacy_frame_scaffold_elevation_deg"] == pytest.approx(
        old_normal_angle, abs=2e-6
    )
    assert old_normal_angle == pytest.approx(90 - first["requested_deg"], abs=2e-6)
    assert first["realized_scaffold_elevation_deg"] == pytest.approx(
        new_normal_angle, abs=2e-6
    )
    assert first["prospective_patch_frame"] == plan["frame"]


@pytest.mark.parametrize("native_distances", [False, True])
def test_missing_legacy_frame_preserves_requested_parent_without_adding_guidance(
    native_distances: bool,
) -> None:
    features, plan, _ = fixture(native_distances=native_distances)
    plan["legacy_frame"] = None
    plan["legacy_frame_error"] = "legacy frame is unsupported for this patch"
    before = snapshot(features)
    record = condition(features, plan, 123, [0, 20], 6.0)
    assert record["record_status"] == "rejected_guide"
    assert record["reason"] == plan["legacy_frame_error"]
    assert record["guide_seed"] == 123
    assert record["cdr_positions"] == [3, 6, 9]
    assert record["generation_only"] is True
    assert record["guidance_definition"] == "legacy_placement_ray"
    assert not record.get("pairs")
    if native_distances:
        assert_feature_noop(before, features)
    else:
        assert not features["distogram_mask"].any()
        assert not features["distogram_emb"].any()
        size = len(features["restype"])
        assert features["distogram_mask"].shape == (size, size)
        assert features["distogram_emb"].shape == (size, size)
        for key in before:
            if isinstance(before[key], np.ndarray):
                np.testing.assert_array_equal(features[key], before[key])


def test_ordinary_control_does_not_require_a_legacy_frame() -> None:
    features, plan, _ = fixture()
    plan["legacy_frame"] = None
    before = snapshot(features)
    record = condition(features, plan, 17, None, 6.0)
    assert record["guidance_definition"] == "ordinary"
    assert "record_status" not in record
    assert_feature_noop(before, features)


@pytest.mark.parametrize(
    "error", ["missing_epitope", "extra_epitope", "changed_target", "token_count"]
)
def test_invalid_runtime_correspondence_fails_before_feature_mutation(
    error: str,
) -> None:
    features, plan, starts = fixture()
    if error == "missing_epitope":
        features["is_epitope"][:] = 0
        expected = "epitope flags"
    elif error == "extra_epitope":
        features["is_epitope"][starts["Left"] + 3] = 1
        expected = "epitope flags"
    elif error == "changed_target":
        features["struct"].chains["Right"].rname[0] = "GLY"
        expected = "target sequence changed"
    else:
        features["restype"] = features["restype"][:-1]
        expected = "one token per protein residue"
    before = snapshot(features)
    with pytest.raises(ValueError, match=expected):
        condition(
            features, plan, 31, [70, 90], 6.0, guidance_definition="surface_elevation"
        )
    assert_feature_noop(before, features)


def test_unknown_definition_is_rejected_without_adding_features() -> None:
    features, plan, _ = fixture()
    before = snapshot(features)
    with pytest.raises(ValueError, match="unknown guidance definition"):
        condition(
            features, plan, 31, [0, 20], 6.0, guidance_definition="ambiguous_angle"
        )
    assert_feature_noop(before, features)


def test_existing_cross_chain_condition_is_not_overwritten_by_an_angle_guide() -> None:
    features, plan, starts = fixture()
    framework, _ = framework_map(SEQUENCE, PATTERN)
    mark = plan["keys"][plan["frame"]["landmarks"][0]]
    target = starts[mark[0]] + mark[1] - 1
    binder = starts["VHH"] + framework[plan["shape"]["fixed_indices"][0]]
    features["distogram_mask"][target, binder] = True
    features["distogram_mask"][binder, target] = True
    features["distogram_emb"][target, binder] = 49
    features["distogram_emb"][binder, target] = 49
    before = snapshot(features)
    with pytest.raises(ValueError, match="overlap"):
        condition(
            features, plan, 31, [70, 90], 6.0, guidance_definition="surface_elevation"
        )
    assert_feature_noop(before, features)


def test_scaffold_coordinate_verification_precedes_distance_feature_mutation() -> None:
    features, plan, _ = fixture()
    plan = copy.deepcopy(plan)
    coordinates = np.asarray(plan["shape"]["scaffold_ca_local"])
    coordinates[plan["shape"]["framework_indices"]] += [4.0, 0.0, 0.0]
    plan["shape"]["scaffold_ca_local"] = coordinates.tolist()
    before = snapshot(features)
    with pytest.raises(ValueError):
        condition(
            features, plan, 31, [70, 90], 6.0, guidance_definition="surface_elevation"
        )
    assert_feature_noop(before, features)
