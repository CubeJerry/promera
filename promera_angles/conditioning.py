"""Generation-only sparse distance features, independent of the model runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.pose_guidance.pose_geometry import (
    AnchorClash,
    chain_sequence,
    cross_distogram,
    framework_map,
    merge_distogram,
)

from .geometry import elevation_placement, placement, signed_elevation
from .gfp_pair_ablation import GFP_PAIR_COUNTS, retain_gfp_pairs


def condition(
    feats: dict[str, Any],
    plan: dict[str, Any],
    seed: int,
    interval: Any,
    standoff: float,
    *,
    guidance_definition: str = "legacy_placement_ray",
    cdr_lateral_offset: Any = (0.0, 0.0),
    gfp_pair_arm: str | None = None,
) -> dict[str, Any]:
    """Add native sparse distance pairs after verifying the transformed scaffold."""
    struct = feats["struct"]
    offsets, n = {}, 0
    for name, chain in struct.chains.items():
        offsets[name] = n
        n += len(chain.rname)
    if n != len(feats["restype"]):
        raise ValueError("angular landmarks require one token per protein residue")
    keys = plan["keys"]

    def token(key):
        chain, pos = key
        return offsets[chain] + pos - 1

    expected = {token(keys[i]) for i in plan["selected"]}
    actual = set()
    for name, sequence in plan["sequences"].items():
        if chain_sequence(struct.chains[name]) != sequence:
            raise ValueError(f"runtime target sequence changed on {name}")
        actual.update(
            offsets[name] + int(i)
            for i in np.flatnonzero(
                feats["is_epitope"][offsets[name] : offsets[name] + len(sequence)]
            )
        )
    if expected != actual:
        raise ValueError(
            "native epitope flags disagree with the plan; check the installed multichain adapter"
        )
    binder = plan["binder_chain"]
    fw, cdr = framework_map(chain_sequence(struct.chains[binder]), plan["framework"])
    record = dict(
        cdr_positions=[p + 1 for p in cdr],
        framework_positions=[p + 1 for p in fw],
        guide_seed=seed,
        guidance_definition="ordinary" if interval is None else guidance_definition,
        generation_only=True,
    )
    if gfp_pair_arm is not None:
        if not plan.get("gfp_pair_ablation") or gfp_pair_arm not in GFP_PAIR_COUNTS:
            raise ValueError("GFP pair subset requires a prepared GFP benchmark")
        record["nominal_guidance_pairs"] = GFP_PAIR_COUNTS[gfp_pair_arm]
    if interval is None:
        if gfp_pair_arm is not None:
            if GFP_PAIR_COUNTS[gfp_pair_arm] != 0:
                raise ValueError("only the GFP ordinary control has no interval")
            record.update(requested_guidance_pairs=0, informative_guidance_pairs=0)
        return record
    if guidance_definition not in ("legacy_placement_ray", "surface_elevation"):
        raise ValueError(f"unknown guidance definition: {guidance_definition}")
    frame = plan["frame"]
    if (
        guidance_definition == "legacy_placement_ray"
        and plan.get("experiment_definition") == "surface_elevation_v1"
    ):
        frame = plan.get("legacy_frame")
        if frame is None:
            return rejected(
                feats,
                n,
                record,
                plan.get("legacy_frame_error", "legacy patch frame unavailable"),
            )
    marks = frame["landmarks"]
    shape = plan["shape"]
    target_marks = np.asarray(plan["coords"])[marks]
    try:
        if guidance_definition == "surface_elevation":
            coords, guide = elevation_placement(
                frame,
                shape,
                interval,
                seed,
                standoff,
                target_anchors=target_marks,
                cdr_lateral_offset=cdr_lateral_offset,
            )
        else:
            coords, guide = placement(
                frame, shape, interval, seed, standoff, target_anchors=target_marks
            )
            if "scaffold_ca_xyz" in guide:
                placed = np.asarray(guide["scaffold_ca_xyz"])
                guide["legacy_frame_scaffold_elevation_deg"] = guide[
                    "realized_scaffold_elevation_deg"
                ]
                guide["realized_scaffold_elevation_deg"] = signed_elevation(
                    placed[shape["framework_indices"]],
                    placed[shape["cdr_indices"]],
                    plan["frame"]["normal"],
                )
                guide["prospective_patch_frame"] = plan["frame"]
        bins, mask, pairs = cross_distogram(
            n,
            [token(keys[i]) for i in marks],
            [offsets[binder] + fw[i] for i in shape["fixed_indices"]],
            target_marks,
            coords,
            16,
        )
    except AnchorClash as exc:
        return rejected(feats, n, record, str(exc))
    if gfp_pair_arm is not None:
        pairs = retain_gfp_pairs(bins, mask, pairs, gfp_pair_arm)
    target_tokens = {token(keys[i]): keys[i] for i in marks}
    binder_tokens = {offsets[binder] + fw[i]: fw[i] + 1 for i in shape["fixed_indices"]}
    for pair in pairs:
        pair["target_residue"] = list(target_tokens[pair["target_token"]])
        pair["binder_residue"] = [binder, binder_tokens[pair["binder_token"]]]
    guide.update(
        target_landmark_xyz=target_marks.tolist(),
        framework_landmark_xyz=coords.tolist(),
        distance_encoding="promera_51_bins_1_to_50_with_overflow",
        informative_guidance_pairs=sum(p["finite_distance_information"] for p in pairs),
        requested_guidance_pairs=len(pairs),
    )
    merge_distogram(feats, bins, mask)
    return dict(record, **guide, pairs=pairs)


def rejected(
    feats: dict[str, Any], n: int, record: dict[str, Any], reason: str
) -> dict[str, Any]:
    """Keep rejected parent metadata and collate keys, without adding a restraint."""
    if "distogram_mask" not in feats:
        feats["distogram_mask"] = np.zeros((n, n), dtype=bool)
        feats["distogram_emb"] = np.zeros((n, n), dtype=np.int64)
    return dict(record, record_status="rejected_guide", reason=reason)
