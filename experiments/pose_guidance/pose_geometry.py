"""CPU-only geometry for the isolated cross-chain conditioning experiment."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np


def digest(path: str | Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def framework_map(sequence: str, pattern: str) -> tuple[list[int], list[int]]:
    """Return zero-based fixed/CDR ordinals; reject ambiguous framework matches."""
    tags = re.findall(r'<cdrh[123]>', pattern)
    if tags != ['<cdrh1>', '<cdrh2>', '<cdrh3>']:
        raise ValueError('framework must contain cdrh1, cdrh2, cdrh3 once, in order')
    parts = re.split(r'<cdrh[123]>', pattern)
    # Enumerate segment matches, rather than silently choosing a repeated motif.
    solutions: list[list[tuple[int, int]]] = []

    def visit(part: int, start: int, spans: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if part == len(parts):
            if start == len(sequence):
                solutions.append(spans)
            return
        text = parts[part]
        if not text:
            raise ValueError('framework segments must be nonempty')
        candidates = [0] if part == 0 else range(start + 1, len(sequence))
        for pos in candidates:
            if sequence.startswith(text, pos):
                visit(part + 1, pos + len(text), spans + [(pos, pos + len(text))])

    visit(0, 0, [])
    if len(solutions) != 1:
        raise ValueError('source/framework mapping is missing or ambiguous; use the matching framework')
    fixed = [i for a, b in solutions[0] for i in range(a, b)]
    fixed_set = set(fixed)
    return fixed, [i for i in range(len(sequence)) if i not in fixed_set]


def exact_crop(query: str, source: str) -> list[int]:
    """Match an identical chain or unambiguous contiguous crop, not author numbers."""
    starts = [i for i in range(len(source) - len(query) + 1) if source.startswith(query, i)]
    if not query or len(starts) != 1:
        raise ValueError('target chain must match the guide exactly or be a unique contiguous crop')
    return list(range(starts[0], starts[0] + len(query)))


def xyz(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or not len(arr) or not np.isfinite(arr).all():
        raise ValueError('expected finite nonempty N x 3 coordinates')
    return arr


def farthest_points(coords: Any, count: int = 4) -> list[int]:
    arr = xyz(coords)
    if len(arr) < count:
        raise ValueError(f'need at least {count} coordinate-bearing landmarks')
    selected = [int(np.argmin(np.sum((arr - arr.mean(0)) ** 2, axis=1)))]
    while len(selected) < count:
        d2 = np.sum((arr[:, None] - arr[selected][None, :]) ** 2, axis=-1).min(1)
        d2[selected] = -1
        selected.append(int(np.argmax(d2)))
    if np.linalg.matrix_rank(arr[selected] - arr[selected].mean(0), tol=1e-6) < 2:
        raise ValueError('landmarks are collinear; orientation is not identifiable')
    return selected


def jitter(coords: Any, seed: int, degrees: float, translation: float) -> np.ndarray:
    """Rigid rotation about the framework centroid and translation in a ball.

    The angle is uniform on [-degrees, degrees], not uniform on SO(3).
    This changes the guide placement, not individual atom coordinates.
    """
    arr = xyz(coords)
    if not np.isfinite([degrees, translation]).all() or not 0 <= degrees <= 180 or translation < 0:
        raise ValueError('invalid rigid-jitter bounds')
    if degrees == 0 and translation == 0:
        return arr.copy()
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(-degrees, degrees))
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    shift = direction * translation * rng.random() ** (1 / 3)
    return (arr - arr.mean(0)) @ rotation.T + arr.mean(0) + shift


def stratified_cone_angles(
    backbone_index: int, count: int, replicate_seed: int,
    min_degrees: float, max_degrees: float,
) -> tuple[float, float, int, int]:
    """Sample reproducibly over the whole cone, not a lucky cluster of draws.

    Independently permute equal-solid-angle polar and equal-azimuth strata
    across the planned backbones within each replicate. The marginals retain
    the original uniform-in-solid-angle policy, while a full 64-backbone
    replicate necessarily explores the available radial and azimuth intervals.
    """
    if count < 1 or not 0 <= backbone_index < count:
        raise ValueError('cone stratum index is outside requested backbone count')
    if not (np.isfinite([min_degrees, max_degrees]).all()
            and 0 <= min_degrees <= max_degrees <= 180):
        raise ValueError('invalid stratified cone angle interval')
    rng = np.random.default_rng(int(replicate_seed))
    polar = rng.permutation(count)
    azimuth = rng.permutation(count)
    polar_quantile = (int(polar[backbone_index]) + float(rng.random(count)[backbone_index])) / count
    azimuth_quantile = (int(azimuth[backbone_index]) + float(rng.random(count)[backbone_index])) / count
    c_low = float(np.cos(np.deg2rad(min_degrees)))
    c_high = float(np.cos(np.deg2rad(max_degrees)))
    angle = float(np.rad2deg(np.arccos(np.clip(
        c_low - polar_quantile * (c_low - c_high), -1.0, 1.0
    ))))
    return angle, 360.0 * azimuth_quantile, int(polar[backbone_index]), int(azimuth[backbone_index])


def cone_placement(
    coords: Any,
    source_axis: Any,
    seed: int,
    min_degrees: float,
    max_degrees: float,
    translation: float = 0.0,
    pivot: Any | None = None,
    *,
    sampled_angle_degrees: float | None = None,
    sampled_azimuth_degrees: float | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Sample a target-relative approach cone around the selected source axis.

    Polar deviation is uniformly distributed in solid angle, not in degrees;
    azimuth is uniform. No atom-wise noise or framework-template conditioning.
    The guide is a *soft* conditioning input, never a guarantee of output angle.
    """
    arr = xyz(coords)
    axis = np.asarray(source_axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-6:
        raise ValueError("source approach axis must be finite and nonzero")
    if not (
        np.isfinite([min_degrees, max_degrees, translation]).all()
        and 0 <= min_degrees <= max_degrees <= 180
        and translation >= 0
    ):
        raise ValueError("invalid approach-angle interval or translation bounds")
    source = axis / np.linalg.norm(axis)
    rng = np.random.default_rng(seed)
    cos_min = np.cos(np.deg2rad(max_degrees))
    cos_max = np.cos(np.deg2rad(min_degrees))
    if (sampled_angle_degrees is None) != (sampled_azimuth_degrees is None):
        raise ValueError('both stratified angle and azimuth must be supplied together')
    if sampled_angle_degrees is None:
        theta = float(np.arccos(np.clip(rng.uniform(cos_min, cos_max), -1, 1)))
        azimuth = float(rng.uniform(0, 2 * np.pi))
        policy = "uniform_solid_angle"
    else:
        if not (
            np.isfinite([sampled_angle_degrees, sampled_azimuth_degrees]).all()
            and min_degrees - 1e-9 <= sampled_angle_degrees <= max_degrees + 1e-9
            and 0 <= sampled_azimuth_degrees < 360
        ):
            raise ValueError('stratified cone sample outside requested interval')
        theta = float(np.deg2rad(sampled_angle_degrees))
        azimuth = float(np.deg2rad(sampled_azimuth_degrees))
        policy = "stratified_solid_angle"
    basis = np.zeros(3)
    basis[int(np.argmin(np.abs(source)))] = 1.0
    u = np.cross(source, basis)
    u /= np.linalg.norm(u)
    v = np.cross(source, u)
    desired = (
        np.cos(theta) * source
        + np.sin(theta) * (np.cos(azimuth) * u + np.sin(azimuth) * v)
    )
    rotation_axis = np.cross(source, desired)
    magnitude = np.linalg.norm(rotation_axis)
    rotation_axis = u if magnitude < 1e-8 else rotation_axis / magnitude
    x, y, z = rotation_axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rotation = np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
    shift = np.zeros(3)
    if translation:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        shift = direction * translation * rng.random() ** (1 / 3)
    center = arr.mean(0) if pivot is None else np.asarray(pivot, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("pose rotation pivot must be a finite xyz point")
    transformed = (arr - center) @ rotation.T + center + shift
    # Record the actual geometric deviation, not merely the requested draw.
    achieved = np.rad2deg(
        np.arccos(np.clip(source @ (rotation @ source), -1, 1))
    )
    return transformed, {
        "guide_approach_deviation_degrees": float(achieved),
        "guide_azimuth_degrees": float(np.rad2deg(azimuth)),
        "guide_translation_angstrom": float(np.linalg.norm(shift)),
        "guide_sampling_policy": policy,
        "guide_pivot": "framework_centroid" if pivot is None else "antigen_patch_centroid",
    }


def cross_distogram(
    n_tokens: int, target_tokens: list[int], binder_tokens: list[int],
    target_xyz: Any, binder_xyz: Any, count: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Encode cross-chain distances using Promera's exact 51-bin convention.

    The model is trained with torch.linspace(1, 50, 50) and a 51-entry
    embedding table. Consequently bin 50 means *distance > 50 Angstrom*.
    It is a valid training bin, but gives no finite-distance resolution.
    Preserve every requested pair and record which guides overflow.
    Reject a guide with inadequate finite cross-chain coverage instead of
    silently discarding long-distance constraints or clipping coordinates.
    """
    target, binder = xyz(target_xyz), xyz(binder_xyz)
    if len(target_tokens) != 4 or len(binder_tokens) != 4 or target.shape != (4, 3) or binder.shape != (4, 3):
        raise ValueError('this pilot requires four target and four binder landmarks')
    tokens = target_tokens + binder_tokens
    if len(set(tokens)) != 8 or any(i < 0 or i >= n_tokens for i in tokens):
        raise ValueError('landmark tokens must be distinct and inside the unpadded sequence')
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 0 <= count <= 16:
        raise ValueError('pair count must be an integer between 0 and 16')
    mask = np.zeros((n_tokens, n_tokens), dtype=bool)
    bins = np.zeros((n_tokens, n_tokens), dtype=np.int64)
    records = []
    boundaries = np.linspace(1, 50, 50)
    order = [(i, (i + offset) % 4) for offset in range(4) for i in range(4)]
    for ti, bi in order[:count]:
        distance = float(np.linalg.norm(target[ti] - binder[bi]))
        if not distance > 1.0:
            raise ValueError(
                f'cross-chain anchor C-alpha separation {distance:.3f} A is <= 1 A; '
                'guide contains a physically implausible clash'
            )
        t, b = target_tokens[ti], binder_tokens[bi]
        bucket = int(np.sum(distance > boundaries))
        overflow = bucket == 50
        mask[t, b] = mask[b, t] = True
        bins[t, b] = bins[b, t] = bucket
        records.append({
            'target_token': t,
            'binder_token': b,
            'distance_angstrom': distance,
            'bin': bucket,
            'distance_overflow': overflow,
            'finite_distance_information': not overflow,
        })

    # Bin 50 (>50 A) is valid model input, but cannot discriminate e.g. 52
    # from 80 A. A guidance set dominated by overflow pairs would be
    # scientifically uninterpretable, so require adequate non-overflow
    # coverage on both the antigen and framework sides.
    informative = [r for r in records if r['finite_distance_information']]
    if count:
        minimum = max(1, (count + 1) // 2)
        minimum_axes = min(3, count)
        n_target = len({r['target_token'] for r in informative})
        n_binder = len({r['binder_token'] for r in informative})
        if len(informative) < minimum or min(n_target, n_binder) < minimum_axes:
            raise ValueError(
                f'insufficient informative cross-chain anchors: {len(informative)}/{count} '
                f'finite-distance pairs (required >= {minimum}); '
                f'{n_target} antigen and {n_binder} binder landmarks '
                f'(required >= {minimum_axes} each); '
                f'{count - len(informative)} pairs are in the trained >50 A overflow bin. '
                'Try a narrower approach-angle interval or a different source pose.'
            )
    return bins, mask, records


def merge_distogram(feats: dict[str, Any], bins: np.ndarray, mask: np.ndarray) -> None:
    """Zero guidance is a feature-level no-op; never add integer bin indices."""
    if not mask.any():
        return
    if 'distogram_mask' in feats:
        old_mask = np.asarray(feats['distogram_mask'], dtype=bool)
        old_bins = np.asarray(feats['distogram_emb'])
        if old_mask.shape != mask.shape or old_bins.shape != bins.shape:
            raise ValueError('existing distogram has an incompatible shape')
        if np.any(old_mask & mask):
            raise ValueError('pose anchors overlap an existing conditioned pair')
        feats['distogram_emb'] = np.where(mask, bins, old_bins)
        feats['distogram_mask'] = old_mask | mask
    else:
        feats['distogram_emb'], feats['distogram_mask'] = bins, mask


def align(moving: Any, reference: Any) -> tuple[np.ndarray, np.ndarray]:
    moving, reference = xyz(moving), xyz(reference)
    if moving.shape != reference.shape or len(moving) < 3:
        raise ValueError('rigid alignment requires at least three matched coordinates')
    if min(np.linalg.matrix_rank(a - a.mean(0), tol=1e-6) for a in (moving, reference)) < 2:
        raise ValueError('cannot align a collinear structure')
    u, _, vt = np.linalg.svd((moving - moving.mean(0)).T @ (reference - reference.mean(0)))
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    return rotation, reference.mean(0) - moving.mean(0) @ rotation


def rmsd(first: Any, second: Any) -> float:
    a, b = xyz(first), xyz(second)
    if a.shape != b.shape:
        raise ValueError('RMSD requires one-to-one coordinates')
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def pose_metrics(reference_target: Any, reference_fw: Any, observed_target: Any,
                 observed_fw: Any, reference_axis: Any,
                 target_patch_center: Any | None = None) -> dict[str, float]:
    """Separate internal shape, body rotation, approach direction and displacement."""
    rt, rf, ot, of = map(xyz, (reference_target, reference_fw, observed_target, observed_fw))
    rotation, shift = align(ot, rt)
    fw = of @ rotation + shift
    body_rotation, body_shift = align(rf, fw)
    axis = np.asarray(reference_axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-6:
        raise ValueError('reference approach axis is undefined')
    axis = axis / np.linalg.norm(axis)
    transported = axis @ body_rotation
    basis = np.zeros(3)
    basis[int(np.argmin(np.abs(axis)))] = 1.0
    u = np.cross(axis, basis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    tilt = float(np.rad2deg(np.arccos(np.clip(axis @ transported, -1, 1))))
    # Azimuth has no geometric meaning at zero tilt; do not invent one.
    azimuth = (float(np.rad2deg(np.arctan2(transported @ v, transported @ u)) % 360)
               if tilt > 1e-5 else None)
    radial_change = None
    if target_patch_center is not None:
        pivot = np.asarray(target_patch_center, dtype=float)
        if pivot.shape != (3,) or not np.isfinite(pivot).all():
            raise ValueError('target patch center is not finite xyz')
        radial_change = float(abs(
            np.linalg.norm(fw.mean(0) - pivot)
            - np.linalg.norm(rf.mean(0) - pivot)
        ))
    return {
        'patch_radius_change_angstrom': radial_change,
        'target_fit_rmsd': rmsd(ot @ rotation + shift, rt),
        'source_framework_rmsd': rmsd(fw, rf),
        'framework_internal_rmsd': rmsd(rf @ body_rotation + body_shift, fw),
        'centroid_shift_angstrom': float(np.linalg.norm(fw.mean(0) - rf.mean(0))),
        'body_rotation_degrees': float(np.rad2deg(np.arccos(np.clip((np.trace(body_rotation) - 1) / 2, -1, 1)))),
        # Transport a fixed source-defined axis; do not redefine it from new CDRs.
        'approach_degrees': tilt,
        'approach_azimuth_degrees': azimuth,
    }


AA3 = dict(zip(
    'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),
    'ARNDCQEGHILKMFPSTWYV',
))
AA3.update({'UNK': 'X', 'MSE': 'M'})


def chain_sequence(chain: Any) -> str:
    return ''.join(AA3.get(str(name), 'X') for name in chain.rname)


def read_chains(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read the actual tinyprot chain/residue order used by this runtime."""
    from tinyprot.structure import Structure

    if Path(path).suffix.lower() not in {'.cif', '.mmcif'}:
        raise ValueError('use an mmCIF guide, not a PDB with potentially remapped chains')
    result = {}
    for name, chain in Structure.from_mmcif(str(path)).chains.items():
        coords = np.full((len(chain.rname), 3), np.nan)
        for index, names in enumerate(chain.aname):
            matches = np.flatnonzero(np.asarray(names).astype(str) == 'CA')
            if len(matches) == 1 and bool(chain.mask[index, matches[0]]):
                coords[index] = chain.coords[index, matches[0]]
        result[str(name)] = {'sequence': chain_sequence(chain), 'coords': coords,
                             'residue_ids': [str(x) for x in chain.ridx]}
    return result


def build_plan(chains: dict[str, Any], schema: dict[str, Any], pattern: str,
               binder_chain: str, guide_binder_chain: str, target_map: dict[str, str]) -> dict[str, Any]:
    """Build traceable ordinal mappings and four spatially distributed landmarks."""
    if guide_binder_chain not in chains or binder_chain in schema:
        raise ValueError('missing guide binder or binder identifier conflicts with target schema')
    if not schema or set(target_map) != set(schema) or len(set(target_map.values())) != len(target_map):
        raise ValueError('target map must map every target chain uniquely')
    if guide_binder_chain in target_map.values():
        raise ValueError('a source chain cannot be both binder and target')
    binder = chains[guide_binder_chain]
    fixed, cdr = framework_map(binder['sequence'], pattern)
    fw = xyz(binder['coords'][fixed])
    cdr_coords = xyz(binder['coords'][cdr])
    targets, labels, mappings, sequences = [], [], {}, {}
    for name, entry in schema.items():
        if entry.get('type') != 'protein' or not isinstance(entry.get('sequence'), str):
            raise ValueError('this experiment supports explicit protein target chains only')
        source = chains[target_map[name]]
        mapping = exact_crop(entry['sequence'], source['sequence'])
        mappings[name] = mapping
        sequences[name] = entry['sequence']
        for ordinal, source_index in enumerate(mapping):
            ca = source['coords'][source_index]
            if np.isfinite(ca).all():
                targets.append(ca)
                labels.append([name, ordinal])
    target = xyz(targets)
    # Choose a local patch around the residue nearest the source CDR centroid.
    center = target[np.argmin(np.linalg.norm(target - cdr_coords.mean(0), axis=1))]
    candidates = np.flatnonzero(np.linalg.norm(target - center, axis=1) <= 15.0)
    target_landmarks = [int(candidates[i]) for i in farthest_points(target[candidates])]
    # Exclude terminal three residues from the landmark pool, not from RMSD.
    pool = [i for i, ordinal in enumerate(fixed) if 3 <= ordinal < len(binder['sequence']) - 3]
    binder_landmarks = [pool[i] for i in farthest_points(fw[pool])]
    axis = cdr_coords.mean(0) - fw.mean(0)
    if np.linalg.norm(axis) < 1e-6:
        raise ValueError('source approach axis is undefined')
    cross_distogram(8, list(range(4)), list(range(4, 8)),
                    target[target_landmarks], fw[binder_landmarks], 16)
    return {
        'schema_version': 1, 'framework': pattern, 'binder_chain': binder_chain,
        'guide_binder_chain': guide_binder_chain, 'target_chain_map': target_map,
        'target_sequences': sequences, 'target_source_ordinals': mappings,
        'target_labels': labels, 'target_coords': target.tolist(),
        'source_framework_ordinals': fixed, 'source_framework_coords': fw.tolist(),
        'source_axis': axis.tolist(), 'target_landmarks': target_landmarks,
        'binder_landmarks': binder_landmarks,
        'indexing': 'zero-based sequence ordinals; never author residue numbers',
    }


def targeted_site_from_reference(
    chains: dict[str, Any], plan: dict[str, Any], cutoff_angstrom: float = 10.0,
    max_residues: int = 7,
) -> dict[str, Any]:
    """Derive a compact, site-only targeted comparator from the source CDRs.

    The CA-distance cutoff defines a *candidate pool*, never an unrestricted
    10-A shell of conditioning residues. Choose at most seven nearby antigen
    positions by CDR proximity (or another explicitly requested site size).
    This mirrors the scale of NOMINEE's usual Epitope Scan signatures while
    keeping the selection deterministic and shared by every targeted arm.
    Sequence ordinals are mapped to actual native template residue IDs later.
    """
    if not np.isfinite(cutoff_angstrom) or not 5 <= cutoff_angstrom <= 25:
        raise ValueError('site C-alpha cutoff must be 5-25 Angstrom')
    if isinstance(max_residues, bool) or not isinstance(max_residues, (int, np.integer)):
        raise ValueError('site-residues must be an integer')
    if not 4 <= int(max_residues) <= 20:
        raise ValueError('site-residues must be between 4 and 20')
    if len(plan['target_sequences']) != 1:
        raise ValueError(
            'targeted comparison currently supports one antigen chain; '
            'do not silently discard an interface spanning multiple chains'
        )
    chain_name = next(iter(plan['target_sequences']))
    binder = chains[plan['guide_binder_chain']]
    _, cdr_ordinals = framework_map(binder['sequence'], plan['framework'])
    cdr_ca = xyz(binder['coords'][cdr_ordinals])
    target_ca = xyz(plan['target_coords'])
    labels = plan['target_labels']
    if len(labels) != len(target_ca):
        raise ValueError('target coordinates and ordinal labels disagree')
    contact_distance = np.linalg.norm(
        target_ca[:, None, :] - cdr_ca[None, :, :], axis=-1
    ).min(axis=1)
    qualifying = [
        (int(ordinal) + 1, float(contact_distance[i]))
        for i, (chain, ordinal) in enumerate(labels)
        if chain == chain_name and contact_distance[i] <= cutoff_angstrom
    ]
    # Reject ambiguous duplicated schema sites; never silently collapse a
    # chain-mapping bug to make the residue count appear correct.
    if len({position for position, _ in qualifying}) != len(qualifying):
        raise ValueError('duplicate target schema positions in site candidate pool')
    if len(qualifying) < 4:
        raise ValueError(
            f'CDR-proximal site has {len(qualifying)} residues within '
            f'{cutoff_angstrom:g} A (required at least four). Check the source '
            'pose or explicitly choose a different site cutoff.'
        )
    # Proximity first, deterministic sequence position for exact ties.
    ranked = sorted(qualifying, key=lambda entry: (entry[1], entry[0]))
    selected = ranked[:int(max_residues)]
    positions = sorted(position for position, _ in selected)
    sequence = plan['target_sequences'][chain_name]
    if any(p < 1 or p > len(sequence) for p in positions):
        raise ValueError('derived epitope position falls outside target sequence')
    return {
        'chain': chain_name,
        'schema_positions_1_based': positions,
        'source_cdr_ca_cutoff_angstrom': float(cutoff_angstrom),
        'source_cdr_residue_count': len(cdr_ordinals),
        'source_target_residue_count': len(positions),
        'requested_site_residue_limit': int(max_residues),
        'source_candidate_pool_count': len(qualifying),
        'selection_policy': 'nearest_target_CA_to_any_source_CDR_CA_within_candidate_cutoff',
        'method': 'rank_nearest_source_CDR_CA_to_target_CA_within_cutoff',
        'residue_numbering': '1-based in exact target schema (not template author IDs)',
        'nearest_cdr_ca_distance_angstrom': {
            str(pos): float(distance) for pos, distance in selected
        },
    }



def native_epitope_template_ids(
    site: dict[str, Any], target_sequence: str, template_chain: dict[str, Any],
    *, min_identity: float = 0.8, min_coverage: float = 0.7,
    aligner: Any | None = None, alignment_report: dict[str, Any] | None = None,
) -> list[str]:
    """Map source-site runtime ordinals to native template residue IDs safely.

    The runtime target is a cropped antigen and the template can contain
    additional residues, gaps, substitutions or unresolved structure entries.
    Use Promera's own semi-global schema-to-template alignment, not
    exact_crop, which rejects legitimate template representations.

    Require the native model's configured identity/coverage floors, a
    one-to-one mapping, and exact amino-acid identity at all selected
    epitope positions. Those conditions prevent a high overall score from
    silently mapping a targeted residue to a different amino acid or gap.
    The downstream PoseGuidedDesign task independently checks the native
    Promera-derived struct.epitope_residues during CPU preflight.
    """
    template_sequence = str(template_chain['sequence'])
    ids = list(template_chain['residue_ids'])
    if not target_sequence or not template_sequence:
        raise ValueError('empty runtime or target template sequence')
    if len(ids) != len(template_sequence):
        raise ValueError('template residue IDs and sequence have incompatible lengths')
    if not (
        np.isfinite([min_identity, min_coverage]).all()
        and 0 < min_identity <= 1 and 0 < min_coverage <= 1
    ):
        raise ValueError('invalid target-template identity or coverage thresholds')

    if aligner is None:
        # This helper is the same one used in promera.inference.design for
        # ordinary targeted epitope mapping and template conditioning.
        # Lazy import keeps CPU geometry tests independent of the model,
        # TinyProt, and CUDA libraries until prepare() actually runs.
        from promera.inference.utils import align_schema_to_template
        aligner = align_schema_to_template

    mapping, stats = aligner(target_sequence, template_sequence)
    identity, coverage = float(stats['identity']), float(stats['coverage'])
    if (
        not np.isfinite([identity, coverage]).all()
        or identity < min_identity or coverage < min_coverage
    ):
        raise ValueError(
            'target-template alignment below configured safety thresholds: '
            f'identity={identity:.3f} (minimum {min_identity:.3f}), '
            f'coverage={coverage:.3f} (minimum {min_coverage:.3f})'
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('target-template alignment is not one-to-one')
    # Select only the user-requested compact site, never all candidate-pool
    # residues within the broad CDR neighbourhood.
    selected = []
    for position in site['schema_positions_1_based']:
        pos = int(position)
        if not 1 <= pos <= len(target_sequence):
            raise ValueError(f'target site position {pos} outside runtime sequence')
        index = mapping.get(pos - 1)
        if index is None:
            raise ValueError(
                f'target site position {pos} has no corresponding template residue; '
                'cannot safely pin this epitope'
            )
        if not 0 <= index < len(template_sequence):
            raise ValueError(f'template index {index} for site position {pos} is invalid')
        if target_sequence[pos - 1] != template_sequence[index] or target_sequence[pos - 1] == 'X':
            raise ValueError(
                f'target site position {pos} maps to a non-identical or unresolved '
                f'template residue ({target_sequence[pos - 1]} vs {template_sequence[index]}); '
                'choose a source-supported target site'
            )
        residue_id = str(ids[index])
        if not residue_id.strip() or residue_id in {'.', '?'}:
            raise ValueError(f'target site position {pos} has no usable template residue ID')
        selected.append(residue_id)

    if len(set(selected)) != len(selected):
        raise ValueError('target template has duplicated selected residue IDs')
    if alignment_report is not None:
        alignment_report.update({
            'method': 'promera.inference.utils.align_schema_to_template',
            'runtime_length': len(target_sequence),
            'template_length': len(template_sequence),
            'identity': identity,
            'coverage': coverage,
            'mapped': int(stats['mapped']),
            'min_identity': float(min_identity),
            'min_coverage': float(min_coverage),
            'schema_to_template_residue_ids': {
                str(int(pos)): str(resid)
                for pos, resid in zip(site['schema_positions_1_based'], selected)
            },
        })
    return selected




def template_author_to_native_ids(
    template_cif: str | Path, label_chain: str,
) -> dict[str, str]:
    """Map PDB *author* antigen IDs onto the native TinyProt template IDs.

    Epitope Scan uses auth_seq_id; Promera's native epitope/template code
    uses Structure.chain.ridx (label_seq_id in this source structure).
    Treating numeric author IDs as native IDs silently targets wrong residues.
    """
    # Standalone streaming parser for the PDB mmCIF atom_site loop.
    # No added dependency is needed in the existing immutable Promera image.
    # Refuse malformed rows and ambiguous mappings rather than guessing.
    import shlex
    required = [
        '_atom_site.group_PDB',
        '_atom_site.label_asym_id', '_atom_site.auth_asym_id',
        '_atom_site.label_seq_id', '_atom_site.auth_seq_id',
        '_atom_site.pdbx_PDB_ins_code',
    ]
    columns: list[str] = []
    in_loop, in_atoms, found = False, False, False
    def atom_records():
        nonlocal in_loop, in_atoms, found, columns
        with Path(template_cif).open(encoding='utf-8') as handle:
            for raw in handle:
                line = raw.strip()
                if line == 'loop_':
                    if in_atoms:
                        break
                    columns, in_loop = [], True
                    continue
                if line == '#':
                    if in_atoms:
                        break
                    columns, in_loop = [], False
                    continue
                if not line:
                    continue
                if in_loop and line.startswith('_'):
                    if in_atoms:
                        raise ValueError('unexpected mmCIF column inside atom_site data')
                    columns.append(line.split()[0])
                    continue
                if not in_loop or not columns:
                    continue
                if not in_atoms:
                    if not all(key in columns for key in required):
                        # Another loop; ignore it until its delimiter.
                        in_loop = False
                        continue
                    in_atoms = True
                    found = True
                if line.startswith(';'):
                    raise ValueError('multiline atom_site values are unsupported')
                tokens = shlex.split(line, comments=False, posix=True)
                if len(tokens) != len(columns):
                    raise ValueError(
                        f'malformed atom_site row: {len(tokens)} values for '
                        f'{len(columns)} columns; use standard PDB mmCIF'
                    )
                selected = {name: tokens[columns.index(name)] for name in required}
                yield tuple(selected[key] for key in required)
    author_to_native: dict[str, str] = {}
    native_to_author: dict[str, str] = {}
    for group, label, author_chain, native_id, author_id, insertion in atom_records():
        if group != 'ATOM' or label != label_chain:
            continue
        if not native_id or native_id in {'.', '?'} or not author_id or author_id in {'.','?'}:
            continue
        if author_chain != label_chain:
            # The Epitope Scan screenshot uses author chain A. Chain aliases
            # need an explicit map, never a silent assumption.
            raise ValueError(
                f'target template label chain {label_chain} has a different author '
                f'chain {author_chain}; resolve the chain map before targeting'
            )
        insertion = '' if insertion in {'.', '?'} else str(insertion)
        author_key = str(author_id) + insertion
        native_id = str(native_id)
        if (author_key in author_to_native and author_to_native[author_key] != native_id
                or native_id in native_to_author and native_to_author[native_id] != author_key):
            raise ValueError('ambiguous native/author residue numbering in target template')
        author_to_native[author_key] = native_id
        native_to_author[native_id] = author_key
    if not found or not author_to_native:
        raise ValueError(f'no valid polymer atom_site loop for target chain {label_chain}')
    return author_to_native


def basin_site_from_author_ids(
    target_sequence: str, template_chain: dict[str, Any],
    author_residue_ids: list[str], chain: str, *,
    author_to_native_ids: dict[str, str],
    min_identity: float = 0.8, min_coverage: float = 0.7,
    aligner: Any | None = None,
) -> dict[str, Any]:
    """Resolve Epitope Scan author IDs through native label IDs into runtime positions.

    Distinguish three numbering systems: PDB author numbering, template
    chain.ridx native IDs and 1-based cropped Promera target-schema positions.
    """
    if not author_residue_ids or len(set(author_residue_ids)) != len(author_residue_ids):
        raise ValueError('basin target residues must be distinct, nonempty author IDs')
    template_seq = str(template_chain['sequence'])
    native_ids = [str(x) for x in template_chain['residue_ids']]
    if len(native_ids) != len(template_seq) or len(set(native_ids)) != len(native_ids):
        raise ValueError('ambiguous or incomplete native target-template residue identifiers')
    if aligner is None:
        from promera.inference.utils import align_schema_to_template
        aligner = align_schema_to_template
    mapping, stats = aligner(target_sequence, template_seq)
    if (float(stats['identity']) < min_identity or float(stats['coverage']) < min_coverage
            or not np.isfinite([stats['identity'], stats['coverage']]).all()
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError('B004 target-template alignment failed identity/coverage/uniqueness checks')
    inverse = {j: i for i, j in mapping.items()}
    template_by_native = {rid: i for i, rid in enumerate(native_ids)}
    positions, selected_native = [], []
    for rid in author_residue_ids:
        template_native_id = author_to_native_ids.get(str(rid))
        if template_native_id is None:
            raise ValueError(f'Epitope Scan author target {chain}:{rid} not found in target mmCIF')
        ti = template_by_native.get(template_native_id)
        if ti is None:
            raise ValueError(
                f'Epitope Scan {chain}:{rid} maps to label_seq_id {template_native_id} '
                'which TinyProt did not retain; cannot condition safely'
            )
        si = inverse.get(ti)
        if si is None or target_sequence[si] != template_seq[ti] or target_sequence[si] == 'X':
            raise ValueError(
                f'Epitope Scan target {chain}:{rid} is absent or changed in runtime target'
            )
        positions.append(si + 1)
        selected_native.append(template_native_id)
    if len(set(positions)) != len(positions):
        raise ValueError('B004 author residues map to duplicate runtime positions')
    site = {
        'chain': chain,
        'schema_positions_1_based': positions,
        'epitope_scan_author_residue_ids': list(author_residue_ids),
        'native_epitope_residue_ids': selected_native,
        'native_epitope_numbering': 'target_template_label_seq_ids',
        'native_template_chain': chain,
        'source_cdr_ca_cutoff_angstrom': 10.0,
        'source_target_residue_count': len(positions),
        'selection_policy': 'epitope_scan_basin_recommended_author_residues',
        'residue_numbering': 'Epitope Scan author -> template native -> runtime schema',
        'template_alignment': {
            'method': 'promera.inference.utils.align_schema_to_template',
            'identity': float(stats['identity']), 'coverage': float(stats['coverage']),
            'runtime_length': len(target_sequence), 'template_length': len(template_seq),
            'author_to_native_to_schema': {
                f'{chain}:{rid}': {'native_label_seq_id': native,
                                  'schema_position_1_based': p}
                for rid, native, p in zip(author_residue_ids, selected_native, positions)
            },
        },
    }
    if native_epitope_template_ids(
        site, target_sequence, template_chain,
        min_identity=min_identity, min_coverage=min_coverage, aligner=aligner,
    ) != selected_native:
        raise ValueError('B004 author/native/schema round-trip failed')
    return site


def site_contact_metrics(chains: dict[str, Any], plan: dict[str, Any]) -> dict[str, float | int | None]:
    """Observed CDR/antigen contacts to the *shared* source-defined site.

    Always measure the full generated CDR (not the reference CDR geometry).
    These are C-alpha proximity diagnostics, not predictions of binding.
    """
    site = plan.get('targeted_site')
    if not site:
        return {
            'site_contacted_residues': None,
            'site_contact_fraction': None,
            'site_contact_share_of_all_contacts': None,
            'cdr_residue_count': None,
        }
    binder = chains[plan['binder_chain']]
    _, cdr_ordinals = framework_map(binder['sequence'], plan['framework'])
    cdr_xyz = xyz(binder['coords'][cdr_ordinals])
    chain = site['chain']
    antigen = xyz(chains[chain]['coords'])
    threshold = float(site['source_cdr_ca_cutoff_angstrom'])
    nearest = np.linalg.norm(
        antigen[:, None, :] - cdr_xyz[None, :, :], axis=-1
    ).min(axis=1)
    contacted = set(int(i) + 1 for i in np.flatnonzero(nearest <= threshold))
    target = set(site['schema_positions_1_based'])
    site_touches = len(contacted & target)
    return {
        'site_contacted_residues': site_touches,
        'site_contact_fraction': site_touches / len(target),
        'site_contact_share_of_all_contacts':
            site_touches / len(contacted) if contacted else 0.0,
        'cdr_residue_count': len(cdr_ordinals),
    }


def observed_arrays(chains: dict[str, Any], plan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for name, sequence in plan['target_sequences'].items():
        if chains[name]['sequence'] != sequence:
            raise ValueError(f'target sequence/order changed in {name}')
    target = xyz([chains[name]['coords'][ordinal] for name, ordinal in plan['target_labels']])
    binder = chains[plan['binder_chain']]
    fixed, _ = framework_map(binder['sequence'], plan['framework'])
    fw = xyz(binder['coords'][fixed])
    if len(fw) != len(plan['source_framework_coords']):
        raise ValueError('framework mapping has changed')
    return target, fw, xyz(binder['coords'])
