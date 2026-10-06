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


def cross_distogram(
    n_tokens: int, target_tokens: list[int], binder_tokens: list[int],
    target_xyz: Any, binder_xyz: Any, count: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Use a nested, coverage-balanced prefix of 16 undirected cross-chain pairs.

    Four pairs do not determine all six rigid-body degrees of freedom. Pair
    count is an experimental variable, not a calibrated strength parameter.
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
    order = [(i, (i + offset) % 4) for offset in range(4) for i in range(4)]
    for ti, bi in order[:count]:
        distance = float(np.linalg.norm(target[ti] - binder[bi]))
        # Saturated overflow bins carry no finite-distance placement information.
        if not 1 < distance < 50:
            raise ValueError(f'anchor distance {distance:.3f} A is outside informative bins (1, 50)')
        t, b = target_tokens[ti], binder_tokens[bi]
        bucket = int(np.sum(distance > np.linspace(1, 50, 50)))
        mask[t, b] = mask[b, t] = True
        bins[t, b] = bins[b, t] = bucket
        records.append({'target_token': t, 'binder_token': b, 'distance_angstrom': distance, 'bin': bucket})
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
                 observed_fw: Any, reference_axis: Any) -> dict[str, float]:
    """Separate internal shape, body rotation, approach direction and displacement."""
    rt, rf, ot, of = map(xyz, (reference_target, reference_fw, observed_target, observed_fw))
    rotation, shift = align(ot, rt)
    fw = of @ rotation + shift
    body_rotation, body_shift = align(rf, fw)
    axis = np.asarray(reference_axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-6:
        raise ValueError('reference approach axis is undefined')
    axis = axis / np.linalg.norm(axis)
    return {
        'target_fit_rmsd': rmsd(ot @ rotation + shift, rt),
        'source_framework_rmsd': rmsd(fw, rf),
        'framework_internal_rmsd': rmsd(rf @ body_rotation + body_shift, fw),
        'centroid_shift_angstrom': float(np.linalg.norm(fw.mean(0) - rf.mean(0))),
        'body_rotation_degrees': float(np.rad2deg(np.arccos(np.clip((np.trace(body_rotation) - 1) / 2, -1, 1)))),
        # Transport a fixed source-defined axis; do not redefine it from new CDRs.
        'approach_degrees': float(np.rad2deg(np.arccos(np.clip(axis @ (axis @ body_rotation), -1, 1)))),
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
