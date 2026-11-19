"""Post-run, reference-aware B004 comparison. CPU only; no model inference.

The B004 footprint overlap is a *proxy*, not a new official Epitope Scan
basin assignment. Epitope Scan's CDR angle/contact definition is reproduced
at the structural level; backbone contacts are necessarily backbone-heavy-atom
only whereas independently predicted refolds may contain sidechains.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

import numpy as np

from pose_geometry import align, framework_map, rmsd

CUTOFF_ANGSTROM = 5.0
CDR_ANGLE_MAX = 20.0
CDR_CONTACT_MIN = 0.60
FRAMEWORK_ANGLE_MIN = 30.0
DOCKQ_MIN_STRICT = 0.23


def _median(values):
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(cleaned) if cleaned else None


def _pct(k, n):
    return 100 * k / n if n else None


def _save_csv(path, rows, fields):
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _reference_label(angle, fraction):
    if angle is None or fraction is None:
        return 'no_contacts'
    if angle <= CDR_ANGLE_MAX and fraction > CDR_CONTACT_MIN:
        return 'cdr_driven'
    if angle >= FRAMEWORK_ANGLE_MIN or (angle > 20 and fraction < 0.50):
        return 'framework_driven'
    return 'mixed_or_unclassified'


def _loop_ranges(sequence, template):
    fixed, cdr = framework_map(sequence, template)
    parts = re.split(r'<cdrh[123]>', template)
    if len(parts) != 4:
        raise ValueError('invalid VHH framework tags')
    pattern = re.compile('^' + re.escape(parts[0]) + '(.*?)' +
                         re.escape(parts[1]) + '(.*?)' +
                         re.escape(parts[2]) + '(.*?)' +
                         re.escape(parts[3]) + '$')
    match = pattern.fullmatch(sequence)
    if not match:
        raise ValueError('VHH sequence does not match framework/CDR split')
    loops = {f'cdr{i}': np.arange(*match.span(i), dtype=int) for i in (1, 2, 3)}
    if any(not len(v) for v in loops.values()) or set(cdr) != set(np.concatenate(list(loops.values()))):
        raise ValueError('invalid loop spans')
    return np.asarray(fixed, dtype=int), loops


def _structure(path, plan):
    """Extract heavy-atom coordinates, residue owners and C-alpha per chain."""
    from tinyprot.structure import Structure
    result = {}
    raw = Structure.from_mmcif(str(path))
    for name, chain in raw.chains.items():
        name = str(name)
        n = len(chain.rname)
        ca = np.full((n, 3), np.nan)
        all_xyz, all_owner = [], []
        for i in range(n):
            labels = np.asarray(chain.aname[i]).astype(str)
            mask = np.asarray(chain.mask[i], dtype=bool)
            if len(labels) != len(mask):
                raise ValueError(f'{path}: atom name/mask mismatch')
            hits = np.flatnonzero((labels == 'CA') & mask)
            if len(hits) == 1:
                ca[i] = chain.coords[i, hits[0]]
            for k in np.flatnonzero(mask):
                atom = labels[k].strip()
                if not atom or atom[0].upper() in {'H', 'D'}:
                    continue
                xyz = np.asarray(chain.coords[i, k], dtype=float)
                if np.isfinite(xyz).all():
                    all_xyz.append(xyz)
                    all_owner.append(i)
        from pose_geometry import chain_sequence
        result[name] = {
            'sequence': chain_sequence(chain),
            'ca': ca,
            'atoms': np.asarray(all_xyz, dtype=float).reshape((-1, 3)),
            'atom_owners': np.asarray(all_owner, dtype=int),
        }
    expected = set(plan['target_sequences']) | {plan['binder_chain']}
    if not expected.issubset(set(result)):
        raise ValueError(f'{path}: missing expected chains {sorted(expected - set(result))}')
    for chain in plan['target_sequences']:
        if result[chain]['sequence'] != plan['target_sequences'][chain]:
            raise ValueError(f'{path}: antigen sequence differs from prepared run')
    return result


def _contacts(struct, plan, fixed, loops):
    """Epitope Scan-compatible 5 A heavy-atom contacts and centroid approach."""
    from scipy.spatial import cKDTree
    binder = struct[plan['binder_chain']]
    coords, owner = [], []
    for name in plan['target_sequences']:
        chain = struct[name]
        if len(chain['atoms']):
            coords.extend(chain['atoms'])
            owner.extend((name, int(i)) for i in chain['atom_owners'])
    if not coords:
        raise ValueError('no antigen heavy atoms available')
    target_tree = cKDTree(np.asarray(coords))
    pairs = target_tree.query_ball_point(binder['atoms'], r=CUTOFF_ANGSTROM)
    contacting_binder, contacting_antigen, cdr_antigen = set(), set(), set()
    cdr_set = set(int(i) for loop in loops.values() for i in loop)
    for i, neighbours in enumerate(pairs):
        if not neighbours:
            continue
        pos = int(binder['atom_owners'][i])
        contacting_binder.add(pos)
        for j in neighbours:
            antigen_residue = owner[j]
            contacting_antigen.add(antigen_residue)
            if pos in cdr_set:
                cdr_antigen.add(antigen_residue)
    if not contacting_binder:
        return {
            'contacting_binder_residues': 0, 'cdr_contacting_binder_residues': 0,
            'cdr_contact_fraction': 0.0, 'epitope_scan_approach_angle_degrees': None,
            'epitope_scan_class': 'no_contacts',
            'contacted_antigen': set(), 'cdr_contacted_antigen': set(),
        }
    fraction = len(contacting_binder.intersection(cdr_set)) / len(contacting_binder)
    ca = binder['ca']
    if not np.isfinite(ca).all():
        raise ValueError('nonfinite VHH C-alpha coordinates')
    interface_ca = np.asarray([struct[name]['ca'][ordinal] for name, ordinal in sorted(contacting_antigen)])
    if not np.isfinite(interface_ca).all():
        raise ValueError('nonfinite contacted antigen C-alpha coordinates')
    fw_center = ca[fixed].mean(axis=0)
    cdr_center = ca[np.concatenate(list(loops.values()))].mean(axis=0)
    interface_center = interface_ca.mean(axis=0)
    a, b = cdr_center - fw_center, interface_center - fw_center
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        raise ValueError('undefined Epitope Scan centroid approach axis')
    angle = float(np.degrees(np.arccos(np.clip(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1,
    ))))
    return {
        'contacting_binder_residues': len(contacting_binder),
        'cdr_contacting_binder_residues': len(contacting_binder.intersection(cdr_set)),
        'cdr_contact_fraction': fraction,
        'epitope_scan_approach_angle_degrees': angle,
        'epitope_scan_class': _reference_label(angle, fraction),
        'contacted_antigen': contacting_antigen,
        'cdr_contacted_antigen': cdr_antigen,
    }


def _dockq(row, plan):
    data = json.loads(row.get('scDockQ_json') or '{}')
    expected = plan['targeted_site']['chain'] + '_' + plan['binder_chain']
    reverse = plan['binder_chain'] + '_' + plan['targeted_site']['chain']
    item = data.get(expected, data.get(reverse))
    if not isinstance(item, dict):
        raise ValueError(f'refold has no scDockQ for required antigen/binder pair ({expected})')
    result = item.get('DockQ')
    return float(result) if isinstance(result, (int, float)) and math.isfinite(result) else None


def _region_metrics(contacts, plan, angle):
    site = plan['targeted_site']
    chain = site['chain']
    required = {(chain, pos - 1) for pos in site['schema_positions_1_based']}
    cdr = contacts['cdr_contacted_antigen']
    all_antigen = contacts['contacted_antigen']
    candidates = {int(i + 1) for name, i in cdr if name == chain}
    region = plan['basin_reference']['regions']
    distances = []
    for name, info in region.items():
        footprint = set(info['footprint_schema_positions'])
        if not footprint or not candidates:
            score = 0.0
        else:
            score = len(candidates & footprint) / len(candidates | footprint)
        distances.append((score, name))
    distances.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_name = distances[0]
    # A nearest historical footprint is not a production basin assignment.
    footprint_b004 = set(region[plan['basin_reference']['basin_id']]['footprint_schema_positions'])
    in_b004 = len(candidates & footprint_b004)
    is_cdr = contacts['epitope_scan_class'] == 'cdr_driven'
    site_cdr = len(cdr & required)
    return {
        'target_residue_any_contacts': len(all_antigen & required),
        'target_residue_cdr_contacts': site_cdr,
        'b004_cdr_footprint_residues': in_b004,
        'b004_cdr_footprint_fraction': in_b004 / len(candidates) if candidates else 0.0,
        'nearest_historical_footprint': best_name if best_score > 0 else '',
        'nearest_footprint_jaccard': best_score,
        'source_relative_in_0_20': float(angle) <= 20,
        'source_neighbourhood_and_cdr_driven': float(angle) <= 20 and is_cdr,
        'b004_like_proxy': (
            float(angle) <= 20 and is_cdr and site_cdr >= 1
            and best_name == plan['basin_reference']['basin_id'] and best_score > 0
        ),
    }


def _as_bool(value):
    return str(value).lower() in {'true', '1', 'yes'}


def _values(row):
    """Reduce nonserializable internal contact sets before CSV writing."""
    return {k: v for k, v in row.items() if not isinstance(v, (np.ndarray, set, dict)) and not k.startswith('_')}


def _diversity(backbones, plan):
    grouped = defaultdict(list)
    for rec in backbones:
        grouped[rec['arm']].append(rec)
    results = []
    for arm, items in sorted(grouped.items()):
        cdr_lengths = set(tuple(len(r['_loops'][f'cdr{i}']) for i in (1, 2, 3)) for r in items)
        pairs, tips = [], []
        nearest = [float('inf') for _ in items]
        for i, a in enumerate(items):
            for j in range(i + 1, len(items)):
                b = items[j]
                dist = float(np.linalg.norm(a['_tip3'] - b['_tip3']))
                tips.append(dist)
                if len(a['_loops']['cdr3']) != len(b['_loops']['cdr3']):
                    continue
                d = rmsd(a['_placed_cdr3'], b['_placed_cdr3'])
                pairs.append(d)
                nearest[i], nearest[j] = min(nearest[i], d), min(nearest[j], d)
        results.append({
            'arm': arm, 'generated_backbones': len(items),
            'distinct_cdr_length_triplets': len(cdr_lengths),
            'cdr3_length_variants': len(set(x[2] for x in cdr_lengths)),
            'cdr3_tip_pair_distance_median_A': _median(tips),
            'equal_length_cdr3_pair_rmsd_median_A': _median(pairs),
            'equal_length_cdr3_pair_count': len(pairs),
            'equal_length_cdr3_pair_fraction_ge2A': (
                sum(v >= 2 for v in pairs) / len(pairs) if pairs else None),
            'equal_length_cdr3_nn_median_A': _median(v for v in nearest if math.isfinite(v)),
        })
    return results


def _aggregate(observations, designs, backbone_records, diversity, totals, manifest):
    result = []
    for arm in dict.fromkeys(job['arm'] for job in manifest['jobs']):
        bb = [r for r in backbone_records if r['arm'] == arm]
        rr = [r for r in observations if r['arm'] == arm and r['kind'] == 'refold']
        dd = [r for r in designs if r['arm'] == arm]
        requested = sum(int(x['requested_backbones']) for x in totals if x['arm'] == arm)
        result.append({
            'arm': arm, 'requested_backbones': requested, 'generated_backbones': len(bb),
            'designed_sequences': len(dd), 'refold_structures': len(rr),
            'backbone_in_source_neighbourhood': sum(r['source_relative_in_0_20'] for r in bb),
            'backbone_cdr_driven': sum(r['epitope_scan_class'] == 'cdr_driven' for r in bb),
            'backbone_framework_driven': sum(r['epitope_scan_class'] == 'framework_driven' for r in bb),
            'backbone_b004_like_proxy': sum(r['b004_like_proxy'] for r in bb),
            'backbone_target_contact_via_cdr': sum(r['target_residue_cdr_contacts'] >= 1 for r in bb),
            'backbone_cdr_angle_median_degrees': _median(r['epitope_scan_approach_angle_degrees'] for r in bb),
            'backbone_delta_cdr_angle_from_source_median_degrees': _median(
                r['delta_epitope_scan_angle_from_source_degrees'] for r in bb),
            'backbone_source_relative_median_degrees': _median(r['source_relative_angle_degrees'] for r in bb),
            'backbone_cdr_fraction_median': _median(r['cdr_contact_fraction'] for r in bb),
            'accepted_scDockQ_gt_023_provisional': sum(d['accepted_scDockQ_gt_023'] for d in dd),
            'accepted_cdr_driven': sum(d['accepted_scDockQ_gt_023'] and d['representative_class']=='cdr_driven' for d in dd),
            'accepted_b004_like_proxy': sum(d['accepted_scDockQ_gt_023'] and d['representative_b004_like_proxy'] for d in dd),
            'designed_vhh_recovery_le3A': sum(d['best_binder_only_ca_rmsd_A'] is not None
                                               and d['best_binder_only_ca_rmsd_A'] <= 3 for d in dd),
            'best_vhh_rmsd_median_A': _median(d['best_binder_only_ca_rmsd_A'] for d in dd),
            'best_cdr3_rmsd_median_A': _median(d['best_cdr3_framework_rmsd_A'] for d in dd),
            'best_own_dockq_median': _median(d['max_scDockQ'] for d in dd),
            'jobs_completed': sum(x['status']=='completed' for x in totals if x['arm']==arm),
            'jobs_requested': sum(1 for x in totals if x['arm']==arm),
        })
    return result


def analyse_basin(root: Path, existing: list[dict], totals: list[dict],
                  plan: dict, manifest: dict) -> None:
    """Read structures once per stage and write interpretable, lightweight outputs."""
    if manifest.get('profile') != 'basin':
        raise ValueError('basin analysis only applies to --profile basin')
    output = root / 'analysis' / 'basin'
    output.mkdir(parents=True, exist_ok=True)
    observations, backbones, errors, grouped = [], [], [], defaultdict(list)
    # Directly measure the reference backbone rather than assume the Atlas's
    # representative-refold approach angle also describes the raw backbone.
    reference = _structure(root / 'inputs' / 'guide.cif', plan)
    source_binder = reference[plan['binder_chain']]
    source_fixed, source_loops = _loop_ranges(source_binder['sequence'], plan['framework'])
    source_contacts = _contacts(reference, plan, source_fixed, source_loops)
    source_angle = source_contacts['epitope_scan_approach_angle_degrees']
    backbone_cache = {}
    for entry in existing:
        path = Path(entry['cif']).resolve()
        try:
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('missing or outside-experiment CIF')
            arm, rep = entry['arm'], int(entry['replicate'])
            kind = entry['kind']
            match = re.search(r'/sample(\d+)/', str(path))
            if not match:
                raise ValueError('unrecognized sample directory')
            sample = int(match.group(1))
            pred = _structure(path, plan)
            binder = pred[plan['binder_chain']]
            ca = binder['ca']
            if not np.isfinite(ca).all():
                raise ValueError('VHH backbone/refold has non-finite CA coordinates')
            fixed, loops = _loop_ranges(binder['sequence'], plan['framework'])
            contact = _contacts(pred, plan, fixed, loops)
            source_relative = float(entry['approach_degrees'])
            region = _region_metrics(contact, plan, source_relative)
            obs = {
                'arm': arm, 'replicate': rep, 'sample_index': sample,
                'kind': kind, 'cif': str(path),
                'source_relative_angle_degrees': source_relative,
                'source_framework_rmsd_A': float(entry['source_framework_rmsd']),
                'epitope_scan_approach_angle_degrees': contact['epitope_scan_approach_angle_degrees'],
                'delta_epitope_scan_angle_from_source_degrees': (
                    abs(float(contact['epitope_scan_approach_angle_degrees']) - float(source_angle))
                    if source_angle is not None and contact['epitope_scan_approach_angle_degrees'] is not None
                    else None
                ),
                'cdr_contact_fraction': contact['cdr_contact_fraction'],
                'contacting_binder_residues': contact['contacting_binder_residues'],
                'cdr_contacting_binder_residues': contact['cdr_contacting_binder_residues'],
                'epitope_scan_class': contact['epitope_scan_class'],
                **region,
                'iptm': float(entry['iptm']) if kind=='refold' and entry.get('iptm') else None,
                'scDockQ': _dockq(entry, plan) if kind=='refold' else None,
                'antigen_aligned_binder_rmsd_A': float(entry['self_binder_rmsd']) if kind=='refold' else None,
                'binder_only_ca_rmsd_A': None,
                'cdr1_framework_rmsd_A': None, 'cdr2_framework_rmsd_A': None,
                'cdr3_framework_rmsd_A': None,
                'sequence': entry.get('sequence', '') if kind=='refold' else '',
            }
            key = (arm, rep, sample)
            if kind == 'backbone':
                rot, shift = align(ca[fixed], np.asarray(plan['source_framework_coords']))
                placed = ca @ rot + shift
                cdr3 = loops['cdr3']
                nmid = min(3, len(cdr3))
                middle = cdr3[(len(cdr3)-nmid)//2 : (len(cdr3)-nmid)//2 + nmid]
                obs['_placed_cdr3'] = placed[cdr3]
                obs['_tip3'] = placed[middle].mean(axis=0)
                obs['_loops'] = loops
                backbone_cache[key] = (ca.copy(), fixed, loops)
                backbones.append(obs)
            else:
                reference = backbone_cache.get(key)
                if reference is None:
                    # The existing analysis groups refolds under their parent backbone,
                    # but does not guarantee that CSV entries are ordered.
                    bp = path.parent.parent / 'backbone.cif'
                    base = _structure(bp, plan)[plan['binder_chain']]
                    bca = base['ca']
                    bfixed, bloops = _loop_ranges(base['sequence'], plan['framework'])
                    reference = (bca, bfixed, bloops)
                    backbone_cache[key] = reference
                base_ca, base_fixed, base_loops = reference
                if ca.shape != base_ca.shape or not np.array_equal(fixed, base_fixed):
                    raise ValueError('refold/backbone C-alpha or framework length mismatch')
                if any(not np.array_equal(loops[name], base_loops[name]) for name in loops):
                    raise ValueError('refold/backbone CDR intervals differ')
                rr, tt = align(ca, base_ca)
                obs['binder_only_ca_rmsd_A'] = rmsd(ca @ rr + tt, base_ca)
                fr, ft = align(ca[fixed], base_ca[fixed])
                placed = ca @ fr + ft
                for i in (1, 2, 3):
                    name = f'cdr{i}'
                    obs[f'{name}_framework_rmsd_A'] = rmsd(placed[loops[name]], base_ca[loops[name]])
                refname = re.fullmatch(r'sample(\d+)_design(\d+)_refold(\d+)', path.stem)
                if refname is None or int(refname.group(1)) != sample:
                    raise ValueError('unrecognized refold identifier')
                obs['design_index'] = int(refname.group(2))
                obs['refold_index'] = int(refname.group(3))
                grouped[(arm, rep, sample, obs['design_index'], obs['sequence'])].append(obs)
            observations.append(obs)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            errors.append({'kind':entry.get('kind'), 'path':entry.get('cif'), 'error':str(error)})
    designs = []
    for (arm, rep, sample, did, seq), rows in sorted(grouped.items()):
        ranked = sorted(
            (r for r in rows if r['scDockQ'] is not None),
            key=lambda r: (-r['scDockQ'], -float(r['iptm'] or 0), r['refold_index']),
        )
        selected = ranked[0] if ranked else None
        designs.append({
            'arm': arm, 'replicate': rep, 'sample_index':sample, 'design_index':did,
            'sequence': seq, 'refolds':len(rows),
            'max_scDockQ': selected['scDockQ'] if selected else None,
            'accepted_scDockQ_gt_023': bool(selected and selected['scDockQ'] > DOCKQ_MIN_STRICT),
            'representative_class': selected['epitope_scan_class'] if selected else '',
            'representative_cdr_angle_degrees': selected['epitope_scan_approach_angle_degrees'] if selected else None,
            'representative_cdr_contact_fraction': selected['cdr_contact_fraction'] if selected else None,
            'representative_source_relative_angle_degrees': selected['source_relative_angle_degrees'] if selected else None,
            'representative_b004_like_proxy': selected['b004_like_proxy'] if selected else False,
            'best_binder_only_ca_rmsd_A': min(r['binder_only_ca_rmsd_A'] for r in rows),
            'best_cdr3_framework_rmsd_A': min(r['cdr3_framework_rmsd_A'] for r in rows),
            'max_iptm': max((r['iptm'] for r in rows if r['iptm'] is not None), default=None),
        })
    diversity = _diversity(backbones, plan)
    summary = _aggregate(observations, designs, backbones, diversity, totals, manifest)
    # Independent SLURM replicates, not five refolds or pairwise RMSDs,
    # define the between-condition experimental repeat.
    by_replicate = []
    for job in manifest['jobs']:
        arm, rep = job['arm'], int(job['replicate'])
        bb = [x for x in backbones if x['arm']==arm and x['replicate']==rep]
        dd = [x for x in designs if x['arm']==arm and x['replicate']==rep]
        by_replicate.append({
            'arm':arm, 'replicate':rep,
            'requested_backbones':int(job['requested_backbones']),
            'generated_backbones':len(bb), 'designed_sequences':len(dd),
            'source_neighbourhood_backbones':sum(x['source_relative_in_0_20'] for x in bb),
            'cdr_driven_backbones':sum(x['epitope_scan_class']=='cdr_driven' for x in bb),
            'b004_like_proxy_backbones':sum(x['b004_like_proxy'] for x in bb),
            'accepted_scDockQ_gt_023_sequences':sum(x['accepted_scDockQ_gt_023'] for x in dd),
            'accepted_cdr_driven_sequences':sum(
                x['accepted_scDockQ_gt_023'] and x['representative_class']=='cdr_driven' for x in dd),
            'accepted_b004_like_proxy_sequences':sum(
                x['accepted_scDockQ_gt_023'] and x['representative_b004_like_proxy'] for x in dd),
        })
    # Refold-specific columns (design/refold indices) must not be dropped just
    # because the first row belongs to a backbone.
    obs_columns = list(dict.fromkeys(
        k for row in observations for k in row if not k.startswith('_')
    ))
    _save_csv(output / 'per_structure.csv', [_values(x) for x in observations], obs_columns)
    _save_csv(output / 'per_sequence.csv', designs, list(designs[0]) if designs else [])
    _save_csv(output / 'by_replicate.csv', by_replicate, list(by_replicate[0]) if by_replicate else [])
    _save_csv(output / 'diversity.csv', diversity, list(diversity[0]) if diversity else [])
    _save_csv(output / 'summary.csv', summary, list(summary[0]) if summary else [])
    (output / 'errors.json').write_text(json.dumps(errors, indent=2)+'\n')
    metrics = {
        'profile': 'basin', 'basin_id': manifest['basin_id'],
        'source_backbone_epitope_scan_approach_angle_degrees': source_angle,
        'source_backbone_epitope_scan_CDR_fraction': source_contacts['cdr_contact_fraction'],
        'source_backbone_epitope_scan_class': source_contacts['epitope_scan_class'],
        'source_pose_sha256': plan['source_pose_sha256'],
        'basin_source_npz_sha256': plan['basin_reference']['npz_sha256'],
        'site_author_ids': plan['targeted_site']['native_epitope_residue_ids'],
        'target_schema_positions': plan['targeted_site']['schema_positions_1_based'],
        'epitope_scan_cutoff_angstrom': CUTOFF_ANGSTROM,
        'epitope_scan_class': 'angle<=20 and CDR binder-contact fraction>0.60; 5A heavy-atom contacts',
        'b004_proxy_definition': (
            'CDR-driven & source-relative angle<=20 & >=1 CDR contact with recommended '
            'B004 target residue & best Jaccard overlap to B004 historic basin footprint'
        ),
        'comparison_warning': (
            'B004-like proxy is not an official post-refold Epitope Scan basin assignment. '
            'Backbone-only atoms undercount contacts relative to fully modeled refolds. '
            'Refold acceptance is provisional scDockQ>0.23, not the production usability audit.'
        ),
        'records': len(observations), 'sequences': len(designs),
        'errors': len(errors), 'complete': (
            not errors and len(backbones)==sum(int(j['requested_backbones']) for j in manifest['jobs'])
            and all(t['status']=='completed' for t in totals)
        ),
        'summary': summary,
        'by_replicate': by_replicate,
        'diversity': diversity,
    }
    (output / 'summary.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    lines = [
        '# B004 targeted versus pose-guided diffusion',
        '',
        'Recommended native target IDs: ' + ', '.join('A:'+x for x in metrics['site_author_ids']),
        'Source pose: B004__G047__5_389; angular guidance: 0–20° relative to this pose.',
        'Measured source BACKBONE Epitope Scan-style approach angle: '
          + str(source_angle) + '°; class: ' + source_contacts['epitope_scan_class']
          + ' (Atlas display may report the selected REFOLD angle instead).',
        'Epitope Scan approach = angle between F→CDR and F→contacted-antigen centroids,',
        'not the same measure as source-relative orientation.',
        '',
        '| Arm | Backbones | Designs | Source 0–20° | CDR-driven (backbones) | B004-like proxy (backbones) | DockQ>0.23 designs | DockQ>0.23 and B004-like |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in summary:
        lines.append(
            f"| {r['arm']} | {r['generated_backbones']}/{r['requested_backbones']} | "
            f"{r['designed_sequences']} | {r['backbone_in_source_neighbourhood']} | "
            f"{r['backbone_cdr_driven']} | {r['backbone_b004_like_proxy']} | "
            f"{r['accepted_scDockQ_gt_023_provisional']} | {r['accepted_b004_like_proxy']} |"
        )
    lines += ['', '## Independent replicate counts', '',
        '| Arm | Replicate | CDR-driven backbones | B004-like proxy backbones | Provisional accepted sequences |',
        '|---|---:|---:|---:|---:|']
    for r in by_replicate:
        lines.append(f"| {r['arm']} | {r['replicate']} | "
                     f"{r['cdr_driven_backbones']}/{r['generated_backbones']} | "
                     f"{r['b004_like_proxy_backbones']}/{r['generated_backbones']} | "
                     f"{r['accepted_scDockQ_gt_023_sequences']}/{r['designed_sequences']} |")
    lines += [
        '',
        '## What each measure can establish',
        '',
        '- The B004-like *proxy* combines source-relative orientation, CDR-driven classification,',
        '  contact with ≥1 recommended target residue through CDRs, and maximal archived',
        '  basin-footprint Jaccard similarity. No 100% target-residue coverage is required.',
        '- CDR-driven backbone rates and accepted/refold CDR-driven rates are separate.',
        '- The accepted-candidate numerator uses one highest-scDockQ refold per designed sequence',
        '  and a strict >0.23 numerical threshold, without pretending to reproduce',
        '  NOMINEE’s complete structural-usability checks.',
        '- CDR3 diversity compares equal-length loops only after framework alignment;',
        '  loop-length combinations and tip distances are reported separately.',
        '- Best binder-only RMSD measures fold recovery independent of docking orientation.',
        '- Per-replicate and per-structure rows remain available; do not treat five',
        '  refolds per sequence or many pairwise CDR RMSDs as independent biological replicates.',
        '',
        '## CDR3 diversity',
        '',
        '| Arm | Length triplets | Same-length CDR3 RMSD median (Å) | CDR3 tip distance median (Å) |',
        '|---|---:|---:|---:|',
    ]
    for r in diversity:
        lines.append(
            f"| {r['arm']} | {r['distinct_cdr_length_triplets']} | "
            f"{r['equal_length_cdr3_pair_rmsd_median_A']} | {r['cdr3_tip_pair_distance_median_A']} |"
        )
    lines += [
        '', '## Fold recovery',
        '',
        '| Arm | Median best VHH RMSD (Å) | Median best CDR3 RMSD (Å) | Designs with VHH RMSD ≤3 Å |',
        '|---|---:|---:|---:|',
    ]
    for r in summary:
        lines.append(
            f"| {r['arm']} | {r['best_vhh_rmsd_median_A']} | "
            f"{r['best_cdr3_rmsd_median_A']} | {r['designed_vhh_recovery_le3A']} |"
        )
    lines += [
        '', '**Important:** the reference B004 screenshot reports a framework-dominated',
        'basin. Its recommended A:339/A:343 site has not been validated as a',
        'CDR-favourable target. Report site-mediated and CDR-driven enrichment',
        'separately rather than calling every site contact a success.',
        '', f"Parse errors: {len(errors)}. Inspect errors.json before using partial results.",
    ]
    (output / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    files = [
        output / name for name in ('RESULTS.md', 'summary.csv', 'summary.json',
                                  'per_structure.csv', 'per_sequence.csv',
                                  'by_replicate.csv', 'diversity.csv', 'errors.json')
    ]
    files += [root/'manifest.json', root/'plan.json', root/'source_atlas_selection.json']
    archive = output / 'basin-analysis.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.is_file():
                zf.write(path, arcname=path.name)
    print(json.dumps({'basin_analysis':str(output), 'archive':str(archive),
                      'backbones':len(backbones), 'sequences':len(designs),
                      'parse_errors':len(errors), 'complete':metrics['complete']}, indent=2))
    if errors:
        raise ValueError('B004 basin report is partial: inspect analysis/basin/errors.json')
