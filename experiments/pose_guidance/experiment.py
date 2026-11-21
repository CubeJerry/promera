#!/usr/bin/env python3
"""Prepare, preflight, run and analyse a pose-guidance pilot. No production writes."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import sys
import tempfile
import time
import statistics
import traceback
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from pose_geometry import (align, build_plan, digest, observed_arrays,
                           pose_metrics, read_chains, rmsd,
                           targeted_site_from_reference, native_epitope_template_ids,
                           site_contact_metrics, cone_placement, jitter, cross_distogram)

HERE = Path(__file__).resolve().parent
ARMS = [('control', 0, 0, 0), ('anchors4', 4, 0, 0), ('anchors8', 8, 0, 0),
        ('anchors16', 16, 0, 0), ('nearby16', 16, 15, 2)]


def parse_angle_range(value: str | None) -> tuple[float, float] | None:
    """Parse explicit [min, max] deviation from the selected approach axis."""
    if value is None:
        return None
    parts = value.strip().split(':')
    if len(parts) != 2:
        raise ValueError('use --angle-range MIN:MAX, e.g. 15:35')
    try:
        low, high = (float(v) for v in parts)
    except ValueError as exc:
        raise ValueError('approach angle bounds must be numbers in degrees') from exc
    if not np.isfinite([low, high]).all() or not 0 <= low <= high <= 180:
        raise ValueError('approach angle range must satisfy 0 <= min <= max <= 180')
    return low, high


def _passes_geometry(metrics: dict[str, float], bounds: dict[str, float]) -> bool:
    """Evaluate measured output in the requested cone, not only an upper limit."""
    angle = metrics['approach_degrees']
    if bounds.get('position_comparison') == 'target_patch_radius':
        radial_change = metrics.get('patch_radius_change_angstrom')
        position_pass = radial_change is not None and radial_change <= bounds['patch_radius_change_angstrom_max']
    else:
        position_pass = metrics['centroid_shift_angstrom'] <= bounds['centroid_shift_angstrom_max']
    return (
        bounds.get('approach_degrees_min', 0) - 1e-5 <= angle
        <= bounds['approach_degrees_max'] + 1e-5
        and position_pass
        and metrics['framework_internal_rmsd'] <= bounds['framework_internal_rmsd_max']
        and metrics['target_fit_rmsd'] <= bounds['target_fit_rmsd_max']
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def resolve(value: str, base: Path) -> Path:
    result = Path(value).expanduser()
    return (result if result.is_absolute() else base / result).resolve()


def prepare(args: argparse.Namespace) -> None:
    base_path = Path(args.base_config).expanduser().resolve()
    cfg = OmegaConf.to_container(OmegaConf.load(base_path), resolve=True)
    if cfg.get('binder', {}).get('type') != 'vhh':
        raise ValueError('base config must be a standard Promera VHH Design config')
    source = resolve(str(cfg['input']), base_path.parent)
    if source.is_dir():
        candidates = sorted(source.glob('*.json'))
        if len(candidates) != 1:
            raise ValueError('pilot needs one target schema; set input to a single JSON file')
        source = candidates[0]
    schema = load_json(source)
    target_names = cfg.get('target_chains') or [x for x in schema if x != 'connections']
    if isinstance(target_names, str):
        target_names = [x.strip() for x in target_names.split(',') if x.strip()]
    if len(set(target_names)) != len(target_names):
        raise ValueError('duplicate target chain identifiers')
    if schema.get('connections'):
        raise ValueError('explicit covalent connections are outside this protein-only pilot')
    schema = {name: schema[name] for name in target_names}
    target_map = json.loads(args.target_map) if args.target_map else {name: name for name in schema}
    pose = Path(args.pose).expanduser().resolve()
    guide_binder = args.guide_binder_chain or cfg['binder']['chain']
    source_chains = read_chains(pose)
    plan = build_plan(source_chains, schema, cfg['binder']['framework'],
                      cfg['binder']['chain'], guide_binder, target_map)
    plan['source_pose_sha256'] = digest(pose)
    if args.profile == 'comparison':
        if getattr(args, 'angle_range', None) is None:
            raise ValueError('comparison requires --angle-range, e.g. 0:20')
        if cfg.get('epitope_residues') or cfg.get('epitope_residue_map'):
            raise ValueError(
                'comparison derives exactly one shared source-CDR target site; '
                'the original task already supplies epitope selection'
            )
        plan['targeted_site'] = targeted_site_from_reference(
            source_chains, plan,
            cutoff_angstrom=float(getattr(args, 'site_cutoff_angstrom', 10.0)),
            max_residues=getattr(args, 'site_residues', 7),
        )
    msa = resolve(args.msa_dir or str(cfg.get('msa_dir') or ''), base_path.parent)
    if not (args.msa_dir or cfg.get('msa_dir')) or not msa.is_dir():
        raise ValueError('provide an existing target MSA cache with --msa-dir')
    root = Path(args.out).expanduser().resolve()
    if root.exists():
        raise ValueError(f'experiment directory already exists: {root}; use a new directory')
    n = args.backbones if args.backbones is not None else (2 if args.profile == 'smoke' else 32)
    replicates = args.replicates if args.replicates is not None else (1 if args.profile == 'smoke' else 3)
    if min(n, replicates, args.batch_size, args.sequences_per_backbone) < 1 or not 0 <= args.seed < 2**32 - replicates:
        raise ValueError('invalid budget, batch size or replicate seed')
    interval = parse_angle_range(getattr(args, 'angle_range', None))
    angle_sampling = getattr(args, 'angle_sampling', 'uniform_solid_angle')
    if angle_sampling not in ('uniform_solid_angle', 'stratified_degrees'):
        raise ValueError('unsupported angular sampling policy')
    if interval is None and angle_sampling != 'uniform_solid_angle':
        raise ValueError('stratified angular sampling requires --angle-range')
    angle_shift = float(getattr(args, 'angle_translation_angstrom', 0.0))
    if not np.isfinite(angle_shift) or angle_shift < 0:
        raise ValueError('angle translation bound must be finite and nonnegative')
    if interval is None and angle_shift:
        raise ValueError('--angle-translation-angstrom requires --angle-range')
    if args.profile == 'comparison':
        # The primary null control sees the identical target epitope but no
        # inter-chain distances. An untargeted arm remains as context only.
        selected_arms = [('control', 0, 0, 0), ('targeted', 0, 0, 0),
                         ('anchors16', 16, 0, 0), ('cone16', 16, 0, angle_shift)]
    else:
        selected_arms = [ARMS[0], ARMS[3]] if args.profile == 'smoke' else list(ARMS)
        if interval is not None:
            selected_arms.append(('cone16', 16, 0, angle_shift))
    # Validate paths before creating the experiment tree.
    target_template = cfg.get('target_template')
    if target_template:
        template = resolve(target_template['path'], base_path.parent)
        if not template.is_file():
            raise ValueError(f'target template not found: {template}')
    if args.profile == 'comparison':
        site = plan['targeted_site']
        if target_template:
            native_chain = str(target_template.get('chain') or site['chain'])
            native_chains = read_chains(template)
            if native_chain not in native_chains:
                raise ValueError(f'target template is missing chain {native_chain!r}')
            alignment_report: dict[str, Any] = {}
            site['native_epitope_residue_ids'] = native_epitope_template_ids(
                site, schema[site['chain']]['sequence'], native_chains[native_chain],
                min_identity=float(target_template.get('min_identity', 0.8)),
                min_coverage=float(target_template.get('min_coverage', 0.7)),
                alignment_report=alignment_report,
            )
            site['template_alignment'] = alignment_report
            site['native_epitope_numbering'] = 'target_template_residue_ids'
            site['native_template_chain'] = native_chain
        else:
            site['native_epitope_residue_ids'] = list(site['schema_positions_1_based'])
            site['native_epitope_numbering'] = 'schema_1_based'
            site['native_template_chain'] = None
    root.mkdir(parents=True)
    inputs = root / 'inputs'
    inputs.mkdir()
    shutil.copy2(pose, inputs / 'guide.cif')
    shutil.copy2(base_path, inputs / 'base_config.yaml')
    write_json(inputs / 'target.json', schema)
    write_json(root / 'plan.json', plan)
    cfg.update(input=str(inputs / 'target.json'), msa_dir=str(msa), target_chains=list(schema),
               num_backbones=n, batch_size=args.batch_size, skip_existing=False,
               save_traj=False, save_distogram=False, save_full_confidence=False)
    cfg['binder']['framework_template'] = None
    cfg.setdefault('inverse_folder', {})['type'] = args.inverse_folder
    cfg['inverse_folder']['num_seqs'] = args.sequences_per_backbone
    cfg['inverse_folder']['backbone_noise'] = 0.0
    cfg.setdefault('data', {})['workers'] = 0
    if target_template:
        shutil.copy2(template, inputs / 'target_template.cif')
        cfg['target_template']['path'] = str(inputs / 'target_template.cif')
    jobs = []
    for label, pairs, angle, shift in selected_arms:
        for rep in range(replicates):
            job_cfg = copy.deepcopy(cfg)
            job_cfg['output'] = str(root / 'results' / label / f'rep{rep}')
            site_enabled = args.profile == 'comparison' and label != 'control'
            if args.profile == 'comparison':
                # Use Promera's *native* targeted epitope configuration, not
                # an extra feature injection after featurization. Mapping
                # schema positions to full-template author IDs means the
                # native task both sets antigen epitope tokens and pins them
                # correctly during target-template subsampling.
                job_cfg['epitope_residues'] = (
                    list(plan['targeted_site']['native_epitope_residue_ids'])
                    if site_enabled else []
                )
                if site_enabled:
                    job_cfg['epitope_chain'] = plan['targeted_site']['chain']
                job_cfg.setdefault('nominee', {})['has_epitope'] = site_enabled
            job_cfg['pose_guidance'] = {
                'plan': str(root / 'plan.json'), 'plan_sha256': digest(root / 'plan.json'),
                'pairs': pairs, 'rotation_degrees': angle, 'translation_angstrom': shift,
                'seed': args.seed + rep, 'targeted_site_enabled': site_enabled,
            }
            if label == 'cone16':
                job_cfg['pose_guidance']['approach_angle_range_degrees'] = list(interval)
                job_cfg['pose_guidance']['angle_sampling'] = angle_sampling
            path = root / 'configs' / f'{len(jobs):03d}_{label}_rep{rep}.json'
            write_json(path, job_cfg)  # JSON is also valid YAML for OmegaConf.
            jobs.append({'index': len(jobs), 'arm': label, 'replicate': rep,
                         'config': str(path.relative_to(root)), 'sha256': digest(path),
                         'requested_backbones': n, 'targeted_site_enabled': site_enabled})
    manifest = {
        'schema_version': 1, 'profile': args.profile, 'jobs': jobs,
        'plan_sha256': digest(root / 'plan.json'), 'source_config_sha256': digest(base_path),
        'guide_sha256': digest(inputs / 'guide.cif'),
        'target_sha256': digest(inputs / 'target.json'),
        'target_template_sha256': digest(inputs / 'target_template.cif') if target_template else None,
        'selection': {'approach_degrees_min': interval[0] if interval else 0,
                      'approach_degrees_max': interval[1] if interval else 25,
                      'position_comparison': 'target_patch_radius' if interval else 'source_centroid_shift',
                      'centroid_shift_angstrom_max': 6,
                      'patch_radius_change_angstrom_max': 6,
                      'framework_internal_rmsd_max': 2, 'self_binder_rmsd_max': 6,
                      'target_fit_rmsd_max': 3},
        'angular_reference': 'deviation_from_selected_source_CDR_to_framework_axis_after_target_alignment',
        'angle_range_requested_degrees': list(interval) if interval else None,
        'angle_sampling': angle_sampling,
        'comparison_primary_contrast': (
            'targeted_only_vs_cone16_same_epitope_and_seeds'
            if args.profile == 'comparison' else None
        ),
        'targeted_site': plan.get('targeted_site'),
        'targeted_site_residue_count': (
            len(plan['targeted_site']['schema_positions_1_based'])
            if plan.get('targeted_site') else 0
        ),
        'backbones_per_replicate': n,
        'replicates': replicates,
        'sequences_per_backbone': args.sequences_per_backbone,
        'inverse_folder': args.inverse_folder,
        'primary_endpoint': (
            'generated_backbone_approach_in_requested_angle_interval_AND_'
            'at_least_three_source_site_antigen_CAs_within_site_cutoff_of_generated_CDR_CAs'
            if args.profile == 'comparison' else None
        ),
        'paired_design': 'matched_random_seed_and_backbone_index_across_arms',
        'selection_note': 'Generation-guidance range is a preference. Evaluate actual unrestrained output in the same interval; these are not validated binding cutoffs.',
        'framework_template_disabled': True, 'refolds_unrestrained': True,
        'budget_note': 'Five diffusion samples per designed sequence; these share a trunk evaluation.',
        'script_sha256': {p.name: digest(p) for p in HERE.iterdir()
                          if p.suffix in {'.py', '.sh', '.slurm'} and not p.name.startswith('test_')},
    }
    write_json(root / 'manifest.json', manifest)
    print(json.dumps({'experiment': str(root), 'jobs': len(jobs),
                      'requested_backbones': n * len(jobs),
                      'requested_refolds': 0 if args.inverse_folder == 'none' else n * len(jobs) * args.sequences_per_backbone * 5,
                      'targeted_site_chain': plan['targeted_site']['chain'] if plan.get('targeted_site') else None,
                      'targeted_site_residues': len(plan['targeted_site']['schema_positions_1_based']) if plan.get('targeted_site') else 0,
                      'site_cutoff_angstrom': plan['targeted_site']['source_cdr_ca_cutoff_angstrom'] if plan.get('targeted_site') else None}, indent=2))


def job_config(root: Path, index: int) -> tuple[dict[str, Any], Any, Path]:
    manifest = load_json(root / 'manifest.json')
    if not 0 <= index < len(manifest['jobs']):
        raise ValueError('job index is outside the prepared experiment')
    path = root / manifest['jobs'][index]['config']
    checks = {path: manifest['jobs'][index]['sha256'], root / 'plan.json': manifest['plan_sha256'],
              root / 'inputs/guide.cif': manifest['guide_sha256'],
              root / 'inputs/target.json': manifest['target_sha256']}
    if manifest.get('target_template_sha256'):
        checks[root / 'inputs/target_template.cif'] = manifest['target_template_sha256']
    for filename, expected in checks.items():
        if digest(filename) != expected:
            raise ValueError(f'prepared input/config changed: {filename}; prepare a new experiment')
    for name, expected in manifest['script_sha256'].items():
        if digest(HERE / name) != expected:
            raise ValueError(f'experiment code changed: {name}; prepare a new experiment')
    return manifest, OmegaConf.load(path), path


def verify_writable_runtime() -> dict[str, str]:
    """Fail early if the isolated GPU image would write into read-only /home.

    Do not modify host HOME. Every checked path must already be redirected
    by the experimental container wrapper to the scratch bind mount.
    """
    keys = (
        'HOME', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME',
        'XDG_RUNTIME_DIR', 'TMPDIR', 'TORCH_HOME',
        'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH',
        'HF_HOME', 'MPLCONFIGDIR',
    )
    checked: dict[str, str] = {}
    for name in keys:
        value = os.environ.get(name, '').strip()
        if not value or not Path(value).is_absolute():
            raise ValueError(
                f'GPU runtime requires an absolute writable {name}, got {value!r}; '
                'launch using the updated pose_guidance/container.sh'
            )
        target = Path(value)
        try:
            target.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode='wb', dir=target, prefix='.pose-runtime-write-'
            ) as handle:
                handle.write(b'ok')
        except OSError as exc:
            raise OSError(
                f'{name}={target} is not writable inside Apptainer; '
                'do not allow GPU compilers or caches to fall back to /home/users'
            ) from exc
        checked[name] = str(target)
    # Some libraries call pathlib.Path.home() rather than reading HOME.
    if Path.home() != Path(checked['HOME']):
        raise ValueError(
            f'Path.home()={Path.home()} differs from configured HOME={checked["HOME"]}'
        )
    return checked


def verify_all_proposed_guides(cfg: Any, plan: dict[str, Any]) -> dict[str, Any]:
    """Cheap full-budget geometry check; no sequence featurization or GPU work.

    Preflight's two real featurized items only check the leading RNG draws.
    A larger run must verify all requested angular-pose draws are encodable
    before SLURM submission, using the same deterministic seeds as pose_task.
    """
    requested = int(cfg.num_backbones)
    if requested < 1:
        raise ValueError('no backbones requested')
    setting = cfg.pose_guidance
    target = np.asarray(plan['target_coords'], dtype=float)[plan['target_landmarks']]
    source_fw = np.asarray(plan['source_framework_coords'], dtype=float)
    axis = np.asarray(plan['source_axis'], dtype=float)
    patch = target.mean(0)
    binder_ids = plan['binder_landmarks']
    requested_pairs = int(setting.pairs)
    min_finite = requested_pairs
    max_overflow = 0
    sampled_angles = []
    sampled_azimuths = []
    for bidx in range(requested):
        seed = int.from_bytes(
            hashlib.sha256(f'{setting.seed}:{bidx}'.encode()).digest()[:4], 'big'
        )
        interval = setting.get('approach_angle_range_degrees')
        if interval is not None:
            framework, guide = cone_placement(
                source_fw, axis, seed, float(interval[0]), float(interval[1]),
                float(setting.translation_angstrom), pivot=patch,
                sampling_policy=str(setting.get('angle_sampling', 'uniform_solid_angle')),
                sample_index=bidx, sample_count=requested, schedule_seed=int(setting.seed),
            )
            sampled_angles.append(float(guide['guide_approach_deviation_degrees']))
            sampled_azimuths.append(float(guide['guide_azimuth_degrees']))
        else:
            framework = jitter(
                source_fw, seed, float(setting.rotation_degrees),
                float(setting.translation_angstrom)
            )
        _, _, records = cross_distogram(
            8, [0, 1, 2, 3], [4, 5, 6, 7],
            target, framework[binder_ids], requested_pairs,
        )
        informative = sum(bool(row['finite_distance_information']) for row in records)
        min_finite = min(min_finite, informative)
        max_overflow = max(max_overflow, requested_pairs - informative)
    angle_histogram = {}
    if sampled_angles:
        low, high = map(float, setting.approach_angle_range_degrees)
        for left in np.arange(low, high, 5.0):
            right = min(float(left + 5.0), high)
            angle_histogram[f'{left:g}-{right:g}'] = sum(
                left <= angle < right or (right == high and angle == right)
                for angle in sampled_angles
            )
    return {
        'checked_backbones': requested,
        'minimum_finite_guidance_pairs': min_finite,
        'maximum_overflow_guidance_pairs': max_overflow,
        'guide_angle_histogram_5_degrees': angle_histogram,
        'guide_azimuth_quadrant_counts': [
            sum(start <= angle < start + 90 for angle in sampled_azimuths)
            for start in (0, 90, 180, 270)
        ],
    }


def preflight(root: Path, index: int, require_gpu: bool = False) -> dict[str, Any]:
    _, cfg, _ = job_config(root, index)
    import torch
    from promera.data.utils import collate
    from promera.inference import design
    from pose_task import PoseGuidedDesign

    if cfg.inverse_folder.type == 'boltzif' and not hasattr(design, 'boltz_if'):
        raise ValueError('this runtime lacks the NOMINEE Boltz-IF adapter; use the current patched NOMINEE image')
    if cfg.get('antigen_symmetry') is not None and not callable(getattr(design, 'evaluate_cofold_symmetry', None)):
        raise ValueError('this image lacks the current NOMINEE declared-symmetry patch; repair the Promera image through dev')
    if len(cfg.get('target_chains') or []) > 1 and 'antigen_chains' not in inspect.signature(design.compute_dockq).parameters:
        raise ValueError('this image lacks the current NOMINEE multichain-antigen patch; repair the Promera image through dev')
    if require_gpu and not torch.cuda.is_available():
        raise ValueError('no CUDA GPU visible; run the GPU stage inside a SLURM GPU allocation')
    # The isolated container opts in during CPU preflight as well as the GPU
    # phase. This catches Apptainer HOME/mount mistakes before sbatch.
    check_runtime = require_gpu or os.environ.get('POSE_VERIFY_WRITABLE_RUNTIME') == '1'
    runtime_writable_dirs = verify_writable_runtime() if check_runtime else None
    task = PoseGuidedDesign(cfg)
    all_guide_geometry = verify_all_proposed_guides(cfg, task.plan)
    if not len(task):
        raise ValueError('no generation items; fresh runs must not inherit completed outputs')
    features = [task[i] for i in range(min(2, len(task)))]
    site = task.plan.get('targeted_site')
    expected_epitope_count = 0
    if site:
        enabled = bool(cfg.pose_guidance.get('targeted_site_enabled', False))
        expected = {pos - 1 for pos in site['schema_positions_1_based']} if enabled else set()
        expected_epitope_count = len(expected)
        for featureset in features:
            offsets, offset = {}, 0
            for name, chain in featureset['struct'].chains.items():
                offsets[name] = offset
                offset += len(chain.rname)
            site_start = offsets[site['chain']]
            n_antigen = len(featureset['struct'].chains[site['chain']].rname)
            actual = set(int(x) for x in np.flatnonzero(
                featureset['is_epitope'][site_start:site_start + n_antigen]
            ))
            if actual != expected:
                raise ValueError(
                    f'targeted comparison epitope mismatch: expected {sorted(expected)}, '
                    f'actual {sorted(actual)}'
                )
    batch = collate(features, token_multiple=int(cfg.get('pad_tokens_to_multiple', 1)))
    if int(cfg.pose_guidance.pairs):
        size = batch['restype'].shape[1]
        if tuple(batch['distogram_mask'].shape) != (len(features), size, size):
            raise ValueError('distogram did not pad on both token dimensions')
    weights = Path(os.environ.get('PROMERA_WEIGHTS', ''))
    if not weights.is_file():
        raise ValueError('PROMERA_WEIGHTS must point to the installed checkpoint')
    report = {'python': platform.python_version(), 'torch': torch.__version__,
              'cuda_visible': torch.cuda.is_available(), 'design_source': inspect.getfile(design),
              'design_sha256': digest(inspect.getfile(design)), 'weights': str(weights),
              'weights_size': weights.stat().st_size,
              'featurized_items': len(features), 'pair_count': int(cfg.pose_guidance.pairs),
              'requested_angle_sampling': cfg.pose_guidance.get('angle_sampling', 'uniform_solid_angle'),
              'requested_approach_angle_range_degrees': (
                  OmegaConf.to_container(cfg.pose_guidance.approach_angle_range_degrees, resolve=True)
                  if cfg.pose_guidance.get('approach_angle_range_degrees') is not None else None
              ),
              'inverse_folder': cfg.inverse_folder.type,
              'image_path': os.environ.get('POSE_RUNTIME_IMAGE'),
              'experiment_commit': os.environ.get('POSE_EXPERIMENT_COMMIT'),
              'runtime_writable_dirs': runtime_writable_dirs,
              'targeted_site_enabled': bool(cfg.pose_guidance.get('targeted_site_enabled', False)),
              'targeted_epitope_tokens': expected_epitope_count,
              'targeted_site_chain': site['chain'] if site else None,
              'targeted_site_schema_positions': site['schema_positions_1_based'] if site else None,
              'all_proposed_guide_geometry': all_guide_geometry}
    report['pose_anchor_diagnostics'] = [
        {
            'backbone_index': int(feat['backbone_idx']),
            'requested_pairs': int(feat['pose_record']['requested_guidance_pairs']),
            'informative_pairs': int(feat['pose_record']['informative_guidance_pairs']),
            'overflow_pairs': int(feat['pose_record']['overflow_guidance_pairs']),
            'max_distance_angstrom': feat['pose_record']['maximum_anchor_distance_angstrom'],
            'guide_angle_degrees': feat['pose_record'].get('guide_approach_deviation_degrees'),
            'guide_stratum_rank': feat['pose_record'].get('guide_stratum_rank'),
            'guide_azimuth_degrees': feat['pose_record'].get('guide_azimuth_degrees'),
        }
        for feat in features
    ]
    msa_paths = getattr(features[0]['struct'], 'msa_summary', {}).get('msa_path', {})
    report['target_msa_sha256'] = {name: digest(msa_paths[name]) for name in task.plan['target_sequences']
                                    if msa_paths.get(name) and Path(msa_paths[name]).is_file()}
    for name in ('promera.model.model', 'promera.diffusion.sampler'):
        module = importlib.import_module(name)
        report[name + '_sha256'] = digest(inspect.getfile(module))
    return report


def run(root: Path, index: int) -> None:
    _, cfg, path = job_config(root, index)
    lock_dir = root / 'locks'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f'{index}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = Path(cfg.output)
        if output.exists() and any(output.iterdir()):
            raise ValueError('output already contains results; use a new experiment, never mix or resume this pilot')
        report = preflight(root, index, require_gpu=True)
        report['weights_sha256'] = digest(report['weights'])
        report['manifest_sha256'] = digest(root / 'manifest.json')
        report['status'] = 'running'
        receipt = root / 'receipts' / f'{index}.json'
        write_json(receipt, report)
        started = time.perf_counter()
        old_argv = sys.argv
        sys.argv = ['promera', '--task', 'pose_task.PoseGuidedDesign', '--task_config', str(path), 'data.workers=0', 'trainer.devices=1']
        try:
            runpy.run_module('promera', run_name='__main__')
            report['status'] = 'completed'
        except BaseException as error:
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            sys.argv = old_argv
            report['wall_seconds'] = time.perf_counter() - started
            write_json(receipt, report)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def wilson95(successes: int, total: int) -> tuple[float | None, float | None]:
    """95% Wilson interval for a binomial backbone recovery fraction."""
    if total <= 0:
        return None, None
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z**2 / total
    mid = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, float(mid - half)), min(1.0, float(mid + half))


def comparison_summary(
    manifest: dict[str, Any], rows: list[dict[str, Any]], totals: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Describe targeted-only versus geometry guidance, not proof of binding."""
    if manifest.get('profile') != 'comparison':
        return [], {}
    arms = list(dict.fromkeys(job['arm'] for job in manifest['jobs']))
    summaries = []
    for arm in arms:
        outputs = [row for row in rows if row['kind'] == 'backbone' and row['arm'] == arm]
        task_rows = [row for row in totals if row['arm'] == arm]
        requested = sum(int(row['requested_backbones']) for row in task_rows)
        n = len(outputs)
        angle_success = sum(bool(row['in_requested_angle']) for row in outputs)
        primary_success = sum(bool(row['primary_hit']) for row in outputs)
        confidence_low, confidence_high = wilson95(primary_success, n)
        angles = [float(row['approach_degrees']) for row in outputs]
        contacts = [float(row['site_contact_fraction']) for row in outputs]
        guided = [row for row in outputs if row.get('guide_approach_deviation_degrees') is not None]
        errors = [float(row['guide_axis_error_degrees']) for row in guided
                  if row.get('guide_axis_error_degrees') is not None]
        polar_errors = [abs(float(row['approach_degrees']) -
                            float(row['guide_approach_deviation_degrees'])) for row in guided]
        planned = [float(row['guide_approach_deviation_degrees']) for row in guided]
        observed = [float(row['approach_degrees']) for row in guided]
        correlation = (
            float(np.corrcoef(planned, observed)[0, 1])
            if len(guided) >= 3 and np.std(planned) > 0 and np.std(observed) > 0
            else None
        )
        summaries.append({
            'arm': arm,
            'targeted_site_enabled': bool(task_rows[0]['targeted_site_enabled']),
            'replicates_requested': len(task_rows),
            'replicates_completed': sum(row['status'] == 'completed' for row in task_rows),
            'status': (
                'completed' if n == requested and all(row['status'] == 'completed' for row in task_rows)
                else 'partial_or_not_run'
            ),
            'requested_backbones': requested,
            'generated_backbones': n,
            'in_angle_range': angle_success,
            'in_angle_range_fraction': angle_success / n if n else None,
            'source_site_contact_3plus': sum(int(row['site_contacted_residues']) >= 3 for row in outputs),
            'primary_geometry_and_site_hits': primary_success,
            'primary_hit_fraction': primary_success / n if n else None,
            'primary_hit_wilson95_low': confidence_low,
            'primary_hit_wilson95_high': confidence_high,
            'median_approach_degrees': statistics.median(angles) if angles else None,
            'median_guide_axis_error_degrees': statistics.median(errors) if errors else None,
            'median_abs_polar_error_degrees': statistics.median(polar_errors) if polar_errors else None,
            'guide_recovered_polar_correlation': correlation,
            'median_source_site_contact_fraction': statistics.median(contacts) if contacts else None,
            'gpu_wall_seconds': sum(float(row['wall_seconds'] or 0) for row in task_rows),
        })
    by_arm = {row['arm']: row for row in summaries}
    target, cone = by_arm['targeted'], by_arm['cone16']
    absolute = (
        float(cone['primary_hit_fraction'] - target['primary_hit_fraction'])
        if cone['primary_hit_fraction'] is not None and target['primary_hit_fraction'] is not None
        else None
    )
    return summaries, {
        'primary_comparison': 'cone16_vs_targeted_same_CDR_derived_antigen_site',
        'angle_range_degrees': manifest['angle_range_requested_degrees'],
        'source_site': manifest['targeted_site'],
        'absolute_primary_hit_fraction_difference': absolute,
        'complete': all(row['status'] == 'completed' for row in summaries),
        'interpretation': (
            'Descriptive comparison of generation geometry and antigen-site '
            'proximity only; not an affinity, neutralization or refold success test.'
        ),
    }


def angle_response(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Five-degree bins test whether generated backbones track *specific* guides."""
    interval = manifest.get('angle_range_requested_degrees')
    if not interval:
        return []
    lower, maximum = map(float, interval)
    guides = [row for row in rows if row['kind'] == 'backbone' and row['arm'] == 'cone16'
              and row.get('guide_approach_deviation_degrees') is not None]
    summaries = []
    while lower < maximum:
        upper = min(lower + 5.0, maximum)
        sample = [row for row in guides
                  if lower <= float(row['guide_approach_deviation_degrees'])
                  and (float(row['guide_approach_deviation_degrees']) < upper
                       or upper == maximum and float(row['guide_approach_deviation_degrees']) <= upper)]
        planned = [float(row['guide_approach_deviation_degrees']) for row in sample]
        actual = [float(row['approach_degrees']) for row in sample]
        errors = [float(row['guide_axis_error_degrees']) for row in sample
                  if row.get('guide_axis_error_degrees') is not None]
        summaries.append({
            'guide_angle_min': lower, 'guide_angle_max': upper,
            'observed_backbones': len(sample),
            'median_guide_angle': statistics.median(planned) if planned else None,
            'median_recovered_angle': statistics.median(actual) if actual else None,
            'median_abs_polar_error_degrees': (
                statistics.median(abs(a - b) for a, b in zip(planned, actual))
                if sample else None
            ),
            'median_guide_axis_error_degrees': statistics.median(errors) if errors else None,
        })
        lower = upper
    return summaries


def analyse(root: Path) -> None:
    manifest = load_json(root / 'manifest.json')
    plan = load_json(root / 'plan.json')
    if digest(root / 'plan.json') != manifest['plan_sha256']:
        raise ValueError('pose plan changed')
    threshold = manifest['selection']
    patch_center = np.asarray(plan['target_coords'], dtype=float)[plan['target_landmarks']].mean(0)
    rows, totals, exports, errors = [], [], {}, []
    runtime_reference = None
    for job in manifest['jobs']:
        _, cfg, _ = job_config(root, job['index'])
        output = Path(cfg.output)
        receipt_path = root / 'receipts' / f"{job['index']}.json"
        receipt = load_json(receipt_path) if receipt_path.exists() else {}
        if receipt:
            runtime = {key: receipt.get(key) for key in ('python', 'torch', 'design_sha256',
                       'weights_sha256', 'target_msa_sha256', 'promera.model.model_sha256',
                       'promera.diffusion.sampler_sha256')}
            if runtime_reference is None:
                runtime_reference = runtime
            elif runtime != runtime_reference:
                errors.append({'path': str(receipt_path), 'error': 'mixed runtime/checkpoint/MSA inputs across conditions'})
        total = dict(job, generated_backbones=0, refolds=0, geometry_matched_refolds=0,
                     unique_matched_sequences=0, matched_backbones=0,
                     in_range_backbones=0, site_contacted_backbones=0, primary_hits=0,
                     status=receipt.get('status', 'not_run'), wall_seconds=receipt.get('wall_seconds', ''))
        designed = sorted(output.rglob('sample*_design*.fasta'))
        total['designed_sequences'] = len(designed)
        total['expected_refolds_from_sequences'] = len(designed) * 5
        matched_sequences = set()
        for backbone in sorted(output.rglob('backbone.cif')):
            total['generated_backbones'] += 1
            try:
                observed_backbone = read_chains(backbone)
                bt, bf, bb = observed_arrays(observed_backbone, plan)
                guidance_file = backbone.parent / 'pose_guidance.json'
                guidance = load_json(guidance_file) if guidance_file.is_file() else {}
                bm = pose_metrics(plan['target_coords'], plan['source_framework_coords'], bt, bf,
                                  plan['source_axis'], target_patch_center=patch_center,
                                  guide_axis=guidance.get('guide_axis_xyz'))
                site_metrics = site_contact_metrics(observed_backbone, plan)
                angle_in_range = (
                    threshold['approach_degrees_min'] - 1e-5
                    <= bm['approach_degrees']
                    <= threshold['approach_degrees_max'] + 1e-5
                )
                site_hits = (
                    int(site_metrics['site_contacted_residues']) >= 3
                    if plan.get('targeted_site') else False
                )
                primary_hit = bool(angle_in_range and site_hits) if plan.get('targeted_site') else None
                backbone_match = _passes_geometry(bm, threshold)
                guide_fields = {
                    'guide_approach_deviation_degrees': guidance.get('guide_approach_deviation_degrees'),
                    'guide_azimuth_degrees': guidance.get('guide_azimuth_degrees'),
                    'guide_sampling_policy': guidance.get('guide_sampling_policy'),
                    'guide_stratum_rank': guidance.get('guide_stratum_rank'),
                    'requested_guidance_pairs': guidance.get('requested_guidance_pairs'),
                    'informative_guidance_pairs': guidance.get('informative_guidance_pairs'),
                    'overflow_guidance_pairs': guidance.get('overflow_guidance_pairs'),
                    'maximum_anchor_distance_angstrom': guidance.get('maximum_anchor_distance_angstrom'),
                }
                total['matched_backbones'] += int(backbone_match)
                total['in_range_backbones'] += int(angle_in_range)
                total['site_contacted_backbones'] += int(site_hits)
                total['primary_hits'] += int(bool(primary_hit))
                rows.append(dict(arm=job['arm'], replicate=job['replicate'], kind='backbone',
                                 cif=str(backbone), geometry_match=backbone_match,
                                 targeted_site_enabled=job.get('targeted_site_enabled', False),
                                 in_requested_angle=angle_in_range, primary_hit=primary_hit,
                                 **site_metrics, **guide_fields, **bm))
                for confidence_path in sorted((backbone.parent / 'refolds').glob('*_confidence.json')):
                    total['refolds'] += 1
                    cif = confidence_path.with_name(confidence_path.name.replace('_confidence.json', '.cif'))
                    try:
                        chains = read_chains(cif)
                        rt, rf, rb = observed_arrays(chains, plan)
                        metrics = pose_metrics(plan['target_coords'], plan['source_framework_coords'], rt, rf,
                                               plan['source_axis'], target_patch_center=patch_center,
                                               guide_axis=guidance.get('guide_axis_xyz'))
                        refold_site_metrics = site_contact_metrics(chains, plan)
                        rotation, translation = align(rt, bt)
                        self_rmsd = rmsd(rb @ rotation + translation, bb)
                        geometry_match = _passes_geometry(metrics, threshold)
                        eligible = geometry_match and self_rmsd <= threshold['self_binder_rmsd_max']
                        confidence = load_json(confidence_path)
                        iptm = confidence.get('complex_iptm')
                        if not isinstance(iptm, (int, float)) or not np.isfinite(iptm):
                            iptm = None
                        sequence = chains[plan['binder_chain']]['sequence']
                        row = dict(arm=job['arm'], replicate=job['replicate'], kind='refold', cif=str(cif),
                                   sequence=sequence, iptm=iptm, scDockQ_json=json.dumps(confidence.get('scDockQ')),
                                   self_binder_rmsd=self_rmsd, geometry_match=geometry_match,
                                   geometry_and_self_consistency=eligible,
                                   targeted_site_enabled=job.get('targeted_site_enabled', False),
                                   in_requested_angle=(
                                       threshold['approach_degrees_min'] - 1e-5
                                       <= metrics['approach_degrees']
                                       <= threshold['approach_degrees_max'] + 1e-5
                                   ),
                                   **refold_site_metrics, **guide_fields, **metrics)
                        rows.append(row)
                        if eligible:
                            total['geometry_matched_refolds'] += 1
                            matched_sequences.add(sequence)
                            current = exports.get(sequence)
                            score = iptm if iptm is not None else -1
                            if current is None or score > current['_score']:
                                exports[sequence] = {'name': 'pose_' + hashlib.sha256(sequence.encode()).hexdigest()[:16],
                                                     'sequence': sequence, 'promera_reference_cif': str(cif),
                                                     'arm': job['arm'], 'replicate': job['replicate'], 'iptm': iptm,
                                                     '_score': score}
                    except (ValueError, KeyError, OSError) as error:
                        errors.append({'path': str(cif), 'error': str(error)})
            except (ValueError, KeyError, OSError) as error:
                errors.append({'path': str(backbone), 'error': str(error)})
        total['unique_matched_sequences'] = len(matched_sequences)
        total['missing_refolds_from_sequences'] = max(0, total['expected_refolds_from_sequences'] - total['refolds'])
        total['missing_backbones'] = job['requested_backbones'] - total['generated_backbones']
        total['geometry_yield_per_requested_backbone'] = total['matched_backbones'] / job['requested_backbones']
        total['primary_hit_fraction'] = total['primary_hits'] / total['generated_backbones'] if total['generated_backbones'] else None
        total['evaluated_angle_min_degrees'] = threshold.get('approach_degrees_min', 0)
        total['evaluated_angle_max_degrees'] = threshold['approach_degrees_max']
        totals.append(total)
    destination = root / 'analysis'
    destination.mkdir(exist_ok=True)
    write_csv(destination / 'structures.csv', rows, ['arm', 'replicate', 'kind', 'cif', 'sequence', 'iptm',
        'scDockQ_json', 'target_fit_rmsd', 'source_framework_rmsd', 'framework_internal_rmsd',
        'centroid_shift_angstrom', 'patch_radius_change_angstrom', 'body_rotation_degrees', 'approach_degrees',
        'guide_axis_error_degrees',
        'guide_approach_deviation_degrees', 'guide_azimuth_degrees', 'guide_sampling_policy',
        'guide_stratum_rank',
        'requested_guidance_pairs', 'informative_guidance_pairs', 'overflow_guidance_pairs',
        'maximum_anchor_distance_angstrom', 'targeted_site_enabled',
        'in_requested_angle', 'primary_hit',
        'site_contacted_residues', 'site_contact_fraction',
        'site_contact_share_of_all_contacts', 'cdr_residue_count',
        'self_binder_rmsd',
        'geometry_match', 'geometry_and_self_consistency'])
    write_csv(destination / 'conditions.csv', totals, ['index', 'arm', 'replicate', 'status', 'requested_backbones',
        'generated_backbones', 'missing_backbones', 'matched_backbones', 'geometry_yield_per_requested_backbone',
        'designed_sequences', 'expected_refolds_from_sequences', 'refolds', 'missing_refolds_from_sequences',
        'geometry_matched_refolds', 'unique_matched_sequences',
        'in_range_backbones', 'site_contacted_backbones', 'primary_hits', 'primary_hit_fraction',
        'targeted_site_enabled',
        'evaluated_angle_min_degrees', 'evaluated_angle_max_degrees', 'wall_seconds'])
    write_csv(destination / 'protenix_inputs.csv', list(exports.values()),
              ['name', 'sequence', 'promera_reference_cif', 'arm', 'replicate', 'iptm'])
    write_csv(destination / 'angle_response.csv', angle_response(manifest, rows),
              ['guide_angle_min', 'guide_angle_max', 'observed_backbones',
               'median_guide_angle', 'median_recovered_angle',
               'median_abs_polar_error_degrees', 'median_guide_axis_error_degrees'])
    if manifest.get('profile') == 'comparison':
        comparisons, effect = comparison_summary(manifest, rows, totals)
        write_csv(destination / 'comparison.csv', comparisons, [
            'arm', 'targeted_site_enabled', 'replicates_requested',
            'replicates_completed', 'status', 'requested_backbones',
            'generated_backbones', 'in_angle_range', 'in_angle_range_fraction',
            'source_site_contact_3plus', 'primary_geometry_and_site_hits',
            'primary_hit_fraction', 'primary_hit_wilson95_low',
            'primary_hit_wilson95_high', 'median_approach_degrees',
            'median_guide_axis_error_degrees', 'median_abs_polar_error_degrees',
            'guide_recovered_polar_correlation',
            'median_source_site_contact_fraction', 'gpu_wall_seconds',
        ])
        write_json(destination / 'comparison.json', effect)
    write_json(destination / 'errors.json', errors)
    incomplete_jobs = [
        {'index': row['index'], 'arm': row['arm'], 'replicate': row['replicate'],
         'status': row['status'], 'missing_backbones': row['missing_backbones']}
        for row in totals
        if row['status'] != 'completed' or row['missing_backbones'] != 0
    ]
    complete = not incomplete_jobs and not errors
    status = {
        'status': 'complete' if complete else 'incomplete',
        'requested_backbones': sum(row['requested_backbones'] for row in totals),
        'generated_backbones': sum(row['generated_backbones'] for row in totals),
        'incomplete_jobs': incomplete_jobs,
        'parse_errors': len(errors),
        'analysis_directory': str(destination),
    }
    write_json(destination / 'status.json', status)
    print(json.dumps({
        **status, 'geometry_matched_unique_sequences': len(exports),
        'note': 'Exports are screening inputs, not validated binders or production passers.',
    }, indent=2))
    if not complete:
        raise ValueError(
            'pose-guidance analysis is incomplete; partial reports were written. '
            'Inspect analysis/status.json and analysis/errors.json before interpreting yields'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--base-config', required=True)
    prep.add_argument('--pose', required=True)
    prep.add_argument('--guide-binder-chain')
    prep.add_argument('--target-map', help='JSON map: runtime chain -> guide chain; default identical IDs')
    prep.add_argument('--msa-dir')
    prep.add_argument('--out', required=True)
    prep.add_argument('--profile', choices=['smoke', 'pilot', 'comparison'], default='smoke')
    prep.add_argument('--site-cutoff-angstrom', type=float, default=10.0,
                      help='comparison: source-CDR CA neighbourhood for site candidates (default 10 A)')
    prep.add_argument('--site-residues', type=int, default=7,
                      help='comparison: number of nearest source-CDR antigen residues to target (default 7)')
    prep.add_argument('--angle-range', metavar='MIN:MAX',
                      help='source-relative angular guidance interval in degrees (e.g. 15:35)')
    prep.add_argument('--angle-sampling', choices=('uniform_solid_angle', 'stratified_degrees'),
                      default='uniform_solid_angle',
                      help='stratified_degrees: even polar-angle coverage and dispersed azimuths')
    prep.add_argument('--angle-translation-angstrom', type=float, default=0.0,
                      help='maximum rigid translation of the angular guide (default: 0)')
    prep.add_argument('--backbones', type=int)
    prep.add_argument('--replicates', type=int)
    prep.add_argument('--batch-size', type=int, default=1)
    prep.add_argument('--seed', type=int, default=20261007)
    prep.add_argument('--inverse-folder', choices=['boltzif', 'abmpnn', 'proteinmpnn', 'none'], default='boltzif')
    prep.add_argument('--sequences-per-backbone', type=int, default=1)
    for command in ('preflight', 'run', 'analyse'):
        item = sub.add_parser(command)
        item.add_argument('--experiment', required=True)
        if command != 'analyse':
            item.add_argument('--job', type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args)
        elif args.command == 'run':
            run(Path(args.experiment).resolve(), args.job)
        elif args.command == 'preflight':
            print(json.dumps(preflight(Path(args.experiment).resolve(), args.job), indent=2))
        else:
            analyse(Path(args.experiment).resolve())
    except (ValueError, OSError, KeyError) as error:
        # A full traceback is essential for GPU-only filesystem/library errors:
        # do not collapse the responsible call site into just "[Errno 30]".
        if isinstance(error, OSError):
            traceback.print_exc()
        parser.exit(2, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
