"""CPU-only repeat contracts: stratified cone coverage and audit-only target gate."""
from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import basin_analysis as ba
import pose_geometry as geometry
from test_pose_guidance import task_module, prepared


def test_cone_stratifies_solid_angle_and_azimuth_per_replicate():
    n = 64
    first = [geometry.stratified_cone_angles(i, n, 100, 0, 20) for i in range(n)]
    second = [geometry.stratified_cone_angles(i, n, 100, 0, 20) for i in range(n)]
    assert first == second
    angles, azimuths, polar_bins, azimuth_bins = map(list, zip(*first))
    assert set(polar_bins) == set(range(n))
    assert set(azimuth_bins) == set(range(n))
    assert min(angles) < 5
    assert max(angles) > 19
    assert all(0 <= angle <= 20 for angle in angles)
    assert all(0 <= phi < 360 for phi in azimuths)
    assert {int(phi // 90) for phi in azimuths} == {0, 1, 2, 3}
    assert first != [geometry.stratified_cone_angles(i, n, 101, 0, 20) for i in range(n)]
    with pytest.raises(ValueError, match='outside requested'):
        geometry.stratified_cone_angles(n, n, 100, 0, 20)


def test_generated_guide_preserves_requested_theta_phi_and_measures_them():
    target = np.array([[0, 0, 0], [8, 0, 0], [0, 8, 0], [0, 0, 8]], dtype=float)
    framework = np.array(
        [[-3, -3, 20], [3, -2, 20], [1, 4, 22], [0, 1, 27], [5, 0, 23]],
        dtype=float,
    )
    axis = np.array([0.0, 0.0, 1.0])
    guided, info = geometry.cone_placement(
        framework, axis, 42, 0, 20, pivot=target.mean(axis=0),
        sampled_angle_degrees=17.5, sampled_azimuth_degrees=250,
    )
    assert info['guide_sampling_policy'] == 'stratified_solid_angle'
    assert info['guide_approach_deviation_degrees'] == pytest.approx(17.5, abs=1e-7)
    assert info['guide_azimuth_degrees'] == pytest.approx(250, abs=1e-7)
    observed = geometry.pose_metrics(target, framework, target, guided, axis)
    assert observed['approach_degrees'] == pytest.approx(17.5, abs=1e-7)
    assert observed['approach_azimuth_degrees'] == pytest.approx(250, abs=1e-7)
    with pytest.raises(ValueError, match='both stratified'):
        geometry.cone_placement(
            framework, axis, 42, 0, 20, sampled_angle_degrees=4,
        )


def test_native_rejection_is_audited_not_silently_changed(task_module):
    module, _ = task_module
    design = sys.modules['promera.inference.design']
    def original(*args, **kwargs):
        return {
            'status':'reject', 'reason':'no_epitope_contacts',
            'failing_epitope_residues':['272', '276'],
            'distance_cutoff_angstrom':8,
        }
    design.evaluate_targeted_backbone_viability = original
    with module.allow_guided_backbones_past_native_gate():
        changed = design.evaluate_targeted_backbone_viability()
        assert changed['status'] == 'bypassed'
        assert changed['native_status'] == 'reject'
        assert changed['native_reason'] == 'no_epitope_contacts'
        assert changed['native_failing_epitope_residues'] == ['272','276']
    assert design.evaluate_targeted_backbone_viability is original
    with pytest.raises(RuntimeError):
        with module.allow_guided_backbones_past_native_gate():
            assert design.evaluate_targeted_backbone_viability()['status'] == 'bypassed'
            raise RuntimeError('simulate downstream crash')
    assert design.evaluate_targeted_backbone_viability is original


def test_angle_audit_counts_real_measurement_not_requested_bin():
    plan = {'angle_range_requested_degrees':[0,20],
            'jobs':[{'arm':'cone16','replicate':0}]}
    entries = [
        {
            'arm':'cone16','replicate':0,
            'source_relative_angle_degrees': observed,
            'source_relative_azimuth_degrees': azim,
            'guide_approach_deviation_degrees': request,
            'guide_azimuth_degrees': azim,
        }
        for observed, request, azim in [
            (2,3,10),(8,6,100),(12,12,190),(19,17,280),(23,19,330)
        ]
    ]
    result = ba._angular_audit(entries, plan)
    assert len(result)==1
    item = result[0]
    assert item['measured_angle_beyond_max']==1
    assert [item[f'measured_angle_bin{k}'] for k in range(1,5)]==[1,1,1,1]
    assert item['measured_azimuth_quadrants_covered']==4
    assert item['measured_angle_median_degrees']==12
    assert item['requested_angles']==5
    assert math.isfinite(item['requested_vs_measured_angle_r'])



@pytest.mark.parametrize('diagnostic', [False, True])
def test_b004_pose_only_guides_preserve_native_targeted_control(
    prepared, monkeypatch, task_module, diagnostic,
):
    """Only the targeted control gets A339/A343 epitope tokens; the pose
    arms retain distance guidance, with the native targeted gate disabled
    by the absence of target residues (not a bypass patch).
    """
    import argparse
    import json
    from pathlib import Path
    import experiment
    from test_pose_guidance import fixtures, synthetic_features

    chains, _, _ = fixtures()
    template_path = prepared.parent / 'template.cif'
    template_path.write_text('mocked target template')
    basin_npz = prepared.parent / 'epitope_basins.npz'
    basin_npz.write_bytes(b'mocked basin signatures')

    cfg_path = prepared.parent / 'base.yaml'
    cfg = json.loads(cfg_path.read_text())
    cfg['target_template'] = {
        'path': str(template_path), 'chain': 'K',
        'min_identity': 0.8, 'min_coverage': 0.7,
    }
    cfg_path.write_text(json.dumps(cfg))

    t_chain = {
        'sequence': 'ACDEF',
        'coords': np.asarray(chains['L']['coords'], dtype=float),
        'residue_ids': ['1', '2', '3', '4', '5'],
    }
    monkeypatch.setattr(
        experiment, 'read_chains',
        lambda path: {'K': t_chain} if Path(path) == template_path else chains,
    )
    monkeypatch.setattr(
        experiment, 'template_author_to_native_ids',
        lambda path, chain: {'339': '1', '343': '3'},
    )
    monkeypatch.setattr(
        experiment, 'basin_site_from_author_ids',
        lambda *args, **kwargs: {
            'chain': 'K',
            'schema_positions_1_based': [1, 3],
            'native_epitope_residue_ids': ['1', '3'],
            'epitope_scan_author_residue_ids': ['339', '343'],
            'source_cdr_ca_cutoff_angstrom': 10.0,
        },
    )
    monkeypatch.setattr(
        experiment, 'basin_reference_from_npz',
        lambda *args, **kwargs: {
            'basin_id': 'B004', 'target_chain': 'K',
            'selected_basin_author_ids': ['339', '343'],
            'regions': {'B004': {
                'footprint_schema_positions': [1, 2, 3],
                'core_schema_positions': [1, 3],
            }},
            'npz_sha256': 'synthetic',
        },
    )
    args = argparse.Namespace(
        base_config=str(cfg_path),
        pose=str(prepared.parent / 'guide.cif'),
        guide_binder_chain='N',
        target_map='{"K":"L"}',
        msa_dir=None,
        out=str(prepared.parent / 'pose-only-basin'),
        backbones=2, replicates=2, batch_size=1,
        profile='basin', seed=100,
        inverse_folder='boltzif', sequences_per_backbone=1,
        angle_range='0:20', angle_translation_angstrom=0.0,
        basin_id='B004', basin_residues='K:339,K:343',
        basin_npz=str(basin_npz), pose_only_guides=True,
        stratified_cone=True, bypass_guided_target_gate=False,
        diagnostic_targeted_cone=diagnostic,
    )
    experiment.prepare(args)
    root = Path(args.out)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['pose_only_guides'] is True
    assert manifest['diagnostic_targeted_cone'] is diagnostic
    expected_arms = (
        ['targeted', 'targeted', 'anchors16', 'anchors16', 'cone16', 'cone16']
        + (['targeted_cone16', 'targeted_cone16'] if diagnostic else [])
    )
    assert [j['arm'] for j in manifest['jobs']] == expected_arms
    expected_conditioning = {
        'targeted': {
            'native_epitope_targeting': True, 'sparse_distance_pairs': 0,
            'target_gate_policy': 'native',
        },
        'anchors16': {
            'native_epitope_targeting': False, 'sparse_distance_pairs': 16,
            'target_gate_policy': 'not_applicable',
        },
        'cone16': {
            'native_epitope_targeting': False, 'sparse_distance_pairs': 16,
            'target_gate_policy': 'not_applicable',
        },
    }
    if diagnostic:
        expected_conditioning['targeted_cone16'] = {
            'native_epitope_targeting': True, 'sparse_distance_pairs': 16,
            'target_gate_policy': 'audit_only',
        }
    assert manifest['arm_conditioning'] == expected_conditioning
    module, _ = task_module
    for j in manifest['jobs']:
        _, config, _ = experiment.job_config(root, j['index'])
        wanted = j['arm'] in {'targeted', 'targeted_cone16'}
        assert list(config.epitope_residues) == (['1', '3'] if wanted else [])
        assert bool(config.nominee.has_epitope) is wanted
        assert bool(config.pose_guidance.targeted_site_enabled) is wanted
        assert bool(config.pose_guidance.bypass_targeted_backbone_gate) is (
            j['arm'] == 'targeted_cone16'
        )
        if j['arm'] == 'targeted_cone16':
            assert not bool(config.pose_guidance.pose_only_guidance)
            assert config.pose_guidance.pairs == 16
            assert config.pose_guidance.stratified_cone is True
            _, baseline, _ = experiment.job_config(
                root, next(row['index'] for row in manifest['jobs']
                           if row['arm'] == 'cone16'
                           and row['replicate'] == j['replicate'])
            )
            # Same guide source, seed, sampling and angle distribution.
            for key in (
                'seed', 'pairs', 'approach_angle_range_degrees',
                'stratified_cone', 'translation_angstrom',
                'rotation_degrees', 'plan_sha256',
            ):
                assert config.pose_guidance[key] == baseline.pose_guidance[key]
        if not wanted:
            task = module.PoseGuidedDesign(config)
            task.features = [synthetic_features(
                'ACDEGGFGHIAAKLMNTTPQRS', 0,
            )]
            features = task[0]
            assert not features['is_epitope'].any()
            assert features['pose_record']['requested_guidance_pairs'] == 16
            if j['arm'] == 'cone16':
                assert config.pose_guidance.stratified_cone is True


def test_pose_only_and_gate_bypass_are_mutually_exclusive(prepared):
    import argparse
    import experiment
    from pathlib import Path
    # Early failure happens before any output directory is created.
    args = argparse.Namespace(
        base_config=str(prepared.parent / 'base.yaml'),
        pose=str(prepared.parent / 'guide.cif'),
        guide_binder_chain='N', target_map='{"K":"L"}',
        msa_dir=None, out=str(prepared.parent / 'invalid-run'),
        backbones=2, replicates=1, batch_size=1,
        profile='smoke', seed=100,
        inverse_folder='boltzif', sequences_per_backbone=1,
        angle_range=None, angle_translation_angstrom=0.0,
        pose_only_guides=True, bypass_guided_target_gate=True,
    )
    with pytest.raises(ValueError, match='only supported for the B004'):
        experiment.prepare(args)
