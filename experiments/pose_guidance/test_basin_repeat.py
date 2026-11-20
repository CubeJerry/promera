"""CPU-only repeat contracts: stratified cone coverage and audit-only target gate."""
from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import basin_analysis as ba
import pose_geometry as geometry
from test_pose_guidance import task_module


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
