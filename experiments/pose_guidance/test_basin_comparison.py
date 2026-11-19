"""CPU-only contracts for the B004 basin targeting / analysis additions."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

import experiment
import pose_geometry as geom
import basin_analysis as ba


def test_author_residue_mapping_rejects_wrong_positions():
    target = 'ACDE'
    template = {'sequence': 'XACDEY',
                'residue_ids': ['101', '270', '271', '272', '273', '274']}
    author_map = {'339': '270', '343': '272', '345': '274'}
    aligner = lambda a,b: ({0:1,1:2,2:3,3:4}, {'identity':1.0,'coverage':1.0})
    site = geom.basin_site_from_author_ids(
        target, template, ['339', '343'], 'A',
        author_to_native_ids=author_map, aligner=aligner,
    )
    assert site['schema_positions_1_based'] == [1,3]
    assert site['native_epitope_residue_ids'] == ['270','272']
    assert site['epitope_scan_author_residue_ids'] == ['339','343']
    assert site['selection_policy'] == 'epitope_scan_basin_recommended_author_residues'
    with pytest.raises(ValueError, match='absent'):
        geom.basin_site_from_author_ids(
            target, template, ['345'],'A', author_to_native_ids=author_map,
            aligner=aligner)
    with pytest.raises(ValueError, match='distinct'):
        geom.basin_site_from_author_ids(
            target,template,['339','339'],'A',author_to_native_ids=author_map,
            aligner=aligner)




def test_real_mmcif_author_and_native_numbering_are_distinct(tmp_path):
    # Synthetic mmCIF uses the same author→label offset as the 8VWP template:
    # auth A:339/A:343 -> label_seq_id 272/276, NOT native 339/343.
    text = """data_test
#
loop_
_atom_site.group_PDB
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM A A 272 339 ? 1 C CA LEU 1.0 2.0 3.0
ATOM A A 276 343 ? 2 C CA VAL 4.0 5.0 6.0
#
"""
    cif = tmp_path/'author-label.cif'
    cif.write_text(text)
    result = geom.template_author_to_native_ids(cif, 'A')
    assert result['339'] == '272'
    assert result['343'] == '276'
    assert result['339'] != '339'


def test_basin_author_parser():
    assert experiment.parse_basin_residues('A:339,A:343') == ('A',['339','343'])
    with pytest.raises(ValueError, match='one chain'):
        experiment.parse_basin_residues('A:339,B:343')
    with pytest.raises(ValueError, match='must resemble'):
        experiment.parse_basin_residues('339,343')


def test_epitope_scan_classification_exact_boundaries():
    assert ba._reference_label(8.5,0.80) == 'cdr_driven'
    assert ba._reference_label(20.0,0.61) == 'cdr_driven'
    assert ba._reference_label(20.0,0.60) == 'mixed_or_unclassified'
    assert ba._reference_label(30.0,0.90) == 'framework_driven'
    assert ba._reference_label(25.0,0.49) == 'framework_driven'
    assert ba._reference_label(None,0.9) == 'no_contacts'


def test_b004_proxy_requires_cdr_driven_and_preserves_partial_contact():
    plan = {
        'targeted_site': {'chain':'A','schema_positions_1_based':[5,6]},
        'basin_reference': {
            'basin_id':'B004',
            'regions': {
                'B004': {'footprint_schema_positions':[5,6,7,8,9]},
                'B003': {'footprint_schema_positions':[1,2,3]},
            },
        },
    }
    contacts = {
        'epitope_scan_class':'cdr_driven',
        'cdr_contacted_antigen':{('A',4),('A',7)},
        'contacted_antigen':{('A',4),('A',7)},
    }
    row = ba._region_metrics(contacts,plan,8.2)
    assert row['target_residue_cdr_contacts'] == 1
    assert row['nearest_historical_footprint'] == 'B004'
    assert row['b004_like_proxy'] is True  # Does not require both sites.
    assert ba._region_metrics(contacts,plan,25)['b004_like_proxy'] is False
    contacts['epitope_scan_class'] = 'framework_driven'
    assert ba._region_metrics(contacts,plan,8.2)['b004_like_proxy'] is False


def test_basin_signature_preserves_full_historical_footprints(tmp_path, monkeypatch):
    path = tmp_path/'epitope_basins.npz'
    np.savez(
        path, basin_id=np.asarray(['B003','B004']),
        residue_chain=np.asarray(['A']*5),
        residue_author_number=np.asarray(['100','101','339','343','345']),
        basin_residue_indices_flat=np.asarray([0,1,2,3,4]),
        basin_residue_offsets=np.asarray([0,2,5]),
        basin_core_residue_indices_flat=np.asarray([0,2,3]),
        basin_core_residue_offsets=np.asarray([0,1,3]),
    )
    module = types.ModuleType('promera.inference.utils')
    module.align_schema_to_template = lambda a,b: (
        {i:i for i in range(len(a))},{'identity':1.0,'coverage':1.0})
    for name in ['promera', 'promera.inference']:
        monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    monkeypatch.setitem(sys.modules,'promera.inference.utils',module)
    template = {'sequence':'ACDEF', 'residue_ids':['900','901','270','272','273']}
    author_map = {'100':'900','101':'901','339':'270','343':'272','345':'273'}
    signature = experiment.basin_reference_from_npz(
        path,'B004','A','ACDEF',template,author_map)
    assert signature['regions']['B004']['footprint_schema_positions'] == [3,4,5]
    assert signature['regions']['B004']['core_schema_positions'] == [3,4]
    assert signature['regions']['B003']['footprint_schema_positions'] == [1,2]
    with pytest.raises(ValueError, match='B008'):
        experiment.basin_reference_from_npz(
            path,'B008','A','ACDEF',template,author_map)


def test_cpu_analysis_submission_uses_afterok():
    path = Path(__file__).with_name('analyse.slurm')
    text = path.read_text()
    assert '#SBATCH --partition=regular' in text
    assert '#SBATCH --gres=' not in text
    assert 'container.sh" analyse' in text
    launch = Path(__file__).with_name('launch_from_atlas.py').read_text()
    assert 'afterok:{array_job_id}' in launch
    assert 'basin-analysis.zip' in launch
