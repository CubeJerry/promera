"""Synthetic CPU contract tests; these do not validate model quality or GPU inference."""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment
import pose_geometry as g

PATTERN = 'ACDE<cdrh1>FGHI<cdrh2>KLMN<cdrh3>PQRS'
SEQ = 'ACDEGGFGHIAAKLMNTTPQRS'


def fixtures():
    t = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5], [4, 3, 2]], dtype=float)
    b = np.array([[i * 0.7, 2 * np.sin(i), 20 + 2 * np.cos(i)] for i in range(len(SEQ))])
    chains = {'L': {'sequence': 'ACDEF', 'coords': t}, 'N': {'sequence': SEQ, 'coords': b}}
    schema = {'K': {'type': 'protein', 'sequence': 'ACDEF'}}
    plan = g.build_plan(chains, schema, PATTERN, 'V', 'N', {'K': 'L'})
    return chains, schema, plan


def test_framework_mapping_varied_lengths():
    f, c = g.framework_map(SEQ, PATTERN)
    assert ''.join(SEQ[i] for i in f) == 'ACDEFGHIKLMNPQRS'
    assert ''.join(SEQ[i] for i in c) == 'GGAATT'
    longer = SEQ.replace('GG', 'GGGG')
    lf, _ = g.framework_map(longer, PATTERN)
    assert len(f) == len(lf)
    assert lf[-1] == f[-1] + 2


@pytest.mark.parametrize('seq,pattern', [
    ('WRONG', PATTERN), ('ACQQDEQQDEQQFGQQHI', 'AC<cdrh1>DE<cdrh2>FG<cdrh3>HI'),
    (SEQ, 'AC<cdrh1>DE'), (SEQ, '<cdrh1>DE<cdrh2>FG<cdrh3>HI')])
def test_invalid_framework(seq, pattern):
    with pytest.raises(ValueError):
        g.framework_map(seq, pattern)


@pytest.mark.parametrize('query,source,expected', [('ABC', 'ABC', [0, 1, 2]), ('BC', 'ABCD', [1, 2])])
def test_exact_crop(query, source, expected):
    assert g.exact_crop(query, source) == expected


@pytest.mark.parametrize('query,source', [('AB', 'ABAB'), ('AB', 'AC'), ('', 'AB')])
def test_crop_rejects_ambiguity_and_mismatch(query, source):
    with pytest.raises(ValueError):
        g.exact_crop(query, source)


def points():
    t = np.array([[0.2, 0, 0], [3.3, 0, 0], [0, 4.4, 0], [0, 0, 2.2]])
    b = t + [4.1, 2.3, 18.7]
    return t, b


@pytest.mark.parametrize('count', [0, 4, 8, 16])
def test_cross_only_symmetric_nested_masks(count):
    t, b = points()
    bins, mask, records = g.cross_distogram(12, [1, 3, 5, 7], [0, 2, 4, 6], t, b, count)
    assert mask.sum() == count * 2
    assert np.array_equal(mask, mask.T)
    assert np.array_equal(bins, bins.T)
    assert not mask[np.ix_([1, 3, 5, 7], [1, 3, 5, 7])].any()
    assert not mask[np.ix_([0, 2, 4, 6], [0, 2, 4, 6])].any()
    _, full, all_records = g.cross_distogram(12, [1, 3, 5, 7], [0, 2, 4, 6], t, b, 16)
    assert not (mask & ~full).any()
    assert records == all_records[:count]


def test_zero_is_noop_and_preserves_existing_target_template():
    t, b = points()
    empty = {'restype': np.zeros(8)}
    zero = g.cross_distogram(8, list(range(4)), list(range(4, 8)), t, b, 0)
    g.merge_distogram(empty, *zero[:2])
    assert set(empty) == {'restype'}
    feats = {'distogram_emb': np.zeros((8, 8), dtype=int), 'distogram_mask': np.zeros((8, 8), dtype=bool)}
    feats['distogram_emb'][0, 1] = 30
    feats['distogram_mask'][0, 1] = True
    positive = g.cross_distogram(8, list(range(4)), list(range(4, 8)), t, b, 8)
    g.merge_distogram(feats, *positive[:2])
    assert feats['distogram_emb'][0, 1] == 30
    with pytest.raises(ValueError, match='overlap'):
        g.merge_distogram(feats, *positive[:2])


@pytest.mark.parametrize('count', [-1, 17, 3.5, True])
def test_bad_pair_count(count):
    with pytest.raises(ValueError):
        g.cross_distogram(8, list(range(4)), list(range(4, 8)), *points(), count)


def test_saturation_and_invalid_tokens_fail():
    t, b = points()
    with pytest.raises(ValueError, match='informative'):
        g.cross_distogram(8, list(range(4)), list(range(4, 8)), t, b + 100, 16)
    with pytest.raises(ValueError, match='distinct'):
        g.cross_distogram(8, list(range(4)), list(range(4)), t, b, 4)


def test_above_50_uses_trained_overflow_bin_and_retains_informative_pairs():
    """52.481 A is a valid bin-50 input, not a reason to abort preflight."""
    t, b = points()
    b = b.copy()
    b[0] = [t[0, 0], t[0, 1], 52.481]
    bins, mask, records = g.cross_distogram(
        8, list(range(4)), list(range(4, 8)), t, b, 16
    )
    assert mask.sum() == 32
    assert len(records) == 16
    assert any(p['distance_angstrom'] == pytest.approx(52.481) for p in records)
    assert sum(p['distance_overflow'] for p in records) == 4
    assert sum(p['finite_distance_information'] for p in records) == 12
    assert all(p['bin'] == 50 for p in records if p['distance_overflow'])
    assert all(0 <= p['bin'] <= 49 for p in records if p['finite_distance_information'])
    assert np.array_equal(bins, bins.T)
    assert np.array_equal(mask, mask.T)
    for p in records:
        actual_bin = int(np.sum(p['distance_angstrom'] > np.linspace(1, 50, 50)))
        assert actual_bin == p['bin']


def test_many_overflow_pairs_fail_with_explicit_geometry_diagnostic():
    t, b = points()
    b = b.copy()
    b[:3] += [0, 0, 55]
    with pytest.raises(ValueError, match='insufficient informative cross-chain anchors'):
        g.cross_distogram(8, list(range(4)), list(range(4, 8)), t, b, 16)


def test_colliding_cross_chain_ca_remains_invalid():
    t, b = points()
    b = b.copy()
    b[0] = t[0]
    with pytest.raises(ValueError, match='physically implausible clash'):
        g.cross_distogram(8, list(range(4)), list(range(4, 8)), t, b, 16)


def test_jitter_is_rigid_and_reproducible():
    _, b = points()
    first = g.jitter(b, 7, 15, 2)
    assert np.array_equal(first, g.jitter(b, 7, 15, 2))
    assert not np.array_equal(first, g.jitter(b, 8, 15, 2))
    assert np.linalg.norm(first.mean(0) - b.mean(0)) <= 2
    np.testing.assert_allclose(np.linalg.norm(first[:, None] - first[None, :], axis=-1),
                               np.linalg.norm(b[:, None] - b[None, :], axis=-1), atol=1e-10)
    np.testing.assert_array_equal(g.jitter(b, 7, 0, 0), b)


def test_global_transform_invariance_and_twist_separation():
    t, b = points()
    theta = np.deg2rad(30)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    metrics = g.pose_metrics(t, b, t @ rotation + 40, b @ rotation + 40, [0, 0, 1])
    assert metrics['source_framework_rmsd'] < 1e-10
    assert metrics['body_rotation_degrees'] < 1e-5
    twist = (b - b.mean(0)) @ rotation + b.mean(0)
    metrics = g.pose_metrics(t, b, t, twist, [0, 0, 1])
    assert metrics['body_rotation_degrees'] == pytest.approx(30)
    assert metrics['approach_degrees'] < 1e-5
    assert metrics['framework_internal_rmsd'] < 1e-10


def test_degenerate_geometry_fails():
    with pytest.raises(ValueError, match='collinear'):
        g.farthest_points([[i, 0, 0] for i in range(5)])
    with pytest.raises(ValueError):
        g.xyz([[np.nan, 0, 0]])


def test_chain_mapping_and_no_mutation():
    chains, schema, plan = fixtures()
    assert plan['target_chain_map'] == {'K': 'L'}
    assert plan['binder_chain'] == 'V'
    assert len(plan['binder_landmarks']) == 4
    assert schema == {'K': {'type': 'protein', 'sequence': 'ACDEF'}}
    with pytest.raises(ValueError):
        g.build_plan(chains, schema, PATTERN, 'K', 'N', {'K': 'L'})
    with pytest.raises(ValueError):
        g.build_plan(chains, schema, PATTERN, 'V', 'N', {'K': 'N'})


def test_multichain_mapping_and_runtime_order():
    chains, schema, _ = fixtures()
    chains['M'] = {'sequence': 'HIKL', 'coords': chains['L']['coords'][:4] + [2, 2, 0]}
    schema['Z'] = {'type': 'protein', 'sequence': 'HIKL'}
    plan = g.build_plan(chains, schema, PATTERN, 'V', 'N', {'K': 'L', 'Z': 'M'})
    observed = {'V': chains['N'], 'Z': chains['M'], 'K': chains['L']}
    target, fw, _ = g.observed_arrays(observed, plan)
    assert target.shape == (9, 3)
    np.testing.assert_allclose(fw, plan['source_framework_coords'])


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    chains, schema, _ = fixtures()
    (tmp_path / 'target.json').write_text(json.dumps(schema))
    (tmp_path / 'guide.cif').write_text('synthetic fixture, not a structure')
    (tmp_path / 'msa').mkdir()
    cfg = {'input': 'target.json', 'msa_dir': 'msa', 'binder': {'type': 'vhh', 'chain': 'V',
           'framework': PATTERN, 'framework_template': {'path': 'unused.cif'}}, 'target_chains': ['K']}
    (tmp_path / 'base.yaml').write_text(json.dumps(cfg))
    monkeypatch.setattr(experiment, 'read_chains', lambda _: chains)
    args = argparse.Namespace(base_config=str(tmp_path / 'base.yaml'), pose=str(tmp_path / 'guide.cif'),
        guide_binder_chain='N', target_map='{"K":"L"}', msa_dir=None, out=str(tmp_path / 'exp'),
        backbones=None, replicates=None, batch_size=1, profile='smoke', seed=100,
        inverse_folder='boltzif', sequences_per_backbone=1)
    experiment.prepare(args)
    return tmp_path / 'exp'


def test_preparation_paired_control_and_hashes(prepared):
    manifest = experiment.load_json(prepared / 'manifest.json')
    assert len(manifest['jobs']) == 2
    assert sum(j['requested_backbones'] for j in manifest['jobs']) == 4
    _, a, _ = experiment.job_config(prepared, 0)
    _, b, _ = experiment.job_config(prepared, 1)
    assert a.pose_guidance.seed == b.pose_guidance.seed
    assert a.pose_guidance.pairs == 0 and b.pose_guidance.pairs == 16
    assert a.binder.framework_template is None
    assert a.inverse_folder.type == 'boltzif'
    assert not a.skip_existing


def test_changed_plan_rejected(prepared):
    with (prepared / 'plan.json').open('a') as handle:
        handle.write(' ')
    with pytest.raises(ValueError, match='changed'):
        experiment.job_config(prepared, 0)


@pytest.fixture
def task_module(monkeypatch):
    # Load the repository's real collator without loading tinyprot's asset database.
    ccd = types.ModuleType('tinyprot.ccd')
    ccd._get_attrs = lambda *args: None
    monkeypatch.setitem(sys.modules, 'tinyprot.ccd', ccd)
    path = Path(__file__).resolve().parents[2] / 'promera/data/utils.py'
    spec = importlib.util.spec_from_file_location('pose_test_collate', path)
    collator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collator)
    design = types.ModuleType('promera.inference.design')

    class FakeDesign:
        def __init__(self, cfg):
            self.cfg = cfg
        def __getitem__(self, idx):
            feat = copy.deepcopy(self.features[idx])
            ids = list(self.cfg.get('epitope_residues', []) or [])
            if ids:
                chain = self.cfg.epitope_chain
                assert not self.cfg.get('target_template'), (
                    'the synthetic feature fixture only supports native schema numbering'
                )
                offsets, offset = {}, 0
                for name, record in feat['struct'].chains.items():
                    offsets[name], offset = offset, offset + len(record.rname)
                for residue in ids:
                    feat['is_epitope'][offsets[chain] + int(residue) - 1] = True
                feat['struct'].epitope_residues = [int(x) for x in ids]
            return feat
        def _refold_batch(self, model, feats):
            return model.pairformer_forward(feats)

    design.Design = FakeDesign
    design._sample_dir = lambda output, name, idx: str(Path(output) / name / f'sample{idx}')
    for name in ('promera', 'promera.data', 'promera.inference'):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules['promera.data'].utils = collator
    monkeypatch.setitem(sys.modules, 'promera.data.utils', collator)
    monkeypatch.setitem(sys.modules, 'promera.inference.design', design)
    name = 'pose_task_contract_test'
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('pose_task.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, collator


def synthetic_features(sequence, index):
    inv = {one: three for three, one in g.AA3.items()}
    binder = types.SimpleNamespace(rname=[inv[a] for a in sequence])
    antigen = types.SimpleNamespace(rname=[inv[a] for a in 'ACDEF'])
    # Deliberately put binder first, with non-A/B runtime chain identifiers.
    struct = types.SimpleNamespace(chains={'V': binder, 'K': antigen})
    return {'struct': struct, 'restype': np.zeros(len(sequence) + 5, dtype=int),
            'backbone_idx': index, 'is_epitope': np.zeros(len(sequence) + 5, dtype=bool)}


def test_task_variable_cdrs_and_real_matrix_collation(task_module, prepared):
    module, collator = task_module
    _, cfg, _ = experiment.job_config(prepared, 1)
    task = module.PoseGuidedDesign(cfg)
    task.features = [synthetic_features(SEQ, 0), synthetic_features(SEQ.replace('GG', 'GGGGG'), 1)]
    features = [task[0], task[1]]
    for feat in features:
        assert feat['distogram_mask'].sum() == 32
        pose_stats = feat['pose_record']
        assert pose_stats['requested_guidance_pairs'] == 16
        assert pose_stats['informative_guidance_pairs'] + pose_stats['overflow_guidance_pairs'] == 16
        assert pose_stats['distance_encoding'] == 'promera_51_bins_1_to_50_with_overflow'
        for record in pose_stats['pairs']:
            assert record['target_token'] >= len(feat['struct'].chains['V'].rname)
            assert record['binder_token'] < len(feat['struct'].chains['V'].rname)
    batch = collator.collate(features, token_multiple=8)
    n = batch['restype'].shape[1]
    assert batch['distogram_mask'].shape == (2, n, n)
    assert not batch['distogram_mask'][0, len(SEQ) + 5:].any()


def test_task_control_does_not_add_conditioning(task_module, prepared):
    module, _ = task_module
    _, cfg, _ = experiment.job_config(prepared, 0)
    task = module.PoseGuidedDesign(cfg)
    task.features = [synthetic_features(SEQ, 0)]
    result = task[0]
    assert 'distogram_mask' not in result
    assert result['pose_record']['pairs'] == []


@pytest.mark.parametrize('key', ['distogram_mask', 'is_epitope'])
def test_refold_guard_rejects_leaks_and_restores_model(task_module, key):
    module, _ = task_module
    task = object.__new__(module.PoseGuidedDesign)
    original = lambda feats: 'refold'
    model = types.SimpleNamespace(pairformer_forward=original)
    assert task._refold_batch(model, {key: np.zeros((2, 2))}) == 'refold'
    with pytest.raises(RuntimeError, match='contamination'):
        task._refold_batch(model, {key: np.ones((2, 2))})
    assert model.pairformer_forward is original


def test_no_outputs_are_not_biological_failures(prepared, monkeypatch):
    experiment.analyse(prepared)
    with (prepared / 'analysis/conditions.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(row['status'] == 'not_run' and row['missing_backbones'] == '2' for row in rows)


def test_analysis_exports_matching_refold_not_highest_wrong_pose(prepared, monkeypatch):
    _, cfg, _ = experiment.job_config(prepared, 0)
    folder = Path(cfg.output) / 'target/sample0'
    (folder / 'refolds').mkdir(parents=True)
    (folder / 'backbone.cif').write_text('synthetic')
    chains, _, _ = fixtures()
    observed = {'K': chains['L'], 'V': chains['N']}
    wrong = copy.deepcopy(observed)
    wrong['V']['coords'] = wrong['V']['coords'] + [20, 0, 0]
    for i, score in enumerate([0.4, 0.95]):
        stem = f'sample0_design0_refold{i}'
        (folder / f'refolds/{stem}.cif').write_text('synthetic')
        experiment.write_json(folder / f'refolds/{stem}_confidence.json', {'complex_iptm': score})
    monkeypatch.setattr(experiment, 'read_chains', lambda path: wrong if 'refold1' in str(path) else observed)
    experiment.analyse(prepared)
    with (prepared / 'analysis/protenix_inputs.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert 'refold0.cif' in rows[0]['promera_reference_cif']
    assert float(rows[0]['iptm']) == 0.4


def test_sampling_seed_restores_cpu_streams(task_module):
    import random
    module, _ = task_module
    before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with module.sampling_seed(123, torch.device('cpu')):
        one = (random.random(), np.random.rand(), torch.rand(3))
    with module.sampling_seed(123, torch.device('cpu')):
        two = (random.random(), np.random.rand(), torch.rand(3))
    assert one[:2] == two[:2]
    assert torch.equal(one[2], two[2])
    assert random.getstate() == before_python
    np.testing.assert_array_equal(np.random.get_state()[1], before_numpy[1])
    assert torch.equal(torch.get_rng_state(), before_torch)


def test_framework_template_cannot_be_enabled(task_module, prepared):
    module, _ = task_module
    _, cfg, _ = experiment.job_config(prepared, 1)
    cfg.binder.framework_template = {'path': 'anything.cif'}
    with pytest.raises(ValueError, match='without a framework template'):
        module.PoseGuidedDesign(cfg)


def test_changed_script_rejected(prepared, monkeypatch, tmp_path):
    replacement = tmp_path / 'scripts'
    replacement.mkdir()
    for script in experiment.HERE.iterdir():
        if not script.is_file():
            continue
        (replacement / script.name).write_text(script.read_text())
    with (replacement / 'pose_task.py').open('a') as handle:
        handle.write('\n# changed\n')
    monkeypatch.setattr(experiment, 'HERE', replacement)
    with pytest.raises(ValueError, match='code changed'):
        experiment.job_config(prepared, 0)


def test_changed_config_rejected(prepared):
    path = sorted((prepared / 'configs').glob('*.json'))[0]
    with path.open('a') as handle:
        handle.write(' ')
    with pytest.raises(ValueError, match='changed'):
        experiment.job_config(prepared, 0)


@pytest.mark.parametrize('command,needs_gpu', [('preflight', False), ('run', True)])
def test_container_argv_and_temporary_cleanup(tmp_path, command, needs_gpu):
    import os
    import subprocess
    binary = tmp_path / 'bin'
    binary.mkdir()
    capture = tmp_path / 'arguments'
    fake = binary / 'apptainer'
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    fake.chmod(0o755)
    work = tmp_path / 'space in workspace'
    work.mkdir()
    assets = tmp_path / 'assets'
    (assets / 'checkpoints').mkdir(parents=True)
    (assets / 'checkpoints/promera_2606.ckpt').write_text('fixture')
    image = tmp_path / 'image.sif'
    image.write_text('fixture, never executed')
    env = dict(os.environ, PATH=f'{binary}:' + os.environ['PATH'], CAPTURE=str(capture),
               POSE_WORK=str(work), PROMERA_IMAGE=str(image), PROMERA_ASSETS=str(assets))
    subprocess.run(['bash', str(Path(__file__).with_name('container.sh')), command,
                    '--experiment', str(work / 'exp')], env=env, check=True)
    args = capture.read_text().splitlines()
    assert ('--nv' in args) is needs_gpu
    assert args[args.index('--pwd') + 1] == str(work)
    assert '/opt/pose_guidance/experiment.py' in args
    exported = {value.partition('=')[0]: value.partition('=')[2]
                for index, value in enumerate(args) if index and args[index - 1] == '--env'}
    expected = {
        'XDG_CACHE_HOME': work / '.cache',
        'XDG_CONFIG_HOME': work / '.config',
        'XDG_DATA_HOME': work / '.local/share',
        'TORCH_HOME': work / '.cache/torch',
        'TORCHINDUCTOR_CACHE_DIR': work / '.cache/torchinductor',
        'TRITON_CACHE_DIR': work / '.cache/triton',
        'CUDA_CACHE_PATH': work / '.cache/nvidia',
        'HF_HOME': work / '.cache/huggingface',
        'MPLCONFIGDIR': work / '.cache/matplotlib',
    }
    for name, value in expected.items():
        assert exported.get(name) == str(value), name
        assert value.is_dir(), name
    assert Path(exported['XDG_RUNTIME_DIR']).parent.parent == work / '.tmp'
    assert exported['TMPDIR'].startswith(str(work / '.tmp'))
    assert exported.get('POSE_VERIFY_WRITABLE_RUNTIME') == '1'
    assert 'HOME' not in exported, 'Apptainer rejects --env HOME and emits a warning'
    assert '--home' in args
    assert args[args.index('--home') + 1] == (
        f"{work / '.home'}:{work / '.home'}"
    )
    assert '--no-home' not in args
    assert '--no-mount' in args
    assert args[args.index('--no-mount') + 1] == 'cwd'
    assert '/usr/bin/env' in args
    env_index = args.index('/usr/bin/env')
    assert args[env_index + 1] == f"HOME={work / '.home'}"
    assert args[env_index + 2] == 'python'
    assert args[env_index + 3] == '/opt/pose_guidance/experiment.py'
    assert (work / '.home').is_dir()
    assert not list((work / '.tmp').iterdir())


def test_launcher_discovers_managed_image_and_assets(tmp_path):
    import os
    import subprocess

    home = tmp_path / "home with spaces"
    config_dir = home / "dev"
    config_dir.mkdir(parents=True)
    parent = tmp_path / "scratch with spaces"
    image_release = parent / ".nominee/images/promera/releases/release42.sif"
    image_release.parent.mkdir(parents=True)
    image_release.write_text("synthetic image")
    (image_release.parent.parent / "current.sif").symlink_to("releases/release42.sif")
    asset_release = parent / ".nominee/assets/promera/releases/release42"
    (asset_release / "checkpoints").mkdir(parents=True)
    (asset_release / "checkpoints/promera_2606.ckpt").write_text("synthetic checkpoint")
    (asset_release.parent.parent / "current").symlink_to("releases/release42")
    (parent / ".nominee/cache/promera/msa").mkdir(parents=True)
    config = config_dir / "config.yaml"
    config.write_text('paths:\n  installation_parent: "$POSE_TEST_PARENT"  # site root\n')
    work = tmp_path / "experiment work"
    work.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_apptainer = bin_dir / "apptainer"
    fake_apptainer.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$POSE_ARG_LOG"\n')
    fake_apptainer.chmod(0o755)
    capture = tmp_path / "arguments"
    env = dict(os.environ, HOME=str(home), POSE_TEST_PARENT=str(parent),
               POSE_WORK=str(work), POSE_ARG_LOG=str(capture),
               PATH=f"{bin_dir}:{os.environ['PATH']}")
    for key in ("PROMERA_IMAGE", "PROMERA_ASSETS", "NOMINEE_HOST_CONFIG",
                "NOMINEE_CONFIG", "POSE_NOMINEE_CONFIG"):
        env.pop(key, None)
    wrapper = Path(__file__).with_name("container.sh")
    command = ["bash", str(wrapper), "preflight", "--experiment", str(work / "exp")]
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    argv = capture.read_text().splitlines()
    assert str(image_release) in argv
    assert any(str(asset_release) + ":/opt/nominee/assets:ro" in arg for arg in argv)
    assert any(str(parent) + ":" + str(parent) + ":ro" in arg for arg in argv)
    assert any(str(parent / ".nominee/cache") + ":/opt/nominee/cache:ro" in arg for arg in argv)
    assert "Promera image:" in result.stderr
    assert not list((work / ".tmp").iterdir())

    # Both explicit overrides work without config, including GPU launch.
    config.unlink()
    env.update(PROMERA_IMAGE=str(image_release), PROMERA_ASSETS=str(asset_release))
    result = subprocess.run(["bash", str(wrapper), "run", "--experiment", "test"],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--nv" in capture.read_text().splitlines()

    # In automatic mode, never silently fall back to old releases.
    env.pop("PROMERA_IMAGE")
    env.pop("PROMERA_ASSETS")
    config.write_text('paths:\n  installation_parent: "$POSE_TEST_PARENT"\n')
    (image_release.parent.parent / "current.sif").unlink()
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "active Promera image not found" in result.stderr


def test_nominee_config_validation_without_host_yaml(tmp_path):
    import os
    import subprocess
    import sys

    parser = Path(__file__).with_name("nominee_paths.py")
    config = tmp_path / "config.yaml"
    config.write_text("paths:\n  installation_parent: ''\n")
    result = subprocess.run([sys.executable, str(parser), str(config)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "blank or unsupported" in result.stderr

    config.write_text('paths:\n  installation_parent: "$UNDEFINED_POSE_TEST_ROOT"\n')
    env = dict(os.environ)
    env.pop("UNDEFINED_POSE_TEST_ROOT", None)
    result = subprocess.run([sys.executable, str(parser), str(config)],
                            capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "undefined environment variable" in result.stderr


@pytest.mark.parametrize("text, expected", [
    ("0:0", (0.0, 0.0)), ("0:25", (0.0, 25.0)),
    ("15:35", (15.0, 35.0)), ("180:180", (180.0, 180.0)),
])
def test_parse_direct_angle_interval(text, expected):
    assert experiment.parse_angle_range(text) == expected


@pytest.mark.parametrize("text", ["15", "35:15", "-1:20", "15:181", "NaN:10", "10:inf", "foo:bar"])
def test_invalid_direct_angle_interval_rejected(text):
    with pytest.raises(ValueError):
        experiment.parse_angle_range(text)


@pytest.mark.parametrize("lo,hi", [(0, 0), (0, 25), (15, 35), (40, 70), (150, 180)])
def test_direct_angle_cone_is_rigid_and_measurably_in_range(lo, hi):
    _, _, plan = fixtures()
    reference_target = np.asarray(plan['target_coords'])
    reference_fw = np.asarray(plan['source_framework_coords'])
    patch = reference_target[plan['target_landmarks']].mean(0)
    for seed in range(32):
        changed, draw = g.cone_placement(
            reference_fw, plan['source_axis'], seed, lo, hi,
            translation=0.0, pivot=patch,
        )
        value = draw['guide_approach_deviation_degrees']
        assert lo - 1e-5 <= value <= hi + 1e-5
        np.testing.assert_allclose(
            np.linalg.norm(changed[:, None] - changed[None, :], axis=-1),
            np.linalg.norm(reference_fw[:, None] - reference_fw[None, :], axis=-1),
            atol=1e-10,
        )
        measured = g.pose_metrics(
            reference_target, reference_fw, reference_target, changed,
            plan['source_axis'], target_patch_center=patch,
        )
        assert measured['approach_degrees'] == pytest.approx(value, abs=1e-5)
        assert measured['patch_radius_change_angstrom'] == pytest.approx(0, abs=1e-9)
        assert measured['framework_internal_rmsd'] == pytest.approx(0, abs=1e-9)
        if seed == 0:
            np.testing.assert_array_equal(
                changed, g.cone_placement(reference_fw, plan['source_axis'], seed, lo, hi, 0, pivot=patch)[0]
            )


def angular_fixture(prepared, *, interval="15:35", translation=0.0):
    args = argparse.Namespace(
        base_config=str(prepared.parent / 'base.yaml'),
        pose=str(prepared.parent / 'guide.cif'),
        guide_binder_chain='N', target_map='{"K":"L"}',
        msa_dir=None, out=str(prepared.parent / 'angle-experiment'),
        backbones=None, replicates=None, batch_size=1,
        profile='smoke', seed=100, inverse_folder='boltzif',
        sequences_per_backbone=1, angle_range=interval,
        angle_translation_angstrom=translation,
    )
    experiment.prepare(args)
    return Path(args.out)


def comparison_fixture(prepared, *, interval="0:20", backbones=3, replicates=2):
    args = argparse.Namespace(
        base_config=str(prepared.parent / 'base.yaml'),
        pose=str(prepared.parent / 'guide.cif'),
        guide_binder_chain='N', target_map='{"K":"L"}',
        msa_dir=None, out=str(prepared.parent / 'comparison-experiment'),
        backbones=backbones, replicates=replicates, batch_size=1,
        profile='comparison', seed=100, inverse_folder='boltzif',
        sequences_per_backbone=1, angle_range=interval,
        angle_translation_angstrom=0.0,
        # This synthetic binder is ~20 A from the tiny synthetic target;
        # the real B004 source yields 33 CDR-proximal antigen residues at 10 A.
        site_cutoff_angstrom=25.0,
        site_residues=7,
    )
    experiment.prepare(args)
    return Path(args.out)


def test_target_site_selects_antigen_schema_ordinals_not_template_ids():
    chains, _, plan = fixtures()
    site = g.targeted_site_from_reference(chains, plan, 25.0)
    assert site['chain'] == 'K'
    assert site['schema_positions_1_based'] == [1, 2, 3, 4, 5]
    assert site['residue_numbering'].startswith('1-based in exact target schema')
    assert all(0 < value <= 25 for value in site['nearest_cdr_ca_distance_angstrom'].values())
    template = {
        'sequence': 'QQACDEFRR',
        'residue_ids': [str(i) for i in range(101, 110)],
    }
    assert g.native_epitope_template_ids(site, 'ACDEF', template) == [
        '103', '104', '105', '106', '107'
    ]
    with pytest.raises(ValueError, match='cutoff'):
        g.targeted_site_from_reference(chains, plan, 30.0)
    with pytest.raises(ValueError, match='CDR-proximal site'):
        g.targeted_site_from_reference(chains, plan, 5.0)


def test_site_cap_selects_seven_closest_from_broad_cdr_neighbourhood():
    chains, _, plan = fixtures()
    plan['target_sequences']['K'] = 'A' * 20
    plan['target_labels'] = [['K', i] for i in range(20)]
    plan['target_coords'] = [[i * 0.2, 0, 0] for i in range(20)]
    site = g.targeted_site_from_reference(chains, plan, cutoff_angstrom=25.0)
    assert site['source_candidate_pool_count'] == 20
    assert site['requested_site_residue_limit'] == 7
    assert site['source_target_residue_count'] == 7
    assert len(site['schema_positions_1_based']) == 7
    assert len(site['nearest_cdr_ca_distance_angstrom']) == 7
    assert site['selection_policy'] == (
        'nearest_target_CA_to_any_source_CDR_CA_within_candidate_cutoff'
    )
    # The selector must respond to user-specified compact site size.
    smaller = g.targeted_site_from_reference(
        chains, plan, cutoff_angstrom=25.0, max_residues=4
    )
    assert smaller['source_target_residue_count'] == 4
    assert set(smaller['schema_positions_1_based']).issubset(
        site['schema_positions_1_based']
    )
    with pytest.raises(ValueError, match='site-residues'):
        g.targeted_site_from_reference(chains, plan, max_residues=33)
    with pytest.raises(ValueError, match='site-residues'):
        g.targeted_site_from_reference(chains, plan, max_residues=True)


def test_comparison_creates_paired_site_only_and_guided_arms(prepared):
    root = comparison_fixture(prepared)
    manifest = experiment.load_json(root / 'manifest.json')
    assert manifest['profile'] == 'comparison'
    assert len(manifest['jobs']) == 8
    assert manifest['angle_range_requested_degrees'] == [0, 20]
    assert manifest['comparison_primary_contrast'] == 'targeted_only_vs_cone16_same_epitope_and_seeds'
    assert manifest['targeted_site']['schema_positions_1_based'] == [1, 2, 3, 4, 5]
    assert manifest['targeted_site_residue_count'] == 5
    assert manifest['targeted_site']['requested_site_residue_limit'] == 7
    assert sum(job['requested_backbones'] for job in manifest['jobs']) == 24
    names = [job['arm'] for job in manifest['jobs']]
    assert names == ['control']*2 + ['targeted']*2 + ['anchors16']*2 + ['cone16']*2
    assert [job['targeted_site_enabled'] for job in manifest['jobs']] == [False]*2 + [True]*6
    by_arm = {}
    for job in manifest['jobs']:
        _, cfg, _ = experiment.job_config(root, job['index'])
        by_arm.setdefault(job['arm'], {})[job['replicate']] = cfg
        if job['arm'] == 'control':
            assert list(cfg.epitope_residues) == []
        else:
            assert list(cfg.epitope_residues) == [1, 2, 3, 4, 5]
        assert cfg.pose_guidance.seed == 100 + job['replicate']
        assert cfg.binder.framework_template is None
    for rep in (0, 1):
        assert all(by_arm[a][rep].pose_guidance.seed == by_arm['targeted'][rep].pose_guidance.seed
                   for a in ('control', 'anchors16', 'cone16'))
        assert by_arm['targeted'][rep].pose_guidance.pairs == 0
        assert by_arm['control'][rep].pose_guidance.pairs == 0
        assert by_arm['anchors16'][rep].pose_guidance.pairs == 16
        assert by_arm['cone16'][rep].pose_guidance.pairs == 16
        assert list(by_arm['cone16'][rep].pose_guidance.approach_angle_range_degrees) == [0, 20]


def test_comparison_all_site_arms_have_identical_epitope_features(task_module, prepared):
    module, _ = task_module
    root = comparison_fixture(prepared)
    outputs = {}
    for index in (0, 2, 4, 6):
        _, cfg, _ = experiment.job_config(root, index)
        task = module.PoseGuidedDesign(cfg)
        task.features = [synthetic_features(SEQ, 0)]
        feat = task[0]
        site = feat['is_epitope'][len(SEQ):]
        outputs[index] = site.copy()
        if index == 0:
            assert not site.any()
            assert 'distogram_mask' not in feat
            assert feat['pose_record']['targeted_site_tokens'] == 0
        else:
            assert site.sum() == 5
            assert all(site)
            assert feat['pose_record']['targeted_site_enabled'] is True
            assert feat['pose_record']['targeted_site_tokens'] == 5
            assert feat['struct'].epitope_residues == [1, 2, 3, 4, 5]
            if index == 2:
                assert 'distogram_mask' not in feat
            else:
                assert feat['distogram_mask'].sum() == 32
    np.testing.assert_array_equal(outputs[2], outputs[4])
    np.testing.assert_array_equal(outputs[2], outputs[6])


def test_comparison_analysis_primary_endpoint_and_wilson(prepared):
    root = comparison_fixture(prepared)
    manifest = experiment.load_json(root / 'manifest.json')
    totals = []
    rows = []
    for arm in ('control', 'targeted', 'anchors16', 'cone16'):
        for rep in (0, 1):
            totals.append({
                'arm': arm, 'replicate': rep, 'status': 'completed',
                'requested_backbones': 2, 'targeted_site_enabled': arm != 'control',
                'wall_seconds': 2.0,
            })
        for angle, contacts in [(7, 4), (11, 1), (58, 8), (10, 5)]:
            hit = 0 <= angle <= 20 and contacts >= 3
            rows.append({
                'kind': 'backbone', 'arm': arm, 'approach_degrees': angle,
                'in_requested_angle': 0 <= angle <= 20, 'primary_hit': hit,
                'site_contacted_residues': contacts,
                'site_contact_fraction': contacts / 5,
            })
    summary, effect = experiment.comparison_summary(manifest, rows, totals)
    assert [row['arm'] for row in summary] == ['control', 'targeted', 'anchors16', 'cone16']
    for row in summary:
        assert row['status'] == 'completed'
        assert row['generated_backbones'] == 4
        assert row['primary_geometry_and_site_hits'] == 2
        assert row['primary_hit_fraction'] == 0.5
        assert 0 <= row['primary_hit_wilson95_low'] < 0.5
        assert 0.5 < row['primary_hit_wilson95_high'] <= 1
    assert effect['complete'] is True
    assert effect['absolute_primary_hit_fraction_difference'] == 0.0
    assert experiment.wilson95(0, 0) == (None, None)


def test_prepare_angle_range_separate_generation_arm(prepared):
    root = angular_fixture(prepared)
    manifest = experiment.load_json(root / 'manifest.json')
    assert [job['arm'] for job in manifest['jobs']] == ['control', 'anchors16', 'cone16']
    assert manifest['selection']['approach_degrees_min'] == 15
    assert manifest['selection']['approach_degrees_max'] == 35
    assert manifest['selection']['position_comparison'] == 'target_patch_radius'
    _, cfg, _ = experiment.job_config(root, 2)
    assert list(cfg.pose_guidance.approach_angle_range_degrees) == [15, 35]
    assert cfg.pose_guidance.pairs == 16
    assert cfg.binder.framework_template is None


def test_target_relative_angular_guide_is_passed_to_design(task_module, prepared):
    module, collator = task_module
    root = angular_fixture(prepared, translation=1.5)
    _, cfg, _ = experiment.job_config(root, 2)
    task = module.PoseGuidedDesign(cfg)
    task.features = [synthetic_features(SEQ, 0), synthetic_features(SEQ.replace('GG', 'GGGG'), 1)]
    features = [task[0], task[1]]
    for item in features:
        record = item['pose_record']
        assert record['guide_sampling_policy'] == 'uniform_solid_angle'
        assert record['guide_pivot'] == 'antigen_patch_centroid'
        assert 15 - 1e-5 <= record['guide_approach_deviation_degrees'] <= 35 + 1e-5
        assert record['guide_translation_angstrom'] <= 1.5
        assert record['approach_angle_range_degrees'] == [15, 35]
        assert item['distogram_mask'].sum() == 32
    padded = collator.collate(features, token_multiple=8)
    assert padded['distogram_mask'].shape[-1] == padded['restype'].shape[-1]


def test_angle_family_evaluation_allows_expected_centroid_movement():
    metrics = {'approach_degrees': 29, 'centroid_shift_angstrom': 12,
               'patch_radius_change_angstrom': 0.2, 'framework_internal_rmsd': 0.1,
               'target_fit_rmsd': 0.1}
    thresholds = {'approach_degrees_min': 15, 'approach_degrees_max': 35,
                  'position_comparison': 'target_patch_radius',
                  'patch_radius_change_angstrom_max': 6,
                  'centroid_shift_angstrom_max': 6,
                  'framework_internal_rmsd_max': 2, 'target_fit_rmsd_max': 3}
    assert experiment._passes_geometry(metrics, thresholds)
    assert not experiment._passes_geometry(dict(metrics, approach_degrees=5), thresholds)
    assert not experiment._passes_geometry(dict(metrics, approach_degrees=50), thresholds)
    assert not experiment._passes_geometry(
        dict(metrics, patch_radius_change_angstrom=7), thresholds
    )


def test_current_nominee_multichain_and_symmetry_options_survive_prepare(prepared):
    base = prepared.parent / 'base.yaml'
    cfg = json.loads(base.read_text())
    cfg['antigen_symmetry'] = {'group': 'C1', 'units': [], 'site_policy': 'equivalent_sites'}
    cfg['antigen_symmetry_validation'] = {'status': 'valid'}
    cfg['epitope_residue_map'] = {'K': [2, 3]}
    base.write_text(json.dumps(cfg))
    root = angular_fixture(prepared)
    _, generated, _ = experiment.job_config(root, 2)
    assert generated.antigen_symmetry.group == 'C1'
    assert generated.antigen_symmetry_validation.status == 'valid'
    assert list(generated.epitope_residue_map.K) == [2, 3]


def _synthetic_atlas_run(tmp_path):
    import hashlib
    from pathlib import Path
    import launch_from_atlas as atlas

    run = tmp_path / "sandbox" / "nominee-runs" / "poseguide"
    scan = run / "analysis" / "epitope_scan"
    scan.mkdir(parents=True)
    pose_id = "B004__G047__5_389"
    seed = scan / "pose_seeds" / "B004" / pose_id
    seed.mkdir(parents=True)
    backbone = seed / "backbone.cif"
    backbone.write_bytes(b"original canonical CDR-guided coordinates")
    original = run / "promera" / "batch_5" / "sample_5_389" / "backbone.cif"
    original.parent.mkdir(parents=True)
    original.write_bytes(backbone.read_bytes())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "pose_seed_id": pose_id,
        "canonical_backbone_path": str(backbone),
        "canonical_backbone_sha256": digest(backbone),
    }
    seed_manifest = seed / "pose_seed.json"
    seed_manifest.write_text(json.dumps(manifest))
    row = {
        "pose_seed_id": pose_id,
        "source_candidate_id": "5_389",
        "basin_id": "B004",
        "geometry_group_id": "G047",
        "classification": "cdr_driven",
        "pose_seed_manifest_path": str(seed_manifest),
        "pose_seed_manifest_sha256": digest(seed_manifest),
        "backbone_path": str(original),
        "backbone_sha256": digest(original),
        "binder_chain": "B",
        "antigen_chains": "A",
        "framework": PATTERN,
    }
    catalog_path = scan / "pose_catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "records": [row]}))
    (run / "promera" / "promera_batch_1.yaml").write_text(
        json.dumps({"binder": {"chain": "B", "type": "vhh",
                              "framework": "WRONG"}, "input": "wrong.json"}))
    (run / "promera" / "input").mkdir(parents=True)
    (run / "promera" / "input" / "target.json").write_text('{"A":{"type":"protein","sequence":"ACDEFG"}}')
    (run / "promera" / "msa").mkdir()
    config = run / "promera" / "promera_batch_5.yaml"
    config.write_text(json.dumps({
        "binder": {"chain": "B", "type": "vhh", "framework": PATTERN},
        "target_chains": ["A"],
        "input": str(run / "promera" / "input" / "target.json"),
        "msa_dir": str(run / "promera" / "msa"),
    }))
    return run, backbone, original, seed_manifest, catalog_path, config


def test_atlas_launch_selects_exact_first_cdr_seed_batch_not_first_task(tmp_path):
    import launch_from_atlas as atlas
    run, backbone, original, manifest, catalog, config = _synthetic_atlas_run(tmp_path)
    selected = atlas.select_source(run, "B004__G047__5_389")
    assert selected["canonical_backbone"] == backbone.resolve()
    assert selected["original_backbone"] == original.resolve()
    assert selected["base_config"] == config
    assert selected["batch_number"] == 5
    assert selected["source_record"]["classification"] == "cdr_driven"


@pytest.mark.parametrize("mutation", ["wrong_class", "changed_backbone", "changed_manifest",
                                       "wrong_framework", "wrong_chain", "missing_batch", "duplicate"])
def test_atlas_launch_refuses_unverified_or_mismatched_source(tmp_path, mutation):
    import launch_from_atlas as atlas
    run, backbone, original, manifest, catalog, config = _synthetic_atlas_run(tmp_path)
    rowset = json.loads(catalog.read_text())
    if mutation == "wrong_class":
        rowset["records"][0]["classification"] = "framework_driven"
    elif mutation == "changed_backbone":
        backbone.write_bytes(b"invalid geometry data")
    elif mutation == "changed_manifest":
        manifest.write_text("different manifest")
    elif mutation == "wrong_framework":
        cfg = json.loads(config.read_text())
        cfg["binder"]["framework"] = "DIFFERENT"
        config.write_text(json.dumps(cfg))
    elif mutation == "wrong_chain":
        cfg = json.loads(config.read_text())
        cfg["target_chains"] = ["Z"]
        config.write_text(json.dumps(cfg))
    elif mutation == "missing_batch":
        config.unlink()
    elif mutation == "duplicate":
        rowset["records"].append(dict(rowset["records"][0]))
    catalog.write_text(json.dumps(rowset))
    with pytest.raises(ValueError):
        atlas.select_source(run, "B004__G047__5_389")


def test_atlas_launcher_external_bind_is_read_only(tmp_path):
    import launch_from_atlas as atlas
    run, *_ = _synthetic_atlas_run(tmp_path)
    selected = atlas.select_source(run)
    parent = tmp_path / "different" / "installation"
    work = parent / "pose-guidance-work"
    external = atlas.extra_binds(selected, parent, work)
    assert f"{run}:{run}:ro" in external
    assert "rw" not in external
    assert len(external.split(",")) >= 1


def test_atlas_launcher_submits_using_repository_root_not_experiments(
    tmp_path, monkeypatch
):
    """Prevent actual SLURM exit 127 caused by duplicated /experiments path."""
    import launch_from_atlas as atlas

    run, *_ = _synthetic_atlas_run(tmp_path)
    site = tmp_path / "dev" / "config.yaml"
    site.parent.mkdir(parents=True)
    site.write_text(f"paths:\n  installation_parent: {tmp_path / 'installation'}\n")
    args = argparse.Namespace(
        run=str(run), pose_id="B004__G047__5_389", angle_range="15:35",
        nominee_config=str(site), work=str(tmp_path / "test-work"),
        dry_run=False, no_submit=False,
    )
    invocations = []
    def simulate_run(argv, **kwargs):
        invocations.append((list(argv), dict(kwargs)))
        env = kwargs["env"]
        source = Path(env["POSE_SOURCE"])
        assert source == Path(atlas.__file__).resolve().parents[2]
        assert (source / "promera").is_dir()
        assert (source / "experiments/pose_guidance/container.sh").is_file()
        assert "experiments/experiments/" not in str(source)
        if argv[0] == "bash" and "prepare" in argv:
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / 'manifest.json').write_text(json.dumps({
                'jobs': [{'index': i, 'arm': arm, 'replicate': 0,
                          'requested_backbones': 2}
                         for i, arm in enumerate(('control', 'anchors16', 'cone16'))],
                'inverse_folder': 'boltzif', 'sequences_per_backbone': 1,
            }))
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(atlas.subprocess, "run", simulate_run)
    monkeypatch.setattr(atlas, "shutil_which_sbatch", lambda: True)
    atlas.launch(args)
    assert len(invocations) == 5
    assert [cmd[0][2] for cmd in invocations[:4]] == [
        "prepare", "preflight", "preflight", "preflight"
    ]
    sbatch, kw = invocations[-1]
    assert sbatch[0] == "sbatch"
    assert "--array=0-2%1" in sbatch
    assert Path(sbatch[-1]) == Path(atlas.__file__).resolve().with_name("run.slurm")
    assert kw["env"]["POSE_SOURCE"] == str(Path(atlas.__file__).resolve().parents[2])
    assert (Path(kw["env"]["POSE_EXPERIMENT"]) / "source_atlas_selection.json").is_file()


def test_atlas_launcher_comparison_dynamic_array_and_budget(tmp_path, monkeypatch):
    import launch_from_atlas as atlas
    run, *_ = _synthetic_atlas_run(tmp_path)
    site = tmp_path / 'dev' / 'config.yaml'
    site.parent.mkdir(parents=True)
    site.write_text(f"paths:\n  installation_parent: {tmp_path / 'installation'}\n")
    args = argparse.Namespace(
        run=str(run), pose_id="B004__G047__5_389", angle_range="0:20",
        nominee_config=str(site), work=str(tmp_path / 'bench-work'),
        profile='comparison', backbones=3, replicates=2,
        site_cutoff_angstrom=10.0, site_residues=7,
        dry_run=False, no_submit=False,
    )
    invocations = []
    def fake_run(argv, **kwargs):
        invocations.append(list(argv))
        if 'prepare' in argv:
            assert argv[argv.index('--profile') + 1] == 'comparison'
            assert argv[argv.index('--backbones') + 1] == '3'
            assert argv[argv.index('--replicates') + 1] == '2'
            assert argv[argv.index('--site-cutoff-angstrom') + 1] == '10.0'
            assert argv[argv.index('--site-residues') + 1] == '7'
            root = Path(argv[argv.index('--out') + 1])
            root.mkdir(parents=True, exist_ok=True)
            (root / 'manifest.json').write_text(json.dumps({
                'jobs': [{'index': i, 'arm': arm, 'replicate': rep,
                          'requested_backbones': 3}
                         for i, (arm, rep) in enumerate(
                             (arm, rep) for arm in ('control', 'targeted', 'anchors16', 'cone16')
                             for rep in range(2))],
                'sequences_per_backbone': 1, 'inverse_folder': 'boltzif',
            }))
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(atlas.subprocess, 'run', fake_run)
    monkeypatch.setattr(atlas, 'shutil_which_sbatch', lambda: True)
    atlas.launch(args)
    assert len(invocations) == 10  # prepare, eight condition preflights, sbatch
    assert invocations[-1][0] == 'sbatch'
    assert '--array=0-7%1' in invocations[-1]
    assert sorted(cmd[-1] for cmd in invocations[1:-1]) == ['0','1','2','3','4','5','6','7']


def test_atlas_launcher_dry_run_selects_and_does_not_submit(tmp_path, capsys):
    import launch_from_atlas as atlas
    run, *_ = _synthetic_atlas_run(tmp_path)
    site = tmp_path / "dev" / "config.yaml"
    site.parent.mkdir(parents=True)
    site.write_text(f"paths:\n  installation_parent: {tmp_path / 'installation'}\n")
    args = argparse.Namespace(
        run=str(run), pose_id="B004__G047__5_389", angle_range="15:35",
        nominee_config=str(site), work=str(tmp_path / "test-work"), dry_run=True,
        no_submit=False,
    )
    atlas.launch(args)
    assert "DRY RUN" in capsys.readouterr().out
    assert not list((tmp_path / "test-work").iterdir())


def test_milton_gpuq_explicitly_requests_one_a30():
    """The one-command atlas launcher submits this exact SLURM script."""
    script = Path(__file__).with_name("run.slurm").read_text()
    assert "#SBATCH --partition=gpuq" in script
    assert "#SBATCH --gres=gpu:A30:1" in script
    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --gres=gpu:1\n" not in script


def test_gpu_runtime_writable_cache_probe_uses_scratch_not_login_home(tmp_path, monkeypatch):
    names = {
        'HOME': '.home',
        'XDG_CACHE_HOME': '.cache',
        'XDG_CONFIG_HOME': '.config',
        'XDG_DATA_HOME': '.local/share',
        'XDG_RUNTIME_DIR': '.tmp/xdg-runtime',
        'TMPDIR': '.tmp',
        'TORCH_HOME': '.cache/torch',
        'TORCHINDUCTOR_CACHE_DIR': '.cache/torchinductor',
        'TRITON_CACHE_DIR': '.cache/triton',
        'CUDA_CACHE_PATH': '.cache/nvidia',
        'HF_HOME': '.cache/huggingface',
        'MPLCONFIGDIR': '.cache/matplotlib',
    }
    for name, relative in names.items():
        monkeypatch.setenv(name, str(tmp_path / relative))
    checked = experiment.verify_writable_runtime()
    assert set(checked) == set(names)
    assert checked['HOME'] == str(tmp_path / '.home')
    assert all(Path(directory).is_dir() for directory in checked.values())
    assert not list(tmp_path.rglob('.pose-runtime-write-*'))


def test_gpu_runtime_probe_detects_missing_writable_cache(tmp_path, monkeypatch):
    for name in (
        'HOME', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME',
        'XDG_RUNTIME_DIR', 'TMPDIR', 'TORCH_HOME', 'TORCHINDUCTOR_CACHE_DIR',
        'TRITON_CACHE_DIR', 'HF_HOME', 'MPLCONFIGDIR',
    ):
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.delenv('CUDA_CACHE_PATH', raising=False)
    with pytest.raises(ValueError, match='CUDA_CACHE_PATH'):
        experiment.verify_writable_runtime()


@pytest.mark.parametrize("job, expected_angles", [
    (0, None),
    (1, None),
    (2, [15.0, 35.0]),
])
def test_real_preflight_report_json_encodes_omegaconf_angle_list(
    prepared, task_module, monkeypatch, job, expected_angles
):
    """Regression: successful featurization must produce valid preflight JSON.

    In particular the cone arm is loaded by OmegaConf as ListConfig, which is
    *not* directly JSON-serializable. This exercises preflight(), not just the
    conversion expression in isolation.
    """
    module, _ = task_module
    root = angular_fixture(prepared)
    # preflight imports the concrete 'pose_task' module name. Point it to
    # the mocked task boundary rather than accidentally importing a second
    # copy of PoseGuidedDesign with unmocked dataset methods.
    monkeypatch.setitem(sys.modules, "pose_task", module)
    module.PoseGuidedDesign.features = [
        synthetic_features(SEQ, 0),
        synthetic_features(SEQ.replace('GG', 'GGGG'), 1),
    ]
    monkeypatch.setattr(
        module.PoseGuidedDesign, "__len__", lambda self: 2, raising=False
    )
    design = sys.modules["promera.inference.design"]
    design.boltz_if = object()
    design.__file__ = str(Path(__file__).with_name("pose_task.py"))
    sys.modules["promera.inference"].design = design
    checkpoint = prepared.parent / "synthetic.ckpt"
    checkpoint.write_bytes(b"not a real checkpoint")
    monkeypatch.setenv("PROMERA_WEIGHTS", str(checkpoint))

    import_module = experiment.importlib.import_module

    def import_stub(name):
        if name in {"promera.model.model", "promera.diffusion.sampler"}:
            return design
        return import_module(name)

    monkeypatch.setattr(experiment.importlib, "import_module", import_stub)
    calls = []
    def fake_writable_probe():
        calls.append(True)
        return {'HOME': '/scratch/pose-test-home'}
    monkeypatch.setattr(experiment, 'verify_writable_runtime', fake_writable_probe)
    monkeypatch.setenv('POSE_VERIFY_WRITABLE_RUNTIME', '1')
    report = experiment.preflight(root, job)
    assert calls == [True], 'CPU preflight must validate container HOME before sbatch'
    assert report['runtime_writable_dirs'] == {'HOME': '/scratch/pose-test-home'}
    assert report["requested_approach_angle_range_degrees"] == expected_angles
    assert report["pair_count"] == (0 if job == 0 else 16)
    if job == 2:
        assert all(
            15 <= item["guide_angle_degrees"] <= 35
            for item in report["pose_anchor_diagnostics"]
        )
    encoded = json.loads(json.dumps(report, indent=2, allow_nan=False))
    assert encoded["requested_approach_angle_range_degrees"] == expected_angles
