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
            return copy.deepcopy(self.features[idx])
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
        for record in feat['pose_record']['pairs']:
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
