"""Opt-in task: cross-chain input conditioning, unchanged backbone sampler.

Import using --task pose_task.PoseGuidedDesign with this directory on PYTHONPATH.
Do not place the whole experimental checkout ahead of a patched runtime package.
"""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Iterator

import numpy as np
import torch
from promera.data import utils as data_utils
from promera.inference.design import Design, _sample_dir

from pose_geometry import (chain_sequence, cone_placement, cross_distogram, digest,
                           framework_map, jitter, merge_distogram, stratified_cone_angles)


@contextmanager
def sampling_seed(seed: int, device: torch.device) -> Iterator[None]:
    """Scope the experimental random stream; ordinary Design is never modified."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def allow_guided_backbones_past_native_gate() -> Iterator[None]:
    """Keep NOMINEE's viability diagnostics, but do not discard guided poses.

    The usual targeted-only control still uses the native gate unchanged.
    All overridden decisions are labelled 'bypassed' and retain the original
    rejection reason in backbone_confidence.json. This temporary replacement
    affects this isolated experiment process only and is restored on exit.
    """
    from promera.inference import design as native_design
    original = getattr(native_design, 'evaluate_targeted_backbone_viability', None)
    if not callable(original):
        raise RuntimeError('cannot bypass target gate: patched NOMINEE viability function is missing')

    def audit_without_reject(*args: Any, **kwargs: Any) -> Any:
        verdict = original(*args, **kwargs)
        if verdict is None or verdict.get('status') != 'reject':
            return verdict
        return {
            **verdict,
            'status': 'bypassed',
            'reason': 'experimental_pose_guided_backbone_gate_bypass',
            'native_status': 'reject',
            'native_reason': verdict.get('reason'),
            'native_failing_epitope_residues': list(
                verdict.get('failing_epitope_residues') or []
            ),
        }

    native_design.evaluate_targeted_backbone_viability = audit_without_reject
    try:
        yield
    finally:
        native_design.evaluate_targeted_backbone_viability = original


class PoseGuidedDesign(Design):
    def __init__(self, cfg: Any) -> None:
        if cfg.binder.type != 'vhh' or cfg.binder.get('framework_template'):
            raise ValueError('use VHH sequence conditioning without a framework template')
        if cfg.get('skip_existing', False):
            raise ValueError('experimental runs require a fresh output directory')
        self.guide = cfg.pose_guidance
        self.plan = json.loads(Path(self.guide.plan).read_text())
        if digest(self.guide.plan) != self.guide.plan_sha256:
            raise ValueError('pose plan changed after experiment preparation')
        if cfg.binder.chain != self.plan['binder_chain'] or cfg.binder.framework != self.plan['framework']:
            raise ValueError('configuration no longer matches the prepared guide')
        if not 0 <= int(self.guide.pairs) <= 16:
            raise ValueError('invalid pose pair count')
        interval = self.guide.get('approach_angle_range_degrees')
        if interval is not None:
            if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)) or len(interval) != 2:
                raise ValueError('approach angle range must have two degree endpoints')
            lo, hi = map(float, interval)
            if not np.isfinite([lo, hi]).all() or not 0 <= lo <= hi <= 180:
                raise ValueError('approach angle range must satisfy 0 <= min <= max <= 180')
            if int(self.guide.pairs) <= 0:
                raise ValueError('angular guidance requires nonzero cross-chain pairs')
        if bool(self.guide.get('bypass_targeted_backbone_gate', False)):
            if not int(self.guide.pairs) or not bool(self.guide.get('targeted_site_enabled', False)):
                raise ValueError('native target gate bypass is restricted to targeted pose-guided arms')
        if bool(self.guide.get('stratified_cone', False)) and interval is None:
            raise ValueError('stratified cone sampling requires an angular guidance interval')
        # The pinned collator otherwise pads these matrices along only one axis.
        # This process-local registration happens only when this task is selected.
        for key in ('distogram_emb', 'distogram_mask'):
            if key not in data_utils._token_pair_keys:
                data_utils._token_pair_keys.append(key)
        super().__init__(cfg)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        with sampling_seed((int(self.guide.seed) + idx) % 2**32, torch.device('cpu')):
            feats = super().__getitem__(idx)
        struct = feats['struct']
        offsets, total = {}, 0
        for name, chain in struct.chains.items():
            offsets[name], total = total, total + len(chain.rname)
        if total != len(feats['restype']):
            raise ValueError('expected one protein token per sequence residue')
        site = self.plan.get('targeted_site')
        site_enabled = bool(self.guide.get('targeted_site_enabled', False))
        if site_enabled:
            if site is None:
                raise ValueError('targeted comparison needs a recorded source site')
            chain = site['chain']
            positions = site['schema_positions_1_based']
            if chain not in offsets or chain != self.cfg.epitope_chain:
                raise ValueError('target site chain must match the selected antigen chain')
            if any(not 1 <= p <= len(struct.chains[chain].rname) for p in positions):
                raise ValueError('selected site residue outside generated antigen')
            if list(self.cfg.epitope_residues) != list(site['native_epitope_residue_ids']):
                raise ValueError('native targeted residue IDs changed after preparation')
            site_tokens = [offsets[chain] + p - 1 for p in positions]
            # Native Promera target featurization must already have applied
            # epitope tokens, pinned the target template, and converted author
            # IDs back to 1-based schema positions. Do NOT add a second mask:
            # that would differ from the ordinary targeted-run pathway.
            actual = {i for i in site_tokens if bool(feats['is_epitope'][i])}
            if actual != set(site_tokens):
                raise ValueError('native targeted epitope features do not match the source site')
            if list(getattr(struct, 'epitope_residues', [])) != list(positions):
                raise ValueError('native mapped target epitope residue ordinals disagree')
        elif site is not None:
            site_tokens = []
            if any(bool(feats['is_epitope'][offsets[chain] + p - 1])
                   for chain in self.plan['target_sequences']
                   for p in range(1, len(struct.chains[chain].rname) + 1)):
                raise ValueError('untargeted comparison leaked epitope flags')
        else:
            site_tokens = []
        for name, sequence in self.plan['target_sequences'].items():
            if chain_sequence(struct.chains[name]) != sequence:
                raise ValueError(f'generated target {name} differs from the prepared schema')
        fixed, _ = framework_map(chain_sequence(struct.chains[self.cfg.binder.chain]), self.plan['framework'])
        if len(fixed) != len(self.plan['source_framework_coords']):
            raise ValueError('generated framework mapping differs from the guide')
        b_idx = int(feats['backbone_idx'])
        seed = int.from_bytes(hashlib.sha256(f'{self.guide.seed}:{b_idx}'.encode()).digest()[:4], 'big')
        interval = self.guide.get('approach_angle_range_degrees')
        if interval is not None:
            kwargs = {}
            if bool(self.guide.get('stratified_cone', False)):
                angle, azimuth, angle_stratum, azimuth_stratum = stratified_cone_angles(
                    b_idx, int(self.cfg.num_backbones), int(self.guide.seed),
                    float(interval[0]), float(interval[1]),
                )
                kwargs = {
                    'sampled_angle_degrees': angle,
                    'sampled_azimuth_degrees': azimuth,
                }
            framework, guide_metadata = cone_placement(
                self.plan['source_framework_coords'], self.plan['source_axis'], seed,
                float(interval[0]), float(interval[1]),
                float(self.guide.translation_angstrom),
                pivot=np.asarray(self.plan['target_coords'], dtype=float)[
                    self.plan['target_landmarks']
                ].mean(axis=0),
                **kwargs,
            )
            if kwargs:
                guide_metadata['guide_angle_stratum'] = angle_stratum
                guide_metadata['guide_azimuth_stratum'] = azimuth_stratum
        else:
            framework = jitter(self.plan['source_framework_coords'], seed,
                               float(self.guide.rotation_degrees), float(self.guide.translation_angstrom))
            guide_metadata = {'guide_sampling_policy': 'legacy_rigid_jitter'}
        target_ids = [self.plan['target_labels'][i] for i in self.plan['target_landmarks']]
        target_tokens = [offsets[name] + ordinal for name, ordinal in target_ids]
        binder_tokens = [offsets[self.cfg.binder.chain] + fixed[i] for i in self.plan['binder_landmarks']]
        bins, mask, records = cross_distogram(
            total, target_tokens, binder_tokens,
            np.asarray(self.plan['target_coords'])[self.plan['target_landmarks']],
            framework[self.plan['binder_landmarks']], int(self.guide.pairs))
        merge_distogram(feats, bins, mask)
        informative_count = sum(bool(p['finite_distance_information']) for p in records)
        overflow_count = len(records) - informative_count
        feats['pose_record'] = {
            'pairs': records, 'jitter_seed': seed,
            'generation_only': True, 'plan_sha256': self.guide.plan_sha256,
            'approach_angle_range_degrees': list(interval) if interval is not None else None,
            'requested_guidance_pairs': len(records),
            'informative_guidance_pairs': informative_count,
            'overflow_guidance_pairs': overflow_count,
            'maximum_anchor_distance_angstrom': max(
                (float(p['distance_angstrom']) for p in records), default=None
            ),
            'distance_encoding': 'promera_51_bins_1_to_50_with_overflow',
            'targeted_site_enabled': site_enabled,
            'targeted_site_tokens': len(site_tokens),
            'targeted_site_chain': site['chain'] if site_enabled else None,
            'bypass_targeted_backbone_gate': bool(
                self.guide.get('bypass_targeted_backbone_gate', False)
            ),
            **guide_metadata,
        }
        return feats

    def _refold_batch(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        """Fail rather than accidentally validating a structurally restrained refold."""
        original = model.pairformer_forward

        def unrestrained(feats: dict[str, Any], *a: Any, **kw: Any) -> Any:
            for key in ('distogram_mask', 'is_epitope'):
                value = feats.get(key)
                if value is not None and bool(value.any()):
                    raise RuntimeError(f'validation contamination: nonzero {key} in refold')
            return original(feats, *a, **kw)

        model.pairformer_forward = unrestrained
        try:
            return super()._refold_batch(model, *args, **kwargs)
        finally:
            model.pairformer_forward = original

    def run_batch(self, model: Any, batch: dict[str, Any]) -> None:
        if int(self.guide.pairs) and not model.cfg.model.disto_embed:
            raise RuntimeError('loaded model disables distogram conditioning')
        ids = [int(x) for x in batch['backbone_idx']]
        seed = int.from_bytes(hashlib.sha256(f'{self.guide.seed}:{ids}'.encode()).digest()[:4], 'big')
        start = time.perf_counter()
        with sampling_seed(seed, batch['restype'].device):
            if bool(self.guide.get('bypass_targeted_backbone_gate', False)):
                with allow_guided_backbones_past_native_gate():
                    super().run_batch(model, batch)
            else:
                super().run_batch(model, batch)
        elapsed = time.perf_counter() - start
        for index, b_idx in enumerate(ids):
            root = Path(_sample_dir(self.cfg.output, batch['name'][index], b_idx))
            root.mkdir(parents=True, exist_ok=True)
            record = dict(batch['pose_record'][index], batch_seed=seed,
                          batch_indices=ids, batch_wall_seconds=elapsed,
                          backbone_exists=(root / 'backbone.cif').is_file())
            (root / 'pose_guidance.json').write_text(json.dumps(record, indent=2) + '\n')
            if not record['backbone_exists']:
                raise RuntimeError(f'backbone missing at {root}; check for an upstream OOM/skip')
