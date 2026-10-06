"""Opt-in task: cross-chain input conditioning, unchanged backbone sampler.

Import using --task pose_task.PoseGuidedDesign with this directory on PYTHONPATH.
Do not place the whole experimental checkout ahead of a patched runtime package.
"""
from __future__ import annotations

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

from pose_geometry import (chain_sequence, cross_distogram, digest,
                           framework_map, jitter, merge_distogram)


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
        for name, sequence in self.plan['target_sequences'].items():
            if chain_sequence(struct.chains[name]) != sequence:
                raise ValueError(f'generated target {name} differs from the prepared schema')
        fixed, _ = framework_map(chain_sequence(struct.chains[self.cfg.binder.chain]), self.plan['framework'])
        if len(fixed) != len(self.plan['source_framework_coords']):
            raise ValueError('generated framework mapping differs from the guide')
        b_idx = int(feats['backbone_idx'])
        seed = int.from_bytes(hashlib.sha256(f'{self.guide.seed}:{b_idx}'.encode()).digest()[:4], 'big')
        framework = jitter(self.plan['source_framework_coords'], seed,
                           float(self.guide.rotation_degrees), float(self.guide.translation_angstrom))
        target_ids = [self.plan['target_labels'][i] for i in self.plan['target_landmarks']]
        target_tokens = [offsets[name] + ordinal for name, ordinal in target_ids]
        binder_tokens = [offsets[self.cfg.binder.chain] + fixed[i] for i in self.plan['binder_landmarks']]
        bins, mask, records = cross_distogram(
            total, target_tokens, binder_tokens,
            np.asarray(self.plan['target_coords'])[self.plan['target_landmarks']],
            framework[self.plan['binder_landmarks']], int(self.guide.pairs))
        merge_distogram(feats, bins, mask)
        feats['pose_record'] = {'pairs': records, 'jitter_seed': seed,
                                'generation_only': True, 'plan_sha256': self.guide.plan_sha256}
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
