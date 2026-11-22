"""Instrument the installed ordinary Design task. No alternative sampler."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pose_guidance.pose_geometry import (
    chain_sequence,
    cross_distogram,
    digest,
    framework_map,
    merge_distogram,
)
from experiments.pose_guidance.pose_task import UnconditionedRefoldMixin, sampling_seed
from promera.data import utils as data_utils
from promera.inference.design import Design, _sample_dir

from .geometry import placement


def condition(feats, plan, seed, interval, standoff):
    struct = feats["struct"]
    offsets, n = {}, 0
    for name, chain in struct.chains.items():
        offsets[name] = n
        n += len(chain.rname)
    if n != len(feats["restype"]):
        raise ValueError("angular landmarks require one token per protein residue")
    keys = plan["keys"]

    def token(key):
        chain, pos = key
        return offsets[chain] + pos - 1

    expected = {token(keys[i]) for i in plan["selected"]}
    actual = set()
    for name, sequence in plan["sequences"].items():
        if chain_sequence(struct.chains[name]) != sequence:
            raise ValueError(f"runtime target sequence changed on {name}")
        actual.update(
            offsets[name] + int(i)
            for i in np.flatnonzero(
                feats["is_epitope"][offsets[name] : offsets[name] + len(sequence)]
            )
        )
    if expected != actual:
        raise ValueError(
            "native epitope flags disagree with the plan; check the installed multichain adapter"
        )
    binder = plan["binder_chain"]
    fw, cdr = framework_map(chain_sequence(struct.chains[binder]), plan["framework"])
    record = dict(cdr_positions=[p + 1 for p in cdr])
    if interval is None:
        return record
    marks = plan["frame"]["landmarks"]
    shape = plan["shape"]
    coords, guide = placement(plan["frame"], shape, interval, seed, standoff)
    bins, mask, pairs = cross_distogram(
        n,
        [token(keys[i]) for i in marks],
        [offsets[binder] + fw[i] for i in shape["fixed_indices"]],
        np.asarray(plan["coords"])[marks],
        coords,
        16,
    )
    target_xyz = np.asarray(plan["coords"])
    v = target_xyz - plan["frame"]["origin"]
    along = v @ np.array(guide["ray"])
    across = np.linalg.norm(v - np.outer(along, guide["ray"]), axis=1)
    guide["ray_obstructed_ca_proxy"] = bool(
        ((along > 4) & (along < standoff + 18) & (across < 4)).any()
    )
    merge_distogram(feats, bins, mask)
    return dict(record, **guide, pairs=pairs)


class TargetedDesign(UnconditionedRefoldMixin, Design):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.spec = cfg.angle_trial
        self.items = [(name, i + self.spec.start) for name, i in self.items]
        if digest(self.spec.plan) != self.spec.sha256:
            raise ValueError("angular plan digest changed")
        self.plan = json.loads(Path(self.spec.plan).read_text())
        if (
            cfg.binder.framework != self.plan["framework"]
            or cfg.binder.chain != self.plan["binder_chain"]
        ):
            raise ValueError("framework differs from prepared plan")
        for key in ("distogram_mask", "distogram_emb"):
            if key not in data_utils._token_pair_keys:
                data_utils._token_pair_keys.append(key)

    def seed(self, stream, ids):
        raw = f"{self.spec.seed}:{stream}:{ids}".encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")

    def __getitem__(self, idx):
        with sampling_seed(self.seed("features", idx), torch.device("cpu")):
            feats = super().__getitem__(idx)
        index = int(feats["backbone_idx"])
        feats["angle_record"] = condition(
            feats,
            self.plan,
            self.seed("angle", index),
            self.spec.interval,
            self.spec.standoff,
        )
        return feats

    def run_batch(self, model, batch):
        if self.spec.interval is not None and not model.cfg.model.disto_embed:
            raise ValueError("model has distance embedding disabled")
        ids = [int(i) for i in batch["backbone_idx"]]
        for name, i, record in zip(batch["name"], ids, batch["angle_record"]):
            folder = Path(_sample_dir(self.cfg.output, name, i))
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "angle.json").write_text(json.dumps(record, allow_nan=False))
        with sampling_seed(self.seed("batch", ids), batch["restype"].device):
            super().run_batch(model, batch)
