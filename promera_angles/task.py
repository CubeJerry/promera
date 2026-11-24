"""Instrument the installed ordinary Design task. No alternative sampler."""

import hashlib
import json
from pathlib import Path

import torch

from experiments.pose_guidance.pose_geometry import digest
from experiments.pose_guidance.pose_task import sampling_seed
from promera.data import utils as data_utils
from promera.inference.design import Design, _sample_dir

from .batches import accepted_batch
from .conditioning import condition
from .refold import UnconditionedRefoldMixin


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
            guidance_definition=self.spec.get(
                "guidance_definition", "legacy_placement_ray"
            ),
            cdr_lateral_offset=self.spec.get("cdr_lateral_offset", [0, 0]),
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
        batch = accepted_batch(batch)
        if batch is None:
            return
        ids = [int(i) for i in batch["backbone_idx"]]
        with sampling_seed(self.seed("batch", ids), batch["restype"].device):
            super().run_batch(model, batch)
