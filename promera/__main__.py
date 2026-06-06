import argparse
import os

_DEFAULT_MODEL_CONFIG = os.path.join(os.path.dirname(__file__), "model/config.yaml")
_DEFAULT_TASK_CONFIG = os.path.join(
    os.path.dirname(__file__), "inference/cofolding.yaml"
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_config", type=str, default=_DEFAULT_MODEL_CONFIG)
parser.add_argument("--task_config", type=str, default=_DEFAULT_TASK_CONFIG)
parser.add_argument("--weights", type=str, default=os.environ.get("PROMERA_WEIGHTS"))
parser.add_argument(
    "--task",
    type=str,
    default="promera.inference.Cofolding",
    help="Dotted path to an inference task class",
)
args, extra = parser.parse_known_args()

from omegaconf import OmegaConf

model_cfg = OmegaConf.merge(
    OmegaConf.load(args.model_config), OmegaConf.from_cli(extra)
)
task_cfg = OmegaConf.merge(OmegaConf.load(args.task_config), OmegaConf.from_cli(extra))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from promera.data.utils import collate
from promera.utils.load_weights import load_weights


def _get_attr_from_path(path):
    import importlib

    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


model_cfg.trainer.devices = int(
    os.environ.get("SLURM_NTASKS_PER_NODE", model_cfg.trainer.devices)
)
torch.set_float32_matmul_precision(model_cfg.set_float32_matmul_precision)

trainer = pl.Trainer(
    **model_cfg.trainer,
    num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
)

model = _get_attr_from_path(model_cfg.model._target_)(model_cfg)

load_weights(args.weights, model)

task_cls = _get_attr_from_path(args.task)
task = task_cls(task_cfg)
model.inference_task = task
loader = DataLoader(
    task, batch_size=1, collate_fn=collate, num_workers=model_cfg.data.workers
)

trainer.predict(model, loader)
