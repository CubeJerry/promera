"""
This is all boilerplate code, safe to skip / ignore!
"""

import logging
import os
import socket
import time
from collections import defaultdict

import wandb
import numpy as np
import pandas as pd
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def get_logger(name):
    logger = logging.Logger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"%(asctime)s [{socket.gethostname()}:%(process)d] [%(levelname)s] %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger(__name__)


def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {
        key: sum([lg.get(key, []) for lg in log_list], [])
        for key in set(sum([list(lg.keys()) for lg in log_list], []))
    }
    return log


def get_log_mean(log):
    out = {}
    for key in log:
        if ":mask" in key:
            continue
        try:
            data = np.array(log[key])
            if f"{key}:mask" in log:
                mask = np.array(log[f"{key}:mask"])
                out[key] = float((data * mask).sum() / mask.sum())
            else:
                out[key] = float(np.nanmean(data))
        except Exception as _:
            pass
    return out


class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        self.iter_step = defaultdict(int)
        self.prefix = None
        self.masks = {}
        if cfg.wandb:
            self.wandb_init()

    def step(self, trainer, prefix):
        self.iter_step[prefix] += 1
        self._log[prefix + "/dur"].append(time.time() - self.last_log_time)
        self.last_log_time = time.time()

        interval = {"train": self.cfg.train_log_freq, "val": self.cfg.val_log_freq}[
            prefix
        ]
        if interval is not None and self.iter_step[prefix] % interval == 0:
            self.print_log(trainer, prefix)

    def log(self, key, data, mask=None):
        def sanitize(arr):
            if arr is None:
                return
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            if isinstance(arr, np.ndarray):
                arr = arr.flatten().tolist()
            if type(arr) in [list, tuple]:
                arr = arr
            else:
                arr = [float(arr)]
            return arr

        data = sanitize(data)
        mask = sanitize(mask)
        if mask is None:
            mask = [1.0] * len(data)

        self._log[f"{self.prefix}/{key}"].extend(data)
        self._log[f"{self.prefix}/{key}:mask"].extend(mask)

    def epoch_end(self, trainer, prefix):
        interval = {"train": self.cfg.train_log_freq, "val": self.cfg.val_log_freq}[
            prefix
        ]
        if interval is None:
            self.print_log(trainer, prefix)

    def print_log(self, trainer, prefix="train", save=False, extra_logs=None):

        assert not save, "print_log(save=True) does not work yet"
        log = self._log
        log = {key: log[key] for key in log if f"{prefix}/" in key}
        log = gather_log(log, trainer.world_size)
        mean_log = get_log_mean(log)

        mean_log.update(
            {
                prefix + "/epoch": trainer.current_epoch,
                prefix + "/global_step": trainer.global_step,
                prefix + "/iter_step": self.iter_step[prefix],
                prefix + "/count": len(log[next(iter(log))]),
            }
        )
        if extra_logs:
            mean_log.update(extra_logs)

        if trainer.is_global_zero:
            logger.info(str(mean_log))
            if self.cfg.wandb:
                self.wandb_log(mean_log)
            if save:
                path = os.path.join(
                    os.environ["MODEL_DIR"],
                    f"{prefix}_{self.trainer.current_epoch}.csv",
                )
                pd.DataFrame(log).to_csv(path)
        for key in list(self._log.keys()):
            if f"{prefix}/" in key:
                del self._log[key]

    @rank_zero_only
    def wandb_init(self):
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=self.cfg.project,
            name=self.cfg.name,
            id=getattr(self.cfg, "id", None),
            resume="allow",
            save_code=True,
        )

    @rank_zero_only
    def wandb_log(self, log):
        wandb.log(log)
