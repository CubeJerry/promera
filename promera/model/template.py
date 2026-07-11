import gc
import os
from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningModule

from ..utils.ema import ExponentialMovingAverage
from ..utils.logger import Logger
from ..utils.scheduler import AlphaFoldLRScheduler
from ..utils.tensor_utils import tensor_tree_map


def _release_inference_memory(context: str = "") -> None:
    """Collect dead Python objects and return unused CUDA blocks to the driver.

    This is deliberately called at batch boundaries, after ``run_batch`` has
    returned and its large pairformer/diffusion/confidence locals are no longer in
    scope.  ``empty_cache`` cannot free live tensors, but at this point it prevents
    the allocator's high-water mark from creeping across a long design run.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        # ipc_collect may be unavailable before CUDA IPC has been initialised.
        pass
    if str(os.environ.get("PROMERA_DEBUG_CUDA_MEM", "0")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        prefix = f"{context}: " if context else ""
        print(
            f"{prefix}cuda_allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
            f"cuda_reserved={torch.cuda.memory_reserved()/1e9:.2f}GB",
            flush=True,
        )


class LightningModuleTemplate(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # self.save_hyperparameters("cfg")
        self.cfg = cfg
        self.rng = np.random.default_rng(seed=137)
        if hasattr(cfg, "logger"):
            self._logger = Logger(cfg.logger)
        else:
            self._logger = None
        self.ema = None
        self.cached_weights = None

    # uncomment this to debug
    def on_before_optimizer_step(self, optimizer):
        quit = False
        for name, p in self.named_parameters():
            if p.requires_grad and p.grad is None:
                print(name, "has no grad")
                quit = True

        if quit:
            exit()

    def gradient_norm(self, module) -> float:
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, module.parameters())
        parameters = filter(lambda p: p.grad is not None, parameters)
        norm = torch.tensor([p.grad.norm(p=2) ** 2 for p in parameters]).sum().sqrt()
        return norm

    def parameter_norm(self, module) -> float:
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, module.parameters())
        norm = (
            torch.tensor([p.detach().norm(p=2) ** 2 for p in parameters]).sum().sqrt()
        )
        return norm

    def on_train_epoch_end(self):
        self._logger.epoch_end(self.trainer, prefix="train")

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()
        self._logger.epoch_end(self.trainer, prefix="val")

    #### EMA SWITCHING ####

    def ensure_ema_in_train_state(self):
        if self.ema is not None:
            if self.ema.device != self.device:
                self.ema.to(self.device)

            if self.cached_weights is not None:
                self.load_state_dict(self.cached_weights)
                self.cached_weights = None

    def ensure_ema_in_val_state(self):
        if self.ema is not None:
            if self.ema.device != self.device:
                self.ema.to(self.device)

            if self.cached_weights is None:
                clone_param = lambda t: t.detach().clone()
                self.cached_weights = tensor_tree_map(clone_param, self.state_dict())

                self.load_state_dict(self.ema.state_dict()["params"])

    ########## EMA CREATION #########
    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.cfg.model.ema and "ema" in checkpoint:
            self.ema = ExponentialMovingAverage(self, self.cfg.model.ema_decay)
            self.ema.load_state_dict(checkpoint["ema"])

        #### weight surgery - assumes pretrained model doesn't use EMA ####
        # state_dict = checkpoint["state_dict"]
        # current_model_state = self.state_dict()
        # for key in state_dict:
        #     requires_grad = False
        #     try:
        #         requires_grad = self.get_parameter(key).requires_grad
        #     except:
        #         pass
        #     if not requires_grad:
        #         state_dict[key] = current_model_state[key]

    def on_train_epoch_start(self):
        if self.cfg.model.ema and self.ema is None:
            self.ema = ExponentialMovingAverage(self, self.cfg.model.ema_decay)
        # if resuming training, do so with new scheduler hparams
        self.lr_schedulers().__dict__.update(self.cfg.optimizer.scheduler)

    ############# EMA HANDLING ########
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.ensure_ema_in_train_state()
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

    def on_train_batch_start(self, *args):
        self._logger.prefix = "train"
        self.ensure_ema_in_train_state()

        if os.environ.get("WEIGHT_SURGERY", "0") == "1":
            if getattr(self, "ckpt1", None) is None:
                print("Loading", os.environ["CKPT1"], flush=True)
                ckpt1 = torch.load(
                    os.environ["CKPT1"], map_location="cpu", weights_only=False
                )
                ckpt1 = ckpt1["ema"]["params"]
                self.ckpt1 = {k: v.to(self.device) for k, v in ckpt1.items()}

                print("Loading", os.environ["CKPT2"], flush=True)
                ckpt2 = torch.load(
                    os.environ["CKPT2"], map_location="cpu", weights_only=False
                )
                ckpt2 = ckpt2["ema"]["params"]
                self.ckpt2 = {k: v.to(self.device) for k, v in ckpt2.items()}
            start = int(os.environ["STEP1"])
            end = int(os.environ["STEP2"])
            r = (self.trainer.global_step - start) / (end - start)
            toload = {
                k: (1 - r) * self.ckpt1[k] + r * self.ckpt2[k] for k in self.ckpt1
            }
            self.load_state_dict(toload, strict=False)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._logger.log("grad_norm", self.gradient_norm(self))
        self._logger.log("param_norm", self.parameter_norm(self))

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self._logger.log("lr", lr)

        if (self.ema is not None) and (
            batch_idx % self.trainer.accumulate_grad_batches == 0
        ):
            self.ema.update(self)

        self._logger.step(self.trainer, "train")

    def on_validation_batch_start(self, *args):
        self._logger.prefix = "val"
        if self.cfg.model.val_ema:
            self.ensure_ema_in_val_state()

    def on_validation_batch_end(self, *args):
        self._logger.step(self.trainer, "val")

    def configure_optimizers(self):
        """Configure the optimizer."""
        cfg = self.cfg.optimizer
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            betas=cfg.adam_betas,
            eps=cfg.adam_eps,
            lr=cfg.base_lr,
        )
        scheduler = AlphaFoldLRScheduler(optimizer, **cfg.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ev = self.evals[dataloader_idx]
        savedir = f'{os.environ["MODEL_DIR"]}/eval_step{self.trainer.global_step}/{ev.cfg.name}'
        savedir = ev.cfg.get("savedir", savedir)
        os.makedirs(savedir, exist_ok=True)
        ev.run_batch(self, batch, savedir=savedir, logger=self._logger)

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # At the start of batch N, all temporary objects from batch N-1 should have
        # fallen out of scope.  Clean before allocating the next large diffusion.
        if batch_idx:
            _release_inference_memory(f"before predict batch {batch_idx}")

    def predict_step(self, batch, batch_idx):
        return self.inference_task.run_batch(self, batch)

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # run_batch has returned, so its local backbone/refold tensors are dead.
        _release_inference_memory(f"after predict batch {batch_idx}")

    def on_predict_epoch_end(self):
        _release_inference_memory("after predict epoch")
