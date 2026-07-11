import os
from collections import defaultdict

import numpy as np
import torch
import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_edm_sched_fn(cfg):
    def edm_sched_fn(t):
        p = cfg.rho
        sigma_max = cfg.sigma_max * cfg.sigma_data
        sigma_min = cfg.sigma_min * cfg.sigma_data
        return (
            sigma_min ** (1 / p)
            + (1 - t) * (sigma_max ** (1 / p) - sigma_min ** (1 / p))
        ) ** p

    return edm_sched_fn


def _truthy(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


class Sampler:
    def __init__(self, schedules, steppers):
        self.schedules = schedules
        self.steppers = steppers

    def sample(
        self,
        model,
        noisy_batch,
        steps=100,
        trunc=None,
        pbar=True,
        record_history=None,
    ):
        """Run diffusion without retaining hundreds of GPU coordinate snapshots.

        Promera's model only needs the final coordinates for normal prediction, but
        the original sampler appended ``x`` and ``x0`` on every step and the model
        stacked those lists unconditionally.  A 200-step, five-sample refold could
        therefore keep 400 full coordinate tensors alive until the whole refold
        returned.  Repeating that inside design made allocator pressure look like a
        leak and could eventually exhaust a 24 GB GPU.

        Full history is opt-in through ``record_history=True`` or
        ``PROMERA_RECORD_TRAJECTORY=1``.  Recorded frames are detached and moved to
        CPU immediately.  When history is disabled, a single final CPU frame is
        returned for backwards compatibility with callers that still stack the
        historical keys.
        """
        if record_history is None:
            record_history = _truthy(os.environ.get("PROMERA_RECORD_TRAJECTORY", "0"))

        extra = defaultdict(list) if record_history else None
        schedule_points = np.linspace(0, 1, steps + 1)
        schedule_steps = list(zip(schedule_points[:-1], schedule_points[1:]))
        iterator = tqdm.tqdm(schedule_steps) if pbar else iter(schedule_steps)

        for t, s in iterator:
            if trunc is not None and t < trunc:
                continue
            sched = {
                key: (schedule(t), schedule(s))
                for key, schedule in self.schedules.items()
            }
            noisy_batch = self.single_step(model, noisy_batch, sched, extra)

        if extra is None:
            # Existing model code expects both keys and stacks each list.  Keep one
            # CPU frame instead of the complete GPU trajectory.
            final_cpu = noisy_batch["coords"].detach().cpu()
            extra = {"traj": [final_cpu], "noisy": [final_cpu]}
        return noisy_batch, extra

    def single_step(self, model_fn, noisy_batch, sched, extra=None, sc=True):
        for stepper in self.steppers:
            stepper.set_step(noisy_batch, sched, extra)

        readout = model_fn(noisy_batch)

        for stepper in self.steppers:
            stepper.advance(noisy_batch, sched, readout, extra)

        return noisy_batch


class EDMDiffusionStepper:
    def __init__(self, cfg=None, mask=None):
        self.cfg = cfg
        self.mask = mask

    def set_step(self, batch, sched, extra=None):
        t, s = sched["coords"]

        cfg = self.cfg
        if cfg.edm_churn:
            gamma = cfg.gamma_0 if s > cfg.gamma_min else 0
            t_hat = t * (gamma + 1)
            noise = (
                cfg.noise_scale
                * np.sqrt(t_hat**2 - t**2)
                * torch.randn_like(batch["coords"])
            )
            batch["coords"] += noise
            batch["coords_sigma"][:] = t_hat
        else:
            batch["coords_sigma"][:] = t

    def advance(self, batch, sched, out, extra=None):
        cfg = self.cfg
        if cfg.edm_churn:
            t, s = sched["coords"]
            x = batch["coords"]
            x0 = out["coords"]
            # boltz alignment
            from ..model.loss.diffusion import weighted_rigid_align

            # coords carry the multiplicity-expanded batch (B * diffusion_samples)
            # while atom_pad_mask is still the un-expanded batch (B). Repeat it to
            # match so the per-sample alignment broadcasts correctly for B > 1.
            atom_mask = batch["atom_pad_mask"].float()
            if atom_mask.shape[0] != x.shape[0]:
                atom_mask = atom_mask.repeat_interleave(
                    x.shape[0] // atom_mask.shape[0], 0
                )

            with torch.autocast("cuda", enabled=False):
                x = weighted_rigid_align(
                    x.float(),
                    x0.float(),
                    atom_mask,
                    atom_mask,
                )
            t_hat = batch["coords_sigma"]
            delta = (x0 - x) / t_hat[..., None, None]
            dt = t_hat - s
            dx = cfg.step_scale * dt[..., None, None] * delta
        else:
            x = batch["coords"]
            t2, t1 = sched["coords"]
            dt = t2 - t1
            g = np.sqrt(2 * t2)
            x0 = out["coords"]
            score = (x0 - x) / t2**2
            noise = torch.randn_like(x)
            gamma = self.cfg.temp_factor
            weight = 2 * (t2 / self.cfg.sigma_data + 1)
            weight = max(weight, 1)
            weight = 0
            ode = 0.5 * g**2 * score * dt
            sde = 0.5 * g**2 * score * dt + g * gamma * np.sqrt(dt) * noise
            dx = ode + weight * sde

        if extra is not None:
            # Trajectory saving is intentionally CPU-backed.  Keeping these frames
            # on CUDA defeats the purpose of an opt-in diagnostic trajectory.
            extra["traj"].append(x0.detach().cpu())
            extra["noisy"].append(x.detach().cpu())
        batch["coords"] = x + dx
