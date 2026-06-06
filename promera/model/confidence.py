# Adapted from https://github.com/jwohlwend/boltz
import torch.nn as nn
from .layers import initialize as init
from .utils import LinearNoBias
from .trunk import PairformerModule
import os
import torch
from torch import nn
from tinyprot.feature import _ntoks


class ConfidenceModule(nn.Module):
    """Confidence module."""

    def __init__(
        self,
        cfg,
        distogram=False,
        inp_only=False,
    ):

        super().__init__()
        self.cfg = cfg
        self.inp_only = inp_only
        token_s, token_z = cfg.dims.token_s, cfg.dims.token_z

        if distogram:
            boundaries = torch.linspace(2, cfg.max_dist, cfg.num_dist_bins - 1)
            self.register_buffer("boundaries", boundaries)
            self.dist_bin_pairwise_embed = nn.Embedding(cfg.num_dist_bins, token_z)
            torch.nn.init.normal_(self.dist_bin_pairwise_embed.weight, std=10.0)

        from tinyprot.feature import _ntoks

        s_input_dim = cfg.dims.token_s + 2 * _ntoks + 2
        self.s_to_z = LinearNoBias(s_input_dim, token_z)
        self.s_to_z_transpose = LinearNoBias(s_input_dim, token_z)
        self.s_norm = nn.LayerNorm(cfg.dims.token_s)
        init.gating_init_(self.s_to_z.weight)
        init.gating_init_(self.s_to_z_transpose.weight)

        self.pairformer_stack = PairformerModule(cfg.pairformer)

        self.confidence_heads = ConfidenceHeads(
            token_s,
            token_z,
        )

    def forward(self, feats, out, multiplicity):

        s_inputs = out["s_inputs"].detach()

        if self.inp_only:
            s = out["s_init"].detach()
            z = out["z_init"].detach()

        else:
            s = out["s"].detach()
            z = out["z"].detach()

        s = self.s_norm(s)

        x_pred = out.get("sample_atom_coords", None)

        z = z.repeat_interleave(multiplicity, 0)
        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )

        token_to_rep_atom = feats["token_to_rep_atom"].repeat_interleave(
            multiplicity, 0
        )

        d = None
        if x_pred is not None:
            if len(x_pred.shape) == 4:
                B, mult, N, _ = x_pred.shape
                x_pred = x_pred.reshape(B * mult, N, -1)

            x_pred_repr = torch.gather(
                x_pred, 1, token_to_rep_atom[..., None].repeat(1, 1, 3)
            )
            d = torch.cdist(x_pred_repr, x_pred_repr)

            distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
            distogram = self.dist_bin_pairwise_embed(distogram)

            z = z + distogram

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        s = s.repeat_interleave(multiplicity, 0)
        # breakpoint()
        s, z = self.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask)

        return self.confidence_heads(
            s=s,
            z=z,
            feats=feats,
            multiplicity=multiplicity,
        )


class ConfidenceHeads(nn.Module):
    """Confidence heads."""

    def __init__(
        self,
        token_s,
        token_z,
        num_plddt_bins=50,
        num_pde_bins=64,
        num_pae_bins=64,
        protenix=False,
    ):

        super().__init__()

        self.to_pde_logits = LinearNoBias(token_z, num_pde_bins)
        self.to_plddt_logits = LinearNoBias(token_s, num_plddt_bins)
        self.to_pae_logits = LinearNoBias(token_z, num_pae_bins)

        torch.nn.init.zeros_(self.to_pde_logits.weight)
        torch.nn.init.zeros_(self.to_plddt_logits.weight)
        torch.nn.init.zeros_(self.to_pae_logits.weight)

        self._protenix = protenix
        if protenix:
            self.pae_ln = nn.LayerNorm(token_z)
            self.pde_ln = nn.LayerNorm(token_z)
            self.plddt_ln = nn.LayerNorm(token_s)

    def forward(
        self,
        s,
        z,
        feats,
        multiplicity=1,
    ):
        if self._protenix:
            plddt_logits = self.to_plddt_logits(self.plddt_ln(s))
            pde_logits = self.to_pde_logits(self.pde_ln(z + z.transpose(1, 2)))
            pae_logits = self.to_pae_logits(self.pae_ln(z))
        else:
            plddt_logits = self.to_plddt_logits(s)
            pde_logits = self.to_pde_logits(z + z.transpose(1, 2))
            pae_logits = self.to_pae_logits(z)

        out_dict = dict(
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
        )

        plddt = compute_aggregated_metric(plddt_logits)
        pde = compute_aggregated_metric(pde_logits, end=32)
        pae = compute_aggregated_metric(pae_logits, end=32)

        out_dict.update(
            dict(
                pde=pde,
                plddt=plddt,
                pae=pae,
            )
        )

        return out_dict


def compute_aggregated_metric(logits, end=1.0):
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=end, step=bin_width, device=logits.device
    )
    probs = nn.functional.softmax(logits, dim=-1)
    plddt = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return plddt


class ContactModule(nn.Module):
    """Classifies predicted contacts (from sampled structure) as true or false."""

    def __init__(self, cfg):
        super().__init__()
        token_s, token_z = cfg.dims.token_s, cfg.dims.token_z
        s_input_dim = cfg.dims.token_s + 2 * _ntoks + 2

        self.s_to_z = LinearNoBias(s_input_dim, token_z)
        self.s_to_z_transpose = LinearNoBias(s_input_dim, token_z)
        self.s_norm = nn.LayerNorm(token_s)
        init.gating_init_(self.s_to_z.weight)
        init.gating_init_(self.s_to_z_transpose.weight)

        boundaries = torch.linspace(2, cfg.max_dist, cfg.num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(cfg.num_dist_bins, token_z)
        torch.nn.init.normal_(self.dist_bin_pairwise_embed.weight, std=10.0)

        self.pairformer_stack = PairformerModule(cfg.pairformer)
        self.to_contact_logits = LinearNoBias(token_z, 1)
        torch.nn.init.zeros_(self.to_contact_logits.weight)

    def forward(self, feats, out, multiplicity):
        s_inputs = out["s_inputs"].detach()
        s = self.s_norm(out["s"].detach())
        z = out["z"].detach()

        z = z.repeat_interleave(multiplicity, 0)
        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )

        # Condition on predicted distances from sampled structure
        x_pred = out["sample_atom_coords"]
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        token_to_rep_atom = feats["token_to_rep_atom"].repeat_interleave(
            multiplicity, 0
        )
        x_pred_repr = torch.gather(
            x_pred, 1, token_to_rep_atom[..., None].repeat(1, 1, 3)
        )
        pred_dist = torch.cdist(x_pred_repr, x_pred_repr)

        distogram = (pred_dist.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        z = z + self.dist_bin_pairwise_embed(distogram)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s = s.repeat_interleave(multiplicity, 0)

        s, z = self.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask)

        contact_logits = self.to_contact_logits(z + z.transpose(1, 2)).squeeze(-1)
        return {"contact_logits": contact_logits, "pred_dist": pred_dist}
