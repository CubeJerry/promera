# Adapted from https://github.com/jwohlwend/boltz
import torch
from torch import nn


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


def symmetry_correction(
    pred_coords, feats, multiplicity=1, nucleotide_weight=5.0, ligand_weight=10.0
):
    dev = pred_coords.device

    token_to_rep_atom = feats["token_to_rep_atom"].repeat_interleave(multiplicity, 0)
    alt_coords = feats["alt_coords"].repeat_interleave(multiplicity, 0)

    # print(alt_coords.shape)

    alt_coords_mask = feats["alt_coords_mask"].repeat_interleave(multiplicity, 0)
    alt_token_coords = torch.gather(
        alt_coords,
        1,
        token_to_rep_atom[..., None, None].repeat(1, 1, alt_coords.shape[2], 3),
    ).permute(2, 0, 1, 3)
    alt_token_mask = torch.gather(
        alt_coords_mask,
        1,
        token_to_rep_atom[..., None].repeat(1, 1, alt_coords.shape[2]),
    ).permute(2, 0, 1)

    pred_token_coords = torch.gather(
        pred_coords, 1, token_to_rep_atom[..., None].repeat(1, 1, 3)
    )

    alt_dmat = torch.cdist(alt_token_coords, alt_token_coords)
    pred_dmat = torch.cdist(pred_token_coords, pred_token_coords)

    token_mask = feats["token_pad_mask"] * alt_token_mask
    pair_mask = (
        token_mask.unsqueeze(-1)
        * token_mask.unsqueeze(-2)
        * (1 - torch.eye(alt_dmat.shape[-1], device=dev))
    )
    proposed_lddt = lddt_dist(
        pred_dmat, alt_dmat, pair_mask, cutoff=15.0, per_atom=False
    )[0]

    idx = proposed_lddt.argmax(0)
    # print(proposed_lddt.max(0), idx)
    alt_coords = torch.gather(
        alt_coords, 2, idx.reshape(-1, 1, 1, 1).expand(*alt_coords.shape[:2], 1, 3)
    )
    alt_coords_mask = torch.gather(
        alt_coords_mask, 2, idx.reshape(-1, 1, 1).expand(*alt_coords.shape[:2], 1)
    )
    return alt_coords.squeeze(2), alt_coords_mask.squeeze(2)


def _all_gather_with_grad(tensor, dim):
    """All-gather tensor across ranks along dim, preserving gradient for the local rank's slice."""
    import torch.distributed as dist

    print("Trying to all gather", flush=True)

    if (
        not (dist.is_available() and dist.is_initialized())
        or dist.get_world_size() == 1
    ):
        return tensor
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.detach())
    gathered[rank] = tensor  # keep gradient path for local rank's samples
    return torch.cat(gathered, dim=dim)


def compute_confidence_loss(
    model_out,
    feats,
    true_coords,
    true_coords_resolved_mask,
    multiplicity=1,
    alpha_pae=0.0,
):
    # Compute losses
    plddt_loss, plddt_mae = compute_plddt_loss(
        model_out["plddt_logits"],
        model_out["sample_atom_coords"],
        true_coords,
        true_coords_resolved_mask,
        feats,
        multiplicity=multiplicity,
    )
    pde_loss, pde_mae = compute_pde_loss(
        model_out["pde_logits"],
        model_out["sample_atom_coords"],
        true_coords,
        true_coords_resolved_mask,
        feats,
        multiplicity=multiplicity,
    )

    pae_loss, pae_mae = compute_pae_loss(
        model_out["pae_logits"],
        model_out["sample_atom_coords"],
        true_coords,
        true_coords_resolved_mask,
        feats,
        multiplicity=multiplicity,
    )

    loss = plddt_loss + pde_loss + pae_loss

    dict_out = {
        "loss": loss,
        "plddt_loss": plddt_loss,
        "plddt_mae": plddt_mae,
        "pde_loss": pde_loss,
        "pde_mae": pde_mae,
        "pae_loss": pae_loss,
        "pae_mae": pae_mae,
    }

    return dict_out


def compute_plddt_loss(
    pred_lddt_logits,
    pred_atom_coords,
    true_atom_coords,
    true_coords_resolved_mask,
    feats,
    multiplicity=1,
):

    token_to_rep_atom = feats["token_to_rep_atom"].repeat_interleave(multiplicity, 0)
    pred_token_coords = torch.gather(
        pred_atom_coords, 1, token_to_rep_atom[..., None].repeat(1, 1, 3)
    )
    true_token_coords = torch.gather(
        true_atom_coords, 1, token_to_rep_atom[..., None].repeat(1, 1, 3)
    )

    pred_dist = torch.cdist(pred_token_coords, pred_token_coords)
    true_dist = torch.cdist(true_token_coords, true_token_coords)

    atom_mask = true_coords_resolved_mask
    token_mask = torch.gather(atom_mask, 1, token_to_rep_atom) * feats["token_pad_mask"]
    pair_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)
    pair_mask = pair_mask * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))

    cutoff = 15 + 15 * (feats["is_dna"] + feats["is_rna"])[:, None, :]

    target_lddt, mask_no_match = lddt_dist(
        pred_dist, true_dist, pair_mask, cutoff, per_atom=True
    )
    pred_lddt = compute_aggregated_metric(pred_lddt_logits)

    # compute loss
    num_bins = pred_lddt_logits.shape[-1]
    bin_index = torch.floor(target_lddt * num_bins).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    lddt_one_hot = nn.functional.one_hot(bin_index, num_classes=num_bins)

    errors = -1 * torch.sum(
        lddt_one_hot * torch.nn.functional.log_softmax(pred_lddt_logits, dim=-1), dim=-1
    )

    loss = torch.sum(errors * token_mask * mask_no_match, dim=-1) / (
        1e-7 + torch.sum(token_mask * mask_no_match, dim=-1)
    )

    # Average over the batch dimension
    loss = torch.mean(loss)

    mae = torch.abs(pred_lddt - target_lddt)
    mae = torch.sum(mae * token_mask * mask_no_match, dim=-1) / (
        1e-7 + torch.sum(token_mask * mask_no_match, dim=-1)
    )
    mae = torch.mean(mae)

    return loss, mae


def compute_pde_loss(
    pred_pde_logits,
    pred_atom_coords,
    true_atom_coords,
    true_coords_resolved_mask,
    feats,
    multiplicity=1,
    max_dist=32.0,
):
    token_to_rep_atom = feats["token_to_rep_atom"].repeat_interleave(multiplicity, 0)
    pred_token_coords = torch.gather(
        pred_atom_coords, 1, token_to_rep_atom[..., None].repeat(1, 1, 3)
    )
    true_token_coords = torch.gather(
        true_atom_coords, 1, token_to_rep_atom[..., None].repeat(1, 1, 3)
    )
    atom_mask = true_coords_resolved_mask
    token_mask = torch.gather(atom_mask, 1, token_to_rep_atom) * feats["token_pad_mask"]

    mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)

    true_d = torch.cdist(true_token_coords, true_token_coords)
    pred_d = torch.cdist(pred_token_coords, pred_token_coords)
    target_pde = torch.abs(true_d - pred_d)

    # compute loss
    num_bins = pred_pde_logits.shape[-1]
    bin_index = torch.floor(target_pde * num_bins / max_dist).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    pde_one_hot = nn.functional.one_hot(bin_index, num_classes=num_bins)

    errors = -1 * torch.sum(
        pde_one_hot * torch.nn.functional.log_softmax(pred_pde_logits, dim=-1),
        dim=-1,
    )
    loss = torch.sum(errors * mask, dim=(-2, -1)) / (
        1e-7 + torch.sum(mask, dim=(-2, -1))
    )

    # Average over the batch dimension
    loss = torch.mean(loss)

    pred_pde = compute_aggregated_metric(pred_pde_logits, end=32)
    mae = torch.abs(torch.clamp(pred_pde, max=32) - torch.clamp(target_pde, max=32))
    mae = torch.sum(mae * mask, dim=(-2, -1)) / (1e-7 + torch.sum(mask, dim=(-2, -1)))
    mae = torch.mean(mae)

    return loss, mae


def compute_collinear_mask(v1, v2):
    norm1 = torch.linalg.vector_norm(v1, axis=-1, keepdims=True)
    norm2 = torch.linalg.vector_norm(v2, axis=-1, keepdims=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = torch.abs(torch.sum(v1 * v2, axis=-1)) < 0.9063
    mask_overlap1 = norm1.squeeze(-1) > 1e-2
    mask_overlap2 = norm2.squeeze(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def express_coordinate_in_frame(atom_coords, a, b, c):
    w1 = (a - b) / (torch.norm(a - b, dim=-1, keepdim=True) + 1e-5)
    w2 = (c - b) / (torch.norm(c - b, dim=-1, keepdim=True) + 1e-5)

    # build orthogonal frame
    e1 = (w1 + w2) / (torch.norm(w1 + w2, dim=-1, keepdim=True) + 1e-5)
    e2 = (w2 - w1) / (torch.norm(w2 - w1, dim=-1, keepdim=True) + 1e-5)
    e3 = torch.linalg.cross(e1, e2)

    # project onto frame basis

    d = atom_coords[:, None, :, :] - b[:, :, None, :]
    x_transformed = torch.cat(
        [
            torch.sum(d * e1[:, :, None, :], dim=-1, keepdim=True),
            torch.sum(d * e2[:, :, None, :], dim=-1, keepdim=True),
            torch.sum(d * e3[:, :, None, :], dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    return x_transformed


def compute_pae_loss(
    pred_pae_logits,
    pred_atom_coords,
    true_atom_coords,
    true_coords_resolved_mask,
    feats,
    multiplicity=1,
    max_dist=32.0,
):
    arange = torch.arange(0, pred_atom_coords.shape[0], device=pred_atom_coords.device)
    frame_idx = feats["frames_idx"].repeat_interleave(multiplicity, 0)
    frame_mask = feats["frames_mask"].repeat_interleave(multiplicity, 0)
    token_to_rep_atom = feats["token_to_rep_atom"].repeat_interleave(multiplicity, 0)

    pred_token_coords = pred_atom_coords[arange[:, None], token_to_rep_atom]
    pred_frames_expanded = pred_atom_coords[arange[:, None, None], frame_idx]
    pred_frame_mask = (
        compute_collinear_mask(
            pred_frames_expanded[..., 1, :] - pred_frames_expanded[..., 0, :],
            pred_frames_expanded[..., 1, :] - pred_frames_expanded[..., 2, :],
        )
        & frame_mask
    )
    pred_coords_transformed = express_coordinate_in_frame(
        pred_token_coords, *pred_frames_expanded.unbind(-2)
    )

    true_token_coords = true_atom_coords[arange[:, None], token_to_rep_atom]
    true_token_mask = true_coords_resolved_mask[arange[:, None], token_to_rep_atom]
    true_token_mask = true_token_mask * feats["token_pad_mask"]
    true_frames_expanded = true_atom_coords[arange[:, None, None], frame_idx]
    true_frame_mask = (
        compute_collinear_mask(
            true_frames_expanded[..., 1, :] - true_frames_expanded[..., 0, :],
            true_frames_expanded[..., 1, :] - true_frames_expanded[..., 2, :],
        )
        & frame_mask
    )
    true_frame_mask &= true_coords_resolved_mask[arange[:, None, None], frame_idx].all(
        -1
    )

    true_coords_transformed = express_coordinate_in_frame(
        true_token_coords, *true_frames_expanded.unbind(-2)
    )

    target_pae = torch.sqrt(
        ((true_coords_transformed - pred_coords_transformed) ** 2).sum(-1) + 1e-8
    )

    pair_mask = (true_frame_mask * pred_frame_mask)[:, :, None] * true_token_mask[
        :, None, :
    ]

    # compute loss
    num_bins = pred_pae_logits.shape[-1]
    bin_index = torch.floor(target_pae * num_bins / max_dist).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    pae_one_hot = nn.functional.one_hot(bin_index, num_classes=num_bins)

    errors = -1 * torch.sum(
        pae_one_hot
        * torch.nn.functional.log_softmax(
            pred_pae_logits.reshape(pae_one_hot.shape), dim=-1
        ),
        dim=-1,
    )
    loss = torch.sum(errors * pair_mask, dim=(-2, -1)) / (
        1e-7 + torch.sum(pair_mask, dim=(-2, -1))
    )
    # Average over the batch dimension
    loss = torch.mean(loss)

    pred_pae = compute_aggregated_metric(pred_pae_logits, end=32)
    mae = torch.abs(torch.clamp(pred_pae, max=32) - torch.clamp(target_pae, max=32))
    mae = torch.sum(mae * pair_mask, dim=(-2, -1)) / (
        1e-7 + torch.sum(pair_mask, dim=(-2, -1))
    )
    mae = torch.mean(mae)

    return loss, mae


def lddt_dist(dmat_predicted, dmat_true, mask, cutoff=15.0, per_atom=False):
    # NOTE: the mask is a pairwise mask which should have the identity elements already masked out
    # Compute mask over distances
    dists_to_score = (dmat_true < cutoff).float() * mask
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    score = 0.25 * (
        (dist_l1 < 0.5).float()
        + (dist_l1 < 1.0).float()
        + (dist_l1 < 2.0).float()
        + (dist_l1 < 4.0).float()
    )

    # Normalize over the appropriate axes.
    if per_atom:
        mask_no_match = torch.sum(dists_to_score, dim=-1) != 0
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=-1))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=-1))
        return score, mask_no_match.float()
    else:
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=(-2, -1)))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=(-2, -1)))
        total = torch.sum(dists_to_score, dim=(-1, -2))
        return score, total
